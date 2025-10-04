"""
人脸换脸谱应用 - GUI版本
功能：实时检测人脸并叠加脸谱效果，支持按钮切换脸谱
作者：[Your Name]
日期：[Current Date]
"""

import cv2
import mediapipe as mp
import numpy as np
from math import hypot, atan2, degrees
import os 
import glob
from typing import List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time

prreview_size = 128

class FaceMaskGUI:
    def __init__(self, root):
        """初始化人脸换脸谱GUI应用"""
        # 设置中文字体支持
        self.root = root
        self.root.title("人脸换脸谱")
        self.root.geometry("1400x950")
        self.root.resizable(True, True)
        # 设置窗口背景颜色为明亮的颜色
        self.root.configure(bg="#FFF8E1")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 初始化应用状态
        self.is_running = False
        self.cap = None
        self.thread = None
        self.auto_change_timer = None
        self.auto_change_enabled = False
        
        # 初始化Mediapipe模块
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.face_masks = []
        self.current_mask_index = 0
        self.mask_thumbnails = []  # 存储缩略图
        self.mask_buttons = []     # 存储按钮
        self.average_mask_size = (0, 0)  # 脸谱平均尺寸
        
        # 配置参数
        self.config = {
            'display_size': (1280, 720),
            'hand_face_threshold': 0.1,
            'mask_alpha': 0.28,
            'original_eye_distance': 100,
            'auto_change_interval': 5  # 自动切换间隔（秒）
        }
        
        # 创建状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        
        # 创建GUI组件
        self.create_widgets()
        
        # 加载脸谱
        self.load_face_masks()
    
    def create_widgets(self):
        """创建GUI组件"""
        # 设置中文字体和样式
        style = ttk.Style()
        style.configure("TButton", font=('Microsoft YaHei', 14), padding=15)
        style.configure("TLabel", font=('Microsoft YaHei', 12))
        style.configure("TLabelFrame", font=('Microsoft YaHei', 14, 'bold'))
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧设置区域
        left_frame = ttk.Frame(main_frame, width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 创建透明度设置框架
        alpha_frame = ttk.LabelFrame(left_frame, text="透明度调整", padding="10")
        alpha_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 垂直排列透明度控件 - 居中对齐
        ttk.Label(alpha_frame, text="透明度:", font=('Microsoft YaHei', 12, 'bold')).pack(pady=5)
        
        # 创建一个容器来居中透明度滑块
        alpha_scale_frame = ttk.Frame(alpha_frame)
        alpha_scale_frame.pack(fill=tk.Y, expand=True)
        
        # 确保透明度滑块与图像高度匹配
        self.alpha_scale = ttk.Scale(alpha_scale_frame, from_=0.1, to=0.9, orient=tk.VERTICAL, 
                                    value=self.config['mask_alpha'], length=600, 
                                    command=self.update_alpha)
        self.alpha_scale.pack(fill=tk.Y, expand=True, padx=20)
        
        # 居中显示透明度数值
        self.alpha_value = ttk.Label(alpha_frame, text=f"{self.config['mask_alpha']:.2f}", font=('Microsoft YaHei', 14, 'bold'))
        self.alpha_value.pack(pady=5)
        
        # 创建右侧控制区域
        right_frame = ttk.Frame(main_frame, width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # 创建当前脸谱预览框架
        preview_frame = ttk.LabelFrame(right_frame, text="当前脸谱", padding="10")
        preview_frame.pack(fill=tk.X, pady=5)
        
        # 使用tk.Frame作为有背景色的容器
        preview_inner = tk.Frame(preview_frame, bg="#FFD1DC", padx=10, pady=10)
        preview_inner.pack(fill=tk.BOTH, expand=True)
        
        # 创建方形的脸部预览区域（尺寸为prreview_size*prreview_size像素）
        self.mask_preview = tk.Label(preview_inner, bg="#FFFFFF", width=prreview_size, height=prreview_size)
        self.mask_preview.pack(pady=5)
        
        # 创建控制按钮框架
        control_frame = ttk.LabelFrame(right_frame, text="控制按钮", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        # 使用tk.Frame作为有背景色的容器
        control_inner = tk.Frame(control_frame, bg="#E6F7FF", padx=10, pady=10)
        control_inner.pack(fill=tk.BOTH, expand=True)
        
        # 创建垂直排列的容器
        buttons_container = tk.Frame(control_inner, bg="#E6F7FF")
        buttons_container.pack(fill=tk.BOTH, expand=True)
        
        # 垂直排列按钮 - 使用tk.Button代替ttk.Button以确保背景色正确显示
        button_width = 15
        button_height = 3
        
        # 创建开始按钮（绿色）
        self.start_button = tk.Button(buttons_container, 
                                     text="开始", 
                                     command=self.start_camera, 
                                     bg="#4CAF50", 
                                     fg="white",
                                     font=('Microsoft YaHei', 16, 'bold'),
                                     width=button_width, 
                                     height=button_height,
                                     relief=tk.RAISED,
                                     bd=5,
                                     cursor="hand2")
        self.start_button.pack(fill=tk.X, pady=10, padx=5)
        
        # 创建停止按钮（红色）
        self.stop_button = tk.Button(buttons_container, 
                                    text="停止", 
                                    command=self.stop_camera, 
                                    state=tk.DISABLED, 
                                    bg="#F44336", 
                                    fg="white",
                                    font=('Microsoft YaHei', 16, 'bold'),
                                    width=button_width, 
                                    height=button_height,
                                    relief=tk.RAISED,
                                    bd=5,
                                    cursor="hand2")
        self.stop_button.pack(fill=tk.X, pady=10, padx=5)
        
        # 创建自动切换脸谱选项
        auto_change_frame = tk.Frame(buttons_container, bg="#E6F7FF")
        auto_change_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # self.auto_change_var = tk.BooleanVar(value=False)
        # self.auto_change_checkbox = tk.Checkbutton(auto_change_frame, 
        #                                           text="自动切换脸谱", 
        #                                           variable=self.auto_change_var, 
        #                                           bg="#E6F7FF",
        #                                           font=('Microsoft YaHei', 12, 'bold'),
        #                                           command=self.toggle_auto_change,
        #                                           cursor="hand2")
        # self.auto_change_checkbox.pack(side=tk.LEFT, padx=5)
        
        # 创建中间视频显示区域
        center_frame = ttk.LabelFrame(main_frame, text="摄像头预览", padding="10")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 使用tk.Frame作为有背景色的容器
        center_inner = tk.Frame(center_frame, bg="#F5F5F5", padx=10, pady=10)
        center_inner.pack(fill=tk.BOTH, expand=True)
        
        # 创建视频显示标签
        self.video_label = tk.Label(center_inner, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # 创建状态栏
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, font=('Microsoft YaHei', 10))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 由于我们现在使用tk.Button而不是ttk.Button，因此不需要ttk样式定义
        # tk.Button可以直接设置背景色、前景色和字体等属性
        
        # 以下是tk.Button相关的配置已在按钮创建时直接应用
        # 包括背景色、前景色、字体、大小、边框和鼠标样式等
    
    def load_face_masks(self, face_dir: str = 'face') -> List[np.ndarray]:
        """加载所有脸谱图像"""
        self.face_masks = []
        self.mask_thumbnails = []
        self.mask_buttons = []
        
        # 支持 png 和 jpg 格式的图片
        image_files = glob.glob(os.path.join(face_dir, '*.png')) + glob.glob(os.path.join(face_dir, '*.jpg'))
        
        total_width = 0
        total_height = 0
        valid_masks = 0
        
        for image_file in image_files:
            mask_img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
            if mask_img is not None:
                self.face_masks.append(mask_img)
                # 计算所有脸谱的平均尺寸
                h, w = mask_img.shape[:2]
                total_width += w
                total_height += h
                valid_masks += 1
            else:
                print(f"无法加载脸谱图像: {image_file}")
        
        if not self.face_masks:
            messagebox.showerror("错误", "没有找到脸谱图像！请确保在'face'目录下有PNG或JPG格式的图像。")
            self.status_var.set("错误: 没有找到脸谱图像")
        else:
            # 计算脸谱的平均尺寸
            if valid_masks > 0:
                avg_width = total_width // valid_masks
                avg_height = total_height // valid_masks
                # 使用中值尺寸（取整数）
                self.average_mask_size = (avg_width, avg_height)
            else:
                self.average_mask_size = (200, 200)  # 默认尺寸
            
            self.status_var.set(f"就绪: 已加载{len(self.face_masks)}个脸谱")
            self.update_mask_preview()
        
        return self.face_masks
    
    def update_mask_preview(self):
        """更新脸谱预览和按钮"""
        if self.face_masks and 0 <= self.current_mask_index < len(self.face_masks):
            # 为所有脸谱创建缩略图和按钮（尺寸为prreview_size*prreview_size像素）
            button_size = (prreview_size, prreview_size)  # 按钮大小
            
            # 创建预览
            mask = self.face_masks[self.current_mask_index].copy()
            # 使用平均尺寸作为预览尺寸基础，确保等比缩放
            h, w = mask.shape[:2]
            scale = min(button_size[0] / w, button_size[1] / h)
            new_width = int(w * scale)
            new_height = int(h * scale)
            
            # 缩放图像
            mask = cv2.resize(mask, (new_width, new_height))
            
            # 创建一个白色背景的方形图像
            square_mask = np.ones((button_size[1], button_size[0], 3), dtype=np.uint8) * 255
            
            # 计算居中放置的坐标
            y_offset = (button_size[1] - new_height) // 2
            x_offset = (button_size[0] - new_width) // 2
            
            # 处理不同通道数的图像
            if len(mask.shape) == 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            elif mask.shape[2] == 4:
                # 分离Alpha通道
                b, g, r, a = cv2.split(mask)
                # 创建带有白色背景的RGB图像
                background = np.full((new_height, new_width, 3), 255, dtype=np.uint8)
                for i in range(3):
                    background[:, :, i] = (a / 255) * mask[:, :, i] + (1 - a / 255) * 255
                mask = background
            
            # 将缩放后的图像放置在中心
            square_mask[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = mask
            
            # 转换为PIL图像
            square_mask = cv2.cvtColor(square_mask, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(square_mask)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # 更新预览标签
            self.mask_preview.config(image=photo)
            self.mask_preview.image = photo
            self.status_var.set(f"当前脸谱: {self.current_mask_index + 1}/{len(self.face_masks)}")
        elif not self.face_masks:
            return
    
    def toggle_auto_change(self):
        """切换自动切换脸谱功能"""
        # self.auto_change_enabled = self.auto_change_var.get()
        
        if self.auto_change_enabled and self.is_running:
            # 启动自动切换定时器
            self.schedule_next_auto_change()
            self.status_var.set(f"自动切换已开启 ({self.config['auto_change_interval']}秒)")
        else:
            # 取消自动切换定时器
            if self.auto_change_timer:
                self.root.after_cancel(self.auto_change_timer)
                self.auto_change_timer = None
            if self.is_running:
                self.status_var.set(f"当前脸谱: {self.current_mask_index + 1}/{len(self.face_masks)}")
    
    def schedule_next_auto_change(self):
        """安排下一次自动切换"""
        if self.auto_change_enabled and self.is_running:
            self.auto_change_timer = self.root.after(self.config['auto_change_interval'] * 1000, 
                                                    self.auto_next_mask)
    
    def auto_next_mask(self):
        """自动切换到下一个脸谱"""
        if self.is_running and self.auto_change_enabled:
            self.next_mask()
            self.schedule_next_auto_change()  # 继续安排下一次切换
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """旋转图像"""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    def is_hand_near_face(
        self, 
        hand_landmarks, 
        face_landmarks, 
        image_width: int, 
        image_height: int
    ) -> bool:
        """检测手是否靠近脸部"""
        # 获取食指尖位置
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # 获取鼻尖位置
        nose_tip = face_landmarks.landmark[1]

        # 计算距离
        distance_x = abs(index_finger_tip.x - nose_tip.x) * image_width
        distance_y = abs(index_finger_tip.y - nose_tip.y) * image_height
        distance = hypot(distance_x, distance_y)
        
        # 使用图像对角线长度作为参考
        diagonal = hypot(image_width, image_height)
        return distance < diagonal * self.config['hand_face_threshold']
    
    def apply_mask(
        self, 
        image: np.ndarray, 
        mask_image: np.ndarray, 
        face_landmarks: List
    ) -> np.ndarray:
        """将脸谱应用到人脸上"""
        if mask_image is None or len(face_landmarks) < 468:
            return image

        h, w = image.shape[:2]
        
        # 计算脸谱旋转角度和缩放比例
        left_eye = face_landmarks[145]
        right_eye = face_landmarks[374]
        angle = -degrees(atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x))
        eye_distance_px = hypot(
            (right_eye.x * w) - (left_eye.x * w),
            (right_eye.y * h) - (left_eye.y * h)
        )
        scale = max(min(eye_distance_px / self.config['original_eye_distance'], 1), 0.1)
        
        try:
            # 调整脸谱大小和角度
            mask_resized = cv2.resize(mask_image, (0, 0), fx=scale, fy=scale)
            mask_rotated = self.rotate_image(mask_resized, angle)
        except Exception as e:
            print(f"调整脸谱大小或角度时出错: {e}")
            return image

        # 计算脸谱位置
        nose_tip = face_landmarks[1]
        center_x = int(nose_tip.x * w - mask_rotated.shape[1] // 2)
        center_y = int(nose_tip.y * h - mask_rotated.shape[0] // 2)

        # 处理边界情况
        start_y = max(center_y, 0)
        end_y = min(center_y + mask_rotated.shape[0], h)
        start_x = max(center_x, 0)
        end_x = min(center_x + mask_rotated.shape[1], w)

        mask_start_y = start_y - center_y
        mask_end_y = mask_start_y + (end_y - start_y)
        mask_start_x = start_x - center_x
        mask_end_x = mask_start_x + (end_x - start_x)

        # 应用脸谱
        if mask_rotated.shape[2] == 4:  # 带Alpha通道的图像
            for i in range(3):
                image[start_y:end_y, start_x:end_x, i] = (
                    image[start_y:end_y, start_x:end_x, i] * (1 - self.config['mask_alpha']) +
                    mask_rotated[mask_start_y:mask_end_y, mask_start_x:mask_end_x, i] * self.config['mask_alpha']
                ).astype(np.uint8)
        else:  # 无Alpha通道的图像
            for i in range(3):
                image[start_y:end_y, start_x:end_x, i] = cv2.addWeighted(
                    image[start_y:end_y, start_x:end_x, i],
                    1 - self.config['mask_alpha'],
                    mask_rotated[mask_start_y:mask_end_y, mask_start_x:mask_end_x, i],
                    self.config['mask_alpha'],
                    0
                )

        return image
    
    def process_frame(self, frame: np.ndarray, face_mesh, hands) -> Tuple[np.ndarray, bool]:
        """处理单帧图像"""
        if frame is None:
            return None, False

        # 预处理图像
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测人脸和手势
        results_faces = face_mesh.process(frame)
        results_hands = hands.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 处理检测结果
        if results_faces.multi_face_landmarks:
            for face_landmarks in results_faces.multi_face_landmarks:
                if not self.auto_change_enabled and results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        # 检查手是否靠近脸
                        h, w = frame.shape[:2]
                        if self.is_hand_near_face(hand_landmarks, face_landmarks, w, h):
                            self.root.after(0, self.next_mask)

                # 应用当前选中的脸谱
                frame = self.apply_mask(frame, self.face_masks[self.current_mask_index], face_landmarks.landmark)

        # 图像后处理 - 保持原始宽高比
        frame = cv2.flip(frame, 1)
        
        # 计算缩放比例，保持宽高比
        h, w = frame.shape[:2]
        target_width, target_height = self.config['display_size']
        
        # 计算缩放比例
        scale = min(target_width / w, target_height / h)
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        # 缩放图像
        frame = cv2.resize(frame, (new_width, new_height))
        
        # 创建一个黑色背景的目标尺寸图像
        result_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # 计算居中放置的坐标
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        # 将缩放后的图像放置在中心
        result_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame
        frame = result_frame

        return frame, True
    
    def start_camera(self):
        """启动摄像头"""
        if not self.face_masks:
            messagebox.showerror("错误", "没有找到脸谱图像！")
            return
            
        if not self.is_running:
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # 根据自动切换选项更新状态
            if self.auto_change_enabled:
                self.status_var.set(f"正在运行 (自动切换: {self.config['auto_change_interval']}秒)...")
                # 启动自动切换定时器
                self.schedule_next_auto_change()
            else:
                self.status_var.set("正在运行...")
            
            # 在新线程中运行摄像头处理
            self.thread = threading.Thread(target=self.run_camera)
            self.thread.daemon = True
            self.thread.start()
    
    def stop_camera(self):
        """停止摄像头"""
        if self.is_running:
            self.is_running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            # 等待线程结束
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)
            
            # 清理摄像头资源
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # 取消自动切换定时器
            if self.auto_change_timer:
                self.root.after_cancel(self.auto_change_timer)
                self.auto_change_timer = None
            
            self.status_var.set(f"已停止: {self.current_mask_index + 1}/{len(self.face_masks)}")
    
    def run_camera(self):
        """在单独线程中运行摄像头"""
        try:
            # 初始化摄像头
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("错误", "无法打开摄像头！"))
                self.root.after(0, self.stop_camera)
                return
            
            # 创建 MediaPipe 实例
            with self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh, \
                 self.mp_hands.Hands(
                     min_detection_confidence=0.5,
                     min_tracking_confidence=0.5) as hands:
                
                while self.is_running:
                    success, frame = self.cap.read()
                    if not success:
                        print("无法捕获帧")
                        time.sleep(0.1)
                        continue
                    
                    # 处理帧
                    processed_frame, _ = self.process_frame(frame, face_mesh, hands)
                    if processed_frame is None:
                        continue
                    
                    # 转换为RGB格式以便在Tkinter中显示
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # 转换为PIL图像
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # 转换为Tkinter图像
                    tk_image = ImageTk.PhotoImage(image=pil_image)
                    
                    # 更新视频标签
                    self.video_label.config(image=tk_image)
                    self.video_label.image = tk_image  # 保持引用，避免被垃圾回收
                    
                    # 短暂延迟以降低CPU使用率
                    time.sleep(0.03)
        except Exception as e:
            print(f"运行摄像头时出错: {e}")
            self.root.after(0, lambda: messagebox.showerror("错误", f"运行时出错: {str(e)}"))
            self.root.after(0, self.stop_camera)
    
    def next_mask(self):
        """切换到下一个脸谱"""
        if self.face_masks:
            self.current_mask_index = (self.current_mask_index + 1) % len(self.face_masks)
            self.update_mask_preview()
            # 更新状态信息
            if self.is_running:
                if self.auto_change_enabled:
                    self.status_var.set(f"自动切换已开启 ({self.config['auto_change_interval']}秒) - 当前脸谱: {self.current_mask_index + 1}/{len(self.face_masks)}")
                else:
                    self.status_var.set(f"当前脸谱: {self.current_mask_index + 1}/{len(self.face_masks)}")
    
    def update_alpha(self, value):
        """更新脸谱透明度"""
        self.config['mask_alpha'] = float(value)
        self.alpha_value.config(text=f"{self.config['mask_alpha']:.2f}")
    
    def on_closing(self):
        """窗口关闭时的处理"""
        self.stop_camera()
        self.root.destroy()
    

def main():
    """主函数"""
    # 创建Tkinter根窗口
    root = tk.Tk()
    
    # 创建应用实例
    app = FaceMaskGUI(root)
    
    # 启动主循环
    try:
        root.mainloop()
    except Exception as e:
        print(f"应用程序错误: {e}")
    finally:
        app.stop_camera()

if __name__ == "__main__":
    main()