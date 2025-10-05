import sys
import os
import glob
import cv2
import numpy as np
from math import hypot, atan2, degrees
from pathlib import Path
import time
import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QVBoxLayout, QHBoxLayout, QFrame, QGridLayout,
    QMessageBox, QStatusBar, QSizePolicy, QDialog, QScrollArea,
    QGroupBox, QRadioButton, QButtonGroup, QToolBar
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QRect
from PySide6.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QPainter, QPen

import mediapipe as mp

# 全局配置
PREVIEW_SIZE = 128
DISPLAY_SIZE = (1280, 720)
MASK_ALPHA_DEFAULT = 0.28
ORIGINAL_EYE_DISTANCE = 100
HAND_FACE_THRESHOLD = 0.1

# 功能模式枚举
class Mode:
    FACE_MASK = 0
    FACE_MESH = 1
    HAND_SKELETON = 2
    BODY_SKELETON = 3


class VideoWorker(QThread):
    """视频处理工作线程"""
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)
    fps_updated = Signal(float)
    face_detected = Signal(np.ndarray, bool)  # 发送帧和人脸检测状态
    mask_swapped = Signal()  # 脸谱切换信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_running = False
        self.face_masks = []
        self.current_mask_index = 0
        self.mask_alpha = MASK_ALPHA_DEFAULT
        self.bg_alpha = 1.0  # 背景透明度
        self.current_mode = Mode.FACE_MASK  # 当前工作模式
        
        # MediaPipe 模型
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        # 绘制工具
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 帧率计算
        self.frame_count = 0
        self.start_time = 0
        
        # 媒体捕获
        self.is_recording = False
        self.video_writer = None
        self.wechat_watermark = None
        self.last_face_switch_time = 0
        self.face_switch_cooldown = 0.5  # 脸谱切换冷却时间（秒）

    def set_running(self, running: bool):
        self._is_running = running

    def set_face_masks(self, masks):
        self.face_masks = masks

    def set_current_mask_index(self, idx):
        self.current_mask_index = idx

    def set_mask_alpha(self, alpha):
        self.mask_alpha = alpha
        
    def set_bg_alpha(self, alpha):
        self.bg_alpha = alpha
        
    def set_mode(self, mode):
        self.current_mode = mode
        
    def start_recording(self, output_path=None):
        # 如果没有提供输出路径，使用默认路径
        if output_path is None:
            videos_dir = Path("videos")
            videos_dir.mkdir(exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(videos_dir / f"video_{timestamp}.mp4")
            self.output_path = output_path
        
        self.is_recording = True
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, DISPLAY_SIZE)
        return output_path
        
    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        # 返回录制的视频路径
        if hasattr(self, 'output_path'):
            return self.output_path
        return None
        
    def take_screenshot(self, output_path):
        # 该方法会在主线程中调用，确保线程安全
        pass
        
    def load_watermark(self, watermark_path):
        if os.path.exists(watermark_path):
            self.wechat_watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
            # 调整水印大小
            if self.wechat_watermark is not None:
                h, w = self.wechat_watermark.shape[:2]
                scale = 0.1  # 缩放比例
                new_h, new_w = int(h * scale), int(w * scale)
                self.wechat_watermark = cv2.resize(self.wechat_watermark, (new_w, new_h))

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
    def draw_face_mesh(self, image, face_landmarks):
        if not face_landmarks:
            return image
        
        # 创建一个透明度混合的图层
        overlay = image.copy()
        
        # 绘制人脸网格
        for face_lm in face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=overlay,
                landmark_list=face_lm,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            self.mp_drawing.draw_landmarks(
                image=overlay,
                landmark_list=face_lm,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        # 根据bg_alpha调整原始图像的透明度
        result = cv2.addWeighted(image, self.bg_alpha, overlay, 1 - self.bg_alpha, 0)
        return result
        
    def draw_hand_skeleton(self, image, hand_landmarks):
        if not hand_landmarks:
            return image
        
        overlay = image.copy()
        
        # 绘制手部骨架
        for hand_lm in hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image=overlay,
                landmark_list=hand_lm,
                connections=mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        result = cv2.addWeighted(image, self.bg_alpha, overlay, 1 - self.bg_alpha, 0)
        return result
        
    def draw_body_skeleton(self, image, pose_landmarks):
        if not pose_landmarks:
            return image
        
        overlay = image.copy()
        
        # 绘制人体骨架
        for pose_lm in pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image=overlay,
                landmark_list=pose_lm,
                connections=mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_pose_connections_style()
            )
        
        result = cv2.addWeighted(image, self.bg_alpha, overlay, 1 - self.bg_alpha, 0)
        return result
        
    def add_watermark(self, image):
        if self.wechat_watermark is None:
            return image
        
        h, w = image.shape[:2]
        watermark_h, watermark_w = self.wechat_watermark.shape[:2]
        
        # 将水印放在右下角
        x_pos = w - watermark_w - 10
        y_pos = h - watermark_h - 10
        
        # 确保水印位置有效
        x_pos = max(0, x_pos)
        y_pos = max(0, y_pos)
        
        # 处理带alpha通道的水印
        if self.wechat_watermark.shape[2] == 4:
            # 分离通道
            b, g, r, a = cv2.split(self.wechat_watermark)
            # 创建RGB版本的水印
            rgb_watermark = cv2.merge((b, g, r))
            # 创建alpha蒙版
            alpha = a / 255.0
            
            # 获取图像上对应位置的区域
            roi = image[y_pos:y_pos+watermark_h, x_pos:x_pos+watermark_w]
            
            # 应用水印
            for c in range(3):
                roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * rgb_watermark[:, :, c]
        else:
            # 直接叠加
            image[y_pos:y_pos+watermark_h, x_pos:x_pos+watermark_w] = self.wechat_watermark
        
        return image
        
    def is_hand_near_face(self, hand_landmarks, face_landmarks, img_w, img_h) -> bool:
        current_time = time.time()
        # 检查冷却时间
        if current_time - self.last_face_switch_time < self.face_switch_cooldown:
            return False
            
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        nose_tip = face_landmarks.landmark[1]
        dx = abs(index_tip.x - nose_tip.x) * img_w
        dy = abs(index_tip.y - nose_tip.y) * img_h
        dist = hypot(dx, dy)
        diagonal = hypot(img_w, img_h)
        
        if dist < diagonal * HAND_FACE_THRESHOLD:
            self.last_face_switch_time = current_time
            return True
        
        return False

    def apply_mask(self, image, mask_img, face_landmarks):
        if mask_img is None or len(face_landmarks) < 468:
            return image

        h, w = image.shape[:2]
        left_eye = face_landmarks[145]
        right_eye = face_landmarks[374]
        angle = -degrees(atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x))
        eye_dist_px = hypot(
            (right_eye.x * w) - (left_eye.x * w),
            (right_eye.y * h) - (left_eye.y * h)
        )
        scale = max(min(eye_dist_px / ORIGINAL_EYE_DISTANCE, 1.0), 0.1)

        try:
            resized = cv2.resize(mask_img, (0, 0), fx=scale, fy=scale)
            rotated = self.rotate_image(resized, angle)
        except Exception as e:
            print(f"Mask resize/rotate error: {e}")
            return image

        nose = face_landmarks[1]
        cx = int(nose.x * w - rotated.shape[1] // 2)
        cy = int(nose.y * h - rotated.shape[0] // 2)

        start_y = max(cy, 0)
        end_y = min(cy + rotated.shape[0], h)
        start_x = max(cx, 0)
        end_x = min(cx + rotated.shape[1], w)

        mask_sy = start_y - cy
        mask_ey = mask_sy + (end_y - start_y)
        mask_sx = start_x - cx
        mask_ex = mask_sx + (end_x - start_x)

        roi = image[start_y:end_y, start_x:end_x]
        mask_roi = rotated[mask_sy:mask_ey, mask_sx:mask_ex]

        if mask_roi.size == 0 or roi.size == 0:
            return image

        if mask_roi.shape[2] == 4:
            alpha = mask_roi[:, :, 3] / 255.0 * self.mask_alpha
            for c in range(3):
                roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * mask_roi[:, :, c]
        else:
            blended = cv2.addWeighted(roi, 1 - self.mask_alpha, mask_roi, self.mask_alpha, 0)
            image[start_y:end_y, start_x:end_x] = blended

        return image

    def run(self):
        # 尝试打开默认摄像头(索引0)
        cap = cv2.VideoCapture(0)
        
        # 如果默认摄像头打开失败，尝试其他常见索引
        if not cap.isOpened():
            for i in range(1, 3):  # 尝试索引1和2
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    break
        
        # 如果所有尝试都失败，发出错误信号
        if not cap.isOpened():
            self.error_occurred.emit("无法打开摄像头！请检查设备连接和权限。")
            return
        
        # 设置摄像头分辨率（如果支持）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 加载微信水印
        self.load_watermark("wechat.jpg")
        
        # 帧率计算初始化
        self.frame_count = 0
        self.start_time = time.time()

        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh, self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands, self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            while self._is_running:
                success, frame = cap.read()
                if not success:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 根据当前模式处理图像
                if self.current_mode == Mode.FACE_MASK:
                    results_face = face_mesh.process(frame)
                    results_hand = hands.process(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    hand_near_face = False
                    if results_face.multi_face_landmarks and results_hand.multi_hand_landmarks:
                        face_lm = results_face.multi_face_landmarks[0]
                        for hand_lm in results_hand.multi_hand_landmarks:
                            if self.is_hand_near_face(hand_lm, face_lm, frame.shape[1], frame.shape[0]):
                                hand_near_face = True
                                break

                    # 发送信号让主线程切换脸谱（线程安全）
                    if hand_near_face:
                        self.frame_ready.emit(None)  # None 表示“请求切换”

                    if results_face.multi_face_landmarks:
                        face_lm = results_face.multi_face_landmarks[0]
                        if self.face_masks and 0 <= self.current_mask_index < len(self.face_masks):
                            mask = self.face_masks[self.current_mask_index]
                            frame = self.apply_mask(frame, mask, face_lm.landmark)
                
                elif self.current_mode == Mode.FACE_MESH:
                    results_face = face_mesh.process(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = self.draw_face_mesh(frame, results_face.multi_face_landmarks)
                
                elif self.current_mode == Mode.HAND_SKELETON:
                    results_hand = hands.process(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = self.draw_hand_skeleton(frame, results_hand.multi_hand_landmarks)
                
                elif self.current_mode == Mode.BODY_SKELETON:
                    results_pose = pose.process(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = self.draw_body_skeleton(frame, results_pose.multi_pose_landmarks)

                # 镜像 + 缩放保持比例
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                target_w, target_h = DISPLAY_SIZE
                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (new_w, new_h))
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                y_off = (target_h - new_h) // 2
                x_off = (target_w - new_w) // 2
                canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

                # 计算并更新帧率
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 1.0:  # 每秒更新一次
                    fps = self.frame_count / elapsed_time
                    self.fps_updated.emit(fps)
                    self.frame_count = 0
                    self.start_time = time.time()

                # 如果正在录制，写入视频
                if self.is_recording and self.video_writer:
                    # 添加水印
                    video_frame = self.add_watermark(canvas.copy())
                    self.video_writer.write(video_frame)

                self.frame_ready.emit(canvas)

                self.msleep(30)

        cap.release()


class FaceMaskApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸换脸谱 - PySide6 版")
        self.resize(1400, 950)

        # 状态
        self.face_masks = []
        self.current_mask_index = 0
        self.is_running = False
        self.current_mode = Mode.FACE_MASK
        self.is_recording = False
        self.worker = None
        self.is_camera_running = False

        # 创建 UI
        self.init_ui()
        
        # 加载资源
        self.load_face_masks()

        # 设置连接
        self.setup_connections()

        # 初始预览
        self.update_mask_preview()
        
    def setup_connections(self):
        """设置信号和槽的连接"""
        # 创建工作线程并连接信号
        if self.worker is None:
            self.worker = VideoWorker()
            self.worker.frame_ready.connect(self.on_frame_ready)
            self.worker.error_occurred.connect(self.on_error)
            self.worker.fps_updated.connect(self.update_fps_display)
            self.worker.face_detected.connect(self.on_face_detected)
            self.worker.mask_swapped.connect(self.on_mask_swapped)

    def init_ui(self):
        # 主窗口背景色
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#FFF8E1"))
        self.setPalette(palette)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # === 顶部工具栏 ===
        top_toolbar = QToolBar("顶部工具栏")
        top_toolbar.setFixedHeight(60)
        main_layout.addWidget(top_toolbar)

        # 核心功能切换区
        function_buttons = []

        # 川剧变脸
        face_mask_btn = QPushButton("川剧变脸")
        face_mask_btn.setCheckable(True)
        face_mask_btn.setChecked(True)  # 默认选中
        face_mask_btn.clicked.connect(lambda: self.switch_mode(Mode.FACE_MASK))
        function_buttons.append(face_mask_btn)
        top_toolbar.addWidget(face_mask_btn)

        # 人脸网格
        face_mesh_btn = QPushButton("人脸网格")
        face_mesh_btn.setCheckable(True)
        face_mesh_btn.clicked.connect(lambda: self.switch_mode(Mode.FACE_MESH))
        function_buttons.append(face_mesh_btn)
        top_toolbar.addWidget(face_mesh_btn)

        # 手指骨架
        hand_skeleton_btn = QPushButton("手指骨架")
        hand_skeleton_btn.setCheckable(True)
        hand_skeleton_btn.clicked.connect(lambda: self.switch_mode(Mode.HAND_SKELETON))
        function_buttons.append(hand_skeleton_btn)
        top_toolbar.addWidget(hand_skeleton_btn)

        # 人体骨架
        body_skeleton_btn = QPushButton("人体骨架")
        body_skeleton_btn.setCheckable(True)
        body_skeleton_btn.clicked.connect(lambda: self.switch_mode(Mode.BODY_SKELETON))
        function_buttons.append(body_skeleton_btn)
        top_toolbar.addWidget(body_skeleton_btn)

        # 为功能按钮创建按钮组，确保互斥
        function_group = QButtonGroup(self)
        for btn in function_buttons:
            function_group.addButton(btn)
        function_group.setExclusive(True)

        # 分隔符
        top_toolbar.addSeparator()

        # 媒体捕获按钮
        screenshot_btn = QPushButton("截图")
        screenshot_btn.clicked.connect(self.take_screenshot)
        top_toolbar.addWidget(screenshot_btn)

        self.record_btn = QPushButton("录制视频")
        self.record_btn.clicked.connect(self.toggle_recording)
        top_toolbar.addWidget(self.record_btn)

        # 分隔符
        top_toolbar.addSeparator()

        # 关于我按钮
        about_btn = QPushButton("关于我")
        about_btn.clicked.connect(self.show_about_dialog)
        top_toolbar.addWidget(about_btn)

        # === 中央区域 ===
        center_widget = QWidget()
        center_layout = QHBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(center_widget, 1)

        # 视频显示区域
        video_frame = QFrame()
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(*DISPLAY_SIZE)
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        video_layout.addWidget(self.video_label)
        center_layout.addWidget(video_frame, 1)

        # 右侧透明度滑块
        right_panel = QWidget()
        right_panel.setFixedWidth(100)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        alpha_label = QLabel("透明度:")
        alpha_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        alpha_label.setAlignment(Qt.AlignCenter)

        self.alpha_slider = QSlider(Qt.Vertical)
        self.alpha_slider.setRange(10, 90)
        self.alpha_slider.setValue(int(MASK_ALPHA_DEFAULT * 100))
        self.alpha_slider.setTickPosition(QSlider.TicksBothSides)
        self.alpha_slider.setTickInterval(20)
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)

        self.alpha_value_label = QLabel(f"{MASK_ALPHA_DEFAULT:.2f}")
        self.alpha_value_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.alpha_value_label.setAlignment(Qt.AlignCenter)

        right_layout.addWidget(alpha_label)
        right_layout.addWidget(self.alpha_slider)
        right_layout.addWidget(self.alpha_value_label)
        right_layout.addStretch()

        center_layout.addWidget(right_panel)

        # === 脸谱展示区域（仅川剧变脸模式显示）=== 
        self.masks_scroll_area = QScrollArea()
        self.masks_scroll_area.setWidgetResizable(True)
        self.masks_scroll_area.setFixedHeight(120)
        self.masks_container = QWidget()
        self.masks_layout = QHBoxLayout(self.masks_container)
        self.masks_layout.setContentsMargins(10, 10, 10, 10)
        self.masks_layout.setSpacing(10)
        self.masks_scroll_area.setWidget(self.masks_container)
        main_layout.addWidget(self.masks_scroll_area)

        # === 开关式控制按钮 ===
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(control_widget)

        self.toggle_button = QPushButton("开始")
        self.toggle_button.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        self.toggle_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 15px 40px;")
        self.toggle_button.clicked.connect(self.toggle_camera)
        control_layout.addWidget(self.toggle_button)

        # === 状态栏 ===
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 状态栏显示项
        self.camera_status_label = QLabel("摄像头: 未开启")
        self.fps_label = QLabel("FPS: 0.0")
        self.status_bar.addWidget(self.camera_status_label)
        self.status_bar.addWidget(QLabel(" | "))
        self.status_bar.addWidget(self.fps_label)
        self.status_bar.showMessage("就绪")

    def load_face_masks(self):
        face_dir = Path("face")
        if not face_dir.exists():
            face_dir.mkdir()
        image_files = list(face_dir.glob("*.png")) + list(face_dir.glob("*.jpg"))
        self.face_masks = []
        for f in image_files:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is not None:
                self.face_masks.append(img)
        if not self.face_masks:
            QMessageBox.critical(self, "错误", "未在 'face' 目录中找到 PNG/JPG 图像！")
            self.status_bar.showMessage("错误: 未找到脸谱图像")
        else:
            self.status_bar.showMessage(f"就绪: 已加载 {len(self.face_masks)} 个脸谱")

    def update_mask_preview(self):
        """更新脸谱预览区域，显示所有脸谱缩略图"""
        if not self.face_masks:
            return
            
        # 清空现有内容
        for i in reversed(range(self.masks_layout.count())):
            widget = self.masks_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
                
        # 为每个脸谱创建缩略图
        THUMBNAIL_SIZE = 100
        
        for i, mask in enumerate(self.face_masks):
            mask_copy = mask.copy()
            h, w = mask_copy.shape[:2]
            scale = min(THUMBNAIL_SIZE / w, THUMBNAIL_SIZE / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(mask_copy, (new_w, new_h))

            # 白底方形
            square = np.ones((THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3), dtype=np.uint8) * 255
            y_off = (THUMBNAIL_SIZE - new_h) // 2
            x_off = (THUMBNAIL_SIZE - new_w) // 2

            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            elif resized.shape[2] == 4:
                b, g, r, a = cv2.split(resized)
                bg = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
                for j in range(3):
                    bg[:, :, j] = (a / 255.0) * resized[:, :, j] + (1 - a / 255.0) * 255
                resized = bg

            square[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            square_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            h_sq, w_sq, ch_sq = square_rgb.shape
            qimg = QImage(square_rgb.data, w_sq, h_sq, ch_sq * w_sq, QImage.Format_RGB888)
            
            # 创建缩略图标签
            thumb_label = QLabel()
            thumb_label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
            thumb_label.setPixmap(QPixmap.fromImage(qimg))
            
            # 当前选中的脸谱添加边框
            if i == self.current_mask_index:
                thumb_label.setStyleSheet("border: 3px solid #4CAF50;")
            else:
                thumb_label.setStyleSheet("border: 1px solid #ccc;")
                
            # 添加点击事件
            thumb_label.mask_index = i  # 存储脸谱索引
            thumb_label.mousePressEvent = lambda event, idx=i: self.on_thumbnail_clicked(idx)
            
            # 添加到布局
            self.masks_layout.addWidget(thumb_label)
            
        self.status_bar.showMessage(f"当前脸谱: {self.current_mask_index + 1}/{len(self.face_masks)}")
        
    def on_thumbnail_clicked(self, index):
        """点击缩略图切换脸谱"""
        if 0 <= index < len(self.face_masks):
            self.current_mask_index = index
            self.update_mask_preview()
            if self.worker and self.worker.isRunning():
                self.worker.set_current_mask_index(index)

    @Slot(int)
    def on_alpha_changed(self, value):
        alpha = value / 100.0
        self.alpha_value_label.setText(f"{alpha:.2f}")
        if hasattr(self, 'worker'):
            self.worker.set_mask_alpha(alpha)

    @Slot(np.ndarray)
    def on_frame_ready(self, frame):
        # 如果frame为None，表示请求切换脸谱
        if frame is None:
            # 切换到下一个脸谱
            if self.face_masks:
                self.current_mask_index = (self.current_mask_index + 1) % len(self.face_masks)
                self.update_mask_preview()
                if self.worker and self.worker.isRunning():
                    self.worker.set_current_mask_index(self.current_mask_index)
            return
        
        # 正常显示视频帧
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))
        
    @Slot(float)
    def update_fps_display(self, fps):
        """更新FPS显示"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
    @Slot(bool)
    def on_face_detected(self, detected):
        """人脸检测回调"""
        if detected and self.current_mode == Mode.FACE_MASK:
            self.status_bar.showMessage("检测到人脸")
        else:
            if self.is_camera_running:
                mode_names = {Mode.FACE_MASK: "川剧变脸", Mode.FACE_MESH: "人脸网格", 
                            Mode.HAND_SKELETON: "手指骨架", Mode.BODY_SKELETON: "人体骨架"}
                self.status_bar.showMessage(f"摄像头运行中 - {mode_names[self.current_mode]}")
                
    @Slot()
    def on_mask_swapped(self):
        """脸谱切换回调"""
        if self.worker and self.current_mode == Mode.FACE_MASK:
            self.current_mask_index = self.worker.current_mask_index
            self.update_mask_preview()
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    @Slot(str)
    def on_error(self, msg):
        QMessageBox.critical(self, "错误", msg)
        self.stop_camera()

    def next_mask(self):
        if self.face_masks:
            self.current_mask_index = (self.current_mask_index + 1) % len(self.face_masks)
            self.update_mask_preview()
            if hasattr(self, 'worker'):
                self.worker.set_current_mask_index(self.current_mask_index)
                
    def toggle_camera(self):
        """切换摄像头开启/关闭状态"""
        if self.is_camera_running:
            self.stop_camera()
        else:
            self.start_camera()
            
    def switch_mode(self, mode):
        """切换功能模式"""
        self.current_mode = mode
        # 根据模式控制界面元素可见性
        if mode == Mode.FACE_MASK:
            self.masks_scroll_area.show()
        else:
            self.masks_scroll_area.hide()
            
        # 如果摄像头正在运行，更新工作线程的模式
        if self.worker and self.worker.isRunning():
            self.worker.set_mode(mode)
            
    def take_screenshot(self):
        """拍摄当前画面截图"""
        if not self.is_camera_running:
            QMessageBox.information(self, "提示", "请先开启摄像头！")
            return
            
        # 确保有图像
        if not hasattr(self, 'video_label') or self.video_label.pixmap() is None:
            QMessageBox.information(self, "提示", "当前没有可截图的画面！")
            return
            
        # 创建screenshots目录
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = screenshots_dir / f"screenshot_{timestamp}.png"
        
        # 保存截图
        if self.video_label.pixmap().save(str(filename)):
            self.status_bar.showMessage(f"截图已保存至: {filename}")
        else:
            QMessageBox.critical(self, "错误", "无法保存截图！")
            
    def toggle_recording(self):
        """切换视频录制状态"""
        if not self.is_camera_running:
            QMessageBox.information(self, "提示", "请先开启摄像头！")
            return
            
        self.is_recording = not self.is_recording
        
        if self.is_recording:
            # 开始录制
            if self.worker:
                self.worker.start_recording()
                self.record_btn.setText("停止录制")
                self.record_btn.setStyleSheet("color: red;")
                self.status_bar.showMessage("开始录制视频...")
        else:
            # 停止录制
            if self.worker:
                video_path = self.worker.stop_recording()
                self.record_btn.setText("录制视频")
                self.record_btn.setStyleSheet("")
                if video_path:
                    self.status_bar.showMessage(f"视频已保存至: {video_path}")
                else:
                    self.status_bar.showMessage("停止录制")
                    
    def show_about_dialog(self):
        """显示关于对话框"""
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("关于")
        about_dialog.resize(400, 300)
        
        layout = QVBoxLayout(about_dialog)
        
        title_label = QLabel("川剧变脸 - 特效相机")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        
        info_label = QLabel("基于OpenCV和MediaPipe的实时人脸特效应用\n\n"
                           "功能特点：\n"
                           "- 川剧变脸：通过手势切换脸谱\n"
                           "- 人脸网格：显示人脸关键点网格\n"
                           "- 手指骨架：显示手部骨架\n"
                           "- 人体骨架：显示人体姿态骨架\n"
                           "- 截图功能：保存当前画面\n"
                           "- 视频录制：录制特效视频")
        info_label.setAlignment(Qt.AlignLeft)
        info_label.setWordWrap(True)
        
        version_label = QLabel("版本 1.0.0")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        
        layout.addWidget(title_label)
        layout.addSpacing(20)
        layout.addWidget(info_label)
        layout.addStretch()
        layout.addWidget(version_label)
        
        about_dialog.exec()

    def start_camera(self):
        """启动摄像头"""
        # 对于川剧变脸模式，需要检查是否有脸谱
        if self.current_mode == Mode.FACE_MASK and not self.face_masks:
            QMessageBox.critical(self, "错误", "请先加载脸谱图像！")
            return
            
        if self.is_camera_running:
            return
            
        # 创建新的工作线程
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            
        self.worker = VideoWorker()
        self.setup_connections()
        
        # 配置工作线程
        self.worker.set_running(True)
        self.worker.set_face_masks(self.face_masks)
        self.worker.set_current_mask_index(self.current_mask_index)
        self.worker.set_mask_alpha(self.alpha_slider.value() / 100.0)
        self.worker.set_mode(self.current_mode)
        
        # 启动线程
        self.worker.start()
        
        # 更新UI状态
        self.is_camera_running = True
        self.camera_status_label.setText("摄像头: 开启")
        self.toggle_button.setText("停止")
        self.toggle_button.setStyleSheet("background-color: #F44336; color: white; padding: 15px 40px;")
        
        # 更新状态消息
        mode_names = {Mode.FACE_MASK: "川剧变脸", Mode.FACE_MESH: "人脸网格", 
                    Mode.HAND_SKELETON: "手指骨架", Mode.BODY_SKELETON: "人体骨架"}
        self.status_bar.showMessage(f"摄像头运行中 - {mode_names[self.current_mode]}")

    def stop_camera(self):
        """停止摄像头"""
        if not self.is_camera_running or not self.worker:
            return
            
        # 停止录制（如果正在进行）
        if self.is_recording:
            self.toggle_recording()
            
        # 停止工作线程
        self.worker.set_running(False)
        
        # 等待线程结束（短超时）
        self.worker.wait(2000)
        
        # 更新UI状态
        self.is_camera_running = False
        self.camera_status_label.setText("摄像头: 未开启")
        self.toggle_button.setText("开始")
        self.toggle_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 15px 40px;")
        
        # 更新状态消息
        self.status_bar.showMessage("就绪")

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = FaceMaskApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()