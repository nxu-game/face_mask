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
    QGroupBox, QRadioButton, QButtonGroup, QToolBar, QProgressBar
)
from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot, QTimer, QRect
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QPainter, QPen, QIcon

import mediapipe as mp

# 全局配置
PREVIEW_SIZE = 128
DISPLAY_SIZE = (1280, 720)
MASK_ALPHA_DEFAULT = 0.28
ORIGINAL_EYE_DISTANCE = 100
# 提高切换脸谱的灵敏度，从0.1增加到0.15
HAND_FACE_THRESHOLD = 1

# 功能模式枚举
class Mode:
    FACE_MASK = 0
    FACE_MESH = 1
    HAND_SKELETON = 2
    BODY_SKELETON = 3


class ResourceLoader(QObject):
    """资源加载器，在单独线程中加载资源"""
    progress_updated = Signal(int, str)  # 进度值, 状态消息
    loading_completed = Signal()         # 加载完成
    loading_failed = Signal(str)         # 加载失败，附带错误信息
    finished = Signal()                  # 任务完成信号
    
    def __init__(self, mode, parent_app):
        super().__init__()
        self.mode = mode
        self.parent_app = parent_app
    
    @Slot()
    def load_resources(self):
        """加载对应模式的资源"""
        try:
            if self.mode == Mode.FACE_MASK:
                # 加载川剧变脸模式所需资源
                self.progress_updated.emit(0, "准备加载川剧变脸资源...")
                time.sleep(0.1)  # 短暂延迟以便UI更新
                
                self.progress_updated.emit(20, "正在加载脸谱图像...")
                # 模拟进度更新
                for i in range(20, 80):
                    time.sleep(0.01)  # 模拟加载延迟
                    self.progress_updated.emit(i, "正在加载脸谱图像...")
                
                # 实际加载资源
                self.parent_app.load_face_masks()
                self.progress_updated.emit(80, "脸谱图像加载完成")
                
                time.sleep(0.1)  # 短暂延迟
                self.progress_updated.emit(100, "资源加载完成")
                
            elif self.mode == Mode.FACE_MESH:
                # 加载人脸网格模式所需资源
                self.progress_updated.emit(0, "准备加载人脸网格资源...")
                time.sleep(0.1)
                self.progress_updated.emit(50, "正在初始化人脸检测模型...")
                time.sleep(0.2)
                self.progress_updated.emit(100, "人脸网格资源加载完成")
                
            elif self.mode == Mode.HAND_SKELETON:
                # 加载手部骨架模式所需资源
                self.progress_updated.emit(0, "准备加载手部骨架资源...")
                time.sleep(0.1)
                self.progress_updated.emit(50, "正在初始化手部检测模型...")
                time.sleep(0.2)
                self.progress_updated.emit(100, "手部骨架资源加载完成")
                
            elif self.mode == Mode.BODY_SKELETON:
                # 加载人体骨架模式所需资源
                self.progress_updated.emit(0, "准备加载人体骨架资源...")
                time.sleep(0.1)
                self.progress_updated.emit(50, "正在初始化人体检测模型...")
                time.sleep(0.2)
                self.progress_updated.emit(100, "人体骨架资源加载完成")
                
            # 发出完成信号
            self.loading_completed.emit()
            
        except Exception as e:
            # 发出失败信号
            self.loading_failed.emit(str(e))
        finally:
            # 发出完成信号
            self.finished.emit()


class VideoWorker(QThread):
    """视频处理工作线程"""
    frame_ready = Signal(np.ndarray)          # 发送处理后的帧（BGR）
    error_occurred = Signal(str)
    fps_updated = Signal(float)
    request_mask_switch = Signal()            # 请求切换脸谱（线程安全）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_running = False
        self.face_masks = []
        self.current_mask_index = 0
        self.mask_alpha = MASK_ALPHA_DEFAULT
        self.bg_alpha = 1.0
        self.current_mode = Mode.FACE_MASK

        # MediaPipe 模型（延迟初始化）
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 帧率计算
        self.frame_count = 0
        self.start_time = time.time()

        # 录制相关
        self.is_recording = False
        self.video_writer = None
        self.wechat_watermark = None
        self.last_face_switch_time = 0
        self.face_switch_cooldown = 0.5

        # 摄像头对象（在 run 中初始化）
        self.cap = None

    def __del__(self):
        # 确保资源释放
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()

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
        if output_path is None:
            videos_dir = Path("videos")
            videos_dir.mkdir(exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(videos_dir / f"video_{timestamp}.mp4")
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, DISPLAY_SIZE)
        self.is_recording = True
        return output_path

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        return getattr(self, 'output_path', None)

    def load_watermark(self, watermark_path):
        if os.path.exists(watermark_path):
            self.wechat_watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
            if self.wechat_watermark is not None:
                h, w = self.wechat_watermark.shape[:2]
                scale = 0.1
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
        overlay = np.zeros_like(image)
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
        # 反转透明度逻辑，使其与用户期望一致：0%表示网格完全可见，100%表示网格完全透明
        # 现在控制摄像头画面的透明度，而不是网格的透明度
        return cv2.addWeighted(image, 1.0 - self.mask_alpha, overlay, 1.0, 0)

    def draw_hand_skeleton(self, image, hand_landmarks):
        if not hand_landmarks:
            return image
        overlay = np.zeros_like(image)
        for hand_lm in hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image=overlay,
                landmark_list=hand_lm,
                connections=mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )
        # 反转透明度逻辑，使其与用户期望一致：0%表示骨架完全可见，100%表示骨架完全透明
        # 现在控制摄像头画面的透明度，而不是骨架的透明度
        return cv2.addWeighted(image, 1.0 - self.mask_alpha, overlay, 1.0, 0)

    def draw_body_skeleton(self, image, pose_landmarks):
        if not pose_landmarks:
            return image
        overlay = np.zeros_like(image)
        # 处理单个pose_landmarks对象（不是列表）
        self.mp_drawing.draw_landmarks(
            image=overlay,
            landmark_list=pose_landmarks,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        # 反转透明度逻辑，使其与用户期望一致：0%表示骨架完全可见，100%表示骨架完全透明
        # 现在控制摄像头画面的透明度，而不是骨架的透明度
        return cv2.addWeighted(image, 1.0 - self.mask_alpha, overlay, 1.0, 0)

    def add_watermark(self, image):
        if self.wechat_watermark is None:
            return image
        h, w = image.shape[:2]
        watermark_h, watermark_w = self.wechat_watermark.shape[:2]
        x_pos = max(0, w - watermark_w - 10)
        y_pos = max(0, h - watermark_h - 10)
        roi = image[y_pos:y_pos+watermark_h, x_pos:x_pos+watermark_w]
        if self.wechat_watermark.shape[2] == 4:
            b, g, r, a = cv2.split(self.wechat_watermark)
            rgb_watermark = cv2.merge((b, g, r))
            alpha = a.astype(np.float32) / 255.0
            for c in range(3):
                roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * rgb_watermark[:, :, c]
        else:
            image[y_pos:y_pos+watermark_h, x_pos:x_pos+watermark_w] = self.wechat_watermark
        return image

    def is_hand_near_face(self, hand_landmarks, face_landmarks, img_w, img_h) -> bool:
        current_time = time.time()
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
        eye_dist_px = hypot((right_eye.x * w) - (left_eye.x * w), (right_eye.y * h) - (left_eye.y * h))
        # 移除缩放上限，让脸谱可以随人脸变大而相应变大
        scale = max(eye_dist_px / ORIGINAL_EYE_DISTANCE, 0.1)
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
        # 反转透明度逻辑，使其与用户期望一致：0%表示完全显示脸谱，100%表示完全透明
        inverted_alpha = 1.0 - self.mask_alpha
        if mask_roi.shape[2] == 4:
            alpha = mask_roi[:, :, 3].astype(np.float32) / 255.0 * inverted_alpha
            for c in range(3):
                roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * mask_roi[:, :, c]
        else:
            blended = cv2.addWeighted(roi, self.mask_alpha, mask_roi, inverted_alpha, 0)
            image[start_y:end_y, start_x:end_x] = blended
        return image

    def scale_and_mirror(self, frame):
        """统一处理镜像 + 等比缩放至 DISPLAY_SIZE"""
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
        return canvas

    def run(self):
        # 尝试打开摄像头（增加更多摄像头索引尝试）
        camera_indices = [0, 1, 2]
        self.cap = None
        
        for i in camera_indices:
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                print(f"成功打开摄像头，索引: {i}")
                break
            else:
                self.cap.release()
                print(f"尝试打开摄像头索引 {i} 失败")
                
        if not self.cap or not self.cap.isOpened():
            print("无法打开任何摄像头！")
            self.error_occurred.emit("无法打开摄像头！请检查设备连接和权限。")
            return

        # 设置较低的分辨率以提高性能
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 打印实际分辨率
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"摄像头实际分辨率: {actual_width}x{actual_height}")
        
        self.load_watermark("wechat.jpg")

        self.frame_count = 0
        self.start_time = time.time()
        frame_emit_count = 0
        
        # 初始化模型为None，按需加载
        face_mesh = None
        hands = None
        pose = None
        
        try:
            while self._is_running:
                success, frame = self.cap.read()
                if not success:
                    print("未成功读取视频帧")
                    continue

                # 确保帧不为空
                if frame is None or frame.size == 0:
                    print("读取到空视频帧")
                    continue

                # 统一在 RGB 空间处理
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                processed_frame = frame.copy()  # BGR

                try:
                    if self.current_mode == Mode.FACE_MASK:
                        # 按需加载人脸和手部模型
                        if face_mesh is None:
                            face_mesh = self.mp_face_mesh.FaceMesh(
                                max_num_faces=1, refine_landmarks=True,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5
                            )
                        if hands is None:
                            hands = self.mp_hands.Hands(
                                min_detection_confidence=0.5, min_tracking_confidence=0.5
                            )
                        
                        results_face = face_mesh.process(frame_rgb)
                        results_hand = hands.process(frame_rgb)

                        if results_face.multi_face_landmarks:
                            face_lm = results_face.multi_face_landmarks[0]
                            if self.face_masks and 0 <= self.current_mask_index < len(self.face_masks):
                                mask = self.face_masks[self.current_mask_index]
                                processed_frame = self.apply_mask(processed_frame, mask, face_lm.landmark)

                            if results_hand.multi_hand_landmarks:
                                for hand_lm in results_hand.multi_hand_landmarks:
                                    if self.is_hand_near_face(hand_lm, face_lm, frame.shape[1], frame.shape[0]):
                                        self.request_mask_switch.emit()
                                        break
                        else:
                            processed_frame = frame.copy()

                    elif self.current_mode == Mode.FACE_MESH:
                        # 按需加载人脸模型
                        if face_mesh is None:
                            face_mesh = self.mp_face_mesh.FaceMesh(
                                max_num_faces=1, refine_landmarks=True,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5
                            )
                        results_face = face_mesh.process(frame_rgb)
                        processed_frame = self.draw_face_mesh(frame.copy(), results_face.multi_face_landmarks)

                    elif self.current_mode == Mode.HAND_SKELETON:
                        # 按需加载手部模型
                        if hands is None:
                            hands = self.mp_hands.Hands(
                                min_detection_confidence=0.5, min_tracking_confidence=0.5
                            )
                        results_hand = hands.process(frame_rgb)
                        processed_frame = self.draw_hand_skeleton(frame.copy(), results_hand.multi_hand_landmarks)

                    elif self.current_mode == Mode.BODY_SKELETON:
                        # 按需加载身体模型
                        if pose is None:
                            pose = self.mp_pose.Pose(
                                min_detection_confidence=0.5, min_tracking_confidence=0.5
                            )
                        results_pose = pose.process(frame_rgb)
                        # 修复属性错误：使用正确的属性名
                        processed_frame = self.draw_body_skeleton(frame.copy(), results_pose.pose_landmarks)
                except Exception as process_error:
                    print(f"帧处理错误: {process_error}")
                    # 出错时使用原始帧
                    processed_frame = frame.copy()

                # 统一缩放镜像
                try:
                    display_frame = self.scale_and_mirror(processed_frame)
                except Exception as scale_error:
                    print(f"缩放镜像错误: {scale_error}")
                    # 出错时使用原始帧
                    display_frame = frame.copy()

                # 帧率更新
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 1.0:
                    fps = self.frame_count / elapsed
                    self.fps_updated.emit(fps)
                    print(f"FPS: {fps:.1f}")
                    self.frame_count = 0
                    self.start_time = time.time()

                # 录制（带水印）
                if self.is_recording and self.video_writer:
                    try:
                        video_frame = self.add_watermark(display_frame.copy())
                        self.video_writer.write(video_frame)
                    except Exception as record_error:
                        print(f"录制错误: {record_error}")

                # 每5帧才发送一次信号，减轻UI负担
                frame_emit_count += 1
                if frame_emit_count % 2 == 0:
                    try:
                        # 确保display_frame有效
                        if display_frame is not None and display_frame.size > 0:
                            self.frame_ready.emit(display_frame)
                            # 仅在控制台定期打印
                            if frame_emit_count % 20 == 0:
                                print(f"已发送帧 #{frame_emit_count}")
                        else:
                            print("尝试发送无效帧")
                    except Exception as emit_error:
                        print(f"发送帧信号错误: {emit_error}")
                
                # 更短的休眠时间，提高响应速度
                self.msleep(15)
        except Exception as e:
            print(f"运行时错误: {e}")
            self.error_occurred.emit(f"程序运行错误: {str(e)}")

        # 清理资源
        if face_mesh:
            face_mesh.close()
        if hands:
            hands.close()
        if pose:
            pose.close()
        self.cap.release()


class FaceMaskApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸换脸谱 - PySide6 版")
        self.resize(1400, 950)

        # 状态
        self.face_masks = []
        self.current_mask_index = 0
        self.is_camera_running = False
        self.current_mode = Mode.FACE_MASK
        self.is_recording = False
        self.worker = None
        self.latest_frame = None  # 用于截图
        self.resources_loaded = False  # 标记资源是否已加载
        
        # 样式变量，设为实例变量以便全局访问
        self.button_style = ""
        self.mode_button_style = ""
        self.control_button_style = ""

        self.init_ui()
        # 默认不加载资源，按需加载
        # self.load_face_masks()
        self.setup_connections()
        # self.update_mask_preview()

    def setup_connections(self):
        if self.worker is None:
            self.worker = VideoWorker()
        # 确保信号槽连接总是建立，无论worker是否为None
        try:
            # 断开之前的连接（如果有）
            try:
                self.worker.frame_ready.disconnect(self.on_frame_ready)
            except:
                pass
            try:
                self.worker.error_occurred.disconnect(self.on_error)
            except:
                pass
            try:
                self.worker.fps_updated.disconnect(self.update_fps_display)
            except:
                pass
            try:
                self.worker.request_mask_switch.disconnect(self.next_mask)
            except:
                pass
            
            # 重新连接信号槽
            self.worker.frame_ready.connect(self.on_frame_ready, Qt.QueuedConnection)
            self.worker.error_occurred.connect(self.on_error, Qt.QueuedConnection)
            self.worker.fps_updated.connect(self.update_fps_display, Qt.QueuedConnection)
            self.worker.request_mask_switch.connect(self.next_mask, Qt.QueuedConnection)
            print("信号槽连接成功建立")
        except Exception as e:
            print(f"建立信号槽连接时出错: {e}")
            # 在UI上显示错误消息
            if self.isVisible():
                self.status_bar.showMessage(f"信号连接错误: {str(e)}")

    def init_ui(self):
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#FFF8E1"))
        self.setPalette(palette)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # === 顶部工具栏 ===
        top_toolbar = QToolBar()
        top_toolbar.setFixedHeight(70)
        top_toolbar.setStyleSheet("background-color: #f0f0f0;")
        main_layout.addWidget(top_toolbar)

        # 工具栏按钮通用样式
        self.button_style = """
            QPushButton {
                padding: 12px 16px;
                border-radius: 12px;
                font-size: 14px;
                font-weight: bold;
                color: white;
                background-color: #4CAF50;
            }
            QPushButton:hover {
                opacity: 0.9;
            }
            QPushButton:checked {
                background-color: #3e8e41;
                border: 2px solid #2e7d32;
            }
        """

        # 模式按钮 - 鲜艳的绿色系
        self.mode_button_style = self.button_style.replace("background-color: #4CAF50;", "background-color: #4CAF50;")
        self.mode_buttons = {}
        modes = [
            (Mode.FACE_MASK, "川剧变脸"),
            (Mode.FACE_MESH, "人脸网格"),
            (Mode.HAND_SKELETON, "手指骨架"),
            (Mode.BODY_SKELETON, "人体骨架"),
        ]
        button_group = QButtonGroup(self)
        icon_map = {
            Mode.FACE_MASK: QIcon(os.path.join('resources', 'icons', 'face.png')),
            Mode.FACE_MESH: QIcon(os.path.join('resources', 'icons', 'mesh.jpg')),
            Mode.HAND_SKELETON: QIcon(os.path.join('resources', 'icons', 'hand.png')),
            Mode.BODY_SKELETON: QIcon(os.path.join('resources', 'icons', 'pose.png'))
        }
        
        for mode, text in modes:
            btn = QPushButton(text)
            btn.setIcon(icon_map[mode])
            btn.setIconSize(btn.sizeHint() / 2)  # 设置图标大小
            btn.setCheckable(True)
            btn.setStyleSheet(self.mode_button_style)
            btn.clicked.connect(lambda _, m=mode: self.switch_mode(m))
            self.mode_buttons[mode] = btn
            button_group.addButton(btn)
            top_toolbar.addWidget(btn)
            # 为模式按钮之间增加小间距
            spacer = QWidget()
            spacer.setFixedWidth(5)
            top_toolbar.addWidget(spacer)
        self.mode_buttons[Mode.FACE_MASK].setChecked(True)

        # 为模式按钮和控制按钮之间增加大间距
        spacer = QWidget()
        spacer.setFixedWidth(20)
        top_toolbar.addWidget(spacer)
        top_toolbar.addSeparator()
        spacer = QWidget()
        spacer.setFixedWidth(20)
        top_toolbar.addWidget(spacer)

        # 控制按钮 - 鲜艳的蓝色系
        self.control_button_style = self.button_style.replace("background-color: #4CAF50;", "background-color: #2196F3;")
        
        # 将开始/停止按钮移到顶部工具栏
        self.toggle_button = QPushButton("开始")
        # 创建一个简单的相机图标
        camera_icon = QIcon()
        self.toggle_button.setIcon(camera_icon)
        self.toggle_button.setIconSize(self.toggle_button.sizeHint() / 2)
        self.toggle_button.setStyleSheet(self.control_button_style)
        self.toggle_button.clicked.connect(self.toggle_camera)
        top_toolbar.addWidget(self.toggle_button)

        # 为控制按钮之间增加大间距
        spacer = QWidget()
        spacer.setFixedWidth(15)
        top_toolbar.addWidget(spacer)
        top_toolbar.addSeparator()
        spacer = QWidget()
        spacer.setFixedWidth(15)
        top_toolbar.addWidget(spacer)

        screenshot_btn = QPushButton("截图")
        # 创建一个简单的相机图标
        camera_icon = QIcon()
        screenshot_btn.setIcon(camera_icon)
        screenshot_btn.setIconSize(screenshot_btn.sizeHint() / 2)
        screenshot_btn.setStyleSheet(self.control_button_style)
        screenshot_btn.clicked.connect(self.take_screenshot)
        top_toolbar.addWidget(screenshot_btn)
        # 为截图和录制按钮之间增加小间距
        spacer = QWidget()
        spacer.setFixedWidth(5)
        top_toolbar.addWidget(spacer)

        self.record_btn = QPushButton("录制视频")
        # 创建一个简单的视频图标
        video_icon = QIcon()
        self.record_btn.setIcon(video_icon)
        self.record_btn.setIconSize(self.record_btn.sizeHint() / 2)
        self.record_btn.setStyleSheet(self.control_button_style)
        self.record_btn.clicked.connect(self.toggle_recording)
        top_toolbar.addWidget(self.record_btn)

        # 为录制按钮和关于按钮之间增加大间距
        spacer = QWidget()
        spacer.setFixedWidth(15)
        top_toolbar.addWidget(spacer)
        top_toolbar.addSeparator()
        spacer = QWidget()
        spacer.setFixedWidth(15)
        top_toolbar.addWidget(spacer)

        about_btn = QPushButton("关于我")
        # 创建一个简单的信息图标
        info_icon = QIcon()
        about_btn.setIcon(info_icon)
        about_btn.setIconSize(about_btn.sizeHint() / 2)
        about_btn.setStyleSheet(self.control_button_style)
        about_btn.clicked.connect(self.show_about_dialog)
        top_toolbar.addWidget(about_btn)

        # === 中央区域 ===
        center_widget = QWidget()
        center_layout = QHBoxLayout(center_widget)
        main_layout.addWidget(center_widget, 1)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: transparent;")
        self.video_label.setMinimumSize(*DISPLAY_SIZE)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        center_layout.addWidget(self.video_label, 1)

        # 右侧面板：透明度滑块
        right_panel = QWidget()
        right_panel.setFixedWidth(120)
        right_panel.setStyleSheet("background-color: #ffffff; border-left: 1px solid #dddddd;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setAlignment(Qt.AlignTop)
        right_layout.addSpacing(30)
        
        alpha_label = QLabel("透明度:")
        alpha_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        alpha_label.setStyleSheet("color: #333333;")
        alpha_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(alpha_label)
        right_layout.addSpacing(15)
        
        # 创建一个自定义的滑块容器
        slider_container = QWidget()
        slider_container.setStyleSheet("background-color: #f5f5f5; border-radius: 15px; padding: 10px;")
        slider_layout = QVBoxLayout(slider_container)
        slider_layout.setContentsMargins(5, 5, 5, 5)
        
        self.alpha_slider = QSlider(Qt.Vertical)
        self.alpha_slider.setRange(10, 90)
        self.alpha_slider.setValue(int(MASK_ALPHA_DEFAULT * 100))
        self.alpha_slider.setEnabled(True)  # 确保滑块可用
        self.alpha_slider.setFixedHeight(200)
        self.alpha_slider.setStyleSheet("""
            QSlider::groove:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF6B6B, stop:1 #4ECDC4);
                width: 10px;
                border-radius: 5px;
            }
            QSlider::handle:vertical {
                background: white;
                border: 2px solid #4ECDC4;
                width: 24px;
                height: 24px;
                margin: -7px 0;
                border-radius: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            QSlider::handle:vertical:hover {
                background: #4ECDC4;
                border-color: #45B7AA;
            }
            QSlider::handle:vertical:pressed {
                background: #45B7AA;
                border-color: #3C9E94;
                box-shadow: 0 1px 2px rgba(0,0,0,0.3);
            }
        """)
        # 确保valueChanged信号连接到正确的槽
        try:
            # 检查是否已经连接了信号
            if hasattr(self, 'on_alpha_changed'):
                # 断开之前的连接（如果有）
                try:
                    self.alpha_slider.valueChanged.disconnect(self.on_alpha_changed)
                except:
                    pass
                # 重新连接信号
                self.alpha_slider.valueChanged.connect(self.on_alpha_changed)
            else:
                # 如果on_alpha_changed方法不存在，则定义它
                def on_alpha_changed(value):
                    self.alpha_value_label.setText(f"{value/100:.2f}")
                    if self.worker and self.worker.isRunning():
                        self.worker.set_mask_alpha(value / 100.0)
                self.on_alpha_changed = on_alpha_changed
                self.alpha_slider.valueChanged.connect(on_alpha_changed)
        except Exception as e:
            print(f"连接透明度滑块信号时出错: {e}")
            # 作为后备方案，直接在滑块旁边显示值
            def show_alpha_value(value):
                self.alpha_value_label.setText(f"{value/100:.2f}")
            self.alpha_slider.valueChanged.connect(show_alpha_value)
        
        slider_layout.addWidget(self.alpha_slider, alignment=Qt.AlignCenter)
        right_layout.addWidget(slider_container)
        
        self.alpha_value_label = QLabel(f"{MASK_ALPHA_DEFAULT:.2f}")
        self.alpha_value_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.alpha_value_label.setStyleSheet("color: #4ECDC4;")
        self.alpha_value_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.alpha_value_label)
        right_layout.addSpacing(20)
        
        # 添加一些简单的操作说明
        help_text = QLabel("拖动滑块调整\n效果透明度")
        help_text.setFont(QFont("Microsoft YaHei", 9))
        help_text.setStyleSheet("color: #666666;")
        help_text.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(help_text)
        
        right_layout.addStretch()
        center_layout.addWidget(right_panel)

        # 脸谱预览
        self.masks_scroll_area = QScrollArea()
        self.masks_scroll_area.setWidgetResizable(True)
        self.masks_scroll_area.setFixedHeight(120)
        self.masks_container = QWidget()
        self.masks_layout = QHBoxLayout(self.masks_container)
        self.masks_layout.setContentsMargins(10, 10, 10, 10)
        self.masks_scroll_area.setWidget(self.masks_container)
        main_layout.addWidget(self.masks_scroll_area)

        # 移除原来的控制按钮区域，控制功能已移至顶部工具栏

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.camera_status_label = QLabel("摄像头: 未开启")
        self.fps_label = QLabel("FPS: 0.0")
        self.status_bar.addWidget(self.camera_status_label)
        self.status_bar.addWidget(QLabel(" | "))
        self.status_bar.addWidget(self.fps_label)

    def load_face_masks(self):
        import sys
        from pathlib import Path
        
        # 获取正确的face目录路径
        try:
            # 检查是否在PyInstaller打包环境中运行
            base_path = Path(sys._MEIPASS)
            face_dir = base_path / "face"
        except AttributeError:
            # 非打包环境，使用相对路径
            face_dir = Path("face")
        
        # 确保face目录存在
        face_dir.mkdir(exist_ok=True)
        
        # 查找所有图片文件
        image_files = list(face_dir.glob("*.png")) + list(face_dir.glob("*.jpg"))
        self.face_masks = []
        
        for f in image_files:
            try:
                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    self.face_masks.append(img)
            except Exception as e:
                print(f"加载图像 {f} 失败: {e}")
        
        if not self.face_masks:
            # 如果在打包环境中没有找到资源，尝试从当前工作目录的face文件夹加载
            if hasattr(sys, '_MEIPASS'):
                cwd_face_dir = Path.cwd() / "face"
                if cwd_face_dir.exists():
                    cwd_image_files = list(cwd_face_dir.glob("*.png")) + list(cwd_face_dir.glob("*.jpg"))
                    for f in cwd_image_files:
                        try:
                            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                            if img is not None:
                                self.face_masks.append(img)
                        except Exception as e:
                            print(f"加载工作目录中的图像 {f} 失败: {e}")
            
            if not self.face_masks:
                QMessageBox.critical(self, "错误", "未在 'face' 目录中找到 PNG/JPG 图像！")

    def update_mask_preview(self):
        # 清空
        while self.masks_layout.count():
            child = self.masks_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        THUMBNAIL_SIZE = 100
        for i, mask in enumerate(self.face_masks):
            mask_copy = mask.copy()
            h, w = mask_copy.shape[:2]
            scale = min(THUMBNAIL_SIZE / w, THUMBNAIL_SIZE / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(mask_copy, (new_w, new_h))

            square = np.ones((THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3), dtype=np.uint8) * 255
            y_off = (THUMBNAIL_SIZE - new_h) // 2
            x_off = (THUMBNAIL_SIZE - new_w) // 2

            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            elif resized.shape[2] == 4:
                b, g, r, a = cv2.split(resized)
                bg = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
                alpha = a / 255.0
                for j in range(3):
                    bg[:, :, j] = alpha * resized[:, :, j] + (1 - alpha) * 255
                resized = bg

            square[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            square_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            qimg = QImage(square_rgb.data, square_rgb.shape[1], square_rgb.shape[0],
                          square_rgb.strides[0], QImage.Format_RGB888)
            thumb_label = QLabel()
            thumb_label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
            thumb_label.setPixmap(QPixmap.fromImage(qimg))
            if i == self.current_mask_index:
                thumb_label.setStyleSheet("border: 3px solid #4CAF50;")
            else:
                thumb_label.setStyleSheet("border: 1px solid #ccc;")

            # 修复 lambda 闭包问题
            thumb_label.mousePressEvent = self.make_thumbnail_click_handler(i)
            self.masks_layout.addWidget(thumb_label)

    def make_thumbnail_click_handler(self, index):
        """生成点击处理函数，避免闭包陷阱"""
        return lambda event: self.on_thumbnail_clicked(index)

    def on_thumbnail_clicked(self, index):
        self.current_mask_index = index
        self.update_mask_preview()
        if self.worker and self.worker.isRunning():
            self.worker.set_current_mask_index(index)

    @Slot(int)
    def on_alpha_changed(self, value):
        alpha = value / 100.0
        self.alpha_value_label.setText(f"{alpha:.2f}")
        if self.worker:
            self.worker.set_mask_alpha(alpha)

    @Slot(np.ndarray)
    def on_frame_ready(self, frame):
        try:
            # 首先确保视频标签存在
            if not hasattr(self, 'video_label'):
                print("错误: video_label不存在")
                return
                
            # 缓存用于截图
            if frame is not None and frame.size > 0:
                self.latest_frame = frame.copy()
            else:
                print("警告: 接收到空帧")
                return
                
            # 检查帧的基本信息
            if len(frame.shape) < 2:
                print(f"错误: 无效的帧形状: {frame.shape}")
                return
                
            # 获取帧的高度和宽度
            h, w = frame.shape[:2]
            ch = 1 if len(frame.shape) == 2 else frame.shape[2]
            print(f"帧信息: {w}x{h}, {ch}通道, 类型: {frame.dtype}")
            
            # 确保图像数据有效
            if frame.size == 0:
                print("错误: 帧数据为空")
                return
                
            # 检查图像格式并正确转换
            if frame.dtype != np.uint8:
                print(f"转换帧类型: {frame.dtype} -> uint8")
                frame = frame.astype(np.uint8)
                
            # 确保图像是3通道BGR格式
            if len(frame.shape) == 2:  # 灰度图
                print("转换灰度图到BGR")
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                ch = 3
            elif frame.shape[2] == 4:  # RGBA
                print("转换RGBA到BGR")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                ch = 3
            elif frame.shape[2] == 3:  # 已经是BGR
                print("帧已是BGR格式")
                pass
            else:
                print(f"错误: 不支持的通道数: {frame.shape[2]}")
                return
                
            # 创建QImage - 确保使用正确的格式
            bytes_per_line = ch * w
            try:
                qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                if qimg.isNull():
                    print("错误: 创建QImage失败")
                    # 尝试使用另一种方法创建QImage
                    qimg = QImage(frame.tobytes(), w, h, bytes_per_line, QImage.Format_BGR888)
                    if qimg.isNull():
                        print("错误: 再次尝试创建QImage失败")
                        return
                    else:
                        print("成功: 使用tobytes()创建QImage")
                else:
                    print("成功: 创建QImage")
            except Exception as img_err:
                print(f"创建QImage异常: {img_err}")
                return
                
            # 创建QPixmap
            try:
                pixmap = QPixmap.fromImage(qimg)
                if pixmap.isNull():
                    print("错误: 创建QPixmap失败")
                    return
                print("成功: 创建QPixmap")
            except Exception as pix_err:
                print(f"创建QPixmap异常: {pix_err}")
                return
                
            # 设置Pixmap到标签 - 使用固定尺寸避免缩放问题
            try:
                # 先设置一个固定的尺寸以确保显示
                self.video_label.setMinimumSize(w, h)
                # 使用KeepAspectRatio确保图像不变形
                scaled_pixmap = pixmap.scaled(
                    self.video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.FastTransformation  # 使用FastTransformation代替SmoothTransformation以提高性能
                )
                self.video_label.setPixmap(scaled_pixmap)
                print("成功: 设置Pixmap到video_label")
            except Exception as set_err:
                print(f"设置Pixmap异常: {set_err}")
                return
                
            # 确保video_label和所有父窗口都可见
            if not self.video_label.isVisible():
                print("显示video_label")
                self.video_label.show()
                
            # 检查并确保所有父窗口都可见
            parent = self.video_label.parentWidget()
            while parent:
                if not parent.isVisible():
                    print(f"显示父窗口: {parent.objectName()}")
                    parent.show()
                parent = parent.parentWidget()
                
            # 确保主窗口可见
            if not self.isVisible():
                print("显示主窗口")
                self.show()
                self.raise_()
            
            # 强制更新和重绘
            try:
                self.video_label.repaint()
                self.video_label.update()
                QApplication.processEvents()
                self.update()
                print("强制重绘完成")
            except Exception as update_err:
                print(f"重绘异常: {update_err}")
                
        except Exception as e:
            # 捕获并打印异常，以便调试
            print(f"on_frame_ready异常: {e}")
            # 在UI上显示错误消息
            self.status_bar.showMessage(f"视频显示错误: {str(e)}")

    @Slot(float)
    def update_fps_display(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    @Slot()
    def next_mask(self):
        if self.face_masks:
            self.current_mask_index = (self.current_mask_index + 1) % len(self.face_masks)
            self.update_mask_preview()
            if self.worker and self.worker.isRunning():
                self.worker.set_current_mask_index(self.current_mask_index)

    @Slot(str)
    def on_error(self, msg):
        QMessageBox.critical(self, "错误", msg)
        self.stop_camera()

    def switch_mode(self, mode):
        self.current_mode = mode
        if mode == Mode.FACE_MASK:
            self.masks_scroll_area.show()
        else:
            self.masks_scroll_area.hide()
        if self.worker and self.worker.isRunning():
            self.worker.set_mode(mode)

    def take_screenshot(self):
        if not self.is_camera_running or self.latest_frame is None:
            QMessageBox.information(self, "提示", "请先开启摄像头并等待画面加载！")
            return
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = screenshots_dir / f"screenshot_{timestamp}.png"
        # 从缓存帧保存并添加水印
        bgr_frame = self.latest_frame.copy()
        # 添加微信水印（与视频中相同）
        if self.worker:
            bgr_frame = self.worker.add_watermark(bgr_frame)
        success = cv2.imwrite(str(filename), bgr_frame)
        if success:
            self.status_bar.showMessage(f"截图已保存至: {filename}")
        else:
            QMessageBox.critical(self, "错误", "无法保存截图！")

    def toggle_recording(self):
        if not self.is_camera_running:
            QMessageBox.information(self, "提示", "请先开启摄像头！")
            return
        self.is_recording = not self.is_recording
        if self.is_recording:
            if self.worker:
                self.worker.start_recording()
                self.record_btn.setText("停止录制")
                # 保留其他样式，只改变颜色
                self.record_btn.setStyleSheet(self.control_button_style.replace("color: white;", "color: red;"))
        else:
            if self.worker:
                path = self.worker.stop_recording()
                self.record_btn.setText("录制视频")
                # 恢复原始样式
                self.record_btn.setStyleSheet(self.control_button_style)
                if path:
                    self.status_bar.showMessage(f"视频已保存至: {path}")

    def show_about_dialog(self):
        # 创建自定义对话框
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("关于")
        about_dialog.resize(450, 350)
        
        # 设置对话框样式，与主UI保持一致
        about_dialog.setStyleSheet("background-color: #FFF8E1;")
        
        layout = QVBoxLayout(about_dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 标题
        title = QLabel("川剧变脸 - 特效相机")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #333333;")
        layout.addWidget(title)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #cccccc;")
        layout.addWidget(line)
        
        # 信息文本
        info = QLabel("基于OpenCV和MediaPipe的实时人脸特效应用\n\n"
                      "- 川剧变脸：手势切换\n- 人脸/手/人体骨架\n- 截图 & 录制")
        info.setWordWrap(True)
        info.setFont(QFont("Microsoft YaHei", 11))
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: #555555;")
        layout.addWidget(info)
        
        # 添加微信图片（如果存在）
        wechat_path = os.path.join(os.getcwd(), "wechat.jpg")
        if os.path.exists(wechat_path):
            wechat_label = QLabel()
            pixmap = QPixmap(wechat_path)
            # 调整图片大小
            scaled_pixmap = pixmap.scaledToWidth(120, Qt.SmoothTransformation)
            wechat_label.setPixmap(scaled_pixmap)
            wechat_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(wechat_label)
            
            wechat_text = QLabel("扫码了解更多")
            wechat_text.setFont(QFont("Microsoft YaHei", 10))
            wechat_text.setAlignment(Qt.AlignCenter)
            wechat_text.setStyleSheet("color: #666666;")
            layout.addWidget(wechat_text)
        
        # 版本信息
        version = QLabel("版本 1.0.0")
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet("color: #777777; font-style: italic;")
        layout.addWidget(version)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # 确定按钮，使用与控制按钮一致的样式
        ok_button = QPushButton("确定")
        ok_button.setStyleSheet("background-color: #2196F3; color: white; padding: 10px 20px; border-radius: 12px; font-size: 14px; font-weight: bold;")
        ok_button.clicked.connect(about_dialog.accept)
        button_layout.addWidget(ok_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        about_dialog.exec()

    def start_camera(self):
        # 启动摄像头（先启动摄像头，再加载资源）
        if self.worker and self.worker.isRunning():
            self.worker.set_running(False)
            self.worker.wait()
        self.worker = VideoWorker()
        self.setup_connections()
        self.worker.set_running(True)
        self.worker.set_face_masks(self.face_masks)
        self.worker.set_current_mask_index(self.current_mask_index)
        self.worker.set_mask_alpha(self.alpha_slider.value() / 100.0)
        self.worker.set_mode(self.current_mode)
        self.worker.start()
        self.is_camera_running = True
        self.camera_status_label.setText("摄像头: 开启")
        
        # 更新按钮样式和文本
        toggle_style = ""
        if self.is_camera_running:
            self.toggle_button.setText("停止")
            toggle_style = "background-color: #F44336; color: white; padding: 12px 16px; border-radius: 12px; font-size: 14px; font-weight: bold;"
        else:
            self.toggle_button.setText("开始")
            toggle_style = "background-color: #4CAF50; color: white; padding: 12px 16px; border-radius: 12px; font-size: 14px; font-weight: bold;"
        self.toggle_button.setStyleSheet(toggle_style)
        
        mode_names = {Mode.FACE_MASK: "川剧变脸", Mode.FACE_MESH: "人脸网格",
                      Mode.HAND_SKELETON: "手指骨架", Mode.BODY_SKELETON: "人体骨架"}
        self.status_bar.showMessage(f"摄像头运行中 - {mode_names[self.current_mode]}")
        
        # 按需加载资源（在单独的线程中进行，不阻塞UI）
        if not self.resources_loaded:
            # 在主UI上显示加载进度条
            if not hasattr(self, 'loading_progress_widget'):
                # 创建加载指示器控件
                self.loading_progress_widget = QWidget(self)
                self.loading_progress_widget.setGeometry(100, 30, 300, 80)
                self.loading_progress_widget.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
                self.loading_progress_widget.setStyleSheet("background-color: rgba(255, 255, 255, 220); border-radius: 10px; border: 1px solid #cccccc;")
                
                # 布局
                loading_layout = QVBoxLayout(self.loading_progress_widget)
                loading_layout.setContentsMargins(15, 15, 15, 15)
                
                # 标签
                self.loading_label = QLabel("正在加载资源，请稍候...")
                self.loading_label.setAlignment(Qt.AlignCenter)
                self.loading_label.setStyleSheet("color: #333333; font-weight: bold;")
                loading_layout.addWidget(self.loading_label)
                
                # 进度条
                self.loading_progress = QProgressBar()
                self.loading_progress.setRange(0, 100)
                self.loading_progress.setValue(0)
                self.loading_progress.setStyleSheet("QProgressBar {border: 1px solid grey; border-radius: 5px; text-align: center;}"
                                                  "QProgressBar::chunk {background-color: #4CAF50; width: 20px;}")
                loading_layout.addWidget(self.loading_progress)
                
            # 显示加载控件
            self.loading_progress_widget.show()
            
            # 创建资源加载线程
            self.resource_thread = QThread()
            self.resource_loader = ResourceLoader(self.current_mode, self)
            self.resource_loader.moveToThread(self.resource_thread)
            
            # 连接信号槽
            self.resource_thread.started.connect(self.resource_loader.load_resources)
            self.resource_loader.progress_updated.connect(self.update_loading_progress)
            self.resource_loader.loading_completed.connect(self.on_resources_loaded)
            self.resource_loader.loading_failed.connect(self.on_resources_failed)
            self.resource_loader.finished.connect(self.resource_thread.quit)
            self.resource_thread.finished.connect(self.resource_thread.deleteLater)
            
            # 启动线程
            self.resource_thread.start()
    
    def update_loading_progress(self, progress, message):
        """更新加载进度条和状态消息"""
        self.loading_progress.setValue(progress)
        self.loading_label.setText(message)
        self.status_bar.showMessage(f"加载中: {message}")
        QApplication.processEvents()
    
    def on_resources_loaded(self):
        """资源加载完成后的回调"""
        self.resources_loaded = True
        self.status_bar.showMessage("资源加载完成")
        
        # 隐藏加载控件
        if hasattr(self, 'loading_progress_widget'):
            self.loading_progress_widget.hide()
            
        # 验证模式特定资源
        if self.current_mode == Mode.FACE_MASK and self.worker:
            self.worker.set_face_masks(self.face_masks)
            self.update_mask_preview()
    
    def on_resources_failed(self, error_msg):
        """资源加载失败后的回调"""
        print(f"资源加载失败: {error_msg}")
        self.status_bar.showMessage(f"资源加载失败: {error_msg}")
        
        # 隐藏加载控件
        if hasattr(self, 'loading_progress_widget'):
            self.loading_progress_widget.hide()
            
        # 显示错误消息，但不停止摄像头
        QMessageBox.critical(self, "错误", f"资源加载失败: {error_msg}\n\n摄像头仍可使用，但部分功能可能受限。")
    
    # 摄像头启动逻辑已移至方法开头
    

    def stop_camera(self):
        if not self.is_camera_running:
            return
        if self.is_recording:
            self.toggle_recording()
        if self.worker:
            self.worker.set_running(False)
            self.worker.wait(2000)
        self.is_camera_running = False
        self.camera_status_label.setText("摄像头: 未开启")
        
        # 更新按钮样式和文本
        if hasattr(self, 'toggle_button'):
            self.toggle_button.setText("开始")
            self.toggle_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 12px 16px; border-radius: 12px; font-size: 14px; font-weight: bold;")
            
        self.status_bar.showMessage("就绪")

    def toggle_camera(self):
        if self.is_camera_running:
            self.stop_camera()
        else:
            self.start_camera()
        # 确保UI状态一致性
        if hasattr(self, 'toggle_button'):
            if self.is_camera_running:
                self.toggle_button.setText("停止")
                self.toggle_button.setStyleSheet("background-color: #F44336; color: white; padding: 12px 16px; border-radius: 12px; font-size: 14px; font-weight: bold;")
            else:
                self.toggle_button.setText("开始")
                self.toggle_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 12px 16px; border-radius: 12px; font-size: 14px; font-weight: bold;")

    def closeEvent(self, event):
        self.stop_camera()
        
        # 创建自定义退出确认对话框
        exit_dialog = QDialog(self)
        exit_dialog.setWindowTitle("确认退出")
        exit_dialog.resize(350, 180)
        
        # 设置对话框样式，与主UI保持一致
        exit_dialog.setStyleSheet("background-color: #FFF8E1;")
        
        layout = QVBoxLayout(exit_dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 提示信息
        message = QLabel("确定要退出程序吗？")
        message.setFont(QFont("Microsoft YaHei", 12))
        message.setAlignment(Qt.AlignCenter)
        message.setStyleSheet("color: #333333;")
        layout.addWidget(message)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # 取消按钮
        cancel_button = QPushButton("取消")
        cancel_button.setStyleSheet("background-color: #f0f0f0; color: #333333; padding: 10px 20px; border-radius: 12px; font-size: 14px; font-weight: bold;")
        cancel_button.clicked.connect(exit_dialog.reject)
        button_layout.addWidget(cancel_button)
        button_layout.addSpacing(10)
        
        # 确定按钮，使用与停止按钮一致的样式
        ok_button = QPushButton("确定")
        ok_button.setStyleSheet("background-color: #F44336; color: white; padding: 10px 20px; border-radius: 12px; font-size: 14px; font-weight: bold;")
        ok_button.clicked.connect(exit_dialog.accept)
        button_layout.addWidget(ok_button)
        
        layout.addLayout(button_layout)
        
        # 显示对话框并处理结果
        if exit_dialog.exec() == QDialog.Accepted:
            event.accept()
        else:
            event.ignore()
            # 如果用户选择不退出，确保摄像头状态正确
            if not self.is_camera_running:
                self.start_camera()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = FaceMaskApp()
    window.show()
    
    # 程序启动时自动尝试打开摄像头
    def try_camera():
        try:
            # 创建临时VideoWorker来测试摄像头
            test_worker = VideoWorker()
            test_worker.moveToThread(QThread.currentThread())
            
            # 尝试打开摄像头
            camera_indices = [0, 1, 2]
            cap = None
            for i in camera_indices:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cap.release()
                    # 摄像头可用，启动主程序摄像头
                    window.start_camera()
                    return True
                else:
                    cap.release()
            
            # 摄像头不可用，显示重试/退出对话框
            while True:
                reply = QMessageBox.question(
                    window,
                    "获取摄像头失败",
                    "无法打开摄像头！请检查设备连接和权限。\n\n是否重试？",
                    QMessageBox.Retry | QMessageBox.Exit
                )
                
                if reply == QMessageBox.Exit:
                    return False
                elif reply == QMessageBox.Retry:
                    # 重试打开摄像头
                    if try_camera():
                        return True
        except Exception as e:
            print(f"摄像头测试失败: {e}")
            return False
    
    # 启动时尝试打开摄像头
    QTimer.singleShot(100, lambda: try_camera())
    sys.exit(app.exec())


if __name__ == "__main__":
    main()