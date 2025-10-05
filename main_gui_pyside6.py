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
from PySide6.QtWidgets import QApplication
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
        overlay = image.copy()
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
        return cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

    def draw_hand_skeleton(self, image, hand_landmarks):
        if not hand_landmarks:
            return image
        overlay = image.copy()
        for hand_lm in hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image=overlay,
                landmark_list=hand_lm,
                connections=mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )
        return cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

    def draw_body_skeleton(self, image, pose_landmarks):
        if not pose_landmarks:
            return image
        overlay = image.copy()
        # 处理单个pose_landmarks对象（不是列表）
        self.mp_drawing.draw_landmarks(
            image=overlay,
            landmark_list=pose_landmarks,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        return cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

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
            alpha = mask_roi[:, :, 3].astype(np.float32) / 255.0 * self.mask_alpha
            for c in range(3):
                roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * mask_roi[:, :, c]
        else:
            blended = cv2.addWeighted(roi, 1 - self.mask_alpha, mask_roi, self.mask_alpha, 0)
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
                print(f"成功打开摄像头索引: {i}")
                break
            else:
                self.cap.release()
                
        if not self.cap or not self.cap.isOpened():
            self.error_occurred.emit("无法打开摄像头！请检查设备连接和权限。")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.load_watermark("wechat.jpg")

        self.frame_count = 0
        self.start_time = time.time()
        
        # 初始化模型为None，按需加载
        face_mesh = None
        hands = None
        pose = None
        
        while self._is_running:
            success, frame = self.cap.read()
            if not success:
                print("摄像头读取失败")
                continue

            # 统一在 RGB 空间处理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
                processed_frame = frame.copy()  # BGR

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

            # 统一缩放镜像
            display_frame = self.scale_and_mirror(processed_frame)

            # 帧率更新
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 1.0:
                fps = self.frame_count / elapsed
                self.fps_updated.emit(fps)
                self.frame_count = 0
                self.start_time = time.time()

            # 录制（带水印）
            if self.is_recording and self.video_writer:
                video_frame = self.add_watermark(display_frame.copy())
                self.video_writer.write(video_frame)

            # 调试：检查帧数据
            if self.frame_count % 30 == 0:  # 每30帧打印一次调试信息
                print(f"帧尺寸: {display_frame.shape}, 数据类型: {display_frame.dtype}, 模式: {self.current_mode}")
                # 检查帧是否全黑
                if np.all(display_frame == 0):
                    print("警告：检测到全黑帧！")
                else:
                    print(f"帧非全黑，像素范围: {display_frame.min()} - {display_frame.max()}")

            # 调试：检查frame_ready信号是否被发射
            if self.frame_count % 10 == 0:
                print(f"发射frame_ready信号: 第{self.frame_count}帧")
            self.frame_ready.emit(display_frame)
            self.msleep(30)

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

        self.init_ui()
        self.load_face_masks()
        self.setup_connections()
        self.update_mask_preview()

    def setup_connections(self):
        if self.worker is None:
            self.worker = VideoWorker()
        # 确保信号槽连接总是建立，无论worker是否为None
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.fps_updated.connect(self.update_fps_display)
        self.worker.request_mask_switch.connect(self.next_mask)

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
        top_toolbar.setFixedHeight(60)
        main_layout.addWidget(top_toolbar)

        self.mode_buttons = {}
        modes = [
            (Mode.FACE_MASK, "川剧变脸"),
            (Mode.FACE_MESH, "人脸网格"),
            (Mode.HAND_SKELETON, "手指骨架"),
            (Mode.BODY_SKELETON, "人体骨架"),
        ]
        button_group = QButtonGroup(self)
        for mode, text in modes:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, m=mode: self.switch_mode(m))
            self.mode_buttons[mode] = btn
            top_toolbar.addWidget(btn)
            button_group.addButton(btn)
        self.mode_buttons[Mode.FACE_MASK].setChecked(True)

        top_toolbar.addSeparator()

        screenshot_btn = QPushButton("截图")
        screenshot_btn.clicked.connect(self.take_screenshot)
        top_toolbar.addWidget(screenshot_btn)

        self.record_btn = QPushButton("录制视频")
        self.record_btn.clicked.connect(self.toggle_recording)
        top_toolbar.addWidget(self.record_btn)

        top_toolbar.addSeparator()

        about_btn = QPushButton("关于我")
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

        # 透明度滑块
        right_panel = QWidget()
        right_panel.setFixedWidth(100)
        right_layout = QVBoxLayout(right_panel)
        alpha_label = QLabel("透明度:")
        alpha_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        alpha_label.setAlignment(Qt.AlignCenter)
        self.alpha_slider = QSlider(Qt.Vertical)
        self.alpha_slider.setRange(10, 90)
        self.alpha_slider.setValue(int(MASK_ALPHA_DEFAULT * 100))
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)
        self.alpha_value_label = QLabel(f"{MASK_ALPHA_DEFAULT:.2f}")
        self.alpha_value_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        self.alpha_value_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(alpha_label)
        right_layout.addWidget(self.alpha_slider)
        right_layout.addWidget(self.alpha_value_label)
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

        # 控制按钮
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        control_layout.setAlignment(Qt.AlignCenter)
        self.toggle_button = QPushButton("开始")
        self.toggle_button.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        self.toggle_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 15px 40px;")
        self.toggle_button.clicked.connect(self.toggle_camera)
        control_layout.addWidget(self.toggle_button)
        main_layout.addWidget(control_widget)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.camera_status_label = QLabel("摄像头: 未开启")
        self.fps_label = QLabel("FPS: 0.0")
        self.status_bar.addWidget(self.camera_status_label)
        self.status_bar.addWidget(QLabel(" | "))
        self.status_bar.addWidget(self.fps_label)

    def load_face_masks(self):
        face_dir = Path("face")
        face_dir.mkdir(exist_ok=True)
        image_files = list(face_dir.glob("*.png")) + list(face_dir.glob("*.jpg"))
        self.face_masks = []
        for f in image_files:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is not None:
                self.face_masks.append(img)
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
            self.latest_frame = frame.copy()  # 缓存用于截图
            h, w, ch = frame.shape
            
            # 确保图像数据有效
            if frame.size == 0:
                print("接收到空帧")
                return
                
            # 检查图像格式并正确转换
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
                
            # 确保图像是3通道BGR格式
            if len(frame.shape) == 2:  # 灰度图
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] == 3:  # 已经是BGR
                pass
            else:
                print(f"未知的图像格式: {frame.shape}")
                return
                
            # 创建QImage
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            
            if qimg.isNull():
                print("QImage创建失败")
                return
            else:
                print(f"QImage创建成功: {qimg.width()}x{qimg.height()}, 格式: {qimg.format()}")
                
            # 创建QPixmap
            pixmap = QPixmap.fromImage(qimg)
            if pixmap.isNull():
                print("QPixmap创建失败")
                return
            else:
                print(f"QPixmap创建成功: {pixmap.width()}x{pixmap.height()}")
                
            # 检查video_label状态
            print(f"video_label尺寸: {self.video_label.size().width()}x{self.video_label.size().height()}")
            print(f"video_label可见性: {self.video_label.isVisible()}")
            print(f"video_label启用状态: {self.video_label.isEnabled()}")
            # 检查父级和布局
            parent_widget = self.video_label.parentWidget()
            print(f"video_label父窗口: {parent_widget}")
            if parent_widget:
                print(f"父窗口布局: {parent_widget.layout()}")
                print(f"父窗口可见性: {parent_widget.isVisible()}")
            # 检查窗口层级和z-index
            print(f"video_label层级: {self.video_label.raise_()}")
            print(f"video_label是否在最顶层: {self.video_label.isTopLevel()}")
                
            # 设置Pixmap
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # 确保video_label和所有父窗口都可见
            if not self.video_label.isVisible():
                print("警告：video_label被隐藏，正在强制显示")
                self.video_label.show()
                
            # 检查并确保所有父窗口都可见
            parent = self.video_label.parentWidget()
            while parent:
                if not parent.isVisible():
                    print(f"警告：父窗口 {parent} 被隐藏，正在强制显示")
                    parent.show()
                parent = parent.parentWidget()
                
            # 确保主窗口可见
            if not self.isVisible():
                print("警告：主窗口被隐藏，正在强制显示")
                self.show()
                self.raise_()
            
            # 强制更新
            self.video_label.repaint()
            # 强制应用程序处理所有待处理的事件
            QApplication.processEvents()
            # 强制整个窗口重绘
            self.update()
            print("Pixmap已设置并触发重绘和窗口刷新")
            
        except Exception as e:
            print(f"显示帧时出错: {e}")
            import traceback
            traceback.print_exc()

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
        # 从缓存帧保存
        bgr_frame = self.latest_frame.copy()
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
                self.record_btn.setStyleSheet("color: red;")
        else:
            if self.worker:
                path = self.worker.stop_recording()
                self.record_btn.setText("录制视频")
                self.record_btn.setStyleSheet("")
                if path:
                    self.status_bar.showMessage(f"视频已保存至: {path}")

    def show_about_dialog(self):
        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("关于")
        about_dialog.resize(400, 300)
        layout = QVBoxLayout(about_dialog)
        title = QLabel("川剧变脸 - 特效相机")
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        info = QLabel("基于OpenCV和MediaPipe的实时人脸特效应用\n\n"
                      "- 川剧变脸：手势切换\n- 人脸/手/人体骨架\n- 截图 & 录制")
        info.setWordWrap(True)
        version = QLabel("版本 1.0.0")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(info)
        layout.addStretch()
        layout.addWidget(version)
        about_dialog.exec()

    def start_camera(self):
        if self.current_mode == Mode.FACE_MASK and not self.face_masks:
            QMessageBox.critical(self, "错误", "请先加载脸谱图像！")
            return
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
        self.toggle_button.setText("停止")
        self.toggle_button.setStyleSheet("background-color: #F44336; color: white; padding: 15px 40px;")
        mode_names = {Mode.FACE_MASK: "川剧变脸", Mode.FACE_MESH: "人脸网格",
                      Mode.HAND_SKELETON: "手指骨架", Mode.BODY_SKELETON: "人体骨架"}
        self.status_bar.showMessage(f"摄像头运行中 - {mode_names[self.current_mode]}")

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
        self.toggle_button.setText("开始")
        self.toggle_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 15px 40px;")
        self.status_bar.showMessage("就绪")

    def toggle_camera(self):
        if self.is_camera_running:
            self.stop_camera()
        else:
            self.start_camera()

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