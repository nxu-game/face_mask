import sys
import os
import glob
import cv2
import numpy as np
from math import hypot, atan2, degrees
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QVBoxLayout, QHBoxLayout, QFrame, QGridLayout,
    QMessageBox, QStatusBar, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QPalette, QColor

import mediapipe as mp

# 全局配置
PREVIEW_SIZE = 128
DISPLAY_SIZE = (1280, 720)
MASK_ALPHA_DEFAULT = 0.28
ORIGINAL_EYE_DISTANCE = 100
HAND_FACE_THRESHOLD = 0.1


class VideoWorker(QThread):
    """视频处理工作线程"""
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_running = False
        self.face_masks = []
        self.current_mask_index = 0
        self.mask_alpha = MASK_ALPHA_DEFAULT
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands

    def set_running(self, running: bool):
        self._is_running = running

    def set_face_masks(self, masks):
        self.face_masks = masks

    def set_current_mask_index(self, idx):
        self.current_mask_index = idx

    def set_mask_alpha(self, alpha):
        self.mask_alpha = alpha

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def is_hand_near_face(self, hand_landmarks, face_landmarks, img_w, img_h) -> bool:
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        nose_tip = face_landmarks.landmark[1]
        dx = abs(index_tip.x - nose_tip.x) * img_w
        dy = abs(index_tip.y - nose_tip.y) * img_h
        dist = hypot(dx, dy)
        diagonal = hypot(img_w, img_h)
        return dist < diagonal * HAND_FACE_THRESHOLD

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
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error_occurred.emit("无法打开摄像头！")
            return

        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh, self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:

            while self._is_running:
                success, frame = cap.read()
                if not success:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

        # 创建 UI
        self.init_ui()
        
        # 加载资源
        self.load_face_masks()

        # 创建工作线程
        self.worker = VideoWorker()
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.error_occurred.connect(self.on_error)

        # 初始预览
        self.update_mask_preview()

    def init_ui(self):
        # 主窗口背景色
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#FFF8E1"))
        self.setPalette(palette)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # === 左侧：透明度控制 ===
        left_frame = QFrame()
        left_frame.setFixedWidth(200)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(10, 10, 10, 10)

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

        left_layout.addWidget(alpha_label)
        left_layout.addWidget(self.alpha_slider)
        left_layout.addWidget(self.alpha_value_label)
        left_layout.addStretch()

        # === 中间：视频显示 ===
        center_frame = QFrame()
        center_layout = QVBoxLayout(center_frame)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(*DISPLAY_SIZE)
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        center_layout.addWidget(self.video_label)

        # === 右侧：控制面板 ===
        right_frame = QFrame()
        right_frame.setFixedWidth(250)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # 预览
        preview_label = QLabel("当前脸谱")
        preview_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        preview_label.setAlignment(Qt.AlignCenter)

        self.mask_preview = QLabel()
        self.mask_preview.setFixedSize(PREVIEW_SIZE, PREVIEW_SIZE)
        self.mask_preview.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.mask_preview.setAlignment(Qt.AlignCenter)

        # 按钮
        self.start_btn = QPushButton("开始")
        self.start_btn.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 15px;")
        self.start_btn.clicked.connect(self.start_camera)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        self.stop_btn.setStyleSheet("background-color: #F44336; color: white; padding: 15px;")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)

        right_layout.addWidget(preview_label)
        right_layout.addWidget(self.mask_preview)
        right_layout.addSpacing(20)
        right_layout.addWidget(self.start_btn)
        right_layout.addWidget(self.stop_btn)
        right_layout.addStretch()

        # 组装
        main_layout.addWidget(left_frame)
        main_layout.addWidget(center_frame, 1)
        main_layout.addWidget(right_frame)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
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
        if not self.face_masks:
            return
        mask = self.face_masks[self.current_mask_index].copy()
        h, w = mask.shape[:2]
        scale = min(PREVIEW_SIZE / w, PREVIEW_SIZE / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(mask, (new_w, new_h))

        # 白底方形
        square = np.ones((PREVIEW_SIZE, PREVIEW_SIZE, 3), dtype=np.uint8) * 255
        y_off = (PREVIEW_SIZE - new_h) // 2
        x_off = (PREVIEW_SIZE - new_w) // 2

        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        elif resized.shape[2] == 4:
            b, g, r, a = cv2.split(resized)
            bg = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
            for i in range(3):
                bg[:, :, i] = (a / 255.0) * resized[:, :, i] + (1 - a / 255.0) * 255
            resized = bg

        square[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        square_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        h, w, ch = square_rgb.shape
        qimg = QImage(square_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.mask_preview.setPixmap(QPixmap.fromImage(qimg))

        self.status_bar.showMessage(f"当前脸谱: {self.current_mask_index + 1}/{len(self.face_masks)}")

    @Slot(int)
    def on_alpha_changed(self, value):
        alpha = value / 100.0
        self.alpha_value_label.setText(f"{alpha:.2f}")
        if hasattr(self, 'worker'):
            self.worker.set_mask_alpha(alpha)

    @Slot(np.ndarray)
    def on_frame_ready(self, frame):
        if frame is None:
            # 请求切换脸谱
            self.next_mask()
            return

        h, w, ch = frame.shape
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

    def start_camera(self):
        if not self.face_masks:
            QMessageBox.critical(self, "错误", "请先加载脸谱图像！")
            return
        if not self.is_running:
            self.is_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_bar.showMessage("正在运行...")

            self.worker.set_running(True)
            self.worker.set_face_masks(self.face_masks)
            self.worker.set_current_mask_index(self.current_mask_index)
            self.worker.set_mask_alpha(self.alpha_slider.value() / 100.0)
            self.worker.start()

    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            self.worker.set_running(False)
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_bar.showMessage("已停止")
            # 等待线程结束（短超时）
            self.worker.wait(2000)

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