"""
人脸换脸谱应用
功能：实时检测人脸并叠加脸谱效果，支持手势和按键切换脸谱
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

class FaceMaskApp:
    def __init__(self):
        """初始化人脸换脸谱应用"""
        # 初始化 Mediapipe 模块
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 初始化视频捕获和脸谱相关变量
        self.cap = None
        self.face_masks = []
        self.current_mask_index = 0
        
        # 配置参数
        self.config = {
            'window_name': 'MediaPipe FaceSwap',
            'display_size': (1280, 960),
            'hand_face_threshold': 0.1,
            'mask_alpha': 0.28,
            'original_eye_distance': 100
        }

    def load_face_masks(self, face_dir: str = 'face') -> List[np.ndarray]:
        """
        加载所有脸谱图像
        
        Args:
            face_dir: 脸谱图片所在目录
            
        Returns:
            包含所有已加载脸谱的列表
        """
        masks = []
        # 支持 png 和 jpg 格式的图片
        image_files = glob.glob(os.path.join(face_dir, '*.png')) + glob.glob(os.path.join(face_dir, '*.jpg'))
        
        for image_file in image_files:
            mask_img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
            if mask_img is not None:
                masks.append(mask_img)
        return masks

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        旋转图像
        
        Args:
            image: 要旋转的图像
            angle: 旋转角度
            
        Returns:
            旋转后的图像
        """
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
        """
        检测手是否靠近脸部
        
        Args:
            hand_landmarks: 手部关键点
            face_landmarks: 脸部关键点
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            如果手靠近脸部返回True，否则返回False
        """
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
        """
        将脸谱应用到人脸上
        
        Args:
            image: 原始图像
            mask_image: 脸谱图像
            face_landmarks: 人脸关键点
            
        Returns:
            添加脸谱后的图像
        """
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
        except:
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

        # 反转透明度逻辑：透明度值越低，脸谱越明显；透明度值越高，脸谱越透明
        # 0% 完全显示脸谱，100% 完全透明
        inverted_alpha = 1.0 - self.config['mask_alpha']
        
        # 应用脸谱
        if mask_rotated.shape[2] == 4:  # 带Alpha通道的图像
            for i in range(3):
                image[start_y:end_y, start_x:end_x, i] = (
                    image[start_y:end_y, start_x:end_x, i] * inverted_alpha +
                    mask_rotated[mask_start_y:mask_end_y, mask_start_x:mask_end_x, i] * self.config['mask_alpha']
                ).astype(np.uint8)
        else:  # 无Alpha通道的图像
            for i in range(3):
                image[start_y:end_y, start_x:end_x, i] = cv2.addWeighted(
                    image[start_y:end_y, start_x:end_x, i],
                    inverted_alpha,
                    mask_rotated[mask_start_y:mask_end_y, mask_start_x:mask_end_x, i],
                    self.config['mask_alpha'],
                    0
                )

        return image

    def process_frame(self, frame: np.ndarray, face_mesh, hands) -> Tuple[np.ndarray, bool]:
        """
        处理单帧图像
        
        Args:
            frame: 输入帧
            face_mesh: MediaPipe face_mesh 实例
            hands: MediaPipe hands 实例
            
        Returns:
            处理后的帧和是否继续处理的标志
        """
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
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        # 检查手是否靠近脸
                        h, w = frame.shape[:2]
                        if self.is_hand_near_face(hand_landmarks, face_landmarks, w, h):
                            self.current_mask_index = (self.current_mask_index + 1) % len(self.face_masks)

                # 应用当前选中的脸谱
                frame = self.apply_mask(frame, self.face_masks[self.current_mask_index], face_landmarks.landmark)

        # 图像后处理
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, self.config['display_size'])

        return frame, True

    def run(self):
        """运行应用程序"""
        # 初始化
        self.cap = cv2.VideoCapture(0)
        self.face_masks = self.load_face_masks()
        
        if not self.face_masks:
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

            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break

                # 处理帧
                processed_frame, should_continue = self.process_frame(frame, face_mesh, hands)
                if not should_continue:
                    break

                # 显示结果
                cv2.imshow(self.config['window_name'], processed_frame)
                
                # 处理键盘输入
                key = cv2.waitKey(5) & 0xFF
                if key == 27:  # ESC键退出
                    break
                elif key == ord('n'):  # 'n'键切换脸谱
                    self.current_mask_index = (self.current_mask_index + 1) % len(self.face_masks)

        # 清理资源
        self.cleanup()

    def cleanup(self):
        """清理资源"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    """主函数"""
    app = FaceMaskApp()
    try:
        app.run()
    except:
        pass
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()



