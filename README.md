# 人脸换脸谱应用

这是一个基于 Python 的实时人脸换脸谱应用，使用 MediaPipe 进行人脸检测和手势识别，可以实时为检测到的人脸添加脸谱效果。

## 功能特点

- 实时人脸检测
- 自动脸谱对齐和缩放
- 手势控制切换脸谱
- 支持键盘快捷键
- 支持多种图片格式（PNG/JPG）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

1. 确保 `face` 目录中包含脸谱图片（支持 PNG 和 JPG 格式）
2. 运行程序：
   ```bash
   python main.py
   ```

## 操作方法

- 将手靠近脸部：自动切换下一个脸谱
- 按 'N' 键：手动切换下一个脸谱
- 按 'ESC' 键：退出程序

## 配置说明

可以在 `FaceMaskApp` 类的 `config` 字典中修改以下参数：

- `window_name`: 窗口名称
- `display_size`: 显示窗口大小
- `hand_face_threshold`: 手脸距离阈值
- `mask_alpha`: 脸谱透明度
- `original_eye_distance`: 眼睛距离基准值 