# 人脸换脸谱应用

一个实时人脸换脸谱应用，提供Python实现方式，使用MediaPipe进行人脸检测和手势识别，能够将传统戏曲脸谱实时应用到检测到的人脸上。

![Face Mask Application Demo](https://github.com/nxu-game/interesting_assets/raw/main/images/face_mask.png)

## 项目特点

- 🎭 **实时人脸检测**：精准识别面部特征
- 🎯 **自动对齐缩放**：根据人脸特征自动调整脸谱位置和大小
- ✋ **手势控制**：将手靠近脸部可自动切换脸谱（非自动模式下）
- ⌨️ **键盘快捷键**：支持键盘操作便捷切换
- 🔄 **自动切换功能**：支持开启/关闭5秒自动切换脸谱模式
- 🖼️ **优化脸谱预览**：128×128像素方形预览区域，等比缩放居中显示
- 🌐 **跨平台实现**：提供Python实现方式

## 实现方式

### 🐍 Python版本

适用于需要本地部署或与其他Python项目集成的场景。

#### 环境要求

```bash
pip install -r requirements.txt
```

#### 使用方法

1. 确保`face`目录中包含脸谱图像（支持PNG和JPG格式）
2. 运行程序：
   ```bash
   python main_gui.py
   ```

#### 操作控制

- **图形界面控制**：
  - 点击"开始"按钮：启动摄像头和人脸检测
  - 点击"停止"按钮：停止摄像头和人脸检测
  - 勾选"自动切换脸谱"复选框：启用5秒自动切换模式
  - 拖动滑块：调整脸谱透明度

- **手势控制**（仅在非自动模式下生效）：
  - 将手靠近脸部：自动切换到下一个脸谱

- **键盘快捷键**：
  - 按`N`键：手动切换到下一个脸谱
  - 按`ESC`键：退出程序

## 隐私声明

本应用实时处理视频流，不会存储任何个人数据。所有处理均在您的设备本地进行。

## 技术栈

### Python版本
- **语言**：Python 3.7+
- **库**：OpenCV, MediaPipe, NumPy, Tkinter


## 项目结构

```
face_mask/
├── face/                # 脸谱图像目录
├── main_gui.py          # Python版本主程序（带图形界面）
├── requirements.txt     # Python依赖配置
├── README.md            # 主项目说明文档
```

## 联系我们

如有任何问题或建议，欢迎联系：

- WeChat: znzatop

![WeChat](wechat.jpg)

## 更多项目

更多有趣的项目请见：https://github.com/nxu-game/interesting_assets.git
