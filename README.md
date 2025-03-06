# Face Mask Application

A real-time face mask application built with Python that uses MediaPipe for face detection and gesture recognition to apply traditional opera masks to detected faces.

![Face Mask Application Demo](face-mask.png)

## Features

- Real-time face detection
- Automatic mask alignment and scaling
- Gesture-based mask switching
- Keyboard shortcuts support
- Multiple image format support (PNG/JPG)

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure the `face` directory contains mask images (PNG and JPG formats supported)
2. Run the program:
   ```bash
   python main.py
   ```

## Controls

- Move hand close to face: Automatically switch to next mask
- Press 'N' key: Manually switch to next mask
- Press 'ESC' key: Exit program

## Configuration

You can modify the following parameters in the `FaceMaskApp` class's `config` dictionary:

- `window_name`: Window title
- `display_size`: Display window size
- `hand_face_threshold`: Hand-face distance threshold
- `mask_alpha`: Mask transparency
- `original_eye_distance`: Base value for eye distance

## Privacy Notice

This application processes video feed in real-time and does not store any personal data. All processing is done locally on your device. 