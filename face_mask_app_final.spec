# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

current_dir = os.getcwd()

# === 1. 二进制依赖 ===
opencv_binaries = collect_dynamic_libs('cv2')
mediapipe_datas = collect_data_files('mediapipe')
pyside6_datas = collect_data_files('PySide6', include_py_files=False, excludes=['*__pycache__*'])

# === 2. MediaPipe tasks 目录 ===
try:
    import mediapipe
    tasks_path = os.path.join(os.path.dirname(mediapipe.__file__), 'tasks')
    if os.path.exists(tasks_path):
        mediapipe_datas.append((tasks_path, 'mediapipe/tasks'))
except Exception as e:
    print(f"Warning: {e}")

# === 3. 应用资源 ===
app_datas = [('face/', 'face/'), ('wechat.jpg', '.')]

# 确保资源列表正确合并
all_datas = app_datas + mediapipe_datas + pyside6_datas

# 添加运行时路径处理支持
import sys
sys.path.append(current_dir)

# === 4. 隐藏导入 ===
hiddenimports = [
    # MediaPipe
    'mediapipe.python.solutions.drawing_utils',
    'mediapipe.python.solutions.drawing_styles',
    'mediapipe.python.solutions.face_mesh',
    'mediapipe.python.solutions.hands',
    'mediapipe.python.solutions.pose',
    # PySide6
    'PySide6.QtMultimedia',
    'PySide6.QtMultimediaWidgets',
    # NumPy
    'numpy.core._multiarray_umath',
    # Matplotlib（最小必要）
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.colors',
    'matplotlib.lines',
    'matplotlib.collections',
    'matplotlib.patches',
    'matplotlib.path',
    'matplotlib.transforms',
    'matplotlib.figure',
    'matplotlib.backends.backend_agg',
]

# === 5. 排除列表（不再排除 pyexpat / xml / pickle）===
excludes = [
    # 第三方大模块
    'tkinter', '_tkinter', 'Tkinter', 'FixTk', 'tix',
    'IPython', 'scipy', 'pandas', 'sklearn',
    # matplotlib GUI 后端
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_gtk',
    'matplotlib.backends.backend_wx',
    'matplotlib.backends.backend_macosx',
    # 开发/网络（可选）
    'unittest', 'pydoc', 'doctest', 'ftplib', 'ssl',
    # 压缩（可选，但不要动 pyexpat）
    'bz2', 'lzma',
    # 注意：没有 'pickle', 'pyexpat', 'xml', 'urllib'！
]

# === 6. 添加PySide6插件支持 ===
# 显式打包PySide6的plugins目录，确保jpg等图像格式能正常加载
import PySide6
pyside6_dir = os.path.dirname(PySide6.__file__)
plugins_src = os.path.join(pyside6_dir, 'plugins')
if os.path.exists(plugins_src):
    all_datas.append((plugins_src, 'PySide6/plugins'))

# === 7. 分析与构建 ===
a = Analysis(
    ['main_gui_pyside6.py'],
    pathex=[current_dir],
    binaries=opencv_binaries,
    datas=all_datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FaceMaskApp',
    debug=False,
    console=False,  # 调试时设为 True
    strip=False,
    upx=True,
    icon=None
)