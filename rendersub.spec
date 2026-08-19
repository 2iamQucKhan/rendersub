# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

added_datas = [
    ('Data', 'Data'),
    ('config/xkiro_prompt_template.json', 'config'),
    ('config/trending_dict.json', 'config'),
]
if os.path.exists('config/app_settings.json'):
    added_datas.append(('config/app_settings.json', 'config'))

added_binaries = []
if os.path.exists('bin/ffmpeg.exe'):
    added_binaries.append(('bin/ffmpeg.exe', 'bin'))
if os.path.exists('bin/ffprobe.exe'):
    added_binaries.append(('bin/ffprobe.exe', 'bin'))

hidden_imports = [
    'PyQt6.sip',
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'cv2',
    'easyocr',
    'edge_tts',
    'pydub',
    'deep_translator',
    'numpy',
    'matplotlib',
    'matplotlib.backends.backend_qtagg',
    'resource_utils',
    'dubber',
    'transcriber',
    'translator',
    'downloader',
    'optimized_pipeline',
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=added_binaries,
    datas=added_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RenderSub',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # Set to True for console output diagnostics
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='RenderSub',
)
