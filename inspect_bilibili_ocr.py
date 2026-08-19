import os
import sys
import cv2
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import transcriber
import translator
import dubber

def inspect_bilibili_video():
    video_path = os.path.join(root_dir, "videos", "宿舍空调哥舍友夏天不开空调_哔哩哔哩_bilibili.mp4")
    print(f"Path: {video_path}")
    print(f"Exists: {os.path.exists(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video file!")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    print(f"Video props: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}, Duration: {duration:.2f}s")
    
    # Test OCR on subtitle region
    bbox = [10, int(height * 0.65), width - 20, int(height * 0.32)]
    print(f"Crop bbox for OCR: {bbox}")
    
    t0 = time.time()
    segments = transcriber.run_hardsub_ocr(
        video_path,
        bbox=bbox,
        ocr_lang="Trung Giản Thể (ch_sim)",
        progress_callback=lambda msg: print(f"[OCR] {msg}", flush=True)
    )
    t_ocr = time.time() - t0
    print(f"OCR finished in {t_ocr:.2f}s, extracted {len(segments)} segments:")
    for idx, s in enumerate(segments, 1):
        print(f"  #{idx:02d} [{s['start']:.2f}s -> {s['end']:.2f}s]: {s.get('text', '')}")

if __name__ == "__main__":
    inspect_bilibili_video()
