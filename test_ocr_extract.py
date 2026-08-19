import os
import sys
import glob
import cv2
import time
import json
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import transcriber

def test_ocr():
    videos_dir = os.path.join(root_dir, "videos")
    files = [f for f in glob.glob(os.path.join(videos_dir, "*.mp4")) if "output_" not in f and "_tested" not in f and "_dubbed" not in f]
    print(f"Danh sách video thật trong videos/: {files}")
    if not files:
        print("Không tìm thấy video.")
        return
        
    target_video = files[0]
    print(f"Đã chọn file video gốc: {os.path.basename(target_video)}")
    
    cap = cv2.VideoCapture(target_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Thuộc tính video: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    # Bbox nằm ở phần dưới video nơi thường chứa phụ đề
    bbox = [20, int(height * 0.70), width - 40, int(height * 0.28)]
    
    print("Bắt đầu chạy OCR quét khung hình video...")
    try:
        segments = transcriber.run_hardsub_ocr(
            target_video,
            bbox=bbox,
            ocr_lang="Tự động (Trung, Việt, Anh)",
            progress_callback=lambda msg: print(f"[OCR Progress] {msg}")
        )
        print(f"Kết quả OCR ({len(segments)} câu):")
        for i, s in enumerate(segments):
            print(f"  #{i+1} [{s['start']:.2f}s -> {s['end']:.2f}s]: {s.get('text', '')}")
    except Exception as e:
        print(f"Lỗi khi chạy OCR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ocr()
