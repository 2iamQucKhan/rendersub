import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

from optimized_pipeline import enable_opencv_hardware_acceleration, ParallelVideoProcessor, SmartFrameInpainter, FFmpegVideoWriter
import dubber

def create_benchmark_video(filename="benchmark_input.mp4", num_frames=150, width=640, height=360, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)
        # Bắt đầu vẽ chuyển động hình tròn phía trên
        cx = (i * 4) % width
        cv2.circle(frame, (cx, 100), 25, (0, 165, 255), -1)
        # Vùng phụ đề cố định phía dưới (Static Subtitle Region)
        cv2.rectangle(frame, (100, 260), (540, 320), (220, 220, 220), -1)
        cv2.putText(frame, "PHU DE GOC CAN XOA VA VE LAI", (110, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
        out.write(frame)
    out.release()
    return os.path.abspath(filename)

def run_legacy_processing(video_path, output_path, crop_bbox):
    """Mô phỏng quy trình xử lý OpenCV cũ (Đơn luồng, không cache absdiff, cv2.VideoWriter chuẩn)"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    x, y, w, h = crop_bbox
    frames_processed = 0

    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Luôn luôn tính cv2.inpaint trên mỗi frame không bỏ qua
        crop = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        inpainted = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
        frame[y:y+h, x:x+w] = inpainted

        out.write(frame)
        frames_processed += 1

    cap.release()
    out.release()
    elapsed = time.time() - start_time
    fps_val = frames_processed / elapsed if elapsed > 0 else 0
    return frames_processed, elapsed, fps_val

def run_optimized_processing(video_path, output_path, crop_bbox):
    """Quy trình mới tối ưu (Multi-threaded Pipeline + Smart Frame Skipping + Hardware Accel)"""
    x, y, w, h = crop_bbox
    crop_bbox_tuple = (x, y, w, h)

    def process_frame(frame, f_idx, total_frames, fps, inpainter):
        # Sử dụng inpainter đệm diff thông minh
        frame = inpainter.process_crop(frame, crop_bbox_tuple, mask_mode="inpaint")
        return frame

    processor = ParallelVideoProcessor(video_path, output_path, process_frame)
    
    start_time = time.time()
    processor.run()
    elapsed = time.time() - start_time

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fps_val = total_frames / elapsed if elapsed > 0 else 0
    return total_frames, elapsed, fps_val

def benchmark_main():
    print("==========================================================")
    print("=== BƯỚC 2: BÀI THỬ NHIỆM ĐO ĐẠC HIỆU NĂNG (BENCHMARK) ===")
    print("==========================================================")

    # Khởi tạo video mẫu 150 frames
    test_video = create_benchmark_video()
    crop_area = (100, 260, 440, 60)
    
    out_legacy = "bench_out_legacy.mp4"
    out_optimized = "bench_out_optimized.mp4"

    # 1. Đo hiệu năng quy trình cũ (Legacy Single-Thread)
    print("-> 1. Đang chạy quy trình CŨ (Single-threaded, No Cache)...")
    frames_old, time_old, fps_old = run_legacy_processing(test_video, out_legacy, crop_area)
    print(f"   + Kết quả CŨ:   {frames_old} frames | Thời gian: {time_old:.3f}s | Tốc độ: {fps_old:.2f} FPS")

    # 2. Đo hiệu năng quy trình MỚI (Optimized Multi-Thread Pipeline)
    print("-> 2. Đang chạy quy trình MỚI (Multi-threaded Pipeline, Smart Diff Cache, HW Accel)...")
    frames_new, time_new, fps_new = run_optimized_processing(test_video, out_optimized, crop_area)
    print(f"   + Kết quả MỚI:  {frames_new} frames | Thời gian: {time_new:.3f}s | Tốc độ: {fps_new:.2f} FPS")

    # 3. So sánh độ tăng tốc
    speedup = (fps_new / fps_old) if fps_old > 0 else 1.0
    time_saved_pct = ((time_old - time_new) / time_old) * 100.0 if time_old > 0 else 0.0

    print("----------------------------------------------------------")
    print(f"🚀 TĂNG TỐC HIỆU NĂNG: Gấp {speedup:.2f}x (Tiết kiệm {time_saved_pct:.1f}% thời gian)")
    print("==========================================================")

    # Dọn dẹp tệp tạm
    for f in [test_video, out_legacy, out_optimized]:
        if os.path.exists(f):
            try: os.remove(f)
            except Exception: pass

    return {
        "fps_old": fps_old,
        "fps_new": fps_new,
        "speedup": speedup
    }

if __name__ == "__main__":
    benchmark_main()
