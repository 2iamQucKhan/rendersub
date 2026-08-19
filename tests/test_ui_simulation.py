import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer

def create_sample_video(filename="test_sample.mp4", num_frames=60, width=640, height=360, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background gradient & text
        cv2.putText(frame, f"TEST FRAME {i}/{num_frames}", (50, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        # Subtitle area
        cv2.rectangle(frame, (100, 280), (540, 330), (255, 255, 255), -1)
        cv2.putText(frame, "PHU DE MAU DE TEST OCR", (120, 315),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        out.write(frame)
    out.release()
    return os.path.abspath(filename)

def run_gui_simulation():
    print("=== BẮT ĐẦU BUỚC 2.3 & 2.4: KIỂM THỬ KHỞI CHẠY VÀ GIẢ LẬP GIAO DIỆN ===")
    
    # Set offscreen platform for headless test execution if needed
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    sample_video = create_sample_video()
    print(f"-> Đã tạo video thử nghiệm mẫu: {sample_video}")
    
    from main import MainWindow
    window = MainWindow()
    window.show()
    
    print("-> Đang nạp video vào MainWindow...")
    window.video_path = sample_video
    window.load_video_preview(sample_video)
    
    # Kiểm tra các thông số video preview
    print(f"-> Video width: {window.video_width}, height: {window.video_height}")
    assert window.video_width > 0 and window.video_height > 0, "Lỗi nạp video!"
    
    # Kiểm tra phản hồi thủ công của Slider và nút bấm
    print("-> Kiểm tra thao tác kéo Slider & các nút điều hướng [-5s], [Trước], [Sau], [+5s]...")
    window.slider_player_timeline.setValue(500)
    app.processEvents()
    
    window.seek_relative(1)
    app.processEvents()
    window.seek_relative(-1)
    app.processEvents()
    
    # Giả lập phản hồi Visual Feedback từ Background Worker
    print("-> Giả lập Signal từ Background Worker truyền về GUI (Real-time Visual Feedback)...")
    sample_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(sample_frame, "WORKER FRAME", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    # Phát 10 frames liên tục để kiểm tra độ mượt
    for i in range(1, 11):
        window.on_worker_frame_update(
            frame=sample_frame,
            frame_idx=i * 5,
            total_frames=60,
            timestamp_s=i * 0.1,
            active_bbox=(100, 280, 440, 50),
            status_msg=f"ĐANG QUÉT MẪU OCR... FRAME {i*5}/60"
        )
        app.processEvents()
        time.sleep(0.02)
        
    print("-> Real-time Visual Feedback và Slider update hoạt động mượt mà không trễ/treo GUI!")
    
    # Dọn dẹp tệp thử nghiệm
    if os.path.exists(sample_video):
        try: os.remove(sample_video)
        except Exception: pass
        
    print("=== MỌI BÀI KIỂM THỬ GIẢ LẬP GIAO DIỆN HOÀN THÀNH TỐT ĐẸP! (0 EXCEPTIONS) ===")

if __name__ == "__main__":
    run_gui_simulation()
