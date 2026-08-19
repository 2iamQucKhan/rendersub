import os
import sys
import cv2
import numpy as np

# Đảm bảo in Tiếng Việt trên Windows Console
sys.stdout.reconfigure(encoding='utf-8')

def test_title_translation_and_rendering():
    print("=" * 70)
    print(" 🧪 KIỂM THỬ ĐỘC LẬP TÍNH NĂNG DỊCH VÀ VẼ TIÊU ĐỀ VIDEO (title_bbox)")
    print("=" * 70)

    video_path = os.path.join("videos", "sample_demo.mp4")
    if not os.path.exists(video_path):
        print(f"❌ Khởi tạo thất bại: Không tìm thấy video mẫu '{video_path}'.")
        return False

    output_dir = os.path.join("output", "test_title_demo")
    os.makedirs(output_dir, exist_ok=True)
    before_img_path = os.path.join(output_dir, "frame_before_title.jpg")
    after_img_path = os.path.join(output_dir, "frame_after_title.jpg")
    out_video_path = os.path.join(output_dir, "sample_with_title.mp4")

    # Mock title_bbox ở vùng phía trên [x=100, y=20, w=440, h=50]
    title_bbox = [100, 20, 440, 50]
    sub_bbox = [100, 280, 440, 50]
    mock_title_text = "TINH NANG DICH TIEU DE MULTI-LINE"

    print(f"1. Video mẫu: {video_path}")
    print(f"2. Giả lập Khung Tiêu Đề (title_bbox): {title_bbox}")
    print(f"3. Nội dung Tiêu đề dịch: '{mock_title_text}'")

    # Lấy frame 0 chụp ảnh trước khi render
    cap = cv2.VideoCapture(video_path)
    ret, frame_orig = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(before_img_path, frame_orig)
        print(f"📷 Đã lưu ảnh GỐC (Trước khi vẽ tiêu đề): {before_img_path}")

    # Chạy dubber.create_dubbed_video để render video có tiêu đề
    import dubber
    mock_segments = [
        {"start": 0.0, "end": 3.0, "text": "Phụ đề thoại mẫu bên dưới", "bbox": sub_bbox}
    ]

    print("\n⚡ Đang thực thi render video có Tiêu đề & Phụ đề...")
    dubber.create_dubbed_video(
        video_path=video_path,
        segments=mock_segments,
        voice="vi-VN-HoaiMyNeural",
        output_video_path=out_video_path,
        burn_subtitles=True,
        selected_bbox=sub_bbox,
        enable_dubbing=False,
        title_text=mock_title_text,
        title_bbox=title_bbox
    )

    # Chụp ảnh sau khi render (Frame 30)
    cap_out = cv2.VideoCapture(out_video_path)
    cap_out.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret_out, frame_after = cap_out.read()
    cap_out.release()

    if ret_out:
        cv2.imwrite(after_img_path, frame_after)
        print(f"\n🎉 RENDER THÀNH CÔNG! Đã lưu ảnh THÀNH PHẨM (Đè Tiêu Đề vàng & Phụ Đề):")
        print(f"   🖼️ Đường dẫn ảnh Demo: {after_img_path}")
        print(f"   🎬 Đường dẫn video Demo: {out_video_path}")
        return True
    else:
        print("❌ Lỗi: Không thể chụp ảnh thành phẩm từ video mới render.")
        return False

if __name__ == "__main__":
    test_title_translation_and_rendering()
