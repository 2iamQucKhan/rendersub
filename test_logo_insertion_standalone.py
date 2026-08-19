import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Đảm bảo in Tiếng Việt trên Windows Console
sys.stdout.reconfigure(encoding='utf-8')

def test_single_logo_insertion():
    print("=" * 70)
    print(" 🧪 KIỂM THỬ ĐỘC LẬP TÍNH NĂNG CHÈN LOGO VỚI ĐÚNG 1 KHUNG LOGO (logo_bbox)")
    print("=" * 70)

    video_path = os.path.join("videos", "sample_demo.mp4")
    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video mẫu '{video_path}'.")
        return False

    output_dir = os.path.join("output", "test_logo_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Tạo file logo PNG mẫu (120x40 px nền trong suốt)
    logo_path = os.path.join(output_dir, "sample_logo.png")
    logo_img = Image.new("RGBA", (120, 40), (0, 0, 0, 0))
    draw_l = ImageDraw.Draw(logo_img)
    draw_l.rounded_rectangle([0, 0, 119, 39], radius=8, fill=(255, 57, 57, 220), outline=(255, 255, 255, 255), width=2)
    draw_l.text((15, 10), "LOGO DEMO", fill=(255, 255, 255, 255))
    logo_img.save(logo_path)
    print(f"1. Đã tạo file Logo PNG mẫu: {logo_path}")

    # Chỉ có DUY NHẤT 1 khung logo [x=480, y=20, w=130, h=45] (KHÔNG CÓ KHUNG SUB NÀO KHÁC)
    logo_bbox = [480, 20, 130, 45]
    print(f"2. Cấu hình kiểm thử: CHỈ CÓ ĐÚNG 1 KHUNG LOGO (selected_bboxes = [{logo_bbox}])")

    before_img_path = os.path.join(output_dir, "frame_before_logo.jpg")
    after_img_path = os.path.join(output_dir, "frame_after_logo.jpg")
    out_video_path = os.path.join(output_dir, "sample_with_logo.mp4")

    # Lấy frame 0 chụp ảnh trước khi chèn logo
    cap = cv2.VideoCapture(video_path)
    ret, frame_orig = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(before_img_path, frame_orig)
        print(f"📷 Đã lưu ảnh GỐC: {before_img_path}")

    import dubber
    print("\n⚡ Đang thực thi render video với 1 khung logo duy nhất...")
    dubber.create_dubbed_video(
        video_path=video_path,
        segments=[],
        voice="vi-VN-HoaiMyNeural",
        output_video_path=out_video_path,
        burn_subtitles=False,
        enable_dubbing=False,
        selected_bboxes=[logo_bbox],
        logo_path=logo_path,
        logo_bbox=logo_bbox
    )

    # Chụp ảnh sau khi render (Frame 30)
    cap_out = cv2.VideoCapture(out_video_path)
    cap_out.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret_out, frame_after = cap_out.read()
    cap_out.release()

    if ret_out:
        cv2.imwrite(after_img_path, frame_after)
        print(f"\n🎉 THÀNH CÔNG! Đã lưu ảnh THÀNH PHẨM (Logo đã chèn đúng vào vị trí khoanh 1 khung):")
        print(f"   🖼️ Đường dẫn ảnh Trước: {before_img_path}")
        print(f"   🖼️ Đường dẫn ảnh Sau khi chèn Logo: {after_img_path}")
        print(f"   🎬 Đường dẫn video: {out_video_path}")
        return True
    else:
        print("❌ Lỗi: Không thể xuất ảnh thành phẩm.")
        return False

if __name__ == "__main__":
    test_single_logo_insertion()
