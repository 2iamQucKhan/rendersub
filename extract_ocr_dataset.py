import os
import cv2
import re
import argparse

def srt_time_to_seconds(t_str):
    t_str = t_str.replace('.', ',')
    h, m, s_ms = t_str.split(':')
    s, ms = s_ms.split(',')
    ms = ms.ljust(3, '0')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def parse_srt(srt_path):
    """Đọc file SRT và trả về danh sách phân đoạn (start_sec, end_sec, text)"""
    segments = []
    if not os.path.exists(srt_path):
        return segments
    
    with open(srt_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        content = f.read()
        
    lines = content.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line or line.isdigit():
            i += 1
            continue
        
        if "-->" in line:
            time_match = re.findall(r'\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}', line)
            if len(time_match) >= 2:
                start_sec = srt_time_to_seconds(time_match[0])
                end_sec = srt_time_to_seconds(time_match[1])
                
                # Đọc nội dung chữ bên dưới
                text_lines = []
                i += 1
                while i < n:
                    next_line = lines[i].strip()
                    if not next_line:
                        break
                    if "-->" in next_line:
                        i -= 1
                        break
                    if next_line.isdigit() and i + 1 < n and "-->" in lines[i+1]:
                        break
                    text_lines.append(next_line)
                    i += 1
                
                text = " ".join(text_lines).strip()
                if text:
                    segments.append({
                        'start': start_sec,
                        'end': end_sec,
                        'text': text
                    })
        i += 1
    return segments

def extract_ocr_dataset(video_path, srt_path, output_dir="ocr_dataset", y_start_pct=0.7, y_end_pct=0.95):
    """
    Tự động hóa cắt ảnh phụ đề từ video và file SRT tương ứng
    y_start_pct và y_end_pct xác định vùng cắt phụ đề theo chiều dọc (ví dụ 70% đến 95% chiều cao video)
    """
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy video {video_path}")
        return
    if not os.path.exists(srt_path):
        print(f"Lỗi: Không tìm thấy file SRT {srt_path}")
        return
        
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    # File nhãn định dạng PaddleOCR: đường_dẫn_ảnh\tNội_dung_chữ
    gt_file_path = os.path.join(output_dir, "rec_gt.txt")
    
    segments = parse_srt(srt_path)
    print(f"Đã đọc được {len(segments)} dòng phụ đề từ SRT.")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps <= 0 or width <= 0 or height <= 0:
        print("Lỗi: Không thể lấy thông tin video.")
        cap.release()
        return
        
    # Tính toán tọa độ crop mặc định theo chiều dọc
    y1 = int(height * y_start_pct)
    y2 = int(height * y_end_pct)
    
    saved_count = 0
    gt_lines = []
    
    for idx, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        text = seg['text']
        
        # Nhảy tới khung hình ở GIỮA thời gian bắt đầu và kết thúc của câu
        target_time = (start + end) / 2.0
        frame_idx = int(target_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Cắt lấy vùng phụ đề
        crop_img = frame[y1:y2, 0:width]
        
        # Lưu file ảnh
        img_name = f"sub_{idx:05d}.jpg"
        img_path = os.path.join(img_dir, img_name)
        cv2.imwrite(img_path, crop_img)
        
        # Định dạng dòng nhãn: images/sub_00001.jpg\tNội dung chữ
        gt_line = f"images/{img_name}\t{text}"
        gt_lines.append(gt_line)
        saved_count += 1
        
    cap.release()
    
    # Ghi file nhãn
    with open(gt_file_path, "a", encoding="utf-8") as gt_file:
        for line in gt_lines:
            gt_file.write(line + "\n")
            
    print(f"Hoàn thành! Đã cắt và lưu thành công {saved_count} ảnh phụ đề vào thư mục '{output_dir}'.")
    print(f"File nhãn lưu tại: {gt_file_path}")

if __name__ == "__main__":
    # Ví dụ cách chạy nhanh bằng dòng lệnh
    # python extract_ocr_dataset.py --video video.mp4 --srt srt_goc/video.srt
    parser = argparse.ArgumentParser(description="Tự động trích xuất Dataset để train OCR từ Video và SRT")
    parser.add_argument("--video", type=str, help="Đường dẫn file video gốc")
    parser.add_argument("--srt", type=str, help="Đường dẫn file SRT tương ứng")
    parser.add_argument("--out", type=str, default="ocr_dataset", help="Thư mục lưu kết quả")
    
    args = parser.parse_args()
    if args.video and args.srt:
        extract_ocr_dataset(args.video, args.srt, args.out)
    else:
        print("Vui lòng chạy lệnh kèm tham số. Ví dụ:")
        print("python extract_ocr_dataset.py --video 'tên_video.mp4' --srt 'tên_sub.srt'")
