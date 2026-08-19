import os
import glob
import datetime
import json
import re

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis > 999:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def convert_json_to_srt(json_path, output_srt_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[-] Không thể đọc file JSON: {e}")
        return False

    # 1. Trích xuất nội dung text từ materials.texts
    text_materials = {}
    texts_list = data.get('materials', {}).get('texts', [])
    for text_mat in texts_list:
        mat_id = text_mat.get('id')
        raw_content = text_mat.get('content', '')
        
        text_val = ""
        # Thử parse content nếu nó là chuỗi JSON
        try:
            content_json = json.loads(raw_content)
            # Cấu trúc của CapCut mới lưu text trong key 'text'
            text_val = content_json.get('text', '')
        except Exception:
            # Nếu không phải JSON, loại bỏ các tag HTML/XML (như <font>...)
            cleaned = re.sub(r'<[^>]*>', '', raw_content)
            text_val = cleaned.strip()
        
        # Fallback regex nếu có định dạng đặc biệt
        if not text_val:
            m = re.search(r'"text"\s*:\s*"([^"]+)"', raw_content)
            if m:
                text_val = m.group(1)
            else:
                text_val = raw_content
                
        text_materials[mat_id] = text_val

    # 2. Lấy các phân đoạn track chứa text và thời gian
    subtitle_segments = []
    for track in data.get('tracks', []):
        # Thông thường track chứa text có attribute = 1 hoặc type = 'text'
        for segment in track.get('segments', []):
            material_id = segment.get('material_id')
            timerange = segment.get('target_timerange') or segment.get('source_timerange')
            
            if timerange and material_id in text_materials:
                start_us = timerange.get('start', 0)
                duration_us = timerange.get('duration', 0)
                
                # CapCut dùng microsecond (1s = 1.000.000us)
                start_sec = start_us / 1000000.0
                end_sec = (start_us + duration_us) / 1000000.0
                
                text = text_materials[material_id]
                if text and text.strip():  # Chỉ lấy phụ đề có chữ
                    subtitle_segments.append({
                        'start': start_sec,
                        'end': end_sec,
                        'text': text.strip()
                    })

    if not subtitle_segments:
        print("[-] Không tìm thấy phụ đề nào trong file dự án này.")
        return False

    # Sắp xếp phụ đề theo thời gian bắt đầu
    subtitle_segments.sort(key=lambda x: x['start'])

    # 3. Ghi ra file SRT
    try:
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            for idx, seg in enumerate(subtitle_segments, 1):
                start_str = format_time(seg['start'])
                end_str = format_time(seg['end'])
                f.write(f"{idx}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{seg['text']}\n\n")
        return True
    except Exception as e:
        print(f"[-] Lỗi khi ghi file SRT: {e}")
        return False

def find_and_select_drafts():
    local_appdata = os.environ.get('LOCALAPPDATA')
    if not local_appdata:
        local_appdata = os.path.expandvars(r'%USERPROFILE%\AppData\Local')

    possible_paths = [
        os.path.join(local_appdata, 'CapCut', 'User Data', 'Projects', 'com.lveditor.draft'),
        os.path.join(local_appdata, 'CapCut', 'User Data', 'Projects', 'com.lved.pc'),
        os.path.join(local_appdata, 'CapCut', 'User Data', 'Projects')
    ]

    draft_files = []
    for path in possible_paths:
        if os.path.exists(path):
            found_files = glob.glob(os.path.join(path, '**', 'draft_content.json'), recursive=True)
            if found_files:
                draft_files = found_files
                break

    if not draft_files:
        print("\n[-] Không tìm thấy thư mục dự án CapCut mặc định.")
        return

    # Sắp xếp các dự án theo thời gian cập nhật mới nhất
    draft_files.sort(key=os.path.getmtime, reverse=True)

    print("\n" + "="*60)
    print(" DANH SÁCH DỰ ÁN CAPCUT TÌM THẤY:")
    print("="*60)
    for idx, file in enumerate(draft_files, 1):
        mtime = os.path.getmtime(file)
        dt = datetime.datetime.fromtimestamp(mtime).strftime('%d/%m/%Y %H:%M:%S')
        folder_name = os.path.basename(os.path.dirname(file))
        if folder_name == 'Timelines':
            # Trường hợp nằm trong thư mục con Timelines
            folder_name = os.path.basename(os.path.dirname(os.path.dirname(file)))
        print(f"{idx:2d}. Dự án: {folder_name}  |  Cập nhật: {dt}")

    print("="*60)
    try:
        choice = input("Nhập số thứ tự dự án bạn muốn xuất SRT (hoặc nhấn Enter để thoát): ").strip()
        if not choice:
            print("Thoát chương trình.")
            return

        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(draft_files):
            print("[-] Số thứ tự không hợp lệ.")
            return

        selected_json = draft_files[choice_idx]
        # Lấy tên dự án để đặt tên file srt xuất ra
        project_name = os.path.basename(os.path.dirname(selected_json))
        if project_name == 'Timelines':
            project_name = os.path.basename(os.path.dirname(os.path.dirname(selected_json)))
            
        # Tạo thư mục 'srt gốc' trong thư mục làm việc hiện tại của tool
        export_dir = os.path.join(os.getcwd(), "srt gốc")
        os.makedirs(export_dir, exist_ok=True)
        
        output_name = f"CapCut_{project_name}.srt"
        output_path = os.path.join(export_dir, output_name)

        print(f"\n[*] Đang chuyển đổi dự án '{project_name}'...")
        if convert_json_to_srt(selected_json, output_path):
            print(f"[+] THÀNH CÔNG! Đã xuất file phụ đề tại:")
            print(f"    👉 {output_path}")
        else:
            print("[-] Xuất phụ đề thất bại.")

    except ValueError:
        print("[-] Vui lòng nhập một số hợp lệ.")
    except KeyboardInterrupt:
        print("\nĐã hủy lệnh.")

if __name__ == "__main__":
    find_and_select_drafts()
