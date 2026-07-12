import os
import asyncio
import subprocess
import cv2
import edge_tts
from pydub import AudioSegment
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_SUPPORTED_VOICES_CACHE = None


def _run_ffmpeg(cmd, action_label):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("Khong tim thay ffmpeg. Vui long cai ffmpeg va them vao PATH.") from exc
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        raise RuntimeError(f"{action_label} that bai:\n{err[-1200:]}") from exc

async def _get_all_voices_async():
    try:
        # Lấy toàn bộ danh sách giọng đọc từ máy chủ Microsoft
        voices = await edge_tts.list_voices()
        result = []
        for v in voices:
            # Dịch giới tính sang Tiếng Việt
            gender_vn = "Nữ" if v["Gender"] == "Female" else "Nam"
            result.append({
                "name": v["ShortName"],
                "desc": f"{v['Locale']} - {v['FriendlyName'].replace('Microsoft', '').replace('Online', '').strip()} ({gender_vn})",
                "locale": v["Locale"]
            })
            
        # Sắp xếp thứ tự ưu tiên: Tiếng Việt (vi-VN) đầu tiên, rồi tới Tiếng Anh (en-), sau đó là các tiếng khác
        def sort_key(item):
            loc = item["locale"].lower()
            if loc == "vi-vn":
                return (0, item["name"])
            elif loc.startswith("en-"):
                return (1, item["name"])
            else:
                return (2, item["name"])
                
        result.sort(key=sort_key)
        return result
    except Exception as e:
        print(f"Khong the lay danh sach giong doc online: {e}")
        return None

def download_google_tts(text, output_path):
    import urllib.request
    import urllib.parse
    try:
        encoded_text = urllib.parse.quote(text)
        url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl=vi&client=tw-ob&q={encoded_text}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            audio_data = response.read()
            with open(output_path, "wb") as f:
                f.write(audio_data)
        return True
    except Exception as e:
        print(f"Error downloading Google TTS: {e}")
        return False

# Lấy danh sách các giọng đọc hỗ trợ của Edge-TTS
def get_supported_voices():
    """
    Trả về danh sách giọng đọc đầy đủ tải trực tiếp từ Edge-TTS.
    Nếu mất mạng, tự động dự phòng sang danh sách giọng đọc cơ bản offline.
    """
    global _SUPPORTED_VOICES_CACHE
    if _SUPPORTED_VOICES_CACHE:
        return list(_SUPPORTED_VOICES_CACHE)

    google_voice = {"name": "google-translate-vi", "desc": "vi-VN - Chị Google (Meme/Hài hước)"}
    try:
        voices = asyncio.run(_get_all_voices_async())
        if voices:
            _SUPPORTED_VOICES_CACHE = [google_voice] + voices
            return list(_SUPPORTED_VOICES_CACHE)
    except Exception:
        pass
        
    # Danh sách dự phòng offline
    _SUPPORTED_VOICES_CACHE = [
        google_voice,
        {"name": "vi-VN-HoaiMyNeural", "desc": "vi-VN - Hoài My (Nữ)"},
        {"name": "vi-VN-NamMinhNeural", "desc": "vi-VN - Nam Minh (Nam)"},
        {"name": "en-US-AriaNeural", "desc": "en-US - Aria (Nữ)"},
        {"name": "en-US-GuyNeural", "desc": "en-US - Guy (Nam)"},
        {"name": "zh-CN-XiaoxiaoNeural", "desc": "zh-CN - Xiaoxiao (Nữ)"},
        {"name": "zh-CN-YunxiNeural", "desc": "zh-CN - Yunxi (Nam)"},
        {"name": "ja-JP-NanamiNeural", "desc": "ja-JP - Nanami (Nữ)"},
        {"name": "ja-JP-KeitaNeural", "desc": "ja-JP - Keita (Nam)"},
        {"name": "ko-KR-SunHiNeural", "desc": "ko-KR - SunHi (Nữ)"},
        {"name": "ko-KR-InJoonNeural", "desc": "ko-KR - InJoon (Nam)"}
    ]
    return list(_SUPPORTED_VOICES_CACHE)

# Hàm sinh file âm thanh TTS cho một câu đơn lẻ
async def _generate_tts_async(text, voice, output_path, rate="+0%", pitch="+0Hz"):
    if voice == "google-translate-vi":
        success = download_google_tts(text, output_path)
        if not success:
            raise Exception("Tải giọng đọc Google TTS thất bại.")
        return
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_path)

def generate_tts(text, voice, output_path, rate="+0%", pitch="+0Hz"):
    """
    Sinh giọng đọc TTS đồng bộ từ Edge-TTS (tự động nhận diện và bíp hóa các cụm ***).
    """
    import re
    # Kiểm tra xem có ký tự *** kiểm duyệt nào không
    parts = re.split(r'(\*+)', text)
    if len(parts) <= 1:
        # Trường hợp không có ***: sinh TTS bình thường
        try:
            asyncio.run(_generate_tts_async(text, voice, output_path, rate, pitch))
            return True
        except Exception as e:
            print(f"Loi khi sinh TTS: {e}")
            return False
            
    # Trường hợp có ***: sinh từng phân đoạn và chèn tiếng bíp
    try:
        from pydub.generators import Sine
        combined_audio = AudioSegment.empty()
        
        # Tiếng bíp tần số 1000Hz, độ dài 400ms, giảm âm lượng -10dB
        beep_sound = Sine(1000).to_audio_segment(duration=400).apply_gain(-10)
        
        # Tạo một thư mục tạm riêng cho các phân đoạn nhỏ
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_tts_parts")
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, part in enumerate(parts):
            if i % 2 == 1:
                # Đây là cụm *** -> Chèn tiếng bíp bíp
                combined_audio += beep_sound
            else:
                # Đây là văn bản thông thường
                part_stripped = part.strip()
                # Nếu rỗng hoặc chỉ toàn ký tự dấu câu/khoảng trắng, bù bằng khoảng lặng ngắn tương ứng
                if not part_stripped or all(c in ",.!? !?，。！？" for c in part_stripped):
                    if len(part) > 0:
                        combined_audio += AudioSegment.silent(duration=len(part) * 50)
                    continue
                
                # Sinh TTS cho phân đoạn nhỏ này
                part_path = os.path.join(temp_dir, f"part_{i}.mp3")
                if os.path.exists(part_path):
                    try:
                        os.remove(part_path)
                    except:
                        pass
                    
                asyncio.run(_generate_tts_async(part, voice, part_path, rate, pitch))
                
                if os.path.exists(part_path):
                    try:
                        part_audio = AudioSegment.from_file(part_path)
                        combined_audio += part_audio
                    except Exception as pe:
                        print(f"Loi doc am thanh phan doan TTS: {pe}")
                    try:
                        os.remove(part_path)
                    except:
                        pass
                else:
                    # Bù khoảng lặng nếu sinh lỗi
                    combined_audio += AudioSegment.silent(duration=300)
                    
        # Dọn dẹp thư mục tạm
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        # Xuất file âm thanh hợp nhất
        combined_audio.export(output_path, format="mp3")
        return True
    except Exception as e:
        print(f"Loi khi sinh TTS kiem duyet bip: {e}")
        return False


# Căn chỉnh tốc độ của file âm thanh bằng FFmpeg
def speed_adjust_audio(input_path, output_path, factor):
    """
    Tăng/giảm tốc độ file âm thanh dùng bộ lọc atempo của FFmpeg.
    Hỗ trợ chuỗi atempo (chain) cho hệ số > 2.0:
    Ví dụ: factor=3.0 → atempo=2.0,atempo=1.5
           factor=5.0 → atempo=2.0,atempo=2.0,atempo=1.25
    """
    factor = max(0.5, factor)  # Không giới hạn trên, chỉ giới hạn dưới 0.5
    # Nếu hệ số tốc độ xấp xỉ 1.0, sao chép trực tiếp không cần xử lý
    if abs(factor - 1.0) < 0.05:
        if input_path != output_path:
            import shutil
            shutil.copy(input_path, output_path)
        return
    
    # Xây dựng chuỗi atempo filters (mỗi mắt xích tối đa 2.0, tối thiểu 0.5)
    filters = []
    remaining = factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    if remaining < 0.5:
        filters.append("atempo=0.5")
    else:
        filters.append(f"atempo={remaining:.4f}")
        
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter:a", ",".join(filters),
        output_path
    ]
    _run_ffmpeg(cmd, "Dieu chinh toc do audio")

# Lấy độ dài thực tế của file âm thanh (giây)
def get_audio_duration(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0
    except Exception:
        return 0.0

# Lấy độ dài thực tế của video (milli-giây)
def get_video_duration_ms(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return 0
        duration = int((total_frames / fps) * 1000)
        cap.release()
        return duration
    except Exception:
        return 0

# Hàm vẽ phụ đề có dấu và hộp đen che sub gốc
def get_font_path(font_name):
    name_clean = font_name.strip()
    
    # Check in Data/fonts first (relative to project)
    project_fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "fonts")
    if os.path.exists(project_fonts_dir):
        # Check if direct file name
        p_path = os.path.join(project_fonts_dir, name_clean)
        if os.path.exists(p_path) and os.path.isfile(p_path):
            return p_path
        # Check with extensions
        for ext in ['.ttf', '.ttc', '.otf']:
            p_path_ext = os.path.join(project_fonts_dir, name_clean + ext)
            if os.path.exists(p_path_ext) and os.path.isfile(p_path_ext):
                return p_path_ext
            
            # Check basename
            base = os.path.basename(name_clean)
            p_path_base = os.path.join(project_fonts_dir, base)
            if os.path.exists(p_path_base) and os.path.isfile(p_path_base):
                return p_path_base
            p_path_base_ext = os.path.join(project_fonts_dir, base + ext)
            if os.path.exists(p_path_base_ext) and os.path.isfile(p_path_base_ext):
                return p_path_base_ext

    # If direct valid path
    if os.path.exists(name_clean) and os.path.isfile(name_clean):
        return name_clean

    font_mapping = {
        "Arial": "arial.ttf",
        "Arial Bold": "arialbd.ttf",
        "Calibri": "calibri.ttf",
        "Calibri Bold": "calibrib.ttf",
        "Segoe UI": "segoeui.ttf",
        "Segoe UI Bold": "segoeuib.ttf",
        "Times New Roman": "times.ttf",
        "Times New Roman Bold": "timesbd.ttf",
        "Tahoma": "tahoma.ttf",
        "Tahoma Bold": "tahomabd.ttf",
        "Courier New": "cour.ttf",
        "Courier New Bold": "courbd.ttf",
        "Consolas": "consola.ttf",
    }
    filename = font_mapping.get(name_clean, name_clean)
    if not filename.lower().endswith(('.ttf', '.ttc', '.otf')):
        filename += '.ttf'
    
    win_font_path = os.path.join("C:\\Windows\\Fonts", filename)
    if os.path.exists(win_font_path):
        return win_font_path
        
    try:
        fonts_dir = "C:\\Windows\\Fonts"
        for f in os.listdir(fonts_dir):
            if f.lower() == filename.lower():
                return os.path.join(fonts_dir, f)
            base, ext = os.path.splitext(f)
            if base.lower() == name_clean.lower():
                return os.path.join(fonts_dir, f)
    except Exception:
        pass
        
    return "C:\\Windows\\Fonts\\arial.ttf"

def wrap_text(text, font, max_width):
    dummy_img = Image.new('RGB', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    
    paragraphs = text.split('\n')
    all_lines = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word]) if current_line else word
            bbox = dummy_draw.textbbox((0, 0), test_line, font=font)
            line_w = bbox[2] - bbox[0]
            if line_w <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    all_lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    all_lines.append(word)
                    current_line = []
        if current_line:
            all_lines.append(" ".join(current_line))
    return all_lines

def get_text_block_size(lines, font):
    dummy_img = Image.new('RGB', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    
    max_w = 0
    line_heights = []
    
    for line in lines:
        bbox = dummy_draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        max_w = max(max_w, w)
        line_heights.append(h)
    
    line_spacing = int(font.size * 0.2)
    total_h = sum(line_heights) + line_spacing * (len(lines) - 1) if lines else 0
    return max_w, total_h, line_heights, line_spacing

def apply_opencv_watermark_removal(frame, bbox, mask_mode="blur"):
    """
    Xóa hoặc che watermark/phụ đề cũ sử dụng OpenCV (Black Box, Gaussian Blur, hoặc Inpaint).
    Hàm này được thiết kế mô-đun hóa để dễ dàng tích hợp thêm các mô hình YOLO/Template Matching trong tương lai.
    """
    h_frame, w_frame, _ = frame.shape
    x, y, w, h = bbox
    x = max(0, min(x, w_frame))
    y = max(0, min(y, h_frame))
    w = max(0, min(w, w_frame - x))
    h = max(0, min(h, h_frame - y))
    
    if w <= 0 or h <= 0:
        return frame
        
    if mask_mode == "black":
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
    elif mask_mode == "blur":
        crop = frame[y:y+h, x:x+w]
        k_w = 51 if 51 < w else (w - 1 if w % 2 == 0 else w)
        k_h = 51 if 51 < h else (h - 1 if h % 2 == 0 else h)
        k_w = max(1, k_w)
        k_h = max(1, k_h)
        if k_w % 2 == 0: k_w = max(1, k_w - 1)
        if k_h % 2 == 0: k_h = max(1, k_h - 1)
        blurred = cv2.GaussianBlur(crop, (k_w, k_h), 0)
        frame[y:y+h, x:x+w] = blurred
    elif mask_mode == "inpaint":
        crop = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        inpainted = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
        frame[y:y+h, x:x+w] = inpainted
        
    return frame

def draw_burned_subtitle(frame, text, bbox, default_bbox=None, font_path=None, preset=None):
    h_frame, w_frame, _ = frame.shape
    
    # 1. Vẽ hộp che phụ đề cũ nếu dùng OpenCV và có bbox
    remove_algo = preset.get("remove_algo", "opencv") if preset else "opencv"
    if remove_algo != "ffmpeg":
        box = bbox or default_bbox
        if box:
            mask_mode = preset.get("mask_mode", "blur") if preset else "blur"
            frame = apply_opencv_watermark_removal(frame, box, mask_mode)
            
    if not text or not text.strip():
        return frame, False
        
    # 2. Chuẩn bị preset
    if preset is None:
        preset = {
            "v_align": "bottom",
            "h_align": "center",
            "margin_v_type": "percent",
            "margin_v_val": 8.0,
            "margin_h_type": "percent",
            "margin_h_val": 5.0,
            "font_name": "Arial",
            "font_size": 20,
            "font_color": [255, 255, 255],
            "outline_color": [0, 0, 0],
            "outline_width": 2,
            "bg_color": [0, 0, 0],
            "bg_opacity": 0,
            "use_bg_box": False
        }
        
    # Tính margin thực tế dựa vào độ phân giải video
    margin_v_val = preset.get("margin_v_val", 8.0)
    margin_v_type = preset.get("margin_v_type", "percent")
    if margin_v_type == "percent":
        margin_v_px = int(h_frame * (margin_v_val / 100.0))
    else:
        margin_v_px = int(margin_v_val)
        
    margin_h_val = preset.get("margin_h_val", 5.0)
    margin_h_type = preset.get("margin_h_type", "percent")
    if margin_h_type == "percent":
        margin_h_px = int(w_frame * (margin_h_val / 100.0))
    else:
        margin_h_px = int(margin_h_val)
        
    max_width = max(100, w_frame - 2 * margin_h_px)
    
    # 3. Tự động xuống dòng và giảm font_size (Auto word-wrap & shrink size)
    font_size = preset.get("font_size", 20)
    min_font_size = 12
    font_name = preset.get("font_name", "Arial")
    font_path_resolved = get_font_path(font_path or font_name)
    
    font = None
    lines = []
    text_w, text_h = 0, 0
    line_heights = []
    line_spacing = 0
    
    # Đo đạc và tự động thu nhỏ font nếu quá giới hạn
    while font_size >= min_font_size:
        try:
            font = ImageFont.truetype(font_path_resolved, font_size)
        except Exception:
            font = ImageFont.load_default()
            
        lines = wrap_text(text, font, max_width)
        text_w, text_h, line_heights, line_spacing = get_text_block_size(lines, font)
        
        # Kiểm tra xem có vừa vặn theo chiều ngang (không có dòng nào vượt max_width) và chiều dọc (<= 25% chiều cao video)
        fits_horizontally = True
        dummy_img = Image.new('RGB', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        for line in lines:
            line_bbox = dummy_draw.textbbox((0, 0), line, font=font)
            line_w = line_bbox[2] - line_bbox[0]
            if line_w > max_width:
                fits_horizontally = False
                break
                
        if fits_horizontally and text_h <= h_frame * 0.25:
            break
            
        if font_size == min_font_size:
            break
        font_size -= 1
        
    # Phát hiện xem có bị tràn hay không (sau khi đã giảm về cỡ min 12px)
    overflowed = False
    if font_size == min_font_size:
        dummy_img = Image.new('RGB', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        for line in lines:
            line_bbox = dummy_draw.textbbox((0, 0), line, font=font)
            line_w = line_bbox[2] - line_bbox[0]
            if line_w > max_width:
                overflowed = True
                break
        if text_h > h_frame * 0.25:
            overflowed = True
        
    # 4. Vẽ văn bản bằng Pillow để hỗ trợ tiếng Việt Unicode
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Tính toạ độ dọc y_start cho khối chữ
    custom_pos = preset.get("custom_pos") if preset else None
    smart_pos = preset.get("smart_pos", False) if preset else False
    box = bbox or default_bbox
    if custom_pos:
        center_y = h_frame * float(custom_pos.get("y_pct", 50.0)) / 100.0
        y_start = int(center_y - (text_h / 2))
    elif smart_pos and box:
        x, y, w, h = box
        center_x = x + (w / 2)
        center_y = y + (h / 2)
        y_start = int(center_y - (text_h / 2))
    else:
        v_align = preset.get("v_align", "bottom")
        if v_align == "top":
            y_start = margin_v_px
        elif v_align == "middle":
            y_start = (h_frame - text_h) // 2
        else: # bottom
            y_start = h_frame - text_h - margin_v_px
            
    y_start = max(0, min(y_start, h_frame - text_h))
    
    h_align = preset.get("h_align", "center")
    font_color = tuple(preset.get("font_color", [255, 255, 255]))
    outline_color = tuple(preset.get("outline_color", [0, 0, 0]))
    outline_width = preset.get("outline_width", 2)
    use_bg_box = preset.get("use_bg_box", False)
    bg_color = tuple(preset.get("bg_color", [0, 0, 0]))
    bg_opacity = preset.get("bg_opacity", 0)
    
    current_y = y_start
    for idx, line in enumerate(lines):
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_w = line_bbox[2] - line_bbox[0]
        line_h = line_bbox[3] - line_bbox[1]
        
        # Căn lề ngang cho dòng hiện tại
        if custom_pos:
            center_x = w_frame * float(custom_pos.get("x_pct", 50.0)) / 100.0
            line_x = int(center_x - (line_w / 2))
        elif smart_pos and box:
            line_x = int(center_x - (line_w / 2))
        elif h_align == "left":
            line_x = margin_h_px
        elif h_align == "right":
            line_x = w_frame - line_w - margin_h_px
        else: # center
            line_x = (w_frame - line_w) // 2
            
        line_x = max(0, min(line_x, w_frame - line_w))
        
        # Vẽ background box
        if use_bg_box and bg_opacity > 0:
            padding = 6
            box_x1 = max(0, line_x - padding)
            box_y1 = max(0, current_y - padding)
            box_x2 = min(w_frame, line_x + line_w + padding)
            box_y2 = min(h_frame, current_y + line_h + padding)
            
            if bg_opacity < 255:
                overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill=bg_color + (bg_opacity,))
                pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(pil_img)
            else:
                draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill=bg_color)
                
        # Vẽ viền chữ và chữ
        draw.text((line_x, current_y - line_bbox[1]), line, font=font, fill=font_color,
                  stroke_width=outline_width, stroke_fill=outline_color)
                  
        current_y += line_h + line_spacing
        
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), overflowed

# Hàm ghi đè phụ đề lên video từng frame
def process_video_subtitles(video_path, segments, output_temp_video, default_bbox=None, preset=None, progress_callback=None, draw_text=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Khong the mo video de ghi phu de.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError("Video khong co FPS, kich thuoc hoac so frame hop le.")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_temp_video, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise ValueError("Khong the tao file video tam de ghi phu de.")
    
    overflowed_segments = []
    sorted_segments = sorted(segments or [], key=lambda s: s.get('start', 0.0))
    seg_idx = 0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp_s = frame_idx / fps
        
        active_text = None
        active_bbox = None
        while seg_idx < len(sorted_segments) and sorted_segments[seg_idx].get('end', 0.0) < timestamp_s:
            seg_idx += 1
        if seg_idx < len(sorted_segments):
            seg = sorted_segments[seg_idx]
            if seg.get('start', 0.0) <= timestamp_s <= seg.get('end', 0.0):
                active_text = seg.get('text', '').strip()
                active_bbox = seg.get('bbox')
                
        if active_text:
            text_to_draw = active_text if draw_text else ""
            frame, overflowed = draw_burned_subtitle(frame, text_to_draw, active_bbox, default_bbox, preset=preset)
            if draw_text and overflowed and active_text not in overflowed_segments:
                overflowed_segments.append(active_text)
        elif default_bbox:
            frame, _ = draw_burned_subtitle(frame, "", None, default_bbox, preset=preset)
            
        out.write(frame)
        frame_idx += 1
        
        if progress_callback and frame_idx % 30 == 0:
            percent = int((frame_idx / total_frames) * 100) if total_frames else 0
            msg = "Đang ghi đè phụ đề lên khung hình..." if draw_text else "Đang thực hiện xử lý ảnh xóa watermark..."
            progress_callback(f"{msg} {percent}% ({frame_idx}/{total_frames})")
            
    cap.release()
    out.release()
    return overflowed_segments

# Tạo lồng tiếng và trộn video cuối cùng
def create_dubbed_video(video_path, segments, voice, output_video_path, bg_volume=0.1, dub_volume=1.0, burn_subtitles=False, selected_bbox=None, preset=None, progress_callback=None):
    """
    Tạo lồng tiếng cho video:
    1. Sinh TTS cho từng câu phụ đề.
    2. Căn chỉnh tốc độ khớp mốc thời gian.
    3. Trộn tất cả câu lồng tiếng vào 1 track âm thanh trống.
    4. Trộn track âm thanh mới với video gốc hoặc video đã ghi đè phụ đề sử dụng FFmpeg.
    """
    temp_dir = os.path.join(os.path.dirname(video_path), "temp_dub")
    os.makedirs(temp_dir, exist_ok=True)
    
    video_dur_ms = get_video_duration_ms(video_path)
    if video_dur_ms <= 0:
        raise ValueError("Không thể lấy độ dài video gốc.")
        
    # 1. Khởi tạo track âm thanh lồng tiếng trống bằng pydub
    dubbed_audio = AudioSegment.silent(duration=video_dur_ms)
    
    total_segments = len(segments)
    
    for idx, seg in enumerate(segments):
        start_time = seg['start']
        end_time = seg['end']
        sub_dur = end_time - start_time
        
        if sub_dur <= 0 or not seg['text'].strip():
            continue
            
        if progress_callback:
            progress_callback(f"Đang sinh giọng đọc AI cho câu {idx+1}/{total_segments}...")
            
        # Tạo file âm thanh tạm thời
        raw_audio_path = os.path.join(temp_dir, f"raw_{idx}.mp3")
        aligned_audio_path = os.path.join(temp_dir, f"aligned_{idx}.mp3")
        
        # Sinh giọng đọc
        success = generate_tts(seg['text'], voice, raw_audio_path)
        if not success or not os.path.exists(raw_audio_path):
            continue
            
        # Tính toán hệ số tốc độ
        tts_dur = get_audio_duration(raw_audio_path)
        if tts_dur <= 0:
            continue
            
        speed_factor = tts_dur / sub_dur
        
        # Căn chỉnh tốc độ đọc nếu tốc độ thực tế dài hơn thời gian phụ đề
        if speed_factor > 1.05:
            # Tăng tốc độ đọc lên để vừa khít phân đoạn
            speed_adjust_audio(raw_audio_path, aligned_audio_path, speed_factor)
        else:
            # Giữ nguyên tốc độ đọc tự nhiên nếu câu lồng tiếng ngắn hơn thời gian phụ đề
            speed_adjust_audio(raw_audio_path, aligned_audio_path, 1.0)
            
        # Ghi đè âm thanh vào track chính theo đúng mốc thời gian (quy đổi ra ms)
        start_ms = int(start_time * 1000)
        target_ms = int(sub_dur * 1000)
        try:
            aligned_audio = AudioSegment.from_file(aligned_audio_path)
            # Hard truncate: cắt bớt audio nếu vẫn dài hơn cửa sổ thời gian cho phép
            if len(aligned_audio) > target_ms:
                aligned_audio = aligned_audio[:target_ms]
            # Overlay âm thanh
            dubbed_audio = dubbed_audio.overlay(aligned_audio, position=start_ms)
        except Exception as e:
            print(f"Loi chen am thanh phan doan {idx}: {e}")
            
    # Xuất file âm thanh lồng tiếng tổng hợp
    final_dubbed_wav = os.path.join(temp_dir, "final_dubbed_voice.wav")
    if progress_callback:
        progress_callback("Đang xuất file lồng tiếng tổng hợp...")
    dubbed_audio.export(final_dubbed_wav, format="wav")
    
    # 2. Xử lý video nếu ghi đè phụ đề hoặc cần xóa watermark bằng OpenCV
    video_to_mix = video_path
    overflowed_segments = []
    
    remove_algo = preset.get("remove_algo", "opencv") if preset else "opencv"
    smart_pos = preset.get("smart_pos", False) if preset else False
    
    # Nếu remove_algo == "opencv" và có selected_bbox, ta vẫn chạy OpenCV để xóa watermark.
    # Nhưng ta sẽ KHÔNG vẽ chữ bằng OpenCV nữa (vì ta sẽ dùng bộ lọc ASS của FFmpeg bên dưới).
    run_opencv_watermark = (selected_bbox and remove_algo == "opencv")
    
    if run_opencv_watermark:
        if progress_callback:
            progress_callback("Đang thực hiện xử lý ảnh xóa watermark bằng OpenCV...")
        temp_burned_video = os.path.join(temp_dir, "temp_burned_video.mp4")
        process_video_subtitles(
            video_path=video_path,
            segments=segments,
            output_temp_video=temp_burned_video,
            default_bbox=selected_bbox,
            preset=preset,
            progress_callback=progress_callback,
            draw_text=False  # Chỉ xóa watermark, không vẽ chữ
        )
        video_to_mix = temp_burned_video
        
    # Tạo tệp phụ đề .ass nếu burn_subtitles là True
    ass_path = None
    if burn_subtitles:
        if progress_callback:
            progress_callback("Đang sinh tệp phụ đề ASS phục vụ kết xuất video...")
        ass_path = os.path.join(temp_dir, "subtitles.ass")
        generate_ass_file(
            segments=segments,
            ass_path=ass_path,
            selected_bbox=selected_bbox,
            preset=preset,
            chk_smart_pos=smart_pos,
            video_path=video_path
        )
        
    # 4. Sử dụng FFmpeg trộn âm thanh lồng tiếng mới vào video
    if progress_callback:
        progress_callback("Đang thực hiện trộn âm thanh và xuất video thành phẩm...")
        
    # Xây dựng các bộ lọc video cho FFmpeg
    ffmpeg_vf = []
    use_delogo = (selected_bbox and remove_algo == "ffmpeg")
    if use_delogo:
        x, y, w, h = selected_bbox
        ffmpeg_vf.append(f"delogo=x={x}:y={y}:w={w}:h={h}")
        
    if burn_subtitles and ass_path:
        escaped_ass_path = ass_path.replace("\\", "/").replace(":", "\\:")
        ffmpeg_vf.append(f"ass='{escaped_ass_path}'")
        
    if bg_volume == 0:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_to_mix,
            "-i", final_dubbed_wav,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-filter:a:1", f"volume={dub_volume}"
        ]
        if ffmpeg_vf:
            cmd.extend([
                "-vf", ",".join(ffmpeg_vf),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23"
            ])
        else:
            cmd.extend([
                "-c:v", "copy"
            ])
        cmd.extend([
            "-shortest",
            output_video_path
        ])
    else:
        # Trộn nhạc nền video gốc (giảm âm lượng) với giọng lồng tiếng (giữ nguyên hoặc tăng âm lượng)
        if ffmpeg_vf:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_to_mix,
                "-i", final_dubbed_wav,
                "-i", video_path,
                "-filter_complex", f"[0:v]{','.join(ffmpeg_vf)}[v_filtered];[2:a]volume={bg_volume}[bg];[1:a]volume={dub_volume}[fg];[bg][fg]amix=inputs=2:duration=first[a]",
                "-map", "[v_filtered]",
                "-map", "[a]",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                output_video_path
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_to_mix,
                "-i", final_dubbed_wav,
                "-i", video_path,
                "-filter_complex", f"[2:a]volume={bg_volume}[bg];[1:a]volume={dub_volume}[fg];[bg][fg]amix=inputs=2:duration=first[a]",
                "-map", "0:v:0",
                "-map", "[a]",
                "-c:v", "copy",
                output_video_path
            ]
        
    _run_ffmpeg(cmd, "Xuat video long tieng")
    
    # Dọn dẹp các tệp tạm thời
    try:
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
    except Exception:
        pass
        
    return output_video_path, overflowed_segments


def format_ass_time(seconds):
    if seconds < 0:
        seconds = 0.0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs == 100:
        cs = 0
        secs += 1
        if secs == 60:
            secs = 0
            mins += 1
            if mins == 60:
                mins = 0
                hrs += 1
    return f"{hrs}:{mins:02d}:{secs:02d}.{cs:02d}"


def generate_ass_file(segments, ass_path, selected_bbox, preset, chk_smart_pos=False, video_path=None):
    width, height = 1920, 1080
    if video_path and os.path.exists(video_path):
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

    if preset is None:
        preset = {}

    font_name = preset.get("font_name", "Arial")
    if font_name.endswith(('.ttf', '.otf', '.ttc')) or "/" in font_name or "\\" in font_name:
        font_name = os.path.splitext(os.path.basename(font_name))[0]

    font_size = preset.get("font_size", 20)
    font_color = preset.get("font_color", [255, 255, 255])
    outline_color = preset.get("outline_color", [0, 0, 0])
    outline_width = preset.get("outline_width", 2)
    bg_color = preset.get("bg_color", [0, 0, 0])
    bg_opacity = preset.get("bg_opacity", 0)
    use_bg_box = preset.get("use_bg_box", False)

    def to_ass_color(rgb, opacity_val=255):
        alpha = 255 - opacity_val
        r, g, b = rgb
        return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"

    primary_color = to_ass_color(font_color)
    out_color = to_ass_color(outline_color)
    back_color = to_ass_color(bg_color, int(bg_opacity * 255 / 100))

    # Khi Smart Pos bật, luôn ép BorderStyle=3 (hộp nền đặc) để che chữ gốc
    if chk_smart_pos:
        border_style = 3
        # Nếu bg_opacity chưa đủ đậm, ép mặc định 80% để hộp nền che kín chữ
        if bg_opacity <= 0:
            back_color = to_ass_color(bg_color, int(80 * 255 / 100))
        # Outline = 0 khi dùng hộp nền đặc Smart Pos (tránh viền thừa bao quanh box)
        outline_width = 0
    else:
        border_style = 3 if use_bg_box else 1

    margin_v_val = preset.get("margin_v_val", 8.0)
    margin_v_type = preset.get("margin_v_type", "percent")
    if margin_v_type == "percent":
        margin_v_px = int(height * (margin_v_val / 100.0))
    else:
        margin_v_px = int(margin_v_val)

    margin_h_val = preset.get("margin_h_val", 5.0)
    margin_h_type = preset.get("margin_h_type", "percent")
    if margin_h_type == "percent":
        margin_h_px = int(width * (margin_h_val / 100.0))
    else:
        margin_h_px = int(margin_h_val)

    custom_pos = preset.get("custom_pos")
    h_map = {"left": 1, "center": 2, "right": 3}
    v_map = {"bottom": 0, "middle": 3, "top": 6}
    alignment = 5 if custom_pos else h_map.get(preset.get("h_align", "center"), 2) + v_map.get(preset.get("v_align", "bottom"), 0)

    lines = []
    lines.append("[Script Info]")
    lines.append("ScriptType: v4.00+")
    lines.append(f"PlayResX: {width}")
    lines.append(f"PlayResY: {height}")
    lines.append("")
    lines.append("[V4+ Styles]")
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
    lines.append(
        f"Style: Default,{font_name},{font_size},{primary_color},&H000000FF,{out_color},{back_color},"
        f"0,0,0,0,100,100,0,0,{border_style},{outline_width},0,{alignment},{margin_h_px},{margin_h_px},{margin_v_px},1"
    )
    lines.append("")
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    for seg in segments:
        text = seg.get('text', '').strip()
        if not text:
            continue

        start_t = format_ass_time(seg['start'])
        end_t = format_ass_time(seg['end'])
        box = seg.get('bbox') or selected_bbox
        formatted_text = text.replace('\n', '\\N')

        if custom_pos:
            center_x = width * float(custom_pos.get("x_pct", 50.0)) / 100.0
            center_y = height * float(custom_pos.get("y_pct", 88.0)) / 100.0
            pos_tag = f"{{\\an5\\pos({center_x:.1f},{center_y:.1f})}}"
            lines.append(f"Dialogue: 0,{start_t},{end_t},Default,,0000,0000,0000,,{pos_tag}{formatted_text}")
        elif chk_smart_pos and box:
            x, y, w, h = box
            center_x = x + (w / 2)
            center_y = y + (h / 2)
            pos_tag = f"{{\\an5\\pos({center_x:.1f},{center_y:.1f})}}"
            lines.append(f"Dialogue: 0,{start_t},{end_t},Default,,0000,0000,0000,,{pos_tag}{formatted_text}")
        else:
            lines.append(f"Dialogue: 0,{start_t},{end_t},Default,,0000,0000,0000,,{formatted_text}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
