import os
import re
import json
from pathlib import Path
import datetime
import time
from typing import List, Dict, Union, Optional
import cv2
import difflib
import whisper

_easyocr_readers_cache = {}
OCR_DEBUG = os.environ.get("SUPERSUBS_OCR_DEBUG") == "1"

_ocr_debug_dir = None

def get_ocr_debug_dir():
    global _ocr_debug_dir
    if not OCR_DEBUG:
        return None
    if _ocr_debug_dir is None:
        _ocr_debug_dir = Path(__file__).resolve().parent / "Data" / "ocr_debug" / time.strftime("%Y%m%d_%H%M%S")
        _ocr_debug_dir.mkdir(parents=True, exist_ok=True)
    return _ocr_debug_dir

def write_ocr_debug(case_id, message):
    debug_dir = get_ocr_debug_dir()
    if not debug_dir:
        return
    with (debug_dir / "ocr_debug.log").open("a", encoding="utf-8") as f:
        f.write(f"[{case_id}] {message}\n")

def save_ocr_debug_image(case_id, name, image):
    debug_dir = get_ocr_debug_dir()
    if not debug_dir:
        return
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    cv2.imwrite(str(debug_dir / f"{case_id}_{safe_name}.png"), image)

def get_easyocr_lang_candidates(ocr_lang):
    text = (ocr_lang or "auto").lower()
    if "phồn" in text or "phon" in text or "ch_tra" in text or "traditional" in text:
        return [['ch_tra', 'en'], ['ch_sim', 'en'], ['en']]
    if "giản" in text or "gian" in text or "ch_sim" in text or "simplified" in text or "trung" in text or "chinese" in text or "auto" in text or "tự động" in text:
        return [['ch_sim', 'en'], ['ch_tra', 'en'], ['en']]
    if "việt" in text or "viet" in text or "vi" in text:
        return [['vi', 'en'], ['ch_sim', 'en'], ['en']]
    if "anh" in text or "english" in text or "en" in text:
        return [['en'], ['ch_sim', 'en']]
    if "nhật" in text or "ja" in text:
        return [['ja', 'en']]
    if "hàn" in text or "ko" in text:
        return [['ko', 'en']]
    if "pháp" in text or "fr" in text:
        return [['fr', 'en']]
    if "đức" in text or "de" in text:
        return [['de', 'en']]
    if "tây ban nha" in text or "spanish" in text or "es" in text:
        return [['es', 'en']]
    if "nga" in text or "ru" in text:
        return [['ru', 'en']]
    if "thái" in text or "thai" in text or "th" in text:
        return [['th', 'en']]
    return [['ch_sim', 'en'], ['ch_tra', 'en'], ['en']]

def get_easyocr_reader(lang_list):
    lang_tuple = tuple(sorted(lang_list))
    if lang_tuple not in _easyocr_readers_cache:
        import easyocr
        import torch
        print(f"[DEBUG] Khởi tạo EasyOCR Reader mới cho ngôn ngữ: {lang_list}")
        _easyocr_readers_cache[lang_tuple] = easyocr.Reader(list(lang_tuple), gpu=torch.cuda.is_available())
    return _easyocr_readers_cache[lang_tuple]

# Hàm phụ trợ chuyển đổi định dạng thời gian SRT sang giây
def srt_time_to_seconds(t_str):
    t_str = t_str.replace('.', ',') # Đảm bảo dấu phẩy thập phân
    h, m, s_ms = t_str.split(':')
    s, ms = s_ms.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

# Hàm phụ trợ chuyển đổi giây sang định dạng thời gian SRT
def seconds_to_srt_time(sec):
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
        if s == 60:
            m += 1
            s = 0
            if m == 60:
                h += 1
                m = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# Hàm phân tích chuỗi SRT thành danh sách phân đoạn
def parse_srt_string(srt_text):
    # regex tìm các block phụ đề SRT
    pattern = re.compile(
        r'(\d+)\s*\n'
        r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n'
        r'((?:[^\n]+\n*)+)', re.MULTILINE
    )
    
    segments = []
    for match in pattern.finditer(srt_text + "\n"):
        try:
            start_sec = srt_time_to_seconds(match.group(2))
            end_sec = srt_time_to_seconds(match.group(3))
            text = match.group(4).strip().replace('\n', ' ')
            segments.append({
                'start': start_sec,
                'end': end_sec,
                'text': text
            })
        except Exception:
            continue
    return segments

# Chuyển đổi danh sách phân đoạn thành chuỗi SRT
def segments_to_srt(segments):
    srt_lines = []
    for idx, seg in enumerate(segments, 1):
        start_str = seconds_to_srt_time(seg['start'])
        end_str = seconds_to_srt_time(seg['end'])
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start_str} --> {end_str}")
        srt_lines.append(seg['text'])
        srt_lines.append("")
    return "\n".join(srt_lines)

# 1. Trích xuất phụ đề bằng Whisper Local
def transcribe_local_whisper(audio_path, model_name="base", progress_callback=None):
    if progress_callback:
        progress_callback("Đang tải mô hình Whisper (Lần đầu có thể mất vài phút)...")
    model = whisper.load_model(model_name)
    
    if progress_callback:
        progress_callback("Đang nhận dạng giọng nói từ âm thanh video...")
    transcribe_kwargs = {
        "verbose": False,
        "task": "transcribe",
        "temperature": (0.0, 0.2, 0.4),
        "beam_size": 5,
        "best_of": 5,
        "condition_on_previous_text": True,
        "fp16": False,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
    }
    try:
        result = model.transcribe(audio_path, **transcribe_kwargs)
    except TypeError:
        safe_kwargs = {k: v for k, v in transcribe_kwargs.items() if k in ("verbose", "task", "fp16")}
        result = model.transcribe(audio_path, **safe_kwargs)
    
    segments = []
    for seg in result.get('segments', []):
        segments.append({
            'start': seg['start'],
            'end': seg['end'],
            'text': seg['text'].strip()
        })
    return segments

# 2. Trích xuất phụ đề bằng Gemini API (Nếu có key)
def transcribe_gemini(audio_path, api_key, progress_callback=None):
    if progress_callback:
        progress_callback("Đang kết nối Gemini API...")
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    if progress_callback:
        progress_callback("Đang tải âm thanh lên máy chủ Gemini...")
    audio_file = genai.upload_file(path=audio_path)
    
    if progress_callback:
        progress_callback("Gemini đang nhận diện giọng nói và căn chỉnh mốc thời gian...")
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "Hãy nghe file âm thanh này và tạo phụ đề chính xác bằng ngôn ngữ gốc nói trong file âm thanh. "
        "Xuất kết quả duy nhất ở định dạng chuẩn phụ đề SRT. "
        "Không thêm bất kỳ giải thích nào khác ngoài nội dung mã SRT."
    )
    response = model.generate_content([audio_file, prompt])
    
    # Xoá tệp tạm trên Cloud
    try:
        genai.delete_file(name=audio_file.name)
    except Exception:
        pass
        
    srt_content = response.text.strip()
    if srt_content.startswith("```"):
        lines = srt_content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        srt_content = "\n".join(lines).strip()
        
    return parse_srt_string(srt_content)

# 3. Trích xuất phụ đề bằng quét vùng chữ cứng OCR trên Video
def merge_bboxes(box1, box2):
    if not box1: return box2
    if not box2: return box1
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[0] + box1[2], box2[0] + box2[2])
    y2 = max(box1[1] + box1[3], box2[1] + box2[3])
    return [x1, y1, x2 - x1, y2 - y1]

def clean_text(text):
    return " ".join(text.strip().split()).lower()

def is_similar(text1, text2, threshold=0.78):
    t1 = clean_text(text1)
    t2 = clean_text(text2)
    if not t1 and not t2:
        return True
    if not t1 or not t2:
        return False
    # So khớp độ tương đồng văn bản
    ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
    return ratio >= threshold

def clean_cjk_spaces(text):
    # CJK Unified Ideographs: \u4e00-\u9fff
    # CJK Symbols and Punctuation: \u3000-\u303f
    # Halfwidth and Fullwidth Forms (for CJK punctuation): \uff00-\uffef
    cjk_char = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]'
    
    # 1. Chữ CJK - Khoảng trắng - Chữ CJK
    text = re.sub(rf'({cjk_char})\s+({cjk_char})', r'\1\2', text)
    text = re.sub(rf'({cjk_char})\s+({cjk_char})', r'\1\2', text) # Chạy lại lần 2 cho chuỗi dài
    
    # 2. Chữ CJK - Khoảng trắng - Dấu câu tiếng Anh
    text = re.sub(rf'({cjk_char})\s+([,.\!?])', r'\1\2', text)
    # 3. Dấu câu tiếng Anh - Khoảng trắng - Chữ CJK
    text = re.sub(rf'([,.\!?])\s+({cjk_char})', r'\1\2', text)
    
    return text

def merge_overlapping_strings(s1, s2):
    s1 = s1.strip()
    s2 = s2.strip()
    if not s1:
        return s2
    if not s2:
        return s1
        
    # Check if one is a substring of the other
    if s2 in s1:
        return s1
    if s1 in s2:
        return s2
        
    # Word level exact overlap (for languages with spaces like Vietnamese, English)
    words1 = s1.split()
    words2 = s2.split()
    max_word_overlap = 0
    for i in range(min(len(words1), len(words2)), 0, -1):
        if words1[-i:] == words2[:i]:
            max_word_overlap = i
            break
            
    if max_word_overlap > 0:
        return " ".join(words1[:-max_word_overlap] + words2)
        
    # Character level exact overlap (for Chinese/Japanese/Korean without spaces)
    max_char_overlap = 0
    for i in range(min(len(s1), len(s2)), 0, -1):
        if s1[-i:] == s2[:i]:
            max_char_overlap = i
            break
            
    if max_char_overlap > 0:
        return s1 + s2[max_char_overlap:]
        
    # FUZZY OVERLAP (to handle OCR noise like "hậu kì" vs "hậu kỉ")
    best_fuzzy_overlap = 0
    best_ratio = 0.0
    min_words = 2
    for i in range(min(len(words1), len(words2)), min_words - 1, -1):
        suffix = " ".join(words1[-i:])
        prefix = " ".join(words2[:i])
        ratio = difflib.SequenceMatcher(None, suffix.lower(), prefix.lower()).ratio()
        if ratio > 0.80 and ratio > best_ratio:
            best_ratio = ratio
            best_fuzzy_overlap = i
            
    if best_fuzzy_overlap > 0:
        return " ".join(words1[:-best_fuzzy_overlap] + words2)
        
    # Character level fuzzy overlap (for CJK)
    best_char_fuzzy_overlap = 0
    best_char_ratio = 0.0
    min_chars = 3
    for i in range(min(len(s1), len(s2)), min_chars - 1, -1):
        suffix = s1[-i:]
        prefix = s2[:i]
        ratio = difflib.SequenceMatcher(None, suffix.lower(), prefix.lower()).ratio()
        if ratio > 0.85 and ratio > best_char_ratio:
            best_char_ratio = ratio
            best_char_fuzzy_overlap = i
            
    if best_char_fuzzy_overlap > 0:
        return s1 + s2[best_char_fuzzy_overlap:]
        
    # No overlap, concatenate
    cjk_char = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]'
    if re.search(cjk_char, s1) or re.search(cjk_char, s2):
        return s1 + s2
    return s1 + " " + s2

def sort_ocr_results(results):
    if not results:
        return []
    
    boxes_with_info = []
    for r in results:
        pts, text, conf = r
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        height = max_y - min_y
        boxes_with_info.append({
            'original': r,
            'center_x': center_x,
            'center_y': center_y,
            'height': height,
            'min_x': min_x,
            'min_y': min_y
        })
        
    # Sắp xếp theo center_y trước để chia dòng
    boxes_with_info.sort(key=lambda b: b['center_y'])
    
    lines = []
    for b in boxes_with_info:
        if not lines:
            lines.append([b])
        else:
            last_line = lines[-1]
            avg_height = sum(item['height'] for item in last_line) / len(last_line)
            # Nếu chênh lệch dòng nhỏ hơn 60% chiều cao trung bình, coi như cùng dòng
            if abs(b['center_y'] - last_line[0]['center_y']) < 0.6 * avg_height:
                last_line.append(b)
            else:
                lines.append([b])
                
    sorted_results = []
    for line in lines:
        # Sắp xếp mỗi dòng từ trái sang phải
        line.sort(key=lambda b: b['center_x'])
        for b in line:
            sorted_results.append(b['original'])
            
    return sorted_results

def filter_ocr_noise(results):
    if not results:
        return []
    filtered = []
    for r in results:
        pts, text, conf = r
        text_stripped = text.strip()
        if not text_stripped:
            continue
        # Bỏ qua nếu độ tin cậy quá thấp (< 25%)
        if conf < 0.15:
            continue
        # Bỏ qua nếu chỉ là 1 dấu câu đơn lẻ bị phát hiện với độ tin cậy dưới 45%
        if len(text_stripped) == 1 and text_stripped in ",.!?'-_~`\"^:;“”‘’'`，。！？-、" and conf < 0.45:
            continue
        filtered.append(r)
    return filtered

def build_ocr_preprocess_variants(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped.copy()
    variants = [("color_original", cropped, 1.0, 1.0)]
    for scale in (2.0, 3.0, 4.0):
        resized_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append((f"gray_x{scale:g}", resized_gray, scale, scale))
        denoised = cv2.bilateralFilter(resized_gray, 5, 50, 50)
        variants.append((f"bilateral_x{scale:g}", denoised, scale, scale))
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
        variants.append((f"clahe_x{scale:g}", contrast, scale, scale))
        blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
        sharpened = cv2.addWeighted(contrast, 1.6, blur, -0.6, 0)
        variants.append((f"sharpen_x{scale:g}", sharpened, scale, scale))
        _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((f"otsu_x{scale:g}", otsu, scale, scale))
        variants.append((f"otsu_inv_x{scale:g}", cv2.bitwise_not(otsu), scale, scale))
        adaptive = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 9
        )
        variants.append((f"adaptive_x{scale:g}", adaptive, scale, scale))
        variants.append((f"adaptive_inv_x{scale:g}", cv2.bitwise_not(adaptive), scale, scale))
        kernel = np.ones((2, 2), np.uint8)
        close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
        variants.append((f"close_x{scale:g}", close, scale, scale))
    return variants

def normalize_ocr_results(results, scale_x=1.0, scale_y=1.0):
    normalized = []
    for pts, text, conf in results or []:
        new_pts = [[float(pt[0]) / scale_x, float(pt[1]) / scale_y] for pt in pts]
        normalized.append((new_pts, text, conf))
    return normalized

def score_ocr_text(text_val, results_list):
    if not text_val:
        return -1.0
    clean = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', '', text_val)
    if not clean.strip():
        return 0.0
    cjk_count = sum(1 for ch in clean if '\u4e00' <= ch <= '\u9fff')
    avg_conf = sum(r[2] for r in results_list) / len(results_list) if results_list else 0.0
    return len(clean) + cjk_count * 1.5 + avg_conf * 12.0

def run_easyocr_variants(cropped, reader, case_id="ocr"):
    best = {"score": -1.0, "results": [], "text": "", "variant": "", "raw_count": 0}
    save_ocr_debug_image(case_id, "crop_original", cropped)
    for name, image, scale_x, scale_y in build_ocr_preprocess_variants(cropped):
        save_ocr_debug_image(case_id, name, image)
        try:
            raw_results = reader.readtext(
                image, detail=1, paragraph=False, decoder='beamsearch', beamWidth=5,
                text_threshold=0.2, low_text=0.15, link_threshold=0.2, mag_ratio=1.0
            )
        except TypeError:
            raw_results = reader.readtext(image, detail=1, paragraph=False)
        except Exception as exc:
            write_ocr_debug(case_id, f"{name}: OCR error: {exc}")
            continue
        normalized = normalize_ocr_results(raw_results, scale_x, scale_y)
        filtered = filter_ocr_noise(normalized)
        sorted_results = sort_ocr_results(filtered)
        text = clean_cjk_spaces(" ".join([r[1] for r in sorted_results]).strip())
        score = score_ocr_text(text, sorted_results)
        write_ocr_debug(case_id, f"{name}: raw={len(raw_results)} filtered={len(filtered)} score={score:.2f} text={text!r}")
        if score > best["score"]:
            best = {"score": score, "results": sorted_results, "text": text, "variant": name, "raw_count": len(raw_results)}
    return best

def ocr_on_bbox(frame, bbox, reader, force_horizontal=False):
    """
    Chạy OCR trên vùng bbox của frame.
    Tự động hỗ trợ quét chữ dọc (Vertical Text) bằng cách chia nhỏ bbox.
    Trả về: (results, text_summary)
    Trong đó results là list của (pts, text, conf) nhưng toạ độ pts tính từ góc trái-trên của bbox gốc.
    """
    x, y, w, h = bbox
    frame_h, frame_w, _ = frame.shape
    
    # 1. Hỗ trợ quét chữ dọc (Vertical Text)
    if not force_horizontal and h > 1.25 * w and w > 10:
        N = int(round(h / w))
        if N < 2:
            N = 2
        slice_h = h // N
        
        combined_results = []
        combined_text_parts = []
        
        for i in range(N):
            slice_y = y + i * slice_h
            slice_bbox = [x, slice_y, w, slice_h]
            # Gọi đệ quy ocr_on_bbox cho phân đoạn nhỏ
            slice_res, slice_text = ocr_on_bbox(frame, slice_bbox, reader)
            
            # Cập nhật toạ độ pts của slice_res về toạ độ của bbox gốc
            for r in slice_res:
                pts, r_text, conf = r
                new_pts = [[pt[0], pt[1] + i * slice_h] for pt in pts]
                combined_results.append((new_pts, r_text, conf))
                
            if slice_text:
                combined_text_parts.append(slice_text)
                
        text_summary = "".join(combined_text_parts) # ghép sát nhau vì là chữ dọc CJK
        
        # So sánh kết quả quét cắt nhỏ dọc với kết quả quét ngang toàn bộ vùng chọn
        res_horiz, text_horiz = ocr_on_bbox(frame, bbox, reader, force_horizontal=True)
        if len(text_horiz.strip()) > len(text_summary.strip()):
            return res_horiz, text_horiz
        elif text_summary.strip():
            return combined_results, text_summary

    # 2. Quét chữ ngang thông thường
    x1 = max(0, min(x, frame_w))
    y1 = max(0, min(y, frame_h))
    x2 = max(0, min(x + w, frame_w))
    y2 = max(0, min(y + h, frame_h))
    
    if x2 <= x1 or y2 <= y1:
        return [], ""
        
    cropped = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    ocr_kwargs = {
        "detail": 1,
        "paragraph": False,
        "decoder": "beamsearch",
        "beamWidth": 5,
        "text_threshold": 0.2,
        "low_text": 0.15,
        "link_threshold": 0.2
    }
    
    def safe_readtext(img):
        try:
            return reader.readtext(img, **ocr_kwargs)
        except TypeError:
            return reader.readtext(img, detail=1, paragraph=False)
            
    def score_result(text_val, results_list):
        if not text_val:
            return -1.0
        import re
        clean = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text_val)
        if not clean.strip():
            return 0.0
        avg_conf = sum(r[2] for r in results_list) / len(results_list) if results_list else 0.0
        return len(clean) + avg_conf * 10.0

    # --- PHƯƠNG PHÁP 1: Ảnh xám phóng to 2.5x ---
    results_a = safe_readtext(resized)
    results_a_filtered = filter_ocr_noise(results_a)
    results_a_sorted = sort_ocr_results(results_a_filtered)
    text_a = " ".join([r[1] for r in results_a_sorted]).strip()
    text_a = clean_cjk_spaces(text_a)
    
    # Nếu kết quả tốt và độ tin cậy khá, trả về luôn để tiết kiệm tài nguyên CPU
    if text_a:
        avg_conf_a = sum(r[2] for r in results_a_sorted) / len(results_a_sorted) if results_a_sorted else 0.0
        if avg_conf_a > 0.40 or len(text_a) >= 3:
            if OCR_DEBUG:
                print(f"[DEBUG] OCR early success with Method A: '{text_a}'", flush=True)
            return results_a_sorted, text_a
            
    # --- PHƯƠNG PHÁP 2: Ảnh màu gốc không phóng to ---
    results_b_raw = safe_readtext(cropped)
    results_b = []
    for pts, r_text, conf in results_b_raw:
        scaled_pts = [[pt[0] * 2.5, pt[1] * 2.5] for pt in pts]
        results_b.append((scaled_pts, r_text, conf))
    results_b_filtered = filter_ocr_noise(results_b)
    results_b_sorted = sort_ocr_results(results_b_filtered)
    text_b = " ".join([r[1] for r in results_b_sorted]).strip()
    text_b = clean_cjk_spaces(text_b)
    
    score_a = score_result(text_a, results_a_sorted)
    score_b = score_result(text_b, results_b_sorted)
    
    if text_a or text_b:
        if score_a >= score_b and text_a:
            return results_a_sorted, text_a
        elif text_b:
            return results_b_sorted, text_b

    # --- PHƯƠNG PHÁP 3: Chỉ thử nhị phân hoá thích ứng khi 2 cách trên đều thất bại ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results_c = safe_readtext(thresh)
    results_c_filtered = filter_ocr_noise(results_c)
    results_c_sorted = sort_ocr_results(results_c_filtered)
    text_c = " ".join([r[1] for r in results_c_sorted]).strip()
    text_c = clean_cjk_spaces(text_c)
    
    thresh_inv = cv2.bitwise_not(thresh)
    results_d = safe_readtext(thresh_inv)
    results_d_filtered = filter_ocr_noise(results_d)
    results_d_sorted = sort_ocr_results(results_d_filtered)
    text_d = " ".join([r[1] for r in results_d_sorted]).strip()
    text_d = clean_cjk_spaces(text_d)
    
    score_c = score_result(text_c, results_c_sorted)
    score_d = score_result(text_d, results_d_sorted)
    
    if score_c >= score_d and text_c:
        return results_c_sorted, text_c
    elif text_d:
        return results_d_sorted, text_d
        
    return [], ""

def run_hardsub_ocr(video_path, bbox, progress_callback=None, ocr_lang="Tự động (Trung, Việt, Anh)"):
    """
    bbox là list [x, y, w, h] toạ độ vùng quét chữ (tính theo pixel gốc của video).
    Quét video với chu kỳ 0.5s để lấy các khung hình, cắt và nhận dạng OCR.
    """
    # Phân tích ngôn ngữ cho EasyOCR
    lang_list = ['vi', 'en'] # Mặc định
    if "Tự động" in ocr_lang or "auto" in ocr_lang:
        lang_list = ['ch_sim', 'en']
    elif "Trung Giản Thể" in ocr_lang or "ch_sim" in ocr_lang:
        lang_list = ['ch_sim', 'en']
    elif "Trung Phồn Thể" in ocr_lang or "ch_tra" in ocr_lang:
        lang_list = ['ch_tra', 'en']
    elif "Việt" in ocr_lang or "vi" in ocr_lang:
        lang_list = ['vi', 'en']
    elif "Anh" in ocr_lang or "en" in ocr_lang:
        lang_list = ['en']
    elif "Nhật" in ocr_lang or "ja" in ocr_lang:
        lang_list = ['ja', 'en']
    elif "Hàn" in ocr_lang or "ko" in ocr_lang:
        lang_list = ['ko', 'en']
    elif "Pháp" in ocr_lang or "fr" in ocr_lang:
        lang_list = ['fr', 'en']
    elif "Đức" in ocr_lang or "de" in ocr_lang:
        lang_list = ['de', 'en']
    elif "Tây Ban Nha" in ocr_lang or "es" in ocr_lang:
        lang_list = ['es', 'en']
    elif "Nga" in ocr_lang or "ru" in ocr_lang:
        lang_list = ['ru', 'en']
    elif "Thái" in ocr_lang or "th" in ocr_lang:
        lang_list = ['th', 'en']

    if progress_callback:
        progress_callback(f"Đang khởi tạo EasyOCR với ngôn ngữ {lang_list} (Sử dụng cache)...")
        
    reader = get_easyocr_reader(lang_list)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Không thể mở tệp tin video.")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError("Khong doc duoc FPS hoac so frame cua video.")
    
    subtitles = []
    x, y, w, h = bbox
    
    # Lay mau day hon de bat kip subtitle ngan.
    sample_seconds = 0.25
    miss_tolerance_seconds = 1.2
    sample_interval_frames = int(fps * sample_seconds)
    if sample_interval_frames < 1:
        sample_interval_frames = 1
        
    current_subtitle = None
    last_seen_text_time = None
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp_s = frame_idx / fps
        frame_h, frame_w, _ = frame.shape
        
        # Đảm bảo toạ độ cắt nằm trong kích thước video
        x1 = max(0, min(x, frame_w))
        y1 = max(0, min(y, frame_h))
        x2 = max(0, min(x + w, frame_w))
        y2 = max(0, min(y + h, frame_h))
        
        text = ""
        frame_bbox = None
        if x2 > x1 and y2 > y1:
            # --- TIỀN XỬ LÝ VÀ CHẠY OCR TỰ ĐỘNG PHÂN CHIA DỌC NẾU CÓ ---
            results, text = ocr_on_bbox(frame, [x, y, w, h], reader)
            
            if text and results:
                x_coords = []
                y_coords = []
                for (box, r_text, conf) in results:
                    for pt in box:
                        x_coords.append(pt[0])
                        y_coords.append(pt[1])
                if x_coords and y_coords:
                    scale = 2.5
                    rx_min = min(x_coords) / scale + x1
                    ry_min = min(y_coords) / scale + y1
                    rx_max = max(x_coords) / scale + x1
                    ry_max = max(y_coords) / scale + y1
                    frame_bbox = [int(rx_min), int(ry_min), int(rx_max - rx_min), int(ry_max - ry_min)]
            
        if progress_callback:
            percent = int((frame_idx / total_frames) * 100)
            progress_callback(f"Đang quét hình ảnh video... {percent}% (Giây thứ {int(timestamp_s)}s)")
            
        # Bộ máy trạng thái (State Machine) gom nhóm text giống nhau
        if text:
            if current_subtitle is None:
                current_subtitle = {
                    'start': timestamp_s,
                    'end': min(timestamp_s + sample_seconds, total_frames / fps),
                    'text': text,
                    'bbox': frame_bbox
                }
                last_seen_text_time = timestamp_s
            else:
                if is_similar(current_subtitle['text'], text):
                    current_subtitle['end'] = min(timestamp_s + sample_seconds, total_frames / fps)
                    if len(clean_text(text)) > len(clean_text(current_subtitle['text'])):
                        current_subtitle['text'] = text
                    current_subtitle['bbox'] = merge_bboxes(current_subtitle.get('bbox'), frame_bbox)
                    last_seen_text_time = timestamp_s
                else:
                    subtitles.append(current_subtitle)
                    current_subtitle = {
                        'start': timestamp_s,
                        'end': min(timestamp_s + sample_seconds, total_frames / fps),
                        'text': text,
                        'bbox': frame_bbox
                    }
                    last_seen_text_time = timestamp_s
        else:
            if current_subtitle is not None:
                if last_seen_text_time is not None and (timestamp_s - last_seen_text_time) <= miss_tolerance_seconds:
                    current_subtitle['end'] = min(timestamp_s + sample_seconds, total_frames / fps)
                else:
                    subtitles.append(current_subtitle)
                    current_subtitle = None
                    last_seen_text_time = None
                
        # Bỏ qua các frame tiếp theo bằng cap.grab() để tối ưu tốc độ mà không bị lỗi seek
        for _ in range(sample_interval_frames - 1):
            if not cap.grab():
                break
            frame_idx += 1
            
        frame_idx += 1
        if frame_idx >= total_frames:
            break
            
    if current_subtitle is not None:
        subtitles.append(current_subtitle)
        
    cap.release()
    
    # Hậu xử lý: Loại bỏ các phân đoạn rác hoặc quá ngắn
    cleaned_subs = []
    for sub in subtitles:
        sub['text'] = sub['text'].strip()
        cjk_count = sum(1 for c in sub['text'] if '\u4e00' <= c <= '\u9fff')
        min_duration = 0.18 if cjk_count >= 2 or len(sub['text']) >= 4 else 0.3
        if sub['text'] and (sub['end'] - sub['start']) >= min_duration:
            cleaned_subs.append(sub)
            
    return cleaned_subs

# 4. Trích xuất/phát hiện phụ đề gốc định hướng theo phân đoạn (Segment-Guided)
def run_segment_guided_ocr(video_path, segments, progress_callback=None, ocr_lang="auto", restrict_region=True):
    import numpy as np
    import torch
    import easyocr
    
    # Phân tích ngôn ngữ cho EasyOCR
    lang_list = ['vi', 'en']
    if "Tự động" in ocr_lang or "auto" in ocr_lang:
        lang_list = ['ch_sim', 'en']
    elif "Trung Giản Thể" in ocr_lang or "ch_sim" in ocr_lang:
        lang_list = ['ch_sim', 'en']
    elif "Trung Phồn Thể" in ocr_lang or "ch_tra" in ocr_lang:
        lang_list = ['ch_tra', 'en']
    elif "Việt" in ocr_lang or "vi" in ocr_lang:
        lang_list = ['vi', 'en']
    elif "Anh" in ocr_lang or "en" in ocr_lang:
        lang_list = ['en']
    elif "Nhật" in ocr_lang or "ja" in ocr_lang:
        lang_list = ['ja', 'en']
    elif "Hàn" in ocr_lang or "ko" in ocr_lang:
        lang_list = ['ko', 'en']
    elif "Pháp" in ocr_lang or "fr" in ocr_lang:
        lang_list = ['fr', 'en']
    elif "Đức" in ocr_lang or "de" in ocr_lang:
        lang_list = ['de', 'en']
    elif "Tây Ban Nha" in ocr_lang or "es" in ocr_lang:
        lang_list = ['es', 'en']
    elif "Nga" in ocr_lang or "ru" in ocr_lang:
        lang_list = ['ru', 'en']
    elif "Thái" in ocr_lang or "th" in ocr_lang:
        lang_list = ['th', 'en']

    if progress_callback:
        gpu_status = "GPU" if torch.cuda.is_available() else "CPU"
        progress_callback(f"Khởi tạo EasyOCR ({gpu_status}) với ngôn ngữ {lang_list} (Sử dụng cache)...")
        
    reader = get_easyocr_reader(lang_list)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Không thể mở tệp tin video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if total_frames <= 0 or fps <= 0:
        cap.release()
        raise ValueError("Không đọc được thuộc tính FPS hoặc số khung hình của video.")

    # 1. Gộp các segment liền kề (gap < 1.0s) để giảm số lần gọi OCR
    groups = []
    current_group = []
    for idx, seg in enumerate(segments):
        if not current_group:
            current_group.append(idx)
        else:
            prev_idx = current_group[-1]
            prev_seg = segments[prev_idx]
            if seg['start'] - prev_seg['end'] < 1.0:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
    if current_group:
        groups.append(current_group)
        
    if progress_callback:
        progress_callback(f"Tổng phân đoạn: {len(segments)}. Sau khi gộp liền kề (<1s): {len(groups)} nhóm quét.")
        
    # Hàm hỗ trợ chụp frame và chạy OCR tại thời điểm t
    def scan_frame_at_time(t):
        frame_idx = int(t * fps)
        if frame_idx >= total_frames:
            frame_idx = total_frames - 1
        if frame_idx < 0:
            frame_idx = 0
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None, 0.0
            
        # Kích thước dải quét dọc:
        top_h = int(height * 0.20)
        bottom_y1 = int(height * 0.65)
        
        # Căn giữa 60% chiều ngang: trục X từ 20% đến 80%
        x1 = int(width * 0.20) if restrict_region else 0
        x2 = int(width * 0.80) if restrict_region else width
        
        # Xếp dọc (Stack) dải trên và dưới để quét 1 lần duy nhất
        if restrict_region:
            crop_top = frame[0:top_h, x1:x2]
            crop_bottom = frame[bottom_y1:height, x1:x2]
            stacked = np.vstack([crop_top, crop_bottom])
        else:
            stacked = frame
            
        # Tiền xử lý ảnh
        gray = cv2.cvtColor(stacked, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        
        results = reader.readtext(resized, detail=1)
        if not results:
            return None, 0.0
            
        best_box = None
        best_conf = 0.0
        
        # pts là toạ độ box trên ảnh phóng to: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        for (pts, text, conf) in results:
            if len(text.strip()) >= 1 and conf > best_conf:
                best_conf = conf
                xs = [pt[0] for pt in pts]
                ys = [pt[1] for pt in pts]
                bx = min(xs) / 2.0
                by = min(ys) / 2.0
                bw = (max(xs) - min(xs)) / 2.0
                bh = (max(ys) - min(ys)) / 2.0
                
                # Ánh xạ tọa độ về video gốc
                if restrict_region:
                    # Xem box nằm ở dải trên hay dưới trong ảnh stacked
                    if by + bh/2.0 < top_h:
                        orig_x = int(bx + x1)
                        orig_y = int(by)
                        orig_w = int(bw)
                        orig_h = int(bh)
                    else:
                        orig_x = int(bx + x1)
                        orig_y = int((by - top_h) + bottom_y1)
                        orig_w = int(bw)
                        orig_h = int(bh)
                else:
                    orig_x = int(bx)
                    orig_y = int(by)
                    orig_w = int(bw)
                    orig_h = int(bh)
                    
                orig_x = max(0, min(orig_x, width))
                orig_y = max(0, min(orig_y, height))
                orig_w = max(1, min(orig_w, width - orig_x))
                orig_h = max(1, min(orig_h, height - orig_y))
                best_box = [orig_x, orig_y, orig_w, orig_h]
                
        return best_box, best_conf

    # 2. Chạy quét từng nhóm
    for g_idx, group in enumerate(groups):
        group_segs = [segments[idx] for idx in group]
        start_time = group_segs[0]['start']
        end_time = group_segs[-1]['end']
        
        # Quét lần 1 tại mốc chính giữa (midpoint)
        t_mid = (start_time + end_time) / 2.0
        best_box, best_conf = scan_frame_at_time(t_mid)
        best_t = t_mid
        
        # Cơ chế thử lại (Retry) tại mốc 20% và 80% nếu confidence < 30%
        if (best_box is None or best_conf < 0.3) and (end_time - start_time > 0.5):
            # Thử lại lần 1 tại mốc 20%
            t_retry1 = start_time + 0.2 * (end_time - start_time)
            box1, conf1 = scan_frame_at_time(t_retry1)
            if conf1 > best_conf:
                best_box, best_conf = box1, conf1
                best_t = t_retry1
                
            # Thử lại lần 2 tại mốc 80%
            if best_box is None or best_conf < 0.3:
                t_retry2 = start_time + 0.8 * (end_time - start_time)
                box2, conf2 = scan_frame_at_time(t_retry2)
                if conf2 > best_conf:
                    best_box, best_conf = box2, conf2
                    best_t = t_retry2
                    
        # Kiểm tra xem kích thước hộp che có bất thường không (nghi ngờ vị trí sub đổi giữa 2 câu)
        is_abnormal = False
        if best_box is not None:
            bx, by, bw, bh = best_box
            if bw > width * 0.9 or bh > 150:
                is_abnormal = True
                
        # Áp dụng toạ độ và độ tin cậy cho tất cả các segment trong nhóm
        for idx in group:
            segments[idx]['bbox'] = best_box
            segments[idx]['ocr_timestamp'] = best_t
            if is_abnormal:
                segments[idx]['confidence'] = 20 # Báo động đỏ (hộp che quá to hoặc bất thường)
            else:
                segments[idx]['confidence'] = int(best_conf * 100)
            
        if progress_callback:
            percent = int((g_idx + 1) / len(groups) * 100)
            progress_callback(f"Đang tìm phụ đề gốc: {percent}% (Nhóm {g_idx+1}/{len(groups)})")
            
    cap.release()
    return segments

def run_instant_ocr(frame, bbox, ocr_lang="Tự động (Trung, Việt, Anh)"):
    """
    Chạy OCR tức thì cho 1 frame đơn lẻ và 1 bbox duy nhất.
    """
    # Phân tích ngôn ngữ cho EasyOCR
    lang_list = ['vi', 'en']
    if "Tự động" in ocr_lang or "auto" in ocr_lang:
        lang_list = ['ch_sim', 'en']
    elif "Trung Giản Thể" in ocr_lang or "ch_sim" in ocr_lang:
        lang_list = ['ch_sim', 'en']
    elif "Trung Phồn Thể" in ocr_lang or "ch_tra" in ocr_lang:
        lang_list = ['ch_tra', 'en']
    elif "Việt" in ocr_lang or "vi" in ocr_lang:
        lang_list = ['vi', 'en']
    elif "Anh" in ocr_lang or "en" in ocr_lang:
        lang_list = ['en']
    elif "Nhật" in ocr_lang or "ja" in ocr_lang:
        lang_list = ['ja', 'en']
    elif "Hàn" in ocr_lang or "ko" in ocr_lang:
        lang_list = ['ko', 'en']
    elif "Pháp" in ocr_lang or "fr" in ocr_lang:
        lang_list = ['fr', 'en']
    elif "Đức" in ocr_lang or "de" in ocr_lang:
        lang_list = ['de', 'en']
    elif "Tây Ban Nha" in ocr_lang or "es" in ocr_lang:
        lang_list = ['es', 'en']
    elif "Nga" in ocr_lang or "ru" in ocr_lang:
        lang_list = ['ru', 'en']
    elif "Thái" in ocr_lang or "th" in ocr_lang:
        lang_list = ['th', 'en']

    reader = get_easyocr_reader(lang_list)
    
    frame_h, frame_w, _ = frame.shape
    x, y, w, h = bbox
    
    x1 = max(0, min(x, frame_w))
    y1 = max(0, min(y, frame_h))
    x2 = max(0, min(x + w, frame_w))
    y2 = max(0, min(y + h, frame_h))
    
    if OCR_DEBUG:
        print(f"[DEBUG] run_instant_ocr: bbox={bbox}, x1={x1}, y1={y1}, x2={x2}, y2={y2}, frame_shape={frame.shape}", flush=True)
    
    results, text_summary = ocr_on_bbox(frame, bbox, reader)
    text_results = []
    
    # Ánh xạ kết quả về toạ độ của crop (tính theo pixel gốc của crop)
    # kết quả: [[box, text, confidence], ...]
    for (pts, text, conf) in results:
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        bx = min(xs) / 2.5
        by = min(ys) / 2.5
        bw = (max(xs) - min(xs)) / 2.5
        bh = (max(ys) - min(ys)) / 2.5
        text_results.append({
            'box': [int(bx), int(by), int(bw), int(bh)],
            'text': clean_cjk_spaces(text),
            'confidence': int(conf * 100)
        })
        
    return text_results, text_summary

# =====================================================================
# BACKEND QUẢN LÝ KỊCH BẢN & TTS
# =====================================================================

import time

DATA_DIR = Path(__file__).resolve().parent / "Data"
CUSTOM_PROMPTS_FILE = DATA_DIR / "custom_prompts.json"
SCRIPT_HISTORY_FILE = DATA_DIR / "script_history.json"
HISTORY_LIMIT = 50

def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_json(path: Path, default):
    ensure_data_dir()
    if not path.exists():
        return default
    try:
        with path.open('r', encoding='utf-8-sig') as f:
            data = json.load(f)
            return data if data is not None else default
    except Exception as e:
        print(f"Error loading JSON from {path}: {e}")
        return default

def save_json(path: Path, data):
    ensure_data_dir()
    try:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(path)
    except Exception as e:
        print(f"Error saving JSON to {path}: {e}")

def clean_script(text: str) -> str:
    # Chuẩn hóa văn bản đầu vào để TTS đọc tự nhiên, không đọc lẫn chỉ dẫn kỹ thuật.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-").replace("…", "...")
    text = re.sub(r'(?m)^\s*(?:[-*+]|\d+[.)])\s+', '', text)
    text = text.replace("`", "").replace("*", "").replace("_", "")

    # Xóa các dòng ghi chú kỹ thuật, tiêu đề kịch bản
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        s_line = line.strip()
        if not s_line:
            continue
        lower_line = s_line.lower()
        # Bỏ dòng tiêu đề/chỉ dẫn thường gặp khi copy từ AI hoặc timeline dựng video.
        strip_prefixes = (
            "hook:", "voice over:", "voice-over:", "vo:", "lời đọc:", "loi doc:"
        )
        for prefix in strip_prefixes:
            if lower_line.startswith(prefix):
                s_line = s_line[len(prefix):].strip()
                lower_line = s_line.lower()
                break
        if not s_line:
            continue

        skip_prefixes = (
            "kịch bản:", "kich ban:", "tiêu đề:", "tieu de:", "caption:",
            "b-roll:", "visual:", "cảnh:", "canh:", "shot:", "sfx:", "hiệu ứng:",
            "hieu ung:", "nhạc nền:", "nhac nen:", "ghi chú:", "ghi chu:",
            "lưu ý:", "luu y:"
        )
        if lower_line.startswith(skip_prefixes):
            continue
        if "ghi chú kỹ thuật" in lower_line or "ghi chu ky thuat" in lower_line:
            continue
        raw_letters = re.sub(r'[^a-zA-ZÀ-Ỹ0-9\s]', '', s_line).strip()
        if raw_letters.isupper() and len(raw_letters) > 5:
            continue
        cleaned_lines.append(s_line)
    text = "\n".join(cleaned_lines)

    # Xóa chỉ dẫn sân khấu/timeline để file audio chỉ còn lời đọc.
    text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[-–]\s*\d{1,2}:\d{2}(?::\d{2})?)?', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'/pause\s*\d+\.?\d*s/', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])([^\s,.!?;:])', r'\1 \2', text)
    text = re.sub(r'([!?.,]){4,}', r'\1\1\1', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def _restore_sentence_tokens(text: str, abbreviations: list[str]) -> str:
    for abbr in abbreviations:
        text = text.replace(f'{abbr}_TEMP_DOT_', f'{abbr}.')
    return text.replace('_TEMP_DECIMAL_DOT_', '.')

def _split_long_voiceover_sentence(sentence: str, max_chars: int) -> list[str]:
    sentence = sentence.strip()
    if len(sentence) <= max_chars:
        return [sentence] if sentence else []

    chunks = []
    current = ""
    parts = re.split(r'([,;:]\s+|\s+-\s+|\n+)', sentence)
    for part in parts:
        if not part:
            continue
        candidate = f"{current}{part}".strip() if current else part.strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current.rstrip(" ,;:-"))
            current = part.strip()
        if len(current) > max_chars:
            words = current.split()
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = word
    if current:
        chunks.append(current.rstrip(" ,;:-"))
    return [c for c in chunks if c]

def _polish_voiceover_sentence(sentence: str) -> str:
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    sentence = sentence.strip(' "\'')
    if not sentence:
        return ""
    if sentence[-1] not in ".!?…。！？":
        sentence += "."
    return sentence[0].upper() + sentence[1:] if sentence else sentence

def split_script_to_sentences(script: str, max_chars: int = 155) -> list[dict]:
    if not script:
        return []
    
    text = clean_script(script)
    
    # 1. Danh sách từ viết tắt phổ biến (không được coi là kết thúc câu khi đi sau dấu chấm)
    abbreviations = [
        "TP", "Mr", "Mrs", "Ms", "Dr", "Prof", "ThS", "TS", "BS", "PST", 
        "Th.S", "T.S", "P.S", "Co", "Ltd", "St", "v.v", "p"
    ]
    
    modified_text = text
    # Dùng \b để chỉ khớp ranh giới từ (tránh khớp nhầm đuôi p của "đẹp")
    for abbr in abbreviations:
        modified_text = re.compile(rf'\b{abbr}\.', re.IGNORECASE).sub(f'{abbr}_TEMP_DOT_', modified_text)
        
    # 2. Ngăn tách ở số thập phân (Ví dụ: "3.5 triệu", "10.500đ")
    modified_text = re.sub(r'(\d)\.(\d)', r'\1_TEMP_DECIMAL_DOT_\2', modified_text)
    
    # 3. Tách câu theo dấu câu tiếng Việt/Anh: . ! ? … 。！？
    pattern = r'([.?!…。！？]+"?\s*)'
    parts = re.split(pattern, modified_text)
    
    raw_sentences = []
    current_sentence = ""
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            current_sentence = part
        else:
            full_sentence = (current_sentence + part).strip()
            # Khôi phục các từ viết tắt và số thập phân
            for abbr in abbreviations:
                full_sentence = full_sentence.replace(f'{abbr}_TEMP_DOT_', f'{abbr}.')
            full_sentence = full_sentence.replace('_TEMP_DECIMAL_DOT_', '.')
            
            # Chuẩn hóa khoảng trắng
            full_sentence = re.sub(r'\s+', ' ', full_sentence)
            if full_sentence:
                raw_sentences.append(full_sentence)
            current_sentence = ""
            
    if current_sentence.strip():
        last_s = current_sentence.strip()
        for abbr in abbreviations:
            last_s = last_s.replace(f'{abbr}_TEMP_DOT_', f'{abbr}.')
        last_s = last_s.replace('_TEMP_DECIMAL_DOT_', '.')
        last_s = re.sub(r'\s+', ' ', last_s)
        if last_s:
            raw_sentences.append(last_s)
            
    # Bỏ các dòng rỗng
    raw_sentences = [s for s in raw_sentences if s.strip()]
    
    # Nếu câu quá dài thì tách thành nhịp ngắn, dễ đọc như voice-over TikTok/Reels/Shorts.
    final_sentences = []
    for s in raw_sentences:
        final_sentences.extend(_split_long_voiceover_sentence(s, max_chars))

    final_sentences = [_polish_voiceover_sentence(s) for s in final_sentences]
    final_sentences = [s for s in final_sentences if s.strip()]
    
    # 5. Trả về list dict dạng yêu cầu
    result = []
    for idx, s in enumerate(final_sentences, 1):
        result.append({
            "index": idx,
            "text": s,
            "needs_regen": True,
            "audio_path": "",
            "duration": 0.0
        })
    return result

def format_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
        if s == 60:
            m += 1
            s = 0
            if m == 60:
                h += 1
                m = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def export_srt_with_silence(
    segments: list[dict],
    output_path: str | Path,
    default_duration: float = 2.5,
    silence_between: float = 0.35
) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    current_time = 0.0
    srt_lines = []
    
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
            
        start = current_time
        duration = seg.get("duration", 0.0)
        if duration <= 0:
            duration = default_duration
        end = start + duration
        
        start_srt = format_srt_time(start)
        end_srt = format_srt_time(end)
        
        idx = seg.get("index", 1)
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start_srt} --> {end_srt}")
        srt_lines.append(f"{text}\n")
        
        current_time = end + silence_between
        
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(srt_lines))
        
    return out_path

def load_custom_prompts() -> list[dict]:
    data = load_json(CUSTOM_PROMPTS_FILE, [])
    return data if isinstance(data, list) else []

def save_custom_prompts(prompts: list[dict]) -> None:
    save_json(CUSTOM_PROMPTS_FILE, prompts if isinstance(prompts, list) else [])

def add_custom_prompt(title: str, content: str, tags: list[str] | None = None) -> dict:
    prompts = load_custom_prompts()
    prompt_id = str(int(time.time() * 1000))
    now_iso = datetime.datetime.now().isoformat()
    new_prompt = {
        "id": prompt_id,
        "title": title,
        "content": content,
        "tags": tags if tags is not None else [],
        "created_at": now_iso,
        "updated_at": now_iso
    }
    prompts.append(new_prompt)
    save_custom_prompts(prompts)
    return new_prompt

def delete_custom_prompt(prompt_id: str) -> bool:
    prompts = load_custom_prompts()
    initial_len = len(prompts)
    prompts = [p for p in prompts if p.get("id") != prompt_id]
    if len(prompts) < initial_len:
        save_custom_prompts(prompts)
        return True
    return False

def load_script_history() -> list[dict]:
    data = load_json(SCRIPT_HISTORY_FILE, [])
    return data if isinstance(data, list) else []

def save_script_history(history: list[dict]) -> None:
    safe_history = history if isinstance(history, list) else []
    save_json(SCRIPT_HISTORY_FILE, safe_history[:HISTORY_LIMIT])

def add_script_history(title: str, script: str, segments_count: int = 0) -> dict:
    history = load_script_history()
    history_id = str(int(time.time() * 1000))
    now_iso = datetime.datetime.now().isoformat()
    new_entry = {
        "id": history_id,
        "title": title,
        "script": script,
        "segments_count": segments_count,
        "created_at": now_iso
    }
    history.insert(0, new_entry)
    if len(history) > HISTORY_LIMIT:
        history = history[:HISTORY_LIMIT]
    save_script_history(history)
    return new_entry
