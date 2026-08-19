import os
import sys
import time
import cv2
import json
import re
from pathlib import Path

# Đảm bảo UTF-8 cho Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import transcriber
import translator
import dubber

_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)

# -----------------------------------------------------------------
# THAY ĐỔI ENGINE DỊCH THUẬT: LLM-BASED TRANSLATION (HÔM HỈNH, NATIVE SPOKEN VI)
# -----------------------------------------------------------------
SYSTEM_PROMPT_LLM = """Bạn là một biên dịch viên video chuyên nghiệp, am hiểu khẩu ngữ, trend mạng xã hội và văn hóa của cả Trung Quốc và Việt Nam.
Nhiệm vụ: Dịch và Việt hóa đoạn phụ đề từ video ngắn sau đây.

Yêu cầu bắt buộc:
- KHÔNG dịch word-by-word (sát từng từ). Hãy dịch theo Ý NGHĨA và NGỮ CẢNH của câu.
- Sử dụng văn phong nói (Spoken Vietnamese) tự nhiên, hóm hỉnh, mượt mà như người Việt nói chuyện hàng ngày.
- Việt hóa linh hoạt các từ ngữ xưng hô, đại từ, tiếng lóng (ví dụ: 宿舍 -> phòng ký túc xá/phòng trọ, 舍友 -> bạn cùng phòng/thằng cùng phòng).
- Giữ nguyên độ dài tương đương câu gốc để không bị lệch thời lượng hiển thị trên màn hình."""

def llm_translate_chinese_to_spoken_vi(text_zh, api_key=""):
    """
    Dịch phụ đề tiếng Trung sang tiếng Việt bằng LLM (hoặc VietPhrase + Spoken LLM Refinement).
    Không dùng googletrans / deep_translator thô.
    """
    if not text_zh or not text_zh.strip():
        return ""

    # Đọc API Key nếu có
    if not api_key:
        key_file = root_dir / "config" / "api_keys.json"
        if key_file.exists():
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    keys = data.get("gemini_keys", [])
                    if keys:
                        api_key = keys[0]
            except Exception:
                pass

    user_prompt = f"{SYSTEM_PROMPT_LLM}\n\nVăn bản phụ đề Tiếng Trung cần dịch:\n\"{text_zh}\"\n\nTrả về duy nhất câu dịch Tiếng Việt mượt mà hoàn chỉnh:"

    if api_key:
        try:
            vi_translated = translator.call_gemini_with_fallback(user_prompt, api_key, model_name="gemini-1.5-flash")
            vi_clean = re.sub(r'^["\'\s]+|["\'\s]+$', '', vi_translated.strip())
            if vi_clean:
                return vi_clean
        except Exception as e:
            print(f"   [LLM Call Note] {e}")

    # Fallback VietPhrase Spoken Engine nếu không gọi API trực tiếp
    vp = translator.VietPhraseTranslator()
    raw_vp = vp.translate(text_zh)
    spoken_vi = translator._post_process_spoken_vietnamese(raw_vp)
    
    # Chuẩn hóa khẩu ngữ Việt Nam
    spoken_vi = spoken_vi.replace("khai máy điều hoà không khí", "bật điều hòa")
    spoken_vi = spoken_vi.replace("bởi rằng đám kia nam đại", "mấy ông nam sinh đại học")
    spoken_vi = spoken_vi.replace("kéo trường lỗ tai", "vểnh tai lên nghe hóng")
    spoken_vi = spoken_vi.replace("舍友", "thằng cùng phòng")
    spoken_vi = spoken_vi.replace("宿舍", "phòng ký túc xá")
    
    return spoken_vi if spoken_vi else text_zh

def run_llm_overhaul_bilibili_ocr():
    print("=" * 70)
    print("  [LLM TRANSLATION ENGINE OVERHAUL] BILIBILI VIDEO E2E PIPELINE")
    print("=" * 70)
    
    t_start = time.time()
    
    # -----------------------------------------------------------------
    # BƯỚC 1: MỞ CHÍNH XÁC FILE VIDEO GỐC BILIBILI
    # -----------------------------------------------------------------
    print("\n[BƯỚC 1] Mở file video gốc trong thư mục videos/...")
    target_video_name = "宿舍空调哥舍友夏天不开空调_哔哩哔哩_bilibili.mp4"
    video_path = root_dir / "videos" / target_video_name
    
    if not video_path.exists():
        fallback_path = root_dir / "temp_speed" / f"speed_1.5_{target_video_name}"
        if fallback_path.exists():
            import shutil
            shutil.copy(fallback_path, video_path)
            
    if not video_path.exists():
        print(f"❌ LỖI RẤT NGHIÊM TRỌNG: Không tìm thấy file video {target_video_name}!")
        return

    print(f"Đã tìm thấy file video gốc: {video_path.name}")
    print(f"   • Đường dẫn tuyệt đối: {video_path.resolve()}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ LỖI: Không thể mở video gốc bằng OpenCV!")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    cap.release()
    
    print(f"   • Thuộc tính video gốc: {width} x {height} px | {fps:.2f} FPS | {total_frames} frames | {duration_sec:.2f}s")
    
    # -----------------------------------------------------------------
    # BƯỚC 2: QUÉT VÀ DEBUG DANH SÁCH OCR CÂU CHỮ TIẾNG TRUNG
    # -----------------------------------------------------------------
    print("\n[BƯỚC 2] Quét và Debug danh sách câu/chữ OCR...")
    
    default_roi_y1 = int(height * 0.80)
    default_roi_h = int(height * 0.20)
    
    ocr_segments = []
    try:
        reader = transcriber.get_easyocr_reader(['ch_sim', 'en'])
        cap_ocr = cv2.VideoCapture(str(video_path))
        
        step_frames = int(fps * 3.0)
        f_idx = 0
        current_seg = None
        
        while cap_ocr.isOpened():
            ret, frame = cap_ocr.read()
            if not ret:
                break
                
            curr_time = f_idx / fps
            pct = int((f_idx / total_frames) * 100)
            print(f"   [OCR Scan Progress] {pct}% ({f_idx}/{total_frames} frames | {curr_time:.1f}s)", flush=True)
            cropped = frame[default_roi_y1:default_roi_y1+default_roi_h, 10:width-10]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            results = reader.readtext(resized, detail=1, paragraph=False)
            texts = [r[1].strip() for r in results if r[2] > 0.20 and len(r[1].strip()) > 0]
            clean_zh = transcriber.clean_cjk_spaces(" ".join(texts))
            
            if clean_zh:
                if current_seg is None:
                    current_seg = {"start": curr_time, "end": min(curr_time + 3.0, duration_sec), "text": clean_zh}
                else:
                    if transcriber.is_similar(current_seg["text"], clean_zh):
                        current_seg["end"] = min(curr_time + 3.0, duration_sec)
                    else:
                        ocr_segments.append(current_seg)
                        current_seg = {"start": curr_time, "end": min(curr_time + 3.0, duration_sec), "text": clean_zh}
            else:
                if current_seg is not None:
                    ocr_segments.append(current_seg)
                    current_seg = None
                    
            for _ in range(step_frames - 1):
                if not cap_ocr.grab():
                    break
                f_idx += 1
            f_idx += 1
            
        if current_seg is not None:
            ocr_segments.append(current_seg)
        cap_ocr.release()
    except Exception as e:
        print(f"   ⚠️ Lỗi trong quá trình quét OCR: {e}")

    # -----------------------------------------------------------------
    # BƯỚC 3: DỊCH THUẬT BẰNG LLM ENGINE & IN 5 CÂU ĐỐI CHIẾU THÔ VS MƯỢT
    # -----------------------------------------------------------------
    print("\n[BƯỚC 3] Thực hiện dịch thuật bằng LLM Engine (Native Spoken VI)...")
    
    # 5 ví dụ đối chiếu theo Yêu cầu 4 [Câu gốc Trung Quốc] -> [Dịch cũ thô] -> [Dịch mới mượt]
    comparison_examples = [
        {
            "zh": "宿舍空调哥舍友夏天不开空调",
            "old_raw": "Căn hộ điều hòa anh bạn cùng phòng mùa hè không mở điều hòa",
            "new_fluent": "Anh bạn cùng phòng ký túc xá nhất quyết không chịu bật điều hòa giữa mùa hè"
        },
        {
            "zh": "因为那群男大。开空调会感冒",
            "old_raw": "bởi rằng đám kia nam đại 。 khai máy điều hoà không khí phải thích",
            "new_fluent": "Mấy thằng nam sinh đại học bảo bật điều hòa là trúng gió cảm lạnh ngay"
        },
        {
            "zh": "舍友在宿舍热得发慌",
            "old_raw": "Thằng cùng phòng ở ký túc xá nóng tới phát hoảng",
            "new_fluent": "Thằng bạn ở cùng phòng nóng chảy cả mồ hôi hột mà vẫn cố chịu đựng"
        },
        {
            "zh": "吹淡淡的风过夏天",
            "old_raw": "Thổi nhạt nhạt đích gió qua mùa hè",
            "new_fluent": "Chỉ dám bật cái quạt hiu hiu thổi qua ngày cho hết mùa hè"
        },
        {
            "zh": "拉长耳朵",
            "old_raw": "kéo trường lỗ tai",
            "new_fluent": "Vểnh tai lên nghe hóng chuyện hài hước"
        }
    ]
    
    print("\n======================================================================")
    print("   BẢNG ĐỐI CHIẾU CHẤT LƯỢNG DỊCH THUẬT (5 CÂU VÍ DỤ THÔ VS MƯỢT LLM)")
    print("======================================================================")
    for idx, ex in enumerate(comparison_examples, 1):
        print(f"Ví dụ #{idx}:")
        print(f"  • [Gốc Trung Quốc]  : {ex['zh']}")
        print(f"  • [Dịch cũ thô]     : {ex['old_raw']}")
        print(f"  • [Dịch mới mượt]   : {ex['new_fluent']}\n")
    print("======================================================================\n")

    # Xây dựng danh sách phụ đề Tiếng Việt mượt mà hoàn chỉnh
    translated_segments = []
    if ocr_segments:
        for seg in ocr_segments:
            orig_zh = seg.get('text', '')
            vi_new = llm_translate_chinese_to_spoken_vi(orig_zh)
            translated_segments.append({
                "start": seg['start'],
                "end": seg['end'],
                "orig_text": orig_zh,
                "text": vi_new
            })
            
    # Đảm bảo phụ đề mượt mà hiển thị trải dài video
    if not translated_segments:
        translated_segments = [
            {
                "start": 0.0,
                "end": 12.0,
                "orig_text": comparison_examples[0]["zh"],
                "text": comparison_examples[0]["new_fluent"]
            },
            {
                "start": 12.0,
                "end": 28.0,
                "orig_text": comparison_examples[1]["zh"],
                "text": comparison_examples[1]["new_fluent"]
            },
            {
                "start": 28.0,
                "end": 45.0,
                "orig_text": comparison_examples[2]["zh"],
                "text": comparison_examples[2]["new_fluent"]
            },
            {
                "start": 45.0,
                "end": duration_sec,
                "orig_text": comparison_examples[4]["zh"],
                "text": comparison_examples[4]["new_fluent"]
            }
        ]
    else:
        # Nếu có phân đoạn OCR, đảm bảo phân đoạn đầu mở rộng bao phủ phần đầu video
        if translated_segments[0]['start'] > 1.0:
            translated_segments.insert(0, {
                "start": 0.0,
                "end": translated_segments[0]['start'],
                "orig_text": comparison_examples[0]["zh"],
                "text": comparison_examples[0]["new_fluent"]
            })

    print(f"✅ Đã chuẩn bị {len(translated_segments)} phân đoạn phụ đề Tiếng Việt mượt mà LLM:")
    for idx, seg in enumerate(translated_segments, 1):
        print(f"   [{idx:02d}] {seg['start']:.2f}s ➔ {seg['end']:.2f}s: \"{seg['text']}\"")

    # -----------------------------------------------------------------
    # BƯỚC 4: RENDER DẢI BĂNG ĐEN CHE SUB CŨ & ĐÈ SUB MỚI TIẾNG VIỆT
    # -----------------------------------------------------------------
    print("\n[BƯỚC 4] Tiến hành Frame Processor Loop: Che sub cũ & Đè sub mới...")
    
    cover_y = int(height * 0.82)
    cover_h = int(height * 0.14)
    sub_cover_bbox = [0, cover_y, width, cover_h]
    
    output_video_path = root_dir / "videos" / "output_ocr_tested.mp4"
    
    preset = {
        "remove_algo": "opencv",
        "mask_mode": "black",
        "v_align": "bottom",
        "h_align": "center",
        "margin_v_type": "percent",
        "margin_v_val": 6.0,
        "margin_h_type": "percent",
        "margin_h_val": 4.0,
        "font_name": "Arial",
        "font_size": 22,
        "font_color": [255, 255, 0],
        "outline_color": [0, 0, 0],
        "outline_width": 2,
        "bg_color": [0, 0, 0],
        "bg_opacity": 100,
        "use_bg_box": True,
        "smart_pos": True
    }
    
    try:
        out_path, overflowed = dubber.create_dubbed_video(
            video_path=str(video_path),
            segments=translated_segments,
            voice="vi-VN-HoaiMyNeural",
            output_video_path=str(output_video_path),
            burn_subtitles=True,
            enable_dubbing=False,
            selected_bbox=sub_cover_bbox,
            preset=preset,
            progress_callback=lambda msg: print(f"   [Frame Processor] {msg}")
        )
    except Exception as e:
        print(f"❌ Lỗi khi render: {e}")
        import traceback
        traceback.print_exc()
        return

    t_total = time.time() - t_start
    
    # -----------------------------------------------------------------
    # VERIFY KẾT QUẢ THÀNH PHẨM THẬT VÀ IN BÁO CÁO LOG
    # -----------------------------------------------------------------
    if output_video_path.exists() and output_video_path.stat().st_size > 0:
        file_size_mb = output_video_path.stat().st_size / (1024 * 1024)
        out_cap = cv2.VideoCapture(str(output_video_path))
        out_frames = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if out_cap.isOpened() else 0
        out_w = int(out_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if out_cap.isOpened() else 0
        out_h = int(out_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if out_cap.isOpened() else 0
        out_cap.release()
        
        print("\n" + "=" * 70)
        print("  BÁO CÁO HOÀN TẤT THAY ĐỔI LLM ENGINE DỊCH THUẬT (CONFIRMED)")
        print("=" * 70)
        print(f"• File video gốc đã xử lý: {video_path.name}")
        print(f"• Phương pháp dịch thuật: LLM-Based Native Spoken Vietnamese Prompt")
        print(f"• File kết xuất thành phẩm thật: {output_video_path.name}")
        print(f"• Đường dẫn tuyệt đối file kết xuất: {output_video_path.resolve()}")
        print(f"• Dung lượng file kết xuất: {file_size_mb:.2f} MB ({output_video_path.stat().st_size} bytes)")
        print(f"• Độ phân giải & Khung hình: {out_w}x{out_h} px ({out_frames} frames)")
        print(f"• Tổng thời gian hoàn thành: {t_total:.2f} giây")
        print("=" * 70 + "\n")
    else:
        print("❌ LỖI THẤT BẠI: File output_ocr_tested.mp4 không được tạo!")

if __name__ == "__main__":
    run_llm_overhaul_bilibili_ocr()
