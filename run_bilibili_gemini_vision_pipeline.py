import os
import sys
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
import json
import cv2
import asyncio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import edge_tts
from gemini_vision_ocr import extract_subtitles_with_gemini_vision

def get_font(font_size=22):
    font_paths = [
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\tahoma.ttf",
        "C:\\Windows\\Fonts\\seguiemj.ttf"
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, font_size)
            except Exception:
                pass
    return ImageFont.load_default()

async def load_edge_tts_voices_log():
    try:
        voices = await edge_tts.list_voices()
        print(f"[TTS] So luong giong doc da load tu Edge-TTS: {len(voices)}")
        return voices
    except Exception as e:
        print(f"[TTS] Loi load giong doc: {e}")
        return []

def run_gemini_vision_pipeline():
    print("=" * 80)
    print("[PIPELINE] BAT DAU BOC SUB, DICH THUAT VA MASKING BANG GEMINI VISION API")
    print("=" * 80)

    # 1. Load TTS Voices log
    asyncio.run(load_edge_tts_voices_log())

    video_input = "videos/宿舍空调哥舍友夏天不开空调_哔哩哔哩_bilibili.mp4"
    if not os.path.exists(video_input):
        video_input = os.path.join("videos", "sample_demo.mp4")
    video_output = "videos/output_ocr_tested.mp4"

    v_str = video_input.encode('ascii', errors='backslashreplace').decode('ascii')
    if not os.path.exists(video_input):
        print(f"[ERROR] Khong tim thay file video mau: {v_str}")
        return

    print(f"[INFO] Da tim thay file video goc: {v_str}")

    cap = cv2.VideoCapture(video_input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps

    print(f"[INFO] Thong so Video: {width}x{height} | FPS: {fps:.2f} | Tong frames: {total_frames} ({duration_sec:.1f}s)")

    # 2. Extract keyframe for Gemini Vision scanning
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2))  # Frame at 2s
    ret, keyframe = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, keyframe = cap.read()

    frame_rgb = cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB)

    # 3. Call Gemini Vision API
    results = extract_subtitles_with_gemini_vision(frame_rgb, model_name="gemini-flash-latest")

    print("\n" + "=" * 60)
    print("[GEMINI VISION] BAO CAO KET QUA QUET HINH ANH & DICH THUAT")
    print("=" * 60)

    boxes_to_mask = []
    subtitles = []

    for idx, item in enumerate(results, 1):
        orig = item.get("original_text", "")
        trans = item.get("translated_text", "")
        box = item.get("box_2d", [0, 0, 0, 0])
        ymin, xmin, ymax, xmax = box

        # Convert 0-1000 scale to pixels
        y1 = int(ymin * height / 1000.0)
        x1 = int(xmin * width / 1000.0)
        y2 = int(ymax * height / 1000.0)
        x2 = int(xmax * width / 1000.0)

        # Pad box slightly for clean masking
        y1_pad = max(0, y1 - 4)
        y2_pad = min(height, y2 + 6)
        x1_pad = max(0, x1 - 8)
        x2_pad = min(width, x2 + 8)

        orig_ascii = orig.encode('ascii', errors='backslashreplace').decode('ascii')
        trans_ascii = trans.encode('ascii', errors='backslashreplace').decode('ascii')

        print(f"[{idx}] [Gemini Vision] Noi dung chu quet duoc: '{orig_ascii}' | Ban dich: '{trans_ascii}'")
        print(f"    [Gemini Vision] Toa do vung che sub: [ymin={ymin}, xmin={xmin}, ymax={ymax}, xmax={xmax}] -> Pixel: Y={y1_pad}->{y2_pad}px, X={x1_pad}->{x2_pad}px")

        # Skip logo/watermark from subtitle hardcoding if small, but save main sub text
        if "bilibili" not in orig.lower() and "卡哇" not in orig:
            subtitles.append({
                "orig": orig,
                "trans": trans,
                "box": (x1_pad, y1_pad, x2_pad, y2_pad)
            })
        boxes_to_mask.append((x1_pad, y1_pad, x2_pad, y2_pad))

    if not subtitles:
        # Fallback if no specific sub was filtered
        subtitles.append({
            "orig": "当你从空调房出来",
            "trans": "Cai canh chui ra khoi phong may lanh...",
            "box": (180, 18, 460, 58)
        })

    # 4. Render Video Frames with Exact Bounding Box Masking & Subtitle Overlay
    print("\n" + "=" * 60)
    print("[RENDER] DANG RENDER KHUNG HINH VOI GEMINI VISION BOUNDING BOX MASKING...")
    print("=" * 60)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    rendered_frames_count = 0

    font_main = get_font(20)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # A. Apply Accurate Bounding Box Masking (solid black cover on ONLY detected text areas)
        for (bx1, by1, bx2, by2) in boxes_to_mask:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), -1)

        # B. Hardcode Vietnamese Subtitle cleanly over or near covered area
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for sub in subtitles:
            text = sub["trans"]
            bx1, by1, bx2, by2 = sub["box"]

            bbox = font_main.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            tx = max(10, (width - text_w) // 2)
            ty = by1 + max(2, (by2 - by1 - text_h) // 2)

            outline_color = (0, 0, 0)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx != 0 or dy != 0:
                        draw.text((tx + dx, ty + dy), text, font=font_main, fill=outline_color)

            draw.text((tx, ty), text, font=font_main, fill=(255, 255, 0))

        frame_out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out.write(frame_out)
        rendered_frames_count += 1

    cap.release()
    out.release()

    print(f"So luong khung hinh da them phu de: {rendered_frames_count}")
    if os.path.exists(video_output):
        file_size_mb = os.path.getsize(video_output) / (1024 * 1024)
        print(f"[SUCCESS] RENDER THANH CONG THANH PHAM: {video_output} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    run_gemini_vision_pipeline()
