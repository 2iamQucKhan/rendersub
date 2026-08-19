#!/usr/bin/env python3
import argparse
import os
from transcriber import run_hardsub_ocr, merge_bboxes, segments_to_srt
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Quét phụ đề cứng (hardsub) trong video và xuất SRT")
    p.add_argument("video", help="Đường dẫn tới file video")
    p.add_argument("-b", "--bbox", help="Vùng phụ đề x,y,w,h (ví dụ: 100,800,1700,200). Nếu không cung cấp sẽ quét toàn khung.", default="")
    p.add_argument("-o", "--output", help="File SRT xuất ra (mặc định cùng thư mục video)", default="")
    p.add_argument("--lang", help="Ngôn ngữ OCR (Tự động/vi/en/zh...)", default="Tự động (Trung, Việt, Anh)")
    p.add_argument("--force-scan", help="Ép quét mạnh (dùng cho chữ mờ)", action="store_true")
    args = p.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"File video không tồn tại: {video}")
        return

    if args.bbox:
        try:
            parts = [int(x) for x in args.bbox.split(",")]
            if len(parts) != 4:
                raise ValueError()
            bbox = parts
        except Exception:
            print("Định dạng --bbox không hợp lệ. Sử dụng x,y,w,h")
            return
    else:
        # Truyền danh sách rỗng để hàm run_hardsub_ocr hiểu nghĩa là tự dò vùng
        bbox = []

    print("Bắt đầu quét phụ đề... (có thể mất thời gian)")
    try:
        subs = run_hardsub_ocr(str(video), bbox, progress_callback=lambda s: print(s), ocr_lang=args.lang, force_scan=args.force_scan)
    except Exception as e:
        print(f"Lỗi khi quét phụ đề: {e}")
        return

    if not subs:
        print("Không tìm thấy phụ đề.")
        return

    # Chuyển sang segments SRT chuẩn
    segs = []
    for s in subs:
        segs.append({
            'start': s['start'],
            'end': s['end'],
            'text': s['text']
        })

    srt_text = segments_to_srt(segs)
    out_path = Path(args.output) if args.output else video.with_suffix('.srt')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        f.write(srt_text)

    print(f"Đã xuất SRT: {out_path}")


if __name__ == '__main__':
    main()
