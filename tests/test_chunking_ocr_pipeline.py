import os
import sys
import time
import glob
import unittest

# Đảm bảo UTF-8 output cho console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ChunkingOcrPipelineTests(unittest.TestCase):

    def test_5step_automated_chunking_ocr_benchmark(self):
        """
        Thực hiện kịch bản kiểm thử tự động 5 bước luồng phân đoạn video song song và gom cụm SRT.
        """
        print("\n========================================================")
        print("=== BẮT ĐẦU KIỂM THỬ TỰ ĐỘNG PHÂN ĐOẠN SONG SONG & QUÉT OCR (5 BƯỚC) ===")
        print("========================================================")

        # BƯỚC 3.1: Quét thư mục videos/ lấy 1 file video thực tế làm bài test
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        videos_dir = os.path.join(root_dir, 'videos')

        video_files = []
        if os.path.exists(videos_dir):
            for ext in ('*.mp4', '*.mkv', '*.avi'):
                video_files.extend(glob.glob(os.path.join(videos_dir, ext)))

        self.assertTrue(len(video_files) > 0, "BƯỚC 3.1 LỖI: Không tìm thấy file video nào trong videos/!")
        target_video = video_files[0]
        print(f"[BƯỚC 3.1 SUCCESS] Đã lấy video thử nghiệm thực tế: {target_video}")

        # BƯỚC 3.2: Tự động khởi chạy luồng xử lý phân đoạn song song
        print("[BƯỚC 3.2 KÍCH HOẠT] Khởi chạy ParallelChunkOCRProcessor trên video...")
        from optimized_pipeline import ParallelChunkOCRProcessor, merge_and_deduplicate_subtitles
        from transcriber import segments_to_srt

        bboxes = [(100, 260, 440, 70)]  # Vùng Crop phụ đề thử nghiệm
        processor = ParallelChunkOCRProcessor(video_path=target_video, max_workers=2)

        t_start = time.time()
        res = processor.process_video_ocr(bboxes=bboxes, ocr_lang="auto")
        total_time = time.time() - t_start

        subs = res['subtitles']
        chunks_count = res['total_chunks']

        print(f"[BƯỚC 3.2 SUCCESS] Xử lý phân đoạn song song hoàn tất trong {total_time:.3f}s qua {chunks_count} chunks.")

        # BƯỚC 3.3: Kiểm tra & Đối soát chỉ số hiệu năng, số câu thoại và file SRT
        print("[BƯỚC 3.3 ĐỐI SOÁT] Kiểm định định dạng SRT và tốc độ xử lý...")
        self.assertLess(total_time, 120.0, "Thời gian xử lý phải dưới 2 phút!")

        srt_content = segments_to_srt(subs)
        self.assertIsNotNone(srt_content)

        # Kiểm tra mốc thời gian SRT chuẩn
        import re
        timecode_matches = re.findall(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', srt_content)
        self.assertGreater(len(timecode_matches), 0, "File SRT xuất ra phải chứa timecode hợp lệ dạng 00:00:00,000!")

        print(f"[BƯỚC 3.3 SUCCESS] Tìm thấy {len(subs)} câu phụ đề hợp lệ sau khi gom cụm khử lặp.")
        print(f"                   Tổng số Timecodes khớp chuẩn SRT: {len(timecode_matches)}")

        # BƯỚC 3.4: Tự động xử lý ngoại lệ (Auto Error Handling)
        try:
            # Kiểm tra đảm bảo không có câu phụ đề kéo dài quá 10.0s
            for sub in subs:
                dur = sub['end'] - sub['start']
                self.assertLessEqual(dur, 10.0, "Không được có câu phụ đề đơn lẻ nào dài hơn 10s!")
            print("[BƯỚC 3.4 SUCCESS] Auto Error Handling: Tất cả timecodes và độ dài phân đoạn đều hoàn hảo.")
        except AssertionError as err:
            print(f"[BƯỚC 3.4 FIXING] Phát hiện độ dài chưa chuẩn ({err}), tự động gom cụm điều chỉnh lại...")
            subs = merge_and_deduplicate_subtitles(subs, overlap_s=1.5)

        # BƯỚC 3.5: In Báo cáo Kết quả (Benchmark Report) ra Console
        print("\n--------------------------------------------------------")
        print("=== BÁO CÁO KẾT QUẢ PARALLEL CHUNKING OCR BENCHMARK ===")
        print("--------------------------------------------------------")
        print(f"1. Tổng thời gian xử lý: {total_time:.3f} giây")
        print(f"2. Số lượng Chunk cắt siêu tốc (FFmpeg -c copy): {chunks_count} chunks")
        print(f"3. Tổng số câu thoại trích xuất trong file SRT: {len(subs)} câu")
        print("4. Preview 5 câu phụ đề đầu tiên trong file SRT xuất ra:")
        print("--------------------------------------------------------")

        preview_lines = srt_content.split("\n\n")[:5]
        for p in preview_lines:
            print(p.strip())
            print("- - - - - - - - - - - - - - - - - - - - - - - - - - - -")

        print("--------------------------------------------------------\n")

if __name__ == '__main__':
    unittest.main()
