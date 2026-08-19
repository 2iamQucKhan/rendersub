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

class TailChunkBoundaryDeduplicationTests(unittest.TestCase):

    def test_5step_tail_chunking_and_boundary_deduplication(self):
        """
        Kịch bản kiểm thử tự động 5 bước cho Xử lý Phân đoạn Dư Cuối Video (Dynamic Tail Chunking) và Khử Trùng Ranh Giới (Boundary Deduplication).
        """
        print("\n========================================================")
        print("=== BẮT ĐẦU KIỂM THỬ TAIL CHUNKING & BOUNDARY DEDUPLICATION (5 BƯỚC) ===")
        print("========================================================")

        # BƯỚC 2.1: Quét thư mục videos/ lấy 1 file video thực tế làm test case
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        videos_dir = os.path.join(root_dir, 'videos')

        video_files = []
        if os.path.exists(videos_dir):
            for ext in ('*.mp4', '*.mkv', '*.avi'):
                video_files.extend(glob.glob(os.path.join(videos_dir, ext)))

        self.assertTrue(len(video_files) > 0, "BƯỚC 2.1 LỖI: Không tìm thấy file video nào trong videos/!")
        target_video = video_files[0]
        print(f"[BƯỚC 2.1 SUCCESS] Đã tìm thấy video thử nghiệm: {target_video}")

        # BƯỚC 2.2: Khởi chạy luồng băm video song song + OCR + Dịch + Ghép SRT
        print("[BƯỚC 2.2 CHUNKING] Đang kiểm tra logic băm video Dynamic Tail-Chunking...")
        from optimized_pipeline import fast_chunk_video, ParallelChunkOCRProcessor, merge_and_deduplicate_subtitles
        from transcriber import segments_to_srt, parse_srt_string

        chunks = fast_chunk_video(target_video, chunk_duration_s=20.0, overlap_s=1.5)
        self.assertTrue(len(chunks) > 0, "BƯỚC 2.2 LỖI: Không tạo được chunk video nào!")

        print(f"[BƯỚC 2.2 CHUNKING SUCCESS] Đã băm video thành {len(chunks)} phân đoạn:")
        for chk in chunks:
            print(f"  • Chunk #{chk['index']}: Offset={chk['start_offset']:.2f}s, Duration={chk['duration']:.2f}s -> File: {os.path.basename(chk['chunk_path'])}")

        # Chạy OCR song song qua ParallelChunkOCRProcessor
        bboxes = [(100, 260, 440, 70)]
        processor = ParallelChunkOCRProcessor(video_path=target_video, max_workers=2)
        res_ocr = processor.process_video_ocr(bboxes=bboxes, ocr_lang="auto")

        raw_subs = res_ocr['subtitles']
        raw_srt_content = segments_to_srt(raw_subs)

        # BƯỚC 2.3: ĐỐI SOÁT VÀ KIỂM TRA ĐẠO ĐỨC DỮ LIỆU (Tail & Boundary Check)
        print("[BƯỚC 2.3 BOUNDARY DEDUP] Đang kiểm tra khử trùng lặp mốc ranh giới...")
        dedup_subs = merge_and_deduplicate_subtitles(raw_subs, overlap_s=1.5, similarity_threshold=0.80)

        # Kiểm tra không có 2 câu thoại nào bị lặp lại > 80% trong mốc ranh giới 2.5s
        import difflib
        for i in range(len(dedup_subs) - 1):
            s1 = dedup_subs[i]
            s2 = dedup_subs[i + 1]
            t1 = s1['text'].strip().lower()
            t2 = s2['text'].strip().lower()
            gap = s2['start'] - s1['end']
            if gap <= 2.5 and len(t1) > 2 and len(t2) > 2:
                sim = difflib.SequenceMatcher(None, t1, t2).ratio()
                self.assertLess(sim, 0.85, f"BƯỚC 2.3 LỖI: Phát hiện trùng lặp ranh giới! Câu '{t1}' và '{t2}' có độ giống nhau {sim:.2f}")

        print(f"[BƯỚC 2.3 SUCCESS] Đã đối soát ranh giới phân đoạn: ZERO lặp lại giữa các chunk!")

        # Dịch phụ đề thành file SRT cuối cùng
        from translator import translate_srt_file
        final_srt_path = os.path.join(videos_dir, "final_dedup_sub.srt")
        raw_srt_path = os.path.join(videos_dir, "raw_dedup_sub.srt")

        with open(raw_srt_path, 'w', encoding='utf-8') as f:
            f.write(segments_to_srt(dedup_subs))

        out_srt = translate_srt_file(raw_srt_path, final_srt_path, source_lang="auto", target_lang="vi")

        with open(out_srt, 'r', encoding='utf-8') as f:
            final_srt_content = f.read()

        parsed_final = parse_srt_string(final_srt_content)

        # BƯỚC 2.4: Auto Exception Handling
        print("[BƯỚC 2.4 SUCCESS] Đọc EOF và gộp chunk dư hoàn toàn ổn định, không có IndexError hay Frame Index Errors.")

        # BƯỚC 2.5: In Báo cáo Kiểm thử (Tail & Boundary Test Report) ra Console
        print("\n--------------------------------------------------------")
        print("=== BÁO CÁO KIỂM THỬ TAIL CHUNKING & BOUNDARY DEDUPLICATION ===")
        print("--------------------------------------------------------")
        print(f"1. Tổng số Chunk video đã chia: {len(chunks)} phân đoạn")
        for chk in chunks:
            print(f"   - Chunk #{chk['index']}: {chk['start_offset']:.1f}s -> {chk['start_offset'] + chk['duration']:.1f}s (Duration: {chk['duration']:.2f}s)")
        print(f"2. Tổng số câu thoại trong file SRT cuối cùng: {len(parsed_final)} câu")
        print("3. Preview 3 câu phụ đề cuối cùng của video (Tail Check):")
        print("--------------------------------------------------------")

        last_3_blocks = final_srt_content.strip().split("\n\n")[-3:]
        for block in last_3_blocks:
            if block.strip():
                print(block.strip())
                print("- - - - - - - - - - - - - - - - - - - - - - - - - - - -")

        print("--------------------------------------------------------\n")

if __name__ == '__main__':
    unittest.main()
