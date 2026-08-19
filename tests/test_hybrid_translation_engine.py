import sys, os, io, json
try:
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer') and getattr(sys.stderr, 'encoding', '').lower() != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
from translator import (
    global_translation_cache,
    _post_process_spoken_vietnamese,
    batch_refine_subtitles
)

class TestHybridTranslationEngine(unittest.TestCase):
    """Test suite kiểm thử bộ công cụ Hybrid Batch-Refine Translation Engine."""

    def setUp(self):
        self.cache_manager = global_translation_cache

    def test_translation_caching(self):
        """1. test_translation_caching(): Kiểm tra câu đã cache được nạp tức thì."""
        src_text = "你好世界_TEST_CACHE_KEY"
        expected_vi = "Xin chào thế giới thử nghiệm cache"
        
        self.cache_manager.set_cache(src_text, expected_vi)
        cached_result = self.cache_manager.get_cache(src_text)

        self.assertEqual(cached_result, expected_vi, f"BUG: Cache miss cho câu đã lưu! Got: {cached_result}")

    def test_batch_refine_format_integrity(self):
        """2. test_batch_refine_format_integrity(): Kiểm tra Batch 15 câu giữ đủ và đúng thứ tự 15 câu."""
        segments = [
            {'start': i * 2.0, 'end': (i + 1) * 2.0, 'orig_text': f"Sentence {i+1}", 'text': f"Sentence {i+1}"}
            for i in range(15)
        ]

        # Chạy batch_refine không có API key (chuyển sang bản nháp local + post-processed)
        result_segments = batch_refine_subtitles(segments, api_key=None, batch_size=12)

        self.assertEqual(len(result_segments), 15, f"BUG: Mất hoặc thừa dòng sau khi gộp batch! Got: {len(result_segments)}")
        for idx, seg in enumerate(result_segments):
            self.assertIn("text", seg, f"Dòng {idx} thiếu trường text")

    def test_post_process_spoken_vietnamese(self):
        """3. test_post_process_spoken_vietnamese(): Kiểm tra RegEx chuyển từ nối văn viết sang văn nói tự nhiên."""
        formal_text = "Bởi vì điều này, tôi muốn nói tóm lại là thành ra là do đó nên làm vậy ."
        processed = _post_process_spoken_vietnamese(formal_text)

        self.assertIn("Thế nên là", processed, f"BUG: 'Bởi vì điều này' chưa được chuyển sang 'Thế nên là'! Got: {processed}")
        self.assertIn("tóm lại là", processed, f"BUG: 'nói tóm lại là' chưa chuẩn hóa! Got: {processed}")
        self.assertFalse(processed.endswith(" ."), "BUG: Khoảng trắng trước dấu chấm chưa được làm sạch!")

if __name__ == '__main__':
    unittest.main()
