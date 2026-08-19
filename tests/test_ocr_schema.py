import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimized_pipeline import merge_and_deduplicate_subtitles

class OCRSchemaTests(unittest.TestCase):
    def test_standard_schema_structure(self):
        sample_subs = [
            {"text": "你好世界", "bbox": [10, 20, 100, 30], "start": 0.0, "end": 2.5, "confidence": 0.98},
            {"text": "测试字幕", "bbox": [10, 20, 100, 30], "start": 2.6, "end": 4.0, "confidence": 0.95}
        ]
        
        merged = merge_and_deduplicate_subtitles(sample_subs)
        self.assertEqual(len(merged), 2)
        for s in merged:
            self.assertIn("text", s)
            self.assertIn("start", s)
            self.assertIn("end", s)
            self.assertIsInstance(s["start"], (int, float))
            self.assertIsInstance(s["end"], (int, float))
            self.assertGreater(s["end"], s["start"])

    def test_deduplication_and_overlap_cleanup(self):
        overlapping_subs = [
            {"text": "欢迎观看本期视频", "bbox": [10, 20, 200, 30], "start": 0.0, "end": 2.0, "confidence": 0.95},
            {"text": "欢迎观看本期视频", "bbox": [10, 20, 200, 30], "start": 1.5, "end": 3.5, "confidence": 0.96},
        ]
        merged = merge_and_deduplicate_subtitles(overlapping_subs, overlap_s=1.5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["start"], 0.0)
        self.assertEqual(merged[0]["end"], 3.5)
        self.assertEqual(merged[0]["text"], "欢迎观看本期视频")

    def test_filter_ocr_garbage_short_noise(self):
        noisy_subs = [
            {"text": ".", "bbox": [0, 0, 10, 10], "start": 0.0, "end": 1.0, "confidence": 0.3},
            {"text": "a", "bbox": [0, 0, 10, 10], "start": 1.0, "end": 2.0, "confidence": 0.2},
            {"text": "Xin chào các bạn", "bbox": [10, 20, 150, 30], "start": 2.0, "end": 4.0, "confidence": 0.99},
        ]
        merged = merge_and_deduplicate_subtitles(noisy_subs)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "Xin chào các bạn")

if __name__ == "__main__":
    unittest.main()
