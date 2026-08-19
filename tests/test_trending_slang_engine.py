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
from translator import TrendingSlangManager, extract_slangs_from_text, apply_trending_slang_replacement
from slang_sync_engine import sync_online_trending_words

class TestTrendingSlangEngine(unittest.TestCase):
    """Test suite kiểm thử toàn bộ hệ thống Auto-Updating Bilibili & Douyin Trending Slang Engine."""

    def setUp(self):
        self.manager = TrendingSlangManager()
        self.manager.load_dict()

    def test_slang_extraction_and_auto_save(self):
        """1. test_slang_extraction_and_auto_save(): Kiểm tra AI bóc tách từ lóng mới từ câu tiếng Trung và lưu tự động."""
        sample_text = "这个视频让我彻底破防了，真的是绝绝子！"
        detected = extract_slangs_from_text(sample_text)

        self.assertTrue(len(detected) >= 2, f"BUG: Bóc tách từ lóng thất bại! Got {detected}")
        zh_keys = [item['zh'] for item in detected]
        self.assertIn("破防", zh_keys)
        self.assertIn("绝绝子", zh_keys)

        # Kiểm tra dữ liệu đã lưu trong file JSON
        slang_dict = self.manager.load_dict()
        self.assertIn("破防", slang_dict)
        self.assertIn("绝绝子", slang_dict)

    def test_slang_masking_in_translation(self):
        """2. test_slang_masking_in_translation(): Kiểm tra các từ lóng trong trending_dict.json được áp dụng chính xác."""
        self.manager.add_or_update_slang("显眼包", "thánh tấu hề", category="Douyin", source="Custom")
        
        raw_text = "这个主角是个显眼包"
        replaced_text = apply_trending_slang_replacement(raw_text)

        self.assertIn("thánh tấu hề", replaced_text, f"BUG: Thay thế từ lóng thất bại! Got: {replaced_text}")

    def test_online_sync_merging(self):
        """3. test_online_sync_merging(): Kiểm tra hợp nhất từ điển online không đè mất từ do người dùng tự nhập."""
        # Thêm từ lóng tùy chỉnh của người dùng
        self.manager.add_or_update_slang("自定义词", "từ tự định nghĩa", category="Custom", source="Custom")

        res = sync_online_trending_words()
        self.assertTrue(res['success'])

        slang_dict = self.manager.load_dict()
        self.assertIn("自定义词", slang_dict, "BUG: Đồng bộ online làm mất từ do người dùng tự nhập!")
        self.assertEqual(slang_dict["自定义词"]["vi"], "từ tự định nghĩa")

if __name__ == '__main__':
    unittest.main()
