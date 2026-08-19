import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import translator
from translator import TrendingSlangManager, VietPhraseTranslator

class TranslationQATests(unittest.TestCase):
    def test_trending_slang_manager(self):
        manager = TrendingSlangManager()
        text = "Lão bản này thật là ngưu bức quá đi"
        replaced = manager.replace_slang(text)
        self.assertIsInstance(replaced, str)
        self.assertTrue(len(replaced) > 0)

    def test_vietphrase_translator(self):
        vp = VietPhraseTranslator()
        translated = vp.translate("你好")
        self.assertIsInstance(translated, str)
        self.assertTrue("chào" in translated.lower() or "hảo" in translated.lower() or len(translated) > 0)

    def test_translate_segments_empty_and_normal(self):
        empty_segs = []
        res = translator.translate_segments(empty_segs, engine="Google Translate")
        self.assertEqual(res, [])

        sample_segs = [
            {"text": "Hello world", "start": 0.0, "end": 2.0},
            {"text": "Good morning", "start": 2.0, "end": 4.0}
        ]
        res = translator.translate_segments(sample_segs, source_lang="en", target_lang="vi", engine="Google Translate")
        self.assertEqual(len(res), 2)
        self.assertTrue(any("chào" in s["text"].lower() or "thế giới" in s["text"].lower() for s in res))

if __name__ == "__main__":
    unittest.main()
