import os
import sys
import tempfile
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimized_pipeline import is_ascii_path, ensure_safe_ascii_video_path, validate_output_video

class UnicodePathsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ascii_detection(self):
        self.assertTrue(is_ascii_path("C:/videos/simple_test.mp4"))
        self.assertFalse(is_ascii_path("C:/videos/中文视频.mp4"))
        self.assertFalse(is_ascii_path("C:/videos/日本語.mp4"))
        self.assertFalse(is_ascii_path("C:/videos/Tiếng Việt.mp4"))
        self.assertFalse(is_ascii_path("C:/videos/Видео.mp4"))

    def test_unicode_video_creation_and_validation(self):
        unicode_name = "Video_Tiếng_Việt_中文_日本語_테스트.mp4"
        unicode_path = os.path.join(self.temp_dir, unicode_name)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(unicode_path, fourcc, 25.0, (64, 64))
        for _ in range(15):
            out.write(np.zeros((64, 64, 3), dtype=np.uint8))
        out.release()

        self.assertTrue(os.path.exists(unicode_path))
        valid, info = validate_output_video(unicode_path, min_duration=0.2)
        self.assertTrue(valid, f"Validation failed for unicode path: {info}")

        safe_path = ensure_safe_ascii_video_path(unicode_path)
        self.assertTrue(os.path.exists(safe_path))

if __name__ == "__main__":
    unittest.main()
