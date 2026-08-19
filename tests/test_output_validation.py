import os
import sys
import tempfile
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimized_pipeline import validate_output_video

class OutputValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_nonexistent_file(self):
        valid, msg = validate_output_video(os.path.join(self.temp_dir, "nonexistent.mp4"))
        self.assertFalse(valid)
        self.assertIn("không tồn tại", msg)

    def test_zero_byte_file(self):
        zero_file = os.path.join(self.temp_dir, "zero.mp4")
        with open(zero_file, "wb") as f:
            pass
        valid, msg = validate_output_video(zero_file)
        self.assertFalse(valid)
        self.assertIn("0 bytes", msg)

    def test_corrupted_file(self):
        corrupt_file = os.path.join(self.temp_dir, "corrupt.mp4")
        with open(corrupt_file, "wb") as f:
            f.write(b"this is not a valid mp4 video content")
        valid, msg = validate_output_video(corrupt_file)
        self.assertFalse(valid)
        self.assertTrue("OpenCV" in msg or "không hợp lệ" in msg)

    def test_valid_generated_video(self):
        valid_file = os.path.join(self.temp_dir, "valid.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(valid_file, fourcc, 25.0, (128, 128))
        for _ in range(25):
            frame = np.zeros((128, 128, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        valid, info = validate_output_video(valid_file, min_duration=0.5)
        self.assertTrue(valid, f"Validation failed: {info}")
        self.assertEqual(info["width"], 128)
        self.assertEqual(info["height"], 128)
        self.assertEqual(info["frame_count"], 25)
        self.assertGreaterEqual(info["duration_sec"], 0.9)

if __name__ == "__main__":
    unittest.main()
