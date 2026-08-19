import os
import sys
import tempfile
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimized_pipeline import FFmpegVideoWriter, validate_output_video, probe_encoder_support

class FFmpegWriterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_probe_encoder_support(self):
        # libx264 should be supported with ffmpeg installed
        libx264_ok = probe_encoder_support("libx264")
        self.assertTrue(libx264_ok or probe_encoder_support("mpeg4"))

        # non-existent codec should return False
        fake_ok = probe_encoder_support("non_existent_codec_xyz_123")
        self.assertFalse(fake_ok)

    def test_writer_clean_output_no_dummy_frame(self):
        out_file = os.path.join(self.temp_dir, "test_clean.mp4")
        w, h, fps = 64, 64, 25.0
        num_frames = 20

        writer = FFmpegVideoWriter(out_file, w, h, fps=fps, codec="auto", atomic=True)
        for i in range(num_frames):
            # Generate a solid colored frame (blue)
            frame = np.full((h, w, 3), (255, 0, 0), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        self.assertTrue(os.path.exists(out_file))
        valid, info = validate_output_video(out_file, min_duration=0.5)
        self.assertTrue(valid, f"Validation failed: {info}")
        self.assertEqual(info["width"], w)
        self.assertEqual(info["height"], h)
        self.assertEqual(info["frame_count"], num_frames)

        # Check that the very first frame is NOT a black dummy frame (0,0,0)
        cap = cv2.VideoCapture(out_file)
        ret, first_frame = cap.read()
        cap.release()
        self.assertTrue(ret)
        # Blue frame mean in B channel should be high (>150), not 0
        b_mean = np.mean(first_frame[:, :, 0])
        self.assertGreater(b_mean, 100, f"First frame was black (mean={b_mean}), dummy frame detected!")

if __name__ == "__main__":
    unittest.main()
