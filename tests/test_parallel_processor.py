import os
import sys
import tempfile
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimized_pipeline import ParallelVideoProcessor, validate_output_video

class ParallelProcessorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_sample_video(self, num_frames=30, width=64, height=64):
        vpath = os.path.join(self.temp_dir, "sample_input.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(vpath, fourcc, 25.0, (width, height))
        for i in range(num_frames):
            # Write frame with frame index value in top-left pixel
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[0, 0, 0] = i % 256
            out.write(frame)
        out.release()
        return vpath

    def test_strict_frame_ordering(self):
        in_path = self._create_sample_video(num_frames=20)
        out_path = os.path.join(self.temp_dir, "sample_ordered_out.mp4")

        # Process function stamps monochrome brightness step (idx * 10) across entire frame
        def process_fn(frame, f_idx, total_f, fps, inpainter):
            val = int(min(255, f_idx * 12))
            out_f = np.full(frame.shape, val, dtype=np.uint8)
            return out_f

        processor = ParallelVideoProcessor(in_path, out_path, process_fn, max_queue_size=16, num_workers=4)
        success = processor.run()
        self.assertTrue(success)

        valid, info = validate_output_video(out_path, min_duration=0.5)
        self.assertTrue(valid, f"Validation failed: {info}")
        self.assertEqual(info["frame_count"], 20)

        # Read output and verify frames are strictly in monotonic ascending order
        cap = cv2.VideoCapture(out_path)
        read_means = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            read_means.append(float(np.mean(frame)))
        cap.release()

        self.assertEqual(len(read_means), 20)
        # Check monotonic increase
        for i in range(len(read_means) - 1):
            self.assertLess(read_means[i], read_means[i + 1], f"Frame order violated at index {i}: {read_means}")

    def test_worker_exception_propagation(self):
        in_path = self._create_sample_video(num_frames=20)
        out_path = os.path.join(self.temp_dir, "sample_error_out.mp4")

        def fault_process_fn(frame, f_idx, total_f, fps, inpainter):
            if f_idx >= 5:
                raise ValueError("Simulated processing worker fault")
            return frame

        processor = ParallelVideoProcessor(in_path, out_path, fault_process_fn, num_workers=2)
        with self.assertRaises(ValueError) as ctx:
            processor.run()
        self.assertIn("Simulated processing worker fault", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
