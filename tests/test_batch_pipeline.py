import os
import sys
import tempfile
import unittest
import numpy as np
import cv2
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import BatchPipelineWorker
from optimized_pipeline import validate_output_video

class BatchPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_dummy_video(self, filename="test.mp4", duration_frames=25):
        path = os.path.join(self.temp_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, 25.0, (64, 64))
        for _ in range(duration_frames):
            out.write(np.zeros((64, 64, 3), dtype=np.uint8))
        out.release()
        return path

    def test_empty_batch_queue(self):
        worker = BatchPipelineWorker([], {}, stop_on_error=False, max_workers=2)
        results = []
        worker.sig_batch_finished.connect(lambda d: results.append(d))
        worker.run()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["total"], 0)
        self.assertEqual(results[0]["completed"], 0)

    def test_batch_execution_real_output_and_workers(self):
        # Create 2 sample video files
        v1 = self._create_dummy_video("clip_1.mp4", duration_frames=20)
        v2 = self._create_dummy_video("clip_2.mp4", duration_frames=20)

        out_dir = os.path.join(self.temp_dir, "batch_out")
        cfg = {
            "output_dir": out_dir,
            "enable_dubbing": False,
            "burn_subtitles": False,
            "ocr_engine": "easyocr"
        }

        queue_items = [
            {"index": 0, "file_path": v1, "config": cfg},
            {"index": 1, "file_path": v2, "config": cfg}
        ]

        worker = BatchPipelineWorker(queue_items, cfg, stop_on_error=False, max_workers=2)

        started_events = []
        progress_events = []
        finished_items = []
        summary_events = []

        worker.sig_item_started.connect(lambda idx, name: started_events.append((idx, name)))
        worker.sig_item_progress.connect(lambda idx, p, m: progress_events.append((idx, p, m)))
        worker.sig_item_finished.connect(lambda idx, ok, p, t, s, e: finished_items.append((idx, ok, p, t, s, e)))
        worker.sig_batch_finished.connect(lambda d: summary_events.append(d))

        worker.run()

        self.assertEqual(len(summary_events), 1)
        summary = summary_events[0]
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["success"], 2)
        self.assertEqual(summary["failed"], 0)

        # Verify real outputs exist on disk and pass validation
        for item in finished_items:
            idx, ok, out_path, elapsed, size_mb, err = item
            self.assertTrue(ok, f"Item {idx} failed: {err}")
            self.assertTrue(os.path.exists(out_path), f"Output file does not exist: {out_path}")
            valid, info = validate_output_video(out_path, min_duration=0.2)
            self.assertTrue(valid, f"Validation failed for output {out_path}: {info}")
            self.assertGreater(size_mb, 0.0)

    def test_batch_cancel_behavior(self):
        v1 = self._create_dummy_video("cancel_1.mp4", duration_frames=20)
        v2 = self._create_dummy_video("cancel_2.mp4", duration_frames=20)

        out_dir = os.path.join(self.temp_dir, "cancel_out")
        cfg = {"output_dir": out_dir, "enable_dubbing": False, "burn_subtitles": False}

        queue_items = [
            {"index": 0, "file_path": v1, "config": cfg},
            {"index": 1, "file_path": v2, "config": cfg}
        ]

        worker = BatchPipelineWorker(queue_items, cfg, max_workers=1)
        worker.cancel()
        summary_events = []
        worker.sig_batch_finished.connect(lambda d: summary_events.append(d))
        worker.run()

        self.assertEqual(len(summary_events), 1)
        self.assertTrue(worker._is_cancelled)

if __name__ == "__main__":
    unittest.main()
