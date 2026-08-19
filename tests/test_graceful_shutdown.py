import sys, os, io
try:
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer') and getattr(sys.stderr, 'encoding', '').lower() != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import unittest

class TestGracefulShutdown(unittest.TestCase):
    """Test suite kiểm thử tính năng ngắt/hủy luồng an toàn (Graceful Shutdown)."""

    def test_cancellation_flag_propagation(self):
        """Kiểm thử cờ check_cancel_func trong ParallelChunkOCRProcessor dừng sớm khi bị hủy."""
        from optimized_pipeline import ParallelChunkOCRProcessor

        cancelled = False
        def check_cancel():
            return cancelled

        processor = ParallelChunkOCRProcessor(video_path="dummy.mp4", max_workers=2)
        
        # Đặt cờ hủy ngay lập tức
        cancelled = True
        res = processor.process_video_ocr(bboxes=[[0,0,100,100]], check_cancel_func=check_cancel)

        self.assertEqual(res['subtitles'], [], "Khi bị hủy, kết quả trả về phải là danh sách rỗng.")
        self.assertEqual(res['total_chunks'], 0, "Khi bị hủy, total_chunks phải = 0.")

    def test_pipeline_worker_stop(self):
        """Kiểm thử cờ _is_cancelled trong FullOneClickPipelineWorker."""
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from main import FullOneClickPipelineWorker

        worker = FullOneClickPipelineWorker(
            video_path="dummy.mp4",
            output_path="out.mp4"
        )
        self.assertFalse(worker._is_cancelled)
        worker.stop()
        self.assertTrue(worker._is_cancelled, "Hàm stop() phải đặt _is_cancelled = True")

if __name__ == '__main__':
    unittest.main()
