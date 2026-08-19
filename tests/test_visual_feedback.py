import os
import sys
import unittest
import numpy as np
import cv2

# Đảm bảo import được module từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visual_feedback import draw_visual_feedback_overlay
import transcriber

class VisualFeedbackTests(unittest.TestCase):

    def setUp(self):
        # Tạo frame numpy OpenCV giả lập (640x360x3)
        self.frame = np.zeros((360, 640, 3), dtype=np.uint8)
        # Thêm vẽ một số chi tiết để kiểm tra
        cv2.circle(self.frame, (320, 180), 50, (255, 0, 0), -1)

    def test_overlay_output_shape_and_type(self):
        """Kiểm tra hàm draw_visual_feedback_overlay trả về mảng ndarray hợp lệ cùng kích thước."""
        result = draw_visual_feedback_overlay(
            frame=self.frame,
            frame_idx=15,
            total_frames=100,
            timestamp_s=0.5,
            status_text="TEST OCR SCANNING...",
            active_bbox=(50, 40, 200, 100),
            scanline_state=3
        )

        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.frame.shape)

    def test_overlay_handles_empty_or_invalid_bbox(self):
        """Kiểm tra khi bbox bị None, rỗng hoặc vượt biên độ phân giải."""
        # 1. BBox = None
        res1 = draw_visual_feedback_overlay(self.frame, 0, 100, 0.0, active_bbox=None)
        self.assertEqual(res1.shape, self.frame.shape)

        # 2. BBox vượt quá chiều rộng/dài của frame
        res2 = draw_visual_feedback_overlay(self.frame, 50, 100, 2.0, active_bbox=(-50, -20, 9999, 8888))
        self.assertEqual(res2.shape, self.frame.shape)

        # 3. Multiple BBoxes
        res3 = draw_visual_feedback_overlay(self.frame, 99, 100, 4.0, active_bbox=[[10, 10, 50, 50], [100, 100, 80, 80]])
        self.assertEqual(res3.shape, self.frame.shape)

    def test_transcriber_callback_signature(self):
        """Kiểm tra xem transcriber.run_hardsub_ocr và run_segment_guided_ocr có nhận tham số frame_callback."""
        import inspect
        sig1 = inspect.signature(transcriber.run_hardsub_ocr)
        self.assertIn('frame_callback', sig1.parameters)

        sig2 = inspect.signature(transcriber.run_segment_guided_ocr)
        self.assertIn('frame_callback', sig2.parameters)


    def test_segment_splitting_does_not_merge_entire_video(self):
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from main import MainWindow
        app_mock = MainWindow()
        single_giant_seg = [{
            'start': 0.0,
            'end': 84.37,
            'text': '[Chữ khó - Nhấp quét Gemini]',
            'confidence': 20
        }]
        res = app_mock.preprocess_extracted_segments(single_giant_seg)
        self.assertGreater(len(res), 1, "Phải phân tách video 84s thành nhiều phân đoạn thay vì 1 câu duy nhất!")
        for seg in res:
            self.assertLessEqual(seg['end'] - seg['start'], 8.0, "Mỗi phân đoạn không được dài hơn 8.0s")


if __name__ == '__main__':
    unittest.main()
