try:
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer') and getattr(sys.stderr, 'encoding', '').lower() != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import unittest
from PyQt6.QtCore import QRect

class TestCropCoordinateMapping(unittest.TestCase):
    """Test suite kiểm thử công thức ánh xạ tọa độ Canvas Crop sang độ phân giải video thực tế."""

    def test_calculate_pixmap_rect_letterbox(self):
        """Kiểm thử tính toán QRect khi video 16:9 hiển thị trên Widget 16:9 và khác aspect ratio (Letterbox)."""
        # Widget size: 640x480 (4:3), Video pixmap: 1920x1080 (16:9)
        w_widget, h_widget = 640, 480
        pix_w, pix_h = 1920, 1080

        scale = min(w_widget / float(pix_w), h_widget / float(pix_h))
        scaled_w = int(pix_w * scale)
        scaled_h = int(pix_h * scale)
        offset_x = (w_widget - scaled_w) // 2
        offset_y = (h_widget - scaled_h) // 2

        rect = QRect(offset_x, offset_y, scaled_w, scaled_h)
        self.assertEqual(rect.x(), 0)
        self.assertEqual(rect.y(), 60)
        self.assertEqual(rect.width(), 640)
        self.assertEqual(rect.height(), 360)

    def test_calculate_pixmap_rect_pillarbox(self):
        """Kiểm thử tính toán QRect khi video 9:16 (dọc) hiển thị trên Widget 16:9 (Pillarbox)."""
        w_widget, h_widget = 640, 360
        pix_w, pix_h = 1080, 1920

        scale = min(w_widget / float(pix_w), h_widget / float(pix_h))
        scaled_w = int(pix_w * scale)
        scaled_h = int(pix_h * scale)
        offset_x = (w_widget - scaled_w) // 2
        offset_y = (h_widget - scaled_h) // 2

        rect = QRect(offset_x, offset_y, scaled_w, scaled_h)
        self.assertEqual(rect.y(), 0)
        self.assertTrue(offset_x > 0, "Pillarbox offset_x phải > 0")

    def test_gui_to_real_coordinate_mapping(self):
        """Kiểm thử ánh xạ điểm drag mouse trên GUI sang tọa độ video thực tế."""
        rect = QRect(0, 60, 640, 360)
        w_video, h_video = 1920, 1080

        start_x, start_y = 100, 120
        end_x, end_y = 300, 240

        x1_rel = max(0, min(start_x - rect.x(), rect.width()))
        y1_rel = max(0, min(start_y - rect.y(), rect.height()))
        x2_rel = max(0, min(end_x - rect.x(), rect.width()))
        y2_rel = max(0, min(end_y - rect.y(), rect.height()))

        x = min(x1_rel, x2_rel)
        y = min(y1_rel, y2_rel)
        w = abs(x1_rel - x2_rel)
        h = abs(y1_rel - y2_rel)

        rx = x / float(rect.width())
        ry = y / float(rect.height())
        rw = w / float(rect.width())
        rh = h / float(rect.height())

        vx = int(rx * w_video)
        vy = int(ry * h_video)
        vw = int(rw * w_video)
        vh = int(rh * h_video)

        self.assertEqual(vx, 300)
        self.assertEqual(vy, 180)
        self.assertEqual(vw, 600)
        self.assertEqual(vh, 360)

if __name__ == '__main__':
    unittest.main()
