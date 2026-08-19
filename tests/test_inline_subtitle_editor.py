import sys, os, io
try:
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer') and getattr(sys.stderr, 'encoding', '').lower() != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

class TestInlineSubtitleEditor(unittest.TestCase):
    """Test suite kiểm thử bộ công cụ Inline Subtitle Editor trên giao diện main.py."""

    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        cls.app = QApplication.instance() or QApplication(sys.argv)
        from main import MainWindow
        cls.win = MainWindow()

    @classmethod
    def tearDownClass(cls):
        cls.win.close()

    def setUp(self):
        self.win.segments = [
            {'start': 1.0, 'end': 4.0, 'orig_text': 'Hello world', 'text': 'Xin chào thế giới'},
            {'start': 4.5, 'end': 8.0, 'orig_text': 'Welcome to tool anti', 'text': 'Chào mừng đến với tool anti'},
            {'start': 8.5, 'end': 12.0, 'orig_text': 'Good morning everyone', 'text': 'Chào buổi sáng mọi người'},
        ]
        self.win.populate_subtitle_table()

    def test_inline_edit_text_sync(self):
        """1. test_inline_edit_text_sync(): Sửa ô văn bản và kiểm tra dữ liệu self.segments cập nhật đúng."""
        # Sửa ô bản dịch ở cột 5 dòng 0
        self.win.table.item(0, 5).setText("Xin chào vũ trụ")
        self.win.on_cell_changed(0, 5)

        self.assertEqual(self.win.segments[0]['text'], "Xin chào vũ trụ",
                         "BUG: Sửa ô 'Phụ đề Dịch' không đồng bộ vào self.segments[0]['text']")

        # Sửa ô phụ đề gốc ở cột 4 dòng 1
        self.win.table.item(1, 4).setText("Welcome to anti tool")
        self.win.on_cell_changed(1, 4)

        self.assertEqual(self.win.segments[1]['orig_text'], "Welcome to anti tool",
                         "BUG: Sửa ô 'Phụ đề Gốc' không đồng bộ vào self.segments[1]['orig_text']")

    def test_search_and_filter_counter(self):
        """2. test_search_and_filter_counter(): Tìm từ khóa và xác nhận đếm đúng số câu khớp."""
        self.win.txt_sub_search.setText("chào")
        self.win.cb_sub_search_filter.setCurrentText("Tất cả")
        self.win.search_subtitles(direction=0)

        counter_text = self.win.lbl_search_count.text()
        self.assertIn("3 câu", counter_text, f"BUG: Đếm từ khóa 'chào' thất bại! Got: {counter_text}")

        # Lọc theo Gốc (OCR) -> không từ nào có 'chào'
        self.win.cb_sub_search_filter.setCurrentText("Gốc (OCR)")
        self.win.search_subtitles(direction=0)
        counter_text_ocr = self.win.lbl_search_count.text()
        self.assertIn("0 / 0", counter_text_ocr, f"BUG: Bộ lọc Gốc OCR thất bại! Got: {counter_text_ocr}")

    def test_timestamp_validation(self):
        """3. test_timestamp_validation(): Kiểm tra không cho phép lưu nếu Start Time >= End Time."""
        orig_start = self.win.segments[0]['start']
        
        # Thử nhập Start Time = 10.0s (lớn hơn End Time = 4.0s)
        from main import format_time_stamp
        invalid_start = format_time_stamp(10.0)
        self.win.table.item(0, 1).setText(invalid_start)
        self.win.on_cell_changed(0, 1)

        # Mốc thời gian phải được khôi phục về giá trị cũ
        self.assertEqual(self.win.segments[0]['start'], orig_start,
                         "BUG: Mốc thời gian Start >= End không bị từ chối/khôi phục!")

    def test_merge_and_split_rows(self):
        """4. test_merge_and_split_rows(): Kiểm tra tính toàn vẹn thời gian và văn bản khi gộp/tách dòng."""
        # Gộp dòng 0 và 1
        self.win.table.selectRow(0)
        # Select row 0 and 1
        for col in range(7):
            self.win.table.item(0, col).setSelected(True)
            self.win.table.item(1, col).setSelected(True)

        self.win.merge_selected_subtitle_rows()

        self.assertEqual(len(self.win.segments), 2, "BUG: Sau khi gộp 2 dòng, tổng số dòng phải giảm từ 3 xuống 2")
        self.assertEqual(self.win.segments[0]['start'], 1.0)
        self.assertEqual(self.win.segments[0]['end'], 8.0)
        self.assertIn("Xin chào thế giới", self.win.segments[0]['text'])
        self.assertIn("Chào mừng", self.win.segments[0]['text'])

        # Tách dòng vừa gộp
        self.win.table.selectRow(0)
        self.win.split_subtitle_row()

        self.assertEqual(len(self.win.segments), 3, "BUG: Sau khi tách dòng, tổng số dòng phải tăng lại lên 3")
        self.assertEqual(self.win.segments[0]['start'], 1.0)
        self.assertEqual(self.win.segments[0]['end'], 4.5)
        self.assertEqual(self.win.segments[1]['start'], 4.5)
        self.assertEqual(self.win.segments[1]['end'], 8.0)

if __name__ == '__main__':
    unittest.main()
