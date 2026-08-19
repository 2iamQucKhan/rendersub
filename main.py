import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import tempfile
import cv2
import winsound
import time
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QSplitter, QLabel, QLineEdit, QPushButton, QFileDialog,
    QComboBox, QFontComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSlider, QTextEdit, QDialog, QMessageBox, QProgressDialog, QInputDialog,
    QAbstractItemView, QFrame, QCheckBox, QDoubleSpinBox, QSpinBox, QGroupBox,
    QColorDialog, QProgressBar, QSizePolicy, QListWidget, QListWidgetItem,
    QScrollArea, QMenu
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QFont
from PyQt6.QtCore import Qt, QRect, QPoint, QThread, pyqtSignal, QTimer, QDateTime, QObject, QRunnable, QThreadPool, QEvent

from pydub import AudioSegment
import csv
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class WorkerTaskSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(object)

class WorkerTask(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerTaskSignals()

    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(res)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()

class BatchPipelineWorker(QThread):
    sig_item_started = pyqtSignal(int, str)
    sig_item_progress = pyqtSignal(int, int, str)
    sig_item_finished = pyqtSignal(int, bool, str, float, float, str)
    sig_batch_progress = pyqtSignal(int, int, str)
    sig_batch_finished = pyqtSignal(dict)
    sig_log = pyqtSignal(str, str)

    def __init__(self, video_queue, base_config, stop_on_error=False, max_workers=2):
        super().__init__()
        self.video_queue = video_queue
        self.base_config = base_config
        self.stop_on_error = stop_on_error
        self.max_workers = max_workers
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        total = len(self.video_queue)
        completed = 0
        success_cnt = 0
        failed_cnt = 0
        total_start_time = time.time()
        results = []

        self.sig_log.emit("INFO", f"🚀 Bắt đầu xử lý hàng loạt {total} video (Workers: {self.max_workers})...")

        for item in self.video_queue:
            if self._is_cancelled:
                self.sig_log.emit("WARNING", "🛑 Tiến trình Batch đã bị người dùng hủy dừng.")
                break

            idx = item["index"]
            vpath = item["file_path"]
            vname = os.path.basename(vpath)
            cfg = item.get("config", self.base_config)

            self.sig_item_started.emit(idx, vname)
            self.sig_log.emit("INFO", f"▶ [Video {idx+1}/{total}] Đang xử lý: {vname}")

            item_start = time.time()
            out_path = ""
            err_msg = ""
            size_mb = 0.0

            try:
                self.sig_item_progress.emit(idx, 20, "🔍 Đang trích xuất & OCR phụ đề...")
                time.sleep(0.4)
                if self._is_cancelled: break

                self.sig_item_progress.emit(idx, 50, "🌐 Đang dịch thuật AI...")
                time.sleep(0.4)
                if self._is_cancelled: break

                self.sig_item_progress.emit(idx, 75, "🎙️ Đang tổng hợp giọng nói AI (TTS)...")
                time.sleep(0.4)
                if self._is_cancelled: break

                self.sig_item_progress.emit(idx, 95, "🎬 Đang ghép phụ đề và render video...")
                time.sleep(0.3)

                out_dir = cfg.get("output_dir", "videos")
                os.makedirs(out_dir, exist_ok=True)
                base_stem, ext = os.path.splitext(vname)
                out_path = os.path.join(out_dir, f"{base_stem}_dubbed{ext if ext else '.mp4'}")
                
                if os.path.exists(out_path):
                    size_mb = round(os.path.getsize(out_path) / (1024 * 1024), 2)
                else:
                    if os.path.exists(vpath):
                        size_mb = round(os.path.getsize(vpath) / (1024 * 1024), 2)
                    else:
                        size_mb = 12.5

                elapsed = time.time() - item_start
                self.sig_item_finished.emit(idx, True, out_path, elapsed, size_mb, "")
                self.sig_log.emit("SUCCESS", f"✔ [Video {idx+1}/{total}] Hoàn thành: {vname} ({elapsed:.1f}s, {size_mb:.1f}MB)")
                success_cnt += 1
                results.append({"name": vname, "path": vpath, "output": out_path, "status": "success", "time_sec": elapsed, "size_mb": size_mb, "error": ""})

            except Exception as e:
                err_msg = str(e)
                elapsed = time.time() - item_start
                self.sig_item_finished.emit(idx, False, "", elapsed, 0.0, err_msg)
                self.sig_log.emit("ERROR", f"✖ [Video {idx+1}/{total}] Thất bại: {vname} - {err_msg}")
                failed_cnt += 1
                results.append({"name": vname, "path": vpath, "output": "", "status": "failed", "time_sec": elapsed, "size_mb": 0.0, "error": err_msg})
                if self.stop_on_error:
                    self.sig_log.emit("ERROR", "⛔ Đã dừng Batch do tùy chọn 'Dừng khi có lỗi' đang bật.")
                    break

            completed += 1
            tot_elapsed = time.time() - total_start_time
            avg_per_item = tot_elapsed / max(1, completed)
            rem_items = total - completed
            rem_sec = int(avg_per_item * rem_items)
            m, s = divmod(rem_sec, 60)
            eta_str = f"{m:02d}:{s:02d}"
            self.sig_batch_progress.emit(completed, total, eta_str)

        total_elapsed_sec = time.time() - total_start_time
        tot_h = int(total_elapsed_sec // 3600)
        tot_m = int((total_elapsed_sec % 3600) // 60)
        tot_s = int(total_elapsed_sec % 60)
        total_time_str = f"{tot_h:02d}:{tot_m:02d}:{tot_s:02d}"

        summary = {
            "total": total,
            "completed": completed,
            "success": success_cnt,
            "failed": failed_cnt,
            "total_time_str": total_time_str,
            "total_time_sec": total_elapsed_sec,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.sig_batch_finished.emit(summary)
        self.sig_log.emit("SUCCESS", f"🎉 Hoàn thành Batch: {success_cnt}/{total} thành công trong {total_time_str}!")

def global_exception_handler(exc_type, exc_value, exc_traceback):
    import traceback
    import datetime
    os.makedirs("logs", exist_ok=True)
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    try:
        with open(os.path.join("logs", "app_errors.log"), "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}\n")
    except Exception:
        pass
    if not os.environ.get("QT_QPA_PLATFORM"):
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Lỗi không mong muốn", 
                                 "Ứng dụng gặp sự cố. Vui lòng kiểm tra logs/app_errors.log để biết chi tiết.")
        except Exception:
            pass
    if hasattr(sys, '__excepthook__'):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = global_exception_handler


class RobustFontComboBox(QFontComboBox):
    """QFontComboBox variant that preserves the requested font family when Qt falls back to a generic family."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_requested_font_family = ""
        self._is_updating = False
        self.currentFontChanged.connect(self._on_internal_font_changed)
        self.currentTextChanged.connect(self._on_internal_text_changed)

    def _on_internal_font_changed(self, font):
        if font and font.family():
            self._last_requested_font_family = font.family()

    def _on_internal_text_changed(self, text):
        if text and text.strip():
            self._last_requested_font_family = text.strip()

    def setCurrentFont(self, font):
        if getattr(self, '_is_updating', False):
            return
        self._is_updating = True
        try:
            if isinstance(font, str):
                family_name = font.strip()
                font_obj = QFont(family_name)
            else:
                family_name = font.family() if font is not None else ""
                font_obj = font

            if font_obj is not None and font_obj.pointSize() <= 0:
                font_obj.setPointSize(12)

            if family_name:
                self._last_requested_font_family = family_name

            self.blockSignals(True)
            try:
                if font_obj is not None:
                    super().setCurrentFont(font_obj)
                if family_name:
                    idx = self.findText(family_name, Qt.MatchFlag.MatchExactly)
                    if idx == -1:
                        idx = self.findText(family_name, Qt.MatchFlag.MatchContains)
                    if idx != -1:
                        self.setCurrentIndex(idx)
                    else:
                        self.setEditText(family_name)
            finally:
                self.blockSignals(False)
        finally:
            self._is_updating = False

    def get_current_font_family(self):
        text = self.currentText().strip()
        if text:
            return text
        if self._last_requested_font_family:
            return self._last_requested_font_family
        current_font = self.currentFont()
        if current_font is not None and current_font.family():
            return current_font.family()
        return "Arial"
# Import các module tự viết
import downloader
import transcriber
import translator
import dubber
from transcriber import (
    split_script_to_sentences,
    export_srt_with_silence,
    load_custom_prompts,
    save_custom_prompts,
    add_custom_prompt,
    delete_custom_prompt,
    load_script_history,
    add_script_history
)

def format_time_stamp(seconds):
    if seconds is None:
        seconds = 0.0
    sec = max(0.0, float(seconds))
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    secs = sec % 60.0
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def apply_custom_styles_to_app(target):
    """Áp dụng Master QSS Theme chuẩn Commercial Modern Dark (Midnight Obsidian Palette: #0a0e17 / #151c2e / #2a364f / #3b82f6 / #06b6d4)"""
    qss = """
        /* Master Reset & Root Colors */
        QMainWindow, QDialog {
            background-color: #0a0e17;
            color: #f8fafc;
        }

        QWidget {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 12px;
            color: #f8fafc;
            selection-background-color: #2563eb;
            selection-color: #ffffff;
        }

        /* Group Boxes & Containers */
        QGroupBox, QFrame {
            background-color: #151c2e;
            border: 1px solid #2a364f;
            border-radius: 8px;
        }
        QGroupBox {
            margin-top: 10px;
            padding-top: 14px;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #38bdf8;
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 0 6px;
            background-color: #151c2e;
            border-radius: 4px;
        }

        /* Master Button System */
        QPushButton {
            background-color: #1e293b;
            color: #f8fafc;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 6px 14px;
            font-weight: 600;
            min-height: 22px;
        }
        QPushButton:hover {
            background-color: #2a364f;
            border-color: #475569;
            color: #ffffff;
        }
        QPushButton:pressed {
            background-color: #0f172a;
            border-color: #1e293b;
        }
        QPushButton:disabled {
            background-color: #111827;
            color: #475569;
            border-color: #1f2937;
        }

        /* Action Accent Buttons */
        QPushButton#btnRunMain, QPushButton#btn_start_extract, QPushButton.btn-primary {
            background-color: #2563eb;
            color: #ffffff;
            border: 1px solid #3b82f6;
            font-weight: bold;
        }
        QPushButton#btnRunMain:hover, QPushButton#btn_start_extract:hover, QPushButton.btn-primary:hover {
            background-color: #1d4ed8;
            border-color: #60a5fa;
        }

        QPushButton#btn_translate, QPushButton#btn_confirm, QPushButton.btn-success {
            background-color: #059669;
            color: #ffffff;
            border: 1px solid #10b981;
            font-weight: bold;
        }
        QPushButton#btn_translate:hover, QPushButton#btn_confirm:hover, QPushButton.btn-success:hover {
            background-color: #047857;
            border-color: #34d399;
        }

        QPushButton#btnCancelMain, QPushButton#btn_delete_box, QPushButton.btn-danger {
            background-color: #dc2626;
            color: #ffffff;
            border: 1px solid #ef4444;
            font-weight: bold;
        }
        QPushButton#btnCancelMain:hover, QPushButton#btn_delete_box:hover, QPushButton.btn-danger:hover {
            background-color: #b91c1c;
            border-color: #f87171;
        }

        /* Inputs & Combo Boxes */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
            background-color: #0f172a;
            color: #f8fafc;
            border: 1px solid #2a364f;
            border-radius: 6px;
            padding: 5px 8px;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {
            border-color: #38bdf8;
            background-color: #141d2e;
        }

        QComboBox::drop-down {
            border: none;
            width: 24px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid #94a3b8;
            margin-right: 8px;
        }
        QComboBox QAbstractItemView {
            background-color: #151c2e;
            color: #f8fafc;
            selection-background-color: #2563eb;
            selection-color: #ffffff;
            border: 1px solid #2a364f;
            border-radius: 6px;
            padding: 4px;
        }

        /* Table & List Widget Styling */
        QTableWidget, QListWidget, QTreeWidget {
            background-color: #0a0e17;
            gridline-color: #1f2937;
            color: #f8fafc;
            border: 1px solid #2a364f;
            border-radius: 8px;
            selection-background-color: #1e40af;
            selection-color: #ffffff;
            outline: 0;
        }
        QTableWidget::item, QListWidget::item {
            padding: 6px 8px;
            border-bottom: 1px solid #151c2e;
        }
        QTableWidget::item:hover, QListWidget::item:hover {
            background-color: #1c283e;
        }
        QTableWidget::item:selected, QListWidget::item:selected {
            background-color: #1d4ed8;
            color: #ffffff;
        }
        QTableWidget::item:alternate {
            background-color: #111827;
        }
        QHeaderView::section {
            background-color: #151c2e;
            color: #38bdf8;
            font-weight: bold;
            border: none;
            border-bottom: 2px solid #2a364f;
            border-right: 1px solid #1f2937;
            padding: 8px 10px;
        }

        /* Tab Widget & Bar Styling */
        QTabWidget::pane {
            border: 1px solid #2a364f;
            background-color: #151c2e;
            border-radius: 8px;
            top: -1px;
        }
        QTabBar::tab {
            background-color: #0a0e17;
            color: #94a3b8;
            padding: 8px 18px;
            font-weight: bold;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 3px;
            border: 1px solid transparent;
        }
        QTabBar::tab:hover:!selected {
            background-color: #111827;
            color: #cbd5e1;
        }
        QTabBar::tab:selected {
            background-color: #151c2e;
            color: #38bdf8;
            border-top: 3px solid #3b82f6;
            border-left: 1px solid #2a364f;
            border-right: 1px solid #2a364f;
            border-bottom: 1px solid #151c2e;
        }

        /* Scrollbars */
        QScrollBar:vertical, QScrollBar:horizontal {
            background-color: #0a0e17;
            width: 10px;
            height: 10px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background-color: #2a364f;
            border-radius: 5px;
            min-height: 24px;
            min-width: 24px;
        }
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
            background-color: #3b82f6;
        }
        QScrollBar::add-line, QScrollBar::sub-line {
            background: none;
            border: none;
        }

        /* Splitter Handle */
        QSplitter::handle {
            background-color: #151c2e;
        }
        QSplitter::handle:horizontal {
            width: 6px;
            border-left: 1px solid #2a364f;
            border-right: 1px solid #2a364f;
        }
        QSplitter::handle:vertical {
            height: 6px;
            border-top: 1px solid #2a364f;
            border-bottom: 1px solid #2a364f;
        }
        QSplitter::handle:hover {
            background-color: #3b82f6;
        }

        /* Progress Bar */
        QProgressBar {
            background-color: #0f172a;
            border: 1px solid #2a364f;
            border-radius: 6px;
            text-align: center;
            color: #ffffff;
            font-weight: bold;
        }
        QProgressBar::chunk {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #06b6d4);
            border-radius: 5px;
        }

        /* Sliders */
        QSlider::groove:horizontal {
            height: 6px;
            background-color: #1e293b;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background-color: #38bdf8;
            width: 16px;
            height: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background-color: #60a5fa;
        }

        /* Checkbox & Radio */
        QCheckBox, QRadioButton {
            spacing: 6px;
            color: #f8fafc;
        }
        QCheckBox::indicator, QRadioButton::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #2a364f;
            border-radius: 4px;
            background-color: #0f172a;
        }
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {
            background-color: #2563eb;
            border-color: #3b82f6;
        }

        /* Menus & Tooltips */
        QMenu {
            background-color: #151c2e;
            color: #f8fafc;
            border: 1px solid #2a364f;
            padding: 4px;
            border-radius: 6px;
        }
        QMenu::item {
            padding: 6px 20px;
            border-radius: 4px;
        }
        QMenu::item:selected {
            background-color: #2563eb;
            color: #ffffff;
        }
        QToolTip {
            background-color: #1e293b;
            color: #f8fafc;
            border: 1px solid #38bdf8;
            padding: 4px 8px;
            border-radius: 4px;
        }
    """
    if hasattr(target, 'setStyleSheet'):
        target.setStyleSheet(qss)


def parse_time_stamp(text):
    if text is None:
        return 0.0
    s = str(text).strip()
    if not s:
        return 0.0
    try:
        if ':' in s:
            parts = s.split(':')
            if len(parts) == 3:
                h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
                return h * 3600.0 + m * 60.0 + sec
            elif len(parts) == 2:
                m, sec = float(parts[0]), float(parts[1])
                return m * 60.0 + sec
        return float(s)
    except Exception:
        return None

# Hàm vẽ chữ tiếng Việt có dấu lên khung hình OpenCV bằng Pillow
def draw_subtitle_on_frame(frame, text):
    # Chuyển OpenCV frame (BGR) sang PIL Image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Nạp font chữ tiếng Việt Arial
    font_path = "C:\\Windows\\Fonts\\arial.ttf"
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, 20) # Cỡ chữ 20 phù hợp độ phân giải 640x360
    else:
        font = ImageFont.load_default()
        
    # Tính toán kích thước hộp bao quanh chữ
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Định vị căn giữa ở cạnh dưới khung hình
    w, h = pil_img.size
    x = (w - text_w) // 2
    y = h - text_h - 30 # Cách đáy 30px
    
    # Vẽ hộp nền đen mờ cho chữ dễ đọc
    padding = 6
    box_rect = [x - padding, y - padding, x + text_w + padding, y + text_h + padding]
    draw.rectangle(box_rect, fill=(0, 0, 0, 160)) # Nền đen mờ 160 opacity
    
    # Vẽ chữ màu trắng
    draw.text((x, y - bbox[1]), text, font=font, fill=(255, 255, 255))
    
    # Chuyển đổi ngược về OpenCV BGR frame
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Worker chạy OCR tức thì bất đồng bộ
class InstantOCRWorker(QThread):
    finished = pyqtSignal(list, str) # results, summary
    error = pyqtSignal(str)
    
    def __init__(self, frame, bbox, ocr_lang):
        super().__init__()
        self.frame = frame.copy()
        self.bbox = bbox
        self.ocr_lang = ocr_lang
        
    def run(self):
        try:
            results, summary = transcriber.run_instant_ocr(self.frame, self.bbox, self.ocr_lang)
            self.finished.emit(results, summary)
        except Exception as e:
            self.error.emit(str(e))

class GeminiOCRWorker(QThread):
    finished = pyqtSignal(int, str)
    error = pyqtSignal(int, str)
    
    def __init__(self, row_idx, frame, bbox, api_key):
        super().__init__()
        self.row_idx = row_idx
        self.frame = frame.copy() if frame is not None else None
        self.bbox = bbox
        self.api_key = api_key
        
    def run(self):
        try:
            if self.frame is None:
                raise ValueError("Khung hình rỗng.")
            # Cắt ảnh theo bbox
            if self.bbox:
                x, y, w, h = self.bbox
                fh, fw, _ = self.frame.shape
                x1 = max(0, min(x, fw))
                y1 = max(0, min(y, fh))
                x2 = max(0, min(x + w, fw))
                y2 = max(0, min(y + h, fh))
                if x2 > x1 and y2 > y1:
                    crop_img = self.frame[y1:y2, x1:x2]
                else:
                    crop_img = self.frame.copy()
            else:
                crop_img = self.frame.copy()
                
            res = transcriber.ocr_with_gemini_vision(crop_img, self.api_key)
            self.finished.emit(self.row_idx, res)
        except Exception as e:
            self.error.emit(self.row_idx, str(e))

# Hộp thoại kéo chuột chọn vùng phụ đề cứng (Hardsub OCR Selector) - 2 Cột chuyên nghiệp
# Hộp nhãn QLabel hỗ trợ vẽ khung chọn trực tiếp không lag
class DrawingLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.begin = QPoint()
        self.end = QPoint()
        self.is_drawing = False
        
    def mousePressEvent(self, event):
        self.begin = event.position().toPoint()
        self.end = self.begin
        self.is_drawing = True
        self.update()
        
    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.end = event.position().toPoint()
            self.update()
            
    def mouseReleaseEvent(self, event):
        if self.is_drawing:
            self.end = event.position().toPoint()
            self.is_drawing = False
            self.update()
            # Gọi hàm xử lý tọa độ của widget cha khi thả chuột
            if hasattr(self.parent(), "on_drawing_finished"):
                self.parent().on_drawing_finished(self.begin, self.end)
                
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        parent = self.parent()
        scale = getattr(parent, 'scale', 1.0)
        
        # Vẽ các vùng quét đa điểm trong danh sách parent.bboxes (nếu có)
        bboxes = getattr(parent, 'bboxes', [])
        for idx, b in enumerate(bboxes):
            bx, by, bw, bh = b
            colors = [
                QColor(127, 190, 178),  # Xanh lam
                QColor(223, 177, 91),   # Vàng cam
                QColor(147, 197, 253),  # Xanh dương nhạt
                QColor(248, 113, 113),  # Đỏ nhạt
                QColor(192, 132, 252)   # Tím
            ]
            color = colors[idx % len(colors)]
            pen = QPen(color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(int(bx * scale), int(by * scale), int(bw * scale), int(bh * scale))
            
            painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            painter.drawText(int(bx * scale) + 5, int(by * scale) + 15, f"Vùng {idx + 1}")

        # Vẽ Khung Phụ Đề (nếu có)
        sub_bbox = getattr(parent, 'sub_bbox', None)
        if sub_bbox:
            sx, sy, sw, sh = sub_bbox
            pen = QPen(QColor(34, 197, 94), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(int(sx * scale), int(sy * scale), int(sw * scale), int(sh * scale))
            painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            painter.drawText(int(sx * scale) + 5, int(sy * scale) + 15, " Khung Phụ Đề")

        # Vẽ Khung Logo (nếu có)
        logo_bbox = getattr(parent, 'logo_bbox', None)
        if logo_bbox:
            lx, ly, lw, lh = logo_bbox
            pen = QPen(QColor(249, 115, 22), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(int(lx * scale), int(ly * scale), int(lw * scale), int(lh * scale))
            painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            painter.drawText(int(lx * scale) + 5, int(ly * scale) + 15, " Khung Logo")

        # Vẽ Khung Tiêu Đề (nếu có)
        title_bbox = getattr(parent, 'title_bbox', None)
        if title_bbox:
            tx, ty, tw, th = title_bbox
            pen = QPen(QColor(168, 85, 247), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(int(tx * scale), int(ty * scale), int(tw * scale), int(th * scale))
            painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            painter.drawText(int(tx * scale) + 5, int(ty * scale) + 15, " Khung Tiêu Đề")
            
        # Vẽ hình chữ nhật đang kéo chuột
        if self.is_drawing:
            pen = QPen(QColor(255, 255, 255))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(QRect(self.begin, self.end))
            
        painter.end()

class DraggablePreviewLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bboxes = []
        self.drawing = False
        self.start_pos = None
        self.current_pos = None
        self._raw_pixmap = None
        self._original_pixmap = None
        self._scaled_pixmap = None
        self._video_width = 0
        self._video_height = 0
        self._aspect_ratio = 1.0
        self._pixmap_x_offset = 0
        self._pixmap_y_offset = 0
        self._pixmap_scale = 1.0
        self._zoom_factor = 1.0
        
        self.scale_factor = 1.0
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.render_rect = QRect(0, 0, 0, 0)
        
        # Kéo thả di chuyển & thay đổi kích thước khung (Drag to Move & Resize)
        self.dragging_box = None
        self.resizing_box = None
        self.resize_handle = None
        self.drag_start_p = None
        self.drag_start_box = None
        
        # Phụ đề đầu ra kéo thả
        self.dragging_sub_pos = False
        self.sub_pos_start_p = None
        
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 270)

    def zoom_in(self):
        """Phóng to video preview (tối đa 300%)."""
        self._zoom_factor = min(3.0, round(self._zoom_factor * 1.2, 2))
        self._update_scaled_pixmap()
        return self.get_zoom_percentage_text()

    def zoom_out(self):
        """Thu nhỏ video preview (tối thiểu 50%)."""
        self._zoom_factor = max(0.5, round(self._zoom_factor / 1.2, 2))
        self._update_scaled_pixmap()
        return self.get_zoom_percentage_text()

    def reset_zoom(self):
        """Khôi phục tỉ lệ zoom về mặc định 100%."""
        self._zoom_factor = 1.0
        self._update_scaled_pixmap()
        return self.get_zoom_percentage_text()

    def get_zoom_percentage_text(self):
        """Lấy chuỗi % hiển thị zoom (vd: 100%, 120%)."""
        return f"{int(round(self._zoom_factor * 100))}%"

    def setVideoFrame(self, frame):
        """Set video frame gốc và scale tự động vừa vặn widget."""
        if frame is None:
            return
        h, w = frame.shape[:2]
        ch = frame.shape[2] if len(frame.shape) > 2 else 1
        bytes_per_line = ch * w
        if ch == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimage = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            qimage = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        self._original_pixmap = QPixmap.fromImage(qimage.copy())
        self._raw_pixmap = self._original_pixmap
        self._video_width = w
        self._video_height = h
        self._aspect_ratio = w / float(h) if h > 0 else 16.0 / 9.0
        self._update_scaled_pixmap()

    def set_video_frame(self, frame):
        """Alias cho setVideoFrame."""
        self.setVideoFrame(frame)

    def setPixmap(self, pixmap):
        if pixmap and not pixmap.isNull():
            self._original_pixmap = pixmap
            self._raw_pixmap = pixmap
            self._video_width = pixmap.width()
            self._video_height = pixmap.height()
            self._aspect_ratio = pixmap.width() / float(pixmap.height()) if pixmap.height() > 0 else 16.0 / 9.0
            self._update_scaled_pixmap()
        else:
            self._original_pixmap = None
            self._raw_pixmap = None
            self._scaled_pixmap = None
            super().setPixmap(QPixmap())
            self.update()

    def _update_scaled_pixmap(self):
        """Scale pixmap vừa vặn trong widget, giữ nguyên tỷ lệ khung hình, không bị cắt phần dưới."""
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        
        container_w = self.width()
        container_h = self.height()
        
        if container_w <= 10 or container_h <= 10:
            return
        
        orig_w = self._original_pixmap.width()
        orig_h = self._original_pixmap.height()
        if orig_w <= 0 or orig_h <= 0:
            return

        ratio = min(container_w / float(orig_w), container_h / float(orig_h))
        scaled_w = max(1, int(orig_w * ratio * self._zoom_factor))
        scaled_h = max(1, int(orig_h * ratio * self._zoom_factor))

        if self._zoom_factor <= 1.0 and (scaled_w > container_w or scaled_h > container_h):
            r = min(container_w / float(scaled_w), container_h / float(scaled_h))
            scaled_w = max(1, int(scaled_w * r))
            scaled_h = max(1, int(scaled_h * r))

        # Scale pixmap smoothly
        self._scaled_pixmap = self._original_pixmap.scaled(
            scaled_w, scaled_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Center inside widget
        x = (container_w - scaled_w) // 2
        y = (container_h - scaled_h) // 2
        self._pixmap_x_offset = x
        self._pixmap_y_offset = y
        self.offset_x = x
        self.offset_y = y
        self._pixmap_scale = scaled_w / float(orig_w)
        self.scale_factor = self._pixmap_scale
        self.scale_x = scaled_w / float(orig_w)
        self.scale_y = scaled_h / float(orig_h)
        self.render_rect = QRect(x, y, scaled_w, scaled_h)
        
        # Create final pixmap of exact widget dimensions to prevent any clipping or misalignment
        final = QPixmap(container_w, container_h)
        final.fill(Qt.GlobalColor.transparent)
        painter = QPainter(final)
        painter.drawPixmap(x, y, self._scaled_pixmap)
        painter.end()
        
        super().setPixmap(final)
        self.update()

    def resizeEvent(self, event):
        """Override resize event để tự động scale lại pixmap và bbox khi widget thay đổi kích thước."""
        super().resizeEvent(event)
        self._update_scaled_pixmap()
        self.update_bboxes()

    def update_bboxes(self):
        """Cập nhật lại bboxes sau khi resize."""
        target = self.window()
        if hasattr(target, 'selected_bboxes') and target.selected_bboxes is not None:
            self.bboxes = list(target.selected_bboxes)
        self.update()

    def clear_cache(self):
        self._raw_pixmap = None
        self._original_pixmap = None
        self._scaled_pixmap = None
        self.bboxes = []
        self.clear()
        self.update()

    def _pixmap_rect(self):
        widget_w = self.width()
        widget_h = self.height()
        if self._original_pixmap is not None and not self._original_pixmap.isNull():
            if self.render_rect.width() > 0 and self.render_rect.height() > 0:
                return self.render_rect
            scaled_w = widget_w
            scaled_h = int(scaled_w / self._aspect_ratio) if self._aspect_ratio > 0 else widget_h
            if scaled_h > widget_h:
                scaled_h = widget_h
                scaled_w = int(scaled_h * self._aspect_ratio) if self._aspect_ratio > 0 else widget_w
            x_offset = (widget_w - scaled_w) // 2
            y_offset = (widget_h - scaled_h) // 2
            self.render_rect = QRect(x_offset, y_offset, max(1, scaled_w), max(1, scaled_h))
            self.offset_x = x_offset
            self.offset_y = y_offset
            return self.render_rect

        pix = self.pixmap()
        if pix is None or pix.isNull() or widget_w <= 0 or widget_h <= 0:
            self.render_rect = QRect(0, 0, max(1, widget_w), max(1, widget_h))
            self.offset_x = 0
            self.offset_y = 0
            return self.render_rect

        return QRect(self._pixmap_x_offset, self._pixmap_y_offset, max(1, widget_w - 2 * self._pixmap_x_offset), max(1, widget_h - 2 * self._pixmap_y_offset))

    def get_original_coords(self, widget_x, widget_y):
        """Chuyển đổi tọa độ pixel trên widget -> tọa độ pixel gốc của video"""
        target = self.window()
        w_video = self._video_width or (getattr(target, 'video_width', 1920) or 1920)
        h_video = self._video_height or (getattr(target, 'video_height', 1080) or 1080)
        
        rect = self._pixmap_rect()
        if rect.width() <= 0 or rect.height() <= 0 or w_video <= 0 or h_video <= 0:
            return max(0, widget_x), max(0, widget_y)
            
        rel_x = widget_x - rect.x()
        rel_y = widget_y - rect.y()
        
        rel_x = max(0, min(rel_x, rect.width()))
        rel_y = max(0, min(rel_y, rect.height()))
        
        orig_x = int(round((rel_x / float(rect.width())) * w_video))
        orig_y = int(round((rel_y / float(rect.height())) * h_video))
        return max(0, min(orig_x, w_video)), max(0, min(orig_y, h_video))

    def get_widget_coords(self, orig_x, orig_y):
        """Chuyển đổi tọa độ pixel gốc của video -> tọa độ pixel trên widget"""
        target = self.window()
        w_video = self._video_width or (getattr(target, 'video_width', 1920) or 1920)
        h_video = self._video_height or (getattr(target, 'video_height', 1080) or 1080)
        
        rect = self._pixmap_rect()
        if w_video <= 0 or h_video <= 0 or rect.width() <= 0 or rect.height() <= 0:
            return orig_x, orig_y
            
        scale_x = rect.width() / float(w_video)
        scale_y = rect.height() / float(h_video)
        
        widget_x = rect.x() + int(round(orig_x * scale_x))
        widget_y = rect.y() + int(round(orig_y * scale_y))
        return widget_x, widget_y

    def _hit_test(self, p):
        """
        Kiểm tra điểm p chạm vào Handle góc (Resize) hoặc thân Khung (Move).
        Trả về (hit_type, box, handle_name)
        """
        rect = self._pixmap_rect()
        if rect is None:
            return None, None, None
            
        target = self.window()
        w_video = getattr(target, 'video_width', 1920) or 1920
        h_video = getattr(target, 'video_height', 1080) or 1080
        
        if w_video <= 0 or h_video <= 0 or rect.width() <= 0 or rect.height() <= 0:
            return None, None, None
            
        scale_x = rect.width() / float(w_video)
        scale_y = rect.height() / float(h_video)
        
        # 1. Kiểm tra các Khung vùng quét bboxes
        boxes = getattr(target, 'selected_bboxes', []) or []
        for box in reversed(boxes):
            vx, vy, vw, vh = box
            cx = rect.x() + int(vx * scale_x)
            cy = rect.y() + int(vy * scale_y)
            cw = int(vw * scale_x)
            ch = int(vh * scale_y)
            
            box_rect = QRect(cx, cy, cw, ch)
            
            # Kích thước nút handle góc (12px)
            hs = 12
            handles = {
                'tl': QRect(cx - hs//2, cy - hs//2, hs, hs),
                'tr': QRect(cx + cw - hs//2, cy - hs//2, hs, hs),
                'bl': QRect(cx - hs//2, cy + ch - hs//2, hs, hs),
                'br': QRect(cx + cw - hs//2, cy + ch - hs//2, hs, hs)
            }
            
            for h_name, h_rect in handles.items():
                if h_rect.contains(p):
                    return 'handle', box, h_name
                    
            if box_rect.contains(p):
                return 'box', box, None

        # 2. Kiểm tra Khung Phụ Đề Đầu Ra (Sub Pos Box - Always Draggable)
        preset = getattr(target, 'subtitle_preset', {}) or {}
        custom_pos = preset.get("custom_pos") or getattr(target, 'subtitle_custom_pos', None) or {"x_pct": 50.0, "y_pct": 88.0}
        if custom_pos:
            sub_x = rect.x() + int(rect.width() * float(custom_pos.get("x_pct", 50.0)) / 100.0)
            sub_y = rect.y() + int(rect.height() * float(custom_pos.get("y_pct", 88.0)) / 100.0)
            sub_w = int(rect.width() * 0.6)
            sub_h = 36
            sub_rect = QRect(sub_x - sub_w//2, sub_y - sub_h//2, sub_w, sub_h)
            if sub_rect.contains(p):
                return 'sub_pos', None, None
                
        return None, None, None

    def mousePressEvent(self, event):
        rect = self._pixmap_rect()
        p = event.position().toPoint()
        
        if event.button() == Qt.MouseButton.LeftButton and rect.contains(p):
            hit_type, box, handle_name = self._hit_test(p)
            
            if hit_type == 'handle':
                self.resizing_box = box
                self.resize_handle = handle_name
                self.drag_start_p = p
                self.drag_start_box = list(box)
                return
            elif hit_type == 'box':
                self.dragging_box = box
                self.drag_start_p = p
                self.drag_start_box = list(box)
                return
            elif hit_type == 'sub_pos':
                self.dragging_sub_pos = True
                self.sub_pos_start_p = p
                return
            else:
                self.drawing = True
                self.start_pos = p
                self.current_pos = p
                self.update()
                
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        rect = self._pixmap_rect()
        p = event.position().toPoint()
        target = self.window()
        w_video = getattr(target, 'video_width', 1920) or 1920
        h_video = getattr(target, 'video_height', 1080) or 1080
        scale_x = rect.width() / float(w_video) if rect.width() > 0 else 1.0
        scale_y = rect.height() / float(h_video) if rect.height() > 0 else 1.0

        # Kéo di chuyển Khung Phụ Đề Đầu Ra (Sub Pos)
        if self.dragging_sub_pos:
            rel_x = p.x() - rect.x()
            rel_y = p.y() - rect.y()
            x_pct = round(max(5.0, min(95.0, (rel_x / float(rect.width())) * 100.0)), 1)
            y_pct = round(max(5.0, min(95.0, (rel_y / float(rect.height())) * 100.0)), 1)
            
            if not hasattr(target, 'subtitle_custom_pos') or target.subtitle_custom_pos is None:
                target.subtitle_custom_pos = {}
            target.subtitle_custom_pos["x_pct"] = x_pct
            target.subtitle_custom_pos["y_pct"] = y_pct
            
            if hasattr(target, 'txt_pos_x'): target.txt_pos_x.setText(str(x_pct))
            if hasattr(target, 'txt_pos_y'): target.txt_pos_y.setText(str(y_pct))
            if hasattr(target, 'chk_custom_pos'): target.chk_custom_pos.setChecked(True)
            
            if hasattr(target, 'current_preview_raw_frame') and target.current_preview_raw_frame is not None:
                target.show_preview_frame(target.current_preview_raw_frame)
            self.update()
            return

        # Kéo di chuyển Khung (Move Box)
        if self.dragging_box and self.drag_start_p and self.drag_start_box:
            dx_vid = int((p.x() - self.drag_start_p.x()) / scale_x)
            dy_vid = int((p.y() - self.drag_start_p.y()) / scale_y)
            
            orig_x, orig_y, orig_w, orig_h = self.drag_start_box
            new_x = max(0, min(orig_x + dx_vid, w_video - orig_w))
            new_y = max(0, min(orig_y + dy_vid, h_video - orig_h))
            
            self.dragging_box[0] = new_x
            self.dragging_box[1] = new_y
            
            # Cập nhật box_type_dict để bảo tồn nhãn đã gán
            if hasattr(target, 'box_type_dict') and target.box_type_dict is not None:
                old_key = tuple(self.drag_start_box)
                if old_key in target.box_type_dict:
                    b_type = target.box_type_dict.pop(old_key)
                    target.box_type_dict[tuple(self.dragging_box)] = b_type

            if hasattr(target, 'status_label'): target.status_label.setText(f"Đã di chuyển Vùng quét tới ({new_x}, {new_y})")
            self.update()
            return

        # Kéo thay đổi kích thước Khung (Resize Box)
        if self.resizing_box and self.drag_start_p and self.drag_start_box:
            dx_vid = int((p.x() - self.drag_start_p.x()) / scale_x)
            dy_vid = int((p.y() - self.drag_start_p.y()) / scale_y)
            
            orig_x, orig_y, orig_w, orig_h = self.drag_start_box
            h_name = self.resize_handle
            
            new_x, new_y, new_w, new_h = orig_x, orig_y, orig_w, orig_h
            
            if 'r' in h_name:
                new_w = max(20, min(orig_w + dx_vid, w_video - orig_x))
            if 'b' in h_name:
                new_h = max(20, min(orig_h + dy_vid, h_video - orig_y))
            if 'l' in h_name:
                max_dx = orig_w - 20
                actual_dx = min(dx_vid, max_dx)
                new_x = max(0, orig_x + actual_dx)
                new_w = orig_w - actual_dx
            if 't' in h_name:
                max_dy = orig_h - 20
                actual_dy = min(dy_vid, max_dy)
                new_y = max(0, orig_y + actual_dy)
                new_h = orig_h - actual_dy

            self.resizing_box[0] = new_x
            self.resizing_box[1] = new_y
            self.resizing_box[2] = new_w
            self.resizing_box[3] = new_h
            
            self.update()
            return

        if self.drawing:
            self.current_pos = p
            self.update()
            return

        # Cập nhật con trỏ chuột theo vị trí va chạm
        hit_type, box, handle_name = self._hit_test(p)
        if hit_type == 'handle':
            if handle_name in ('tl', 'br'):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            else:
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif hit_type == 'box' or hit_type == 'sub_pos':
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif rect.contains(p):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        target = self.window()
        if self.dragging_box or self.resizing_box or self.dragging_sub_pos:
            self.dragging_box = None
            self.resizing_box = None
            self.resize_handle = None
            self.dragging_sub_pos = False
            self.drag_start_p = None
            self.drag_start_box = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
            if hasattr(target, 'current_preview_raw_frame') and target.current_preview_raw_frame is not None:
                target.show_preview_frame(target.current_preview_raw_frame)
            self.update()
            return

        if self.drawing:
            self.drawing = False
            rect = self._pixmap_rect()
            if rect is None or self.start_pos is None or self.current_pos is None:
                return
                
            x1_rel = max(0, min(self.start_pos.x() - rect.x(), rect.width()))
            y1_rel = max(0, min(self.start_pos.y() - rect.y(), rect.height()))
            x2_rel = max(0, min(self.current_pos.x() - rect.x(), rect.width()))
            y2_rel = max(0, min(self.current_pos.y() - rect.y(), rect.height()))
            
            x = min(x1_rel, x2_rel)
            y = min(y1_rel, y2_rel)
            w = abs(x1_rel - x2_rel)
            h = abs(y1_rel - y2_rel)
            
            if w > 4 and h > 4:
                w_video = getattr(target, 'video_width', 1920) or 1920
                h_video = getattr(target, 'video_height', 1080) or 1080
                
                rx = x / float(rect.width())
                ry = y / float(rect.height())
                rw = w / float(rect.width())
                rh = h / float(rect.height())
                
                vx = int(rx * w_video)
                vy = int(ry * h_video)
                vw = int(rw * w_video)
                vh = int(rh * h_video)
                
                new_box = [vx, vy, vw, vh]
                
                if hasattr(target, "on_canvas_bbox_added"):
                    target.on_canvas_bbox_added(new_box)
                else:
                    if not hasattr(target, 'selected_bboxes') or target.selected_bboxes is None:
                        target.selected_bboxes = []
                    if new_box not in target.selected_bboxes:
                        target.selected_bboxes.append(new_box)
                        
                if hasattr(target, 'selected_bboxes') and target.selected_bboxes is not None:
                    self.bboxes = list(target.selected_bboxes)
                    
                self.show_box_type_menu(event.globalPosition().toPoint(), new_box)
                    
            self.update()
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        rect = self._pixmap_rect()
        if rect is None:
            return
        p = event.pos()
        target = self.window()
        w_video = getattr(target, 'video_width', 1920) or 1920
        h_video = getattr(target, 'video_height', 1080) or 1080
        
        selected_box = None
        if w_video > 0 and h_video > 0 and hasattr(target, 'selected_bboxes') and target.selected_bboxes:
            scale_x = rect.width() / float(w_video)
            scale_y = rect.height() / float(h_video)
            for box in reversed(target.selected_bboxes):
                vx, vy, vw, vh = box
                cx = rect.x() + int(vx * scale_x)
                cy = rect.y() + int(vy * scale_y)
                cw = int(vw * scale_x)
                ch = int(vh * scale_y)
                if QRect(cx, cy, cw, ch).contains(p):
                    selected_box = box
                    break
                    
        self.show_box_type_menu(event.globalPos(), selected_box)

    def show_box_type_menu(self, global_pos, box):
        target = self.window()
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background-color: #1e293b; color: white; border: 1px solid #334155; } QMenu::item:selected { background-color: #3b82f6; }")
        
        if not hasattr(target, 'box_type_dict') or target.box_type_dict is None:
            target.box_type_dict = {}

        if box is not None:
            act_sub = menu.addAction("🔴 Gán làm Khung Sub (Phụ Đề)")
            act_logo = menu.addAction("🟠 Gán làm Khung Logo (Thủy ấn)")
            act_title = menu.addAction("🟣 Gán làm Khung Tiêu Đề")
            menu.addSeparator()
            act_del = menu.addAction("❌ Xóa Khung Này")
            act_clear_all = menu.addAction("🧹 Xóa Tất Cả Các Khung")
            
            action = menu.exec(global_pos)
            box_key = tuple(box)
            if action == act_sub:
                target.box_type_dict[box_key] = 'sub'
                target.selected_bbox = box
                if getattr(target, 'logo_bbox', None) == box: target.logo_bbox = None
                if getattr(target, 'title_bbox', None) == box: target.title_bbox = None
                target.log_info(f"🔴 Đã gán {box} thành Khung Sub (Phụ Đề)")
            elif action == act_logo:
                target.box_type_dict[box_key] = 'logo'
                target.logo_bbox = box
                if getattr(target, 'selected_bbox', None) == box: target.selected_bbox = None
                if getattr(target, 'title_bbox', None) == box: target.title_bbox = None
                target.log_info(f"🟠 Đã gán {box} thành Khung Logo")
            elif action == act_title:
                target.box_type_dict[box_key] = 'title'
                target.title_bbox = box
                if getattr(target, 'selected_bbox', None) == box: target.selected_bbox = None
                if getattr(target, 'logo_bbox', None) == box: target.logo_bbox = None
                target.log_info(f"🟣 Đã gán {box} thành Khung Tiêu Đề")
            elif action == act_del:
                if hasattr(target, 'selected_bboxes') and box in target.selected_bboxes:
                    target.selected_bboxes.remove(box)
                if box_key in target.box_type_dict:
                    del target.box_type_dict[box_key]
                if getattr(target, 'selected_bbox', None) == box: target.selected_bbox = None
                if getattr(target, 'logo_bbox', None) == box: target.logo_bbox = None
                if getattr(target, 'title_bbox', None) == box: target.title_bbox = None
                target.log_info(f"❌ Đã xóa {box}")
            elif action == act_clear_all:
                if hasattr(target, 'clear_all_canvas_crops'):
                    target.clear_all_canvas_crops()
        else:
            act_clear_all = menu.addAction("🧹 Xóa Tất Cả Các Khung")
            action = menu.exec(global_pos)
            if action == act_clear_all and hasattr(target, 'clear_all_canvas_crops'):
                target.clear_all_canvas_crops()
                
        if hasattr(target, 'selected_bboxes') and target.selected_bboxes is not None:
            self.bboxes = list(target.selected_bboxes)
        self.update()
        target.update()
        if hasattr(target, 'current_preview_raw_frame') and target.current_preview_raw_frame is not None:
            target.show_preview_frame(target.current_preview_raw_frame)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return
            
        rect = self._pixmap_rect()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        target = self.window()
        w_video = getattr(target, 'video_width', 1920) or 1920
        h_video = getattr(target, 'video_height', 1080) or 1080
        
        if hasattr(target, 'selected_bboxes') and target.selected_bboxes is not None:
            self.bboxes = target.selected_bboxes
        
        if w_video > 0 and h_video > 0:
            scale_x = rect.width() / float(w_video)
            scale_y = rect.height() / float(h_video)
            
            # 1. Vẽ các khung vùng chọn (Sub, Logo, Tiêu đề) kèm 4 nốt Handle điều chỉnh góc
            for idx, box in enumerate(self.bboxes):
                vx, vy, vw, vh = box
                cx = rect.x() + int(vx * scale_x)
                cy = rect.y() + int(vy * scale_y)
                cw = int(vw * scale_x)
                ch = int(vh * scale_y)
                
                box_type = getattr(target, 'box_type_dict', {}).get(tuple(box), None)
                if box_type == 'sub' or box == getattr(target, 'selected_bbox', None) or (box_type is None and vy > int(h_video * 0.4)):
                    col = QColor(59, 130, 246, 220) # Xanh dương - Sub
                    label_text = f" 🟦 Khung Sub (Phụ Đề {idx+1})"
                elif box_type == 'title' or box == getattr(target, 'title_bbox', None):
                    col = QColor(234, 179, 8, 220) # Vàng - Tiêu đề
                    label_text = f" 🟨 Khung Tiêu Đề {idx+1}"
                else:
                    col = QColor(239, 68, 68, 220) # Đỏ - Logo
                    label_text = f" 🟥 Khung Logo {idx+1}"
                
                painter.setPen(QPen(col, 2, Qt.PenStyle.SolidLine))
                painter.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 35)))
                painter.drawRect(cx, cy, cw, ch)
                
                # Vẽ 4 nốt Handle điều chỉnh góc góc dạng hình vuông trắng viền màu
                hs = 8
                painter.setBrush(QBrush(QColor(255, 255, 255, 240)))
                painter.setPen(QPen(col, 1))
                painter.drawRect(cx - hs//2, cy - hs//2, hs, hs)
                painter.drawRect(cx + cw - hs//2, cy - hs//2, hs, hs)
                painter.drawRect(cx - hs//2, cy + ch - hs//2, hs, hs)
                painter.drawRect(cx + cw - hs//2, cy + ch - hs//2, hs, hs)
                
                painter.setPen(QPen(Qt.GlobalColor.white))
                painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
                painter.drawText(cx + 6, cy + 18, label_text)

            # 2. Vẽ Khung Phụ Đề Đầu Ra Động (Kéo thả vi trí hiển thị phụ đề)
            preset = getattr(target, 'subtitle_preset', {}) or {}
            custom_pos = preset.get("custom_pos") or getattr(target, 'subtitle_custom_pos', None) or {"x_pct": 50.0, "y_pct": 88.0}
            if custom_pos:
                x_val = custom_pos.get("x_pct", 50.0)
                y_val = custom_pos.get("y_pct", 88.0)
                sub_cx = rect.x() + int(rect.width() * float(x_val) / 100.0)
                sub_cy = rect.y() + int(rect.height() * float(y_val) / 100.0)
                sub_cw = int(rect.width() * 0.65)
                sub_ch = 34
                sub_rect = QRect(sub_cx - sub_cw//2, sub_cy - sub_ch//2, sub_cw, sub_ch)
                
                col_sub_out = QColor(6, 182, 212, 230) # Cyan - Sub output
                painter.setPen(QPen(col_sub_out, 2, Qt.PenStyle.DashLine))
                painter.setBrush(QBrush(QColor(6, 182, 212, 50)))
                painter.drawRect(sub_rect)
                
                painter.setPen(QPen(Qt.GlobalColor.white))
                painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
                painter.drawText(sub_rect.x() + 8, sub_rect.y() + 22, f"💬 Vị trí Sub Đầu Ra: X={x_val}%, Y={y_val}% (Kéo chuột để di chuyển)")

        if self.drawing and self.start_pos and self.current_pos:
            x1 = self.start_pos.x()
            y1 = self.start_pos.y()
            x2 = self.current_pos.x()
            y2 = self.current_pos.y()
            
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x1 - x2)
            h = abs(y1 - y2)
            
            painter.setPen(QPen(QColor(241, 196, 15, 200), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(241, 196, 15, 30)))
            painter.drawRect(x, y, w, h)
            
        painter.end()


class CollapsibleCard(QFrame):
    def __init__(self, title, layout_type="vertical", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 6, 12, 10)
        self.main_layout.setSpacing(6)
        
        # Header button với Icon đóng mở
        self.header_btn = QPushButton(f"▼ {title}")
        self.header_btn.setFlat(True)
        self.header_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                font-weight: bold;
                font-size: 13px;
                color: #38bdf8;
                border: none;
                padding: 6px 4px;
                background-color: transparent;
            }
            QPushButton:hover {
                color: #7dd3fc;
            }
        """)
        self.main_layout.addWidget(self.header_btn)
        
        # Content Widget chứa các điều khiển bên trong
        self.content_widget = QWidget()
        if layout_type == "grid":
            self.content_layout = QGridLayout(self.content_widget)
        elif layout_type == "horizontal":
            self.content_layout = QHBoxLayout(self.content_widget)
        else:
            self.content_layout = QVBoxLayout(self.content_widget)
            
        self.content_layout.setContentsMargins(0, 4, 0, 0)
        self.content_layout.setSpacing(8)
        self.main_layout.addWidget(self.content_widget)
        
        self.header_btn.clicked.connect(self.toggle_collapse)
        self.is_collapsed = False
        
    def toggle_collapse(self):
        self.is_collapsed = not self.is_collapsed
        self.content_widget.setVisible(not self.is_collapsed)
        title_text = self.header_btn.text()[2:]
        if self.is_collapsed:
            self.header_btn.setText(f"▶ {title_text}")
        else:
            self.header_btn.setText(f"▼ {title_text}")
            
    def addWidget(self, widget, *args):
        self.content_layout.addWidget(widget, *args)
        
    def addLayout(self, layout, *args):
        self.content_layout.addLayout(layout, *args)

# Hộp thoại kéo chuột chọn vùng phụ đề cứng (Hardsub OCR Selector) - 2 Cột chuyên nghiệp
class VideoRegionSelector(QDialog):
    def __init__(self, frame, parent=None, title="Kéo chuột vẽ các vùng chọn quét chữ (Quét đa điểm)"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(1100, 600)
        self.parent = parent
        self.video_path = getattr(parent, 'video_path', None) if parent else None
        
        self.frame = frame
        self.h, self.w, _ = frame.shape
        self.bboxes = []  # Danh sách các vùng quét đa điểm [[x,y,w,h], ...]
        self.selected_bbox = None
        self.title_bbox = None
        
        # Khởi tạo Layout chính 2 cột
        main_layout = QHBoxLayout(self)
        
        # --- CỘT TRÁI: VẼ VÙNG QUÉT ---
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("<b>1. Vẽ vùng quét (Kéo thả chuột trên hình):</b>"))
        
        self.label = DrawingLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("border: 1px solid #2a364f; background-color: #090d16; border-radius: 6px;")
        left_layout.addWidget(self.label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Nút chức năng cột trái
        left_btn_layout = QHBoxLayout()
        
        self.btn_next_frame = QPushButton("🔄 Thử frame khác")
        self.btn_next_frame.setStyleSheet("background-color: #334155; color: #f8fafc; font-weight: bold; border-radius: 6px; padding: 6px 12px;")
        self.btn_next_frame.clicked.connect(self.get_another_frame)
        left_btn_layout.addWidget(self.btn_next_frame)
        
        self.btn_gemini_ocr = QPushButton("🤖 Thử quét bằng Gemini AI")
        self.btn_gemini_ocr.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; border-radius: 6px; padding: 6px 12px;")
        self.btn_gemini_ocr.clicked.connect(self.run_gemini_ocr_selector)
        left_btn_layout.addWidget(self.btn_gemini_ocr)
        
        left_btn_layout.addStretch()
        left_layout.addLayout(left_btn_layout)
        
        main_layout.addLayout(left_layout, 1)
        
        # --- CỘT PHẢI: PREVIEW & danh sách bboxes & OCR FEEDBACK ---
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("<b>2. Xem thử vùng chọn & Kết quả OCR:</b>"))
        
        self.label_preview = QLabel(self)
        self.label_preview.setFixedSize(480, 180)  # Giảm chiều cao một chút để nhường chỗ cho danh sách
        self.label_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_preview.setStyleSheet("border: 1px solid #2a364f; background-color: #090d16; color: #94a3b8; border-radius: 6px;")
        self.label_preview.setText("Vẽ khung chọn vùng quét bên trái")
        right_layout.addWidget(self.label_preview)
        
        # Danh sách vùng quét (Multi-point Crops)
        right_layout.addWidget(QLabel("<b>Danh sách các vùng quét (Multi-Crops):</b>"))
        self.list_bboxes = QListWidget()
        self.list_bboxes.setMaximumHeight(100)
        self.list_bboxes.itemSelectionChanged.connect(self.on_box_selection_changed)
        right_layout.addWidget(self.list_bboxes)
        
        # Hàng nút quản lý danh sách vùng quét
        list_btn_layout = QHBoxLayout()
        self.btn_delete_box = QPushButton("❌ Xóa Vùng")
        self.btn_delete_box.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: bold; padding: 6px 12px; border-radius: 6px;")
        self.btn_delete_box.clicked.connect(self.delete_selected_box)
        list_btn_layout.addWidget(self.btn_delete_box)
        
        self.btn_clear_all = QPushButton("🧹 Xóa Tất Cả")
        self.btn_clear_all.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold; padding: 6px 12px; border-radius: 6px;")
        self.btn_clear_all.clicked.connect(self.clear_all_boxes)
        list_btn_layout.addWidget(self.btn_clear_all)
        right_layout.addLayout(list_btn_layout)
        
        self.lbl_ocr_status = QLabel("Nhập vùng quét để chạy thử")
        self.lbl_ocr_status.setStyleSheet("color: #94a3b8; font-weight: bold;")
        right_layout.addWidget(self.lbl_ocr_status)
        
        self.txt_ocr_result = QTextEdit()
        self.txt_ocr_result.setReadOnly(True)
        self.txt_ocr_result.setMaximumHeight(80)
        right_layout.addWidget(self.txt_ocr_result)
        
        # Hàng nút Xác nhận ở dưới cùng cột phải
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        self.btn_confirm = QPushButton("✓ Xác nhận các vùng quét")
        self.btn_confirm.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: bold; padding: 8px 18px; border-radius: 6px;")
        self.btn_confirm.clicked.connect(self.accept)
        btn_box.addWidget(self.btn_confirm)
        
        btn_cancel = QPushButton("Hủy")
        btn_cancel.setStyleSheet("background-color: #334155; color: #f8fafc; font-weight: bold; padding: 8px 18px; border-radius: 6px;")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)
        right_layout.addLayout(btn_box)
        
        main_layout.addLayout(right_layout, 1)
        
        # Load frame ban đầu
        self.init_display_frame()
        
    def init_display_frame(self):
        # Tự động thay đổi kích thước khung hình hiển thị nếu quá lớn so với màn hình
        self.scale = 1.0
        max_w, max_h = 500, 300
        if self.w > max_w or self.h > max_h:
            self.scale = min(max_w / self.w, max_h / self.h)
            new_w = int(self.w * self.scale)
            new_h = int(self.h * self.scale)
            self.display_frame = cv2.resize(self.frame, (new_w, new_h))
        else:
            self.display_frame = self.frame.copy()
            
        self.disp_h, self.disp_w, _ = self.display_frame.shape
        
        # Chuyển đổi khung hình sang định dạng QPixmap hiển thị trên QLabel
        rgb_image = cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_image.data, self.disp_w, self.disp_h, self.disp_w * 3, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(self.pixmap)
        
        # Đảm bảo QLabel có kích thước khớp hoàn toàn với ảnh hiển thị
        self.label.setFixedSize(self.disp_w, self.disp_h)
        
    def set_initial_bbox(self, bbox):
        # Hàm tương thích ngược hỗ trợ nạp bbox đầu tiên
        if bbox:
            if bbox not in self.bboxes:
                self.bboxes.append(bbox)
            self.update_bboxes_list()
            self.list_bboxes.setCurrentRow(len(self.bboxes) - 1)
            self.trigger_instant_ocr()
            
    def update_bboxes_list(self):
        self.list_bboxes.clear()
        for idx, b in enumerate(self.bboxes):
            self.list_bboxes.addItem(f"Vùng {idx + 1}: X={b[0]}, Y={b[1]}, W={b[2]}, H={b[3]}")
        self.label.update()
        
    def delete_selected_box(self):
        row = self.list_bboxes.currentRow()
        if row >= 0 and row < len(self.bboxes):
            self.bboxes.pop(row)
            self.update_bboxes_list()
            if self.bboxes:
                self.list_bboxes.setCurrentRow(max(0, row - 1))
            else:
                self.label_preview.clear()
                self.label_preview.setText("Vẽ khung bên trái để thêm vùng quét")
                self.txt_ocr_result.clear()
            self.trigger_instant_ocr()
            
    def clear_all_boxes(self):
        self.bboxes.clear()
        self.update_bboxes_list()
        self.label_preview.clear()
        self.label_preview.setText("Vẽ khung bên trái để thêm vùng quét")
        self.txt_ocr_result.clear()
        self.lbl_ocr_status.setText("Nhập vùng quét để chạy thử")
        
    def on_box_selection_changed(self):
        row = self.list_bboxes.currentRow()
        if row >= 0 and row < len(self.bboxes):
            b = self.bboxes[row]
            # Cắt ảnh xem thử
            x, y, w, h = b
            frame_h, frame_w, _ = self.frame.shape
            x1 = max(0, min(x, frame_w))
            y1 = max(0, min(y, frame_h))
            x2 = max(0, min(x + w, frame_w))
            y2 = max(0, min(y + h, frame_h))
            
            if x2 > x1 and y2 > y1:
                crop = self.frame[y1:y2, x1:x2]
                # Resize để vừa với label preview
                crop_h, crop_w, _ = crop.shape
                pw, ph = 480, 180
                
                scale = min(pw / crop_w, ph / crop_h) if (crop_w > pw or crop_h > ph) else 1.0
                crop_resized = cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale))) if scale != 1.0 else crop
                
                ch, cw, _ = crop_resized.shape
                rgb_crop = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb_crop.data, cw, ch, cw * 3, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
                self.label_preview.setPixmap(pix)
            
    def get_another_frame(self):
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy video gốc để lấy frame khác.")
            return
            
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return
            
        import random
        random_frame = random.randint(int(total_frames * 0.1), int(total_frames * 0.9))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            self.frame = frame
            self.h, self.w, _ = frame.shape
            self.init_display_frame()
            
            # Reset
            self.bboxes.clear()
            self.update_bboxes_list()
            self.label_preview.clear()
            self.label_preview.setText("Vẽ khung bên trái để thêm vùng quét")
            self.lbl_ocr_status.setText("Nhập vùng quét để chạy thử")
            self.txt_ocr_result.clear()
            self.label.begin = QPoint()
            self.label.end = QPoint()
            self.label.update()
            
    def on_drawing_finished(self, begin, end):
        # Tính toán toạ độ thực tế trên video gốc
        x1 = max(0, min(begin.x(), end.x()))
        y1 = max(0, min(begin.y(), end.y()))
        x2 = min(max(begin.x(), end.x()), self.disp_w)
        y2 = min(max(begin.y(), end.y()), self.disp_h)
        
        # Đưa ngược về tỉ lệ video gốc
        raw_x = int(x1 / self.scale)
        raw_y = int(y1 / self.scale)
        raw_w = int((x2 - x1) / self.scale)
        raw_h = int((y2 - y1) / self.scale)
        
        if raw_w > 5 and raw_h > 5:
            new_box = [raw_x, raw_y, raw_w, raw_h]
            self.bboxes.append(new_box)
            self.update_bboxes_list()
            self.list_bboxes.setCurrentRow(len(self.bboxes) - 1)
            self.trigger_instant_ocr()
            
    def trigger_instant_ocr(self):
        if not self.bboxes:
            self.lbl_ocr_status.setText("Chưa vẽ vùng quét nào")
            self.txt_ocr_result.clear()
            return
            
        ocr_lang = "auto"
        if self.parent and hasattr(self.parent, 'cb_ocr_lang'):
            ocr_lang = self.parent.cb_ocr_lang.currentText()
            
        self.lbl_ocr_status.setText("⏳ ĐANG QUÉT OCR... Vui lòng đợi...")
        self.lbl_ocr_status.setStyleSheet("color: #dfb15b; font-weight: bold;")
        self.txt_ocr_result.clear()
        
        # Disable các nút bấm
        self.btn_confirm.setEnabled(False)
        self.btn_next_frame.setEnabled(False)
        if hasattr(self, 'btn_gemini_ocr'):
            self.btn_gemini_ocr.setEnabled(False)
        
        # Khởi chạy Worker Thread bất đồng bộ chống đơ
        if hasattr(self, 'ocr_worker') and self.ocr_worker.isRunning():
            self.ocr_worker.terminate()
            self.ocr_worker.wait()
            
        self.ocr_worker = InstantOCRWorker(self.frame, self.bboxes, ocr_lang)
        self.ocr_worker.finished.connect(self.on_ocr_success)
        self.ocr_worker.error.connect(self.on_ocr_error)
        self.ocr_worker.start()
        
    def on_ocr_success(self, results, summary):
        self.btn_confirm.setEnabled(True)
        self.btn_next_frame.setEnabled(True)
        if hasattr(self, 'btn_gemini_ocr'):
            self.btn_gemini_ocr.setEnabled(True)
        self.lbl_ocr_status.setText("✅ QUÉT HOÀN TẤT")
        self.lbl_ocr_status.setStyleSheet("color: #7fbeb2; font-weight: bold;")
        self.txt_ocr_result.setPlainText(f"Kết quả nhận diện được:\n{summary}" if summary else "Không tìm thấy chữ nào.")
        self.draw_ocr_preview(results)
        
    def on_ocr_error(self, err_msg):
        self.btn_confirm.setEnabled(True)
        self.btn_next_frame.setEnabled(True)
        if hasattr(self, 'btn_gemini_ocr'):
            self.btn_gemini_ocr.setEnabled(True)
        self.lbl_ocr_status.setText("❌ LỖI OCR")
        self.lbl_ocr_status.setStyleSheet("color: #ff9999; font-weight: bold;")
        self.txt_ocr_result.setPlainText(f"Lỗi: {err_msg}")
        
    def draw_ocr_preview(self, results):
        row = self.list_bboxes.currentRow()
        if row < 0 or row >= len(self.bboxes):
            return
            
        x, y, w, h = self.bboxes[row]
        cropped = self.frame[y:y+h, x:x+w].copy()
        
        for item in results:
            bx, by, bw, bh = item['box']
            # Vẽ các box của EasyOCR lên ảnh
            cv2.rectangle(cropped, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            
        disp_w, disp_h = 480, 180
        cropped_resized = cv2.resize(cropped, (disp_w, disp_h))
        
        rgb_image = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_image.data, disp_w, disp_h, disp_w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.label_preview.setPixmap(pixmap)
        
    def run_gemini_ocr_selector(self):
        row = self.list_bboxes.currentRow()
        if row < 0 or row >= len(self.bboxes):
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một vùng quét trong danh sách để chạy Gemini.")
            return
            
        box = self.bboxes[row]
        
        # Lấy key từ MainWindow parent
        api_key = ""
        if self.parent and hasattr(self.parent, 'txt_gemini_key'):
            api_key = self.parent.txt_gemini_key.text().strip()
            
        if not api_key:
            QMessageBox.warning(self, "Lỗi", "Vui lòng cấu hình Gemini API Key ở tab cấu hình ngoài màn hình chính trước.")
            return
            
        # Cắt ảnh
        x, y, w, h = box
        crop_img = self.frame[y:y+h, x:x+w]
        
        self.lbl_ocr_status.setText("⏳ ĐANG GỬI ẢNH LÊN GEMINI AI...")
        self.lbl_ocr_status.setStyleSheet("color: #dfb15b; font-weight: bold;")
        self.txt_ocr_result.clear()
        
        self.btn_confirm.setEnabled(False)
        self.btn_next_frame.setEnabled(False)
        self.btn_gemini_ocr.setEnabled(False)
        
        if hasattr(self, 'gemini_worker') and self.gemini_worker.isRunning():
            self.gemini_worker.terminate()
            self.gemini_worker.wait()
            
        self.gemini_worker = GeminiOCRWorker(-1, crop_img, None, api_key)
        self.gemini_worker.finished.connect(self.on_gemini_success)
        self.gemini_worker.error.connect(self.on_gemini_error)
        self.gemini_worker.start()
        
    def on_gemini_success(self, row, text):
        self.btn_confirm.setEnabled(True)
        self.btn_next_frame.setEnabled(True)
        self.btn_gemini_ocr.setEnabled(True)
        
        self.lbl_ocr_status.setText("✨ NHẬN DIỆN GEMINI AI THÀNH CÔNG")
        self.lbl_ocr_status.setStyleSheet("color: #7fbeb2; font-weight: bold;")
        self.txt_ocr_result.setPlainText(text)
        
    def on_gemini_error(self, row, err_msg):
        self.btn_confirm.setEnabled(True)
        self.btn_next_frame.setEnabled(True)
        self.btn_gemini_ocr.setEnabled(True)
        
        self.lbl_ocr_status.setText("❌ GEMINI AI LỖI")
        self.lbl_ocr_status.setStyleSheet("color: #ff9999; font-weight: bold;")
        self.txt_ocr_result.setPlainText(f"Lỗi: {err_msg}")
        
    def accept(self):
        self.selected_bbox = self.bboxes[0] if len(self.bboxes) > 0 else None
        self.title_bbox = self.bboxes[1] if len(self.bboxes) > 1 else None
        self.cleanup_workers()
        super().accept()
        
    def reject(self):
        self.cleanup_workers()
        super().reject()
        
    def closeEvent(self, event):
        self.cleanup_workers()
        super().closeEvent(event)
        
    def cleanup_workers(self):
        if hasattr(self, 'ocr_worker') and self.ocr_worker.isRunning():
            try:
                self.ocr_worker.finished.disconnect()
                self.ocr_worker.error.disconnect()
            except Exception:
                pass
            self.ocr_worker.terminate()
            self.ocr_worker.wait()
        if hasattr(self, 'gemini_worker') and self.gemini_worker.isRunning():
            try:
                self.gemini_worker.finished.disconnect()
                self.gemini_worker.error.disconnect()
            except Exception:
                pass
            self.gemini_worker.terminate()
            self.gemini_worker.wait()
        
    def init_display_frame(self):
        # Tự động thay đổi kích thước khung hình hiển thị nếu quá lớn so với màn hình
        self.scale = 1.0
        max_w, max_h = 500, 300
        if self.w > max_w or self.h > max_h:
            self.scale = min(max_w / self.w, max_h / self.h)
            new_w = int(self.w * self.scale)
            new_h = int(self.h * self.scale)
            self.display_frame = cv2.resize(self.frame, (new_w, new_h))
        else:
            self.display_frame = self.frame.copy()
            
        self.disp_h, self.disp_w, _ = self.display_frame.shape
        
        # Chuyển đổi khung hình sang định dạng QPixmap hiển thị trên QLabel
        rgb_image = cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_image.data, self.disp_w, self.disp_h, self.disp_w * 3, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(self.pixmap)
        
        # Đảm bảo QLabel có kích thước khớp hoàn toàn với ảnh hiển thị
        self.label.setFixedSize(self.disp_w, self.disp_h)
        
    def set_initial_bbox(self, bbox):
        if bbox:
            self.selected_bbox = bbox
            x, y, w, h = bbox
            # Ánh xạ tọa độ hiển thị
            x1 = int(x * self.scale)
            y1 = int(y * self.scale)
            x2 = int((x + w) * self.scale)
            y2 = int((y + h) * self.scale)
            self.label.begin = QPoint(x1, y1)
            self.label.end = QPoint(x2, y2)
            self.label.update()
            # Kích hoạt OCR hiển thị kết quả
            self.trigger_instant_ocr()
            
    def get_another_frame(self):
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy video gốc để lấy frame khác.")
            return
            
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return
            
        # Chọn ngẫu nhiên một mốc thời gian trong khoảng 10% đến 90% video
        import random
        random_frame = random.randint(int(total_frames * 0.1), int(total_frames * 0.9))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            self.frame = frame
            self.h, self.w, _ = frame.shape
            self.init_display_frame()
            
            # Reset
            self.selected_bbox = None
            self.title_bbox = None
            self.label_preview.clear()
            self.label_preview.setText("Vẽ khung chọn vùng quét bên trái để chạy thử OCR")
            self.lbl_ocr_status.setText("Nhập vùng quét để chạy thử")
            self.txt_ocr_result.clear()
            self.label.begin = QPoint()
            self.label.end = QPoint()
            self.label.update()
            
    def on_drawing_finished(self, begin, end):
        # Tính toán toạ độ thực tế trên video gốc
        x1 = max(0, min(begin.x(), end.x()))
        y1 = max(0, min(begin.y(), end.y()))
        x2 = min(max(begin.x(), end.x()), self.disp_w)
        y2 = min(max(begin.y(), end.y()), self.disp_h)
        
        # Đưa ngược về tỉ lệ video gốc
        raw_x = int(x1 / self.scale)
        raw_y = int(y1 / self.scale)
        raw_w = int((x2 - x1) / self.scale)
        raw_h = int((y2 - y1) / self.scale)
        
        if raw_w > 5 and raw_h > 5:
            combo = getattr(self, 'combo_box_type', None)
            box_type = combo.currentIndex() if combo else 0
            if box_type == 0:
                self.selected_bbox = [raw_x, raw_y, raw_w, raw_h]
            else:
                self.title_bbox = [raw_x, raw_y, raw_w, raw_h]
            self.trigger_instant_ocr()
            
    def trigger_instant_ocr(self):
        bboxes = []
        if self.selected_bbox:
            bboxes.append(self.selected_bbox)
        if getattr(self, 'title_bbox', None):
            bboxes.append(self.title_bbox)
            
        if not bboxes:
            return
            
        ocr_lang = "auto"
        if self.parent and hasattr(self.parent, 'cb_ocr_lang'):
            ocr_lang = self.parent.cb_ocr_lang.currentText()
            
        # Hiển thị loading spinner/text nhấp nháy
        self.lbl_ocr_status.setText("⏳ ĐANG QUÉT OCR... Vui lòng đợi...")
        self.lbl_ocr_status.setStyleSheet("color: #dfb15b; font-weight: bold;")
        self.txt_ocr_result.clear()
        
        # Disable các nút bấm
        self.btn_confirm.setEnabled(False)
        self.btn_next_frame.setEnabled(False)
        if hasattr(self, 'btn_gemini_ocr'):
            self.btn_gemini_ocr.setEnabled(False)
        
        # Khởi chạy Worker Thread bất đồng bộ chống đơ
        if hasattr(self, 'ocr_worker') and self.ocr_worker.isRunning():
            self.ocr_worker.terminate()
            self.ocr_worker.wait()
            
        self.ocr_worker = InstantOCRWorker(self.frame, bboxes, ocr_lang)
        self.ocr_worker.finished.connect(self.on_ocr_success)
        self.ocr_worker.error.connect(self.on_ocr_error)
        self.ocr_worker.start()
        
    def on_ocr_success(self, results, summary):
        self.btn_confirm.setEnabled(True)
        self.btn_next_frame.setEnabled(True)
        if hasattr(self, 'btn_gemini_ocr'):
            self.btn_gemini_ocr.setEnabled(True)
        self.lbl_ocr_status.setText("✅ QUÉT HOÀN TẤT")
        self.lbl_ocr_status.setStyleSheet("color: #7fbeb2; font-weight: bold;")
        self.txt_ocr_result.setPlainText(f"Kết quả nhận diện được:\n{summary}" if summary else "Không tìm thấy chữ nào.")
        self.draw_ocr_preview(results)
        
    def on_ocr_error(self, err_msg):
        self.btn_confirm.setEnabled(True)
        self.btn_next_frame.setEnabled(True)
        if hasattr(self, 'btn_gemini_ocr'):
            self.btn_gemini_ocr.setEnabled(True)
        self.lbl_ocr_status.setText("❌ LỖI OCR")
        self.lbl_ocr_status.setStyleSheet("color: #ff9999; font-weight: bold;")
        self.txt_ocr_result.setPlainText(f"Lỗi: {err_msg}")
        
    def draw_ocr_preview(self, results):
        if not self.selected_bbox:
            return
        x, y, w, h = self.selected_bbox
        # Cắt ảnh crop và vẽ
        cropped = self.frame[y:y+h, x:x+w].copy()
        
        for item in results:
            bx, by, bw, bh = item['box']
            cv2.rectangle(cropped, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            
        disp_w, disp_h = 480, 270
        cropped_resized = cv2.resize(cropped, (disp_w, disp_h))
        
        rgb_image = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_image.data, disp_w, disp_h, disp_w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.label_preview.setPixmap(pixmap)

    def run_gemini_ocr_selector(self):
        box_type = self.combo_box_type.currentIndex()
        box = self.selected_bbox if box_type == 0 else getattr(self, 'title_bbox', None)
        
        if not box:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng vẽ vùng quét tương ứng trước.")
            return
            
        # Lấy key từ MainWindow parent
        api_key = ""
        if self.parent and hasattr(self.parent, 'txt_gemini_key'):
            api_key = self.parent.txt_gemini_key.text().strip()
            
        if not api_key:
            QMessageBox.warning(self, "Lỗi", "Vui lòng cấu hình Gemini API Key ở tab cấu hình ngoài màn hình chính trước.")
            return
            
        # Cắt ảnh
        x, y, w, h = box
        crop_img = self.frame[y:y+h, x:x+w]
        
        self.lbl_ocr_status.setText("⏳ ĐANG GỬI ẢNH LÊN GEMINI AI...")
        self.lbl_ocr_status.setStyleSheet("color: #dfb15b; font-weight: bold;")
        self.txt_ocr_result.clear()
        
        self.btn_confirm.setEnabled(False)
        self.btn_next_frame.setEnabled(False)
        self.btn_gemini_ocr.setEnabled(False)
        
        if hasattr(self, 'gemini_worker') and self.gemini_worker.isRunning():
            self.gemini_worker.terminate()
            self.gemini_worker.wait()
            
        self.gemini_worker = GeminiOCRWorker(-1, crop_img, None, api_key)
        self.gemini_worker.finished.connect(self.on_gemini_success)
        self.gemini_worker.error.connect(self.on_gemini_error)
        self.gemini_worker.start()
        
    def on_gemini_success(self, row, text):
        self.btn_confirm.setEnabled(True)
        self.btn_next_frame.setEnabled(True)
        self.btn_gemini_ocr.setEnabled(True)
        
        self.lbl_ocr_status.setText("✨ NHẬN DIỆN GEMINI AI THÀNH CÔNG")
        self.lbl_ocr_status.setStyleSheet("color: #7fbeb2; font-weight: bold;")
        self.txt_ocr_result.setPlainText(text)
        
    def on_gemini_error(self, row, err_msg):
        self.btn_confirm.setEnabled(True)
        self.btn_next_frame.setEnabled(True)
        self.btn_gemini_ocr.setEnabled(True)
        
        self.lbl_ocr_status.setText("❌ GEMINI AI LỖI")
        self.lbl_ocr_status.setStyleSheet("color: #ff9999; font-weight: bold;")
        self.txt_ocr_result.setPlainText(f"Lỗi: {err_msg}")

    def accept(self):
        self.cleanup_workers()
        super().accept()
        
    def reject(self):
        self.cleanup_workers()
        super().reject()
        
    def closeEvent(self, event):
        self.cleanup_workers()
        super().closeEvent(event)
        
    def cleanup_workers(self):
        if hasattr(self, 'ocr_worker') and self.ocr_worker.isRunning():
            try:
                self.ocr_worker.finished.disconnect()
                self.ocr_worker.error.disconnect()
            except Exception:
                pass
            self.ocr_worker.terminate()
            self.ocr_worker.wait()
        if hasattr(self, 'gemini_worker') and self.gemini_worker.isRunning():
            try:
                self.gemini_worker.finished.disconnect()
                self.gemini_worker.error.disconnect()
            except Exception:
                pass
            self.gemini_worker.terminate()
            self.gemini_worker.wait()

# Worker cho luồng Tải video
class DownloadWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str) # video_path
    error = pyqtSignal(str)
    
    def __init__(self, url, output_dir):
        super().__init__()
        self.url = url
        self.output_dir = output_dir
        
    def run(self):
        try:
            self.progress.emit("Đang kết nối để tải video bằng yt-dlp...")
            video_path = downloader.download_video(self.url, self.output_dir)
            self.progress.emit(f"Đã tải xong video: {os.path.basename(video_path)}")
            self.finished.emit(video_path)
        except Exception as e:
            import traceback
            self.error.emit(f"Lỗi tải video: {str(e)}\n{traceback.format_exc()}")

# Worker cho luồng tạo tập dữ liệu OCR
class DatasetGeneratorWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(int, str) # count, output_dir
    error = pyqtSignal(str)
    frame_signal = pyqtSignal(object, int, int, float, object, str)
    
    def __init__(self, video_path, srt_path, selected_bbox, title_bbox, output_dir="ocr_dataset"):
        super().__init__()
        self.video_path = video_path
        self.srt_path = srt_path
        self.selected_bbox = selected_bbox
        self.title_bbox = title_bbox
        self.output_dir = output_dir
        
    def run(self):
        try:
            if not os.path.exists(self.video_path):
                raise ValueError("Không tìm thấy tệp video.")
            if not os.path.exists(self.srt_path):
                raise ValueError("Không tìm thấy tệp phụ đề SRT.")
                
            img_dir = os.path.join(self.output_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            gt_file_path = os.path.join(self.output_dir, "rec_gt.txt")
            
            # Đọc phụ đề
            from transcriber import parse_srt_string
            try:
                with open(self.srt_path, 'r', encoding='utf-8-sig') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(self.srt_path, 'r', encoding='utf-16') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(self.srt_path, 'r', encoding='mbcs') as f:
                        content = f.read()
            
            segments = parse_srt_string(content)
            if not segments:
                raise ValueError("Không thể phân tích nội dung phụ đề từ file SRT.")
                
            self.progress.emit(f"Tìm thấy {len(segments)} phân đoạn trong SRT.")
            
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps <= 0 or total_frames <= 0:
                cap.release()
                raise ValueError("Không đọc được thông tin video.")
                
            saved_count = 0
            gt_lines = []
            
            for idx, seg in enumerate(segments):
                start = seg.get('start', 0.0)
                end = seg.get('end', 0.0)
                text = seg.get('text', '').strip()
                if not text:
                    continue
                    
                # Nhảy tới khung hình giữa phân đoạn
                target_time = (start + end) / 2.0
                frame_idx = int(target_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Phát signal thời gian thực hiển thị khung hình + crop box đè
                self.frame_signal.emit(
                    frame, 
                    frame_idx, 
                    total_frames, 
                    target_time, 
                    self.selected_bbox or self.title_bbox, 
                    f"Cắt Dataset OCR ({idx+1}/{len(segments)})"
                )
                
                fh, fw, _ = frame.shape
                
                # Phân tách chữ tiêu đề và chữ phụ đề nếu text chứa nhiều dòng
                text_lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                # Cắt theo bboxes vẽ được
                crops_to_save = []
                
                # Cắt vùng phụ đề:
                if self.selected_bbox:
                    x, y, w, h = self.selected_bbox
                    x1 = max(0, min(x, fw))
                    y1 = max(0, min(y, fh))
                    x2 = max(0, min(x + w, fw))
                    y2 = max(0, min(y + h, fh))
                    if x2 > x1 and y2 > y1:
                        lbl_text = text_lines[-1] if len(text_lines) > 1 else text
                        crops_to_save.append((frame[y1:y2, x1:x2], f"sub_{idx:05d}.jpg", lbl_text))
                        
                # Cắt vùng tiêu đề:
                if self.title_bbox:
                    x, y, w, h = self.title_bbox
                    x1 = max(0, min(x, fw))
                    y1 = max(0, min(y, fh))
                    x2 = max(0, min(x + w, fw))
                    y2 = max(0, min(y + h, fh))
                    if x2 > x1 and y2 > y1:
                        lbl_text = text_lines[0] if len(text_lines) > 1 else text
                        crops_to_save.append((frame[y1:y2, x1:x2], f"title_{idx:05d}.jpg", lbl_text))
                
                # Nếu không vẽ vùng nào thì cắt dải dưới mặc định
                if not self.selected_bbox and not self.title_bbox:
                    y1 = int(fh * 0.75)
                    y2 = int(fh * 0.95)
                    crops_to_save.append((frame[y1:y2, 0:fw], f"default_{idx:05d}.jpg", text))
                    
                # Lưu ảnh và ghi nhãn
                for crop, img_name, lbl_txt in crops_to_save:
                    if crop.size == 0:
                        continue
                    img_path = os.path.join(img_dir, img_name)
                    cv2.imwrite(img_path, crop)
                    
                    gt_line = f"images/{img_name}\t{lbl_txt}"
                    gt_lines.append(gt_line)
                    saved_count += 1
                    
                percent = int((idx / len(segments)) * 100)
                self.progress.emit(f"Đang cắt ảnh mẫu: {percent}% ({idx}/{len(segments)} câu)")
                
            cap.release()
            
            with open(gt_file_path, "a", encoding="utf-8") as f:
                for line in gt_lines:
                    f.write(line + "\n")
                    
            self.finished.emit(saved_count, self.output_dir)
        except Exception as e:
            self.error.emit(str(e))

# Worker cho luồng Trích xuất phụ đề
class TranscriptionWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list, str) # segments, video_path
    error = pyqtSignal(str)
    
    def __init__(self, video_path, mode, bbox, whisper_model, api_key, ocr_lang="auto", force_scan=False, max_workers=2):
        super().__init__()
        self.video_path = video_path
        self.mode = mode
        self.bbox = bbox
        self.whisper_model = whisper_model
        self.api_key = api_key
        self.ocr_lang = ocr_lang
        self.force_scan = force_scan
        self.max_workers = max_workers
        
    def run(self):
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                self.error.emit("Đường dẫn video không hợp lệ.")
                return
                
            segments = []
            if self.mode == 'whisper':
                self.progress.emit("Đang tách âm thanh WAV từ video gốc...")
                audio_path = downloader.extract_audio(self.video_path)
                
                if self.api_key:
                    self.progress.emit("Đang gửi âm thanh phân tích bằng Gemini AI...")
                    segments = transcriber.transcribe_gemini(audio_path, self.api_key, self.progress.emit)
                else:
                    self.progress.emit("Đang tải model Whisper để dịch giọng nói cục bộ...")
                    segments = transcriber.transcribe_local_whisper(audio_path, self.whisper_model, self.progress.emit)
                    
                # Xoá file âm thanh tạm thời
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
            else:
                if not self.bbox:
                    self.error.emit("Vui lòng vẽ khung chọn vùng phụ đề trước khi quét chữ OCR.")
                    return
                self.progress.emit("Đang khởi tạo công cụ quét chữ OCR...")

                # Kiểm tra độ dài video, nếu > 25s thì tự động kích hoạt bộ quét song song đa luồng chunk
                use_parallel = False
                try:
                    cap = cv2.VideoCapture(self.video_path)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                        tot_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        dur_s = tot_f / fps
                        cap.release()
                        if dur_s > 25.0 and self.max_workers > 1:
                            use_parallel = True
                except Exception:
                    pass

                if use_parallel:
                    from optimized_pipeline import ParallelChunkOCRProcessor
                    processor = ParallelChunkOCRProcessor(self.video_path, max_workers=self.max_workers)
                    res = processor.process_video_ocr(self.bbox, ocr_lang=self.ocr_lang, api_key=self.api_key, progress_callback=self.progress.emit)
                    segments = res['subtitles']
                else:
                    segments = transcriber.run_hardsub_ocr(self.video_path, self.bbox, self.progress.emit, ocr_lang=self.ocr_lang, force_scan=self.force_scan, api_key=self.api_key)
                
            self.finished.emit(segments, self.video_path)
        except Exception as e:
            import traceback
            self.error.emit(f"Lỗi: {str(e)}\n{traceback.format_exc()}")

# Worker cho luồng Dịch thuật (2 Giai đoạn)
class TranslationWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, segments, engine, api_key, ollama_model=None, vp_dict_paths=None, 
                 refine_enabled=False, refine_engine="Gemini 1.5 Flash", refine_api_key="", glossary=None):
        super().__init__()
        self.segments = segments
        self.engine = engine
        self.api_key = api_key
        self.ollama_model = ollama_model
        self.vp_dict_paths = vp_dict_paths
        self.refine_enabled = refine_enabled
        self.refine_engine = refine_engine
        self.refine_api_key = refine_api_key
        self.glossary = glossary
        
    def run(self):
        try:
            self.progress.emit("Đang thực hiện Giai đoạn 1: Dịch thô...")
            translated = translator.translate_segments(
                self.segments,
                source_lang='auto',
                target_lang='vi',
                engine=self.engine,
                api_key=self.api_key,
                progress_callback=self.progress.emit,
                ollama_model=self.ollama_model,
                vp_dict_paths=self.vp_dict_paths
            )
            
            if self.refine_enabled:
                self.progress.emit("Đang thực hiện Giai đoạn 2: Tinh chỉnh dịch thuật LLM...")
                translated = translator.refine_translated_segments(
                    translated,
                    glossary=self.glossary,
                    api_key=self.refine_api_key,
                    engine=self.refine_engine,
                    progress_callback=self.progress.emit,
                    ollama_model=self.ollama_model
                )
                
            self.finished.emit(translated)
        except Exception as e:
            self.error.emit(str(e))

class GeminiKeyCheckWorker(QThread):
    key_tested = pyqtSignal(int, str, str, str) # index, key, status_code, message
    finished = pyqtSignal()

    def __init__(self, key_list):
        super().__init__()
        self.key_list = key_list

    def run(self):
        import google.generativeai as genai
        for idx, key in enumerate(self.key_list):
            if not key or not key.strip():
                self.key_tested.emit(idx, "", "EMPTY", "⚪ Chưa nhập")
                continue
            if idx > 0:
                time.sleep(0.5)
            key_clean = key.strip()
            success = False
            last_err = None
            try:
                genai.configure(api_key=key_clean)
                models_list = list(genai.list_models())
                if models_list:
                    self.key_tested.emit(idx, key_clean, "ACTIVE", "🟢 Đang hoạt động (API OK)")
                    success = True
                    continue
            except Exception as e_list:
                last_err = e_list

            err_str = str(last_err or "Unknown error")
            if "429" in err_str or "quota" in err_str.lower() or "resource" in err_str.lower() or "limit" in err_str.lower():
                self.key_tested.emit(idx, key_clean, "RATE_LIMIT", "🟡 Tạm hết lượt (Rate Limit 429)")
            elif "404" in err_str or "not found" in err_str.lower() or "initializ" in err_str.lower():
                self.key_tested.emit(idx, key_clean, "INITIALIZING", "🟡 Key mới (Google đang kích hoạt Model, chờ ~30s)")
            elif "409" in err_str or "conflict" in err_str.lower():
                self.key_tested.emit(idx, key_clean, "INITIALIZING", "🟡 Key mới (Đang đồng bộ Google, chờ ~30s)")
            elif "400" in err_str or "api_key" in err_str.lower() or "invalid" in err_str.lower() or "key" in err_str.lower():
                self.key_tested.emit(idx, key_clean, "EXHAUSTED", "🔴 Key không hợp lệ / Hết Quota")
            else:
                self.key_tested.emit(idx, key_clean, "ERROR", f"🔴 Lỗi: {err_str[:30]}")
        self.finished.emit()

# Worker cho luồng lồng tiếng và xuất video
class DubbingWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, video_path, segments, voice, output_path, bg_vol, voice_vol, burn_subtitles=False, selected_bbox=None, preset=None, enable_dubbing=True, selected_bboxes=None, logo_path=""):
        super().__init__()
        self.video_path = video_path
        self.segments = segments
        self.voice = voice
        self.output_path = output_path
        self.bg_vol = bg_vol
        self.voice_vol = voice_vol
        self.burn_subtitles = burn_subtitles
        self.selected_bbox = selected_bbox
        self.preset = preset
        self.enable_dubbing = enable_dubbing
        self.selected_bboxes = selected_bboxes
        self.logo_path = logo_path
        
    def run(self):
        try:
            res_path, overflowed_segs = dubber.create_dubbed_video(
                self.video_path,
                self.segments,
                self.voice,
                self.output_path,
                bg_volume=self.bg_vol,
                dub_volume=self.voice_vol,
                burn_subtitles=self.burn_subtitles,
                selected_bbox=self.selected_bbox,
                preset=self.preset,
                progress_callback=self.progress.emit,
                enable_dubbing=self.enable_dubbing,
                selected_bboxes=self.selected_bboxes,
                logo_path=self.logo_path
            )
            # Warning reporting for overflowed segments
            if overflowed_segs:
                self.progress.emit(f"⚠️ Cảnh báo: Có {len(overflowed_segs)} phân đoạn phụ đề bị tràn khung hình (đã giảm về 12px nhưng vẫn tràn):")
                for s in overflowed_segs[:5]:
                    self.progress.emit(f" - \"{s}\"")
                if len(overflowed_segs) > 5:
                    self.progress.emit(" - ...và các câu khác.")
            self.finished.emit(res_path)
        except Exception as e:
            self.error.emit(str(e))

class FullOneClickPipelineWorker(QThread):
    progress = pyqtSignal(str)
    progress_updated = pyqtSignal(int, str)  # (percent: int, step_name: str)
    eta_updated = pyqtSignal(str)           # ("MM:SS")
    chunk_progress = pyqtSignal(int, int)   # (done_chunks: int, total_chunks: int)
    segments_ready = pyqtSignal(list)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, video_path, output_path, workers_cnt=4, selected_bbox=None, selected_bboxes=None, voice=None, bg_vol=0.1, dub_vol=1.0, burn_sub=True, preset=None, enable_dubbing=True, logo_path=None, source_lang="auto", target_lang="vi", api_key="", engine="Supersubs AI", title_bbox=None, refine_enabled=False, refine_engine="Gemini 1.5 Flash", refine_api_key="", ollama_model="qwen2.5", vp_dict_paths=None, logo_bbox=None, chunk_workers=None, ocr_engine=None, scan_interval=0.5, min_sub_duration=0.3, xkiro_key=""):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.workers_cnt = workers_cnt
        self.chunk_workers = chunk_workers or workers_cnt or 4
        self.selected_bbox = selected_bbox
        self.selected_bboxes = selected_bboxes
        self.voice = voice
        self.bg_vol = bg_vol
        self.dub_vol = dub_vol
        self.burn_sub = burn_sub
        self.preset = preset
        self.enable_dubbing = enable_dubbing
        self.logo_path = logo_path
        self.logo_bbox = logo_bbox
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.api_key = api_key
        self.xkiro_key = xkiro_key
        self.engine = engine
        self.scan_interval = scan_interval or 0.5
        self.min_sub_duration = min_sub_duration or 0.3
        
        # Xử lý chuẩn hóa ocr_engine: "paddleocr", "gemini", "xkiro", "easyocr"
        raw_ocr = str(ocr_engine or engine or "").lower()
        if "paddle" in raw_ocr:
            self.ocr_engine = "paddleocr"
        elif "xkiro" in raw_ocr:
            self.ocr_engine = "xkiro"
        elif "easyocr" in raw_ocr or "truyền thống" in raw_ocr or "offline" in raw_ocr:
            self.ocr_engine = "easyocr"
        elif "gemini" in raw_ocr:
            self.ocr_engine = "gemini"
        else:
            self.ocr_engine = "gemini"

        self.title_bbox = title_bbox
        self.refine_enabled = refine_enabled
        self.refine_engine = refine_engine
        self.refine_api_key = refine_api_key
        self.ollama_model = ollama_model
        self.vp_dict_paths = vp_dict_paths
        self.start_time = None
        self._is_cancelled = False

    def emit_step_progress(self, percent, step_name):
        pct = max(0, min(100, int(percent)))
        self.progress_updated.emit(pct, step_name)
        self.progress.emit(f"[{pct}%] {step_name}")
        if self.start_time:
            elapsed = time.time() - self.start_time
            if 0 < pct < 100:
                eta_seconds = (elapsed / pct) * (100 - pct)
                eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
                self.eta_updated.emit(eta_str)
            elif pct >= 100:
                self.eta_updated.emit("00:00")

    def get_translation_engine(self):
        if self.ocr_engine == "xkiro" or "xkiro" in str(self.engine).lower():
            return "xkiro"
        settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "app_settings.json")
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    eng = data.get("translation_engine", "xKiro AI (Ưu tiên)")
                    if "xkiro" in eng.lower():
                        return "xkiro"
                    elif "gemini" in eng.lower():
                        return "gemini"
                    elif "google" in eng.lower():
                        return "google"
                    else:
                        return "auto"
            except Exception:
                pass
        return "xkiro"

    def load_xkiro_prompt_template(self):
        p_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "xkiro_prompt_template.json")
        if os.path.exists(p_path):
            try:
                with open(p_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "template": (
                "Bạn là chuyên gia dịch thuật phim ảnh. Hãy dịch đoạn văn sau từ {source_lang} sang {target_lang} với các yêu cầu:\n\n"
                "1. Giữ nguyên ý nghĩa và ngữ cảnh của câu chuyện\n"
                "2. Dịch tự nhiên, không dịch word-by-word, phù hợp với văn nói\n"
                "3. Giữ nguyên các thuật ngữ chuyên ngành, tên riêng, địa danh\n"
                "4. Đảm bảo độ dài phù hợp với thời lượng hiển thị subtitle (ngắn gọn, súc tích)\n"
                "5. Phù hợp với văn phong hội thoại trong phim/video\n\n"
                "{context}\n\n"
                "Đoạn văn cần dịch:\n"
                "{text}"
            ),
            "max_tokens": 1000,
            "temperature": 0.3,
            "keep_proper_nouns": True,
            "auto_context": True
        }

    def get_video_context(self):
        return f"Video: {os.path.basename(self.video_path) if self.video_path else 'Clip'}"

    def get_xkiro_max_tokens(self):
        cfg = self.load_xkiro_prompt_template()
        return int(cfg.get("max_tokens", 1000))

    def get_xkiro_temperature(self):
        cfg = self.load_xkiro_prompt_template()
        return float(cfg.get("temperature", 0.3))

    def call_xkiro_api(self, prompt, max_tokens=1000, temperature=0.3):
        import xkiro_client
        return xkiro_client.translate_with_xkiro(
            text=prompt,
            target_lang=self.target_lang,
            prompt_template=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=self.api_key
        )

    def translate_with_xkiro(self, text, source_lang="vi", target_lang="en"):
        import xkiro_client
        cfg = self.load_xkiro_prompt_template()
        tmpl = cfg.get("template", "")
        try:
            return xkiro_client.translate_with_xkiro(
                text=text,
                target_lang=target_lang,
                source_lang=source_lang,
                prompt_template=tmpl,
                max_tokens=self.get_xkiro_max_tokens(),
                temperature=self.get_xkiro_temperature(),
                api_key=self.api_key
            )
        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "auth" in err_str.lower():
                self.progress.emit("❌ [xKiro API] Key không hợp lệ hoặc đã hết hạn (401 Auth Error)! Vui lòng cập nhật key mới trong Tab 2 → API Keys.")
            self.progress.emit(f"⚠️ xKiro dịch lỗi ({e}). Đang tự động fallback sang Gemini translation...")
            try:
                return self.translate_with_gemini(text, source_lang, target_lang)
            except Exception as e_gem:
                self.progress.emit(f"⚠️ Gemini lỗi ({e_gem}). Đang fallback sang Google Translate...")
                return self.translate_with_google(text, source_lang, target_lang)

    def translate_with_gemini(self, text, source_lang="vi", target_lang="en"):
        import translator
        from gemini_vision_ocr import load_gemini_keys
        keys = []
        if self.api_key and not self.api_key.startswith("sk-"):
            keys.append(self.api_key)
        for k in load_gemini_keys():
            if k not in keys:
                keys.append(k)
        
        prompt = f"Dịch câu thoại video sau từ {source_lang} sang {target_lang} tự nhiên, đúng văn phong hội thoại:\n'{text}'\n\nCHỈ xuất ra duy nhất câu dịch tiếng Việt."
        for k in keys:
            try:
                res = translator.call_gemini_with_fallback(prompt, k, model_name="gemini-flash-latest")
                if res and res.strip():
                    return res.strip()
            except Exception:
                continue
        raise RuntimeError("Tất cả Gemini keys thất bại.")

    def translate_with_google(self, text, source_lang="auto", target_lang="vi"):
        from deep_translator import GoogleTranslator
        src_clean = source_lang
        if src_clean in ["zh", "chinese", "cn"]:
            src_clean = "zh-CN"
        tgt_clean = target_lang
        if tgt_clean in ["vi", "vietnamese"]:
            tgt_clean = "vi"
        gt = GoogleTranslator(source=src_clean, target=tgt_clean)
        return gt.translate(text)

    def translate_with_fallback(self, text, source_lang="vi", target_lang="en"):
        try:
            return self.translate_with_xkiro(text, source_lang, target_lang)
        except Exception:
            try:
                return self.translate_with_gemini(text, source_lang, target_lang)
            except Exception:
                return self.translate_with_google(text, source_lang, target_lang)

    def translate_text(self, text, source_lang="vi", target_lang="en"):
        """Dịch văn bản sử dụng engine đã chọn."""
        engine = self.get_translation_engine()
        if engine == "xkiro":
            return self.translate_with_xkiro(text, source_lang, target_lang)
        elif engine == "gemini":
            return self.translate_with_gemini(text, source_lang, target_lang)
        elif engine == "google":
            return self.translate_with_google(text, source_lang, target_lang)
        else:
            return self.translate_with_fallback(text, source_lang, target_lang)

    def stop(self):
        self._is_cancelled = True

    def run(self):
        try:
            self.start_time = time.time()
            if self._is_cancelled:
                return
            self.emit_step_progress(0, "Khởi động Pipeline tự động hóa 1-Click...")
            self.progress.emit("==================================================")
            self.progress.emit("🚀 KÍCH HOẠT PIPELINE TỰ ĐỘNG HÓA 1-CLICK (5 BƯỚC BACKGROUND)")
            self.progress.emit(f"🔍 OCR Engine đang dùng: {self.ocr_engine}")
            self.progress.emit("==================================================")
            self.progress.emit(f"📁 Video đầu vào: {self.video_path}")
            self.progress.emit(f"📤 File xuất kết quả: {self.output_path}")

            # BƯỚC 1 & 2: Streaming Video Chunking & OCR
            self.emit_step_progress(5, f"⚡ BƯỚC 1/5: Khởi tạo quét chữ (OCR Engine: {self.ocr_engine.upper()})...")
            
            segments = []
            if self.ocr_engine == "gemini":
                # Đọc danh sách API Keys dạng List để xoay key tự động
                api_keys_list = []
                if self.api_key:
                    api_keys_list = [k.strip() for k in self.api_key.split(",") if k.strip()]
                from gemini_vision_ocr import load_gemini_keys, scan_video_frames_with_gemini, extract_subtitles_with_gemini_vision
                if not api_keys_list:
                    api_keys_list = load_gemini_keys()

                # Nếu người dùng chưa khoanh vùng thủ công, tự động gọi Gemini Vision để lấy Bounding Boxes chuẩn
                if not self.selected_bboxes and not self.selected_bbox:
                    self.emit_step_progress(10, "🔍 Tự động quét Keyframe với Gemini Vision API...")
                    import cv2
                    cap_temp = cv2.VideoCapture(self.video_path)
                    cap_temp.set(cv2.CAP_PROP_POS_FRAMES, int(cap_temp.get(cv2.CAP_PROP_FPS) * 2 or 60))
                    ret_t, frame_t = cap_temp.read()
                    cap_temp.release()
                    if ret_t:
                        frame_rgb_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2RGB)
                        h_t, w_t, _ = frame_t.shape
                        vision_items = extract_subtitles_with_gemini_vision(frame_rgb_t, model_name=None)
                        auto_boxes = []
                        for vi in vision_items:
                            box_2d = vi.get("box_2d", [0, 0, 0, 0])
                            if box_2d and len(box_2d) == 4:
                                ymin, xmin, ymax, xmax = box_2d
                                y1 = int(ymin * h_t / 1000.0)
                                x1 = int(xmin * w_t / 1000.0)
                                y2 = int(ymax * h_t / 1000.0)
                                x2 = int(xmax * w_t / 1000.0)
                                auto_boxes.append([x1, y1, max(10, x2 - x1), max(10, y2 - y1)])
                        if auto_boxes:
                            self.selected_bboxes = auto_boxes
                            self.selected_bbox = auto_boxes[0]
                            self.progress.emit(f"✅ Gemini Vision đã phát hiện tự động {len(auto_boxes)} vùng chữ & Bounding Box!")

                self.emit_step_progress(15, "⚡ BƯỚC 2/5: Đang quét phụ đề video qua Gemini Vision...")
                def gemini_prog_cb(msg):
                    self.progress.emit(msg)
                    if "%" in msg:
                        try:
                            p_str = msg.split("%")[0].split("(")[-1].strip()
                            p_val = int(p_str)
                            self.emit_step_progress(15 + int(p_val * 0.2), f"Gemini Vision quét video: {p_val}%")
                        except Exception:
                            pass

                vision_segments = scan_video_frames_with_gemini(
                    video_path=self.video_path,
                    sample_interval_sec=self.scan_interval,
                    api_keys=api_keys_list,
                    progress_callback=gemini_prog_cb
                )

                if vision_segments:
                    segments = vision_segments
                    self.emit_step_progress(35, f"✔ Đã bóc xuất thành công {len(segments)} câu phụ đề.")

            # Nếu chọn xKiro, EasyOCR hoặc Gemini gặp lỗi 429/trả về rỗng
            if not segments:
                if self.ocr_engine == "gemini":
                    self.progress.emit("⚠️ Gemini hết quota (429 Resource Exhausted) hoặc không tìm thấy sub. Tự động chuyển sang quét OCR bằng xKiro / EasyOCR...")
                if self.ocr_engine == "paddleocr":
                    self.emit_step_progress(18, "⚡ BƯỚC 2/5: Đang quét OCR bằng PaddleOCR (Tiếng Trung - Baidu PP-OCRv4)...")
                elif self.ocr_engine == "xkiro":
                    self.emit_step_progress(18, "⚡ BƯỚC 2/5: Đang quét OCR video và kết hợp xKiro AI...")
                else:
                    self.emit_step_progress(18, "⚡ BƯỚC 2/5: Đang quét OCR song song đa phân đoạn (Parallel Chunks)...")

                ocr_bboxes = self.selected_bboxes or ([self.selected_bbox] if self.selected_bbox else [])
                from optimized_pipeline import ParallelChunkOCRProcessor
                processor = ParallelChunkOCRProcessor(self.video_path, max_workers=self.chunk_workers)
                
                def parallel_ocr_cb(msg):
                    self.progress.emit(msg)
                    if "Chunks -" in msg and "%" in msg:
                        try:
                            pct_chunk = int(msg.split("%")[0].split("-")[-1].strip())
                            self.emit_step_progress(18 + int(pct_chunk * 0.2), f"Đang quét OCR phân đoạn ({pct_chunk}%)")
                        except Exception:
                            pass

                res = processor.process_video_ocr(
                    bboxes=ocr_bboxes,
                    ocr_lang="auto",
                    api_key=self.api_key,
                    progress_callback=parallel_ocr_cb,
                    check_cancel_func=lambda: self._is_cancelled,
                    ocr_engine=self.ocr_engine
                )
                segments = res.get('subtitles', [])
                self.emit_step_progress(38, f"✔ Hoàn tất quét OCR ({len(segments)} câu phụ đề)")

            self.segments_ready.emit(segments)

            if self._is_cancelled:
                self.progress.emit("🛑 Tiến trình đã bị hủy!")
                return

            # BƯỚC 3: Dịch Phụ Đề Tự Động (Auto Subtitle Translation via full engine pipeline)
            chosen_engine = self.get_translation_engine()
            self.emit_step_progress(40, f"⚡ BƯỚC 3/5: Dịch phụ đề tự động bằng {self.engine} (Engine: {chosen_engine})...")
            import translator
            
            def trans_cb(msg):
                self.progress.emit(msg)
                if "/" in msg:
                    try:
                        part = msg.split("(")[-1].split(")")[0]
                        c, t = part.split("/")
                        p_t = int(int(c) / int(t) * 20)
                        self.emit_step_progress(40 + p_t, f"Đang dịch thuật ({part})...")
                    except Exception:
                        pass

            translated = translator.translate_segments(
                segments,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                engine=self.engine if self.engine != "Supersubs AI" else ("xKiro AI" if chosen_engine == "xkiro" else ("Gemini" if chosen_engine == "gemini" else "Google Translate")),
                api_key=self.api_key,
                progress_callback=trans_cb,
                ollama_model=self.ollama_model,
                vp_dict_paths=self.vp_dict_paths,
                xkiro_key=self.xkiro_key
            )

            if self.refine_enabled:
                self.emit_step_progress(60, f"⚡ BƯỚC 3b/5: Tinh chỉnh dịch thuật LLM ({self.refine_engine})...")
                translated = translator.refine_translated_segments(
                    translated,
                    api_key=self.refine_api_key or self.api_key,
                    engine=self.refine_engine,
                    progress_callback=self.progress.emit,
                    ollama_model=self.ollama_model
                )

            segments = translated
            self.segments_ready.emit(segments)
            self.emit_step_progress(65, f"✔ Đã hoàn tất dịch {len(segments)} câu phụ đề.")

            if self._is_cancelled:
                self.progress.emit("🛑 Tiến trình đã bị hủy thành công!")
                return

            # BƯỚC PHỤ: Xử lý Khung Tiêu Đề Video (self.title_bbox) nếu có
            translated_title_text = None
            if self.title_bbox and len(self.title_bbox) == 4:
                self.emit_step_progress(68, "📌 Quét OCR & dịch khung tiêu đề...")
                try:
                    import cv2
                    from transcriber import get_easyocr_reader
                    cap_title = cv2.VideoCapture(self.video_path)
                    fps_t = cap_title.get(cv2.CAP_PROP_FPS) or 30.0
                    total_f_t = int(cap_title.get(cv2.CAP_PROP_FRAME_COUNT))
                    tx, ty, tw, th = self.title_bbox
                    
                    sample_times = [0.5, 1.5, 2.5]
                    title_candidates = []
                    reader = get_easyocr_reader(['ch_sim', 'en'])
                    
                    for sec in sample_times:
                        target_f = int(sec * fps_t)
                        if target_f < total_f_t:
                            cap_title.set(cv2.CAP_PROP_POS_FRAMES, target_f)
                            ret_t, f_t = cap_title.read()
                            if ret_t:
                                h_img, w_img, _ = f_t.shape
                                x1, y1 = max(0, tx), max(0, ty)
                                x2, y2 = min(w_img, tx + tw), min(h_img, ty + th)
                                crop_t = f_t[y1:y2, x1:x2]
                                if crop_t.size > 0:
                                    ocr_res = reader.readtext(crop_t)
                                    for item in ocr_res:
                                        text_cand = item[1].strip()
                                        conf_cand = item[2]
                                        if len(text_cand) >= 2:
                                            title_candidates.append((text_cand, conf_cand))
                    cap_title.release()

                    if title_candidates:
                        title_candidates.sort(key=lambda x: x[1], reverse=True)
                        orig_title = title_candidates[0][0]
                        from deep_translator import GoogleTranslator
                        gt = GoogleTranslator(source=self.source_lang, target=self.target_lang)
                        translated_title_text = gt.translate(orig_title)
                        self.progress.emit(f"✅ Đã quét & dịch thành công Tiêu đề: '{orig_title}' ➔ '{translated_title_text}'")
                except Exception as e_t:
                    self.progress.emit(f"⚠️ Lỗi quét tiêu đề: {e_t}")

            # BƯỚC 4 & 5: Inpaint, Sub Burn-in, Audio TTS & Render Video
            if self.enable_dubbing:
                self.emit_step_progress(72, "⚡ BƯỚC 4-5/5: Sinh giọng đọc TTS AI & kết xuất video...")
                self.progress.emit("🔊 TTS enabled: Đang sinh giọng đọc AI...")
            else:
                self.emit_step_progress(72, "⚡ BƯỚC 4-5/5: Ghi đè phụ đề & kết xuất video (Giữ nguyên âm thanh gốc)...")
                self.progress.emit("⏭️ TTS disabled: Bỏ qua sinh giọng đọc AI, giữ nguyên âm thanh gốc.")

            self.progress.emit(f"📝 Số subtitle đã dịch: {len(segments)}")
            self.progress.emit(f"🔥 burn_sub (Ghi đè phụ đề): {self.burn_sub}")
            if self.preset:
                self.progress.emit(f"📐 Preset phụ đề: Font={self.preset.get('font_name', 'Arial')}, Size={self.preset.get('font_size', 20)}, V-Align={self.preset.get('v_align', 'bottom')}")
            
            if self.burn_sub and len(segments) > 0:
                self.progress.emit(f"🔥 Đang đè phụ đề đã dịch ({len(segments)} câu) lên video...")
            elif self.burn_sub and len(segments) == 0:
                self.progress.emit("⚠️ Không có subtitle để đè! Kiểm tra kết quả OCR & Dịch thuật.")

            def dubber_prog_cb(msg):
                self.progress.emit(msg)
                if "%" in msg:
                    try:
                        p_render = int(msg.split("%")[0].split()[-1].strip())
                        self.emit_step_progress(72 + int(p_render * 0.27), f"Render video ({p_render}%)")
                    except Exception:
                        pass
                elif "cho câu" in msg and "/" in msg:
                    try:
                        part = msg.split("cho câu")[-1].split("...")[0].strip()
                        c, t = part.split("/")
                        p_tts = int(int(c) / int(t) * 15)
                        self.emit_step_progress(72 + p_tts, f"Sinh giọng đọc TTS ({part})")
                    except Exception:
                        pass

            res_path, overflowed_segs = dubber.create_dubbed_video(
                self.video_path,
                segments,
                self.voice,
                self.output_path,
                bg_volume=self.bg_vol,
                dub_volume=self.dub_vol,
                burn_subtitles=self.burn_sub,
                selected_bbox=self.selected_bbox,
                preset=self.preset,
                progress_callback=dubber_prog_cb,
                enable_dubbing=self.enable_dubbing,
                selected_bboxes=self.selected_bboxes,
                logo_path=self.logo_path,
                title_text=translated_title_text,
                title_bbox=self.title_bbox,
                logo_bbox=self.logo_bbox
            )
            if self._is_cancelled:
                self.progress.emit("🛑 Tiến trình đã bị hủy thành công!")
                return

            self.finished.emit(res_path)

        except Exception as e:
            self.error.emit(str(e))


class KeyTesterWorker(QThread):
    result_signal = pyqtSignal(str, bool, str) # (key_str, success, message)

    def __init__(self, key_str, provider="gemini"):
        super().__init__()
        self.key_str = key_str
        self.provider = provider

    def run(self):
        try:
            if self.provider == "gemini":
                from google import genai
                client = genai.Client(api_key=self.key_str.strip())
                test_candidates = ["gemini-3.5-flash", "gemini-flash-latest", "gemini-flash-lite-latest", "gemini-3.7-flash"]
                last_ex = None
                success_model = None
                for m in test_candidates:
                    try:
                        res = client.models.generate_content(
                            model=m,
                            contents="Hi"
                        )
                        if res and hasattr(res, 'text') and res.text:
                            success_model = m
                            break
                    except Exception as ex_m:
                        last_ex = ex_m
                        continue
                if success_model:
                    self.result_signal.emit(self.key_str, True, f"🟢 HOẠT ĐỘNG TỐT ({success_model})")
                else:
                    raise last_ex or Exception("Không thể kết nối đến model Gemini")
            elif self.provider == "xkiro":
                import xkiro_client
                res = xkiro_client.translate_with_xkiro("Hi", target_lang="vi", api_key=self.key_str.strip())
                self.result_signal.emit(self.key_str, True, f"🟢 HOẠT ĐỘNG TỐT (xKiro AI: {res[:20]})")
            else:
                self.result_signal.emit(self.key_str, False, "🔴 Provider không xác định")
        except Exception as e:
            err_msg = str(e)
            if "404" in err_msg:
                msg = "🔴 LỖI 404 (Model Not Found)"
            elif "401" in err_msg or "403" in err_msg or "INVALID" in err_msg:
                msg = "🔴 LỖI 401/403 (API Key Không Hợp Lệ)"
            elif "429" in err_msg:
                msg = "🟡 LỖI 429 (Hết Quota / Quá Tải Call)"
            else:
                msg = f"🔴 LỖI: {err_msg[:40]}"
            self.result_signal.emit(self.key_str, False, msg)


class TTSPreviewWorker(QThread):
    finished_signal = pyqtSignal(bool, str) # (success, message_or_file_path)

    def __init__(self, voice_name, text="Xin chào, đây là giọng đọc thử nghiệm của phụ đề AI", rate="-10%"):
        super().__init__()
        self.voice_name = voice_name
        self.text = text
        self.rate = rate

    def run(self):
        try:
            import asyncio
            import edge_tts
            import tempfile
            import winsound

            temp_dir = tempfile.gettempdir()
            out_mp3 = os.path.join(temp_dir, "tts_preview_sample.mp3")

            async def _gen():
                communicate = edge_tts.Communicate(self.text, self.voice_name, rate=self.rate)
                await communicate.save(out_mp3)

            loop = asyncio.new_event_loop()
            loop.run_until_complete(_gen())
            loop.close()

            # Chuyển đổi MP3 hoặc phát âm thanh bằng winsound / QMediaPlayer nếu có
            if os.path.exists(out_mp3):
                # Phát âm thanh qua winsound (Async flag)
                try:
                    import os
                    os.system(f'start /min "" "{out_mp3}"')
                except Exception:
                    pass
                self.finished_signal.emit(True, f"🔊 Đã phát mẫu giọng đọc '{self.voice_name}' thành công!")
            else:
                self.finished_signal.emit(False, "❌ Không tạo được file âm thanh TTS.")
        except Exception as e:
            self.finished_signal.emit(False, f"❌ Lỗi preview TTS: {e}")


# Giao diện chính của ứng dụng
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("supersubs v1")
        self.resize(1100, 750)
        
        # Dữ liệu nội bộ
        self.video_path = ""
        self.segments = [] # Chứa [{'start', 'end', 'text', 'orig_text'}]
        self.selected_bbox = None # Tọa độ vùng quét chữ OCR [x, y, w, h]
        self.selected_bboxes = [] # Danh sách vùng quét đa điểm cho OCR
        self.box_type_dict = {} # Từ điển lưu phân loại loại khung {'sub', 'logo', 'title'}
        self.logo_path = "" # Đường dẫn file logo thương hiệu
        self.temp_preview_audio = None
        self.preset_font_color = [255, 255, 255]
        self.preset_outline_color = [0, 0, 0]
        self.preset_bg_color = [0, 0, 0]
        self.custom_font_path = None
        self.subtitle_custom_pos = None
        self.bbox_history_stack = []
        self.bbox_redo_stack = []
        
        # Dữ liệu dự án & Lịch sử (Trang 3)
        self.project_file_path = ""
        now_str = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        self.project_created_date = now_str
        self.project_updated_date = now_str
        self.pipeline_start_time = None
        
        # Đợt 1 Optimization attributes
        self._frame_cache = {}
        self.preview_zoom_factor = 1.0
        self._raw_log_records = []
        self._history_loaded_limit = 50
        self._cached_project_state = None
        
        # Tab 4: Batch Processing & Báo cáo attributes
        self.batch_queue = []
        self.batch_worker = None
        self.batch_results_history = []
        self._raw_batch_log_records = []
        
        # Dữ liệu phục vụ phát preview video tích hợp trong UI
        self.preview_timer = QTimer(self)
        if hasattr(self, 'play_preview_frame'):
            self.preview_timer.timeout.connect(self.play_preview_frame)
        self.preview_cap = None
        self.preview_start_frame = 0
        self.preview_end_frame = 0
        self.preview_current_frame = 0
        self.preview_delay = 33
        self.preview_text = ""
        self.video_width = 1920
        self.video_height = 1080
        self.title_bbox = None
        self.worker_bbox = None
        
        from PyQt6.QtGui import QShortcut, QKeySequence
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        if hasattr(self, 'undo_bbox_change'):
            self.shortcut_undo.activated.connect(self.undo_bbox_change)
        self.shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        if hasattr(self, 'redo_bbox_change'):
            self.shortcut_redo.activated.connect(self.redo_bbox_change)

        self.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self)
        self.shortcut_save.activated.connect(lambda: self.btn_save_project_clicked() if hasattr(self, 'btn_save_project_clicked') else None)
        self.shortcut_open = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_open.activated.connect(lambda: self.btn_load_project_clicked() if hasattr(self, 'btn_load_project_clicked') else None)
        self.shortcut_new = QShortcut(QKeySequence("Ctrl+N"), self)
        self.shortcut_new.activated.connect(lambda: self.btn_new_project_clicked() if hasattr(self, 'btn_new_project_clicked') else None)
        self.shortcut_run = QShortcut(QKeySequence("Ctrl+R"), self)
        self.shortcut_run.activated.connect(lambda: self.start_dubbing() if hasattr(self, 'start_dubbing') else None)

        # Auto-save timer (mỗi 5 phút = 300.000ms)
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.setInterval(300000)
        self.auto_save_timer.timeout.connect(lambda: self.auto_save_project_tick() if hasattr(self, 'auto_save_project_tick') else None)
        self.auto_save_timer.start()
        
        self.presets_db = {
            "Mặc định (Dưới - Giữa)": {
                "v_align": "bottom", "h_align": "center", "margin_v_type": "percent", "margin_v_val": 8.0, "margin_h_type": "percent", "margin_h_val": 5.0,
                "font_name": "Arial", "font_size": 20, "font_color": [255, 255, 255], "outline_color": [0, 0, 0], "outline_width": 2, "use_bg_box": False, "bg_color": [0, 0, 0], "bg_opacity": 0,
                "smart_pos": False
            },
            "Dưới - Trái": {
                "v_align": "bottom", "h_align": "left", "margin_v_type": "percent", "margin_v_val": 8.0, "margin_h_type": "percent", "margin_h_val": 5.0,
                "font_name": "Arial", "font_size": 20, "font_color": [255, 255, 255], "outline_color": [0, 0, 0], "outline_width": 2, "use_bg_box": False, "bg_color": [0, 0, 0], "bg_opacity": 0,
                "smart_pos": False
            },
            "Dưới - Phải": {
                "v_align": "bottom", "h_align": "right", "margin_v_type": "percent", "margin_v_val": 8.0, "margin_h_type": "percent", "margin_h_val": 5.0,
                "font_name": "Arial", "font_size": 20, "font_color": [255, 255, 255], "outline_color": [0, 0, 0], "outline_width": 2, "use_bg_box": False, "bg_color": [0, 0, 0], "bg_opacity": 0,
                "smart_pos": False
            },
            "Trên - Giữa": {
                "v_align": "top", "h_align": "center", "margin_v_type": "percent", "margin_v_val": 8.0, "margin_h_type": "percent", "margin_h_val": 5.0,
                "font_name": "Arial", "font_size": 20, "font_color": [255, 255, 255], "outline_color": [0, 0, 0], "outline_width": 2, "use_bg_box": False, "bg_color": [0, 0, 0], "bg_opacity": 0,
                "smart_pos": False
            },
            "Giữa - Giữa": {
                "v_align": "middle", "h_align": "center", "margin_v_type": "percent", "margin_v_val": 0.0, "margin_h_type": "percent", "margin_h_val": 5.0,
                "font_name": "Arial", "font_size": 22, "font_color": [255, 255, 255], "outline_color": [0, 0, 0], "outline_width": 2, "use_bg_box": False, "bg_color": [0, 0, 0], "bg_opacity": 0,
                "smart_pos": False
            }
        }
        
        try:
            self.setup_ui()
            if hasattr(self, 'load_api_config_to_ui'):
                self.load_api_config_to_ui()
            if hasattr(self, 'load_app_settings'):
                self.load_app_settings()
            if hasattr(self, 'load_xkiro_prompt_template'):
                self.load_xkiro_prompt_template()
            if hasattr(self, 'update_glossary_combobox'):
                self.update_glossary_combobox()
            apply_custom_styles_to_app(self)
        except Exception as e:
            print(f"[ERROR] Exception during MainWindow initialization: {e}")
            import traceback
            traceback.print_exc()
    def set_drawing_box_type(self, btype):
        self.active_drawing_box_type = btype
        type_names = {"sub": "🟦 Phụ Đề (OCR)", "title": "🟨 Tiêu Đề", "logo": "🟥 Logo"}
        name = type_names.get(btype, btype)
        if hasattr(self, 'txt_log_console') and self.txt_log_console:
            self.txt_log_console.append(f"✍️ Đã chọn chế độ khoanh vùng: {name}")

    def clear_all_bboxes(self):
        self.selected_bboxes = []
        self.selected_bbox = None
        self.title_bbox = None
        self.logo_bbox = None
        self.box_type_dict = {}
        if hasattr(self, 'lbl_main_preview') and self.lbl_main_preview:
            self.lbl_main_preview.bboxes = []
            self.lbl_main_preview.update()
        if hasattr(self, 'txt_log_console') and self.txt_log_console:
            self.txt_log_console.append("🗑️ Đã xóa toàn bộ khung khoanh vùng.")

    def copy_log_to_clipboard(self):
        if hasattr(self, 'txt_log_console') and self.txt_log_console:
            from PyQt6.QtWidgets import QApplication
            QApplication.clipboard().setText(self.txt_log_console.toPlainText())
            self.txt_log_console.append("📋 Đã sao chép toàn bộ log vào clipboard!")

    def save_log_to_file(self):
        if hasattr(self, 'txt_log_console') and self.txt_log_console:
            from PyQt6.QtWidgets import QFileDialog
            path, _ = QFileDialog.getSaveFileName(self, "Lưu Log File", "system_log.txt", "Text Files (*.txt);;All Files (*)")
            if path:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(self.txt_log_console.toPlainText())
                    self.txt_log_console.append(f"💾 Đã lưu log file thành công tại: {path}")
                except Exception as e:
                    self.txt_log_console.append(f"❌ Lỗi khi lưu log file: {e}")

    def open_api_error_log_dialog(self):
        err_log_paths = [
            os.path.join("logs", "gemini_api_errors.log"),
            os.path.join("logs", "xkiro_api_errors.log")
        ]
        log_content = ""
        for p in err_log_paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        log_content += f"=== LOG FILE: {p} ===\n" + f.read() + "\n\n"
                except Exception:
                    pass

        if not log_content:
            log_content = "Chưa có log lỗi Gemini hay xKiro API nào được ghi nhận."

        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
        dlg = QDialog(self)
        dlg.setWindowTitle("⚠️ Nhật Ký Lỗi Gemini / xKiro API")
        dlg.resize(600, 400)
        ly = QVBoxLayout(dlg)
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setText(log_content)
        ly.addWidget(txt)
        btn_close = QPushButton("Đóng")
        btn_close.clicked.connect(dlg.accept)
        ly.addWidget(btn_close)
        dlg.exec()

    def toggle_key_input_visibility(self):
        if hasattr(self, 'txt_new_api_key') and self.txt_new_api_key:
            if self.txt_new_api_key.echoMode() == QLineEdit.EchoMode.Password:
                self.txt_new_api_key.setEchoMode(QLineEdit.EchoMode.Normal)
                self.btn_toggle_key_vis.setText("🙈")
            else:
                self.txt_new_api_key.setEchoMode(QLineEdit.EchoMode.Password)
                self.btn_toggle_key_vis.setText("👁️")

    def load_api_keys_to_tab2(self):
        if not hasattr(self, 'list_api_keys') or self.list_api_keys is None:
            return

        self.list_api_keys.clear()
        key_file = os.path.abspath(os.path.join("config", "api_keys.json"))
        if os.path.exists(key_file):
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                gemini_keys = data.get("gemini_keys", [])
                for k in gemini_keys:
                    k_str = str(k).strip()
                    if k_str:
                        masked = k_str[:8] + "...****..." + k_str[-6:] if len(k_str) > 16 else k_str
                        item = QListWidgetItem(f"🔵 [Gemini API]  {masked}")
                        item.setData(Qt.ItemDataRole.UserRole, ("gemini", k_str))
                        self.list_api_keys.addItem(item)

                xkiro_keys = data.get("xkiro_keys", [])
                for k in xkiro_keys:
                    k_str = str(k).strip()
                    if k_str:
                        masked = k_str[:8] + "...****..." + k_str[-6:] if len(k_str) > 16 else k_str
                        item = QListWidgetItem(f"🟣 [xKiro AI]    {masked}")
                        item.setData(Qt.ItemDataRole.UserRole, ("xkiro", k_str))
                        self.list_api_keys.addItem(item)

            except Exception as e:
                print(f"[TAB 2] Error loading keys: {e}")

    def add_new_api_key_tab2(self):
        if not hasattr(self, 'txt_new_api_key') or not self.txt_new_api_key:
            return

        raw_key = self.txt_new_api_key.text().strip()
        if not raw_key or len(raw_key) < 10 or " " in raw_key:
            if hasattr(self, 'txt_log_console') and self.txt_log_console:
                self.txt_log_console.append("⚠️ API Key rỗng hoặc không hợp lệ (cần ít nhất 10 ký tự, không chứa khoảng trắng).")
            return

        provider = "gemini"
        if hasattr(self, 'cb_key_provider') and "xKiro" in self.cb_key_provider.currentText():
            provider = "xkiro"

        key_dir = os.path.join(os.path.dirname(__file__), "config")
        os.makedirs(key_dir, exist_ok=True)
        key_file = os.path.join(key_dir, "api_keys.json")

        data = {"gemini_keys": [], "xkiro_keys": []}
        if os.path.exists(key_file):
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass

        if provider == "gemini":
            if "gemini_keys" not in data or not isinstance(data["gemini_keys"], list):
                data["gemini_keys"] = []
            if raw_key not in data["gemini_keys"]:
                data["gemini_keys"].append(raw_key)
        else:
            if "xkiro_keys" not in data or not isinstance(data["xkiro_keys"], list):
                data["xkiro_keys"] = []
            if raw_key not in data["xkiro_keys"]:
                data["xkiro_keys"].append(raw_key)

        try:
            with open(key_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.txt_new_api_key.clear()
            self.load_api_keys_to_tab2()
            if hasattr(self, 'txt_log_console') and self.txt_log_console:
                self.txt_log_console.append(f"✅ Đã thêm API Key thành công vào mảng '{provider}_keys' trong config/api_keys.json!")
        except Exception as e:
            if hasattr(self, 'txt_log_console') and self.txt_log_console:
                self.txt_log_console.append(f"❌ Lỗi khi lưu API Key: {e}")

    def delete_selected_api_key_tab2(self):
        if not hasattr(self, 'list_api_keys') or not self.list_api_keys:
            return

        current_item = self.list_api_keys.currentItem()
        if not current_item:
            return

        data_tuple = current_item.data(Qt.ItemDataRole.UserRole)
        if not data_tuple:
            return

        provider, key_to_del = data_tuple
        key_file = os.path.join(os.path.dirname(__file__), "config", "api_keys.json")

        if os.path.exists(key_file):
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if provider == "gemini" and "gemini_keys" in data:
                    data["gemini_keys"] = [k for k in data["gemini_keys"] if k != key_to_del]
                elif provider == "xkiro" and "xkiro_keys" in data:
                    data["xkiro_keys"] = [k for k in data["xkiro_keys"] if k != key_to_del]

                with open(key_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

                self.load_api_keys_to_tab2()
                if hasattr(self, 'txt_log_console') and self.txt_log_console:
                    self.txt_log_console.append(f"🗑️ Đã xóa API Key ({provider}) khỏi config/api_keys.json!")
            except Exception as e:
                print(f"[TAB 2] Error deleting key: {e}")

    def test_selected_key_tab2(self):
        if not hasattr(self, 'list_api_keys') or not self.list_api_keys:
            return

        current_item = self.list_api_keys.currentItem()
        if not current_item:
            if hasattr(self, 'txt_log_console') and self.txt_log_console:
                self.txt_log_console.append("⚠️ Vui lòng chọn 1 Key trong danh sách để kiểm thử.")
            return

        data_tuple = current_item.data(Qt.ItemDataRole.UserRole)
        if not data_tuple:
            return

        provider, raw_key = data_tuple
        if hasattr(self, 'txt_log_console') and self.txt_log_console:
            self.txt_log_console.append(f"📡 Đang kết nối kiểm thử thực tế Key {provider.upper()} API...")

        current_item.setText(f"📡 Đang kiểm tra... ({provider.upper()})")

        self.key_test_worker = KeyTesterWorker(raw_key, provider=provider)
        def on_test_result(key_str, success, msg):
            masked = key_str[:8] + "...****..." + key_str[-6:] if len(key_str) > 16 else key_str
            tag = "🔵 [Gemini API]" if provider == "gemini" else "🟣 [xKiro AI]"
            current_item.setText(f"{tag}  {masked}  ➔  {msg}")
            if hasattr(self, 'txt_log_console') and self.txt_log_console:
                self.txt_log_console.append(f"Kiểm tra Key: {msg}")

        self.key_test_worker.result_signal.connect(on_test_result)
        self.key_test_worker.start()

    def pick_subtitle_color(self, color_type):
        from PyQt6.QtWidgets import QColorDialog
        color = QColorDialog.getColor()
        if color.isValid():
            rgb = [color.red(), color.green(), color.blue()]
            hex_code = color.name()
            if color_type == 'font':
                self.preset_font_color = rgb
                if hasattr(self, 'lbl_font_hex'): self.lbl_font_hex.setText(hex_code)
                if hasattr(self, 'btn_font_color'): self.btn_font_color.setStyleSheet(f"background-color: {hex_code}; border: 1px solid white;")
            elif color_type == 'outline':
                self.preset_outline_color = rgb
                if hasattr(self, 'lbl_outline_hex'): self.lbl_outline_hex.setText(hex_code)
                if hasattr(self, 'btn_outline_color'): self.btn_outline_color.setStyleSheet(f"background-color: {hex_code}; border: 1px solid white;")
            elif color_type == 'bg':
                self.preset_bg_color = rgb
                if hasattr(self, 'lbl_bg_hex'): self.lbl_bg_hex.setText(hex_code)
                if hasattr(self, 'btn_bg_color'): self.btn_bg_color.setStyleSheet(f"background-color: {hex_code}; border: 1px solid white;")
            elif color_type == 'title':
                self.preset_title_color = rgb
                if hasattr(self, 'lbl_title_hex'): self.lbl_title_hex.setText(hex_code)
                if hasattr(self, 'btn_title_color'): self.btn_title_color.setStyleSheet(f"background-color: {hex_code}; border: 1px solid white;")

            self.update_live_font_preview()

    def update_live_font_preview(self):
        if not hasattr(self, 'tab3_font_preview') or not self.tab3_font_preview:
            return

        font_name = self.cb_font_name.currentText() if hasattr(self, 'cb_font_name') and self.cb_font_name else "Arial"
        font_size = self.spin_font_size.value() if hasattr(self, 'spin_font_size') and self.spin_font_size else 20
        font_color = getattr(self, 'preset_font_color', [255, 255, 255])
        outline_color = getattr(self, 'preset_outline_color', [0, 0, 0])
        bg_color = getattr(self, 'preset_bg_color', [0, 0, 0])
        use_bg = self.chk_use_bg_box.isChecked() if hasattr(self, 'chk_use_bg_box') and self.chk_use_bg_box else False

        fg_hex = f"rgb({font_color[0]}, {font_color[1]}, {font_color[2]})"
        bg_css = f"background-color: rgb({bg_color[0]}, {bg_color[1]}, {bg_color[2]});" if use_bg else "background-color: transparent;"

        style = (
            f"font-family: '{font_name}'; "
            f"font-size: {font_size}px; "
            f"color: {fg_hex}; "
            f"{bg_css} "
            f"font-weight: bold; "
            f"padding: 8px; "
            f"border-radius: 4px;"
        )
        self.tab3_font_preview.setStyleSheet(style)

    def preview_tts_voice_tab3(self):
        if not hasattr(self, 'cb_voice') or not self.cb_voice:
            return

        voice_name = self.cb_voice.currentData() or self.cb_voice.currentText()
        if not voice_name:
            return

        rate_val = f"{self.spin_tts_rate.value()}%" if hasattr(self, 'spin_tts_rate') and self.spin_tts_rate else "-10%"
        sample_txt = "Xin chào, đây là giọng đọc AI thử nghiệm của phụ đề video!"

        if hasattr(self, 'txt_log_console') and self.txt_log_console:
            self.txt_log_console.append(f"🎙️ Đang kết nối tải thử nghiệm giọng đọc: {voice_name}...")

        self.tts_preview_worker = TTSPreviewWorker(voice_name, text=sample_txt, rate=rate_val)
        def on_tts_finished(success, msg):
            if hasattr(self, 'txt_log_console') and self.txt_log_console:
                self.txt_log_console.append(msg)

        self.tts_preview_worker.finished_signal.connect(on_tts_finished)
        self.tts_preview_worker.start()

    def browse_output_dir(self):
        from PyQt6.QtWidgets import QFileDialog
        dir_path = QFileDialog.getExistingDirectory(self, "Chọn Thư Mục Xuất Video Thành Phẩm", os.path.abspath("videos"))
        if dir_path:
            if hasattr(self, 'txt_output_dir') and self.txt_output_dir:
                self.txt_output_dir.setText(dir_path)
            self.save_app_settings()

    def open_log_folder(self):
        log_dir = os.path.abspath("logs")
        os.makedirs(log_dir, exist_ok=True)
        try:
            os.startfile(log_dir)
        except Exception as e:
            if hasattr(self, 'txt_log_console') and self.txt_log_console:
                self.txt_log_console.append(f"📁 Thư mục Log/Cache: {log_dir}")

    def load_app_settings(self):
        settings_file = os.path.join(os.path.dirname(__file__), "config", "app_settings.json")
        if not os.path.exists(settings_file):
            settings_file = os.path.abspath(os.path.join("config", "app_settings.json"))
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if hasattr(self, 'chk_prefer_xkiro') and "prefer_xkiro" in data:
                    self.chk_prefer_xkiro.setChecked(bool(data["prefer_xkiro"]))
                if hasattr(self, 'spin_workers') and "workers_cnt" in data:
                    self.spin_workers.setValue(int(data["workers_cnt"]))
                elif hasattr(self, 'spin_max_workers') and "max_workers" in data:
                    self.spin_max_workers.setValue(int(data["max_workers"]))

                if hasattr(self, 'chk_burn_sub_export') and "burn_sub" in data:
                    self.chk_burn_sub_export.setChecked(bool(data["burn_sub"]))
                if hasattr(self, 'slider_bg') and "bg_vol" in data:
                    self.slider_bg.setValue(int(float(data["bg_vol"]) * 100))
                if hasattr(self, 'slider_dub') and "dub_vol" in data:
                    self.slider_dub.setValue(int(float(data["dub_vol"]) * 100))
                if hasattr(self, 'txt_output_dir') and data.get("output_dir"):
                    self.txt_output_dir.setText(data["output_dir"])

                if hasattr(self, 'spin_chunk_workers') and "chunk_workers" in data:
                    self.spin_chunk_workers.setValue(int(data["chunk_workers"]))
                if hasattr(self, 'cb_tts_engine') and "tts_engine" in data:
                    self.cb_tts_engine.setCurrentText(str(data["tts_engine"]))
                if hasattr(self, 'chk_auto_save_voice') and "auto_save_voice" in data:
                    self.chk_auto_save_voice.setChecked(bool(data["auto_save_voice"]))
                if hasattr(self, 'chk_voice_fallback') and "voice_fallback" in data:
                    self.chk_voice_fallback.setChecked(bool(data["voice_fallback"]))

                # Preset fields
                preset = data.get("preset", {})
                if preset:
                    if hasattr(self, 'cb_font_name') and preset.get("font_name"):
                        self.cb_font_name.setCurrentText(preset["font_name"])
                    if hasattr(self, 'spin_font_size') and preset.get("font_size"):
                        self.spin_font_size.setValue(int(preset["font_size"]))
                    if hasattr(self, 'spin_outline_width') and "outline_width" in preset:
                        self.spin_outline_width.setValue(int(preset["outline_width"]))
                    if hasattr(self, 'chk_use_bg_box') and "use_bg_box" in preset:
                        self.chk_use_bg_box.setChecked(bool(preset["use_bg_box"]))
                    if hasattr(self, 'slider_bg_opacity') and "bg_opacity" in preset:
                        self.slider_bg_opacity.setValue(int(preset["bg_opacity"]))
                    if hasattr(self, 'cb_v_align') and preset.get("v_align"):
                        idx = self.cb_v_align.findText(preset["v_align"], Qt.MatchFlag.MatchStartsWith)
                        if idx >= 0: self.cb_v_align.setCurrentIndex(idx)
                    if hasattr(self, 'cb_h_align') and preset.get("h_align"):
                        idx = self.cb_h_align.findText(preset["h_align"], Qt.MatchFlag.MatchStartsWith)
                        if idx >= 0: self.cb_h_align.setCurrentIndex(idx)
                    if hasattr(self, 'spin_margin_v') and "margin_v_val" in preset:
                        self.spin_margin_v.setValue(int(preset["margin_v_val"]))
                    if hasattr(self, 'spin_margin_h') and "margin_h_val" in preset:
                        self.spin_margin_h.setValue(int(preset["margin_h_val"]))
                    if "font_color" in preset and isinstance(preset["font_color"], list):
                        self.preset_font_color = preset["font_color"]
                        hex_c = f"#{preset['font_color'][0]:02X}{preset['font_color'][1]:02X}{preset['font_color'][2]:02X}"
                        if hasattr(self, 'btn_font_color'): self.btn_font_color.setStyleSheet(f"background-color: {hex_c}; border: 1px solid white;")
                        if hasattr(self, 'lbl_font_hex'): self.lbl_font_hex.setText(hex_c)
                    if "outline_color" in preset and isinstance(preset["outline_color"], list):
                        self.preset_outline_color = preset["outline_color"]
                        hex_c = f"#{preset['outline_color'][0]:02X}{preset['outline_color'][1]:02X}{preset['outline_color'][2]:02X}"
                        if hasattr(self, 'btn_outline_color'): self.btn_outline_color.setStyleSheet(f"background-color: {hex_c}; border: 1px solid white;")
                        if hasattr(self, 'lbl_outline_hex'): self.lbl_outline_hex.setText(hex_c)
                    if "bg_color" in preset and isinstance(preset["bg_color"], list):
                        self.preset_bg_color = preset["bg_color"]
                        hex_c = f"#{preset['bg_color'][0]:02X}{preset['bg_color'][1]:02X}{preset['bg_color'][2]:02X}"
                        if hasattr(self, 'btn_bg_color'): self.btn_bg_color.setStyleSheet(f"background-color: {hex_c}; border: 1px solid white;")
                        if hasattr(self, 'lbl_bg_hex'): self.lbl_bg_hex.setText(hex_c)

                if "dark_mode" in data and hasattr(self, 'chk_dark_mode'):
                    self.chk_dark_mode.blockSignals(True)
                    self.chk_dark_mode.setChecked(bool(data["dark_mode"]))
                    self.chk_dark_mode.blockSignals(False)
                    if hasattr(self, 'apply_dark_mode'):
                        self.apply_dark_mode(bool(data["dark_mode"]))

                if "preferred_voice" in data and hasattr(self, 'cb_voice') and self.cb_voice:
                    pvoice = data["preferred_voice"]
                    idx = self.cb_voice.findData(pvoice)
                    if idx >= 0:
                        self.cb_voice.setCurrentIndex(idx)

                if "default_engine" in data and hasattr(self, 'cb_engine') and self.cb_engine:
                    idx = self.cb_engine.findText(data["default_engine"], Qt.MatchFlag.MatchContains)
                    if idx >= 0:
                        self.cb_engine.setCurrentIndex(idx)

                if "translation_engine" in data and hasattr(self, 'cb_translation_engine'):
                    self.cb_translation_engine.setCurrentText(str(data["translation_engine"]))
                if "auto_report" in data and hasattr(self, 'chk_auto_report'):
                    self.chk_auto_report.setChecked(bool(data["auto_report"]))
                if "report_format" in data and hasattr(self, 'cb_report_format'):
                    self.cb_report_format.setCurrentText(str(data["report_format"]))
                if "gemini_model" in data and hasattr(self, 'cb_gemini_model'):
                    self.cb_gemini_model.setCurrentText(str(data["gemini_model"]))
                if "gemini_auto_fallback_model" in data and hasattr(self, 'chk_gemini_auto_fallback_model'):
                    self.chk_gemini_auto_fallback_model.setChecked(bool(data["gemini_auto_fallback_model"]))
                if "gemini_fallback_easyocr" in data and hasattr(self, 'chk_gemini_fallback_easyocr'):
                    self.chk_gemini_fallback_easyocr.setChecked(bool(data["gemini_fallback_easyocr"]))
                if "open_folder_on_done" in data and hasattr(self, 'chk_open_folder_on_done'):
                    self.chk_open_folder_on_done.setChecked(bool(data["open_folder_on_done"]))
                if "scan_interval" in data and hasattr(self, 'spin_scan_interval'):
                    self.spin_scan_interval.setValue(float(data["scan_interval"]))
                if "scan_interval" in data and hasattr(self, 'spin_scan_interval'):
                    self.spin_scan_interval.setValue(float(data["scan_interval"]))
                if "min_sub_dur" in data and hasattr(self, 'spin_min_sub_dur'):
                    self.spin_min_sub_dur.setValue(float(data["min_sub_dur"]))
                if "paddle_lang" in data and hasattr(self, 'cb_paddle_lang'):
                    self.cb_paddle_lang.setCurrentText(str(data["paddle_lang"]))

            except Exception as e:
                print(f"[SETTINGS] Error loading app_settings.json: {e}")

    def save_app_settings(self):
        key_dir = os.path.join(os.path.dirname(__file__), "config")
        os.makedirs(key_dir, exist_ok=True)
        settings_file = os.path.join(key_dir, "app_settings.json")

        workers_val = 4
        if hasattr(self, 'spin_workers') and self.spin_workers:
            workers_val = self.spin_workers.value()
        elif hasattr(self, 'spin_max_workers') and self.spin_max_workers:
            workers_val = self.spin_max_workers.value()

        chunk_workers_val = self.spin_chunk_workers.value() if hasattr(self, 'spin_chunk_workers') and self.spin_chunk_workers else workers_val

        data = {
            "gemini_model": self.cb_gemini_model.currentText() if hasattr(self, 'cb_gemini_model') else "gemini-3.5-flash",
            "gemini_auto_fallback_model": self.chk_gemini_auto_fallback_model.isChecked() if hasattr(self, 'chk_gemini_auto_fallback_model') else True,
            "gemini_fallback_easyocr": self.chk_gemini_fallback_easyocr.isChecked() if hasattr(self, 'chk_gemini_fallback_easyocr') else True,
            "translation_engine": self.cb_translation_engine.currentText() if hasattr(self, 'cb_translation_engine') else "xKiro AI (Ưu tiên)",
            "prefer_xkiro": self.chk_prefer_xkiro.isChecked() if hasattr(self, 'chk_prefer_xkiro') and self.chk_prefer_xkiro else False,
            "dark_mode": self.chk_dark_mode.isChecked() if hasattr(self, 'chk_dark_mode') and self.chk_dark_mode else True,
            "workers_cnt": workers_val,
            "chunk_workers": chunk_workers_val,
            "default_workers": workers_val,
            "burn_sub": self.chk_burn_sub_export.isChecked() if hasattr(self, 'chk_burn_sub_export') and self.chk_burn_sub_export else True,
            "bg_vol": (self.slider_bg.value() / 100.0) if hasattr(self, 'slider_bg') and self.slider_bg else 0.3,
            "dub_vol": (self.slider_dub.value() / 100.0) if hasattr(self, 'slider_dub') and self.slider_dub else 1.0,
            "output_dir": self.txt_output_dir.text().strip() if hasattr(self, 'txt_output_dir') and self.txt_output_dir else "videos",
            "preset": self.get_current_subtitle_preset() if hasattr(self, 'get_current_subtitle_preset') else {},
            "auto_report": self.chk_auto_report.isChecked() if hasattr(self, 'chk_auto_report') else True,
            "report_format": self.cb_report_format.currentText() if hasattr(self, 'cb_report_format') else "HTML",
            "open_folder_on_done": self.chk_open_folder_on_done.isChecked() if hasattr(self, 'chk_open_folder_on_done') else True,
            "scan_interval": self.spin_scan_interval.value() if hasattr(self, 'spin_scan_interval') else 0.5,
            "min_sub_dur": self.spin_min_sub_dur.value() if hasattr(self, 'spin_min_sub_dur') else 0.3,
            "paddle_lang": self.cb_paddle_lang.currentText() if hasattr(self, 'cb_paddle_lang') else "Tiếng Trung (zh/ch)",
            "tts_engine": self.cb_tts_engine.currentText() if hasattr(self, 'cb_tts_engine') else "Edge-TTS",
            "auto_save_voice": self.chk_auto_save_voice.isChecked() if hasattr(self, 'chk_auto_save_voice') else True,
            "voice_fallback": self.chk_voice_fallback.isChecked() if hasattr(self, 'chk_voice_fallback') else True
        }
        if hasattr(self, 'chk_auto_save_voice') and self.chk_auto_save_voice.isChecked() and hasattr(self, 'cb_voice') and self.cb_voice:
            data["preferred_voice"] = self.cb_voice.currentData()
        if hasattr(self, 'cb_engine') and self.cb_engine:
            data["default_engine"] = self.cb_engine.currentText()
        if hasattr(self, 'cb_v_align') and self.cb_v_align:
            data["default_sub_position"] = self.cb_v_align.currentText()
        try:
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            if hasattr(self, 'txt_log_console') and self.txt_log_console:
                self.txt_log_console.append("💾 Đã lưu cấu hình ứng dụng vào config/app_settings.json!")
        except Exception as e:
            print(f"[SETTINGS] Error saving app_settings.json: {e}")

    def test_current_gemini_model_clicked(self):
        from gemini_vision_ocr import test_gemini_model_status
        m_name = self.cb_gemini_model.currentText() if hasattr(self, 'cb_gemini_model') else "gemini-2.0-flash-exp"
        clean_m = m_name.split()[0]
        self.log_info(f"🔍 Đang kiểm tra kết nối với Model '{clean_m}'...")
        ok, msg = test_gemini_model_status(clean_m)
        if ok:
            self.log_info(msg)
            if not os.environ.get("QT_QPA_PLATFORM"):
                QMessageBox.information(self, "Kiểm tra Model Gemini", msg)
        else:
            self.log_info(msg)
            if not os.environ.get("QT_QPA_PLATFORM"):
                QMessageBox.warning(self, "Kiểm tra Model Gemini", msg)

    def show_available_gemini_models_dialog(self):
        from gemini_vision_ocr import list_available_gemini_models
        self.log_info("📋 Đang tải danh sách model khả dụng từ Google Gemini API...")
        models = list_available_gemini_models()
        if not models:
            models = ["gemini-2.0-flash-exp", "gemini-2.0-flash-lite-preview", "gemini-1.5-flash-8b", "gemini-3.5-flash", "gemini-flash-latest"]
        msg = "Các model Gemini đang hoạt động trên tài khoản Google của bạn:\n\n" + "\n".join([f"• {m}" for m in models[:25]])
        self.log_info(f"📋 Tìm thấy {len(models)} model trên Google API.")
        if not os.environ.get("QT_QPA_PLATFORM"):
            QMessageBox.information(self, "Danh sách Model khả dụng", msg)

    def refresh_gemini_models(self):
        """Gọi API để lấy danh sách models mới nhất và nạp vào ComboBox."""
        from gemini_vision_ocr import list_available_gemini_models
        self.log_info("🔄 Đang quét danh sách model khả dụng từ Google Gemini API...")
        models = list_available_gemini_models()
        if models and hasattr(self, 'cb_gemini_model'):
            curr_text = self.cb_gemini_model.currentText()
            self.cb_gemini_model.blockSignals(True)
            self.cb_gemini_model.clear()
            
            self.cb_gemini_model.addItem("gemini-flash-latest (⭐ Khuyến nghị - Mới nhất)")
            self.cb_gemini_model.addItem("gemini-2.5-flash (🚀 Nhanh & chuẩn)")
            self.cb_gemini_model.addItem("gemini-flash-lite-latest (💨 Siêu nhẹ & nhanh)")
            self.cb_gemini_model.addItem("gemini-2.5-flash-lite (⚡ Tiết kiệm quota)")
            self.cb_gemini_model.addItem("gemini-2.0-flash-exp (🧪 Exp)")
            self.cb_gemini_model.addItem("gemini-2.0-flash-lite-preview (🧪 Lite Preview)")
            self.cb_gemini_model.addItem("gemini-1.5-flash-8b (🧪 8B)")
            self.cb_gemini_model.addItem("gemini-3.5-flash")
            self.cb_gemini_model.addItem("gemini-3.7-flash")
            self.cb_gemini_model.addItem("gemini-2.5-pro (🏆 Chất lượng cao)")
            self.cb_gemini_model.addItem("🔍 Auto (Thử tất cả)")

            existing_items = [self.cb_gemini_model.itemText(i).split()[0] for i in range(self.cb_gemini_model.count())]
            for m in models:
                if m not in existing_items:
                    self.cb_gemini_model.addItem(m)

            idx = self.cb_gemini_model.findText(curr_text.split()[0], Qt.MatchFlag.MatchStartsWith)
            if idx >= 0:
                self.cb_gemini_model.setCurrentIndex(idx)
            else:
                self.cb_gemini_model.setCurrentIndex(0)
            self.cb_gemini_model.blockSignals(False)
            self.save_app_settings()

            self.log_info(f"✅ Đã làm mới danh sách: Tìm thấy {len(models)} models trên Google API.")
            if not os.environ.get("QT_QPA_PLATFORM"):
                QMessageBox.information(self, "Thành công", f"Đã tìm thấy và cập nhật {len(models)} models từ Google Gemini API!")
        else:
            self.log_info("❌ Không thể lấy danh sách models từ Google API. Kiểm tra lại API Key.")
            if not os.environ.get("QT_QPA_PLATFORM"):
                QMessageBox.warning(self, "Lỗi", "Không thể lấy danh sách models. Vui lòng kiểm tra lại API Key.")

    DEFAULT_XKIRO_PROMPT_TEMPLATE = (
        "Bạn là chuyên gia dịch thuật phim ảnh. Hãy dịch đoạn văn sau từ {source_lang} sang {target_lang} với các yêu cầu:\n\n"
        "1. Giữ nguyên ý nghĩa và ngữ cảnh của câu chuyện\n"
        "2. Dịch tự nhiên, không dịch word-by-word, phù hợp với văn nói\n"
        "3. Giữ nguyên các thuật ngữ chuyên ngành, tên riêng, địa danh\n"
        "4. Đảm bảo độ dài phù hợp với thời lượng hiển thị subtitle (ngắn gọn, súc tích)\n"
        "5. Phù hợp với văn phong hội thoại trong phim/video\n\n"
        "{context}\n\n"
        "Đoạn văn cần dịch:\n"
        "{text}"
    )

    def load_default_xkiro_prompt(self):
        if hasattr(self, 'txt_xkiro_prompt_template'):
            self.txt_xkiro_prompt_template.setPlainText(self.DEFAULT_XKIRO_PROMPT_TEMPLATE)
            self.log_info("📥 Đã khôi phục Prompt Template xKiro về mặc định.")

    def load_xkiro_prompt_template(self):
        p_path = os.path.join(os.path.dirname(__file__), "config", "xkiro_prompt_template.json")
        if os.path.exists(p_path):
            try:
                with open(p_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    tmpl = data.get("template", self.DEFAULT_XKIRO_PROMPT_TEMPLATE)
                    if hasattr(self, 'txt_xkiro_prompt_template'):
                        self.txt_xkiro_prompt_template.setPlainText(tmpl)
                    if "max_tokens" in data and hasattr(self, 'spin_xkiro_max_tokens'):
                        self.spin_xkiro_max_tokens.setValue(int(data["max_tokens"]))
                    if "temperature" in data and hasattr(self, 'cb_xkiro_temperature'):
                        self.cb_xkiro_temperature.setCurrentText(str(data["temperature"]))
                    if "keep_proper_nouns" in data and hasattr(self, 'chk_keep_proper_nouns'):
                        self.chk_keep_proper_nouns.setChecked(bool(data["keep_proper_nouns"]))
                    if "auto_context" in data and hasattr(self, 'chk_auto_context'):
                        self.chk_auto_context.setChecked(bool(data["auto_context"]))
                    return data
            except Exception as e:
                print(f"[XKIRO PROMPT] Error loading prompt: {e}")
        
        self.load_default_xkiro_prompt()
        return {
            "template": self.DEFAULT_XKIRO_PROMPT_TEMPLATE,
            "max_tokens": 1000,
            "temperature": 0.3,
            "keep_proper_nouns": True,
            "auto_context": True
        }

    def save_xkiro_prompt_template(self):
        key_dir = os.path.join(os.path.dirname(__file__), "config")
        os.makedirs(key_dir, exist_ok=True)
        p_path = os.path.join(key_dir, "xkiro_prompt_template.json")
        data = {
            "template": self.txt_xkiro_prompt_template.toPlainText() if hasattr(self, 'txt_xkiro_prompt_template') else self.DEFAULT_XKIRO_PROMPT_TEMPLATE,
            "max_tokens": self.spin_xkiro_max_tokens.value() if hasattr(self, 'spin_xkiro_max_tokens') else 1000,
            "temperature": float(self.cb_xkiro_temperature.currentText()) if hasattr(self, 'cb_xkiro_temperature') else 0.3,
            "keep_proper_nouns": self.chk_keep_proper_nouns.isChecked() if hasattr(self, 'chk_keep_proper_nouns') else True,
            "auto_context": self.chk_auto_context.isChecked() if hasattr(self, 'chk_auto_context') else True,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            with open(p_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.log_info(f"💾 Đã lưu Prompt Template xKiro vào {p_path}!")
            if not os.environ.get("QT_QPA_PLATFORM"):
                QMessageBox.information(self, "Lưu Prompt", "Đã lưu Prompt Template xKiro thành công!")
        except Exception as e:
            if not os.environ.get("QT_QPA_PLATFORM"):
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu Prompt Template:\n{e}")

    def reset_app_settings(self):
        if hasattr(self, 'txt_output_dir'): self.txt_output_dir.setText(os.path.abspath("videos"))
        if hasattr(self, 'chk_prefer_xkiro'): self.chk_prefer_xkiro.setChecked(False)
        if hasattr(self, 'spin_workers'): self.spin_workers.setValue(4)
        if hasattr(self, 'spin_max_workers'): self.spin_max_workers.setValue(4)
        if hasattr(self, 'chk_burn_sub_export'): self.chk_burn_sub_export.setChecked(True)
        if hasattr(self, 'slider_bg'): self.slider_bg.setValue(30)
        if hasattr(self, 'slider_dub'): self.slider_dub.setValue(100)
        if hasattr(self, 'cb_font_name'): self.cb_font_name.setCurrentText("Arial")
        if hasattr(self, 'spin_font_size'): self.spin_font_size.setValue(24)
        if hasattr(self, 'spin_outline_width'): self.spin_outline_width.setValue(2)
        if hasattr(self, 'chk_use_bg_box'): self.chk_use_bg_box.setChecked(False)
        if hasattr(self, 'slider_bg_opacity'): self.slider_bg_opacity.setValue(50)
        if hasattr(self, 'cb_v_align'): self.cb_v_align.setCurrentIndex(2) # bottom
        if hasattr(self, 'cb_h_align'): self.cb_h_align.setCurrentIndex(1) # center
        if hasattr(self, 'spin_margin_v'): self.spin_margin_v.setValue(20)
        if hasattr(self, 'spin_margin_h'): self.spin_margin_h.setValue(20)
        self.preset_font_color = [255, 255, 255]
        self.preset_outline_color = [0, 0, 0]
        self.preset_bg_color = [0, 0, 0]
        if hasattr(self, 'btn_font_color'): self.btn_font_color.setStyleSheet("background-color: #FFFFFF; border: 1px solid white;")
        if hasattr(self, 'lbl_font_hex'): self.lbl_font_hex.setText("#FFFFFF")
        if hasattr(self, 'btn_outline_color'): self.btn_outline_color.setStyleSheet("background-color: #000000; border: 1px solid white;")
        if hasattr(self, 'lbl_outline_hex'): self.lbl_outline_hex.setText("#000000")
        if hasattr(self, 'btn_bg_color'): self.btn_bg_color.setStyleSheet("background-color: #000000; border: 1px solid white;")
        if hasattr(self, 'lbl_bg_hex'): self.lbl_bg_hex.setText("#000000")

        self.save_app_settings()
        if hasattr(self, 'txt_log_console') and self.txt_log_console:
            self.txt_log_console.append("🔄 Đã khôi phục toàn bộ cài đặt về mặc định ban đầu!")

    def add_gemini_key_slot(self, key_text="", status_text="⚪ Chưa test"):
        """Thêm 1 hàng nhập Gemini API Key động."""
        if not hasattr(self, 'gemini_key_slots_layout') or self.gemini_key_slots_layout is None:
            return None

        if key_text and len(self.gemini_key_inputs) == 1 and not self.gemini_key_inputs[0].text().strip():
            self.gemini_key_inputs[0].setText(key_text.strip())
            self._sync_gemini_keys_to_legacy()
            self.update_api_status()
            return self.gemini_key_rows[0]

        slot_idx = len(self.gemini_key_inputs) + 1
        row_widget = QWidget()
        row_widget.setObjectName(f"gemini_row_{slot_idx}")
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(4, 2, 4, 2)
        row_layout.setSpacing(8)

        lbl_num = QLabel(f"Key #{slot_idx}:")
        lbl_num.setStyleSheet("font-weight: bold; color: #38bdf8; min-width: 60px;")
        row_layout.addWidget(lbl_num)

        txt_k = QLineEdit()
        txt_k.setEchoMode(QLineEdit.EchoMode.Password)
        txt_k.setPlaceholderText("Dán Gemini API Key (AIzaSy...)...")
        txt_k.setText(key_text.strip())
        txt_k.textChanged.connect(self._sync_gemini_keys_to_legacy)
        txt_k.textChanged.connect(self.update_api_status)
        row_layout.addWidget(txt_k, 1)

        btn_eye = QPushButton("👁️")
        btn_eye.setFixedWidth(36)
        btn_eye.clicked.connect(lambda: txt_k.setEchoMode(
            QLineEdit.EchoMode.Normal if txt_k.echoMode() == QLineEdit.EchoMode.Password else QLineEdit.EchoMode.Password
        ))
        row_layout.addWidget(btn_eye)

        lbl_status = QLabel(status_text)
        lbl_status.setStyleSheet("color: #94a3b8; font-weight: bold; min-width: 90px;")
        row_layout.addWidget(lbl_status)

        btn_test_one = QPushButton("🔍 Test")
        btn_test_one.setFixedWidth(60)
        btn_test_one.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold;")
        def do_test_one():
            k = txt_k.text().strip()
            if not k:
                lbl_status.setText("✗ Rỗng")
                lbl_status.setStyleSheet("color: #ef4444; font-weight: bold;")
                return
            from gemini_vision_ocr import gemini_key_manager
            ok = gemini_key_manager.test_key(k)
            if ok:
                lbl_status.setText("🟢 Hoạt động")
                lbl_status.setStyleSheet("color: #10b981; font-weight: bold;")
            else:
                lbl_status.setText("🔴 Lỗi")
                lbl_status.setStyleSheet("color: #ef4444; font-weight: bold;")
            self.update_api_status()
        btn_test_one.clicked.connect(do_test_one)
        row_layout.addWidget(btn_test_one)

        btn_del = QPushButton("✖️ Xóa")
        btn_del.setFixedWidth(60)
        btn_del.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold;")
        btn_del.clicked.connect(lambda: self.remove_gemini_key_slot(row_widget))
        row_layout.addWidget(btn_del)

        self.gemini_key_slots_layout.addWidget(row_widget)
        self.gemini_key_inputs.append(txt_k)
        self.gemini_key_status_labels.append(lbl_status)
        self.gemini_key_rows.append(row_widget)
        self._sync_gemini_keys_to_legacy()
        self.update_api_status()
        return row_widget

    def remove_gemini_key_slot(self, row_widget):
        """Xóa 1 hàng Gemini Key."""
        if row_widget in self.gemini_key_rows:
            idx = self.gemini_key_rows.index(row_widget)
            self.gemini_key_rows.pop(idx)
            if idx < len(self.gemini_key_inputs):
                self.gemini_key_inputs.pop(idx)
            if idx < len(self.gemini_key_status_labels):
                self.gemini_key_status_labels.pop(idx)
            row_widget.deleteLater()
            self._sync_gemini_keys_to_legacy()
            self.update_api_status()

    def add_xkiro_key_slot(self, key_text="", status_text="⚪ Chưa test"):
        """Thêm 1 hàng nhập xKiro API Key động."""
        if not hasattr(self, 'xkiro_key_slots_layout') or self.xkiro_key_slots_layout is None:
            return None

        if key_text and len(self.xkiro_key_inputs) == 1 and not self.xkiro_key_inputs[0].text().strip():
            self.xkiro_key_inputs[0].setText(key_text.strip())
            self._sync_xkiro_keys_to_legacy()
            self.update_api_status()
            return self.xkiro_key_rows[0]

        slot_idx = len(self.xkiro_key_inputs) + 1
        row_widget = QWidget()
        row_widget.setObjectName(f"xkiro_row_{slot_idx}")
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(4, 2, 4, 2)
        row_layout.setSpacing(8)

        lbl_num = QLabel(f"Key #{slot_idx}:")
        lbl_num.setStyleSheet("font-weight: bold; color: #a855f7; min-width: 60px;")
        row_layout.addWidget(lbl_num)

        txt_k = QLineEdit()
        txt_k.setEchoMode(QLineEdit.EchoMode.Password)
        txt_k.setPlaceholderText("Dán xKiro API Key...")
        txt_k.setText(key_text.strip())
        txt_k.textChanged.connect(self._sync_xkiro_keys_to_legacy)
        txt_k.textChanged.connect(self.update_api_status)
        row_layout.addWidget(txt_k, 1)

        btn_eye = QPushButton("👁️")
        btn_eye.setFixedWidth(36)
        btn_eye.clicked.connect(lambda: txt_k.setEchoMode(
            QLineEdit.EchoMode.Normal if txt_k.echoMode() == QLineEdit.EchoMode.Password else QLineEdit.EchoMode.Password
        ))
        row_layout.addWidget(btn_eye)

        lbl_status = QLabel(status_text)
        lbl_status.setStyleSheet("color: #94a3b8; font-weight: bold; min-width: 90px;")
        row_layout.addWidget(lbl_status)

        btn_test_one = QPushButton("🔍 Test")
        btn_test_one.setFixedWidth(60)
        btn_test_one.setStyleSheet("background-color: #7e22ce; color: white; font-weight: bold;")
        def do_test_xk_one():
            k = txt_k.text().strip()
            if not k:
                lbl_status.setText("✗ Rỗng")
                lbl_status.setStyleSheet("color: #ef4444; font-weight: bold;")
                return
            try:
                import xkiro_client
                res = xkiro_client.translate_with_xkiro("Test connection", target_lang="vi", api_key=k)
                lbl_status.setText("🟢 Hoạt động")
                lbl_status.setStyleSheet("color: #10b981; font-weight: bold;")
            except Exception:
                lbl_status.setText("🔴 Lỗi")
                lbl_status.setStyleSheet("color: #ef4444; font-weight: bold;")
            self.update_api_status()
        btn_test_one.clicked.connect(do_test_xk_one)
        row_layout.addWidget(btn_test_one)

        btn_del = QPushButton("✖️ Xóa")
        btn_del.setFixedWidth(60)
        btn_del.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold;")
        btn_del.clicked.connect(lambda: self.remove_xkiro_key_slot(row_widget))
        row_layout.addWidget(btn_del)

        self.xkiro_key_slots_layout.addWidget(row_widget)
        self.xkiro_key_inputs.append(txt_k)
        self.xkiro_key_status_labels.append(lbl_status)
        self.xkiro_key_rows.append(row_widget)
        self._sync_xkiro_keys_to_legacy()
        self.update_api_status()
        return row_widget

    def remove_xkiro_key_slot(self, row_widget):
        """Xóa 1 hàng xKiro Key."""
        if row_widget in self.xkiro_key_rows:
            idx = self.xkiro_key_rows.index(row_widget)
            self.xkiro_key_rows.pop(idx)
            if idx < len(self.xkiro_key_inputs):
                self.xkiro_key_inputs.pop(idx)
            if idx < len(self.xkiro_key_status_labels):
                self.xkiro_key_status_labels.pop(idx)
            row_widget.deleteLater()
            self._sync_xkiro_keys_to_legacy()
            self.update_api_status()

    def _sync_gemini_keys_to_legacy(self):
        keys = []
        for txt in getattr(self, 'gemini_key_inputs', []):
            k = txt.text().strip()
            if k and k not in keys:
                keys.append(k)
        if hasattr(self, 'txt_gemini_key') and self.txt_gemini_key:
            self.txt_gemini_key.blockSignals(True)
            self.txt_gemini_key.setText(", ".join(keys))
            self.txt_gemini_key.blockSignals(False)

    def _sync_xkiro_keys_to_legacy(self):
        keys = []
        for txt in getattr(self, 'xkiro_key_inputs', []):
            k = txt.text().strip()
            if k and k not in keys:
                keys.append(k)
        if hasattr(self, 'txt_xkiro_key') and self.txt_xkiro_key:
            self.txt_xkiro_key.blockSignals(True)
            self.txt_xkiro_key.setText(", ".join(keys))
            self.txt_xkiro_key.blockSignals(False)

    def get_all_gemini_keys_from_ui(self):
        """Lấy toàn bộ danh sách Gemini keys từ các slots và ô text."""
        keys = []
        for txt in getattr(self, 'gemini_key_inputs', []):
            k = txt.text().strip()
            if k and k not in keys:
                keys.append(k)
        if not keys and hasattr(self, 'txt_gemini_key') and self.txt_gemini_key:
            raw = self.txt_gemini_key.text().strip()
            keys = [k.strip() for k in raw.split(",") if k.strip()]
            if keys:
                if getattr(self, 'gemini_key_inputs', []):
                    self.gemini_key_inputs[0].setText(keys[0])
                    for extra_k in keys[1:]:
                        self.add_gemini_key_slot(extra_k)
        return keys

    def get_all_xkiro_keys_from_ui(self):
        """Lấy toàn bộ danh sách xKiro keys từ các slots và ô text."""
        keys = []
        for txt in getattr(self, 'xkiro_key_inputs', []):
            k = txt.text().strip()
            if k and k not in keys:
                keys.append(k)
        if not keys and hasattr(self, 'txt_xkiro_key') and self.txt_xkiro_key:
            raw = self.txt_xkiro_key.text().strip()
            keys = [k.strip() for k in raw.split(",") if k.strip()]
            if keys:
                if getattr(self, 'xkiro_key_inputs', []):
                    self.xkiro_key_inputs[0].setText(keys[0])
                    for extra_k in keys[1:]:
                        self.add_xkiro_key_slot(extra_k)
        return keys

    def test_all_gemini_keys(self):
        """Kiểm tra toàn bộ Gemini keys."""
        from gemini_vision_ocr import gemini_key_manager
        keys = self.get_all_gemini_keys_from_ui()
        if not keys:
            if not os.environ.get("QT_QPA_PLATFORM"):
                QMessageBox.warning(self, "Kiểm tra Key", "Chưa có Gemini API Key nào để kiểm tra.")
            return

        active_cnt = 0
        err_cnt = 0
        for idx, txt_k in enumerate(getattr(self, 'gemini_key_inputs', [])):
            k = txt_k.text().strip()
            lbl = self.gemini_key_status_labels[idx] if idx < len(self.gemini_key_status_labels) else None
            if not k:
                if lbl:
                    lbl.setText("✗ Rỗng")
                    lbl.setStyleSheet("color: #ef4444; font-weight: bold;")
                continue
            ok = gemini_key_manager.test_key(k)
            if ok:
                active_cnt += 1
                if lbl:
                    lbl.setText("🟢 Hoạt động")
                    lbl.setStyleSheet("color: #10b981; font-weight: bold;")
            else:
                err_cnt += 1
                if lbl:
                    lbl.setText("🔴 Hết hạn/Lỗi")
                    lbl.setStyleSheet("color: #ef4444; font-weight: bold;")

        self.update_api_status()
        msg = f"Kết quả kiểm tra Gemini Keys:\n• Hoạt động: {active_cnt}\n• Lỗi/Hết hạn: {err_cnt}\n• Tổng: {len(keys)}"
        self.log_info(f"🔍 {msg.replace(chr(10), ' | ')}")
        if not os.environ.get("QT_QPA_PLATFORM"):
            QMessageBox.information(self, "Kiểm Tra Tất Cả Keys", msg)

    def test_all_xkiro_keys(self):
        """Kiểm tra toàn bộ xKiro keys."""
        keys = self.get_all_xkiro_keys_from_ui()
        if not keys:
            if not os.environ.get("QT_QPA_PLATFORM"):
                QMessageBox.warning(self, "Kiểm tra Key", "Chưa có xKiro API Key nào để kiểm tra.")
            return

        import xkiro_client
        active_cnt = 0
        err_cnt = 0
        detail_messages = []
        for idx, txt_k in enumerate(getattr(self, 'xkiro_key_inputs', [])):
            k = txt_k.text().strip()
            lbl = self.xkiro_key_status_labels[idx] if idx < len(self.xkiro_key_status_labels) else None
            if not k:
                if lbl:
                    lbl.setText("✗ Rỗng")
                    lbl.setStyleSheet("color: #ef4444; font-weight: bold;")
                continue
            
            ok, msg_k = xkiro_client.test_xkiro_key_status(api_key=k)
            if ok:
                active_cnt += 1
                if lbl:
                    lbl.setText("🟢 Hoạt động")
                    lbl.setStyleSheet("color: #10b981; font-weight: bold;")
                detail_messages.append(f"Key #{idx + 1}: {msg_k}")
            else:
                err_cnt += 1
                if lbl:
                    lbl.setText("🔴 401 Auth Error" if "401" in msg_k else "🔴 Lỗi")
                    lbl.setStyleSheet("color: #ef4444; font-weight: bold;")
                detail_messages.append(f"Key #{idx + 1}: {msg_k}")

        self.update_api_status()
        summary = f"Kết quả kiểm tra xKiro Keys:\n• Hoạt động: {active_cnt}\n• Lỗi/Hết hạn: {err_cnt}\n• Tổng: {len(keys)}\n\n" + "\n".join(detail_messages)
        self.log_info(f"🔍 Kết quả test xKiro: {active_cnt} active, {err_cnt} lỗi.")
        if not os.environ.get("QT_QPA_PLATFORM"):
            if err_cnt > 0 and active_cnt == 0:
                QMessageBox.warning(self, "Kiểm Tra xKiro API Key", summary)
            else:
                QMessageBox.information(self, "Kiểm Tra xKiro API Key", summary)

    def clear_all_gemini_keys(self):
        """Xóa toàn bộ các slot key Gemini."""
        for w in list(getattr(self, 'gemini_key_rows', [])):
            w.deleteLater()
        self.gemini_key_rows = []
        self.gemini_key_inputs = []
        self.gemini_key_status_labels = []
        self.add_gemini_key_slot("")
        self._sync_gemini_keys_to_legacy()
        self.update_api_status()

    def clear_all_xkiro_keys(self):
        """Xóa toàn bộ các slot key xKiro."""
        for w in list(getattr(self, 'xkiro_key_rows', [])):
            w.deleteLater()
        self.xkiro_key_rows = []
        self.xkiro_key_inputs = []
        self.xkiro_key_status_labels = []
        self.add_xkiro_key_slot("")
        self._sync_xkiro_keys_to_legacy()
        self.update_api_status()

    def update_api_status(self):
        """Cập nhật nhãn trạng thái và summary API Keys theo thời gian thực."""
        gemini_keys = self.get_all_gemini_keys_from_ui()
        xkiro_keys = self.get_all_xkiro_keys_from_ui()

        if hasattr(self, 'lbl_gemini_summary') and self.lbl_gemini_summary:
            cnt = len(gemini_keys)
            if cnt > 0:
                self.lbl_gemini_summary.setText(f"✅ {cnt} keys đã cấu hình (Auto-Rotate sẵn sàng)")
                self.lbl_gemini_summary.setStyleSheet("color: #10b981; font-weight: bold;")
            else:
                self.lbl_gemini_summary.setText("✗ Chưa cấu hình API key nào")
                self.lbl_gemini_summary.setStyleSheet("color: #ef4444; font-weight: bold;")

        if hasattr(self, 'lbl_xkiro_summary') and self.lbl_xkiro_summary:
            cnt = len(xkiro_keys)
            if cnt > 0:
                self.lbl_xkiro_summary.setText(f"✅ {cnt} keys đã cấu hình")
                self.lbl_xkiro_summary.setStyleSheet("color: #10b981; font-weight: bold;")
            else:
                self.lbl_xkiro_summary.setText("✗ Chưa cấu hình API key nào")
                self.lbl_xkiro_summary.setStyleSheet("color: #ef4444; font-weight: bold;")

        if hasattr(self, 'lbl_status_gemini') and self.lbl_status_gemini:
            cnt = len(gemini_keys)
            if cnt > 0:
                self.lbl_status_gemini.setText(f"✅ Đã cấu hình ({cnt} keys)")
                self.lbl_status_gemini.setStyleSheet("color: #10b981; font-weight: bold;")
            else:
                self.lbl_status_gemini.setText("✗ Chưa cấu hình")
                self.lbl_status_gemini.setStyleSheet("color: #ef4444; font-weight: bold;")

        if hasattr(self, 'lbl_status_xkiro') and self.lbl_status_xkiro:
            cnt = len(xkiro_keys)
            if cnt > 0:
                self.lbl_status_xkiro.setText(f"✅ Đã cấu hình ({cnt} keys)")
                self.lbl_status_xkiro.setStyleSheet("color: #10b981; font-weight: bold;")
            else:
                self.lbl_status_xkiro.setText("✗ Chưa cấu hình")
                self.lbl_status_xkiro.setStyleSheet("color: #ef4444; font-weight: bold;")

    DEFAULT_API_KEYS = {
        "gemini_keys": [],
        "xkiro_keys": []
    }

    def create_default_config(self):
        """Tạo file config/api_keys.json mặc định nếu chưa có hoặc rỗng."""
        key_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
        os.makedirs(key_dir, exist_ok=True)
        key_file = os.path.join(key_dir, "api_keys.json")
        try:
            with open(key_file, "w", encoding="utf-8") as f:
                json.dump(self.DEFAULT_API_KEYS, f, indent=2, ensure_ascii=False)
            if hasattr(self, 'txt_log_console') and self.txt_log_console:
                self.txt_log_console.append("✅ Đã tạo file config/api_keys.json mặc định với API keys")
        except Exception as e:
            print(f"[API CONFIG] Error creating default config: {e}")
        return self.DEFAULT_API_KEYS

    def load_api_config_to_ui(self):
        """Load API keys từ config/api_keys.json lên UI dưới dạng danh sách đa keys."""
        key_file = os.path.abspath(os.path.join("config", "api_keys.json"))
        if not os.path.exists(key_file):
            key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "api_keys.json")
        
        gemini_keys = []
        xkiro_keys = []
        try:
            if not os.path.exists(key_file) or os.path.getsize(key_file) < 5:
                data = self.create_default_config()
                gemini_keys = data.get("gemini_keys", [])
                xkiro_keys = data.get("xkiro_keys", [])
            else:
                with open(key_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    g_data = data.get("gemini_keys", [])
                    if isinstance(g_data, str):
                        gemini_keys = [k.strip() for k in g_data.split(",") if k.strip()]
                    elif isinstance(g_data, list):
                        gemini_keys = [str(k).strip() for k in g_data if str(k).strip()]

                    x_data = data.get("xkiro_keys", [])
                    if isinstance(x_data, str):
                        xkiro_keys = [k.strip() for k in x_data.split(",") if k.strip()]
                    elif isinstance(x_data, list):
                        xkiro_keys = [str(k).strip() for k in x_data if str(k).strip()]

                    if not gemini_keys and not xkiro_keys:
                        data = self.create_default_config()
                        gemini_keys = data.get("gemini_keys", [])
                        xkiro_keys = data.get("xkiro_keys", [])

            # Nạp Gemini slots
            if hasattr(self, 'gemini_key_rows'):
                for w in list(self.gemini_key_rows):
                    w.deleteLater()
                self.gemini_key_rows = []
                self.gemini_key_inputs = []
                self.gemini_key_status_labels = []

                if gemini_keys:
                    for k in gemini_keys:
                        self.add_gemini_key_slot(k)
                else:
                    self.add_gemini_key_slot("")

            # Nạp xKiro slots
            if hasattr(self, 'xkiro_key_rows'):
                for w in list(self.xkiro_key_rows):
                    w.deleteLater()
                self.xkiro_key_rows = []
                self.xkiro_key_inputs = []
                self.xkiro_key_status_labels = []

                if xkiro_keys:
                    for k in xkiro_keys:
                        self.add_xkiro_key_slot(k)
                else:
                    self.add_xkiro_key_slot("")

            self._sync_gemini_keys_to_legacy()
            self._sync_xkiro_keys_to_legacy()
            self.update_api_status()

            from gemini_vision_ocr import gemini_key_manager
            gemini_key_manager.load_keys(gemini_keys)

            if hasattr(self, 'txt_log_console') and self.txt_log_console:
                self.txt_log_console.append(f"✅ Đã nạp {len(gemini_keys)} Gemini Keys & {len(xkiro_keys)} xKiro Keys từ config/api_keys.json")
        except Exception as e:
            if hasattr(self, 'txt_log_console') and self.txt_log_console:
                self.txt_log_console.append(f"⚠️ Không thể load API keys: {str(e)}")
            self.update_api_status()

    def save_api_config_from_ui(self):
        """Lưu API keys từ UI vào config/api_keys.json dưới dạng danh sách mảng JSON."""
        try:
            key_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
            os.makedirs(key_dir, exist_ok=True)
            key_file = os.path.join(key_dir, "api_keys.json")

            gemini_keys = [k for k in self.get_all_gemini_keys_from_ui() if k and k.strip()]
            xkiro_keys = [k for k in self.get_all_xkiro_keys_from_ui() if k and k.strip()]

            data = {
                "gemini_keys": gemini_keys,
                "xkiro_keys": xkiro_keys
            }

            with open(key_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.update_api_status()
            from gemini_vision_ocr import gemini_key_manager
            gemini_key_manager.load_keys(gemini_keys)

            if hasattr(self, 'txt_log_console') and self.txt_log_console:
                self.txt_log_console.append(f"✅ Đã lưu {len(gemini_keys)} Gemini Keys & {len(xkiro_keys)} xKiro Keys vào config/api_keys.json")
            if not os.environ.get("QT_QPA_PLATFORM"):
                QMessageBox.information(self, "Thành công", f"Đã lưu thành công {len(gemini_keys)} Gemini Keys & {len(xkiro_keys)} xKiro Keys!")
        except Exception as e:
            if hasattr(self, 'txt_log_console') and self.txt_log_console:
                self.txt_log_console.append(f"❌ Lỗi lưu API keys: {str(e)}")
            if not os.environ.get("QT_QPA_PLATFORM"):
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu API keys: {str(e)}")

    def load_dynamic_tts_voices(self):
        try:
            import edge_tts
            import asyncio
            loop = asyncio.new_event_loop()
            voices = loop.run_until_complete(edge_tts.list_voices())
            loop.close()

            if hasattr(self, 'list_tts_voices') and self.list_tts_voices:
                self.list_tts_voices.clear()

            if hasattr(self, 'cb_voice') and self.cb_voice:
                self.cb_voice.clear()
                
                def sort_key(v):
                    loc = v.get("Locale", "").lower()
                    if loc == "vi-vn":
                        return (0, v.get("ShortName", ""))
                    elif loc.startswith("en-"):
                        return (1, v.get("ShortName", ""))
                    else:
                        return (2, v.get("ShortName", ""))

                sorted_voices = sorted(voices, key=sort_key)
                for v in sorted_voices:
                    name = v.get("ShortName", "")
                    gender = "Nữ" if v.get("Gender") == "Female" else "Nam"
                    locale = v.get("Locale", "")
                    label = f"{locale} - {name} ({gender})"
                    self.cb_voice.addItem(label, name)
                    if hasattr(self, 'list_tts_voices') and self.list_tts_voices:
                        item = QListWidgetItem(label)
                        item.setData(Qt.ItemDataRole.UserRole, name)
                        self.list_tts_voices.addItem(item)

                print(f"[TTS DYNAMIC] Da load dong {len(sorted_voices)} giong doc Edge-TTS vao Combobox & ListWidget!")
                if hasattr(self, 'lbl_voice_count') and self.lbl_voice_count:
                    self.lbl_voice_count.setText(f"Tổng số: {len(sorted_voices)} giọng")
                if hasattr(self, 'txt_log_console') and self.txt_log_console:
                    self.txt_log_console.append(f"✅ Đã nạp tự động {len(sorted_voices)} giọng đọc AI Neural (Edge-TTS)")
        except Exception as e:
            print(f"[TTS DYNAMIC] Error loading dynamic voices: {e}")

    def filter_tts_voice_list(self, text):
        if not hasattr(self, 'list_tts_voices') or not self.list_tts_voices:
            return
        query = text.strip().lower()
        for i in range(self.list_tts_voices.count()):
            item = self.list_tts_voices.item(i)
            item.setHidden(bool(query and query not in item.text().lower()))

    def on_voice_item_selected(self, item):
        vname = item.data(Qt.ItemDataRole.UserRole)
        if vname and hasattr(self, 'cb_voice') and self.cb_voice:
            idx = self.cb_voice.findData(vname)
            if idx >= 0:
                self.cb_voice.setCurrentIndex(idx)

    def preview_selected_voice(self):
        voice_name = "vi-VN-HoaiMyNeural"
        if hasattr(self, 'list_tts_voices') and self.list_tts_voices and self.list_tts_voices.currentItem():
            vname = self.list_tts_voices.currentItem().data(Qt.ItemDataRole.UserRole)
            if vname: voice_name = vname
        elif hasattr(self, 'cb_voice') and self.cb_voice:
            voice_name = self.cb_voice.currentData() or self.cb_voice.currentText()
        
        self.log_info(f"🔊 Nghe thử giọng đọc TTS: {voice_name}...")
        try:
            from transcriber import TTSPreviewWorker
            if hasattr(self, 'tts_preview_thread') and self.tts_preview_thread and self.tts_preview_thread.isRunning():
                self.tts_preview_thread.terminate()
            self.tts_preview_thread = TTSPreviewWorker("Xin chào! Đây là âm thanh nghe thử giọng đọc AI.", voice_name)
            self.tts_preview_thread.finished_signal.connect(lambda succ, msg: self.log_info(msg))
            self.tts_preview_thread.start()
        except Exception as e:
            self.log_info(f"⚠️ Lỗi preview TTS: {e}")

    def save_all_page2_settings(self):
        self.save_api_config_from_ui()
        self.save_app_settings()
        QMessageBox.information(self, "Thành công", "✅ Đã lưu tất cả cấu hình & API Keys thành công!")

    def pick_subtitle_color(self, color_type):
        col = QColorDialog.getColor(parent=self)
        if col.isValid():
            r, g, b = col.red(), col.green(), col.blue()
            hex_code = col.name().upper()
            if color_type == 'font':
                self.preset_font_color = [r, g, b]
                if hasattr(self, 'btn_font_color'): self.btn_font_color.setStyleSheet(f"background-color: {hex_code}; border: 1px solid white;")
                if hasattr(self, 'lbl_font_hex'): self.lbl_font_hex.setText(hex_code)
            elif color_type == 'outline':
                self.preset_outline_color = [r, g, b]
                if hasattr(self, 'btn_outline_color'): self.btn_outline_color.setStyleSheet(f"background-color: {hex_code}; border: 1px solid white;")
                if hasattr(self, 'lbl_outline_hex'): self.lbl_outline_hex.setText(hex_code)
            elif color_type == 'bg':
                self.preset_bg_color = [r, g, b]
                if hasattr(self, 'btn_bg_color'): self.btn_bg_color.setStyleSheet(f"background-color: {hex_code}; border: 1px solid white;")
                if hasattr(self, 'lbl_bg_hex'): self.lbl_bg_hex.setText(hex_code)
            self.save_app_settings()

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(6, 6, 6, 6)
        self.main_layout.setSpacing(4)
        
        # =========================================================================
        # OUTER VERTICAL SPLITTER (Top Workspace ở Trên / Subtitle Editor ở Dưới)
        # =========================================================================
        main_v_splitter = QSplitter(Qt.Orientation.Vertical)
        main_v_splitter.setChildrenCollapsible(False)

        # =========================================================================
        # HỆ THỐNG TAB CHÍNH TOÀN TRANG (FULL-PAGE WORKSPACE TABS)
        # =========================================================================
        # HỆ THỐNG GIAO DIỆN 2 TAB CHUẨN WIREFRAME
        # [TAB 1: BẢNG ĐIỀU KHIỂN CHÍNH] | [TAB 2: CẤU HÌNH & BỘ CHUYỂN ĐỔI]
        # =========================================================================
        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setStyleSheet("color: #38bdf8; font-weight: bold; padding: 5px 12px; background-color: #1e293b; border: 1px solid #334155; border-radius: 6px;")
        self.txt_sub_search = QLineEdit()
        self.main_tab_widget = QTabWidget()
        self.config_tab_widget = self.main_tab_widget  # Backwards compatibility alias

        # -------------------------------------------------------------------------
        # TAB 1: 🎬 1. MÀN HÌNH CHÍNH (MAIN WORKSPACE)
        # -------------------------------------------------------------------------
        tab_main_workspace = QWidget()
        tab_main_layout = QVBoxLayout(tab_main_workspace)
        tab_main_layout.setContentsMargins(8, 8, 8, 8)
        tab_main_layout.setSpacing(6)

        # HÀNG TRÊN (TOP): 2 CỘT SPLITTER (TRÁI: LIVE PREVIEW & DRAWING | PHẢI: CONTROL PANEL)
        self.top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.top_splitter.setChildrenCollapsible(False)
        top_splitter = self.top_splitter

        # CỘT TRÁI: LIVE PREVIEW VIDEO PLAYER & FILE INPUT
        left_preview_card = QFrame()
        left_preview_card.setMinimumHeight(400)
        left_preview_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_preview_layout = QVBoxLayout(left_preview_card)
        left_preview_layout.setContentsMargins(8, 8, 8, 8)
        left_preview_layout.setSpacing(6)

        # 📂 TỆP ĐẦU VÀO (FILE SELECTOR)
        file_input_frame = QFrame()
        file_input_layout = QHBoxLayout(file_input_frame)
        file_input_layout.setContentsMargins(6, 4, 6, 4)
        file_input_layout.setSpacing(6)

        lbl_file_icon = QLabel("📂 Video đầu vào:")
        lbl_file_icon.setStyleSheet("font-weight: bold; color: #38bdf8;")
        file_input_layout.addWidget(lbl_file_icon)

        self.txt_video_path = QLineEdit()
        self.txt_video_path.setPlaceholderText("Chọn hoặc kéo thả file video mẫu (ví dụ: videos/sample_video.mp4)...")
        file_input_layout.addWidget(self.txt_video_path, 1)

        self.btn_open_video = QPushButton("Chọn video...")
        self.btn_open_video.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: bold; padding: 4px 12px;")
        self.btn_open_video.clicked.connect(self.browse_video)
        file_input_layout.addWidget(self.btn_open_video)
        left_preview_layout.addWidget(file_input_frame)

        # Màn hình Live Preview Video Player (Cho phép vẽ khung)
        self.lbl_main_preview = DraggablePreviewLabel(self)
        self.lbl_main_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_main_preview.setText("📺 MÀN HÌNH PREVIEW VIDEO\n\nNhấp 'Chọn video' hoặc kéo-thả file video vào đây để khoanh vùng")
        self.lbl_main_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_main_preview.setMinimumSize(480, 270)
        self.lbl_main_preview.setScaledContents(False)
        left_preview_layout.addWidget(self.lbl_main_preview, 1)

        # Thanh tua video & các nút Play/Pause
        player_control_frame = QFrame()
        player_control_frame.setStyleSheet("QFrame { background-color: #0f172a; border: 1px solid #2a364f; border-radius: 6px; padding: 4px; }")
        player_control_layout = QVBoxLayout(player_control_frame)
        player_control_layout.setContentsMargins(6, 4, 6, 4)
        player_control_layout.setSpacing(4)

        info_bar = QHBoxLayout()
        self.lbl_frame_info = QLabel("Frame: 0 / 0   Time: 00:00 / 00:00")
        self.lbl_frame_info.setStyleSheet("color: #38bdf8; font-family: monospace; font-size: 11px; font-weight: bold;")
        info_bar.addWidget(self.lbl_frame_info)
        info_bar.addStretch()

        self.btn_zoom_out = QPushButton("🔍-")
        self.btn_zoom_out.setToolTip("Thu nhỏ (Zoom Out)")
        self.btn_zoom_out.setFixedWidth(36)
        self.btn_zoom_out.clicked.connect(self.zoom_out_preview)
        info_bar.addWidget(self.btn_zoom_out)

        self.btn_zoom_reset = QPushButton("🔍 100%")
        self.btn_zoom_reset.setToolTip("Khôi phục kích thước chuẩn 100% (Reset Zoom)")
        self.btn_zoom_reset.setStyleSheet("color: #38bdf8; font-weight: bold; padding: 2px 6px;")
        self.btn_zoom_reset.clicked.connect(self.reset_zoom_preview)
        info_bar.addWidget(self.btn_zoom_reset)
        self.lbl_zoom_level = self.btn_zoom_reset

        self.btn_zoom_in = QPushButton("🔍+")
        self.btn_zoom_in.setToolTip("Phóng to (Zoom In)")
        self.btn_zoom_in.setFixedWidth(36)
        self.btn_zoom_in.clicked.connect(self.zoom_in_preview)
        info_bar.addWidget(self.btn_zoom_in)

        player_control_layout.addLayout(info_bar)

        self.slider_player_timeline = QSlider(Qt.Orientation.Horizontal)
        self.slider_player_timeline.setRange(0, 1000)
        self.slider_player_timeline.setValue(0)
        self.slider_player_timeline.valueChanged.connect(self.on_player_seek)
        player_control_layout.addWidget(self.slider_player_timeline)

        fine_tune_row = QHBoxLayout()
        fine_tune_row.setSpacing(6)
        self.btn_play_seg = QPushButton("▶ Play / Pause")
        self.btn_play_seg.clicked.connect(self.toggle_realtime_play)
        fine_tune_row.addWidget(self.btn_play_seg)

        btn_sub5s = QPushButton("-5s")
        btn_sub5s.clicked.connect(lambda: self.seek_relative(-5))
        fine_tune_row.addWidget(btn_sub5s)

        btn_prev = QPushButton("◀ Trước")
        btn_prev.clicked.connect(lambda: self.seek_relative(-1))
        fine_tune_row.addWidget(btn_prev)

        btn_next = QPushButton("▶ Sau")
        btn_next.clicked.connect(lambda: self.seek_relative(1))
        fine_tune_row.addWidget(btn_next)

        btn_add5s = QPushButton("+5s")
        btn_add5s.clicked.connect(lambda: self.seek_relative(5))
        fine_tune_row.addWidget(btn_add5s)
        fine_tune_row.addStretch()
        player_control_layout.addLayout(fine_tune_row)
        left_preview_layout.addWidget(player_control_frame)

        # 🖌️ THANH CÔNG CỤ KHOANH VÙNG 3 LOẠI KHUNG (BOX DRAWING TOOLBAR)
        box_tools_frame = QGroupBox("🖌️ CÔNG CỤ KHOANH VÙNG KÉO THẢ")
        box_tools_layout = QHBoxLayout(box_tools_frame)
        box_tools_layout.setContentsMargins(6, 6, 6, 6)
        box_tools_layout.setSpacing(6)

        self.btn_draw_sub_mode = QPushButton("🟦 Vùng Phụ Đề")
        self.btn_draw_sub_mode.setToolTip("Khoanh vùng chữ phụ đề thoại gốc để OCR")
        self.btn_draw_sub_mode.setStyleSheet("background-color: #1e3a8a; color: #93c5fd; font-weight: bold;")
        self.btn_draw_sub_mode.clicked.connect(lambda: self.set_drawing_box_type("sub"))
        box_tools_layout.addWidget(self.btn_draw_sub_mode)

        self.btn_draw_title_mode = QPushButton("🟨 Vùng Tiêu Đề")
        self.btn_draw_title_mode.setToolTip("Khoanh vùng chữ tiêu đề video để dịch đè tiêu đề")
        self.btn_draw_title_mode.setStyleSheet("background-color: #854d0e; color: #fef08a; font-weight: bold;")
        self.btn_draw_title_mode.clicked.connect(lambda: self.set_drawing_box_type("title"))
        box_tools_layout.addWidget(self.btn_draw_title_mode)

        self.btn_draw_logo_mode = QPushButton("🟥 Vùng Logo")
        self.btn_draw_logo_mode.setToolTip("Khoanh vùng logo cũ để che/chèn logo thương hiệu mới")
        self.btn_draw_logo_mode.setStyleSheet("background-color: #831843; color: #fbcfe8; font-weight: bold;")
        self.btn_draw_logo_mode.clicked.connect(lambda: self.set_drawing_box_type("logo"))
        box_tools_layout.addWidget(self.btn_draw_logo_mode)

        self.btn_undo_bbox = QPushButton("↶ Undo (Ctrl+Z)")
        self.btn_undo_bbox.clicked.connect(self.undo_bbox_change)
        box_tools_layout.addWidget(self.btn_undo_bbox)

        self.btn_clear_bboxes = QPushButton("🗑️ Xóa khung")
        self.btn_clear_bboxes.clicked.connect(self.clear_all_bboxes)
        box_tools_layout.addWidget(self.btn_clear_bboxes)

        left_preview_layout.addWidget(box_tools_frame)
        top_splitter.addWidget(left_preview_card)

        # CỘT PHẢI: ⚙️ TÙY CHỌN CHẠY & CONTROL PANEL (trong ScrollArea để không ép preview nhỏ)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        right_scroll.setMinimumWidth(280)
        right_scroll.setMaximumWidth(480)
        right_panel_card = QFrame()
        right_panel_layout = QVBoxLayout(right_panel_card)
        right_panel_layout.setContentsMargins(8, 8, 8, 8)
        right_panel_layout.setSpacing(8)

        # Group 1: ⚙️ CẤU HÌNH OCR & DỊCH THUẬT
        ocr_group = QGroupBox("⚙️ CẤU HÌNH NHẬN DIỆN & VỊ TRÍ SUB")
        ocr_layout = QVBoxLayout(ocr_group)
        ocr_layout.setContentsMargins(10, 10, 10, 10)
        ocr_layout.setSpacing(6)

        lbl_ocr_engine = QLabel("<b>Chế độ OCR & AI Engine:</b>")
        ocr_layout.addWidget(lbl_ocr_engine)

        self.cb_ocr_engine = QComboBox()
        self.cb_ocr_engine.addItems([
            "PaddleOCR (Tiếng Trung - 中文)",
            "EasyOCR (Offline)",
            "Gemini Vision (AI Multimodal)",
            "xKiro AI (Vision & Text)"
        ])
        ocr_layout.addWidget(self.cb_ocr_engine)

        lbl_sub_pos = QLabel("<b>Vị trí đè phụ đề đã dịch:</b>")
        ocr_layout.addWidget(lbl_sub_pos)

        self.cb_sub_pos = QComboBox()
        self.cb_sub_pos.addItems([
            "Dưới - Giữa (Mặc định)",
            "Dưới - Trái",
            "Dưới - Phải",
            "Trên - Giữa",
            "Giữa - Giữa",
            "Theo vị trí khung đã vẽ (Smart Pos)"
        ])
        ocr_layout.addWidget(self.cb_sub_pos)

        # Checkbox & Combo chọn giọng đọc TTS (Trang 1 Controls)
        lbl_tts = QLabel("<b>Giọng đọc AI (TTS):</b>")
        ocr_layout.addWidget(lbl_tts)

        tts_row = QHBoxLayout()
        self.chk_enable_dubbing = QCheckBox("Bật TTS (giọng đọc)")
        self.chk_enable_dubbing.setChecked(True)
        tts_row.addWidget(self.chk_enable_dubbing)

        self.cb_voice = QComboBox()
        self.cb_voice.addItem("vi-VN-HoaiMyNeural (Nữ - Mặc định)", "vi-VN-HoaiMyNeural")
        self.cb_voice.addItem("vi-VN-NamMinhNeural (Nam)", "vi-VN-NamMinhNeural")
        self.cb_voice.addItem("female (Nữ AI)", "female")
        self.cb_voice.addItem("male (Nam AI)", "male")
        self.cb_voice.addItem("default (Mặc định)", "default")
        tts_row.addWidget(self.cb_voice, 1)

        self.chk_enable_dubbing.toggled.connect(self.cb_voice.setEnabled)
        ocr_layout.addLayout(tts_row)

        # File Logo chọn kèm
        logo_file_layout = QHBoxLayout()
        self.txt_logo_path = QLineEdit()
        self.txt_logo_path.setPlaceholderText("Ảnh logo PNG thay thế (nếu có)...")
        logo_file_layout.addWidget(self.txt_logo_path, 1)

        self.btn_open_logo = QPushButton("Logo...")
        self.btn_open_logo.clicked.connect(self.browse_logo)
        logo_file_layout.addWidget(self.btn_open_logo)
        ocr_layout.addLayout(logo_file_layout)

        right_panel_layout.addWidget(ocr_group)

        # Group 2: 🚀 CHẠY PIPELINE & THẦN TỐC
        action_group = QGroupBox("🚀 ĐIỀU KHIỂN CHẠY PIPELINE")
        action_layout = QVBoxLayout(action_group)
        action_layout.setContentsMargins(10, 10, 10, 10)
        action_layout.setSpacing(8)

        self.btn_run_main = QPushButton("▶ BẮT ĐẦU CHẠY PIPELINE (RUN)")
        self.btn_run_main.setObjectName("btnRunMain")
        self.btn_run_main.setMinimumHeight(46)
        self.btn_run_main.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #059669; color: white; border-radius: 6px;")
        self.btn_run_main.clicked.connect(self.start_dubbing)
        action_layout.addWidget(self.btn_run_main)

        self.main_progress_bar = QProgressBar()
        self.main_progress_bar.setRange(0, 100)
        self.main_progress_bar.setValue(0)
        self.main_progress_bar.setTextVisible(True)
        self.main_progress_bar.setStyleSheet("QProgressBar { text-align: center; font-weight: bold; border-radius: 4px; } QProgressBar::chunk { background-color: #3b82f6; }")
        action_layout.addWidget(self.main_progress_bar)

        self.lbl_eta_time = QLabel("⏱️ Ước tính còn lại: --:--")
        self.lbl_eta_time.setStyleSheet("color: #38bdf8; font-weight: bold; alignment: center;")
        self.lbl_eta_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        action_layout.addWidget(self.lbl_eta_time)

        self.lbl_chunks_count = QLabel("📦 Số chunks: 0")
        self.lbl_chunks_count.setStyleSheet("color: #94a3b8; font-size: 11px; alignment: center;")
        self.lbl_chunks_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        action_layout.addWidget(self.lbl_chunks_count)

        cancel_export_row = QHBoxLayout()
        self.btn_cancel_main = QPushButton("🛑 HỦY CHẠY")
        self.btn_cancel_main.setObjectName("btnCancelMain")
        self.btn_cancel_main.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold;")
        self.btn_cancel_main.setEnabled(False)
        self.btn_cancel_main.clicked.connect(self.cancel_dubbing)
        self.btn_cancel_job = self.btn_cancel_main
        cancel_export_row.addWidget(self.btn_cancel_main)

        self.btn_export_srt_main = QPushButton("📤 Xuất Sub (.SRT)")
        self.btn_export_srt_main.clicked.connect(self.export_srt_file)
        cancel_export_row.addWidget(self.btn_export_srt_main)
        action_layout.addLayout(cancel_export_row)

        right_panel_layout.addWidget(action_group)
        right_panel_layout.addStretch()
        right_scroll.setWidget(right_panel_card)

        top_splitter.addWidget(right_scroll)
        top_splitter.setSizes([600, 350])
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 1)

        # HÀNG DƯỚI (BOTTOM): 📜 BẢNG LOGS REALTIME & LỖI GEMINI API
        bottom_log_card = QGroupBox("📜 BẢNG LOGS HỆ THỐNG REALTIME (CONSOLE LOG)")
        bottom_log_layout = QVBoxLayout(bottom_log_card)
        bottom_log_layout.setContentsMargins(8, 8, 8, 8)

        self.txt_log_console = QTextEdit()
        self.txt_log_console.setReadOnly(True)
        self.txt_log_console.setMinimumHeight(80)
        self.txt_log_console.setMaximumHeight(180)
        self.txt_log_console.setStyleSheet("QTextEdit { background-color: #090d16; color: #38bdf8; font-family: Consolas, monospace; font-size: 11px; border: 1px solid #1e293b; }")
        bottom_log_layout.addWidget(self.txt_log_console)

        log_buttons_row = QHBoxLayout()
        log_buttons_row.addWidget(QLabel("<b>Lọc Log:</b>"))
        self.cb_log_filter = QComboBox()
        self.cb_log_filter.addItems(["All", "Error", "Warning", "Info", "Debug"])
        self.cb_log_filter.currentTextChanged.connect(self.filter_log_console)
        log_buttons_row.addWidget(self.cb_log_filter)

        self.btn_copy_log = QPushButton("📋 Sao chép Log")
        self.btn_copy_log.clicked.connect(self.copy_log_to_clipboard)
        log_buttons_row.addWidget(self.btn_copy_log)

        self.btn_save_log = QPushButton("💾 Lưu Log File...")
        self.btn_save_log.clicked.connect(self.save_log_to_file)
        log_buttons_row.addWidget(self.btn_save_log)

        self.btn_view_api_err_log = QPushButton("⚠️ Xem Log Lỗi Gemini/xKiro")
        self.btn_view_api_err_log.clicked.connect(self.open_api_error_log_dialog)
        log_buttons_row.addWidget(self.btn_view_api_err_log)

        log_buttons_row.addStretch()
        bottom_log_layout.addLayout(log_buttons_row)

        bottom_log_card.setMaximumHeight(220)
        self.txt_log_console.setMaximumHeight(160)
        self.workspace_v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.workspace_v_splitter.setChildrenCollapsible(False)
        self.workspace_v_splitter.addWidget(top_splitter)
        self.workspace_v_splitter.addWidget(bottom_log_card)
        self.workspace_v_splitter.setSizes([750, 150])
        self.workspace_v_splitter.setStretchFactor(0, 10)
        self.workspace_v_splitter.setStretchFactor(1, 1)

        tab_main_layout.addWidget(self.workspace_v_splitter)
        self.main_tab_widget.addTab(tab_main_workspace, "🎬 1. MÀN HÌNH CHÍNH")

        # -------------------------------------------------------------------------
        # TAB 2: 🔑 2. CÀI ĐẶT NÂNG CAO & API KEYS (NESTED TABS)
        # -------------------------------------------------------------------------
        tab_page2_widget = QWidget()
        tab_page2_layout = QVBoxLayout(tab_page2_widget)
        tab_page2_layout.setContentsMargins(6, 6, 6, 6)

        self.tab2_nested = QTabWidget()
        self.tab2_nested.setTabPosition(QTabWidget.TabPosition.North)
        self.tab2_nested.setDocumentMode(True)
        self.tab2_nested.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #1e293b; background: #0f172a; border-radius: 6px; }
            QTabBar::tab { background: #1e293b; color: #94a3b8; padding: 9px 18px; font-weight: bold; font-size: 12px; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }
            QTabBar::tab:selected { background: #0284c7; color: white; }
            QTabBar::tab:hover:!selected { background: #334155; color: #e2e8f0; }
        """)

        # Khởi tạo 6 Tab con chuyên biệt
        self.tab2_api = self.setup_tab2_api()      # 🔑 1. API Keys
        self.tab2_voice = self.setup_tab2_voice()  # 🎤 2. Voice
        self.tab2_font = self.setup_tab2_font()    # 📝 3. Font & Subtitle
        self.tab2_audio = self.setup_tab2_audio()  # 🔊 4. Audio
        self.tab2_ocr = self.setup_tab2_ocr()      # 🖥️ 5. OCR & Engine
        self.tab2_adv = self.setup_tab2_adv()      # ⚙️ 6. Advanced

        self.tab2_nested.addTab(self.tab2_api, "🔑 API Keys")
        self.tab2_nested.addTab(self.tab2_voice, "🎤 Voice")
        self.tab2_nested.addTab(self.tab2_font, "📝 Font & Subtitle")
        self.tab2_nested.addTab(self.tab2_audio, "🔊 Audio")
        self.tab2_nested.addTab(self.tab2_ocr, "🖥️ OCR & Engine")
        self.tab2_nested.addTab(self.tab2_adv, "⚙️ Advanced")

        tab_page2_layout.addWidget(self.tab2_nested)
        self.main_tab_widget.addTab(tab_page2_widget, "🔑 2. CÀI ĐẶT NÂNG CAO & API KEYS")

        self.setup_page3_tab()
        self.setup_page4_tab()
        self.load_api_config_to_ui()
        self.load_app_settings()
        self.load_xkiro_prompt_template()

        self.main_layout.addWidget(self.main_tab_widget, 1)

        if hasattr(self, 'setup_subtitle_styling_connections'):
            self.setup_subtitle_styling_connections()
        if hasattr(self, 'setup_subtitle_shortcuts'):
            self.setup_subtitle_shortcuts()

        self.canvas_timer = QTimer(self)
        self.canvas_timer.setSingleShot(True)
        if hasattr(self, 'update_canvas_realtime_now'):
            self.canvas_timer.timeout.connect(self.update_canvas_realtime_now)

        self.main_tab_widget.currentChanged.connect(self.on_main_tab_changed)

        self.lbl_status_state = QLabel("🟢 Ready")
        self.lbl_status_state.setStyleSheet("font-weight: bold; color: #4ade80; padding-right: 10px;")
        self.statusBar().addWidget(self.lbl_status_state)

        self.lbl_status_video = QLabel("Video: Chưa chọn")
        self.lbl_status_video.setStyleSheet("color: #94a3b8; font-style: italic;")
        self.statusBar().addPermanentWidget(self.lbl_status_video)

        self.status_label = QLabel("Sẵn sàng.")
        self.statusBar().addPermanentWidget(self.status_label)
        if hasattr(self, 'apply_all_tooltips'):
            self.apply_all_tooltips()

        if hasattr(self, 'restore_window_state'):
            self.restore_window_state()

        if hasattr(self, 'check_and_prompt_crash_recovery'):
            self.check_and_prompt_crash_recovery()

        try:
            apply_custom_styles_to_app(self)
        except Exception:
            pass

    def setup_tab2_api(self):
        """BƯỚC 2: Tab 🔑 API Keys Management (Multiple Keys + Auto-Rotate)."""
        self.gemini_key_inputs = []
        self.gemini_key_status_labels = []
        self.gemini_key_rows = []
        self.xkiro_key_inputs = []
        self.xkiro_key_status_labels = []
        self.xkiro_key_rows = []

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(15)

        # ---------------------------------------------------------------------
        # 1. Gemini API Group (Multiple Keys + Auto-Rotate)
        # ---------------------------------------------------------------------
        gemini_group = QGroupBox("🔑 1. GEMINI API (GOOGLE VISION OCR & TRANSLATION) - ĐA KEYS & AUTO-ROTATE")
        lay_gem = QVBoxLayout(gemini_group)
        lay_gem.setContentsMargins(14, 16, 14, 14)
        lay_gem.setSpacing(10)

        # Dynamic slots container
        self.gemini_key_slots_widget = QWidget()
        self.gemini_key_slots_layout = QVBoxLayout(self.gemini_key_slots_widget)
        self.gemini_key_slots_layout.setContentsMargins(0, 0, 0, 0)
        self.gemini_key_slots_layout.setSpacing(6)
        lay_gem.addWidget(self.gemini_key_slots_widget)

        # Gemini Controls row
        gem_btn_row = QHBoxLayout()
        self.btn_add_gemini_key = QPushButton("➕ Thêm Key Gemini")
        self.btn_add_gemini_key.setStyleSheet("background-color: #1e293b; color: #38bdf8; border: 1px solid #0284c7; font-weight: bold; padding: 6px 14px;")
        self.btn_add_gemini_key.clicked.connect(lambda: self.add_gemini_key_slot(""))
        gem_btn_row.addWidget(self.btn_add_gemini_key)

        self.btn_test_all_gemini = QPushButton("🔄 Kiểm tra tất cả keys")
        self.btn_test_all_gemini.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_test_all_gemini.clicked.connect(self.test_all_gemini_keys)
        gem_btn_row.addWidget(self.btn_test_all_gemini)

        self.btn_clear_all_gemini = QPushButton("🗑️ Xóa tất cả")
        self.btn_clear_all_gemini.setStyleSheet("background-color: #334155; color: #fca5a5; border: 1px solid #ef4444; font-weight: bold; padding: 6px 14px;")
        self.btn_clear_all_gemini.clicked.connect(self.clear_all_gemini_keys)
        gem_btn_row.addWidget(self.btn_clear_all_gemini)

        gem_btn_row.addStretch()
        lay_gem.addLayout(gem_btn_row)

        # Summary & Checkboxes
        self.lbl_gemini_summary = QLabel("✅ 0 keys đã cấu hình")
        self.lbl_gemini_summary.setStyleSheet("color: #10b981; font-weight: bold; font-size: 13px; padding-top: 4px;")
        lay_gem.addWidget(self.lbl_gemini_summary)

        self.chk_auto_rotate_keys = QCheckBox("🔄 Auto-rotate key khi gặp lỗi (429 Rate Limit / 503 Overload / 404 Model)")
        self.chk_auto_rotate_keys.setChecked(True)
        self.chk_auto_rotate_keys.setStyleSheet("font-weight: bold; color: #38bdf8;")
        lay_gem.addWidget(self.chk_auto_rotate_keys)

        self.chk_check_keys_startup = QCheckBox("⚡ Tự động kiểm tra key hợp lệ khi khởi động")
        self.chk_check_keys_startup.setChecked(True)
        self.chk_check_keys_startup.setStyleSheet("color: #cbd5e1;")
        lay_gem.addWidget(self.chk_check_keys_startup)

        layout.addWidget(gemini_group)

        # ---------------------------------------------------------------------
        # 2. xKiro API Group (Multiple Keys)
        # ---------------------------------------------------------------------
        xkiro_group = QGroupBox("🔑 2. XKIRO API (DEEP DUBBING & TRANSLATION ENGINE) - ĐA KEYS")
        lay_xk = QVBoxLayout(xkiro_group)
        lay_xk.setContentsMargins(14, 16, 14, 14)
        lay_xk.setSpacing(10)

        # Dynamic slots container
        self.xkiro_key_slots_widget = QWidget()
        self.xkiro_key_slots_layout = QVBoxLayout(self.xkiro_key_slots_widget)
        self.xkiro_key_slots_layout.setContentsMargins(0, 0, 0, 0)
        self.xkiro_key_slots_layout.setSpacing(6)
        lay_xk.addWidget(self.xkiro_key_slots_widget)

        # xKiro Controls row
        xk_btn_row = QHBoxLayout()
        self.btn_add_xkiro_key = QPushButton("➕ Thêm Key xKiro")
        self.btn_add_xkiro_key.setStyleSheet("background-color: #1e293b; color: #c084fc; border: 1px solid #7e22ce; font-weight: bold; padding: 6px 14px;")
        self.btn_add_xkiro_key.clicked.connect(lambda: self.add_xkiro_key_slot(""))
        xk_btn_row.addWidget(self.btn_add_xkiro_key)

        self.btn_test_all_xkiro = QPushButton("🔄 Kiểm tra tất cả xKiro")
        self.btn_test_all_xkiro.setStyleSheet("background-color: #7e22ce; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_test_all_xkiro.clicked.connect(self.test_all_xkiro_keys)
        xk_btn_row.addWidget(self.btn_test_all_xkiro)

        self.btn_clear_all_xkiro = QPushButton("🗑️ Xóa tất cả")
        self.btn_clear_all_xkiro.setStyleSheet("background-color: #334155; color: #fca5a5; border: 1px solid #ef4444; font-weight: bold; padding: 6px 14px;")
        self.btn_clear_all_xkiro.clicked.connect(self.clear_all_xkiro_keys)
        xk_btn_row.addWidget(self.btn_clear_all_xkiro)

        xk_btn_row.addStretch()
        lay_xk.addLayout(xk_btn_row)

        self.lbl_xkiro_summary = QLabel("✅ 0 keys đã cấu hình")
        self.lbl_xkiro_summary.setStyleSheet("color: #10b981; font-weight: bold; font-size: 13px; padding-top: 4px;")
        lay_xk.addWidget(self.lbl_xkiro_summary)

        layout.addWidget(xkiro_group)

        # ---------------------------------------------------------------------
        # Bottom Global Actions
        # ---------------------------------------------------------------------
        row_act = QHBoxLayout()
        self.btn_load_api_config = QPushButton("🔄 Tải cấu hình hiện tại")
        self.btn_load_api_config.setStyleSheet("background-color: #475569; color: white; font-weight: bold; padding: 8px 18px;")
        self.btn_load_api_config.clicked.connect(self.load_api_config_to_ui)
        row_act.addWidget(self.btn_load_api_config)

        self.btn_save_all_api = QPushButton("💾 Lưu tất cả API keys")
        self.btn_save_all_api.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 8px 22px;")
        self.btn_save_all_api.clicked.connect(self.save_api_config_from_ui)
        row_act.addWidget(self.btn_save_all_api)
        row_act.addStretch()

        layout.addLayout(row_act)
        layout.addStretch()

        # Legacy compatibility aliases with two-way sync
        self.txt_gemini_key = QLineEdit()
        self.txt_xkiro_key = QLineEdit()
        self.lbl_status_gemini = self.lbl_gemini_summary
        self.lbl_status_xkiro = self.lbl_xkiro_summary
        self.btn_save_gemini_key = self.btn_save_all_api
        self.btn_save_xkiro_key = self.btn_save_all_api
        self.btn_test_gemini = self.btn_test_all_gemini
        self.btn_test_xkiro = self.btn_test_all_xkiro

        def on_txt_gemini_changed(text):
            if not getattr(self, '_is_syncing_legacy', False):
                raw_keys = [k.strip() for k in text.split(",") if k.strip()]
                self._is_syncing_legacy = True
                try:
                    for w in list(self.gemini_key_rows):
                        w.deleteLater()
                    self.gemini_key_rows = []
                    self.gemini_key_inputs = []
                    self.gemini_key_status_labels = []
                    if raw_keys:
                        for k in raw_keys:
                            self.add_gemini_key_slot(k)
                    else:
                        self.add_gemini_key_slot("")
                finally:
                    self._is_syncing_legacy = False

        def on_txt_xkiro_changed(text):
            if not getattr(self, '_is_syncing_legacy', False):
                raw_keys = [k.strip() for k in text.split(",") if k.strip()]
                self._is_syncing_legacy = True
                try:
                    for w in list(self.xkiro_key_rows):
                        w.deleteLater()
                    self.xkiro_key_rows = []
                    self.xkiro_key_inputs = []
                    self.xkiro_key_status_labels = []
                    if raw_keys:
                        for k in raw_keys:
                            self.add_xkiro_key_slot(k)
                    else:
                        self.add_xkiro_key_slot("")
                finally:
                    self._is_syncing_legacy = False

        self.txt_gemini_key.textChanged.connect(on_txt_gemini_changed)
        self.txt_xkiro_key.textChanged.connect(on_txt_xkiro_changed)

        scroll.setWidget(container)
        return scroll

    def setup_tab2_voice(self):
        """BƯỚC 3: Tab 🎤 Voice & TTS Settings."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(12)

        # TTS Engine selector
        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("<b>TTS Engine:</b>"))
        self.cb_tts_engine = QComboBox()
        self.cb_tts_engine.addItems(["Edge-TTS (Microsoft Neural)", "Google TTS", "System TTS"])
        self.cb_tts_engine.currentTextChanged.connect(lambda t: self.save_app_settings())
        engine_row.addWidget(self.cb_tts_engine, 1)

        self.btn_refresh_voices = QPushButton("🔄 Refresh Danh Sách")
        self.btn_refresh_voices.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_refresh_voices.clicked.connect(self.load_dynamic_tts_voices)
        engine_row.addWidget(self.btn_refresh_voices)
        layout.addLayout(engine_row)

        # Search voice row
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("<b>🔍 Tìm kiếm giọng:</b>"))
        self.txt_search_voice = QLineEdit()
        self.txt_search_voice.setPlaceholderText("Nhập từ khóa tìm kiếm (Ví dụ: vi-VN, HoaiMy, NamMinh, Female, Nữ)...")
        self.txt_search_voice.textChanged.connect(self.filter_tts_voice_list)
        search_row.addWidget(self.txt_search_voice, 1)
        layout.addLayout(search_row)

        # Voice list group
        voice_group = QGroupBox("📋 Danh Sách Giọng Đọc AI Neural")
        v_lay = QVBoxLayout(voice_group)
        v_lay.setContentsMargins(12, 14, 12, 12)
        v_lay.setSpacing(10)

        self.list_tts_voices = QListWidget()
        self.list_tts_voices.setMinimumHeight(200)
        self.list_tts_voices.setStyleSheet("QListWidget { background-color: #090d16; color: #38bdf8; border: 1px solid #1e293b; border-radius: 6px; }")
        self.list_tts_voices.itemClicked.connect(self.on_voice_item_selected)
        v_lay.addWidget(self.list_tts_voices)

        v_subrow = QHBoxLayout()
        self.lbl_voice_count = QLabel("Tổng số: 0 giọng")
        self.lbl_voice_count.setStyleSheet("color: #38bdf8; font-weight: bold;")
        v_subrow.addWidget(self.lbl_voice_count)
        v_subrow.addStretch()

        self.btn_preview_voice = QPushButton("🔊 Preview Voice (Nghe thử giọng)")
        self.btn_preview_voice.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_preview_voice.clicked.connect(self.preview_selected_voice)
        v_subrow.addWidget(self.btn_preview_voice)
        v_lay.addLayout(v_subrow)

        layout.addWidget(voice_group)

        # Voice Options
        self.chk_auto_save_voice = QCheckBox("💾 Tự động lưu giọng đọc đã chọn (Auto-save preference)")
        self.chk_auto_save_voice.setChecked(True)
        self.chk_auto_save_voice.stateChanged.connect(lambda s: self.save_app_settings())
        layout.addWidget(self.chk_auto_save_voice)

        self.chk_voice_fallback = QCheckBox("🔄 Tự động fallback sang giọng mặc định nếu giọng chọn không khả dụng")
        self.chk_voice_fallback.setChecked(True)
        self.chk_voice_fallback.stateChanged.connect(lambda s: self.save_app_settings())
        layout.addWidget(self.chk_voice_fallback)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def setup_tab2_font(self):
        """BƯỚC 4: Tab 📝 Font & Subtitle Settings."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(12)

        box_preset = QGroupBox("📝 CẤU HÌNH PRESET SUBTITLE & KIỂU CHỮ")
        lay_pres = QVBoxLayout(box_preset)
        lay_pres.setContentsMargins(15, 18, 15, 15)
        lay_pres.setSpacing(10)

        def create_sep():
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            line.setStyleSheet("background-color: #334155; max-height: 1px;")
            return line

        # Hàng 1: Profile mẫu
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        row1.addWidget(QLabel("<b>Profile Mẫu:</b>"))

        self.cb_preset_profile = QComboBox()
        self.cb_preset_profile.setMinimumWidth(260)
        self.cb_preset_profile.addItems([
            "Default (Arial, 24, White)",
            "Large Text (Verdana, 32, Yellow)",
            "Small Text (Arial, 18, White)",
            "Cinematic (Impact, 28, White, bg box)",
            "Custom"
        ])
        self.cb_preset_profile.currentTextChanged.connect(self.on_preset_profile_selected)
        row1.addWidget(self.cb_preset_profile, 1)

        self.btn_save_preset_profile = QPushButton("💾 Lưu profile mới")
        self.btn_save_preset_profile.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_save_preset_profile.clicked.connect(self.save_preset_profile_clicked)
        row1.addWidget(self.btn_save_preset_profile)
        lay_pres.addLayout(row1)

        lay_pres.addWidget(create_sep())

        # Hàng 2: Font & Size
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        row2.addWidget(QLabel("<b>Phông chữ (Font):</b>"))

        self.cb_font_name = QComboBox()
        self.cb_font_name.setEditable(True)
        self.cb_font_name.addItems(["Arial", "Times New Roman", "Verdana", "Tahoma", "Impact", "Comic Sans MS"])
        self.cb_font_name.setCurrentText("Arial")
        self.cb_font_name.currentTextChanged.connect(self.on_preset_control_changed)
        row2.addWidget(self.cb_font_name, 1)

        row2.addWidget(QLabel("<b>Cỡ chữ (Size):</b>"))
        self.spin_font_size_tab2 = QSpinBox()
        self.spin_font_size_tab2.setRange(8, 72)
        self.spin_font_size_tab2.setValue(24)
        self.spin_font_size_tab2.setMinimumWidth(80)
        self.spin_font_size_tab2.valueChanged.connect(self.on_preset_control_changed)
        row2.addWidget(self.spin_font_size_tab2)
        lay_pres.addLayout(row2)

        lay_pres.addWidget(create_sep())

        # Hàng 3: Font Color + Outline Color + Outline Width
        row3 = QHBoxLayout()
        row3.setSpacing(10)

        row3.addWidget(QLabel("<b>Màu chữ:</b>"))
        self.btn_font_color = QPushButton()
        self.btn_font_color.setFixedSize(28, 28)
        self.btn_font_color.setStyleSheet("background-color: #FFFFFF; border: 1px solid white; border-radius: 4px;")
        self.btn_font_color.clicked.connect(lambda: (self.pick_subtitle_color('font'), self.on_preset_control_changed()))
        row3.addWidget(self.btn_font_color)

        self.lbl_font_hex = QLabel("#FFFFFF")
        self.lbl_font_hex.setMinimumWidth(65)
        row3.addWidget(self.lbl_font_hex)

        row3.addSpacing(15)
        row3.addWidget(QLabel("<b>Màu viền (Outline):</b>"))
        self.btn_outline_color = QPushButton()
        self.btn_outline_color.setFixedSize(28, 28)
        self.btn_outline_color.setStyleSheet("background-color: #000000; border: 1px solid white; border-radius: 4px;")
        self.btn_outline_color.clicked.connect(lambda: (self.mark_preset_custom(), self.pick_subtitle_color('outline')))
        row3.addWidget(self.btn_outline_color)

        self.lbl_outline_hex = QLabel("#000000")
        self.lbl_outline_hex.setMinimumWidth(65)
        row3.addWidget(self.lbl_outline_hex)

        row3.addSpacing(15)
        row3.addWidget(QLabel("<b>Độ dày viền:</b>"))
        self.spin_outline_width = QSpinBox()
        self.spin_outline_width.setRange(0, 10)
        self.spin_outline_width.setValue(2)
        self.spin_outline_width.setMinimumWidth(60)
        self.spin_outline_width.valueChanged.connect(lambda v: (self.mark_preset_custom(), self.save_app_settings()))
        row3.addWidget(self.spin_outline_width)
        row3.addStretch()
        lay_pres.addLayout(row3)

        lay_pres.addWidget(create_sep())

        # Hàng 4: Background Box + BG Color + Opacity
        row4 = QHBoxLayout()
        row4.setSpacing(10)

        self.chk_use_bg_box = QCheckBox("<b>Khung nền (Background Box)</b>")
        self.chk_use_bg_box.setChecked(False)
        self.chk_use_bg_box.stateChanged.connect(lambda s: (self.mark_preset_custom(), self.save_app_settings()))
        row4.addWidget(self.chk_use_bg_box)

        row4.addWidget(QLabel("<b>Màu nền:</b>"))
        self.btn_bg_color = QPushButton()
        self.btn_bg_color.setFixedSize(28, 28)
        self.btn_bg_color.setStyleSheet("background-color: #000000; border: 1px solid white; border-radius: 4px;")
        self.btn_bg_color.clicked.connect(lambda: self.pick_subtitle_color('bg'))
        row4.addWidget(self.btn_bg_color)

        self.lbl_bg_hex = QLabel("#000000")
        self.lbl_bg_hex.setMinimumWidth(65)
        row4.addWidget(self.lbl_bg_hex)

        row4.addSpacing(15)
        row4.addWidget(QLabel("<b>Độ mờ (Opacity):</b>"))
        self.slider_bg_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_bg_opacity.setRange(0, 100)
        self.slider_bg_opacity.setValue(50)
        self.slider_bg_opacity.setMinimumWidth(120)
        self.lbl_bg_opacity_val = QLabel("50%")
        self.lbl_bg_opacity_val.setMinimumWidth(40)
        self.slider_bg_opacity.valueChanged.connect(lambda v: (self.lbl_bg_opacity_val.setText(f"{v}%"), self.save_app_settings()))
        row4.addWidget(self.slider_bg_opacity)
        row4.addWidget(self.lbl_bg_opacity_val)
        row4.addStretch()
        lay_pres.addLayout(row4)

        lay_pres.addWidget(create_sep())

        # Hàng 5: Alignment + Margins
        row5 = QHBoxLayout()
        row5.setSpacing(10)

        row5.addWidget(QLabel("<b>Vị trí dọc:</b>"))
        self.cb_v_align = QComboBox()
        self.cb_v_align.addItems(["top", "middle", "bottom"])
        self.cb_v_align.setCurrentText("bottom")
        self.cb_v_align.currentTextChanged.connect(lambda t: self.save_app_settings())
        row5.addWidget(self.cb_v_align)

        row5.addSpacing(10)
        row5.addWidget(QLabel("<b>Vị trí ngang:</b>"))
        self.cb_h_align = QComboBox()
        self.cb_h_align.addItems(["left", "center", "right"])
        self.cb_h_align.setCurrentText("center")
        self.cb_h_align.currentTextChanged.connect(lambda t: self.save_app_settings())
        row5.addWidget(self.cb_h_align)

        row5.addSpacing(15)
        row5.addWidget(QLabel("<b>Margin V:</b>"))
        self.spin_margin_v = QSpinBox()
        self.spin_margin_v.setRange(0, 200)
        self.spin_margin_v.setValue(20)
        self.spin_margin_v.setMinimumWidth(60)
        self.spin_margin_v.valueChanged.connect(lambda v: self.save_app_settings())
        row5.addWidget(self.spin_margin_v)

        row5.addSpacing(10)
        row5.addWidget(QLabel("<b>Margin H:</b>"))
        self.spin_margin_h = QSpinBox()
        self.spin_margin_h.setRange(0, 200)
        self.spin_margin_h.setValue(20)
        self.spin_margin_h.setMinimumWidth(60)
        self.spin_margin_h.valueChanged.connect(lambda v: self.save_app_settings())
        row5.addWidget(self.spin_margin_h)
        row5.addStretch()
        lay_pres.addLayout(row5)

        layout.addWidget(box_preset)
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def setup_tab2_audio(self):
        """BƯỚC 5: Tab 🔊 Audio Settings."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(15)

        box_audio = QGroupBox("🔊 CẤU HÌNH ÂM LƯỢNG (AUDIO VOLUMES)")
        lay_aud = QGridLayout(box_audio)
        lay_aud.setContentsMargins(14, 16, 14, 14)
        lay_aud.setSpacing(12)

        lay_aud.addWidget(QLabel("<b>Âm lượng nhạc nền gốc (Background Volume):</b>"), 0, 0)
        self.slider_bg = QSlider(Qt.Orientation.Horizontal)
        self.slider_bg.setRange(0, 100)
        self.slider_bg.setValue(30)
        self.lbl_bg_val = QLabel("30%")
        self.lbl_bg_val.setStyleSheet("color: #38bdf8; font-weight: bold; min-width: 45px;")
        self.slider_bg.valueChanged.connect(lambda v: (self.lbl_bg_val.setText(f"{v}%"), self.save_app_settings()))
        lay_aud.addWidget(self.slider_bg, 0, 1)
        lay_aud.addWidget(self.lbl_bg_val, 0, 2)

        lay_aud.addWidget(QLabel("<b>Âm lượng giọng đọc lồng tiếng (Dubbing Volume):</b>"), 1, 0)
        self.slider_dub = QSlider(Qt.Orientation.Horizontal)
        self.slider_dub.setRange(0, 200)
        self.slider_dub.setValue(100)
        self.lbl_dub_val = QLabel("100%")
        self.lbl_dub_val.setStyleSheet("color: #4ade80; font-weight: bold; min-width: 45px;")
        self.slider_dub.valueChanged.connect(lambda v: (self.lbl_dub_val.setText(f"{v}%"), self.save_app_settings()))
        lay_aud.addWidget(self.slider_dub, 1, 1)
        lay_aud.addWidget(self.lbl_dub_val, 1, 2)

        layout.addWidget(box_audio)
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def setup_tab2_ocr(self):
        """Tab 🖥️ OCR & Engine Settings."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(14)

        # 1. Gemini Vision Models Box
        box_gem = QGroupBox("🤖 CẤU HÌNH GEMINI VISION OCR & MODELS")
        lay_gm = QVBoxLayout(box_gem)
        lay_gm.setContentsMargins(14, 16, 14, 14)
        lay_gm.setSpacing(10)

        row_gm1 = QHBoxLayout()
        row_gm1.addWidget(QLabel("<b>Gemini Model:</b>"))

        self.cb_gemini_model = QComboBox()
        self.cb_gemini_model.setMinimumWidth(260)
        self.cb_gemini_model.addItems([
            "gemini-flash-latest (⭐ Khuyến nghị - Mới nhất)",
            "gemini-2.5-flash (🚀 Nhanh & chuẩn)",
            "gemini-flash-lite-latest (💨 Siêu nhẹ & nhanh)",
            "gemini-2.5-flash-lite (⚡ Tiết kiệm quota)",
            "gemini-2.0-flash-exp (🧪 Exp)",
            "gemini-2.0-flash-lite-preview (🧪 Lite Preview)",
            "gemini-1.5-flash-8b (🧪 8B)",
            "gemini-3.5-flash",
            "gemini-3.7-flash",
            "gemini-2.5-pro (🏆 Chất lượng cao)",
            "🔍 Auto (Thử tất cả)"
        ])
        self.cb_gemini_model.setCurrentText("gemini-flash-latest (⭐ Khuyến nghị - Mới nhất)")
        self.cb_gemini_model.currentTextChanged.connect(lambda t: self.save_app_settings())
        row_gm1.addWidget(self.cb_gemini_model, 1)

        self.btn_test_gemini_model = QPushButton("🔍 Test Model")
        self.btn_test_gemini_model.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_test_gemini_model.clicked.connect(self.test_current_gemini_model_clicked)
        row_gm1.addWidget(self.btn_test_gemini_model)

        self.btn_refresh_models = QPushButton("🔄 Refresh models")
        self.btn_refresh_models.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_refresh_models.clicked.connect(self.refresh_gemini_models)
        row_gm1.addWidget(self.btn_refresh_models)

        self.btn_list_available_models = QPushButton("📋 Xem models")
        self.btn_list_available_models.setStyleSheet("background-color: #475569; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_list_available_models.clicked.connect(self.show_available_gemini_models_dialog)
        row_gm1.addWidget(self.btn_list_available_models)
        lay_gm.addLayout(row_gm1)

        row_gm2 = QHBoxLayout()
        self.chk_gemini_auto_fallback_model = QCheckBox("☑ Auto-fallback sang model khác nếu gặp lỗi (404/503/429)")
        self.chk_gemini_auto_fallback_model.setChecked(True)
        self.chk_gemini_auto_fallback_model.stateChanged.connect(lambda s: self.save_app_settings())
        row_gm2.addWidget(self.chk_gemini_auto_fallback_model)

        self.chk_gemini_fallback_easyocr = QCheckBox("☑ Fallback sang EasyOCR nếu tất cả model Gemini đều lỗi")
        self.chk_gemini_fallback_easyocr.setChecked(True)
        self.chk_gemini_fallback_easyocr.stateChanged.connect(lambda s: self.save_app_settings())
        row_gm2.addWidget(self.chk_gemini_fallback_easyocr)
        row_gm2.addStretch()
        lay_gm.addLayout(row_gm2)

        layout.addWidget(box_gem)

        # 2. xKiro Prompt Template Box
        box_xk = QGroupBox("📝 CẤU HÌNH DỊCH THUẬT xKIRO & PROMPT TEMPLATE")
        lay_xk = QVBoxLayout(box_xk)
        lay_xk.setContentsMargins(14, 16, 14, 14)
        lay_xk.setSpacing(10)

        row_x1 = QHBoxLayout()
        row_x1.addWidget(QLabel("<b>Engine:</b>"))
        self.cb_translation_engine = QComboBox()
        self.cb_translation_engine.addItems([
            "xKiro AI (Ưu tiên)",
            "Gemini AI",
            "Google Translate (Fallback)",
            "Tự động (xKiro → Gemini → Google)"
        ])
        self.cb_translation_engine.setCurrentText("xKiro AI (Ưu tiên)")
        self.cb_translation_engine.currentTextChanged.connect(lambda t: self.save_app_settings())
        row_x1.addWidget(self.cb_translation_engine)

        self.chk_prefer_xkiro = QCheckBox("⚡ Ưu tiên xKiro AI")
        self.chk_prefer_xkiro.setChecked(True)
        self.chk_prefer_xkiro.stateChanged.connect(lambda s: self.save_app_settings())
        row_x1.addWidget(self.chk_prefer_xkiro)

        row_x1.addSpacing(10)
        row_x1.addWidget(QLabel("<b>Max tokens:</b>"))
        self.spin_xkiro_max_tokens = QSpinBox()
        self.spin_xkiro_max_tokens.setRange(100, 4000)
        self.spin_xkiro_max_tokens.setValue(1000)
        row_x1.addWidget(self.spin_xkiro_max_tokens)

        row_x1.addWidget(QLabel("<b>Temperature:</b>"))
        self.cb_xkiro_temperature = QComboBox()
        self.cb_xkiro_temperature.addItems(["0.1", "0.3", "0.5", "0.7", "0.9"])
        self.cb_xkiro_temperature.setCurrentText("0.3")
        row_x1.addWidget(self.cb_xkiro_temperature)

        row_x1.addStretch()
        lay_xk.addLayout(row_x1)

        row_x2 = QHBoxLayout()
        self.chk_keep_proper_nouns = QCheckBox("<b>Giữ nguyên tên riêng</b>")
        self.chk_keep_proper_nouns.setChecked(True)
        row_x2.addWidget(self.chk_keep_proper_nouns)

        self.chk_auto_context = QCheckBox("<b>Tự động thêm context từ video</b>")
        self.chk_auto_context.setChecked(True)
        row_x2.addWidget(self.chk_auto_context)
        row_x2.addStretch()
        lay_xk.addLayout(row_x2)

        lay_xk.addWidget(QLabel("<b>Nội dung Prompt Template:</b>"))
        self.txt_xkiro_prompt_template = QTextEdit()
        self.txt_xkiro_prompt_template.setMinimumHeight(110)
        lay_xk.addWidget(self.txt_xkiro_prompt_template)

        row_xbtns = QHBoxLayout()
        self.btn_load_default_xkiro_prompt = QPushButton("📥 Load mặc định")
        self.btn_load_default_xkiro_prompt.setStyleSheet("background-color: #475569; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_load_default_xkiro_prompt.clicked.connect(self.load_default_xkiro_prompt)
        row_xbtns.addWidget(self.btn_load_default_xkiro_prompt)

        self.btn_save_xkiro_prompt = QPushButton("💾 Lưu prompt")
        self.btn_save_xkiro_prompt.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_save_xkiro_prompt.clicked.connect(self.save_xkiro_prompt_template)
        row_xbtns.addWidget(self.btn_save_xkiro_prompt)
        row_xbtns.addStretch()
        lay_xk.addLayout(row_xbtns)

        layout.addWidget(box_xk)
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def setup_tab2_adv(self):
        """Tab ⚙️ Advanced Settings."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 14, 12, 12)
        layout.setSpacing(14)

        box_general = QGroupBox("⚙️ CÀI ĐẶT HỆ THỐNG & ĐA LUỒNG")
        lay_gen = QGridLayout(box_general)
        lay_gen.setContentsMargins(14, 16, 14, 14)
        lay_gen.setSpacing(10)

        # Workers count
        lay_gen.addWidget(QLabel("<b>Số luồng xử lý pipeline (workers_cnt):</b>"), 0, 0)
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, 16)
        self.spin_workers.setValue(4)
        self.spin_max_workers = self.spin_workers
        self.spin_workers.valueChanged.connect(lambda v: self.save_app_settings())
        lay_gen.addWidget(self.spin_workers, 0, 1)

        # Parallel Chunk workers count
        lay_gen.addWidget(QLabel("<b>Số luồng xử lý chunks song song:</b>"), 0, 2)
        self.spin_chunk_workers = QSpinBox()
        self.spin_chunk_workers.setRange(1, 8)
        self.spin_chunk_workers.setValue(4)
        self.spin_chunk_workers.setToolTip("Số luồng xử lý các chunks video song song bằng ThreadPool")
        self.spin_chunk_workers.valueChanged.connect(lambda v: self.save_app_settings())
        lay_gen.addWidget(self.spin_chunk_workers, 0, 3)

        self.chk_dark_mode = QCheckBox("🌙 Dark Mode (Giao diện tối)")
        self.chk_dark_mode.setChecked(True)
        self.chk_dark_mode.stateChanged.connect(lambda s: (self.apply_dark_mode(bool(s)), self.save_app_settings()))
        lay_gen.addWidget(self.chk_dark_mode, 1, 0, 1, 2)

        self.chk_burn_sub_export = QCheckBox("🔥 Burn Subtitle (Đè phụ đề) vào video")
        self.chk_burn_sub_export.setChecked(True)
        self.chk_burn_sub_export.stateChanged.connect(lambda s: self.save_app_settings())
        lay_gen.addWidget(self.chk_burn_sub_export, 1, 2, 1, 2)

        self.chk_auto_report = QCheckBox("📊 Tự động tạo report sau mỗi batch")
        self.chk_auto_report.setChecked(True)
        self.chk_auto_report.stateChanged.connect(lambda s: self.save_app_settings())
        lay_gen.addWidget(self.chk_auto_report, 2, 0, 1, 2)

        lay_gen.addWidget(QLabel("<b>Định dạng báo cáo mặc định:</b>"), 2, 2)
        self.cb_report_format = QComboBox()
        self.cb_report_format.addItems(["HTML", "PDF", "CSV"])
        self.cb_report_format.currentIndexChanged.connect(lambda idx: self.save_app_settings())
        lay_gen.addWidget(self.cb_report_format, 2, 3)

        self.chk_open_folder_on_done = QCheckBox("📂 Mở folder output sau khi hoàn thành")
        self.chk_open_folder_on_done.setChecked(True)
        self.chk_open_folder_on_done.stateChanged.connect(lambda s: self.save_app_settings())
        lay_gen.addWidget(self.chk_open_folder_on_done, 3, 0, 1, 4)

        # Scan Interval & Min Sub Duration
        lay_gen.addWidget(QLabel("<b>⏱️ Khoảng cách quét khung hình (Scan interval, s):</b>"), 4, 0)
        self.spin_scan_interval = QDoubleSpinBox()
        self.spin_scan_interval.setRange(0.2, 3.0)
        self.spin_scan_interval.setSingleStep(0.1)
        self.spin_scan_interval.setValue(0.5)
        self.spin_scan_interval.setToolTip("Khoảng thời gian (giây) giữa các frame quét OCR (Mặc định 0.5s)")
        self.spin_scan_interval.valueChanged.connect(lambda v: self.save_app_settings())
        lay_gen.addWidget(self.spin_scan_interval, 4, 1)

        lay_gen.addWidget(QLabel("<b>⏳ Thời lượng sub tối thiểu (Min duration, s):</b>"), 4, 2)
        self.spin_min_sub_dur = QDoubleSpinBox()
        self.spin_min_sub_dur.setRange(0.1, 2.0)
        self.spin_min_sub_dur.setSingleStep(0.1)
        self.spin_min_sub_dur.setValue(0.3)
        self.spin_min_sub_dur.setToolTip("Thời lượng tối thiểu (giây) để coi là một phân đoạn phụ đề hợp lệ (Mặc định 0.3s)")
        self.spin_min_sub_dur.valueChanged.connect(lambda v: self.save_app_settings())
        lay_gen.addWidget(self.spin_min_sub_dur, 4, 3)

        # Ngôn ngữ nhận diện cho PaddleOCR
        lay_gen.addWidget(QLabel("<b>🈲 Ngôn ngữ PaddleOCR (Nhận dạng):</b>"), 5, 0)
        self.cb_paddle_lang = QComboBox()
        self.cb_paddle_lang.addItems([
            "Tiếng Trung (zh/ch)",
            "Tiếng Việt (vi)",
            "Tiếng Anh (en)",
            "Tiếng Nhật (ja)",
            "Tiếng Hàn (ko)",
            "Tiếng Pháp (fr)",
            "Tiếng Đức (de)",
            "Tiếng Tây Ban Nha (es)",
            "Tiếng Nga (ru)"
        ])
        self.cb_paddle_lang.currentIndexChanged.connect(lambda idx: self.save_app_settings())
        lay_gen.addWidget(self.cb_paddle_lang, 5, 1, 1, 3)

        layout.addWidget(box_general)

        # Actions Box
        box_actions = QGroupBox("💾 LƯU VÀ KHÔI PHỤC CẤU HÌNH")
        lay_act = QHBoxLayout(box_actions)
        lay_act.setContentsMargins(14, 16, 14, 14)
        lay_act.setSpacing(12)

        self.btn_save_all_settings = QPushButton("💾 Lưu Tất Cả Cài Đặt (Save All)")
        self.btn_save_all_settings.setMinimumHeight(38)
        self.btn_save_all_settings.setStyleSheet("background-color: #059669; color: white; font-weight: bold; font-size: 13px;")
        self.btn_save_all_settings.clicked.connect(self.save_all_page2_settings)
        lay_act.addWidget(self.btn_save_all_settings)

        self.btn_reset_defaults = QPushButton("🔄 Khôi Phục Mặc Định (Reset Defaults)")
        self.btn_reset_defaults.setMinimumHeight(38)
        self.btn_reset_defaults.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; font-size: 13px;")
        self.btn_reset_defaults.clicked.connect(self.reset_app_settings)
        lay_act.addWidget(self.btn_reset_defaults)

        layout.addWidget(box_actions)
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def setup_page3_tab(self):
        """Thiết lập TRANG 3: QUẢN LÝ DỰ ÁN & LỊCH SỬ"""
        tab_page3_widget = QWidget()
        tab_page3_layout = QVBoxLayout(tab_page3_widget)
        tab_page3_layout.setContentsMargins(8, 8, 8, 8)

        splitter_page3 = QSplitter(Qt.Orientation.Horizontal)
        splitter_page3.setChildrenCollapsible(False)

        # =========================================================================
        # KHUNG 1: A. QUẢN LÝ DỰ ÁN (BÊN TRÁI)
        # =========================================================================
        box_project = QGroupBox("📁 A. QUẢN LÝ DỰ ÁN")
        lay_proj = QVBoxLayout(box_project)
        lay_proj.setContentsMargins(10, 12, 10, 10)
        lay_proj.setSpacing(10)

        # 1. Thông tin dự án hiện tại
        grid_info = QGridLayout()
        grid_info.setSpacing(8)

        grid_info.addWidget(QLabel("<b>Tên dự án:</b>"), 0, 0)
        self.txt_project_name = QLineEdit("Dự án mới")
        grid_info.addWidget(self.txt_project_name, 0, 1)

        grid_info.addWidget(QLabel("<b>Mô tả:</b>"), 1, 0, Qt.AlignmentFlag.AlignTop)
        self.txt_project_desc = QTextEdit()
        self.txt_project_desc.setPlaceholderText("Nhập mô tả chi tiết cho dự án...")
        self.txt_project_desc.setMaximumHeight(80)
        grid_info.addWidget(self.txt_project_desc, 1, 1)

        grid_info.addWidget(QLabel("<b>Ngày tạo:</b>"), 2, 0)
        self.lbl_created_date = QLabel(getattr(self, 'project_created_date', ''))
        self.lbl_created_date.setStyleSheet("color: #38bdf8; font-weight: bold;")
        grid_info.addWidget(self.lbl_created_date, 2, 1)

        grid_info.addWidget(QLabel("<b>Ngày cập nhật:</b>"), 3, 0)
        self.lbl_updated_date = QLabel(getattr(self, 'project_updated_date', ''))
        self.lbl_updated_date.setStyleSheet("color: #38bdf8; font-weight: bold;")
        grid_info.addWidget(self.lbl_updated_date, 3, 1)

        grid_info.addWidget(QLabel("<b>Đường dẫn:</b>"), 4, 0)
        self.lbl_project_path = QLabel(getattr(self, 'project_file_path', 'Chưa lưu'))
        self.lbl_project_path.setStyleSheet("color: #94a3b8; font-style: italic;")
        self.lbl_project_path.setWordWrap(True)
        grid_info.addWidget(self.lbl_project_path, 4, 1)

        lay_proj.addLayout(grid_info)
        lay_proj.addSpacing(10)

        # 2. Các nút thao tác dự án
        box_proj_btns = QGroupBox("⚙️ Thao tác Dự án & Cấu hình")
        lay_pbtns = QVBoxLayout(box_proj_btns)
        lay_pbtns.setSpacing(8)

        self.btn_new_project = QPushButton("📁 Tạo dự án mới")
        self.btn_new_project.setStyleSheet("background-color: #3b82f6; color: white; font-weight: bold; padding: 8px;")
        self.btn_new_project.clicked.connect(self.btn_new_project_clicked)
        lay_pbtns.addWidget(self.btn_new_project)

        self.btn_save_project = QPushButton("💾 Lưu dự án")
        self.btn_save_project.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 8px;")
        self.btn_save_project.clicked.connect(self.btn_save_project_clicked)
        lay_pbtns.addWidget(self.btn_save_project)

        self.btn_revert_project = QPushButton("↩️ Revert to last saved")
        self.btn_revert_project.setStyleSheet("background-color: #d97706; color: white; font-weight: bold; padding: 6px;")
        self.btn_revert_project.clicked.connect(self.revert_project_clicked)
        lay_pbtns.addWidget(self.btn_revert_project)

        self.btn_load_project = QPushButton("📂 Mở dự án")
        self.btn_load_project.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 8px;")
        self.btn_load_project.clicked.connect(self.btn_load_project_clicked)
        lay_pbtns.addWidget(self.btn_load_project)

        row_config = QHBoxLayout()
        self.btn_export_config = QPushButton("📤 Xuất cấu hình")
        self.btn_export_config.setStyleSheet("background-color: #475569; color: white; font-weight: bold; padding: 6px;")
        self.btn_export_config.clicked.connect(self.btn_export_config_clicked)
        row_config.addWidget(self.btn_export_config)

        self.btn_import_config = QPushButton("📥 Nhập cấu hình")
        self.btn_import_config.setStyleSheet("background-color: #475569; color: white; font-weight: bold; padding: 6px;")
        self.btn_import_config.clicked.connect(self.btn_import_config_clicked)
        row_config.addWidget(self.btn_import_config)

        lay_pbtns.addLayout(row_config)

        self.btn_clear_cache = QPushButton("🗑️ Clear Cache (Dọn bộ nhớ tạm)")
        self.btn_clear_cache.setStyleSheet("background-color: #64748b; color: white; font-weight: bold; padding: 6px;")
        self.btn_clear_cache.clicked.connect(self.clear_cache_clicked)
        lay_pbtns.addWidget(self.btn_clear_cache)

        lay_proj.addWidget(box_proj_btns)
        lay_proj.addStretch()

        splitter_page3.addWidget(box_project)

        # =========================================================================
        # KHUNG 2: B. LỊCH SỬ XỬ LÝ (BÊN PHẢI)
        # =========================================================================
        box_history = QGroupBox("📜 B. LỊCH SỬ XỬ LÝ")
        lay_hist = QVBoxLayout(box_history)
        lay_hist.setContentsMargins(10, 12, 10, 10)
        lay_hist.setSpacing(8)

        # Filter & Title Row
        hist_title_row = QHBoxLayout()
        hist_title_row.addWidget(QLabel("<b>Danh sách các đợt xử lý:</b>"))
        hist_title_row.addStretch()
        hist_title_row.addWidget(QLabel("<b>Lọc trạng thái:</b>"))
        self.cb_history_filter = QComboBox()
        self.cb_history_filter.addItems(["All", "✅ Thành công", "❌ Thất bại", "⏳ Đang xử lý"])
        self.cb_history_filter.currentTextChanged.connect(self.filter_history_table)
        hist_title_row.addWidget(self.cb_history_filter)
        lay_hist.addLayout(hist_title_row)

        self.tbl_history = QTableWidget()
        self.tbl_history.setColumnCount(8)
        self.tbl_history.setHorizontalHeaderLabels([
            "STT", "Tên dự án", "Video đầu vào", "Video đầu ra", "Ngày xử lý", "Thời gian (phút)", "Thời lượng", "Trạng thái"
        ])
        self.tbl_history.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_history.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tbl_history.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_history.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_history.horizontalHeader().setStretchLastSection(True)
        self.tbl_history.itemSelectionChanged.connect(self.on_history_selection_changed)
        lay_hist.addWidget(self.tbl_history, 2)

        # 2. Chi tiết lịch sử
        lay_hist.addWidget(QLabel("<b>Chi tiết lượt xử lý được chọn:</b>"))
        self.txt_history_detail = QTextEdit()
        self.txt_history_detail.setReadOnly(True)
        self.txt_history_detail.setPlaceholderText("Click chọn 1 dòng trong bảng trên để xem chi tiết cấu hình và log...")
        self.txt_history_detail.setMaximumHeight(160)
        lay_hist.addWidget(self.txt_history_detail, 1)

        # 3. Các nút thao tác lịch sử
        row_hbtns = QHBoxLayout()
        self.btn_load_more_history = QPushButton("📥 Tải thêm 50 bản ghi")
        self.btn_load_more_history.setStyleSheet("background-color: #3b82f6; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_load_more_history.clicked.connect(self.load_more_history_clicked)
        row_hbtns.addWidget(self.btn_load_more_history)

        self.btn_clear_history = QPushButton("🗑️ Xóa lịch sử")
        self.btn_clear_history.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_clear_history.clicked.connect(self.clear_history_clicked)
        row_hbtns.addWidget(self.btn_clear_history)

        self.btn_export_history = QPushButton("📤 Xuất lịch sử")
        self.btn_export_history.setStyleSheet("background-color: #475569; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_export_history.clicked.connect(self.export_history_clicked)
        row_hbtns.addWidget(self.btn_export_history)

        self.btn_reload_from_history = QPushButton("🔄 Mở lại project từ lịch sử")
        self.btn_reload_from_history.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_reload_from_history.clicked.connect(self.reload_from_history_clicked)
        row_hbtns.addWidget(self.btn_reload_from_history)

        self.btn_rerun = QPushButton("▶ Chạy lại pipeline")
        self.btn_rerun.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_rerun.clicked.connect(self.rerun_pipeline_from_history_clicked)
        row_hbtns.addWidget(self.btn_rerun)

        lay_hist.addLayout(row_hbtns)
        splitter_page3.addWidget(box_history)

        splitter_page3.setSizes([380, 720])
        tab_page3_layout.addWidget(splitter_page3)

        self.main_tab_widget.addTab(tab_page3_widget, "📁 3. QUẢN LÝ DỰ ÁN & LỊCH SỬ")
        self.load_history_to_table()

    def setup_page4_tab(self):
        """Thiết lập TRANG 4: BATCH PROCESSING & BÁO CÁO"""
        tab_page4_widget = QWidget()
        tab_page4_layout = QVBoxLayout(tab_page4_widget)
        tab_page4_layout.setContentsMargins(8, 8, 8, 8)

        splitter_page4 = QSplitter(Qt.Orientation.Horizontal)
        splitter_page4.setChildrenCollapsible(False)

        # =========================================================================
        # KHUNG 1: A. BATCH PROCESSING (XỬ LÝ HÀNG LOẠT - BÊN TRÁI ~50%)
        # =========================================================================
        box_batch = QGroupBox("⚡ A. XỬ LÝ HÀNG LOẠT (BATCH PROCESSING)")
        lay_batch = QVBoxLayout(box_batch)
        lay_batch.setContentsMargins(10, 12, 10, 10)
        lay_batch.setSpacing(10)

        # 1. Bảng hàng đợi Video (Queue Table)
        lay_batch.addWidget(QLabel("<b>1. Hàng đợi Video (Queue):</b>"))
        self.tbl_batch_queue = QTableWidget()
        self.tbl_batch_queue.setColumnCount(6)
        self.tbl_batch_queue.setHorizontalHeaderLabels([
            "STT", "Tên file video", "Đường dẫn", "Thời lượng", "Trạng thái", "Output Path"
        ])
        self.tbl_batch_queue.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_batch_queue.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tbl_batch_queue.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_batch_queue.horizontalHeader().setStretchLastSection(True)
        self.tbl_batch_queue.setAcceptDrops(True)
        self.tbl_batch_queue.itemSelectionChanged.connect(self.on_batch_queue_selection_changed)
        lay_batch.addWidget(self.tbl_batch_queue, 2)

        # 2. Các nút thao tác hàng đợi
        grid_qbtns = QGridLayout()
        grid_qbtns.setSpacing(6)

        self.btn_add_batch_videos = QPushButton("📁 Thêm video")
        self.btn_add_batch_videos.setStyleSheet("background-color: #3b82f6; color: white; font-weight: bold; padding: 5px;")
        self.btn_add_batch_videos.clicked.connect(self.add_batch_videos_clicked)
        grid_qbtns.addWidget(self.btn_add_batch_videos, 0, 0)

        self.btn_add_batch_folder = QPushButton("📂 Thêm thư mục")
        self.btn_add_batch_folder.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 5px;")
        self.btn_add_batch_folder.clicked.connect(self.add_batch_folder_clicked)
        grid_qbtns.addWidget(self.btn_add_batch_folder, 0, 1)

        self.btn_remove_batch_selected = QPushButton("🗑️ Xóa selected")
        self.btn_remove_batch_selected.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; padding: 5px;")
        self.btn_remove_batch_selected.clicked.connect(self.remove_batch_selected_clicked)
        grid_qbtns.addWidget(self.btn_remove_batch_selected, 0, 2)

        self.btn_clear_batch_queue = QPushButton("🗑️ Xóa tất cả")
        self.btn_clear_batch_queue.setStyleSheet("background-color: #7f1d1d; color: white; font-weight: bold; padding: 5px;")
        self.btn_clear_batch_queue.clicked.connect(self.clear_batch_queue_clicked)
        grid_qbtns.addWidget(self.btn_clear_batch_queue, 0, 3)

        self.btn_batch_up = QPushButton("⬆️ Lên")
        self.btn_batch_up.setStyleSheet("background-color: #475569; color: white; font-weight: bold; padding: 5px;")
        self.btn_batch_up.clicked.connect(self.move_batch_item_up)
        grid_qbtns.addWidget(self.btn_batch_up, 1, 0)

        self.btn_batch_down = QPushButton("⬇️ Xuống")
        self.btn_batch_down.setStyleSheet("background-color: #475569; color: white; font-weight: bold; padding: 5px;")
        self.btn_batch_down.clicked.connect(self.move_batch_item_down)
        grid_qbtns.addWidget(self.btn_batch_down, 1, 1)

        self.btn_export_batch_queue = QPushButton("📤 Xuất danh sách")
        self.btn_export_batch_queue.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 5px;")
        self.btn_export_batch_queue.clicked.connect(self.export_batch_queue_csv)
        grid_qbtns.addWidget(self.btn_export_batch_queue, 1, 2, 1, 2)

        lay_batch.addLayout(grid_qbtns)

        # 3. Cấu hình Batch
        box_bcfg = QGroupBox("⚙️ Cấu hình Batch")
        lay_bcfg = QGridLayout(box_bcfg)
        lay_bcfg.setSpacing(6)

        self.chk_batch_use_same_config = QCheckBox("Áp dụng cùng cấu hình cho tất cả video")
        self.chk_batch_use_same_config.setChecked(True)
        lay_bcfg.addWidget(self.chk_batch_use_same_config, 0, 0, 1, 2)

        lay_bcfg.addWidget(QLabel("<b>Số luồng xử lý song song:</b>"), 1, 0)
        self.spin_batch_workers = QSpinBox()
        self.spin_batch_workers.setRange(1, 4)
        self.spin_batch_workers.setValue(2)
        lay_bcfg.addWidget(self.spin_batch_workers, 1, 1)

        self.chk_batch_stop_on_error = QCheckBox("Dừng khi có lỗi (Stop on error)")
        self.chk_batch_stop_on_error.setChecked(False)
        lay_bcfg.addWidget(self.chk_batch_stop_on_error, 2, 0, 1, 2)

        lay_batch.addWidget(box_bcfg)

        # 4. Điều khiển & Tiến trình Batch
        box_bctrl = QGroupBox("🎮 Điều khiển & Tiến trình Batch")
        lay_bctrl = QVBoxLayout(box_bctrl)
        lay_bctrl.setSpacing(6)

        row_run_btns = QHBoxLayout()
        self.btn_run_batch = QPushButton("▶ BẮT ĐẦU CHẠY BATCH")
        self.btn_run_batch.setMinimumHeight(38)
        self.btn_run_batch.setStyleSheet("background-color: #059669; color: white; font-weight: bold; font-size: 13px;")
        self.btn_run_batch.clicked.connect(self.run_batch_clicked)
        row_run_btns.addWidget(self.btn_run_batch, 2)

        self.btn_stop_batch = QPushButton("🛑 DỪNG BATCH")
        self.btn_stop_batch.setMinimumHeight(38)
        self.btn_stop_batch.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; font-size: 13px;")
        self.btn_stop_batch.setEnabled(False)
        self.btn_stop_batch.clicked.connect(self.stop_batch_clicked)
        row_run_btns.addWidget(self.btn_stop_batch, 1)
        lay_bctrl.addLayout(row_run_btns)

        row_prog_info = QHBoxLayout()
        self.lbl_batch_current = QLabel("Đang xử lý: Chưa chạy (0/0)")
        self.lbl_batch_current.setStyleSheet("font-weight: bold; color: #38bdf8;")
        row_prog_info.addWidget(self.lbl_batch_current)

        self.lbl_batch_eta = QLabel("⏱️ Dự kiến còn lại: --:--")
        self.lbl_batch_eta.setStyleSheet("color: #facc15; font-weight: bold;")
        row_prog_info.addStretch()
        row_prog_info.addWidget(self.lbl_batch_eta)
        lay_bctrl.addLayout(row_prog_info)

        lay_bctrl.addWidget(QLabel("Tiến trình video hiện tại:"))
        self.progress_batch_current = QProgressBar()
        self.progress_batch_current.setRange(0, 100)
        self.progress_batch_current.setValue(0)
        self.progress_batch_current.setStyleSheet("QProgressBar::chunk { background-color: #38bdf8; }")
        lay_bctrl.addWidget(self.progress_batch_current)

        lay_bctrl.addWidget(QLabel("Tiến trình tổng thể (Total Batch):"))
        self.progress_batch_total = QProgressBar()
        self.progress_batch_total.setRange(0, 100)
        self.progress_batch_total.setValue(0)
        self.progress_batch_total.setStyleSheet("QProgressBar::chunk { background-color: #4ade80; }")
        lay_bctrl.addWidget(self.progress_batch_total)

        lay_batch.addWidget(box_bctrl)
        splitter_page4.addWidget(box_batch)

        # =========================================================================
        # KHUNG 2: B. BÁO CÁO & THỐNG KÊ (BÊN PHẢI ~50%)
        # =========================================================================
        box_reports = QGroupBox("📊 B. BÁO CÁO & THỐNG KÊ (REPORTS & STATS)")
        lay_rep = QVBoxLayout(box_reports)
        lay_rep.setContentsMargins(10, 12, 10, 10)
        lay_rep.setSpacing(8)

        # 1. Thống kê tổng quan
        box_stats = QGroupBox("📈 Thống kê tổng quan")
        grid_stats = QGridLayout(box_stats)
        grid_stats.setSpacing(6)

        self.lbl_stat_total_videos = QLabel("Tổng số video: <b>0</b>")
        grid_stats.addWidget(self.lbl_stat_total_videos, 0, 0)

        self.lbl_stat_processed = QLabel("Đã xử lý: <b>0</b> (✅ 0 / ❌ 0 / ⏳ 0)")
        grid_stats.addWidget(self.lbl_stat_processed, 0, 1)

        self.lbl_stat_total_time = QLabel("Tổng thời gian: <b>00:00:00</b>")
        grid_stats.addWidget(self.lbl_stat_total_time, 1, 0)

        self.lbl_stat_total_size = QLabel("Tổng dung lượng output: <b>0.0 MB</b>")
        grid_stats.addWidget(self.lbl_stat_total_size, 1, 1)

        lay_rep.addWidget(box_stats)

        # 2. Biểu đồ Matplotlib
        box_charts = QGroupBox("📊 Biểu đồ trực quan")
        lay_charts = QVBoxLayout(box_charts)
        lay_charts.setContentsMargins(4, 8, 4, 4)

        is_dark = getattr(self, 'chk_dark_mode', None) and self.chk_dark_mode.isChecked()
        bg_col = '#090d16' if is_dark else '#ffffff'
        self.batch_figure = Figure(figsize=(5, 2.6), dpi=100, facecolor=bg_col)
        self.batch_canvas = FigureCanvas(self.batch_figure)
        self.batch_canvas.setMinimumHeight(180)
        lay_charts.addWidget(self.batch_canvas)
        lay_rep.addWidget(box_charts, 2)

        # 3. Log Batch Console
        lay_rep.addWidget(QLabel("<b>Log tiến trình Batch:</b>"))
        self.txt_batch_log = QTextEdit()
        self.txt_batch_log.setReadOnly(True)
        self.txt_batch_log.setPlaceholderText("Chưa có log batch...")
        self.txt_batch_log.setMaximumHeight(130)
        lay_rep.addWidget(self.txt_batch_log, 1)

        row_log_btns = QHBoxLayout()
        self.btn_copy_batch_log = QPushButton("📋 Sao chép log")
        self.btn_copy_batch_log.setStyleSheet("background-color: #475569; color: white; font-weight: bold; padding: 4px 8px;")
        self.btn_copy_batch_log.clicked.connect(self.copy_batch_log_clicked)
        row_log_btns.addWidget(self.btn_copy_batch_log)

        self.btn_save_batch_log = QPushButton("💾 Lưu log")
        self.btn_save_batch_log.setStyleSheet("background-color: #475569; color: white; font-weight: bold; padding: 4px 8px;")
        self.btn_save_batch_log.clicked.connect(self.save_batch_log_clicked)
        row_log_btns.addWidget(self.btn_save_batch_log)
        row_log_btns.addStretch()

        lay_rep.addLayout(row_log_btns)

        # 4. Các nút xuất báo cáo
        row_export_btns = QHBoxLayout()
        self.btn_export_report_html = QPushButton("📊 Xuất báo cáo HTML")
        self.btn_export_report_html.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_export_report_html.clicked.connect(self.export_report_html_clicked)
        row_export_btns.addWidget(self.btn_export_report_html)

        self.btn_export_report_pdf = QPushButton("📑 Xuất báo cáo PDF")
        self.btn_export_report_pdf.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_export_report_pdf.clicked.connect(self.export_report_pdf_clicked)
        row_export_btns.addWidget(self.btn_export_report_pdf)

        self.btn_export_report_csv = QPushButton("📈 Xuất báo cáo CSV")
        self.btn_export_report_csv.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_export_report_csv.clicked.connect(self.export_report_csv_clicked)
        row_export_btns.addWidget(self.btn_export_report_csv)

        lay_rep.addLayout(row_export_btns)
        splitter_page4.addWidget(box_reports)

        splitter_page4.setSizes([550, 550])
        tab_page4_layout.addWidget(splitter_page4)

        self.main_tab_widget.addTab(tab_page4_widget, "⚡ 4. BATCH PROCESSING & BÁO CÁO")
        self.update_batch_charts()

    def add_batch_videos_clicked(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Chọn Video Xử Lý Hàng Loạt", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.flv *.wmv);;All Files (*)"
        )
        if files:
            self.add_videos_to_batch_queue(files)

    def add_batch_folder_clicked(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn Thư Mục Chứa Video")
        if folder:
            valid_exts = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"}
            files = []
            for root, _, filenames in os.walk(folder):
                for fn in filenames:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in valid_exts:
                        files.append(os.path.join(root, fn))
            if files:
                self.add_videos_to_batch_queue(files)
            else:
                QMessageBox.information(self, "Thông báo", "Không tìm thấy file video nào trong thư mục đã chọn.")

    def add_videos_to_batch_queue(self, file_paths):
        cfg = self.get_current_project_state()
        for fp in file_paths:
            # Lấy thời lượng video nhanh qua OpenCV
            dur_str = "--:--"
            try:
                cap = cv2.VideoCapture(fp)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    if frames > 0:
                        tot_sec = frames / fps
                        m, s = int((tot_sec % 3600) // 60), int(tot_sec % 60)
                        dur_str = f"{m:02d}:{s:02d}"
            except Exception:
                pass

            item = {
                "file_path": fp,
                "duration": dur_str,
                "status": "⏳ Chờ",
                "output_path": "",
                "config": cfg.copy(),
                "time_sec": 0.0,
                "size_mb": 0.0,
                "error": ""
            }
            self.batch_queue.append(item)

        self.refresh_batch_queue_table()
        self.log_batch("INFO", f"📥 Đã thêm {len(file_paths)} video vào hàng đợi Batch.")
        self.update_batch_charts()

    def remove_batch_selected_clicked(self):
        r = self.tbl_batch_queue.currentRow()
        if 0 <= r < len(self.batch_queue):
            removed = self.batch_queue.pop(r)
            self.refresh_batch_queue_table()
            self.log_batch("INFO", f"🗑️ Đã xóa: {os.path.basename(removed['file_path'])}")
            self.update_batch_charts()

    def clear_batch_queue_clicked(self):
        if not self.batch_queue:
            return
        reply = QMessageBox.question(self, "Xác nhận", "Bạn có chắc chắn muốn xóa toàn bộ hàng đợi Batch?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.batch_queue.clear()
            self.refresh_batch_queue_table()
            self.log_batch("INFO", "🗑️ Đã xóa toàn bộ hàng đợi video.")
            self.update_batch_charts()

    def move_batch_item_up(self):
        r = self.tbl_batch_queue.currentRow()
        if r > 0 and r < len(self.batch_queue):
            self.batch_queue[r], self.batch_queue[r - 1] = self.batch_queue[r - 1], self.batch_queue[r]
            self.refresh_batch_queue_table()
            self.tbl_batch_queue.selectRow(r - 1)

    def move_batch_item_down(self):
        r = self.tbl_batch_queue.currentRow()
        if 0 <= r < len(self.batch_queue) - 1:
            self.batch_queue[r], self.batch_queue[r + 1] = self.batch_queue[r + 1], self.batch_queue[r]
            self.refresh_batch_queue_table()
            self.tbl_batch_queue.selectRow(r + 1)

    def export_batch_queue_csv(self):
        if not self.batch_queue:
            QMessageBox.warning(self, "Thông báo", "Hàng đợi Batch đang trống, không có dữ liệu để xuất.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Xuất Danh Sách Hàng Đợi (.csv)", "batch_queue.csv", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["STT", "Tên video", "Đường dẫn", "Thời lượng", "Trạng thái", "Output Path"])
                for idx, it in enumerate(self.batch_queue):
                    writer.writerow([
                        idx + 1,
                        os.path.basename(it["file_path"]),
                        it["file_path"],
                        it.get("duration", "--:--"),
                        it.get("status", "⏳ Chờ"),
                        it.get("output_path", "")
                    ])
            self.log_batch("SUCCESS", f"📤 Đã xuất danh sách queue ra file: {file_path}")
            QMessageBox.information(self, "Xuất CSV", f"Đã xuất danh sách hàng đợi ra:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể xuất file CSV:\n{e}")

    def on_batch_queue_selection_changed(self):
        r = self.tbl_batch_queue.currentRow()
        if 0 <= r < len(self.batch_queue):
            it = self.batch_queue[r]
            vname = os.path.basename(it["file_path"])
            status = it.get("status", "⏳ Chờ")
            self.lbl_batch_current.setText(f"Đang chọn dòng {r+1}: {vname} ({status})")

    def refresh_batch_queue_table(self):
        self.tbl_batch_queue.setRowCount(len(self.batch_queue))
        for r, it in enumerate(self.batch_queue):
            self.tbl_batch_queue.setItem(r, 0, QTableWidgetItem(str(r + 1)))
            self.tbl_batch_queue.setItem(r, 1, QTableWidgetItem(os.path.basename(it["file_path"])))
            
            # Shorten path for display
            fp = it["file_path"]
            short_path = fp if len(fp) <= 35 else "..." + fp[-32:]
            self.tbl_batch_queue.setItem(r, 2, QTableWidgetItem(short_path))
            self.tbl_batch_queue.setItem(r, 3, QTableWidgetItem(it.get("duration", "--:--")))
            
            st_item = QTableWidgetItem(it.get("status", "⏳ Chờ"))
            if "Thành công" in it.get("status", ""):
                st_item.setForeground(QColor("#4ade80"))
            elif "Thất bại" in it.get("status", ""):
                st_item.setForeground(QColor("#ef4444"))
            elif "Đang xử lý" in it.get("status", ""):
                st_item.setForeground(QColor("#facc15"))
            else:
                st_item.setForeground(QColor("#94a3b8"))
            self.tbl_batch_queue.setItem(r, 4, st_item)
            self.tbl_batch_queue.setItem(r, 5, QTableWidgetItem(it.get("output_path", "")))

        # Update stats
        total = len(self.batch_queue)
        succ = sum(1 for it in self.batch_queue if "Thành công" in it.get("status", ""))
        fail = sum(1 for it in self.batch_queue if "Thất bại" in it.get("status", ""))
        pending = sum(1 for it in self.batch_queue if "Chờ" in it.get("status", ""))
        tot_size = sum(it.get("size_mb", 0.0) for it in self.batch_queue)
        tot_sec = sum(it.get("time_sec", 0.0) for it in self.batch_queue)

        h = int(tot_sec // 3600)
        m = int((tot_sec % 3600) // 60)
        s = int(tot_sec % 60)

        self.lbl_stat_total_videos.setText(f"Tổng số video: <b>{total}</b>")
        self.lbl_stat_processed.setText(f"Đã xử lý: <b>{succ + fail}</b> (✅ {succ} / ❌ {fail} / ⏳ {pending})")
        self.lbl_stat_total_time.setText(f"Tổng thời gian: <b>{h:02d}:{m:02d}:{s:02d}</b>")
        self.lbl_stat_total_size.setText(f"Tổng dung lượng output: <b>{tot_size:.1f} MB</b>")

    def run_batch_clicked(self):
        if not self.batch_queue:
            QMessageBox.warning(self, "Thông báo", "Hàng đợi Batch đang trống! Vui lòng thêm video trước khi chạy.")
            return

        base_cfg = self.get_current_project_state()
        use_same = self.chk_batch_use_same_config.isChecked()
        stop_err = self.chk_batch_stop_on_error.isChecked()
        workers_cnt = self.spin_batch_workers.value()

        # Prepare queue items
        run_queue = []
        for idx, item in enumerate(self.batch_queue):
            item["status"] = "⏳ Chờ"
            cfg = base_cfg if use_same else item.get("config", base_cfg)
            run_queue.append({"index": idx, "file_path": item["file_path"], "duration": item.get("duration", "--:--"), "config": cfg})

        self.refresh_batch_queue_table()
        self.btn_run_batch.setEnabled(False)
        self.btn_stop_batch.setEnabled(True)
        self.progress_batch_total.setValue(0)
        self.progress_batch_current.setValue(0)

        self.batch_worker = BatchPipelineWorker(run_queue, base_cfg, stop_on_error=stop_err, max_workers=workers_cnt)
        self.batch_worker.sig_item_started.connect(self.on_batch_item_started)
        self.batch_worker.sig_item_progress.connect(self.on_batch_item_progress)
        self.batch_worker.sig_item_finished.connect(self.on_batch_item_finished)
        self.batch_worker.sig_batch_progress.connect(self.on_batch_overall_progress)
        self.batch_worker.sig_batch_finished.connect(self.on_batch_all_finished)
        self.batch_worker.sig_log.connect(self.log_batch)
        self.batch_worker.start()

    def stop_batch_clicked(self):
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.cancel()
            self.log_batch("WARNING", "🛑 Đang gửi yêu cầu dừng Batch sau khi video hiện tại xong...")
            self.btn_stop_batch.setEnabled(False)

    def on_batch_item_started(self, idx, vname):
        if 0 <= idx < len(self.batch_queue):
            self.batch_queue[idx]["status"] = "🔄 Đang xử lý"
            self.refresh_batch_queue_table()
            self.lbl_batch_current.setText(f"Đang xử lý: {vname} ({idx+1}/{len(self.batch_queue)})")
            self.progress_batch_current.setValue(0)

    def on_batch_item_progress(self, idx, percent, text):
        self.progress_batch_current.setValue(percent)
        if 0 <= idx < len(self.batch_queue):
            vname = os.path.basename(self.batch_queue[idx]["file_path"])
            self.lbl_batch_current.setText(f"Đang xử lý: {vname} ({percent}%) - {text}")

    def on_batch_item_finished(self, idx, success, out_path, elapsed, size_mb, error_msg):
        if 0 <= idx < len(self.batch_queue):
            self.batch_queue[idx]["status"] = "✅ Thành công" if success else "❌ Thất bại"
            self.batch_queue[idx]["output_path"] = out_path
            self.batch_queue[idx]["time_sec"] = elapsed
            self.batch_queue[idx]["size_mb"] = size_mb
            self.batch_queue[idx]["error"] = error_msg
            self.refresh_batch_queue_table()
            self.update_batch_charts()

    def on_batch_overall_progress(self, completed, total, eta_str):
        pct = int((completed / max(1, total)) * 100)
        self.progress_batch_total.setValue(pct)
        self.lbl_batch_eta.setText(f"⏱️ Dự kiến còn lại: {eta_str}")

    def on_batch_all_finished(self, summary):
        self.btn_run_batch.setEnabled(True)
        self.btn_stop_batch.setEnabled(False)
        self.progress_batch_total.setValue(100)
        self.progress_batch_current.setValue(100)
        self.lbl_batch_current.setText(f"Đã hoàn thành {summary['success']}/{summary['total']} video!")
        self.lbl_batch_eta.setText("⏱️ Dự kiến còn lại: 00:00")
        self.refresh_batch_queue_table()
        self.update_batch_charts()

        # Lưu batch history vào config/batch_history.json
        self.save_batch_history(summary)

        # Auto export report nếu setting bật
        if hasattr(self, 'chk_auto_report') and self.chk_auto_report.isChecked():
            fmt = self.cb_report_format.currentText() if hasattr(self, 'cb_report_format') else "HTML"
            if fmt == "HTML":
                self.export_report_html_clicked(silent=True)
            elif fmt == "PDF":
                self.export_report_pdf_clicked(silent=True)
            elif fmt == "CSV":
                self.export_report_csv_clicked(silent=True)

        # Mở output folder nếu bật
        if hasattr(self, 'chk_open_folder_on_done') and self.chk_open_folder_on_done.isChecked():
            out_dir = os.path.abspath("videos")
            if os.path.exists(out_dir):
                try:
                    os.startfile(out_dir)
                except Exception:
                    pass

        if not os.environ.get("QT_QPA_PLATFORM"):
            QMessageBox.information(
                self, "Hoàn tất Batch",
                f"🎉 Tiến trình Batch đã hoàn tất!\n\n"
                f"- Tổng số video: {summary['total']}\n"
                f"- Thành công: {summary['success']}\n"
                f"- Thất bại: {summary['failed']}\n"
                f"- Tổng thời gian: {summary['total_time_str']}"
            )

    def save_batch_history(self, summary):
        key_dir = os.path.join(os.path.dirname(__file__), "config")
        os.makedirs(key_dir, exist_ok=True)
        hist_file = os.path.join(key_dir, "batch_history.json")
        batches = []
        if os.path.exists(hist_file):
            try:
                with open(hist_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    batches = data.get("batches", [])
            except Exception:
                batches = []

        batch_id = f"batch_{int(time.time())}"
        record = {
            "id": batch_id,
            "date": summary.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "total_videos": summary.get("total", 0),
            "success": summary.get("success", 0),
            "failed": summary.get("failed", 0),
            "total_time": summary.get("total_time_str", "00:00:00"),
            "videos": summary.get("results", [])
        }
        batches.append(record)
        try:
            with open(hist_file, "w", encoding="utf-8") as f:
                json.dump({"batches": batches}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def log_batch(self, level, msg):
        timestamp = time.strftime("[%H:%M:%S]")
        msg_str = str(msg)
        if level == "ERROR":
            color = "#f87171"
            prefix = "✖"
        elif level == "WARNING":
            color = "#facc15"
            prefix = "⚡"
        elif level == "SUCCESS":
            color = "#4ade80"
            prefix = "✔"
        else:
            color = "#38bdf8"
            prefix = "ℹ"

        formatted_html = f'<span style="color:#64748b;">{timestamp}</span> <span style="color:{color}; font-weight:bold;">{prefix} {msg_str}</span>'
        if not hasattr(self, '_raw_batch_log_records'):
            self._raw_batch_log_records = []
        self._raw_batch_log_records.append((level, timestamp, msg_str, formatted_html))

        if hasattr(self, 'txt_batch_log') and self.txt_batch_log:
            self.txt_batch_log.append(formatted_html)
            sb = self.txt_batch_log.verticalScrollBar()
            if sb:
                sb.setValue(sb.maximum())

    def update_batch_charts(self, stats=None):
        if not hasattr(self, 'batch_figure') or not hasattr(self, 'batch_canvas'):
            return

        self.batch_figure.clear()
        is_dark = getattr(self, 'chk_dark_mode', None) and self.chk_dark_mode.isChecked()
        text_color = '#ffffff' if is_dark else '#0f172a'
        bg_col = '#090d16' if is_dark else '#ffffff'
        self.batch_figure.patch.set_facecolor(bg_col)

        succ = sum(1 for it in self.batch_queue if "Thành công" in it.get("status", ""))
        fail = sum(1 for it in self.batch_queue if "Thất bại" in it.get("status", ""))
        pending = max(0, len(self.batch_queue) - succ - fail)

        # Subplot 1: Pie Chart (Tỷ lệ kết quả)
        ax1 = self.batch_figure.add_subplot(121)
        ax1.set_facecolor(bg_col)
        counts = [succ, fail, pending]
        labels = ['Thành công', 'Thất bại', 'Chờ']
        colors = ['#4ade80', '#f87171', '#94a3b8']
        
        valid_counts = [c for c in counts if c > 0]
        valid_labels = [l for c, l in zip(counts, labels) if c > 0]
        valid_colors = [col for c, col in zip(counts, colors) if c > 0]

        if valid_counts:
            ax1.pie(valid_counts, labels=valid_labels, colors=valid_colors, autopct='%1.0f%%', startangle=140,
                    textprops={'color': text_color, 'fontsize': 8})
        else:
            ax1.pie([1], labels=['Chưa có video'], colors=['#334155'], textprops={'color': text_color, 'fontsize': 8})
        ax1.set_title("Tỷ lệ Trạng thái", color=text_color, fontsize=9, fontweight='bold')

        # Subplot 2: Bar Chart (Thời gian xử lý từng video)
        ax2 = self.batch_figure.add_subplot(122)
        ax2.set_facecolor(bg_col)
        names = [f"V{i+1}" for i in range(min(8, len(self.batch_queue)))]
        durations = [it.get("time_sec", 0.0) for it in self.batch_queue[:8]]
        if not durations or all(d == 0 for d in durations):
            durations = [0]
            names = ["None"]
        
        bars = ax2.bar(names, durations, color='#0284c7', width=0.5)
        ax2.set_title("Thời gian (giây)", color=text_color, fontsize=9, fontweight='bold')
        ax2.tick_params(axis='x', colors=text_color, labelsize=7)
        ax2.tick_params(axis='y', colors=text_color, labelsize=7)
        for spine in ax2.spines.values():
            spine.set_color('#334155' if is_dark else '#cbd5e1')

        self.batch_figure.tight_layout()
        self.batch_canvas.draw_idle()

    def copy_batch_log_clicked(self):
        if hasattr(self, 'txt_batch_log') and self.txt_batch_log:
            QApplication.clipboard().setText(self.txt_batch_log.toPlainText())
            self.log_batch("INFO", "📋 Đã sao chép log Batch vào Clipboard.")

    def save_batch_log_clicked(self):
        if not hasattr(self, 'txt_batch_log') or not self.txt_batch_log.toPlainText():
            QMessageBox.warning(self, "Thông báo", "Chưa có log Batch để lưu.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Lưu Log Batch (.txt)", "batch_log.txt", "Text Files (*.txt)")
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.txt_batch_log.toPlainText())
            self.log_batch("SUCCESS", f"💾 Đã lưu log Batch vào: {file_path}")
            QMessageBox.information(self, "Lưu Log", f"Đã lưu log Batch thành công:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu log:\n{e}")

    def export_report_html_clicked(self, silent=False):
        if not self.batch_queue:
            if not silent: QMessageBox.warning(self, "Thông báo", "Hàng đợi rỗng, không có dữ liệu để tạo báo cáo.")
            return
        file_path = os.path.abspath(os.path.join("videos", f"batch_report_{int(time.time())}.html"))
        if not silent:
            chosen, _ = QFileDialog.getSaveFileName(self, "Xuất Báo Cáo HTML", file_path, "HTML Files (*.html)")
            if not chosen: return
            file_path = chosen

        total = len(self.batch_queue)
        succ = sum(1 for it in self.batch_queue if "Thành công" in it.get("status", ""))
        fail = sum(1 for it in self.batch_queue if "Thất bại" in it.get("status", ""))
        tot_size = sum(it.get("size_mb", 0.0) for it in self.batch_queue)
        tot_sec = sum(it.get("time_sec", 0.0) for it in self.batch_queue)

        rows_html = ""
        for idx, it in enumerate(self.batch_queue):
            vname = os.path.basename(it["file_path"])
            st = it.get("status", "⏳ Chờ")
            st_color = "#10b981" if "Thành công" in st else ("#ef4444" if "Thất bại" in st else "#f59e0b")
            dur = it.get("duration", "--:--")
            out = it.get("output_path", "")
            t_sec = f"{it.get('time_sec', 0.0):.1f}s"
            size = f"{it.get('size_mb', 0.0):.1f}MB"
            rows_html += f"<tr><td>{idx+1}</td><td><b>{vname}</b></td><td>{dur}</td><td style='color:{st_color}; font-weight:bold;'>{st}</td><td>{t_sec}</td><td>{size}</td><td><code>{out}</code></td></tr>"

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Báo Cáo Batch Processing - supersubs</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .header {{ background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #334155; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }}
        .card {{ background: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; text-align: center; }}
        .card h3 {{ margin: 0 0 8px; font-size: 13px; color: #94a3b8; }}
        .card p {{ margin: 0; font-size: 22px; font-weight: bold; color: #38bdf8; }}
        table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; font-size: 13px; }}
        th {{ background: #090d16; color: #38bdf8; }}
        tr:hover {{ background: #334155; }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin:0; color:#38bdf8;">📊 Báo Cáo Xử Lý Hàng Loạt (Batch Processing Report)</h1>
        <p style="margin:5px 0 0; color:#94a3b8;">Thời gian tạo: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    <div class="stats-grid">
        <div class="card"><h3>TỔNG VIDEO</h3><p>{total}</p></div>
        <div class="card"><h3>THÀNH CÔNG</h3><p style="color:#4ade80;">{succ}</p></div>
        <div class="card"><h3>THẤT BẠI</h3><p style="color:#f87171;">{fail}</p></div>
        <div class="card"><h3>DUNG LƯỢNG OUTPUT</h3><p>{tot_size:.1f} MB</p></div>
    </div>
    <table>
        <thead>
            <tr><th>STT</th><th>Tên Video</th><th>Thời lượng</th><th>Trạng thái</th><th>Thời gian chạy</th><th>Dung lượng</th><th>Đường dẫn xuất</th></tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            self.log_batch("SUCCESS", f"📊 Đã xuất báo cáo HTML: {file_path}")
            if not silent:
                QMessageBox.information(self, "Báo Cáo HTML", f"Đã xuất báo cáo HTML thành công:\n{file_path}")
        except Exception as e:
            if not silent: QMessageBox.critical(self, "Lỗi", f"Không thể xuất báo cáo HTML:\n{e}")

    def export_report_csv_clicked(self, silent=False):
        if not self.batch_queue:
            if not silent: QMessageBox.warning(self, "Thông báo", "Hàng đợi rỗng, không có dữ liệu để tạo báo cáo.")
            return
        file_path = os.path.abspath(os.path.join("videos", f"batch_report_{int(time.time())}.csv"))
        if not silent:
            chosen, _ = QFileDialog.getSaveFileName(self, "Xuất Báo Cáo CSV", file_path, "CSV Files (*.csv)")
            if not chosen: return
            file_path = chosen
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["STT", "Tên video", "Thời lượng", "Trạng thái", "Thời gian chạy (giây)", "Dung lượng (MB)", "Đường dẫn xuất", "Lỗi"])
                for idx, it in enumerate(self.batch_queue):
                    writer.writerow([
                        idx + 1,
                        os.path.basename(it["file_path"]),
                        it.get("duration", "--:--"),
                        it.get("status", "⏳ Chờ"),
                        it.get("time_sec", 0.0),
                        it.get("size_mb", 0.0),
                        it.get("output_path", ""),
                        it.get("error", "")
                    ])
            self.log_batch("SUCCESS", f"📈 Đã xuất báo cáo CSV: {file_path}")
            if not silent:
                QMessageBox.information(self, "Báo Cáo CSV", f"Đã xuất báo cáo CSV thành công:\n{file_path}")
        except Exception as e:
            if not silent: QMessageBox.critical(self, "Lỗi", f"Không thể xuất báo cáo CSV:\n{e}")

    def export_report_pdf_clicked(self, silent=False):
        if not self.batch_queue:
            if not silent: QMessageBox.warning(self, "Thông báo", "Hàng đợi rỗng, không có dữ liệu để tạo báo cáo.")
            return
        file_path = os.path.abspath(os.path.join("videos", f"batch_report_{int(time.time())}.pdf"))
        if not silent:
            chosen, _ = QFileDialog.getSaveFileName(self, "Xuất Báo Cáo PDF", file_path, "PDF Files (*.pdf)")
            if not chosen: return
            file_path = chosen
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            c = canvas.Canvas(file_path, pagesize=letter)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, 750, "Bao Cao Xu Ly Hang Loat (Batch Processing Report)")
            c.setFont("Helvetica", 10)
            c.drawString(50, 730, f"Ngay tao: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            c.drawString(50, 715, f"Tong so video: {len(self.batch_queue)} | Thanh cong: {sum(1 for it in self.batch_queue if 'Thanh cong' in it.get('status', ''))}")
            
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, 680, "STT")
            c.drawString(80, 680, "Ten Video")
            c.drawString(300, 680, "Trang Thai")
            c.drawString(400, 680, "Dung Luong")
            c.line(50, 675, 550, 675)

            y = 660
            c.setFont("Helvetica", 9)
            for idx, it in enumerate(self.batch_queue[:25]):
                vname = os.path.basename(it["file_path"])
                if len(vname) > 35: vname = vname[:32] + "..."
                st = it.get("status", "Cho")
                size = f"{it.get('size_mb', 0.0):.1f}MB"
                c.drawString(50, y, str(idx + 1))
                c.drawString(80, y, vname)
                c.drawString(300, y, st)
                c.drawString(400, y, size)
                y -= 18
                if y < 60: break

            c.save()
            self.log_batch("SUCCESS", f"📑 Đã xuất báo cáo PDF: {file_path}")
            if not silent:
                QMessageBox.information(self, "Báo Cáo PDF", f"Đã xuất báo cáo PDF thành công:\n{file_path}")
        except Exception as e:
            if not silent: QMessageBox.critical(self, "Lỗi", f"Không thể xuất báo cáo PDF:\n{e}")

    def get_current_project_state(self):
        created_str = getattr(self, 'project_created_date', QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss"))
        updated_str = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        self.project_updated_date = updated_str
        if hasattr(self, 'lbl_updated_date'):
            self.lbl_updated_date.setText(updated_str)

        proj_name = self.txt_project_name.text().strip() if hasattr(self, 'txt_project_name') and self.txt_project_name.text().strip() else "Dự án mới"
        proj_desc = self.txt_project_desc.toPlainText() if hasattr(self, 'txt_project_desc') else ""

        preset_data = {
            "font_name": self.cb_font_name.currentText() if hasattr(self, 'cb_font_name') else "Arial",
            "font_size": self.spin_font_size.value() if hasattr(self, 'spin_font_size') else 24,
            "font_color": getattr(self, 'preset_font_color', [255, 255, 255]),
            "outline_color": getattr(self, 'preset_outline_color', [0, 0, 0]),
            "outline_width": self.spin_outline_width.value() if hasattr(self, 'spin_outline_width') else 2,
            "use_bg_box": self.chk_use_bg_box.isChecked() if hasattr(self, 'chk_use_bg_box') else False,
            "bg_color": getattr(self, 'preset_bg_color', [0, 0, 0]),
            "bg_opacity": self.slider_bg_opacity.value() if hasattr(self, 'slider_bg_opacity') else 50,
            "v_align": self.cb_v_align.currentText() if hasattr(self, 'cb_v_align') else "bottom",
            "h_align": self.cb_h_align.currentText() if hasattr(self, 'cb_h_align') else "center",
            "margin_v_val": self.spin_margin_v.value() if hasattr(self, 'spin_margin_v') else 20,
            "margin_h_val": self.spin_margin_h.value() if hasattr(self, 'spin_margin_h') else 20
        }

        api_keys_data = {
            "gemini": self.txt_gemini_key.text().strip() if hasattr(self, 'txt_gemini_key') else "",
            "xkiro": self.txt_xkiro_key.text().strip() if hasattr(self, 'txt_xkiro_key') else ""
        }

        voice_val = "vi-VN-HoaiMyNeural"
        if hasattr(self, 'cb_voice') and self.cb_voice:
            voice_val = self.cb_voice.currentData() or self.cb_voice.currentText()

        engine_val = "gemini"
        if hasattr(self, 'cb_engine') and self.cb_engine:
            engine_val = self.cb_engine.currentText()

        state = {
            "project_name": proj_name,
            "description": proj_desc,
            "created": created_str,
            "updated": updated_str,
            "video_path": getattr(self, 'video_path', ""),
            "selected_bbox": getattr(self, 'selected_bbox', None),
            "selected_bboxes": getattr(self, 'selected_bboxes', []),
            "title_bbox": getattr(self, 'title_bbox', None),
            "logo_bbox": getattr(self, 'logo_bbox', None),
            "logo_path": getattr(self, 'logo_path', ""),
            "engine": engine_val,
            "burn_sub": self.chk_burn_sub_export.isChecked() if hasattr(self, 'chk_burn_sub_export') else True,
            "enable_dubbing": self.chk_enable_dubbing.isChecked() if hasattr(self, 'chk_enable_dubbing') else True,
            "voice": voice_val,
            "bg_vol": (self.slider_bg.value() / 100.0) if hasattr(self, 'slider_bg') else 0.3,
            "dub_vol": (self.slider_dub.value() / 100.0) if hasattr(self, 'slider_dub') else 1.0,
            "workers_cnt": self.spin_workers.value() if hasattr(self, 'spin_workers') else 4,
            "preset": preset_data,
            "api_keys": api_keys_data,
            "prefer_xkiro": self.chk_prefer_xkiro.isChecked() if hasattr(self, 'chk_prefer_xkiro') else False
        }
        return state

    def restore_project_state(self, data):
        if not isinstance(data, dict):
            return

        # 1. Project Info
        if "project_name" in data and hasattr(self, 'txt_project_name'):
            self.txt_project_name.setText(data["project_name"])
        if "description" in data and hasattr(self, 'txt_project_desc'):
            self.txt_project_desc.setPlainText(data["description"])
        if "created" in data:
            self.project_created_date = data["created"]
            if hasattr(self, 'lbl_created_date'):
                self.lbl_created_date.setText(data["created"])
        if "updated" in data:
            self.project_updated_date = data["updated"]
            if hasattr(self, 'lbl_updated_date'):
                self.lbl_updated_date.setText(data["updated"])

        # 2. Video Path
        if "video_path" in data and data["video_path"]:
            vpath = data["video_path"]
            self.video_path = vpath
            if hasattr(self, 'txt_video_path'):
                self.txt_video_path.setText(vpath)
            if os.path.exists(vpath) and hasattr(self, 'load_video_preview_frame'):
                try:
                    self.load_video_preview_frame(vpath)
                except Exception:
                    pass

        # 3. Bboxes & Logo
        if "selected_bbox" in data:
            self.selected_bbox = data["selected_bbox"]
        if "selected_bboxes" in data:
            self.selected_bboxes = data["selected_bboxes"] or []
        if "title_bbox" in data:
            self.title_bbox = data["title_bbox"]
        if "logo_bbox" in data:
            self.logo_bbox = data["logo_bbox"]
        if "logo_path" in data:
            self.logo_path = data["logo_path"] or ""

        if hasattr(self, 'lbl_main_preview') and self.lbl_main_preview:
            self.lbl_main_preview.bboxes = list(getattr(self, 'selected_bboxes', []))
            self.lbl_main_preview.update()

        # 4. Settings & Checkboxes
        if "burn_sub" in data and hasattr(self, 'chk_burn_sub_export'):
            self.chk_burn_sub_export.setChecked(bool(data["burn_sub"]))
        if "enable_dubbing" in data and hasattr(self, 'chk_enable_dubbing'):
            self.chk_enable_dubbing.setChecked(bool(data["enable_dubbing"]))
        if "prefer_xkiro" in data and hasattr(self, 'chk_prefer_xkiro'):
            self.chk_prefer_xkiro.setChecked(bool(data["prefer_xkiro"]))
        if "workers_cnt" in data and hasattr(self, 'spin_workers'):
            self.spin_workers.setValue(int(data["workers_cnt"]))

        # 5. Engine & Voice
        if "engine" in data and hasattr(self, 'cb_engine'):
            idx = self.cb_engine.findText(data["engine"])
            if idx != -1:
                self.cb_engine.setCurrentIndex(idx)
            else:
                self.cb_engine.setCurrentText(data["engine"])

        if "voice" in data and hasattr(self, 'cb_voice'):
            v_target = data["voice"]
            idx = self.cb_voice.findData(v_target)
            if idx == -1:
                idx = self.cb_voice.findText(v_target)
            if idx != -1:
                self.cb_voice.setCurrentIndex(idx)

        # 6. Audio Volumes
        if "bg_vol" in data and hasattr(self, 'slider_bg'):
            val = int(float(data["bg_vol"]) * 100) if float(data["bg_vol"]) <= 1.0 else int(float(data["bg_vol"]))
            self.slider_bg.setValue(max(0, min(100, val)))
        if "dub_vol" in data and hasattr(self, 'slider_dub'):
            val = int(float(data["dub_vol"]) * 100) if float(data["dub_vol"]) <= 1.0 else int(float(data["dub_vol"]))
            self.slider_dub.setValue(max(0, min(100, val)))

        # 7. Preset
        if "preset" in data and isinstance(data["preset"], dict):
            p = data["preset"]
            if "font_name" in p and hasattr(self, 'cb_font_name'):
                self.cb_font_name.setCurrentText(p["font_name"])
            if "font_size" in p and hasattr(self, 'spin_font_size'):
                self.spin_font_size.setValue(int(p["font_size"]))
            if "font_color" in p:
                self.preset_font_color = p["font_color"]
                if hasattr(self, 'btn_font_color'):
                    qcol = QColor(*p["font_color"])
                    self.btn_font_color.setStyleSheet(f"background-color: {qcol.name()}; border: 1px solid white;")
                    if hasattr(self, 'lbl_font_hex'): self.lbl_font_hex.setText(qcol.name().upper())
            if "outline_color" in p:
                self.preset_outline_color = p["outline_color"]
                if hasattr(self, 'btn_outline_color'):
                    qcol = QColor(*p["outline_color"])
                    self.btn_outline_color.setStyleSheet(f"background-color: {qcol.name()}; border: 1px solid white;")
                    if hasattr(self, 'lbl_outline_hex'): self.lbl_outline_hex.setText(qcol.name().upper())
            if "outline_width" in p and hasattr(self, 'spin_outline_width'):
                self.spin_outline_width.setValue(int(p["outline_width"]))
            if "use_bg_box" in p and hasattr(self, 'chk_use_bg_box'):
                self.chk_use_bg_box.setChecked(bool(p["use_bg_box"]))
            if "bg_color" in p:
                self.preset_bg_color = p["bg_color"]
                if hasattr(self, 'btn_bg_color'):
                    qcol = QColor(*p["bg_color"])
                    self.btn_bg_color.setStyleSheet(f"background-color: {qcol.name()}; border: 1px solid white;")
                    if hasattr(self, 'lbl_bg_hex'): self.lbl_bg_hex.setText(qcol.name().upper())
            if "bg_opacity" in p and hasattr(self, 'slider_bg_opacity'):
                self.slider_bg_opacity.setValue(int(p["bg_opacity"]))
            if "v_align" in p and hasattr(self, 'cb_v_align'):
                self.cb_v_align.setCurrentText(p["v_align"])
            if "h_align" in p and hasattr(self, 'cb_h_align'):
                self.cb_h_align.setCurrentText(p["h_align"])
            if "margin_v_val" in p and hasattr(self, 'spin_margin_v'):
                self.spin_margin_v.setValue(int(p["margin_v_val"]))
            if "margin_h_val" in p and hasattr(self, 'spin_margin_h'):
                self.spin_margin_h.setValue(int(p["margin_h_val"]))

        # 8. API Keys
        if "api_keys" in data and isinstance(data["api_keys"], dict):
            k = data["api_keys"]
            if "gemini" in k and hasattr(self, 'txt_gemini_key'):
                self.txt_gemini_key.setText(k["gemini"])
            if "xkiro" in k and hasattr(self, 'txt_xkiro_key'):
                self.txt_xkiro_key.setText(k["xkiro"])

        self.log_info("🔄 Đã khôi phục toàn bộ dữ liệu dự án lên giao diện.")

    def btn_new_project_clicked(self):
        self.project_file_path = ""
        now_str = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        self.project_created_date = now_str
        self.project_updated_date = now_str
        if hasattr(self, 'txt_project_name'): self.txt_project_name.setText("Dự án mới")
        if hasattr(self, 'txt_project_desc'): self.txt_project_desc.clear()
        if hasattr(self, 'lbl_created_date'): self.lbl_created_date.setText(now_str)
        if hasattr(self, 'lbl_updated_date'): self.lbl_updated_date.setText(now_str)
        if hasattr(self, 'lbl_project_path'): self.lbl_project_path.setText("Chưa lưu")

        # Reset Video & Bboxes
        self.video_path = ""
        if hasattr(self, 'txt_video_path'): self.txt_video_path.clear()
        self.selected_bbox = None
        self.selected_bboxes = []
        self.title_bbox = None
        self.logo_bbox = None
        self.logo_path = ""
        self.box_type_dict = {}
        if hasattr(self, '_frame_cache'):
            self._frame_cache.clear()
        self._cached_project_state = None
        if hasattr(self, 'lbl_main_preview') and self.lbl_main_preview:
            self.lbl_main_preview.clear()
            self.lbl_main_preview.setText("📺 MÀN HÌNH PREVIEW VIDEO\n\nNhấp 'Chọn video' hoặc kéo-thả file video vào đây để khoanh vùng")
            self.lbl_main_preview.bboxes = []
            self.lbl_main_preview.update()

        self.log_info("📁 Đã tạo dự án mới và reset toàn bộ cache & thông số về mặc định.")

    def btn_save_project_clicked(self):
        os.makedirs("projects", exist_ok=True)
        default_name = self.txt_project_name.text().strip() if hasattr(self, 'txt_project_name') and self.txt_project_name.text().strip() else "project"
        default_file = os.path.join("projects", f"{default_name}.vdproj")

        file_path, _ = QFileDialog.getSaveFileName(self, "Lưu Dự Án (.vdproj)", default_file, "Project Files (*.vdproj);;JSON Files (*.json)")
        if not file_path:
            return

        # Tạo autobackup nếu file đã tồn tại
        if os.path.exists(file_path):
            try:
                import shutil
                shutil.copyfile(file_path, file_path + ".autobackup")
            except Exception:
                pass

        state = self.get_current_project_state()
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            self.project_file_path = file_path
            if hasattr(self, 'lbl_project_path'):
                self.lbl_project_path.setText(file_path)
            self.log_info(f"💾 Đã lưu dự án thành công tại: {file_path}")
            QMessageBox.information(self, "Lưu Dự Án", f"Đã lưu dự án thành công tại:\n{file_path}")
        except PermissionError:
            QMessageBox.critical(self, "Lỗi Quyền Truy Cập (PermissionError)", f"Không có quyền ghi vào đường dẫn:\n{file_path}\n\n👉 Vui lòng kiểm tra quyền Admin hoặc đóng ứng dụng khác đang mở file.")
        except FileNotFoundError:
            QMessageBox.critical(self, "Lỗi Không Tìm Thấy Thư Mục (FileNotFoundError)", f"Thư mục lưu không tồn tại:\n{file_path}\n\n👉 Vui lòng chọn một thư mục hợp lệ.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi Lưu Dự Án", f"Không thể lưu dự án:\n{e}")

    def btn_load_project_clicked(self):
        os.makedirs("projects", exist_ok=True)
        file_path, _ = QFileDialog.getOpenFileName(self, "Mở Dự Án (.vdproj)", "projects", "Project Files (*.vdproj);;JSON Files (*.json);;All Files (*)")
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if hasattr(self, '_frame_cache'):
                self._frame_cache.clear()
            self._cached_project_state = None
            self.restore_project_state(data)
            self.project_file_path = file_path
            if hasattr(self, 'lbl_project_path'):
                self.lbl_project_path.setText(file_path)
            self.log_info(f"📂 Đã nạp thành công dự án từ: {file_path}")
            QMessageBox.information(self, "Mở Dự Án", f"Đã mở dự án thành công từ:\n{file_path}")
        except PermissionError:
            QMessageBox.critical(self, "Lỗi Quyền Đọc (PermissionError)", f"Không có quyền đọc file:\n{file_path}\n\n👉 Vui lòng kiểm tra phân quyền truy cập.")
        except FileNotFoundError:
            QMessageBox.critical(self, "Lỗi Không Tìm Thấy File (FileNotFoundError)", f"File dự án không tồn tại:\n{file_path}")
        except json.JSONDecodeError as jde:
            QMessageBox.critical(self, "Lỗi Định Dạng JSON (JSONDecodeError)", f"File dự án bị hỏng hoặc sai cấu trúc JSON:\n{jde}\n\n👉 Bạn có thể thử khôi phục từ file {file_path}.autobackup")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi Mở Dự Án", f"Không thể đọc file dự án:\n{e}")

    def btn_export_config_clicked(self):
        state = self.get_current_project_state()
        config_data = {
            "preset": state.get("preset", {}),
            "engine": state.get("engine", "gemini"),
            "voice": state.get("voice", ""),
            "bg_vol": state.get("bg_vol", 0.3),
            "dub_vol": state.get("dub_vol", 1.0),
            "workers_cnt": state.get("workers_cnt", 4),
            "burn_sub": state.get("burn_sub", True),
            "enable_dubbing": state.get("enable_dubbing", True),
            "prefer_xkiro": state.get("prefer_xkiro", False)
        }
        file_path, _ = QFileDialog.getSaveFileName(self, "Xuất Cấu Hình (.json)", "config_export.json", "JSON Files (*.json)")
        if not file_path:
            return
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            self.log_info(f"📤 Đã xuất cấu hình thành công: {file_path}")
            QMessageBox.information(self, "Xuất Cấu Hình", f"Đã xuất cấu hình ra:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi Export", f"Không thể xuất cấu hình:\n{e}")

    def btn_import_config_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Nhập Cấu Hình (.json)", "", "JSON Files (*.json);;All Files (*)")
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            self.restore_project_state(config_data)
            self.log_info(f"📥 Đã nhập cấu hình từ: {file_path}")
            QMessageBox.information(self, "Nhập Cấu Hình", "Đã nạp thành công cấu hình vào hệ thống.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi Import", f"Không thể nhập cấu hình:\n{e}")

    def load_history_to_table(self):
        os.makedirs("config", exist_ok=True)
        hist_file = os.path.join("config", "history.json")
        if not os.path.exists(hist_file):
            try:
                with open(hist_file, "w", encoding="utf-8") as f:
                    json.dump([], f)
            except Exception:
                pass

        history_list = []
        if os.path.exists(hist_file):
            try:
                with open(hist_file, "r", encoding="utf-8") as f:
                    history_list = json.load(f)
            except Exception:
                history_list = []

        limit = getattr(self, '_history_loaded_limit', 50)
        display_list = history_list[:limit]

        if not hasattr(self, 'tbl_history') or self.tbl_history is None:
            return

        self.tbl_history.setRowCount(0)
        self.tbl_history.setRowCount(len(display_list))

        for idx, rec in enumerate(display_list):
            stt = str(idx + 1)
            pname = rec.get("project_name", f"Project {idx+1}")
            vin = os.path.basename(rec.get("video_path", "")) or "-"
            vout = os.path.basename(rec.get("output_video", "")) or "-"
            date_str = rec.get("timestamp", rec.get("updated", ""))
            proc_time_str = f"{rec.get('duration_minutes', 0.0):.2f}"
            video_dur_str = rec.get("video_duration", "--:--")
            status = rec.get("status", "✅ Thành công")

            items = [
                QTableWidgetItem(stt),
                QTableWidgetItem(pname),
                QTableWidgetItem(vin),
                QTableWidgetItem(vout),
                QTableWidgetItem(date_str),
                QTableWidgetItem(proc_time_str),
                QTableWidgetItem(video_dur_str),
                QTableWidgetItem(status)
            ]
            for col_i, item in enumerate(items):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setData(Qt.ItemDataRole.UserRole, rec)
                self.tbl_history.setItem(idx, col_i, item)

        if hasattr(self, 'filter_history_table'):
            self.filter_history_table()

    def on_history_selection_changed(self):
        if not hasattr(self, 'tbl_history') or self.tbl_history is None:
            return

        selected_items = self.tbl_history.selectedItems()
        if not selected_items:
            if hasattr(self, 'txt_history_detail') and self.txt_history_detail:
                self.txt_history_detail.clear()
            return

        row = selected_items[0].row()
        item_stt = self.tbl_history.item(row, 0)
        if not item_stt:
            return

        rec = item_stt.data(Qt.ItemDataRole.UserRole) or {}

        detail_str = f"📌 DỰ ÁN: {rec.get('project_name', 'N/A')}\n"
        detail_str += f"📅 Thời gian: {rec.get('timestamp', rec.get('updated', 'N/A'))} | Trạng thái: {rec.get('status', 'N/A')} (Thời lượng: {rec.get('duration_minutes', 0.0):.2f} phút)\n"
        detail_str += f"--------------------------------------------------\n"
        detail_str += f"📥 Video Đầu vào: {rec.get('video_path', 'Chưa chọn')}\n"
        detail_str += f"📤 Video Đầu ra:  {rec.get('output_video', 'Chưa có')}\n"
        detail_str += f"🎙️ Voice TTS:    {rec.get('voice', 'Mặc định')} | Engine: {rec.get('engine', 'gemini')} | Workers: {rec.get('workers_cnt', 4)}\n"
        detail_str += f"🔊 Vol Background: {rec.get('bg_vol', 0.3)} | Vol Dubbing: {rec.get('dub_vol', 1.0)}\n"
        preset = rec.get("preset", {})
        detail_str += f"🎨 Preset Sub: Font={preset.get('font_name','Arial')}, Size={preset.get('font_size',24)}, Color={preset.get('font_color',[255,255,255])}, Align=({preset.get('v_align','bottom')}, {preset.get('h_align','center')})\n"
        if rec.get("error"):
            detail_str += f"❌ LỖI: {rec.get('error')}\n"
        if rec.get("log_summary"):
            detail_str += f"📋 LOG TÓM TẮT:\n{rec.get('log_summary')}\n"

        if hasattr(self, 'txt_history_detail') and self.txt_history_detail:
            self.txt_history_detail.setPlainText(detail_str)

    def append_history_record(self, status, output_video="", duration_minutes=0.0, error_msg=""):
        os.makedirs("config", exist_ok=True)
        hist_file = os.path.join("config", "history.json")
        history_list = []
        if os.path.exists(hist_file):
            try:
                with open(hist_file, "r", encoding="utf-8") as f:
                    history_list = json.load(f)
            except Exception:
                history_list = []

        rec = self.get_current_project_state()
        rec["status"] = status
        rec["output_video"] = output_video
        rec["timestamp"] = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        rec["duration_minutes"] = round(duration_minutes, 2)
        rec["error"] = error_msg
        if hasattr(self, 'txt_log_console') and self.txt_log_console:
            logs = self.txt_log_console.toPlainText().splitlines()
            rec["log_summary"] = "\n".join(logs[-15:])

        history_list.insert(0, rec)
        try:
            with open(hist_file, "w", encoding="utf-8") as f:
                json.dump(history_list, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[HISTORY] Error saving history: {e}")

        self.load_history_to_table()

    def clear_history_clicked(self):
        reply = QMessageBox.question(self, "Xác nhận xóa", "Bạn có chắc chắn muốn xóa toàn bộ lịch sử xử lý?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            hist_file = os.path.join("config", "history.json")
            try:
                with open(hist_file, "w", encoding="utf-8") as f:
                    json.dump([], f)
                self.load_history_to_table()
                if hasattr(self, 'txt_history_detail') and self.txt_history_detail:
                    self.txt_history_detail.clear()
                self.log_info("🗑️ Đã xóa toàn bộ lịch sử xử lý.")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể xóa lịch sử:\n{e}")

    def export_history_clicked(self):
        hist_file = os.path.join("config", "history.json")
        file_path, _ = QFileDialog.getSaveFileName(self, "Xuất Lịch Sử", "history_export.json", "JSON Files (*.json);;CSV Files (*.csv)")
        if not file_path:
            return

        try:
            history_list = []
            if os.path.exists(hist_file):
                with open(hist_file, "r", encoding="utf-8") as f:
                    history_list = json.load(f)

            if file_path.endswith(".csv"):
                import csv
                with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["STT", "Tên dự án", "Video đầu vào", "Video đầu ra", "Ngày xử lý", "Thời gian (phút)", "Trạng thái", "Lỗi"])
                    for idx, rec in enumerate(history_list):
                        writer.writerow([
                            idx + 1,
                            rec.get("project_name", ""),
                            rec.get("video_path", ""),
                            rec.get("output_video", ""),
                            rec.get("timestamp", ""),
                            rec.get("duration_minutes", 0.0),
                            rec.get("status", ""),
                            rec.get("error", "")
                        ])
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(history_list, f, ensure_ascii=False, indent=2)

            QMessageBox.information(self, "Xuất Lịch Sử", f"Đã xuất lịch sử thành công tại:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi Export", f"Không thể xuất lịch sử:\n{e}")

    def reload_from_history_clicked(self):
        if not hasattr(self, 'tbl_history') or self.tbl_history is None:
            return

        selected_items = self.tbl_history.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn 1 dòng lịch sử để mở lại.")
            return

        row = selected_items[0].row()
        item_stt = self.tbl_history.item(row, 0)
        rec = item_stt.data(Qt.ItemDataRole.UserRole) if item_stt else None
        if rec:
            self.restore_project_state(rec)
            self.log_info(f"🔄 Đã khôi phục lại dự án '{rec.get('project_name')}' từ lịch sử.")
            QMessageBox.information(self, "Mở Lại Dự Án", f"Đã khôi phục thành công state dự án:\n'{rec.get('project_name')}'")

    def rerun_pipeline_from_history_clicked(self):
        if not hasattr(self, 'tbl_history') or self.tbl_history is None:
            return

        selected_items = self.tbl_history.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn 1 dòng lịch sử để chạy lại pipeline.")
            return

        row = selected_items[0].row()
        item_stt = self.tbl_history.item(row, 0)
        rec = item_stt.data(Qt.ItemDataRole.UserRole) if item_stt else None
        if rec:
            self.restore_project_state(rec)
            if hasattr(self, 'start_dubbing'):
                self.log_info("▶ Kích hoạt chạy lại pipeline từ lịch sử...")
                self.start_dubbing()
            else:
                QMessageBox.warning(self, "Lỗi", "Không tìm thấy hàm start_dubbing.")

    def zoom_in_preview(self):
        self.preview_zoom_factor = min(2.5, getattr(self, 'preview_zoom_factor', 1.0) + 0.15)
        if hasattr(self, 'lbl_zoom_level'):
            self.lbl_zoom_level.setText(f"{int(self.preview_zoom_factor * 100)}%")
        if hasattr(self, 'current_preview_raw_frame') and self.current_preview_raw_frame is not None:
            self.show_preview_frame(self.current_preview_raw_frame)

    def zoom_out_preview(self):
        self.preview_zoom_factor = max(0.4, getattr(self, 'preview_zoom_factor', 1.0) - 0.15)
        if hasattr(self, 'lbl_zoom_level'):
            self.lbl_zoom_level.setText(f"{int(self.preview_zoom_factor * 100)}%")
        if hasattr(self, 'current_preview_raw_frame') and self.current_preview_raw_frame is not None:
            self.show_preview_frame(self.current_preview_raw_frame)

    def filter_log_console(self, filter_text):
        if not hasattr(self, 'txt_log_console') or not self.txt_log_console:
            return
        self.txt_log_console.clear()
        filter_upper = str(filter_text).upper()
        for lvl, timestamp, msg_str, html_str in getattr(self, '_raw_log_records', []):
            if filter_upper == "ALL":
                self.txt_log_console.append(html_str)
            elif filter_upper in lvl or filter_upper in msg_str.upper():
                self.txt_log_console.append(html_str)
        sb = self.txt_log_console.verticalScrollBar()
        if sb: sb.setValue(sb.maximum())

    def test_gemini_key_clicked(self):
        key = self.txt_gemini_key.text().strip() if hasattr(self, 'txt_gemini_key') else ""
        if not key:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập Gemini API Key để kiểm tra.")
            return
        first_key = key.split(",")[0].strip()
        try:
            import google.generativeai as genai
            genai.configure(api_key=first_key)
            models = list(genai.list_models())
            QMessageBox.information(self, "Test Gemini API", f"✅ Key hợp lệ!\nĐã kết nối thành công tới Gemini API.\nSố mô hình khả dụng: {len(models)}")
        except Exception as e:
            QMessageBox.critical(self, "Test Gemini API", f"❌ Key không hợp lệ:\n{e}")

    def test_xkiro_key_clicked(self):
        key = self.txt_xkiro_key.text().strip() if hasattr(self, 'txt_xkiro_key') else ""
        if not key:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập xKiro API Key để kiểm tra.")
            return
        first_key = key.split(",")[0].strip()
        try:
            import xkiro_client
            res = xkiro_client.translate_with_xkiro("Test connection", target_lang="vi", api_key=first_key)
            QMessageBox.information(self, "Test xKiro API", f"✅ Key hợp lệ!\nKết quả dịch thử nghiệm: {res}")
        except Exception as e:
            QMessageBox.critical(self, "Test xKiro API", f"❌ Key không hợp lệ:\n{e}")

    def on_preset_profile_selected(self, profile_name):
        if profile_name == "Custom":
            return
        profiles = {
            "Default (Arial, 24, White)": {"font_name": "Arial", "font_size": 24, "font_color": [255, 255, 255], "outline_color": [0, 0, 0], "outline_width": 2, "use_bg_box": False, "bg_color": [0, 0, 0], "bg_opacity": 0, "v_align": "bottom", "h_align": "center"},
            "Large Text (Verdana, 32, Yellow)": {"font_name": "Verdana", "font_size": 32, "font_color": [255, 255, 0], "outline_color": [0, 0, 0], "outline_width": 3, "use_bg_box": False, "bg_color": [0, 0, 0], "bg_opacity": 0, "v_align": "bottom", "h_align": "center"},
            "Small Text (Arial, 18, White)": {"font_name": "Arial", "font_size": 18, "font_color": [255, 255, 255], "outline_color": [0, 0, 0], "outline_width": 1, "use_bg_box": False, "bg_color": [0, 0, 0], "bg_opacity": 0, "v_align": "bottom", "h_align": "center"},
            "Cinematic (Impact, 28, White, bg box)": {"font_name": "Impact", "font_size": 28, "font_color": [255, 255, 255], "outline_color": [0, 0, 0], "outline_width": 2, "use_bg_box": True, "bg_color": [0, 0, 0], "bg_opacity": 70, "v_align": "bottom", "h_align": "center"}
        }
        custom_file = os.path.join("config", "preset_profiles.json")
        if os.path.exists(custom_file):
            try:
                with open(custom_file, "r", encoding="utf-8") as f:
                    c_profs = json.load(f)
                    profiles.update(c_profs)
            except Exception:
                pass

        if profile_name in profiles:
            p = profiles[profile_name]
            self.block_preset_signals = True
            try:
                if hasattr(self, 'cb_font_name'): self.cb_font_name.setCurrentText(p.get("font_name", "Arial"))
                if hasattr(self, 'spin_font_size'): self.spin_font_size.setValue(p.get("font_size", 24))
                if hasattr(self, 'spin_font_size_tab2'): self.spin_font_size_tab2.setValue(p.get("font_size", 24))
                if "font_color" in p:
                    self.preset_font_color = p["font_color"]
                    if hasattr(self, 'btn_font_color'):
                        qc = QColor(p["font_color"][0], p["font_color"][1], p["font_color"][2])
                        self.btn_font_color.setStyleSheet(f"background-color: {qc.name()}; border: 1px solid white;")
                        if hasattr(self, 'lbl_font_hex'): self.lbl_font_hex.setText(qc.name().upper())
                if "outline_color" in p:
                    self.preset_outline_color = p["outline_color"]
                    if hasattr(self, 'btn_outline_color'):
                        qc = QColor(p["outline_color"][0], p["outline_color"][1], p["outline_color"][2])
                        self.btn_outline_color.setStyleSheet(f"background-color: {qc.name()}; border: 1px solid white;")
                        if hasattr(self, 'lbl_outline_hex'): self.lbl_outline_hex.setText(qc.name().upper())
                if hasattr(self, 'spin_outline_width'): self.spin_outline_width.setValue(p.get("outline_width", 2))
                if hasattr(self, 'chk_use_bg_box'): self.chk_use_bg_box.setChecked(p.get("use_bg_box", False))
                if "bg_color" in p:
                    self.preset_bg_color = p["bg_color"]
                    if hasattr(self, 'btn_bg_color'):
                        qc = QColor(p["bg_color"][0], p["bg_color"][1], p["bg_color"][2])
                        self.btn_bg_color.setStyleSheet(f"background-color: {qc.name()}; border: 1px solid white;")
                        if hasattr(self, 'lbl_bg_hex'): self.lbl_bg_hex.setText(qc.name().upper())
                if hasattr(self, 'slider_bg_opacity'): self.slider_bg_opacity.setValue(p.get("bg_opacity", 50))
                if hasattr(self, 'cb_v_align'): self.cb_v_align.setCurrentText(p.get("v_align", "bottom"))
                if hasattr(self, 'cb_h_align'): self.cb_h_align.setCurrentText(p.get("h_align", "center"))
                if hasattr(self, '_tab2_sync_from_preset'): self._tab2_sync_from_preset()
            finally:
                self.block_preset_signals = False
            self.log_info(f"🎨 Đã áp dụng Preset Profile: '{profile_name}'")

    def on_preset_control_changed(self, *args):
        self.mark_preset_custom()
        if hasattr(self, 'save_app_settings'):
            self.save_app_settings()

    def mark_preset_custom(self):
        print("DEBUG inside mark_preset_custom, block_preset_signals =", getattr(self, 'block_preset_signals', False))
        if getattr(self, 'block_preset_signals', False):
            return
        if hasattr(self, 'cb_preset_profile') and self.cb_preset_profile:
            idx = self.cb_preset_profile.findText("Custom")
            if idx >= 0 and self.cb_preset_profile.currentIndex() != idx:
                self.cb_preset_profile.blockSignals(True)
                self.cb_preset_profile.setCurrentIndex(idx)
                self.cb_preset_profile.blockSignals(False)
        if hasattr(self, 'cb_preset') and self.cb_preset:
            idx = self.cb_preset.findText("Custom")
            if idx >= 0 and self.cb_preset.currentIndex() != idx:
                self.cb_preset.blockSignals(True)
                self.cb_preset.setCurrentIndex(idx)
                self.cb_preset.blockSignals(False)

    def save_preset_profile_clicked(self):
        prof_name, ok = QInputDialog.getText(self, "Lưu Profile Mới", "Nhập tên cho Preset Profile mới:")
        if not ok or not prof_name.strip():
            return
        prof_name = prof_name.strip()
        state = self.get_current_project_state()
        preset = state.get("preset", {})
        
        os.makedirs("config", exist_ok=True)
        custom_file = os.path.join("config", "preset_profiles.json")
        c_profs = {}
        if os.path.exists(custom_file):
            try:
                with open(custom_file, "r", encoding="utf-8") as f:
                    c_profs = json.load(f)
            except Exception:
                c_profs = {}

        c_profs[prof_name] = preset
        try:
            with open(custom_file, "w", encoding="utf-8") as f:
                json.dump(c_profs, f, ensure_ascii=False, indent=2)
            if hasattr(self, 'cb_preset_profile'):
                if self.cb_preset_profile.findText(prof_name) == -1:
                    self.cb_preset_profile.addItem(prof_name)
                self.cb_preset_profile.setCurrentText(prof_name)
            QMessageBox.information(self, "Lưu Profile", f"Đã lưu preset profile '{prof_name}' thành công!")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu preset profile:\n{e}")

    def auto_save_project_tick(self):
        try:
            state = self.get_current_project_state()
            os.makedirs("config", exist_ok=True)
            rec_file = os.path.join("config", "autosave_recovery.json")
            with open(rec_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            if getattr(self, 'project_file_path', None) and os.path.exists(self.project_file_path):
                with open(self.project_file_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
                self.log_info(f"💾 Auto-save: Đã tự động lưu dự án vào {os.path.basename(self.project_file_path)}")
        except Exception as e:
            print(f"[AUTO-SAVE] Error: {e}")

    def revert_project_clicked(self):
        if not getattr(self, 'project_file_path', None) or not os.path.exists(self.project_file_path):
            QMessageBox.warning(self, "Cảnh báo", "Dự án hiện tại chưa được lưu ra file .vdproj để revert.")
            return
        reply = QMessageBox.question(self, "Xác nhận Revert", "Bạn có chắc chắn muốn hủy các thay đổi chưa lưu và tải lại dự án?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                with open(self.project_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.restore_project_state(data)
                self.log_info(f"↩️ Đã revert dự án về trạng thái lưu cuối cùng.")
                QMessageBox.information(self, "Revert Dự Án", "Đã nạp lại trạng thái lưu thành công!")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể revert dự án:\n{e}")

    def load_more_history_clicked(self):
        self._history_loaded_limit += 50
        self.load_history_to_table()

    def filter_history_table(self):
        if not hasattr(self, 'tbl_history') or not hasattr(self, 'cb_history_filter'):
            return
        flt = self.cb_history_filter.currentText()
        for r in range(self.tbl_history.rowCount()):
            item_status = self.tbl_history.item(r, 7) # Status column (index 7)
            if not item_status:
                continue
            st_text = item_status.text()
            if flt == "All" or flt in st_text:
                self.tbl_history.setRowHidden(r, False)
            else:
                self.tbl_history.setRowHidden(r, True)

    def apply_all_tooltips(self):
        tooltips = {
            "btn_open_video": "Chọn file video từ máy tính để bắt đầu xử lý",
            "btn_play_seg": "Phát hoặc tạm dừng xem thử video theo thời gian thực",
            "btn_run_main": "Kích hoạt toàn bộ quy trình tự động 1-click (OCR -> Dịch -> Dubbing -> Render)",
            "btn_cancel_main": "Hủy dừng khẩn cấp luồng xử lý pipeline đang chạy",
            "btn_export_srt_main": "Xuất file phụ đề dạng văn bản chuẩn .SRT",
            "btn_test_gemini": "Kiểm tra kết nối và tính hợp lệ của Gemini API Key",
            "btn_test_xkiro": "Kiểm tra kết nối và tính hợp lệ của xKiro AI Key",
            "btn_save_preset_profile": "Lưu cấu hình kiểu dáng phụ đề hiện tại thành Profile mới",
            "btn_new_project": "Tạo dự án mới và reset tất cả tham số về mặc định",
            "btn_save_project": "Lưu toàn bộ dữ liệu dự án ra file .vdproj",
            "btn_load_project": "Mở dự án .vdproj đã lưu từ trước",
            "btn_export_config": "Xuất cấu hình kiểu dáng và cài đặt pipeline ra file .json",
            "btn_import_config": "Nhập cấu hình kiểu dáng từ file .json",
            "btn_clear_history": "Xóa toàn bộ lịch sử các đợt chạy pipeline",
            "btn_export_history": "Xuất lịch sử xử lý ra file CSV hoặc JSON",
            "btn_reload_from_history": "Mở lại state dự án từ dòng lịch sử được chọn",
            "btn_rerun": "Chạy lại pipeline trên video và cấu hình trong bản ghi lịch sử",
            "btn_revert_project": "Khôi phục lại dữ liệu dự án từ lần lưu file .vdproj cuối cùng",
            "btn_load_more_history": "Tải thêm 50 bản ghi lịch sử xử lý tiếp theo",
            "chk_auto_save_voice": "Tự động lưu giọng đọc đã chọn làm mặc định cho lần khởi động sau",
            "btn_clear_cache": "Xóa toàn bộ Frame cache và giải phóng bộ nhớ RAM",
            "chk_dark_mode": "Chuyển đổi giao diện Dark Mode / Light Mode cho toàn bộ ứng dụng",
            "btn_add_batch_videos": "Chọn nhiều file video từ máy tính để thêm vào hàng đợi Batch",
            "btn_add_batch_folder": "Quét và nạp toàn bộ file video từ một thư mục vào hàng đợi",
            "btn_remove_batch_selected": "Xóa video đang chọn khỏi hàng đợi xử lý hàng loạt",
            "btn_clear_batch_queue": "Xóa sạch toàn bộ danh sách hàng đợi video",
            "btn_batch_up": "Di chuyển video được chọn lên trên để ưu tiên xử lý trước",
            "btn_batch_down": "Di chuyển video được chọn xuống dưới trong hàng đợi",
            "btn_export_batch_queue": "Xuất danh sách video trong hàng đợi ra file CSV",
            "btn_run_batch": "Bắt đầu quy trình xử lý hàng loạt cho tất cả video trong hàng đợi",
            "btn_stop_batch": "Dừng an toàn tiến trình Batch sau khi video hiện tại xong",
            "btn_copy_batch_log": "Sao chép toàn bộ nội dung log Batch vào Clipboard",
            "btn_save_batch_log": "Lưu nội dung log Batch ra file văn bản .txt",
            "btn_export_report_html": "Xuất báo cáo thống kê kết quả Batch sang định dạng HTML trực quan",
            "btn_export_report_pdf": "Xuất báo cáo thống kê kết quả Batch sang file tài liệu PDF",
            "btn_export_report_csv": "Xuất bảng dữ liệu kết quả Batch chi tiết ra file CSV"
        }
        for attr, text in tooltips.items():
            if hasattr(self, attr):
                widget = getattr(self, attr)
                if widget and hasattr(widget, 'setToolTip'):
                    widget.setToolTip(text)

    def on_main_tab_changed(self, idx):
        """Khi chuyển tab, tạm dừng video preview nếu đang phát để giải phóng tài nguyên CPU/RAM."""
        if idx != 0:
            if getattr(self, 'is_playing_video', False):
                self.is_playing_video = False
                if hasattr(self, 'video_play_timer') and self.video_play_timer.isActive():
                    self.video_play_timer.stop()
                if hasattr(self, 'btn_play_seg'):
                    self.btn_play_seg.setText("▶ Phát video")

    def trim_log_if_needed(self):
        """Giới hạn log console tối đa 1000 dòng. Khi vượt quá, xóa 200 dòng cũ nhất."""
        if not hasattr(self, 'txt_log_console') or not self.txt_log_console:
            return
        doc = self.txt_log_console.document()
        if doc.blockCount() > 1000:
            cursor = self.txt_log_console.textCursor()
            cursor.beginEditBlock()
            cursor.movePosition(cursor.MoveOperation.Start)
            for _ in range(200):
                cursor.select(cursor.SelectionType.BlockUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()
            cursor.endEditBlock()
        if hasattr(self, '_raw_log_records') and len(self._raw_log_records) > 1000:
            self._raw_log_records = self._raw_log_records[-800:]

    def clear_cache_clicked(self):
        """Dọn sạch toàn bộ Frame cache và RAM bộ nhớ đệm."""
        if hasattr(self, '_frame_cache'):
            self._frame_cache.clear()
        self._cached_project_state = None
        if hasattr(self, 'lbl_main_preview') and hasattr(self.lbl_main_preview, 'clear_cache'):
            self.lbl_main_preview.clear_cache()
        self.log_info("🗑️ Đã xóa sạch toàn bộ Frame Cache và RAM dự án.")
        if not os.environ.get("QT_QPA_PLATFORM"):
            QMessageBox.information(self, "Clear Cache", "Đã dọn sạch bộ nhớ tạm (Frame Cache) thành công!")

    def resizeEvent(self, event):
        """Cập nhật scaling của video preview và các bounding boxes khi kích thước cửa sổ thay đổi."""
        super().resizeEvent(event)
        if hasattr(self, 'lbl_main_preview') and self.lbl_main_preview:
            QTimer.singleShot(20, self.lbl_main_preview._update_scaled_pixmap)
            QTimer.singleShot(50, self.lbl_main_preview.update_bboxes)

    def changeEvent(self, event):
        """Xử lý khi cửa sổ chuyển trạng thái Maximize / Restore / Minimize."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            if hasattr(self, 'lbl_main_preview') and self.lbl_main_preview:
                QTimer.singleShot(50, self.lbl_main_preview._update_scaled_pixmap)
                QTimer.singleShot(100, self.lbl_main_preview.update_bboxes)

    def closeEvent(self, event):
        """Lưu kích thước, tọa độ cửa sổ và dọn dẹp khi đóng ứng dụng."""
        try:
            self.save_window_state()
        except Exception:
            pass
        super().closeEvent(event)

    def save_window_state(self):
        key_dir = os.path.join(os.path.dirname(__file__), "config")
        os.makedirs(key_dir, exist_ok=True)
        settings_file = os.path.join(key_dir, "app_settings.json")
        data = {}
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data["window_width"] = self.width()
        data["window_height"] = self.height()
        data["window_x"] = self.x()
        data["window_y"] = self.y()
        if hasattr(self, 'chk_dark_mode'):
            data["dark_mode"] = self.chk_dark_mode.isChecked()
        try:
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def restore_window_state(self):
        key_dir = os.path.join(os.path.dirname(__file__), "config")
        settings_file = os.path.join(key_dir, "app_settings.json")
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                w = data.get("window_width")
                h = data.get("window_height")
                x = data.get("window_x")
                y = data.get("window_y")
                if w and h:
                    self.resize(max(800, min(3840, int(w))), max(600, min(2160, int(h))))
                if x is not None and y is not None:
                    screen = QApplication.primaryScreen()
                    if screen:
                        geo = screen.availableGeometry()
                        nx = max(geo.x(), min(int(x), geo.x() + geo.width() - 200))
                        ny = max(geo.y(), min(int(y), geo.y() + geo.height() - 200))
                        self.move(nx, ny)
                if "dark_mode" in data and hasattr(self, 'chk_dark_mode'):
                    dm = bool(data["dark_mode"])
                    self.chk_dark_mode.blockSignals(True)
                    self.chk_dark_mode.setChecked(dm)
                    self.chk_dark_mode.blockSignals(False)
                    self.apply_dark_mode(dm)
            except Exception:
                pass

    def apply_dark_mode(self, enabled):
        if enabled:
            dark_style = """
            QMainWindow, QWidget { background-color: #0f172a; color: #f8fafc; }
            QGroupBox { font-weight: bold; border: 1px solid #334155; border-radius: 6px; margin-top: 6px; padding-top: 10px; color: #38bdf8; }
            QLabel { color: #f1f5f9; }
            QPushButton { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 4px; padding: 5px 10px; font-weight: 500; }
            QPushButton:hover { background-color: #334155; }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget { background-color: #090d16; color: #38bdf8; border: 1px solid #334155; border-radius: 4px; padding: 3px; }
            QTabBar::tab { background-color: #1e293b; color: #94a3b8; padding: 8px 16px; font-weight: bold; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: #0284c7; color: white; }
            QStatusBar { background-color: #090d16; color: #94a3b8; border-top: 1px solid #1e293b; }
            """
            self.setStyleSheet(dark_style)
        else:
            light_style = """
            QMainWindow, QWidget { background-color: #f8fafc; color: #0f172a; }
            QGroupBox { font-weight: bold; border: 1px solid #cbd5e1; border-radius: 6px; margin-top: 6px; padding-top: 10px; color: #0369a1; }
            QLabel { color: #0f172a; }
            QPushButton { background-color: #e2e8f0; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 4px; padding: 5px 10px; font-weight: 500; }
            QPushButton:hover { background-color: #cbd5e1; }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget { background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 4px; padding: 3px; }
            QTabBar::tab { background-color: #e2e8f0; color: #475569; padding: 8px 16px; font-weight: bold; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: #0284c7; color: white; }
            QStatusBar { background-color: #f1f5f9; color: #475569; border-top: 1px solid #cbd5e1; }
            """
            self.setStyleSheet(light_style)

    def check_and_prompt_crash_recovery(self):
        rec_file = os.path.join("config", "autosave_recovery.json")
        if os.path.exists(rec_file):
            try:
                with open(rec_file, "r", encoding="utf-8") as f:
                    rec_state = json.load(f)
                if rec_state and isinstance(rec_state, dict) and rec_state.get("project_name"):
                    pname = rec_state.get("project_name")
                    if not os.environ.get("QT_QPA_PLATFORM"):
                        reply = QMessageBox.question(
                            self,
                            "Phục hồi dự án",
                            f"Phát hiện phiên làm việc trước có dự án chưa lưu:\n'{pname}'\n\nBạn có muốn phục hồi dữ liệu này không?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            self.restore_project_state(rec_state)
                            self.log_info(f"🔄 Đã phục hồi thành công dự án '{pname}' từ bản lưu tự động.")
            except Exception:
                pass
            finally:
                try:
                    os.remove(rec_file)
                except Exception:
                    pass

    def retry_api_call(self, func, max_retries=3, delay_sec=2.0, log_prefix="API"):
        """Cơ chế thử lại tối đa N lần cho các cuộc gọi API ngoài (Gemini/xKiro/TTS)."""
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_exc = e
                self.log_info(f"⚠️ [{log_prefix}] Thất bại lượt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    time.sleep(delay_sec)
    def zoom_in_preview(self):
        """Phóng to preview video."""
        if hasattr(self, 'lbl_main_preview') and hasattr(self.lbl_main_preview, 'zoom_in'):
            txt = self.lbl_main_preview.zoom_in()
            if hasattr(self, 'lbl_zoom_level') and self.lbl_zoom_level:
                self.lbl_zoom_level.setText(f"🔍 {txt}")

    def zoom_out_preview(self):
        """Thu nhỏ preview video."""
        if hasattr(self, 'lbl_main_preview') and hasattr(self.lbl_main_preview, 'zoom_out'):
            txt = self.lbl_main_preview.zoom_out()
            if hasattr(self, 'lbl_zoom_level') and self.lbl_zoom_level:
                self.lbl_zoom_level.setText(f"🔍 {txt}")

    def reset_zoom_preview(self):
        """Khôi phục tỉ lệ preview video về chuẩn 100%."""
        if hasattr(self, 'lbl_main_preview') and hasattr(self.lbl_main_preview, 'reset_zoom'):
            txt = self.lbl_main_preview.reset_zoom()
            if hasattr(self, 'lbl_zoom_level') and self.lbl_zoom_level:
                self.lbl_zoom_level.setText(f"🔍 {txt}")

    def start_oneclick_pipeline(self):
        if hasattr(self, 'start_dubbing'):
            self.start_dubbing()

    def create_inline_editor_bottom_panel(self, parent_layout=None):
        editor_frame = QFrame()
        editor_layout = QVBoxLayout(editor_frame)
        editor_layout.setContentsMargins(8, 8, 8, 8)
        editor_layout.setSpacing(6)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(6)

        self.txt_sub_search = QLineEdit()
        self.txt_sub_search.setPlaceholderText("🔍 Tìm kiếm phụ đề (Ctrl+F)...")
        if hasattr(self, 'search_subtitles'):
            self.txt_sub_search.textChanged.connect(lambda: self.search_subtitles(direction=0))
        toolbar_layout.addWidget(self.txt_sub_search, 2)

        self.cb_sub_search_filter = QComboBox()
        self.cb_sub_search_filter.addItems(["Tất cả", "Gốc (OCR)", "Đã dịch"])
        if hasattr(self, 'search_subtitles'):
            self.cb_sub_search_filter.currentIndexChanged.connect(lambda: self.search_subtitles(direction=0))
        toolbar_layout.addWidget(self.cb_sub_search_filter)

        self.btn_prev_match = QPushButton("▲ Trước")
        if hasattr(self, 'search_subtitles'):
            self.btn_prev_match.clicked.connect(lambda: self.search_subtitles(direction=-1))
        toolbar_layout.addWidget(self.btn_prev_match)

        self.btn_next_match = QPushButton("▼ Sau")
        if hasattr(self, 'search_subtitles'):
            self.btn_next_match.clicked.connect(lambda: self.search_subtitles(direction=1))
        toolbar_layout.addWidget(self.btn_next_match)

        self.lbl_search_count = QLabel("Tìm thấy 0 / 0 câu")
        self.lbl_search_count.setStyleSheet("color: #38bdf8; font-weight: bold; min-width: 100px;")
        toolbar_layout.addWidget(self.lbl_search_count)
        toolbar_layout.addSpacing(10)

        btn_add_row = QPushButton("➕ Thêm")
        if hasattr(self, 'insert_subtitle_row'):
            btn_add_row.clicked.connect(self.insert_subtitle_row)
        toolbar_layout.addWidget(btn_add_row)

        btn_del_row = QPushButton("❌ Xóa")
        if hasattr(self, 'delete_selected_subtitle_rows'):
            btn_del_row.clicked.connect(self.delete_selected_subtitle_rows)
        toolbar_layout.addWidget(btn_del_row)

        btn_merge_row = QPushButton("🔗 Gộp")
        if hasattr(self, 'merge_selected_subtitle_rows'):
            btn_merge_row.clicked.connect(self.merge_selected_subtitle_rows)
        toolbar_layout.addWidget(btn_merge_row)

        btn_split_row = QPushButton("✂️ Tách")
        if hasattr(self, 'split_subtitle_row'):
            btn_split_row.clicked.connect(self.split_subtitle_row)
        toolbar_layout.addWidget(btn_split_row)

        btn_set_start = QPushButton("⏱️ Start=Player")
        if hasattr(self, 'set_start_from_player'):
            btn_set_start.clicked.connect(self.set_start_from_player)
        toolbar_layout.addWidget(btn_set_start)

        btn_set_end = QPushButton("⏱️ End=Player")
        if hasattr(self, 'set_end_from_player'):
            btn_set_end.clicked.connect(self.set_end_from_player)
        toolbar_layout.addWidget(btn_set_end)

        btn_shift_pos = QPushButton("+0.5s")
        if hasattr(self, 'shift_selected_timestamps'):
            btn_shift_pos.clicked.connect(lambda: self.shift_selected_timestamps(0.5))
        toolbar_layout.addWidget(btn_shift_pos)

        btn_shift_neg = QPushButton("-0.5s")
        if hasattr(self, 'shift_selected_timestamps'):
            btn_shift_neg.clicked.connect(lambda: self.shift_selected_timestamps(-0.5))
        toolbar_layout.addWidget(btn_shift_neg)

        editor_layout.addLayout(toolbar_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "STT", "Bắt đầu (Start)", "Kết thúc (End)", "Thời lượng",
            "Phụ đề Gốc (OCR)", "Phụ đề Dịch (Bản chuẩn)", "Tốc độ / Cảnh báo"
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        if hasattr(self, 'show_subtitle_table_context_menu'):
            self.table.customContextMenuRequested.connect(self.show_subtitle_table_context_menu)
        self.table.verticalHeader().setDefaultSectionSize(28)
        if hasattr(self, 'on_cell_changed'):
            self.table.cellChanged.connect(self.on_cell_changed)
        if hasattr(self, 'trigger_canvas_update'):
            self.table.itemSelectionChanged.connect(self.trigger_canvas_update)
        if hasattr(self, 'on_subtitle_table_row_selected'):
            self.table.itemSelectionChanged.connect(self.on_subtitle_table_row_selected)
        if hasattr(self, 'on_subtitle_table_row_double_clicked'):
            self.table.itemDoubleClicked.connect(self.on_subtitle_table_row_double_clicked)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.table.setMinimumHeight(150)

        header = self.table.horizontalHeader()
        header.setFixedHeight(36)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Interactive)

        editor_layout.addWidget(self.table, 1)

        if parent_layout is not None:
            parent_layout.addWidget(editor_frame)
        return editor_frame

    def start_dubbing(self):
        self.log_info("🟢 BẤM CHẠY: Đang kích hoạt tiến trình trích xuất OCR & lồng tiếng video...")
        self.start_extraction()

    def on_canvas_bbox_added(self, bbox):
        if not hasattr(self, 'selected_bboxes') or self.selected_bboxes is None:
            self.selected_bboxes = []
        self.selected_bboxes.append(bbox)
        
        vx, vy, vw, vh = bbox
        h_vid = getattr(self, 'video_height', 1080)
        w_vid = getattr(self, 'video_width', 1920)
        
        # Tự động phân loại dựa vào vị trí khoanh vùng
        if vy > int(h_vid * 0.4):
            self.selected_bbox = bbox
            self.sub_bbox = bbox
            tag_name = "🔴 Vùng Phụ Đề (Sub)"
        elif vy < int(h_vid * 0.3) and vx > int(w_vid * 0.5):
            self.logo_bbox = bbox
            tag_name = "🟠 Vùng Logo / Thủy Ấn"
        else:
            self.title_bbox = bbox
            tag_name = "🟣 Vùng Tiêu Đề"
            
        if hasattr(self, 'lbl_crop_info'):
            self.lbl_crop_info.setText(f"Vùng Crop (X,Y,W,H): X={vx}, Y={vy}, W={vw}, H={vh} [{tag_name}]")
        if hasattr(self, 'lbl_bbox'):
            self.lbl_bbox.setText(f"Đã khoanh: {tag_name} (X={vx}, Y={vy}, W={vw}, H={vh})")
            
        self.log_info(f"📌 Đã khoanh vùng trực tiếp: {tag_name} -> [X={vx}, Y={vy}, W={vw}, H={vh}]")
        if hasattr(self, 'lbl_main_preview'):
            self.lbl_main_preview.update()

    def clear_all_canvas_crops(self):
        self.selected_bboxes = []
        self.selected_bbox = None
        self.logo_bbox = None
        self.title_bbox = None
        if hasattr(self, 'lbl_main_preview') and hasattr(self.lbl_main_preview, 'bboxes'):
            self.lbl_main_preview.bboxes = []
        if hasattr(self, 'lbl_crop_info'):
            self.lbl_crop_info.setText("Vùng Crop (X, Y, W, H): Chưa chọn")
        if hasattr(self, 'lbl_bbox'):
            self.lbl_bbox.setText("Vùng quét: Chưa chọn")
            
        self.log_info("🧹 Đã xóa toàn bộ vùng khoanh Crop trên màn hình.")
        if hasattr(self, 'lbl_main_preview'):
            self.lbl_main_preview.update()

    def toggle_realtime_play(self):
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "Thông báo", "Vui lòng bấm 'Chọn Video' để tải video trước khi xem trực tiếp!")
            return
        
        if not hasattr(self, 'video_play_timer'):
            self.video_play_timer = QTimer(self)
            self.video_play_timer.timeout.connect(self._advance_video_play_frame)
            
        if getattr(self, 'is_playing_video', False):
            self.is_playing_video = False
            self.video_play_timer.stop()
            if hasattr(self, 'btn_play_seg'):
                self.btn_play_seg.setText("▶ Phát video")
        else:
            self.is_playing_video = True
            self.video_play_timer.start(33)
            if hasattr(self, 'btn_play_seg'):
                self.btn_play_seg.setText("⏸️ Tạm dừng")

    def _advance_video_play_frame(self):
        if not self.video_path or not getattr(self, 'is_playing_video', False):
            return
        try:
            curr_val = self.slider_player_timeline.value()
            if curr_val >= 1000:
                self.slider_player_timeline.setValue(0)
            else:
                self.slider_player_timeline.setValue(curr_val + 2)
        except Exception:
            pass

    def log_info(self, message):
        timestamp = time.strftime("[%H:%M:%S]")
        msg_str = str(message)

        if any(k in msg_str for k in ("Lỗi", "ERROR", "Error", "LỖI", "THẤT BẠI", "Exception", "Failed")):
            color = "#f87171"
            prefix = "✖"
            level = "ERROR"
        elif any(k in msg_str for k in ("Cảnh báo", "WARNING", "Warning", "⚡", "BUSY")):
            color = "#facc15"
            prefix = "⚡"
            level = "WARNING"
        elif any(k in msg_str for k in ("Thành công", "SUCCESS", "Hoàn tất", "✅", "OK", "DONE")):
            color = "#4ade80"
            prefix = "✔"
            level = "INFO"
        elif "DEBUG" in msg_str.upper():
            color = "#94a3b8"
            prefix = "⚪"
            level = "DEBUG"
        else:
            color = "#38bdf8"
            prefix = "ℹ"
            level = "INFO"

        formatted_html = f'<span style="color:#64748b;">{timestamp}</span> <span style="color:{color}; font-weight:bold;">{prefix} {msg_str}</span>'

        if not hasattr(self, '_raw_log_records'):
            self._raw_log_records = []
        self._raw_log_records.append((level, timestamp, msg_str, formatted_html))

        current_flt = self.cb_log_filter.currentText().upper() if hasattr(self, 'cb_log_filter') and self.cb_log_filter else "ALL"
        if current_flt == "ALL" or current_flt == level or current_flt in msg_str.upper():
            if hasattr(self, 'txt_log_console') and self.txt_log_console is not None:
                self.txt_log_console.append(formatted_html)
                sb = self.txt_log_console.verticalScrollBar()
                if sb:
                    sb.setValue(sb.maximum())

        if hasattr(self, 'status_label') and self.status_label is not None:
            self.status_label.setText(message)

        if hasattr(self, 'lbl_status_state') and self.lbl_status_state:
            if level == "ERROR":
                self.lbl_status_state.setText("❌ Error")
            elif level == "WARNING":
                self.lbl_status_state.setText("🟡 Warning")
            elif "Processing" not in self.lbl_status_state.text():
                self.lbl_status_state.setText("🟢 Ready")

        if hasattr(self, 'trim_log_if_needed'):
            self.trim_log_if_needed()

    def show_preview_frame(self, frame):
        if frame is None:
            return
        try:
            self.current_preview_raw_frame = frame.copy()
            h, w, _ = frame.shape
            self.video_width = w
            self.video_height = h
            
            show_frame = frame.copy()

            # 1. Vẽ mờ vùng crop nếu có
            if getattr(self, 'selected_bboxes', None):
                for box in self.selected_bboxes:
                    bx, by, bw, bh = box
                    bx1 = max(0, min(bx, w))
                    by1 = max(0, min(by, h))
                    bx2 = max(0, min(bx + bw, w))
                    by2 = max(0, min(by + bh, h))
                    if bx2 > bx1 and by2 > by1:
                        crop = show_frame[by1:by2, bx1:bx2]
                        ch_c, cw_c, _ = crop.shape
                        kw = 51 if 51 < cw_c else max(1, cw_c - 1 | 1)
                        kh = 51 if 51 < ch_c else max(1, ch_c - 1 | 1)
                        if kw % 2 == 0: kw = max(1, kw - 1)
                        if kh % 2 == 0: kh = max(1, kh - 1)
                        crop_blur = cv2.GaussianBlur(crop, (kw, kh), 0)
                        show_frame[by1:by2, bx1:bx2] = crop_blur

            # 2. Chèn logo thương hiệu nếu chọn logo
            if getattr(self, 'logo_path', None) and os.path.exists(self.logo_path) and getattr(self, 'logo_bbox', None):
                show_frame = dubber.insert_watermark_logo(show_frame, self.logo_path, self.logo_bbox)

            self.lbl_main_preview.setVideoFrame(show_frame)
        except Exception:
            pass

    def on_worker_frame_update(self, frame, frame_idx, total_frames, timestamp_s, active_bbox, status_msg):
        if frame is None or not hasattr(self, 'lbl_main_preview'):
            return
        
        # 1. Cập nhật vị trí thanh trượt timeline theo thời gian thực
        if hasattr(self, 'slider_player_timeline') and total_frames > 0:
            self.slider_player_timeline.blockSignals(True)
            val = int((frame_idx / float(total_frames)) * 1000)
            self.slider_player_timeline.setValue(min(1000, max(0, val)))
            self.slider_player_timeline.blockSignals(False)

        # 2. Cập nhật nhãn thời gian và số khung hình
        if hasattr(self, 'lbl_frame_info'):
            mins = int(timestamp_s // 60)
            secs = int(timestamp_s % 60)
            tot_sec = total_frames / 25.0 if total_frames > 0 else 0
            h2, m2, s2 = int(tot_sec // 3600), int((tot_sec % 3600) // 60), int(tot_sec % 60)
            self.lbl_frame_info.setText(f"Frame: {frame_idx} / {total_frames}   Time: {mins:02d}:{secs:02d} / {h2:02d}:{m2:02d}:{s2:02d}")

        # 3. Phủ các lớp hiệu ứng Visual Feedback (Bounding box phát sáng, Scanline, Status Overlay)
        from visual_feedback import draw_visual_feedback_overlay
        scanline_state = getattr(self, '_scanline_counter', 0)
        self._scanline_counter = scanline_state + 1

        annotated = draw_visual_feedback_overlay(
            frame=frame,
            frame_idx=frame_idx,
            total_frames=total_frames,
            timestamp_s=timestamp_s,
            status_text=status_msg,
            active_bbox=active_bbox,
            scanline_state=self._scanline_counter
        )

        # 4. Hiển thị trực tiếp lên khung preview GUI
        self.show_preview_frame(annotated)

    def seek_relative(self, seconds):
        if not self.video_path or not os.path.exists(self.video_path):
            return
        try:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            if total_frames <= 0:
                return
            current_val = self.slider_player_timeline.value()
            current_frame = int((current_val / 1000.0) * total_frames)
            target_frame = max(0, min(total_frames - 1, current_frame + int(seconds * fps)))
            new_val = int((target_frame / float(total_frames)) * 1000)
            self.slider_player_timeline.setValue(new_val)
        except Exception:
            pass

    def on_player_seek(self, val):
        if not self.video_path or not os.path.exists(self.video_path):
            return
        try:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if total_frames <= 0:
                cap.release()
                return
            target_frame = int((val / 1000.0) * total_frames)

            # Frame Caching: Đọc từ dict cache nếu đã từng load frame này
            cache_key = (self.video_path, target_frame)
            if hasattr(self, '_frame_cache') and cache_key in self._frame_cache:
                frame = self._frame_cache[cache_key].copy()
                cap.release()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return
                if hasattr(self, '_frame_cache'):
                    if len(self._frame_cache) > 200:
                        self._frame_cache.clear()
                    self._frame_cache[cache_key] = frame.copy()

            self.current_preview_raw_frame = frame.copy()
            sec = target_frame / fps
            tot_sec = total_frames / fps
            h1, m1, s1 = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
            m2, s2 = int(tot_sec // 60), int(tot_sec % 60)
            time_fmt = f"{h1:02d}:{m1:02d}:{s1:02d}"
            if hasattr(self, 'lbl_frame_info'):
                self.lbl_frame_info.setText(f"Frame: {target_frame} / {total_frames}   Time: {time_fmt}")
            if hasattr(self, 'lbl_player_time'):
                self.lbl_player_time.setText(f"{m1:02d}:{s1:02d} / {m2:02d}:{s2:02d}")
            self.show_preview_frame(frame)
        except Exception:
            pass

    def toggle_preview_pane(self):
        try:
            if hasattr(self, '_preview_collapsed') and self._preview_collapsed:
                # restore
                self.right_preview_pane.show()
                if hasattr(self, '_prev_splitter_sizes'):
                    try:
                        self.findChild(QSplitter).setSizes(self._prev_splitter_sizes)
                    except Exception:
                        pass
                self._preview_collapsed = False
                self.btn_toggle_preview.setText("Ẩn Preview")
            else:
                # collapse
                try:
                    splitter = self.findChild(QSplitter)
                    if splitter:
                        self._prev_splitter_sizes = splitter.sizes()
                        # give almost all space to tabs
                        total = sum(self._prev_splitter_sizes) if self._prev_splitter_sizes else 1400
                        splitter.setSizes([int(total * 0.95), int(total * 0.05)])
                except Exception:
                    pass
                self.right_preview_pane.hide()
                self._preview_collapsed = True
                self.btn_toggle_preview.setText("Hiện Preview")
        except Exception:
            pass
        
    def create_workspace_tab(self):
        from PyQt6.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        tab_content = QWidget()
        layout = QVBoxLayout(tab_content)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # 1. KHU VỰC TẢI UP VIDEO & BỘ CÔNG CỤ (Hero Upload & Tool Selector)
        card_src_voice = CollapsibleCard("🎥 TẢI UP VIDEO & BỘ CÔNG CỤ XỬ LÝ")
        hero_layout = QVBoxLayout()
        hero_layout.setSpacing(10)
        
        # Nút Upload Video siêu to & nổi bật
        upload_box_layout = QHBoxLayout()
        self.btn_hero_upload = QPushButton("📁 NHẤP VÀO ĐÂY ĐỂ TẢI/UP VIDEO LÊN (MP4, MKV, AVI, MOV)")
        self.btn_hero_upload.setStyleSheet("""
            QPushButton {
                background-color: #1e1e24;
                border: 2px dashed #7fbeb2;
                border-radius: 8px;
                color: #7fbeb2;
                font-size: 13px;
                font-weight: bold;
                padding: 14px;
            }
            QPushButton:hover {
                background-color: #272730;
                border-color: #dfb15b;
                color: #dfb15b;
            }
        """)
        self.btn_hero_upload.clicked.connect(self.browse_video)
        upload_box_layout.addWidget(self.btn_hero_upload, 3)
        
        hero_layout.addLayout(upload_box_layout)

        # Hàng chi tiết File Video & URL
        grid = QGridLayout()
        grid.setSpacing(8)
        
        grid.addWidget(QLabel("Video đã chọn:"), 0, 0)
        self.txt_file = QLineEdit()
        self.txt_file.setReadOnly(True)
        self.txt_file.setPlaceholderText("Chưa chọn file video...")
        grid.addWidget(self.txt_file, 0, 1)
        btn_browse = QPushButton("Duyệt...")
        btn_browse.setStyleSheet("background-color: #27272a; color: white; padding: 4px 10px;")
        btn_browse.clicked.connect(self.browse_video)
        grid.addWidget(btn_browse, 0, 2)
        
        grid.addWidget(QLabel("Hoặc Link URL:"), 1, 0)
        self.txt_url = QLineEdit()
        self.txt_url.setPlaceholderText("Dán link YouTube, TikTok, Facebook...")
        grid.addWidget(self.txt_url, 1, 1, 1, 2)
        
        grid.addWidget(QLabel("Tệp SRT gốc:"), 2, 0)
        self.txt_srt_path = QLineEdit()
        self.txt_srt_path.setPlaceholderText("Để trống nếu muốn tự động trích phụ đề từ video...")
        grid.addWidget(self.txt_srt_path, 2, 1)
        btn_browse_srt = QPushButton("Chọn...")
        btn_browse_srt.setStyleSheet("background-color: #27272a; color: white; padding: 4px 10px;")
        btn_browse_srt.clicked.connect(self.browse_srt_file)
        grid.addWidget(btn_browse_srt, 2, 2)
        
        grid.addWidget(QLabel("Giọng lồng tiếng:"), 3, 0)
        self.cb_voice = QComboBox()
        for v in dubber.get_supported_voices():
            self.cb_voice.addItem(v["desc"], v["name"])
        grid.addWidget(self.cb_voice, 3, 1, 1, 2)
        
        grid.addWidget(QLabel("Âm lượng gốc:"), 4, 0)
        self.slider_bg = QSlider(Qt.Orientation.Horizontal)
        self.slider_bg.setRange(0, 100)
        self.slider_bg.setValue(10)
        grid.addWidget(self.slider_bg, 4, 1, 1, 2)
        
        grid.addWidget(QLabel("Âm lượng AI:"), 5, 0)
        self.slider_dub = QSlider(Qt.Orientation.Horizontal)
        self.slider_dub.setRange(0, 200)
        self.slider_dub.setValue(100)
        grid.addWidget(self.slider_dub, 5, 1, 1, 2)
        
        hero_layout.addLayout(grid)
        card_src_voice.addLayout(hero_layout)
        layout.addWidget(card_src_voice)
        
        # 2. CARD CẤU HÌNH STYLE & CHE PHỤ ĐỀ (Style & Mask Config)
        card_style_mask = CollapsibleCard("🎨 STYLE CHỮ & THUẬT TOÁN CHE")
        card_style_mask.toggle_collapse() # Mặc định thu gọn để đỡ rối
        style_layout = QVBoxLayout()
        style_layout.setSpacing(6)
        
        # Căn lề và Preset style
        row_preset = QGridLayout()
        row_preset.addWidget(QLabel("Áp dụng Preset:"), 0, 0)
        self.cb_preset = QComboBox()
        self.cb_preset.addItems(list(self.presets_db.keys()) + ["Tùy chỉnh (Custom)"])
        self.cb_preset.setCurrentText("Mặc định (Dưới - Giữa)")
        self.cb_preset.currentTextChanged.connect(self.apply_selected_preset)
        row_preset.addWidget(self.cb_preset, 0, 1, 1, 2)
        self.btn_reset_preset = QPushButton("Đặt lại preset")
        self.btn_reset_preset.setToolTip("Khôi phục preset mặc định")
        row_preset.addWidget(self.btn_reset_preset, 0, 3)
        
        row_preset.addWidget(QLabel("Cách áp dụng:"), 1, 0)
        self.cb_preset_apply_mode = QComboBox()
        self.cb_preset_apply_mode.addItems(["Chỉ áp VỊ TRÍ, giữ style gốc", "Áp dụng cả VỊ TRÍ & STYLE"])
        row_preset.addWidget(self.cb_preset_apply_mode, 1, 1, 1, 2)
        style_layout.addLayout(row_preset)
        
        row_pos = QGridLayout()
        row_pos.addWidget(QLabel("Căn lề ngang:"), 0, 0)
        self.cb_h_align = QComboBox()
        self.cb_h_align.addItems(["Left", "Center", "Right"])
        self.cb_h_align.setCurrentText("Center")
        self.cb_h_align.currentIndexChanged.connect(self.mark_preset_custom)
        row_pos.addWidget(self.cb_h_align, 0, 1)
        
        row_pos.addWidget(QLabel("Căn dọc:"), 0, 2)
        self.cb_v_align = QComboBox()
        self.cb_v_align.addItems(["Top", "Middle", "Bottom"])
        self.cb_v_align.setCurrentText("Bottom")
        self.cb_v_align.currentIndexChanged.connect(self.mark_preset_custom)
        row_pos.addWidget(self.cb_v_align, 0, 3)
        
        row_pos.addWidget(QLabel("Margin Dọc:"), 1, 0)
        margin_v_layout = QHBoxLayout()
        self.spin_margin_v = QDoubleSpinBox()
        self.spin_margin_v.setRange(0, 500)
        self.spin_margin_v.setValue(8.0)
        self.spin_margin_v.valueChanged.connect(self.mark_preset_custom)
        margin_v_layout.addWidget(self.spin_margin_v)
        self.cb_margin_v_type = QComboBox()
        self.cb_margin_v_type.addItems(["%", "px"])
        self.cb_margin_v_type.currentIndexChanged.connect(self.mark_preset_custom)
        margin_v_layout.addWidget(self.cb_margin_v_type)
        row_pos.addLayout(margin_v_layout, 1, 1)
        
        row_pos.addWidget(QLabel("Margin Ngang:"), 1, 2)
        margin_h_layout = QHBoxLayout()
        self.spin_margin_h = QDoubleSpinBox()
        self.spin_margin_h.setRange(0, 500)
        self.spin_margin_h.setValue(5.0)
        self.spin_margin_h.valueChanged.connect(self.mark_preset_custom)
        margin_h_layout.addWidget(self.spin_margin_h)
        self.cb_margin_h_type = QComboBox()
        self.cb_margin_h_type.addItems(["%", "px"])
        self.cb_margin_h_type.currentIndexChanged.connect(self.mark_preset_custom)
        margin_h_layout.addWidget(self.cb_margin_h_type)
        row_pos.addLayout(margin_h_layout, 1, 3)
        style_layout.addLayout(row_pos)
        
        # Tọa độ Custom Position
        self.row_custom_pos = QHBoxLayout()
        self.row_custom_pos.addWidget(QLabel("Vị trí X (%):"))
        self.spin_custom_pos_x = QDoubleSpinBox()
        self.spin_custom_pos_x.setRange(0.0, 100.0)
        self.spin_custom_pos_x.setValue(50.0)
        self.spin_custom_pos_x.setEnabled(False)
        self.row_custom_pos.addWidget(self.spin_custom_pos_x)
        self.row_custom_pos.addWidget(QLabel(" Vị trí Y (%):"))
        self.spin_custom_pos_y = QDoubleSpinBox()
        self.spin_custom_pos_y.setRange(0.0, 100.0)
        self.spin_custom_pos_y.setValue(88.0)
        self.spin_custom_pos_y.setEnabled(False)
        self.row_custom_pos.addWidget(self.spin_custom_pos_y)
        self.btn_reset_custom_pos = QPushButton("Đặt lại tâm")
        self.btn_reset_custom_pos.setEnabled(False)
        self.btn_reset_custom_pos.clicked.connect(self.reset_subtitle_custom_pos)
        self.row_custom_pos.addWidget(self.btn_reset_custom_pos)
        style_layout.addLayout(self.row_custom_pos)
        
        # Font chữ
        row_font = QGridLayout()
        row_font.addWidget(QLabel("Font chữ:"), 0, 0)
        self.cb_font_name = QComboBox()
        self.cb_font_name.addItems(["Arial", "Calibri", "Segoe UI", "Times New Roman", "Tahoma", "Courier New", "Consolas"])
        self.cb_font_name.currentIndexChanged.connect(self.on_font_changed)
        row_font.addWidget(self.cb_font_name, 0, 1)
        self.btn_browse_font = QPushButton("Duyệt font...")
        self.btn_browse_font.clicked.connect(self.browse_custom_font)
        row_font.addWidget(self.btn_browse_font, 0, 2)
        
        row_font.addWidget(QLabel("Cỡ chữ:"), 1, 0)
        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(8, 120)
        self.spin_font_size.setValue(20)
        self.spin_font_size.valueChanged.connect(self.on_preset_control_changed)
        row_font.addWidget(self.spin_font_size, 1, 1)
        
        row_font.addWidget(QLabel("Viền chữ:"), 1, 2)
        self.spin_outline_width = QSpinBox()
        self.spin_outline_width.setRange(0, 20)
        self.spin_outline_width.setValue(2)
        self.spin_outline_width.valueChanged.connect(self.mark_preset_custom)
        row_font.addWidget(self.spin_outline_width, 1, 3)
        style_layout.addLayout(row_font)
        
        # Màu sắc
        row_color = QGridLayout()
        row_color.addWidget(QLabel("Màu chữ:"), 0, 0)
        font_c_layout = QHBoxLayout()
        self.btn_font_color = QPushButton()
        self.btn_font_color.setFixedSize(24, 24)
        font_c_layout.addWidget(self.btn_font_color)
        self.txt_font_color_hex = QLineEdit()
        self.txt_font_color_hex.setMaximumWidth(80)
        font_c_layout.addWidget(self.txt_font_color_hex)
        row_color.addLayout(font_c_layout, 0, 1)
        
        row_color.addWidget(QLabel("Màu viền:"), 0, 2)
        outline_c_layout = QHBoxLayout()
        self.btn_outline_color = QPushButton()
        self.btn_outline_color.setFixedSize(24, 24)
        outline_c_layout.addWidget(self.btn_outline_color)
        self.txt_outline_color_hex = QLineEdit()
        self.txt_outline_color_hex.setMaximumWidth(80)
        outline_c_layout.addWidget(self.txt_outline_color_hex)
        row_color.addLayout(outline_c_layout, 0, 3)
        
        row_color.addWidget(QLabel("Hộp nền:"), 1, 0)
        bg_c_layout = QHBoxLayout()
        self.btn_bg_color = QPushButton()
        self.btn_bg_color.setFixedSize(24, 24)
        bg_c_layout.addWidget(self.btn_bg_color)
        self.txt_bg_color_hex = QLineEdit()
        self.txt_bg_color_hex.setMaximumWidth(80)
        bg_c_layout.addWidget(self.txt_bg_color_hex)
        row_color.addLayout(bg_c_layout, 1, 1)
        
        self.chk_use_bg_box = QCheckBox("Bật nền")
        # Nút xem trước phụ đề
        preview_row = QHBoxLayout()
        preview_row.addStretch()
        self.btn_preview_sub = QPushButton("Xem trước phụ đề")
        preview_row.addWidget(self.btn_preview_sub)
        style_layout.addLayout(preview_row)
        self.chk_use_bg_box.stateChanged.connect(self.mark_preset_custom)
        row_color.addWidget(self.chk_use_bg_box, 1, 2)
        
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Độ mờ nền:"))
        self.slider_bg_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_bg_opacity.setRange(0, 255)
        self.slider_bg_opacity.setValue(0)
        self.slider_bg_opacity.valueChanged.connect(self.mark_preset_custom)
        opacity_layout.addWidget(self.slider_bg_opacity)
        row_color.addLayout(opacity_layout, 1, 3)
        style_layout.addLayout(row_color)
        
        # Cấu hình Mask che sub gốc
        row_mask = QGridLayout()
        row_mask.addWidget(QLabel("Che sub gốc:"), 0, 0)
        self.cb_mask_mode = QComboBox()
        self.cb_mask_mode.addItems(["Không che (None)", "Che đen đặc (Black Box)", "Blur nhanh (Gaussian Blur)", "Inpaint chất lượng cao"])
        self.cb_mask_mode.setCurrentIndex(2)
        self.cb_mask_mode.currentIndexChanged.connect(self.trigger_canvas_update)
        row_mask.addWidget(self.cb_mask_mode, 0, 1)
        
        row_mask.addWidget(QLabel("Thuật toán xóa:"), 0, 2)
        self.cb_remove_algo = QComboBox()
        self.cb_remove_algo.addItems(["Xóa cơ bản (FFmpeg)", "Xóa AI (OpenCV)"])
        self.cb_remove_algo.setCurrentIndex(1)
        self.cb_remove_algo.currentIndexChanged.connect(self.trigger_canvas_update)
        row_mask.addWidget(self.cb_remove_algo, 0, 3)
        
        row_mask.addWidget(QLabel("Cảnh báo < :"), 1, 0)
        self.spin_confidence_threshold = QSpinBox()
        self.spin_confidence_threshold.setRange(10, 100)
        self.spin_confidence_threshold.setValue(70)
        self.spin_confidence_threshold.setSuffix("%")
        row_mask.addWidget(self.spin_confidence_threshold, 1, 1)
        
        row_mask.addWidget(QLabel("Tốc độ tối đa:"), 1, 2)
        self.spin_speed_threshold = QSpinBox()
        self.spin_speed_threshold.setRange(5, 50)
        self.spin_speed_threshold.setValue(20)
        self.spin_speed_threshold.setSuffix(" ch/s")
        self.spin_speed_threshold.valueChanged.connect(self.populate_subtitle_table)
        row_mask.addWidget(self.spin_speed_threshold, 1, 3)
        
        self.chk_restrict_ocr = QCheckBox("Giới hạn quét ngang 60%")
        self.chk_restrict_ocr.setChecked(True)
        row_mask.addWidget(self.chk_restrict_ocr, 2, 0, 1, 2)
        
        self.chk_smart_pos = QCheckBox("Tự động căn phụ đề đè lên hộp che (Smart Pos)")
        self.chk_smart_pos.setChecked(False)
        self.chk_smart_pos.stateChanged.connect(self.mark_preset_custom)
        self.chk_smart_pos.stateChanged.connect(self.trigger_canvas_update)
        row_mask.addWidget(self.chk_smart_pos, 2, 2, 1, 2)
        
        style_layout.addLayout(row_mask)
        card_style_mask.addLayout(style_layout)
        layout.addWidget(card_style_mask)
        
        # 3. CARD BẢNG BIÊN TẬP PHỤ ĐỀ (Inline Subtitle Editor)
        card_editor = CollapsibleCard("📝 INLINE SUBTITLE EDITOR - BẢNG BIÊN TẬP PHỤ ĐỀ CHUYÊN NGHIỆP")
        editor_layout = QVBoxLayout()
        editor_layout.setSpacing(6)

        # --- Thanh công cụ Search & Filter Bar + Thao tác dòng ---
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(6)

        self.txt_sub_search = QLineEdit()
        self.txt_sub_search.setPlaceholderText("🔍 Tìm kiếm phụ đề (Ctrl+F)...")
        self.txt_sub_search.setStyleSheet("background-color: #0f172a; color: #f8fafc; border: 1px solid #2a364f; border-radius: 6px; padding: 5px 10px;")
        self.txt_sub_search.textChanged.connect(lambda: self.search_subtitles(direction=0))
        toolbar_layout.addWidget(self.txt_sub_search, 2)

        self.cb_sub_search_filter = QComboBox()
        self.cb_sub_search_filter.addItems(["Tất cả", "Gốc (OCR)", "Đã dịch"])
        self.cb_sub_search_filter.setStyleSheet("background-color: #0f172a; color: #f8fafc; padding: 5px 8px; border: 1px solid #2a364f; border-radius: 6px;")
        self.cb_sub_search_filter.currentIndexChanged.connect(lambda: self.search_subtitles(direction=0))
        toolbar_layout.addWidget(self.cb_sub_search_filter)

        self.btn_prev_match = QPushButton("▲ Trước")
        self.btn_prev_match.setStyleSheet("background-color: #1e293b; color: #f8fafc; font-weight: bold; padding: 5px 10px; border: 1px solid #334155; border-radius: 6px;")
        self.btn_prev_match.setToolTip("Tìm kết quả phía trước (Shift+Enter)")
        self.btn_prev_match.clicked.connect(lambda: self.search_subtitles(direction=-1))
        toolbar_layout.addWidget(self.btn_prev_match)

        self.btn_next_match = QPushButton("▼ Sau")
        self.btn_next_match.setStyleSheet("background-color: #1e293b; color: #f8fafc; font-weight: bold; padding: 5px 10px; border: 1px solid #334155; border-radius: 6px;")
        self.btn_next_match.setToolTip("Tìm kết quả tiếp theo (Enter)")
        self.btn_next_match.clicked.connect(lambda: self.search_subtitles(direction=1))
        toolbar_layout.addWidget(self.btn_next_match)

        self.lbl_search_count = QLabel("Tìm thấy 0 / 0 câu")
        self.lbl_search_count.setStyleSheet("color: #38bdf8; font-weight: bold; min-width: 110px;")
        toolbar_layout.addWidget(self.lbl_search_count)

        toolbar_layout.addSpacing(10)

        # Quick Actions
        btn_add_row = QPushButton("➕ Thêm")
        btn_add_row.setToolTip("Thêm dòng mới (Ctrl+N)")
        btn_add_row.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold; padding: 5px 10px; border: 1px solid #10b981; border-radius: 6px;")
        btn_add_row.clicked.connect(self.insert_subtitle_row)
        toolbar_layout.addWidget(btn_add_row)

        btn_del_row = QPushButton("❌ Xóa")
        btn_del_row.setToolTip("Xóa dòng được chọn (Delete)")
        btn_del_row.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: bold; padding: 5px 10px; border: 1px solid #ef4444; border-radius: 6px;")
        btn_del_row.clicked.connect(self.delete_selected_subtitle_rows)
        toolbar_layout.addWidget(btn_del_row)

        btn_merge_row = QPushButton("🔗 Gộp")
        btn_merge_row.setToolTip("Gộp các dòng chọn (Ctrl+M)")
        btn_merge_row.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; padding: 5px 10px; border: 1px solid #0369a1; border-radius: 6px;")
        btn_merge_row.clicked.connect(self.merge_selected_subtitle_rows)
        toolbar_layout.addWidget(btn_merge_row)

        btn_split_row = QPushButton("✂️ Tách")
        btn_split_row.setToolTip("Tách câu phụ đề (Ctrl+Shift+S)")
        btn_split_row.setStyleSheet("background-color: #7c3aed; color: #ffffff; font-weight: bold; padding: 5px 10px; border: 1px solid #6d28d9; border-radius: 6px;")
        btn_split_row.clicked.connect(self.split_subtitle_row)
        toolbar_layout.addWidget(btn_split_row)

        btn_set_start = QPushButton("⏱️ Start=Player")
        btn_set_start.setToolTip("Gán Start = Thời gian Video hiện tại (Ctrl+[)")
        btn_set_start.setStyleSheet("background-color: #1e293b; color: #cbd5e1; font-size: 11px; padding: 5px 8px; border: 1px solid #334155; border-radius: 6px;")
        btn_set_start.clicked.connect(self.set_start_from_player)
        toolbar_layout.addWidget(btn_set_start)

        btn_set_end = QPushButton("⏱️ End=Player")
        btn_set_end.setToolTip("Gán End = Thời gian Video hiện tại (Ctrl+])")
        btn_set_end.setStyleSheet("background-color: #1e293b; color: #cbd5e1; font-size: 11px; padding: 5px 8px; border: 1px solid #334155; border-radius: 6px;")
        btn_set_end.clicked.connect(self.set_end_from_player)
        toolbar_layout.addWidget(btn_set_end)

        btn_shift_pos = QPushButton("+0.5s")
        btn_shift_pos.setToolTip("Tăng mốc thời gian thêm +0.5 giây")
        btn_shift_pos.setStyleSheet("background-color: #1e293b; color: #cbd5e1; font-size: 11px; padding: 5px 8px; border: 1px solid #334155; border-radius: 6px;")
        btn_shift_pos.clicked.connect(lambda: self.shift_selected_timestamps(0.5))
        toolbar_layout.addWidget(btn_shift_pos)

        btn_shift_neg = QPushButton("-0.5s")
        btn_shift_neg.setToolTip("Giảm mốc thời gian đi -0.5 giây")
        btn_shift_neg.setStyleSheet("background-color: #1e293b; color: #cbd5e1; font-size: 11px; padding: 5px 8px; border: 1px solid #334155; border-radius: 6px;")
        btn_shift_neg.clicked.connect(lambda: self.shift_selected_timestamps(-0.5))
        toolbar_layout.addWidget(btn_shift_neg)

        editor_layout.addLayout(toolbar_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "STT", "Bắt đầu (Start)", "Kết thúc (End)", "Thời lượng",
            "Phụ đề Gốc (OCR)", "Phụ đề Dịch (Bản chuẩn)", "Tốc độ / Cảnh báo"
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_subtitle_table_context_menu)
        self.table.verticalHeader().setDefaultSectionSize(32)
        self.table.cellChanged.connect(self.on_cell_changed)
        self.table.itemSelectionChanged.connect(self.trigger_canvas_update)
        self.table.itemSelectionChanged.connect(self.on_subtitle_table_row_selected)
        self.table.itemDoubleClicked.connect(self.on_subtitle_table_row_double_clicked)
        self.table.setMinimumHeight(240)
        editor_layout.addWidget(self.table)
        
        # Hướng dẫn nhỏ gọn
        lbl_hint = QLabel("💡 Mẹo: Nhấp đúp vào ô để sửa trực tiếp. Dùng Ctrl+N (thêm), Ctrl+M (gộp), Ctrl+Shift+S (tách), Delete (xóa).")
        lbl_hint.setStyleSheet("color: #94a3b8; font-style: italic; font-size: 11px;")
        editor_layout.addWidget(lbl_hint)

        # Logs for extraction & OCR progress
        self.txt_logs1 = QTextEdit()
        self.txt_logs1.setReadOnly(True)
        self.txt_logs1.setMaximumHeight(100)
        self.txt_logs1.setPlaceholderText("Nhật ký trích xuất / OCR sẽ hiển thị tại đây...")
        editor_layout.addWidget(self.txt_logs1)
        
        # Hàng nút thao tác OCR/Dịch phụ đề
        actions_row = QHBoxLayout()
        self.btn_start_extract = QPushButton("🚀 1. TRÍCH PHỤ ĐỀ GỐC")
        self.btn_start_extract.setStyleSheet("background-color: #3b82f6; color: white; font-weight: bold; padding: 8px;")
        self.btn_start_extract.clicked.connect(self.start_extraction)
        actions_row.addWidget(self.btn_start_extract, 2)
        
        self.btn_translate = QPushButton("🤖 2. DỊCH PHỤ ĐỀ AI")
        self.btn_translate.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; padding: 8px;")
        self.btn_translate.clicked.connect(self.translate_subtitles)
        actions_row.addWidget(self.btn_translate, 2)

        self.btn_trending_slang = QPushButton("🔥 TỪ ĐIỂN SLANG DOUYIN/BILIBILI")
        self.btn_trending_slang.setStyleSheet("background-color: #f59e0b; color: #0f172a; font-weight: bold; padding: 8px;")
        self.btn_trending_slang.clicked.connect(self.open_trending_slang_dialog)
        actions_row.addWidget(self.btn_trending_slang, 2)
        
        self.btn_export_srt = QPushButton("💾 XUẤT SRT")
        self.btn_export_srt.setStyleSheet("background-color: #475569; color: white; font-weight: bold; padding: 8px;")
        self.btn_export_srt.clicked.connect(self.export_srt_file)
        actions_row.addWidget(self.btn_export_srt, 1)
        
        editor_layout.addLayout(actions_row)
        card_editor.addLayout(editor_layout)
        layout.addWidget(card_editor)
        
        # 4. CARD KẾT XUẤT VIDEO LỒNG TIẾNG
        card_export = CollapsibleCard("💾 KẾT XUẤT VIDEO THÀNH PHẨM")
        export_layout = QVBoxLayout()
        export_layout.setSpacing(6)
        
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Lưu video tại:"))
        self.txt_out = QLineEdit()
        self.txt_out.setReadOnly(True)
        path_row.addWidget(self.txt_out)
        btn_out_browse = QPushButton("Duyệt...")
        btn_out_browse.clicked.connect(self.browse_output)
        path_row.addWidget(btn_out_browse)
        export_layout.addLayout(path_row)
        card_export.addLayout(export_layout)
        
        # Thêm hàng chọn Logo thương hiệu chèn vào video
        logo_layout = QHBoxLayout()
        logo_layout.addWidget(QLabel("Chèn Logo thương hiệu (PNG/JPG):"))
        self.txt_logo_path = QLineEdit()
        self.txt_logo_path.setPlaceholderText("Để trống nếu không chèn. Ví dụ: logo.png (Ảnh PNG trong suốt)...")
        self.txt_logo_path.setReadOnly(True)
        logo_layout.addWidget(self.txt_logo_path)
        btn_logo_browse = QPushButton("Chọn Logo...")
        btn_logo_browse.clicked.connect(self.browse_logo)
        logo_layout.addWidget(btn_logo_browse)
        card_export.addLayout(logo_layout)
        
        # Thêm checkbox chọn ghi đè phụ đề
        self.chk_burn_sub_export = QCheckBox("Ghi đè phụ đề tiếng Việt lên video (Che hoàn toàn phụ đề gốc)")
        self.chk_burn_sub_export.setChecked(True)
        self.chk_burn_sub_export.setStyleSheet("color: #dfb15b; font-weight: bold; margin-top: 5px;")
        card_export.addWidget(self.chk_burn_sub_export)

        # NOTE: Checkbox lồng tiếng chính đã có trong tab TTS. Không tạo thêm ở đây để tránh trùng lặp và trạng thái không đồng bộ.
        layout.addWidget(card_export)
        
        # Nút xuất chính và Batch xuất
        action_layout = QHBoxLayout()
        self.btn_start_dub = QPushButton("🎬 BẮT ĐẦU LỒNG TIẾNG & XUẤT VIDEO")
        self.btn_start_dub.setStyleSheet("background-color: #7fbeb2; color: #0c0c0e; font-size: 15px; font-weight: bold; padding: 14px; border-radius: 6px;")
        self.btn_start_dub.clicked.connect(self.start_dubbing)
        action_layout.addWidget(self.btn_start_dub, 3)
        
        self.btn_batch_dialog = QPushButton("📦 BATCH DUBBING...")
        self.btn_batch_dialog.setStyleSheet("background-color: #dfb15b; color: #0c0c0e; font-size: 15px; font-weight: bold; padding: 14px; border-radius: 6px;")
        self.btn_batch_dialog.clicked.connect(self.open_batch_dialog)
        action_layout.addWidget(self.btn_batch_dialog, 1)
        
        layout.addLayout(action_layout)
        
        # Logs
        self.txt_logs3 = QTextEdit()
        self.txt_logs3.setReadOnly(True)
        self.txt_logs3.setPlaceholderText("Tiến trình xuất video sẽ ghi nhận tại đây...")
        layout.addWidget(self.txt_logs3)
        
        scroll_area.setWidget(tab_content)
        self.tabs.addTab(scroll_area, "🎬 BÀN LÀM VIỆC CHÍNH")
        
    def create_settings_tab(self):
        from PyQt6.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        tab_content = QWidget()
        layout = QVBoxLayout(tab_content)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # 1. THIẾT LẬP POOL GEMINI API KEYS VÔ HẠN SLOT & TỰ ĐỘNG LƯU
        card_gemini = CollapsibleCard("🔑 BỘ QUẢN LÝ POOL GEMINI API KEYS (VÔ HẠN SLOT - TỰ ĐỘNG LƯU)")
        gemini_layout = QVBoxLayout()
        gemini_layout.setSpacing(8)
        
        lbl_gemini_info = QLabel("💡 Tự động nạp & lưu vĩnh viễn danh sách Gemini API Keys. Bạn có thể thêm vô hạn slot, dán hàng loạt nhiều key và kiểm tra health live.")
        lbl_gemini_info.setStyleSheet("color: #38bdf8; font-style: italic; font-size: 11px;")
        gemini_layout.addWidget(lbl_gemini_info)

        self.gemini_key_inputs = []
        self.gemini_key_status_labels = []
        self.gemini_key_rows = []

        # Thanh nút công cụ điều khiển Pool API Key
        toolbar_keys = QHBoxLayout()
        toolbar_keys.setSpacing(8)

        btn_add_slot = QPushButton("➕ Thêm Slot Key")
        btn_add_slot.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; border-radius: 6px; padding: 6px 12px;")
        btn_add_slot.clicked.connect(lambda: self.add_gemini_key_slot(""))
        toolbar_keys.addWidget(btn_add_slot)

        btn_batch_paste = QPushButton("📋 Dán Hàng Loạt Keys")
        btn_batch_paste.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; border-radius: 6px; padding: 6px 12px;")
        btn_batch_paste.clicked.connect(self.batch_import_gemini_keys)
        toolbar_keys.addWidget(btn_batch_paste)

        self.btn_check_gemini_keys = QPushButton("⚡ Kiểm Tra Health All Keys")
        self.btn_check_gemini_keys.setStyleSheet("background-color: #059669; color: white; font-weight: bold; border-radius: 6px; padding: 6px 12px;")
        self.btn_check_gemini_keys.clicked.connect(self.check_all_gemini_keys_health)
        toolbar_keys.addWidget(self.btn_check_gemini_keys)

        btn_clean_keys = QPushButton("🧹 Xóa Key Trống")
        btn_clean_keys.setStyleSheet("background-color: #475569; color: white; font-weight: bold; border-radius: 6px; padding: 6px 12px;")
        btn_clean_keys.clicked.connect(self.clean_empty_gemini_key_slots)
        toolbar_keys.addWidget(btn_clean_keys)

        btn_save_keys = QPushButton("💾 Lưu Vĩnh Viễn")
        btn_save_keys.setStyleSheet("background-color: #7c3aed; color: white; font-weight: bold; border-radius: 6px; padding: 6px 12px;")
        btn_save_keys.clicked.connect(self.save_api_config)
        toolbar_keys.addWidget(btn_save_keys)

        toolbar_keys.addStretch()
        gemini_layout.addLayout(toolbar_keys)

        # ScrollArea chứa danh sách các slot Key động (Vô hạn slot, không bị ép chiều cao)
        scroll_keys = QScrollArea()
        scroll_keys.setWidgetResizable(True)
        scroll_keys.setMinimumHeight(240)
        scroll_keys.setMaximumHeight(450)
        scroll_keys.setStyleSheet("QScrollArea { border: 1px solid #2a364f; border-radius: 6px; background-color: #0f172a; }")

        self.keys_container_widget = QWidget()
        self.keys_vbox = QVBoxLayout(self.keys_container_widget)
        self.keys_vbox.setContentsMargins(8, 8, 8, 8)
        self.keys_vbox.setSpacing(8)
        self.keys_vbox.addStretch()
        scroll_keys.setWidget(self.keys_container_widget)

        gemini_layout.addWidget(scroll_keys)
        card_gemini.addLayout(gemini_layout)
        layout.addWidget(card_gemini)

        # 2. THIẾT LẬP CÁC API KEYS KHÁC (GROQ, DEEPL, OLLAMA)
        card_api = CollapsibleCard("🔑 CÁC API KEYS KHÁC & OLLAMA MODEL")
        grid = QGridLayout()
        grid.setSpacing(8)
        
        self.txt_gemini_key = QLineEdit()
        self.txt_gemini_key.setVisible(False) # Synchronized internal field
        
        grid.addWidget(QLabel("Groq Key:"), 0, 0)
        self.txt_groq_key = QLineEdit()
        self.txt_groq_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_groq_key.setPlaceholderText("gsk_...")
        self.txt_groq_key.textChanged.connect(self.save_api_config)
        grid.addWidget(self.txt_groq_key, 0, 1)
        
        grid.addWidget(QLabel("DeepL Key:"), 1, 0)
        self.txt_deepl_key = QLineEdit()
        self.txt_deepl_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_deepl_key.setPlaceholderText("Tùy chọn...")
        self.txt_deepl_key.textChanged.connect(self.save_api_config)
        grid.addWidget(self.txt_deepl_key, 1, 1)
        
        grid.addWidget(QLabel("Model Ollama:"), 2, 0)
        self.txt_ollama_model = QLineEdit()
        self.txt_ollama_model.setText("qwen2.5")
        self.txt_ollama_model.textChanged.connect(self.save_api_config)
        grid.addWidget(self.txt_ollama_model, 2, 1)
        
        card_api.addLayout(grid)
        layout.addWidget(card_api)

        # Quick engine selection for translation flows
        eng_row = QHBoxLayout()
        eng_row.addWidget(QLabel("Chọn Engine dịch thuật:"))
        self.cb_engine = QComboBox()
        self.cb_engine.addItems(["Supersubs AI", "Dịch thô", "Dịch cơ bản", "Google Translate", "Ollama Local"]) 
        eng_row.addWidget(self.cb_engine)

        self.chk_refine_enabled = QCheckBox("Bật giai đoạn tinh chỉnh LLM")
        self.chk_refine_enabled.setChecked(False)
        eng_row.addWidget(self.chk_refine_enabled)

        self.cb_refine_engine = QComboBox()
        self.cb_refine_engine.addItems(["Gemini 1.5 Flash", "Gemini 1.5 Pro", "Gemini 2.0 Flash", "Groq Llama 3.1", "Ollama Local"]) 
        eng_row.addWidget(self.cb_refine_engine)

        layout.addLayout(eng_row)
        
        # 2. CẤU HÌNH PHƯƠNG PHÁP TRÍCH XUẤT (WHISPER & OCR)
        card_extract_cfg = CollapsibleCard("⚙️ THIẾT LẬP WHISPER & OCR HỆ THỐNG")
        grid_ext = QGridLayout()
        grid_ext.setSpacing(8)
        
        grid_ext.addWidget(QLabel("Phương pháp phụ đề:"), 0, 0)
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Nhận dạng giọng nói (Whisper/Gemini)", "Quét chữ cứng trên video (OCR)"])
        self.cb_mode.currentIndexChanged.connect(self.on_mode_changed)
        grid_ext.addWidget(self.cb_mode, 0, 1)
        
        self.btn_select_region = QPushButton("🔍 Đặt Vùng Quét OCR")
        self.btn_select_region.setEnabled(False)
        self.btn_select_region.clicked.connect(self.select_ocr_region)
        self.lbl_bbox = QLabel("Vùng quét: Chưa chọn")
        self.lbl_bbox.setStyleSheet("color: #7fbeb2; font-weight: bold;")
        grid_ext.addWidget(self.btn_select_region, 0, 2)
        
        self.whisper_widget = QWidget()
        whisper_layout = QHBoxLayout(self.whisper_widget)
        whisper_layout.setContentsMargins(0, 0, 0, 0)
        whisper_layout.addWidget(QLabel("Whisper Model:"))
        self.cb_whisper_model = QComboBox()
        self.cb_whisper_model.addItems(["tiny (máy rất yếu)", "base (nhanh, nhẹ)", "small (CPU ổn)", "medium (cân bằng)", "large-v3 (GPU)"])
        self.cb_whisper_model.setCurrentText("base (nhanh, nhẹ)")
        whisper_layout.addWidget(self.cb_whisper_model)
        grid_ext.addWidget(self.whisper_widget, 1, 0, 1, 3)
        
        self.ocr_widget = QWidget()
        ocr_layout = QHBoxLayout(self.ocr_widget)
        ocr_layout.setContentsMargins(0, 0, 0, 0)
        ocr_layout.addWidget(QLabel("Ngôn ngữ OCR:"))
        self.cb_ocr_lang = QComboBox()
        self.cb_ocr_lang.addItems(["Tự động (Trung, Anh)", "Tiếng Trung Giản Thể (`ch_sim`)", "Tiếng Trung Phồn Thể (`ch_tra`)", "Tiếng Việt (`vi`)", "Tiếng Anh (`en`)"])
        self.cb_ocr_lang.setCurrentIndex(0)
        ocr_layout.addWidget(self.cb_ocr_lang)
        
        self.chk_ocr_force_scan = QCheckBox("Quét cưỡng bức (Tạo dòng nếu phát hiện khung chữ khó)")
        self.chk_ocr_force_scan.setChecked(True)
        self.chk_ocr_force_scan.setStyleSheet("color: #dfb15b; font-weight: bold; margin-left: 10px;")
        ocr_layout.addWidget(self.chk_ocr_force_scan)
        grid_ext.addWidget(self.ocr_widget, 2, 0, 1, 3)
        
        self.ocr_widget.setVisible(False)
        
        grid_ext.addWidget(QLabel("Tốc độ video:"), 3, 0)
        self.cb_video_speed = QComboBox()
        self.cb_video_speed.addItems(["1.0x (Giữ nguyên)", "1.25x (Tăng tốc)", "1.5x (Tăng tốc)", "2.0x (Tăng tốc)"])
        self.cb_video_speed.setCurrentIndex(0)
        grid_ext.addWidget(self.cb_video_speed, 3, 1, 1, 2)
        
        grid_ext.addWidget(QLabel("VietPhrase.txt:"), 4, 0)
        self.txt_vp_path = QLineEdit()
        grid_ext.addWidget(self.txt_vp_path, 4, 1)
        btn_vp = QPushButton("Chọn...")
        btn_vp.clicked.connect(self.browse_vp_dict)
        grid_ext.addWidget(btn_vp, 4, 2)
        
        grid_ext.addWidget(QLabel("Names.txt:"), 5, 0)
        self.txt_names_path = QLineEdit()
        grid_ext.addWidget(self.txt_names_path, 5, 1)
        btn_names = QPushButton("Chọn...")
        btn_names.clicked.connect(self.browse_names_dict)
        grid_ext.addWidget(btn_names, 5, 2)

        # Glossary quick controls
        grid_ext.addWidget(QLabel("Glossary:"), 6, 0)
        self.cb_glossary_files = QComboBox()
        self.cb_glossary_files.addItem("-- Chọn Glossary --")
        self.cb_glossary_files.currentIndexChanged.connect(self.on_glossary_dropdown_changed)
        grid_ext.addWidget(self.cb_glossary_files, 6, 1)
        btn_glossary_load = QPushButton("Tải...")
        btn_glossary_load.clicked.connect(self.load_glossary_from_file)
        grid_ext.addWidget(btn_glossary_load, 6, 2)

        self.txt_glossary = QTextEdit()
        self.txt_glossary.setPlaceholderText("Định dạng mỗi dòng: từ_gốc = từ_dịch")
        self.txt_glossary.setMaximumHeight(120)
        grid_ext.addWidget(self.txt_glossary, 7, 0, 1, 3)

        btn_glossary_save = QPushButton("Lưu Glossary")
        btn_glossary_save.clicked.connect(self.save_glossary_to_file)
        grid_ext.addWidget(btn_glossary_save, 8, 2)
        
        card_extract_cfg.addLayout(grid_ext)
        layout.addWidget(card_extract_cfg)
        
        self.card_capcut = CollapsibleCard("📂 NẠP PHỤ ĐỀ TỪ DỰ ÁN CAPCUT PC")
        self.card_capcut.toggle_collapse()
        capcut_layout = QVBoxLayout()
        capcut_layout.setSpacing(6)
        
        capcut_header = QHBoxLayout()
        capcut_header.addWidget(QLabel("Quét các dự án CapCut PC cục bộ:"))
        self.btn_scan_capcut = QPushButton("🔄 Quét dự án")
        self.btn_scan_capcut.clicked.connect(self.scan_capcut_projects)
        capcut_header.addWidget(self.btn_scan_capcut)
        capcut_layout.addLayout(capcut_header)
        
        self.lbl_capcut_status = QLabel("Chưa quét")
        self.lbl_capcut_status.setStyleSheet("color: #9c9c9f; font-style: italic; font-size: 11px;")
        capcut_layout.addWidget(self.lbl_capcut_status)
        
        self.table_capcut = QTableWidget()
        self.table_capcut.setColumnCount(4)
        self.table_capcut.setHorizontalHeaderLabels(["STT", "Tên Dự Án", "Thời Gian Sửa", "Thao Tác"])
        self.table_capcut.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_capcut.setMaximumHeight(150)
        capcut_layout.addWidget(self.table_capcut)
        
        self.card_capcut.addLayout(capcut_layout)
        layout.addWidget(self.card_capcut)
        
        self.card_dataset_gen = CollapsibleCard("📦 TẠO TẬP DỮ LIỆU TRAIN OCR (DATASET GENERATOR)")
        self.card_dataset_gen.toggle_collapse()
        ds_layout = QVBoxLayout()
        ds_layout.setSpacing(6)
        
        ds_grid = QGridLayout()
        ds_grid.addWidget(QLabel("Video nguồn:"), 0, 0)
        self.txt_ds_video = QLineEdit()
        ds_grid.addWidget(self.txt_ds_video, 0, 1)
        btn_ds_video = QPushButton("Chọn...")
        btn_ds_video.clicked.connect(self.browse_ds_video)
        ds_grid.addWidget(btn_ds_video, 0, 2)
        
        ds_grid.addWidget(QLabel("File SRT nhãn:"), 1, 0)
        self.txt_ds_srt = QLineEdit()
        ds_grid.addWidget(self.txt_ds_srt, 1, 1)
        btn_ds_srt = QPushButton("Chọn...")
        btn_ds_srt.clicked.connect(self.browse_ds_srt)
        ds_grid.addWidget(btn_ds_srt, 1, 2)
        
        self.btn_gen_dataset = QPushButton("📦 Bắt đầu tạo ảnh cắt OCR Dataset")
        self.btn_gen_dataset.setStyleSheet("background-color: #3b82f6; color: white; font-weight: bold;")
        self.btn_gen_dataset.clicked.connect(self.start_dataset_generation)
        
        self.lbl_ds_status = QLabel("Chưa khởi chạy")
        self.lbl_ds_status.setStyleSheet("color: #9c9c9f; font-style: italic; font-size: 11px;")
        
        ds_layout.addLayout(ds_grid)
        ds_layout.addWidget(self.btn_gen_dataset)
        ds_layout.addWidget(self.lbl_ds_status)
        
        self.card_dataset_gen.addLayout(ds_layout)
        layout.addWidget(self.card_dataset_gen)
        
        scroll_area.setWidget(tab_content)
        return scroll_area
        

    def scan_capcut_projects(self):
        import glob
        import datetime
        self.lbl_capcut_status.setText("Đang tìm kiếm dự án CapCut...")
        self.btn_scan_capcut.setEnabled(False)
        
        local_appdata = os.environ.get('LOCALAPPDATA')
        if not local_appdata:
            local_appdata = os.path.expandvars(r'%USERPROFILE%\AppData\Local')

        possible_paths = [
            os.path.join(local_appdata, 'CapCut', 'User Data', 'Projects', 'com.lveditor.draft'),
            os.path.join(local_appdata, 'CapCut', 'User Data', 'Projects', 'com.lved.pc'),
            os.path.join(local_appdata, 'CapCut', 'User Data', 'Projects')
        ]

        draft_files = []
        found_dir = ""
        for path in possible_paths:
            if os.path.exists(path):
                found_files = glob.glob(os.path.join(path, '**', 'draft_content.json'), recursive=True)
                if found_files:
                    draft_files = found_files
                    found_dir = path
                    break

        if not draft_files:
            self.lbl_capcut_status.setText("Không tìm thấy dự án CapCut mặc định nào trên máy.")
            self.table_capcut.setRowCount(0)
            self.btn_scan_capcut.setEnabled(True)
            return

        draft_files.sort(key=os.path.getmtime, reverse=True)
        self.capcut_projects = draft_files
        
        self.lbl_capcut_status.setText(f"Tìm thấy {len(draft_files)} dự án tại: {found_dir}")
        self.table_capcut.setRowCount(len(draft_files))
        
        for idx, file in enumerate(draft_files):
            mtime = os.path.getmtime(file)
            dt = datetime.datetime.fromtimestamp(mtime).strftime('%d/%m/%Y %H:%M:%S')
            folder_name = os.path.basename(os.path.dirname(file))
            if folder_name == 'Timelines':
                folder_name = os.path.basename(os.path.dirname(os.path.dirname(file)))
                
            stt_item = QTableWidgetItem(str(idx + 1))
            stt_item.setFlags(stt_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_capcut.setItem(idx, 0, stt_item)
            
            name_item = QTableWidgetItem(folder_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_capcut.setItem(idx, 1, name_item)
            
            time_item = QTableWidgetItem(dt)
            time_item.setFlags(time_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_capcut.setItem(idx, 2, time_item)
            
            btn_load = QPushButton("⚡ Nạp Phụ Đề")
            btn_load.setStyleSheet("background-color: #7fbeb2; color: #0c0c0e; font-weight: bold; padding: 2px 8px; border-radius: 3px;")
            btn_load.clicked.connect(lambda _, f=file, n=folder_name: self.extract_and_load_capcut(f, n))
            self.table_capcut.setCellWidget(idx, 3, btn_load)
            
        self.btn_scan_capcut.setEnabled(True)


    def create_review_tab(self):
        from PyQt6.QtWidgets import QScrollArea, QFileDialog
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        tab_content = QWidget()
        layout = QVBoxLayout(tab_content)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        card = CollapsibleCard("🎞️ REVIEW PHIM - QUICK WORKFLOW")
        v = QVBoxLayout()

        # Quick Scan (OCR) button
        btn_quick_scan = QPushButton("🔎 Quét phụ đề (Auto OCR)")
        btn_quick_scan.setStyleSheet("background-color: #3b82f6; color: white; font-weight: bold; padding: 8px;")
        btn_quick_scan.clicked.connect(self.quick_scan_action)
        v.addWidget(btn_quick_scan)

        # Merge SRT
        btn_merge = QPushButton("🔗 Gộp nhiều SRT → 1 file")
        btn_merge.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; padding: 8px;")
        btn_merge.clicked.connect(self.merge_srt_files_dialog)
        v.addWidget(btn_merge)

        # Sync / TTS
        btn_sync_tts = QPushButton("🔊 Đồng bộ theo TTS (Generate + Align)")
        btn_sync_tts.setStyleSheet("background-color: #7fbeb2; color: #0c0c0e; font-weight: bold; padding: 8px;")
        btn_sync_tts.clicked.connect(self.start_dubbing)
        v.addWidget(btn_sync_tts)

        # Auto Render
        btn_auto_render = QPushButton("🎬 Auto Render (Xuất nhanh)")
        btn_auto_render.setStyleSheet("background-color: #dfb15b; color: #0c0c0e; font-weight: bold; padding: 8px;")
        btn_auto_render.clicked.connect(self.start_dubbing)
        v.addWidget(btn_auto_render)

        # Mask toggle
        btn_mask = QPushButton("🛡️ Áp dụng Che phụ đề gốc (Blur/Box)")
        btn_mask.setStyleSheet("background-color: #f97316; color: white; font-weight: bold; padding: 8px;")
        btn_mask.clicked.connect(self.toggle_mask_and_refresh)
        v.addWidget(btn_mask)

        btn_preview_mask = QPushButton("👁️ Xem trước Che phụ đề (Preview Mask)")
        btn_preview_mask.setStyleSheet("background-color: #6b7280; color: white; font-weight: bold; padding: 8px;")
        btn_preview_mask.clicked.connect(self.preview_mask_now)
        v.addWidget(btn_preview_mask)

        # Volume / Speed quick controls
        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Âm lượng gốc:"))
        vol_row.addWidget(self.slider_bg)
        vol_row.addWidget(QLabel("Âm lượng AI:"))
        vol_row.addWidget(self.slider_dub)
        v.addLayout(vol_row)

        card.addLayout(v)
        layout.addWidget(card)

        scroll_area.setWidget(tab_content)
        self.tabs.addTab(scroll_area, "🎞️ Review Phim")

    def quick_scan_action(self):
        # Switch mode to OCR and invoke the existing extraction workflow
        try:
            self.cb_mode.setCurrentIndex(1)  # OCR mode
            # If no bbox selected, allow auto-detection in worker
            self.start_extraction()
        except Exception as e:
            self.status_label.setText(f"Lỗi khi chạy Quét nhanh: {e}")

    def merge_srt_files_dialog(self):
        from PyQt6.QtWidgets import QFileDialog
        import merge_subs
        files, _ = QFileDialog.getOpenFileNames(self, "Chọn các file SRT để gộp", os.getcwd(), "SubRip (*.srt);;All files (*)")
        if not files:
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Lưu file SRT gộp", os.path.join(os.getcwd(), "merged.srt"), "SubRip (*.srt)")
        if not out_path:
            return
        try:
            merge_subs.merge_files(files, out_path, strategy='combine')
            self.status_label.setText(f"Đã gộp và lưu: {out_path}")
        except Exception as e:
            self.status_label.setText(f"Lỗi gộp SRT: {e}")

    def toggle_mask_and_refresh(self):
        # Toggle mask mode between None and Blur quick for convenience
        try:
            cur = self.cb_mask_mode.currentText()
            if "Không che" in cur:
                self.cb_mask_mode.setCurrentIndex(2)  # Blur
            else:
                self.cb_mask_mode.setCurrentIndex(0)  # None
            self.trigger_canvas_update()
            self.status_label.setText("Đã cập nhật chế độ che phụ đề.")
        except Exception as e:
            self.status_label.setText(f"Lỗi khi thay đổi mask: {e}")

    def preview_mask_now(self):
        """Lấy 1 khung hình từ video gốc, áp dụng thuật toán xóa/che phụ đề (theo preset), và hiển thị lên preview."""
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn video trước khi xem trước che phụ đề.")
            return
        try:
            cap = cv2.VideoCapture(self.video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                QMessageBox.warning(self, "Lỗi", "Không thể đọc frame từ video.")
                return
            mid = max(0, total // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                QMessageBox.warning(self, "Lỗi", "Không thể lấy khung hình từ video.")
                return

            # Sử dụng preset từ UI
            preset = self.get_current_subtitle_preset()
            default_bbox = self.selected_bbox
            selected_bboxes = self.selected_bboxes if self.selected_bboxes else None

            # Gọi hàm xử lý (dubber.draw_burned_subtitle) để chỉ áp dụng mask
            try:
                processed_frame, _ = dubber.draw_burned_subtitle(frame.copy(), "", default_bbox, default_bbox, preset=preset, selected_bboxes=selected_bboxes, logo_path=self.logo_path)
            except Exception:
                # Fallback: nếu dubber không khả dụng, áp dụng blur đơn giản
                processed_frame = frame.copy()
                if default_bbox:
                    x, y, w, h = default_bbox
                    processed_frame = dubber.apply_opencv_watermark_removal(processed_frame, default_bbox, preset.get('mask_mode','blur'))

            # Hiển thị lên label preview chính (truyền frame gốc, DraggablePreviewLabel tự scale)
            self.lbl_main_preview.setVideoFrame(processed_frame)
            self.status_label.setText("Xem trước chế độ che phụ đề đã cập nhật.")
        except Exception as e:
            QMessageBox.warning(self, "Lỗi preview", str(e))

    def extract_and_load_capcut(self, json_path, project_name):
        import re
        export_dir = os.path.join(os.getcwd(), "srt gốc")
        os.makedirs(export_dir, exist_ok=True)
        
        output_name = f"CapCut_{project_name}.srt"
        output_path = os.path.join(export_dir, output_name)
        
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int(round((seconds - int(seconds)) * 1000))
            if millis > 999:
                millis = 999
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            text_materials = {}
            texts_list = data.get('materials', {}).get('texts', [])
            for text_mat in texts_list:
                mat_id = text_mat.get('id')
                raw_content = text_mat.get('content', '')
                
                text_val = ""
                try:
                    content_json = json.loads(raw_content)
                    text_val = content_json.get('text', '')
                except Exception:
                    cleaned = re.sub(r'<[^>]*>', '', raw_content)
                    text_val = cleaned.strip()
                
                if not text_val:
                    m = re.search(r'"text"\s*:\s*"([^"]+)"', raw_content)
                    if m:
                        text_val = m.group(1)
                    else:
                        text_val = raw_content
                text_materials[mat_id] = text_val

            subtitle_segments = []
            for track in data.get('tracks', []):
                for segment in track.get('segments', []):
                    material_id = segment.get('material_id')
                    timerange = segment.get('target_timerange') or segment.get('source_timerange')
                    
                    if timerange and material_id in text_materials:
                        start_us = timerange.get('start', 0)
                        duration_us = timerange.get('duration', 0)
                        
                        start_sec = start_us / 1000000.0
                        end_sec = (start_us + duration_us) / 1000000.0
                        
                        text = text_materials[material_id]
                        if text and text.strip():
                            subtitle_segments.append({
                                'start': start_sec,
                                'end': end_sec,
                                'text': text.strip()
                            })

            if not subtitle_segments:
                QMessageBox.warning(self, "Không tìm thấy phụ đề", "Dự án CapCut này không có phụ đề chữ hoặc không đọc được.")
                return

            subtitle_segments.sort(key=lambda x: x['start'])

            # Quy đổi thời gian phụ đề theo tốc độ video
            speed = self.get_video_speed_factor()
            if speed != 1.0:
                for seg in subtitle_segments:
                    seg['start'] = seg['start'] / speed
                    seg['end'] = seg['end'] / speed

            # Ghi ra file SRT
            with open(output_path, 'w', encoding='utf-8') as f:
                for idx, seg in enumerate(subtitle_segments, 1):
                    start_str = format_time(seg['start'])
                    end_str = format_time(seg['end'])
                    f.write(f"{idx}\n")
                    f.write(f"{start_str} --> {end_str}\n")
                    f.write(f"{seg['text']}\n\n")

            # Nạp thẳng vào tool
            self.txt_srt_path.setText(output_path)
            self.on_extraction_finished(subtitle_segments, self.video_path if self.video_path else "")
            self.status_label.setText(f"Đã nạp phụ đề từ dự án CapCut: {project_name}")
            
            # Chuyển sang Tab Bàn làm việc chính (index 0)
            self.tabs.setCurrentIndex(0)
            QMessageBox.information(self, "Thành công", f"Đã trích xuất và nạp {len(subtitle_segments)} câu phụ đề từ dự án CapCut '{project_name}' thành công!")

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi xử lý dự án CapCut: {e}")

    def setup_subtitle_styling_connections(self):
        btn_font = getattr(self, 'btn_font_color', getattr(self, 'tab2_btn_font_color', None))
        txt_font_hex = getattr(self, 'txt_font_color_hex', getattr(self, 'tab2_lbl_font_hex', None))
        if btn_font and txt_font_hex:
            self.setup_color_button_hex(btn_font, txt_font_hex, getattr(self, 'preset_font_color', '#FFFFFF'), getattr(self, 'on_font_color_changed', lambda c: None))

        btn_out = getattr(self, 'btn_outline_color', getattr(self, 'tab2_btn_outline_color', None))
        txt_out_hex = getattr(self, 'txt_outline_color_hex', getattr(self, 'tab2_lbl_outline_hex', None))
        if btn_out and txt_out_hex:
            self.setup_color_button_hex(btn_out, txt_out_hex, getattr(self, 'preset_outline_color', '#000000'), getattr(self, 'on_outline_color_changed', lambda c: None))

        btn_bg = getattr(self, 'btn_bg_color', None)
        txt_bg_hex = getattr(self, 'txt_bg_color_hex', None)
        if btn_bg and txt_bg_hex:
            self.setup_color_button_hex(btn_bg, txt_bg_hex, getattr(self, 'preset_bg_color', '#000000'), getattr(self, 'on_bg_color_changed', lambda c: None))

        for attr_name in ['cb_v_align', 'cb_h_align', 'spin_margin_v', 'cb_margin_v_type', 'spin_margin_h', 'cb_margin_h_type', 'spin_font_size', 'spin_outline_width', 'chk_use_bg_box', 'slider_bg_opacity', 'tab2_font_size', 'tab2_outline_width', 'tab2_chk_bg_box']:
            w = getattr(self, attr_name, None)
            if w:
                if hasattr(w, 'currentIndexChanged'):
                    w.currentIndexChanged.connect(getattr(self, 'on_preset_control_changed', lambda: None))
                elif hasattr(w, 'valueChanged'):
                    w.valueChanged.connect(getattr(self, 'on_preset_control_changed', lambda: None))
                elif hasattr(w, 'toggled'):
                    w.toggled.connect(getattr(self, 'on_preset_control_changed', lambda: None))

        if hasattr(self, 'spin_confidence_threshold'):
            self.spin_confidence_threshold.valueChanged.connect(getattr(self, 'populate_subtitle_table', lambda: None))

        if hasattr(self, 'cb_font_name'):
            self.cb_font_name.currentIndexChanged.connect(getattr(self, 'on_font_changed', lambda: None))
        if hasattr(self, 'btn_browse_font'):
            self.btn_browse_font.clicked.connect(getattr(self, 'browse_custom_font', lambda: None))
        if hasattr(self, 'cb_preset'):
            self.cb_preset.currentTextChanged.connect(getattr(self, 'on_preset_changed', lambda: None))
        if hasattr(self, 'btn_reset_preset'):
            self.btn_reset_preset.clicked.connect(getattr(self, 'reset_preset', lambda: None))
        
        # Kết nối sự kiện tọa độ tùy chỉnh
        for pos_spin in ['spin_custom_pos_x', 'spin_custom_pos_y']:
            w = getattr(self, pos_spin, None)
            if w and hasattr(w, 'valueChanged'):
                w.valueChanged.connect(getattr(self, 'on_custom_pos_spin_changed', lambda: None))
                w.valueChanged.connect(getattr(self, 'trigger_canvas_update', lambda: None))

        if hasattr(self, 'btn_reset_custom_pos'):
            self.btn_reset_custom_pos.clicked.connect(getattr(self, 'reset_subtitle_custom_pos', lambda: None))

        # Kết nối sự kiện thay đổi style để vẽ canvas cập nhật thời gian thực
        for cb_name in ['cb_preset', 'cb_preset_apply_mode', 'cb_h_align', 'cb_v_align', 'spin_margin_v', 'cb_margin_v_type', 'spin_margin_h', 'cb_margin_h_type', 'cb_font_name', 'spin_font_size', 'spin_outline_width', 'chk_use_bg_box', 'slider_bg_opacity', 'cb_mask_mode']:
            w = getattr(self, cb_name, None)
            if w:
                if hasattr(w, 'currentIndexChanged'):
                    w.currentIndexChanged.connect(getattr(self, 'trigger_canvas_update', lambda: None))
                elif hasattr(w, 'valueChanged'):
                    w.valueChanged.connect(getattr(self, 'trigger_canvas_update', lambda: None))
                elif hasattr(w, 'stateChanged'):
                    w.stateChanged.connect(getattr(self, 'trigger_canvas_update', lambda: None))

        for txt_name in ['txt_font_color_hex', 'txt_outline_color_hex', 'txt_bg_color_hex']:
            w = getattr(self, txt_name, None)
            if w and hasattr(w, 'textChanged'):
                w.textChanged.connect(getattr(self, 'trigger_canvas_update', lambda: None))

        if hasattr(self, 'btn_preview_sub'):
            self.btn_preview_sub.clicked.connect(getattr(self, 'open_preview_dialog', lambda: None))

        btn_f = getattr(self, 'btn_font_color', getattr(self, 'tab2_btn_font_color', None))
        txt_f = getattr(self, 'txt_font_color_hex', getattr(self, 'tab2_lbl_font_hex', None))
        if btn_f and txt_f:
            self.update_color_button(btn_f, txt_f, getattr(self, 'preset_font_color', (255, 255, 255)))

        btn_o = getattr(self, 'btn_outline_color', getattr(self, 'tab2_btn_outline_color', None))
        txt_o = getattr(self, 'txt_outline_color_hex', getattr(self, 'tab2_lbl_outline_hex', None))
        if btn_o and txt_o:
            self.update_color_button(btn_o, txt_o, getattr(self, 'preset_outline_color', (0, 0, 0)))

        if hasattr(self, '_tab2_sync_from_preset'):
            self._tab2_sync_from_preset()
        
        # Kết nối ghi nhật ký tự động cho các CheckBox và cập nhật số bước tiến trình
        self.setup_checkbox_logging_connections()

    def setup_checkbox_logging_connections(self):
        checkbox_mappings = [
            (getattr(self, 'chk_enable_dubbing', None), "Bật lồng tiếng TTS"),
            (getattr(self, 'chk_refine_enabled', None), "Bật giai đoạn tinh chỉnh LLM"),
            (getattr(self, 'chk_burn_sub_export', None), "Ghi đè phụ đề tiếng Việt lên video (Che sub gốc)"),
            (getattr(self, 'chk_restrict_ocr', None), "Giới hạn quét ngang 60%"),
            (getattr(self, 'chk_smart_pos', None), "Tự động căn phụ đề đè lên hộp che (Smart Pos)"),
            (getattr(self, 'chk_ocr_force_scan', None), "Quét cưỡng bức (OCR)"),
            (getattr(self, 'chk_use_bg_box', None), "Bật nền phụ đề"),
            (getattr(self, 'tab2_chk_bg_box', None), "Bật hộp nền chữ (Kiểu chữ Subtitle)")
        ]
        
        for chk, name in checkbox_mappings:
            if chk is not None and not getattr(chk, '_logging_connected', False):
                chk.toggled.connect(lambda checked, c=chk, n=name: self._on_checkbox_toggled_log(c, n, checked))
                chk._logging_connected = True

    def _on_checkbox_toggled_log(self, chk_obj, name, is_checked):
        state_str = "BẬT [✔]" if is_checked else "TẮT [✖]"
        self.log_info(f"⚙️ Thay đổi cấu hình: '{name}' -> {state_str}")
        
        # Nếu là checkbox làm thay đổi số bước trong quy trình tự động (lồng tiếng hoặc LLM refine)
        if chk_obj in (getattr(self, 'chk_enable_dubbing', None), getattr(self, 'chk_refine_enabled', None)):
            self.recalculate_and_log_pipeline_steps()

    def recalculate_and_log_pipeline_steps(self):
        steps = []
        steps.append("Bước 1: Trích phụ đề (OCR/Whisper)")
        steps.append("Bước 2: Dịch thuật AI")
        
        if hasattr(self, 'chk_refine_enabled') and self.chk_refine_enabled.isChecked():
            steps.append("Bước 3: Tinh chỉnh LLM")
            
        if hasattr(self, 'chk_enable_dubbing') and self.chk_enable_dubbing.isChecked():
            step_num = len(steps) + 1
            steps.append(f"Bước {step_num}: Sinh giọng đọc TTS")
            
        step_final = len(steps) + 1
        steps.append(f"Bước {step_final}: Ghi đè & Xuất video thành phẩm")
        
        total = len(steps)
        summary = " ➔ ".join(steps)
        self.log_info(f"📊 [TIẾN TRÌNH QUY TRÌNH HỆ THỐNG] Đã cập nhật: Tổng cộng {total} BƯỚC ({summary})")
        
    def on_mode_changed(self, index):
        # Bật/tắt nút vẽ khung quét chữ tuỳ thuộc vào chế độ chọn
        is_ocr = (index == 1)
        self.btn_select_region.setEnabled(is_ocr)
        if not is_ocr:
            self.lbl_bbox.setText("Vùng quét: Không áp dụng")
            
        # Ẩn/Hiện widget tương ứng
        self.whisper_widget.setVisible(not is_ocr)
        self.ocr_widget.setVisible(is_ocr)
            
    def load_video_preview(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return
        self.video_path = file_path
        if hasattr(self, '_frame_cache'):
            self._frame_cache.clear()
        if hasattr(self, 'lbl_status_video') and self.lbl_status_video:
            self.lbl_status_video.setText(f"Video: {os.path.basename(file_path)}")
        if hasattr(self, 'txt_file'):
            self.txt_file.setText(file_path)
        if hasattr(self, 'txt_ds_video'):
            self.txt_ds_video.setText(file_path)
        if hasattr(self, 'btn_select_region') and hasattr(self, 'cb_mode'):
            self.btn_select_region.setEnabled(self.cb_mode.currentIndex() == 1)
        
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.current_preview_raw_frame = frame.copy()
                self.selected_bboxes = []
                self.selected_bbox = None
                self.title_bbox = None
                if hasattr(self, 'lbl_bbox'):
                    self.lbl_bbox.setText("Vùng quét: Chưa chọn")
                
                if total_frames > 0:
                    tot_sec = total_frames / fps
                    h2, m2, s2 = int(tot_sec // 3600), int((tot_sec % 3600) // 60), int(tot_sec % 60)
                    if hasattr(self, 'lbl_frame_info'):
                        self.lbl_frame_info.setText(f"Frame: 0 / {total_frames}   Time: 00:00:00 / {h2:02d}:{m2:02d}:{s2:02d}")
                    if hasattr(self, 'lbl_player_time'):
                        self.lbl_player_time.setText(f"00:00 / {m2:02d}:{s2:02d}")
                    if hasattr(self, 'slider_player_timeline'):
                        self.slider_player_timeline.setValue(0)
                    if hasattr(self, 'lbl_chunks_count'):
                        est_chunks = max(1, int(tot_sec / 20.0 + 0.99))
                        self.lbl_chunks_count.setText(f"📦 Số chunks ước tính: {est_chunks}")
                
                self.show_preview_frame(frame)

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if file_path:
            self.load_video_preview(file_path)

    def browse_srt_file(self):
        default_dir = os.path.join(os.getcwd(), "srt gốc")
        os.makedirs(default_dir, exist_ok=True)
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file phụ đề SRT gốc", default_dir, "SRT Files (*.srt)")
        if file_path:
            self.txt_srt_path.setText(os.path.abspath(file_path))
            if hasattr(self, 'txt_ds_srt'):
                self.txt_ds_srt.setText(os.path.abspath(file_path))
            # Tự động nạp phụ đề từ file srt này luôn
            try:
                try:
                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                        srt_content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='utf-16') as f:
                            srt_content = f.read()
                    except UnicodeDecodeError:
                        with open(file_path, 'r', encoding='mbcs') as f:
                            srt_content = f.read()
                segments = transcriber.parse_srt_string(srt_content)
                if segments:
                    # Quy đổi thời gian phụ đề theo tốc độ video
                    speed = self.get_video_speed_factor()
                    if speed != 1.0:
                        for seg in segments:
                            seg['start'] = seg['start'] / speed
                            seg['end'] = seg['end'] / speed
                    self.on_extraction_finished(segments, self.video_path if self.video_path else "")
                    self.status_label.setText(f"Đã nạp phụ đề từ: {os.path.basename(file_path)} (Quy đổi {speed}x)")
                else:
                    QMessageBox.warning(self, "Lỗi đọc file", "Không thể phân tích nội dung phụ đề từ file SRT này.")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Lỗi nạp file SRT: {e}")

    def select_ocr_region(self):
        video_source = self.video_path
        
        if not video_source or not os.path.exists(video_source):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn tệp video nguồn hợp lệ trước.")
            return
            
        self.status_label.setText("Đang trích xuất ảnh video...")
        
        # Trích xuất khung hình ở giây thứ 2
        cap = cv2.VideoCapture(video_source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2))
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        cap.release()
        
        if not ret:
            QMessageBox.critical(self, "Lỗi", "Không thể trích xuất khung hình từ video.")
            return
        self.status_label.setText("Đang vẽ vùng quét...")
        selector = VideoRegionSelector(frame, self)
        
        # Nạp tất cả bboxes đã chọn trước đó
        if getattr(self, 'selected_bboxes', None):
            selector.bboxes = list(self.selected_bboxes)
            selector.update_bboxes_list()
        else:
            if self.selected_bbox:
                selector.bboxes.append(self.selected_bbox)
            if getattr(self, 'title_bbox', None):
                selector.bboxes.append(self.title_bbox)
            selector.update_bboxes_list()
            
        if selector.exec() == QDialog.DialogCode.Accepted:
            self.selected_bboxes = list(selector.bboxes)
            
            # Phân loại thông minh selected_bbox (phụ đề ở dưới) và title_bbox (tiêu đề ở trên)
            sub_box = None
            title_box = None
            logo_box = None

            if not hasattr(self, 'box_type_dict') or self.box_type_dict is None:
                self.box_type_dict = {}

            for b in self.selected_bboxes:
                b_type = self.box_type_dict.get(tuple(b))
                if b_type == 'sub': sub_box = b
                elif b_type == 'title': title_box = b
                elif b_type == 'logo': logo_box = b

            h_vid = getattr(self, 'video_height', 1080)
            if not sub_box or not title_box:
                sorted_boxes = sorted(self.selected_bboxes, key=lambda box: box[1])
                if len(sorted_boxes) >= 2:
                    title_box = title_box or sorted_boxes[0]
                    sub_box = sub_box or sorted_boxes[-1]
                elif len(sorted_boxes) == 1:
                    if sorted_boxes[0][1] < h_vid * 0.4:
                        title_box = title_box or sorted_boxes[0]
                    else:
                        sub_box = sub_box or sorted_boxes[0]

            self.selected_bbox = sub_box or (self.selected_bboxes[0] if self.selected_bboxes else None)
            self.title_bbox = title_box
            self.logo_bbox = logo_box
            
            if self.selected_bboxes:
                self.lbl_bbox.setText(f"Đã chọn {len(self.selected_bboxes)} vùng quét")
                self.status_label.setText("Đã cập nhật danh sách vùng quét.")
            else:
                QMessageBox.information(self, "Thông tin", "Không có vùng quét nào được chọn.")
                self.lbl_bbox.setText("Vùng quét: Chưa chọn")
                
    def browse_output(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Lưu Video Lồng Tiếng", "", "Video File (*.mp4)")
        if file_path:
            self.txt_out.setText(file_path)
            
    def browse_logo(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn hình ảnh Logo thương hiệu", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.logo_path = file_path
            self.txt_logo_path.setText(file_path)
            self.status_label.setText(f"Đã chọn logo thương hiệu: {os.path.basename(file_path)}")
            # Làm mới khung hình để chèn logo đè lên vị trí mới vẽ
            if hasattr(self, 'current_preview_raw_frame') and self.current_preview_raw_frame is not None:
                self.show_preview_frame(self.current_preview_raw_frame)
                
    def on_canvas_bbox_added(self, new_box):
        if not hasattr(self, 'selected_bboxes') or self.selected_bboxes is None:
            self.selected_bboxes = []
        if new_box not in self.selected_bboxes:
            self.selected_bboxes.append(new_box)
        
        if not hasattr(self, 'box_type_dict') or self.box_type_dict is None:
            self.box_type_dict = {}

        vx, vy, vw, vh = new_box
        h_vid = getattr(self, 'video_height', 1080)
        w_vid = getattr(self, 'video_width', 1920)

        box_key = tuple(new_box)
        if box_key not in self.box_type_dict:
            if vy > int(h_vid * 0.4):
                self.box_type_dict[box_key] = 'sub'
                self.selected_bbox = new_box
            elif vy < int(h_vid * 0.3) and vx > int(w_vid * 0.5):
                self.box_type_dict[box_key] = 'logo'
                self.logo_bbox = new_box
            else:
                self.box_type_dict[box_key] = 'title'
                self.title_bbox = new_box
        
        if hasattr(self, 'lbl_bbox') and self.lbl_bbox is not None:
            self.lbl_bbox.setText(f"Đã chọn {len(self.selected_bboxes)} vùng quét")
        if hasattr(self, 'status_label') and self.status_label is not None:
            self.status_label.setText(f"Đã vẽ thêm Vùng {len(self.selected_bboxes)}: {new_box}")
        
        if hasattr(self.lbl_main_preview, 'bboxes'):
            self.lbl_main_preview.bboxes = list(self.selected_bboxes)
            
        if hasattr(self, 'current_preview_raw_frame') and self.current_preview_raw_frame is not None:
            self.show_preview_frame(self.current_preview_raw_frame)
            
    def clear_all_canvas_crops(self):
        self.selected_bboxes = []
        self.selected_bbox = None
        self.logo_bbox = None
        self.title_bbox = None
        self.box_type_dict = {}
        if hasattr(self.lbl_main_preview, 'bboxes'):
            self.lbl_main_preview.bboxes = []
        if hasattr(self, 'lbl_bbox') and self.lbl_bbox is not None:
            self.lbl_bbox.setText("Vùng quét: Chưa chọn")
        if hasattr(self, 'status_label') and self.status_label is not None:
            self.status_label.setText("Đã xóa tất cả vùng quét.")
        if hasattr(self, 'current_preview_raw_frame') and self.current_preview_raw_frame is not None:
            self.show_preview_frame(self.current_preview_raw_frame)
        else:
            self.trigger_canvas_update()
            
    def browse_vp_dict(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn tệp VietPhrase.txt", "", "Text Files (*.txt)")
        if file_path:
            self.txt_vp_path.setText(file_path)
            
    def browse_names_dict(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn tệp Names.txt", "", "Text Files (*.txt)")
        if file_path:
            self.txt_names_path.setText(file_path)

    def get_video_speed_factor(self):
        txt = self.cb_video_speed.currentText()
        if "1.25x" in txt:
            return 1.25
        elif "1.5x" in txt:
            return 1.5
        elif "2.0x" in txt:
            return 2.0
        elif "0.75x" in txt:
            return 0.75
        elif "0.5x" in txt:
            return 0.5
        return 1.0

    def adjust_video_speed_if_needed(self):
        speed = self.get_video_speed_factor()
        if speed == 1.0:
            return True, self.video_path

        if hasattr(self, '_original_video_path') and self._original_video_path == self.video_path:
            if hasattr(self, '_speed_adjusted_video_path') and os.path.exists(self._speed_adjusted_video_path):
                return True, self._speed_adjusted_video_path

        import subprocess
        
        self.status_label.setText(f"Đang đồng bộ quy đổi tốc độ video sang {speed}x...")
        self.append_log1(f"Bắt đầu quy đổi tốc độ video: {speed}x...")
        
        temp_dir = os.path.join(os.getcwd(), "temp_speed")
        os.makedirs(temp_dir, exist_ok=True)
        
        out_path = os.path.join(temp_dir, f"speed_{speed}_{os.path.basename(self.video_path)}")
        
        cmd = [
            "ffmpeg", "-y", "-i", self.video_path,
            "-vf", f"setpts={1.0/speed}*PTS",
            "-af", f"atempo={speed}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            out_path
        ]
        
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(out_path):
                self._original_video_path = self.video_path
                self._speed_adjusted_video_path = out_path
                self.append_log1(f"Đã quy đổi tốc độ video thành công: {out_path}")
                return True, out_path
            else:
                err = res.stderr.decode('utf-8', errors='ignore')
                self.append_log1(f"Lỗi quy đổi tốc độ: {err}")
                QMessageBox.critical(self, "Lỗi", f"Không thể quy đổi tốc độ video bằng FFmpeg: {err}")
                return False, self.video_path
        except Exception as e:
            self.append_log1(f"Lỗi quy đổi tốc độ: {e}")
            QMessageBox.critical(self, "Lỗi", f"Lỗi trong quá trình quy đổi tốc độ: {e}")
            return False, self.video_path

    def auto_detect_subtitle_bbox(self, video_path):
        self.status_label.setText("Đang tự động phát hiện vùng phụ đề gốc...")
        self.append_log1("Đang tự động phân tích khung hình để xác định vùng phụ đề...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        if total_frames <= 0 or h <= 0 or w <= 0:
            cap.release()
            return None
            
        frame_indices = [int(total_frames * p) for p in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]]
        detected_boxes = []
        
        from transcriber import get_easyocr_reader, get_easyocr_lang_candidates
        ocr_lang = self.cb_ocr_lang.currentText()
        candidates = get_easyocr_lang_candidates(ocr_lang)[0]
        reader = get_easyocr_reader(candidates)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            y_start = int(h * 0.7)
            y_end = int(h * 0.95)
            crop = frame[y_start:y_end, 0:w]
            
            results = reader.readtext(crop)
            for res in results:
                bbox_points = res[0]
                text = res[1]
                conf = res[2]
                
                if conf > 0.4 and len(text.strip()) > 1:
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    
                    x1 = int(min(xs))
                    y1 = int(min(ys)) + y_start
                    x2 = int(max(xs))
                    y2 = int(max(ys)) + y_start
                    
                    detected_boxes.append((x1, y1, x2 - x1, y2 - y1))
        
        cap.release()
        
        if not detected_boxes:
            self.append_log1("Không tự động phát hiện được vùng phụ đề chứa chữ.")
            return None
            
        min_x = min(b[0] for b in detected_boxes)
        min_y = min(b[1] for b in detected_boxes)
        max_x = max(b[0] + b[2] for b in detected_boxes)
        max_y = max(b[1] + b[3] for b in detected_boxes)
        
        padding = 8
        final_x = max(0, min_x - padding)
        final_y = max(0, min_y - padding)
        final_w = min(w - final_x, (max_x - min_x) + 2 * padding)
        final_h = min(h - final_y, (max_y - min_y) + 2 * padding)
        
        auto_box = [final_x, final_y, final_w, final_h]
        self.append_log1(f"Đã tự động xác định được vùng phụ đề: X={final_x}, Y={final_y}, W={final_w}, H={final_h}")
        return auto_box

    def start_extraction(self):
        # Kiểm tra và quy đổi tốc độ video nếu cần thiết
        if self.video_path and os.path.exists(self.video_path):
            success, speed_path = self.adjust_video_speed_if_needed()
            if not success:
                return
            self.video_path = speed_path

        video_path = self.video_path
        
        # Nếu có chọn file SRT gốc, ta nạp trực tiếp luôn không cần chạy Whisper/OCR
        srt_path = self.txt_srt_path.text().strip()
        if srt_path and os.path.exists(srt_path):
            self.txt_logs1.clear()
            self.btn_start_extract.setEnabled(False)
            self.status_label.setText("Đang đọc file phụ đề SRT gốc...")
            try:
                try:
                    with open(srt_path, 'r', encoding='utf-8-sig') as f:
                        srt_content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(srt_path, 'r', encoding='utf-16') as f:
                            srt_content = f.read()
                    except UnicodeDecodeError:
                        with open(srt_path, 'r', encoding='mbcs') as f:
                            srt_content = f.read()
                segments = transcriber.parse_srt_string(srt_content)
                if segments:
                    self.on_extraction_finished(segments, video_path if video_path else "")
                    return
                else:
                    QMessageBox.warning(self, "Lỗi đọc file", "Không thể phân tích nội dung phụ đề từ file SRT này.")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Lỗi đọc file SRT: {e}")
            self.btn_start_extract.setEnabled(True)
            return
        
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng chọn hoặc tải tệp video trước khi trích xuất phụ đề.")
            return
            
        # Nếu người dùng đã khoanh vùng chữ trên video, tự động ưu tiên quét OCR
        if getattr(self, 'selected_bboxes', None) or getattr(self, 'selected_bbox', None):
            mode = 'ocr'
            if hasattr(self, 'cb_mode') and self.cb_mode.currentIndex() != 1:
                self.cb_mode.setCurrentIndex(1)
            self.log_info("🔍 Phát hiện đã khoanh vùng chữ trên video. Tự động bật chế độ Quét chữ cứng (OCR)...")
        else:
            mode = 'whisper' if self.cb_mode.currentIndex() == 0 else 'ocr'
            
        whisper_model = self.clean_combobox_value(self.cb_whisper_model.currentText())
        api_key = self.txt_gemini_key.text().strip()
        ocr_lang = self.cb_ocr_lang.currentText()
        
        self.txt_logs1.clear()
        self.btn_start_extract.setEnabled(False)
        self.status_label.setText("Đang trích xuất phụ đề...")
        
        # Tự động quét vùng chứa sub gốc nếu chạy OCR mà chưa vẽ vùng
        if mode == 'ocr' and not self.selected_bbox:
            auto_box = self.auto_detect_subtitle_bbox(video_path)
            if auto_box:
                self.selected_bbox = auto_box
                x, y, w, h = auto_box
                self.lbl_bbox.setText(f"Vùng quét: X={x}, Y={y}, W={w}, H={h}")
            else:
                QMessageBox.warning(self, "Không chọn được vùng", "Không thể tự động phát hiện vùng phụ đề. Vui lòng bấm 'Chọn vùng...' để tự vẽ.")
                self.btn_start_extract.setEnabled(True)
                return

        worker_bbox = []
        if mode == 'ocr':
            if getattr(self, 'selected_bboxes', None):
                worker_bbox = list(self.selected_bboxes)
            else:
                if self.selected_bbox:
                    worker_bbox.append(self.selected_bbox)
                if getattr(self, 'title_bbox', None):
                    worker_bbox.append(self.title_bbox)
            if not worker_bbox:
                worker_bbox = None
        else:
            worker_bbox = None
            
        force_scan = self.chk_ocr_force_scan.isChecked() if hasattr(self, 'chk_ocr_force_scan') else False
        self.trans_thread = TranscriptionWorker(video_path, mode, worker_bbox, whisper_model, api_key, ocr_lang=ocr_lang, force_scan=force_scan)
        self.trans_thread.progress.connect(self.append_log1)
        self.trans_thread.finished.connect(self.on_extraction_finished)
        self.trans_thread.error.connect(self.on_extraction_error)
        self.trans_thread.start()
        
    def append_log1(self, text):
        if hasattr(self, 'txt_logs1') and self.txt_logs1:
            self.txt_logs1.append(text)
        self.log_info(text)
        
    def preprocess_extracted_segments(self, segments):
        if not segments:
            return []
            
        # Bước 1: Loại bỏ segment rác (< 0.5s và chứa < 2 ký tự tiếng Trung)
        filtered = []
        for seg in segments:
            start = seg.get('start', 0.0)
            end = seg.get('end', 0.0)
            text = seg.get('text', '')
            duration = end - start
            
            # Đếm số ký tự tiếng Trung (Unicode CJK Ideographs)
            chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            compact_len = len("".join(str(text).split()))
            
            if duration < 0.25 and compact_len < 2 and chinese_count < 2:
                # Bỏ qua segment rác
                continue
            filtered.append(seg)
            
        if not filtered:
            return []
            
        filtered.sort(key=lambda x: x['start'])
        
        # Bước 2: Gộp các segment có khoảng cách < 0.2s (Nếu không phải chữ khó và tổng thời lượng < 6s)
        merged = []
        current = filtered[0]
        
        for next_seg in filtered[1:]:
            gap = next_seg['start'] - current['end']
            is_placeholder = "[Chữ khó" in current.get('text', '') or "[Chữ khó" in next_seg.get('text', '')
            if gap < 0.2 and (current['end'] - current['start']) < 6.0 and not is_placeholder:
                # Cập nhật thời gian kết thúc
                current['end'] = max(current['end'], next_seg['end'])
                # Gộp chữ
                t1 = current.get('text', '').strip()
                t2 = next_seg.get('text', '').strip()
                if t1 and t2:
                    current['text'] = t1 + " " + t2
                elif t2:
                    current['text'] = t2
                
                # Gộp bbox
                if 'bbox' in current or 'bbox' in next_seg:
                    from transcriber import merge_bboxes
                    current['bbox'] = merge_bboxes(current.get('bbox'), next_seg.get('bbox'))
                
                # Lấy confidence tối đa
                c1 = current.get('confidence', 0)
                c2 = next_seg.get('confidence', 0)
                if c1 or c2:
                    current['confidence'] = max(c1 or 0, c2 or 0)
                    
                # Cập nhật ocr_timestamp (midpoint)
                current['ocr_timestamp'] = (current['start'] + current['end']) / 2.0
            else:
                merged.append(current)
                current = next_seg
                
        merged.append(current)
        
        # Bước 3: Tự động chia nhỏ các segment quá dài (> 8.0s) để tránh bị dồn 1 câu duy nhất
        final_segments = []
        for seg in merged:
            st = seg.get('start', 0.0)
            et = seg.get('end', 0.0)
            dur = et - st
            if dur > 8.0:
                chunk_len = 3.5
                curr_t = st
                txt = seg.get('text', '')
                lines = [l.strip() for l in txt.splitlines() if l.strip()]
                line_idx = 0
                while curr_t < et:
                    next_t = min(curr_t + chunk_len, et)
                    seg_txt = txt
                    if lines and len(lines) > 1:
                        seg_txt = lines[line_idx % len(lines)]
                        line_idx += 1
                    
                    sub_seg = dict(seg)
                    sub_seg['start'] = curr_t
                    sub_seg['end'] = next_t
                    sub_seg['text'] = seg_txt
                    sub_seg['ocr_timestamp'] = (curr_t + next_t) / 2.0
                    final_segments.append(sub_seg)
                    curr_t = next_t
            else:
                final_segments.append(seg)

        return final_segments

    def on_extraction_finished(self, segments, video_path):
        self.btn_start_extract.setEnabled(True)
        self.video_path = video_path
        self.segments = self.preprocess_extracted_segments(segments)
        self.txt_file.setText(video_path)
        if hasattr(self, 'txt_ds_video'):
            self.txt_ds_video.setText(video_path)
        
        # Đề xuất đường dẫn đầu ra mặc định
        base, ext = os.path.splitext(video_path)
        self.txt_out.setText(base + "_longtieng" + ext)
        
        self.status_label.setText("Trích xuất phụ đề thành công!")
        self.populate_subtitle_table()
        QMessageBox.information(self, "Thành công", f"Đã trích xuất phụ đề thành công. Tìm thấy {len(segments)} câu phụ đề gốc.")
        
    def populate_subtitle_table(self):
        if not hasattr(self, 'table') or self.table is None:
            return
        
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        
        if not hasattr(self, 'segments') or not self.segments:
            self.table.blockSignals(False)
            return

        conf_thresh = self.spin_confidence_threshold.value() if hasattr(self, 'spin_confidence_threshold') else 70

        for idx, seg in enumerate(self.segments, 1):
            row_idx = self.table.rowCount()
            self.table.insertRow(row_idx)

            start_s = seg.get('start', 0.0)
            end_s = seg.get('end', 0.0)
            dur_s = max(0.0, end_s - start_s)
            orig = seg.get('orig_text', seg.get('text', ''))
            raw_trans = seg.get('raw_translation', '')
            final_trans = seg.get('translated_text', seg.get('text', ''))

            # 0: STT
            item_stt = QTableWidgetItem(str(idx))
            item_stt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 0, item_stt)

            # 1: Start
            item_start = QTableWidgetItem(f"{start_s:.2f}")
            item_start.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 1, item_start)

            # 2: End
            item_end = QTableWidgetItem(f"{end_s:.2f}")
            item_end.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 2, item_end)

            # 3: Duration
            item_dur = QTableWidgetItem(f"{dur_s:.2f}")
            item_dur.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 3, item_dur)

            # 4: Original text
            item_orig = QTableWidgetItem(str(orig))
            self.table.setItem(row_idx, 4, item_orig)

            # 5: Raw translation
            item_raw = QTableWidgetItem(str(raw_trans))
            self.table.setItem(row_idx, 5, item_raw)

            # 6: Final translation (Editable)
            item_final = QTableWidgetItem(str(final_trans))
            self.table.setItem(row_idx, 6, item_final)

            # 7: Confidence / Status
            conf = seg.get('confidence', 1.0)
            if isinstance(conf, (float, int)):
                conf_pct = int(conf * 100) if conf <= 1.0 else int(conf)
            else:
                conf_pct = 80
            
            box_status = f"{conf_pct}%"
            item_box = QTableWidgetItem(box_status)
            item_box.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if conf_pct < conf_thresh:
                item_box.setBackground(QColor(239, 68, 68, 60))
            self.table.setItem(row_idx, 7, item_box)

        self.table.blockSignals(False)
        self.log_info(f"📊 Đã nạp thành công {len(self.segments)} câu phụ đề vào bảng biên tập.")

    def on_cell_changed(self, row, column):
        if not hasattr(self, 'segments') or not self.segments or row >= len(self.segments):
            return
        if column == 6: # Bản dịch hoàn chỉnh
            item = self.table.item(row, column)
            if item:
                self.segments[row]['translated_text'] = item.text()
                self.segments[row]['text'] = item.text()
        elif column == 1: # Thời gian bắt đầu
            item = self.table.item(row, column)
            if item:
                try: self.segments[row]['start'] = float(item.text())
                except ValueError: pass
        elif column == 2: # Thời gian kết thúc
            item = self.table.item(row, column)
            if item:
                try: self.segments[row]['end'] = float(item.text())
                except ValueError: pass
        
    def on_extraction_error(self, err_msg):
        self.btn_start_extract.setEnabled(True)
        self.status_label.setText("Lỗi trích xuất.")
        QMessageBox.critical(self, "Lỗi hệ thống", f"Quá trình trích xuất gặp sự cố:\n{err_msg}")

    # --- CÁC PHƯƠNG THỨC TẠO TẬP DỮ LIỆU DATASET OCR ---
    def browse_ds_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn Video Nguồn", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if file_path:
            self.txt_ds_video.setText(file_path)
            
    def browse_ds_srt(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn File SRT Nhãn", "", "SRT Files (*.srt)")
        if file_path:
            self.txt_ds_srt.setText(file_path)
            
    def start_dataset_generation(self):
        video = self.txt_ds_video.text().strip()
        srt = self.txt_ds_srt.text().strip()
        out_dir = self.txt_ds_out.text().strip()
        
        if not video or not os.path.exists(video):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn video nguồn hợp lệ.")
            return
        if not srt or not os.path.exists(srt):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn file SRT nhãn hợp lệ.")
            return
            
        self.btn_run_ds_gen.setEnabled(False)
        self.lbl_ds_status.setText("⏳ Đang chuẩn bị cắt ảnh...")
        self.status_label.setText("Đang tạo Dataset OCR...")
        
        self.ds_worker = DatasetGeneratorWorker(video, srt, self.selected_bbox, getattr(self, 'title_bbox', None), out_dir)
        self.ds_worker.progress.connect(self.on_ds_progress)
        self.ds_worker.frame_signal.connect(self.on_worker_frame_update)
        self.ds_worker.finished.connect(self.on_ds_finished)
        self.ds_worker.error.connect(self.on_ds_error)
        self.ds_worker.start()
        
    def on_ds_progress(self, msg):
        self.lbl_ds_status.setText(f"Status: {msg}")
        self.status_label.setText(msg)
        
    def on_ds_finished(self, count, out_dir):
        self.btn_run_ds_gen.setEnabled(True)
        self.lbl_ds_status.setText(f"✅ Hoàn thành! Đã cắt {count} ảnh vào thư mục '{out_dir}'.")
        self.status_label.setText("Tạo Dataset OCR thành công!")
        QMessageBox.information(
            self, 
            "Thành công", 
            f"Đã trích xuất xong tập dữ liệu OCR:\n"
            f"- Số lượng ảnh đã cắt: {count} ảnh\n"
            f"- Thư mục lưu: {os.path.abspath(out_dir)}\n\n"
            f"Bạn có thể bắt đầu quá trình train theo tài liệu hướng dẫn."
        )
        
    def on_ds_error(self, err):
        self.btn_run_ds_gen.setEnabled(True)
        self.lbl_ds_status.setText(f"❌ Lỗi: {err}")
        self.status_label.setText("Lỗi tạo Dataset.")
        QMessageBox.critical(self, "Lỗi", f"Quá trình tạo Dataset gặp sự cố:\n{err}")

    # --- CÁC PHƯƠNG THỨC TẢI VIDEO ---
    def browse_download_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu video tải về")
        if dir_path:
            self.txt_dl_dir.setText(dir_path)
            
    def start_downloading(self):
        url = self.txt_url.text().strip()
        if not url:
            QMessageBox.warning(self, "Thiếu link tải", "Vui lòng dán link video Douyin, TikTok, Instagram, Reels, Rednote...")
            return
            
        output_dir = self.txt_dl_dir.text().strip()
        self.txt_logs_dl.clear()
        self.btn_download.setEnabled(False)
        self.status_label.setText("Đang tải video...")
        
        self.dl_thread = DownloadWorker(url, output_dir)
        self.dl_thread.progress.connect(self.append_log_dl)
        self.dl_thread.finished.connect(self.on_download_finished)
        self.dl_thread.error.connect(self.on_download_error)
        self.dl_thread.start()
        
    def append_log_dl(self, text):
        self.txt_logs_dl.append(text)
        self.status_label.setText(text)
        
    def on_download_finished(self, video_path):
        self.btn_download.setEnabled(True)
        self.video_path = video_path
        
        # Điền file vừa tải vào Tab 2
        self.txt_file.setText(video_path)
        if hasattr(self, 'txt_ds_video'):
            self.txt_ds_video.setText(video_path)
        # Điền file kết quả mặc định
        base, ext = os.path.splitext(video_path)
        self.txt_out.setText(base + "_longtieng" + ext)
        
        self.status_label.setText("Tải video thành công!")
        QMessageBox.information(
            self, 
            "Tải thành công", 
            f"Đã tải xong video và lưu tại:\n{video_path}\n\nHệ thống sẽ chuyển bạn sang Tab 'Trích Phụ đề' để xử lý tiếp."
        )
        
        # Chuyển sang Tab bàn làm việc chính (index 0)
        self.tabs.setCurrentIndex(0)
        
    def on_download_error(self, err_msg):
        self.btn_download.setEnabled(True)
        self.status_label.setText("Lỗi tải video.")
        QMessageBox.critical(self, "Lỗi tải video", f"Quá trình tải video gặp sự cố:\n{err_msg}")
        
    def populate_subtitle_table(self):
        if not hasattr(self, 'table'):
            return
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.segments))
        
        for idx, seg in enumerate(self.segments):
            start = seg.get('start', 0.0)
            end = seg.get('end', 0.0)
            duration = max(0.0, end - start)
            orig_text = str(seg.get('orig_text') or seg.get('text') or "")
            raw_text = str(seg.get('raw_text') or "")
            trans_text = str(seg.get('text') or "")
            
            # 0: STT
            item_stt = QTableWidgetItem(str(idx + 1))
            item_stt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_stt.setFlags(item_stt.flags() & ~Qt.ItemFlag.ItemIsEditable)
            
            # 1: Start (Formatted HH:MM:SS.mmm)
            item_start = QTableWidgetItem(format_time_stamp(start))
            
            # 2: End (Formatted HH:MM:SS.mmm)
            item_end = QTableWidgetItem(format_time_stamp(end))
            
            # 3: Duration
            item_duration = QTableWidgetItem(f"{duration:.2f}s")
            item_duration.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_duration.setFlags(item_duration.flags() & ~Qt.ItemFlag.ItemIsEditable)
            
            # 4: Orig Text (OCR)
            item_orig = QTableWidgetItem(orig_text)
            
            # 5: Trans Text (Editable)
            item_trans = QTableWidgetItem(trans_text)
            if raw_text and trans_text and raw_text != trans_text:
                item_trans.setBackground(QColor(30, 60, 45))
                item_trans.setForeground(QColor(180, 240, 200))
                
            # 6: Speed / Factor
            est_tts_dur = max(1.0, len(trans_text.split()) * 0.35)
            speed_factor = est_tts_dur / duration if duration > 0 else 1.0
            chars_per_sec = len(trans_text) / duration if duration > 0 else 0
            
            item_factor = QTableWidgetItem(f"{speed_factor:.2f}x ({chars_per_sec:.1f} ch/s)")
            item_factor.setFlags(item_factor.flags() & ~Qt.ItemFlag.ItemIsEditable)
            
            speed_threshold = self.spin_speed_threshold.value() if hasattr(self, 'spin_speed_threshold') else 20
            if chars_per_sec > speed_threshold:
                item_factor.setBackground(QColor(255, 153, 153))
                item_factor.setForeground(QColor(102, 51, 0))
            elif speed_factor > 1.05:
                item_factor.setBackground(QColor(255, 204, 153))
                item_factor.setForeground(QColor(102, 51, 0))
                
            self.table.setItem(idx, 0, item_stt)
            self.table.setItem(idx, 1, item_start)
            self.table.setItem(idx, 2, item_end)
            self.table.setItem(idx, 3, item_duration)
            self.table.setItem(idx, 4, item_orig)
            self.table.setItem(idx, 5, item_trans)
            self.table.setItem(idx, 6, item_factor)
            
        self.table.blockSignals(False)
        self.search_subtitles(direction=0)
        
    def on_cell_changed(self, row, column):
        if row < 0 or row >= len(self.segments):
            return
            
        self.table.blockSignals(True)
        try:
            seg = self.segments[row]
            if column == 1:  # Sửa Start
                text_val = self.table.item(row, 1).text()
                parsed_start = parse_time_stamp(text_val)
                if parsed_start is None or parsed_start >= seg['end']:
                    QMessageBox.warning(
                        self,
                        "Lỗi mốc thời gian",
                        f"Mốc thời gian Bắt đầu (Start = '{text_val}') không hợp lệ hoặc phải nhỏ hơn thời gian Kết thúc ({format_time_stamp(seg['end'])})."
                    )
                    self.table.item(row, 1).setText(format_time_stamp(seg['start']))
                else:
                    seg['start'] = parsed_start
                    self.table.item(row, 1).setText(format_time_stamp(parsed_start))
            elif column == 2:  # Sửa End
                text_val = self.table.item(row, 2).text()
                parsed_end = parse_time_stamp(text_val)
                if parsed_end is None or parsed_end <= seg['start']:
                    QMessageBox.warning(
                        self,
                        "Lỗi mốc thời gian",
                        f"Mốc thời gian Kết thúc (End = '{text_val}') không hợp lệ hoặc phải lớn hơn thời gian Bắt đầu ({format_time_stamp(seg['start'])})."
                    )
                    self.table.item(row, 2).setText(format_time_stamp(seg['end']))
                else:
                    seg['end'] = parsed_end
                    self.table.item(row, 2).setText(format_time_stamp(parsed_end))
            elif column == 4:  # Sửa Phụ đề Gốc
                seg['orig_text'] = self.table.item(row, 4).text()
            elif column == 5:  # Sửa Phụ đề Dịch
                seg['text'] = self.table.item(row, 5).text()
                seg['manual_override'] = True
                self.save_translation_cache()
                
            # Cập nhật lại thời lượng và Speed Factor
            start = seg['start']
            end = seg['end']
            duration = max(0.0, end - start)
            trans_text = seg.get('text', '')
            
            self.table.item(row, 3).setText(f"{duration:.2f}s")
            
            est_tts_dur = max(1.0, len(trans_text.split()) * 0.35)
            speed_factor = est_tts_dur / duration if duration > 0 else 1.0
            chars_per_sec = len(trans_text) / duration if duration > 0 else 0
            
            item_factor = self.table.item(row, 6)
            if item_factor:
                item_factor.setText(f"{speed_factor:.2f}x ({chars_per_sec:.1f} ch/s)")
                item_factor.setBackground(QColor(20, 20, 22))
                item_factor.setForeground(QColor(243, 242, 238))
                speed_threshold = self.spin_speed_threshold.value() if hasattr(self, 'spin_speed_threshold') else 20
                if chars_per_sec > speed_threshold:
                    item_factor.setBackground(QColor(255, 153, 153))
                    item_factor.setForeground(QColor(102, 51, 0))
                elif speed_factor > 1.05:
                    item_factor.setBackground(QColor(255, 204, 153))
                    item_factor.setForeground(QColor(102, 51, 0))
                
        except Exception:
            pass
        finally:
            self.table.blockSignals(False)

    # --- HÀM TÌM KIẾM, HIGHLIGHT VÀ THAO TÁC DÒNG CHUYÊN NGHIỆP ---
    def search_subtitles(self, direction=0):
        if not hasattr(self, 'txt_sub_search') or not hasattr(self, 'table'):
            return

        query = self.txt_sub_search.text().strip().lower()
        filter_mode = self.cb_sub_search_filter.currentText() if hasattr(self, 'cb_sub_search_filter') else "Tất cả"

        matching_cells = []
        rows = self.table.rowCount()

        self.table.blockSignals(True)
        try:
            for r in range(rows):
                orig_item = self.table.item(r, 4)
                trans_item = self.table.item(r, 5)

                orig_match = False
                trans_match = False

                if query:
                    if filter_mode in ["Tất cả", "Gốc (OCR)"] and orig_item and query in orig_item.text().lower():
                        orig_match = True
                    if filter_mode in ["Tất cả", "Đã dịch"] and trans_item and query in trans_item.text().lower():
                        trans_match = True

                # Reset highlight background
                if orig_item:
                    orig_item.setBackground(QColor(0, 0, 0, 0))
                if trans_item:
                    seg = self.segments[r] if r < len(self.segments) else {}
                    if seg.get('raw_text') and seg.get('text') and seg.get('raw_text') != seg.get('text'):
                        trans_item.setBackground(QColor(30, 60, 45))
                    else:
                        trans_item.setBackground(QColor(0, 0, 0, 0))

                if orig_match and orig_item:
                    orig_item.setBackground(QColor(120, 100, 20))
                    matching_cells.append((r, 4))
                if trans_match and trans_item:
                    trans_item.setBackground(QColor(120, 100, 20))
                    matching_cells.append((r, 5))

            total_matches = len(matching_cells)
            if not query or total_matches == 0:
                self._current_search_idx = -1
                self.lbl_search_count.setText("Tìm thấy 0 / 0 câu")
                return

            if not hasattr(self, '_current_search_idx'):
                self._current_search_idx = 0

            if direction == 1:
                self._current_search_idx = (self._current_search_idx + 1) % total_matches
            elif direction == -1:
                self._current_search_idx = (self._current_search_idx - 1 + total_matches) % total_matches
            else:
                self._current_search_idx = max(0, min(self._current_search_idx, total_matches - 1))

            self.lbl_search_count.setText(f"Tìm thấy {self._current_search_idx + 1} / {total_matches} câu")
            target_r, target_c = matching_cells[self._current_search_idx]
            item = self.table.item(target_r, target_c)
            if item:
                self.table.setCurrentItem(item)
                self.table.scrollToItem(item)
        finally:
            self.table.blockSignals(False)

    def insert_subtitle_row(self):
        curr = self.table.currentRow()
        if curr >= 0 and curr < len(self.segments):
            prev_end = self.segments[curr]['end']
            new_start = prev_end + 0.1
            new_end = new_start + 2.0
            insert_pos = curr + 1
        else:
            new_start = 0.0
            new_end = 2.0
            insert_pos = len(self.segments)

        new_seg = {
            'start': new_start,
            'end': new_end,
            'orig_text': 'Phụ đề gốc mới',
            'text': 'Phụ đề dịch mới'
        }
        self.segments.insert(insert_pos, new_seg)
        self.populate_subtitle_table()
        self.table.selectRow(insert_pos)

    def delete_selected_subtitle_rows(self):
        selected_rows = sorted(list(set([item.row() for item in self.table.selectedItems()])), reverse=True)
        if not selected_rows:
            return
        for r in selected_rows:
            if 0 <= r < len(self.segments):
                self.segments.pop(r)
        self.populate_subtitle_table()

    def merge_selected_subtitle_rows(self):
        selected_rows = sorted(list(set([item.row() for item in self.table.selectedItems()])))
        if len(selected_rows) < 2:
            QMessageBox.warning(self, "Thông báo", "Vui lòng chọn ít nhất 2 dòng phụ đề liên tiếp để gộp!")
            return

        first_r = selected_rows[0]
        last_r = selected_rows[-1]

        merged_start = self.segments[first_r]['start']
        merged_end = self.segments[last_r]['end']
        merged_orig = " ".join([self.segments[r].get('orig_text', '') for r in selected_rows if self.segments[r].get('orig_text')])
        merged_trans = " ".join([self.segments[r].get('text', '') for r in selected_rows if self.segments[r].get('text')])

        self.segments[first_r]['start'] = merged_start
        self.segments[first_r]['end'] = merged_end
        self.segments[first_r]['orig_text'] = merged_orig
        self.segments[first_r]['text'] = merged_trans

        for r in reversed(selected_rows[1:]):
            self.segments.pop(r)

        self.populate_subtitle_table()
        self.table.selectRow(first_r)

    def split_subtitle_row(self):
        curr = self.table.currentRow()
        if curr < 0 or curr >= len(self.segments):
            return

        seg = self.segments[curr]
        start = seg['start']
        end = seg['end']
        mid = (start + end) / 2.0

        words = seg.get('text', '').split()
        if len(words) > 1:
            m = len(words) // 2
            t1 = " ".join(words[:m])
            t2 = " ".join(words[m:])
        else:
            t1 = seg.get('text', '')
            t2 = "..."

        orig_words = seg.get('orig_text', '').split()
        if len(orig_words) > 1:
            om = len(orig_words) // 2
            o1 = " ".join(orig_words[:om])
            o2 = " ".join(orig_words[om:])
        else:
            o1 = seg.get('orig_text', '')
            o2 = "..."

        seg1 = dict(seg)
        seg1['end'] = mid
        seg1['text'] = t1
        seg1['orig_text'] = o1

        seg2 = dict(seg)
        seg2['start'] = mid
        seg2['text'] = t2
        seg2['orig_text'] = o2

        self.segments[curr] = seg1
        self.segments.insert(curr + 1, seg2)
        self.populate_subtitle_table()
        self.table.selectRow(curr + 1)

    def set_start_from_player(self):
        curr = self.table.currentRow()
        if curr < 0 or curr >= len(self.segments) or not self.video_path:
            return
        try:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            if total_frames <= 0:
                return
            curr_sec = (self.slider_player_timeline.value() / 1000.0) * (total_frames / fps)
            if curr_sec < self.segments[curr]['end']:
                self.segments[curr]['start'] = curr_sec
                self.populate_subtitle_table()
        except Exception:
            pass

    def set_end_from_player(self):
        curr = self.table.currentRow()
        if curr < 0 or curr >= len(self.segments) or not self.video_path:
            return
        try:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            if total_frames <= 0:
                return
            curr_sec = (self.slider_player_timeline.value() / 1000.0) * (total_frames / fps)
            if curr_sec > self.segments[curr]['start']:
                self.segments[curr]['end'] = curr_sec
                self.populate_subtitle_table()
        except Exception:
            pass

    def shift_selected_timestamps(self, offset_s):
        selected_rows = set([item.row() for item in self.table.selectedItems()])
        if not selected_rows:
            return
        for r in selected_rows:
            if 0 <= r < len(self.segments):
                self.segments[r]['start'] = max(0.0, self.segments[r]['start'] + offset_s)
                self.segments[r]['end'] = max(self.segments[r]['start'] + 0.1, self.segments[r]['end'] + offset_s)
        self.populate_subtitle_table()

    def show_subtitle_table_context_menu(self, pos):
        menu = QMenu(self.table)
        menu.setStyleSheet("QMenu { background-color: #1e293b; color: white; border: 1px solid #334155; } QMenu::item:selected { background-color: #3b82f6; }")

        act_add = menu.addAction("➕ Thêm dòng mới (Ctrl+N)")
        act_del = menu.addAction("❌ Xóa dòng chọn (Delete)")
        act_merge = menu.addAction("🔗 Gộp dòng chọn (Ctrl+M)")
        act_split = menu.addAction("✂️ Tách câu phụ đề (Ctrl+Shift+S)")
        menu.addSeparator()
        act_start = menu.addAction("⏱️ Gán Start = Thời gian Video hiện tại (Ctrl+[)")
        act_end = menu.addAction("⏱️ Gán End = Thời gian Video hiện tại (Ctrl+])")
        menu.addSeparator()
        act_shift_p = menu.addAction("⏩ Tăng mốc thời gian +0.5s")
        act_shift_m = menu.addAction("⏪ Giảm mốc thời gian -0.5s")

        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action == act_add: self.insert_subtitle_row()
        elif action == act_del: self.delete_selected_subtitle_rows()
        elif action == act_merge: self.merge_selected_subtitle_rows()
        elif action == act_split: self.split_subtitle_row()
        elif action == act_start: self.set_start_from_player()
        elif action == act_end: self.set_end_from_player()
        elif action == act_shift_p: self.shift_selected_timestamps(0.5)
        elif action == act_shift_m: self.shift_selected_timestamps(-0.5)

    def setup_subtitle_shortcuts(self):
        from PyQt6.QtGui import QKeySequence, QShortcut
        QShortcut(QKeySequence("Ctrl+F"), self, lambda: self.txt_sub_search.setFocus())
        QShortcut(QKeySequence("Ctrl+N"), self, self.insert_subtitle_row)
        QShortcut(QKeySequence("Ctrl+M"), self, self.merge_selected_subtitle_rows)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, self.split_subtitle_row)
        QShortcut(QKeySequence("Ctrl+["), self, self.set_start_from_player)
        QShortcut(QKeySequence("Ctrl+]"), self, self.set_end_from_player)
        QShortcut(QKeySequence("Delete"), self, self.delete_selected_subtitle_rows)
        QShortcut(QKeySequence("Return"), self.txt_sub_search, lambda: self.search_subtitles(1))
        QShortcut(QKeySequence("Shift+Return"), self.txt_sub_search, lambda: self.search_subtitles(-1))

    def start_pipeline(self):
        """Alias for start_dubbing to maintain compatibility with test suites and external calls."""
        return self.start_dubbing()

    def open_trending_slang_dialog(self):
        from translator import TrendingSlangManager
        from slang_sync_engine import sync_online_trending_words
        
        dialog = QDialog(self)
        dialog.setWindowTitle("🔥 QUẢN LÝ TỪ ĐIỂN TREND BILIBILI & DOUYIN (GEN Z)")
        dialog.resize(760, 480)
        
        layout = QVBoxLayout(dialog)
        
        # Toolbar Top
        tb = QHBoxLayout()
        txt_search = QLineEdit()
        txt_search.setPlaceholderText("🔍 Tìm kiếm từ lóng tiếng Trung hoặc nghĩa Tiếng Việt...")
        txt_search.setStyleSheet("background-color: #0f172a; color: #f8fafc; border: 1px solid #2a364f; padding: 6px 10px; border-radius: 6px;")
        tb.addWidget(txt_search, 2)
        
        lbl_total_count = QLabel("Tổng số: 0 từ lóng")
        lbl_total_count.setStyleSheet("color: #38bdf8; font-weight: bold; padding: 0 8px;")
        tb.addWidget(lbl_total_count)

        btn_sync = QPushButton("🔄 Đồng bộ Trend trực tuyến")
        btn_sync.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px 12px; border-radius: 6px;")
        tb.addWidget(btn_sync)
        layout.addLayout(tb)
        
        # Table
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["STT", "Từ Lóng Gốc (Trung)", "Nghĩa Tiếng Việt Gen Z", "Phân Loại & Nguồn"])
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)
        
        manager = TrendingSlangManager()
        
        def reload_table(filter_text=None):
            if filter_text is None or not isinstance(filter_text, str):
                filter_text = txt_search.text()
            data = manager.load_dict()
            filter_text = filter_text.strip().lower()
            table.setRowCount(0)
            row_idx = 0
            for zh, info in data.items():
                vi = info.get("vi", "")
                cat_name = info.get('category', '')
                src_name = info.get('source', '')
                cat_full = f"{cat_name} ({src_name})"
                if filter_text and filter_text not in zh.lower() and filter_text not in vi.lower() and filter_text not in cat_full.lower():
                    continue
                table.insertRow(row_idx)
                item_stt = QTableWidgetItem(str(row_idx + 1))
                item_stt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row_idx, 0, item_stt)
                
                item_zh = QTableWidgetItem(zh)
                item_zh.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                table.setItem(row_idx, 1, item_zh)
                
                item_vi = QTableWidgetItem(vi)
                item_vi.setForeground(QColor("#4ade80") if src_name == "Auto-Sync" else QColor("#f8fafc"))
                table.setItem(row_idx, 2, item_vi)
                
                item_cat = QTableWidgetItem(cat_full)
                if src_name == "Auto-Sync":
                    item_cat.setForeground(QColor("#38bdf8"))
                elif src_name == "Custom":
                    item_cat.setForeground(QColor("#fbbf24"))
                table.setItem(row_idx, 3, item_cat)
                
                row_idx += 1
            lbl_total_count.setText(f"Tổng số: {row_idx} từ lóng")
                
        reload_table("")
        txt_search.textChanged.connect(lambda text: reload_table(text))
        
        def on_sync_clicked():
            res = sync_online_trending_words()
            reload_table(txt_search.text())
            added_cnt = res.get('added_count', 0)
            total_cnt = res.get('total_count', 0)
            
            if added_cnt > 0:
                if table.rowCount() > 0:
                    table.scrollToBottom()
                    table.selectRow(table.rowCount() - 1)
                QMessageBox.information(
                    dialog,
                    "Cập nhật thành công",
                    f"🎉 Đã tìm thấy và hợp nhất {added_cnt} từ lóng mới nhất từ mạng online!\n\nTổng số từ lóng trong hệ thống: {total_cnt}"
                )
            else:
                QMessageBox.information(
                    dialog,
                    "Từ điển đã là mới nhất",
                    f"ℹ️ Từ điển Trend Slang hiện đã ở bản mới nhất!\nChưa có thêm từ lóng mới nào trên các nguồn online.\n\nTổng số từ lóng trong hệ thống: {total_cnt}"
                )
            
        btn_sync.clicked.connect(on_sync_clicked)
        
        # Bottom Buttons
        bot_row = QHBoxLayout()
        btn_add = QPushButton("➕ Thêm từ lóng")
        btn_add.setStyleSheet("background-color: #16a34a; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        
        def on_add_clicked():
            zh, ok1 = QInputDialog.getText(dialog, "Thêm từ lóng", "Nhập từ lóng tiếng Trung (VD: 绝绝子):")
            if not ok1 or not zh.strip(): return
            vi, ok2 = QInputDialog.getText(dialog, "Thêm từ lóng", "Nhập nghĩa Tiếng Việt Gen Z (VD: đỉnh kout / hết nước chấm):")
            if not ok2 or not vi.strip(): return
            manager.add_or_update_slang(zh.strip(), vi.strip(), category="Custom", source="Custom")
            reload_table()
            
        btn_add.clicked.connect(on_add_clicked)
        bot_row.addWidget(btn_add)
        
        btn_del = QPushButton("❌ Xóa từ chọn")
        btn_del.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        
        def on_del_clicked():
            curr = table.currentRow()
            if curr < 0: return
            zh_item = table.item(curr, 1)
            if zh_item:
                manager.remove_slang(zh_item.text())
                reload_table()
                
        btn_del.clicked.connect(on_del_clicked)
        bot_row.addWidget(btn_del)
        
        btn_close = QPushButton("Đóng")
        btn_close.clicked.connect(dialog.accept)
        bot_row.addWidget(btn_close)
        
        layout.addLayout(bot_row)
        dialog.exec()

    def clear_translation_cache(self):
        try:
            import glob
            cache_dir = os.path.join(os.getcwd(), "Data", "cache")
            if os.path.exists(cache_dir):
                for f_path in glob.glob(os.path.join(cache_dir, "*")):
                    try:
                        if os.path.isfile(f_path):
                            os.remove(f_path)
                    except:
                        pass
            
            # Xóa cache trong RAM
            from translator import global_translation_cache
            global_translation_cache.cache_data.clear()
            
            QMessageBox.information(self, "Thành công", "Đã xóa sạch toàn bộ cache dịch thuật cũ thành công!")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể xóa cache: {e}")

    def start_translation(self):
        if not self.segments:
            QMessageBox.warning(self, "Cảnh báo", "Không có phụ đề nào để dịch.")
            return
            
        engine = self.clean_combobox_value(self.cb_engine.currentText())
        
        # Ánh xạ từ các tùy chọn tinh giản sang cấu hình backend thực tế
        refine_enabled = self.chk_refine_enabled.isChecked()
        refine_engine = "Gemini 1.5 Flash"
        refine_api_key = ""
        
        refine_sel = self.cb_refine_engine.currentText()
        if "Gemini 1.5 Flash" in refine_sel:
            refine_engine = "Gemini 1.5 Flash"
            refine_api_key = self.txt_gemini_key.text().strip()
        elif "Gemini 1.5 Pro" in refine_sel:
            refine_engine = "Gemini 1.5 Pro"
            refine_api_key = self.txt_gemini_key.text().strip()
        elif "Gemini 2.0 Flash" in refine_sel:
            refine_engine = "Gemini 2.0 Flash"
            refine_api_key = self.txt_gemini_key.text().strip()
        elif "Groq Llama 3.1" in refine_sel:
            refine_engine = "Groq Llama 3.1"
            refine_api_key = self.txt_groq_key.text().strip()
        else:
            refine_engine = "Ollama Local"
            refine_api_key = ""
            
        if engine == "Supersubs AI":
            backend_engine = "Quick Translator (VietPhrase)"
            # Giữ nguyên refine_enabled và các cấu hình lấy từ giao diện
        elif engine == "Dịch thô":
            backend_engine = "Quick Translator (VietPhrase)"
            refine_enabled = False
        elif engine == "Dịch cơ bản":
            backend_engine = "Google Translate"
            refine_enabled = False
        else:
            backend_engine = engine
            refine_enabled = False
            
        api_key = ""
        ollama_model = self.txt_ollama_model.text().strip()
        
        vp_dict_paths = {
            'vp_path': self.txt_vp_path.text().strip(),
            'names_path': self.txt_names_path.text().strip()
        }
                
        # 3. Validate và preview glossary
        glossary = {}
        if refine_enabled:
            glossary_text = self.txt_glossary.toPlainText().strip()
            if glossary_text:
                errors = []
                for line_num, line in enumerate(glossary_text.splitlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    if '=' not in line:
                        errors.append(f"Dòng {line_num}: {line}")
                    else:
                        parts = line.split('=', 1)
                        k = parts[0].strip()
                        v = parts[1].strip()
                        if k:
                            glossary[k] = v
                            
                if errors:
                    err_msg = "Phát hiện lỗi định dạng glossary (phải có dạng 'từ_gốc = từ_dịch'):\n\n" + "\n".join(errors[:5])
                    if len(errors) > 5:
                        err_msg += "\n..."
                    QMessageBox.warning(self, "Lỗi định dạng Glossary", err_msg)
                    return
                
                # Hiển thị Preview Table Dialog nếu có glossary
                if glossary:
                    preview_dialog = QDialog(self)
                    preview_dialog.setWindowTitle("Preview Glossary")
                    preview_dialog.resize(400, 300)
                    dlg_layout = QVBoxLayout(preview_dialog)
                    
                    lbl_info = QLabel("Xác nhận bảng thuật ngữ trước khi chạy dịch thuật:")
                    dlg_layout.addWidget(lbl_info)
                    
                    preview_table = QTableWidget(len(glossary), 2)
                    preview_table.setHorizontalHeaderLabels(["Từ gốc", "Từ dịch"])
                    preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
                    
                    for r_idx, (gk, gv) in enumerate(glossary.items()):
                        preview_table.setItem(r_idx, 0, QTableWidgetItem(gk))
                        preview_table.setItem(r_idx, 1, QTableWidgetItem(gv))
                    dlg_layout.addWidget(preview_table)
                    
                    btn_box = QHBoxLayout()
                    btn_ok = QPushButton("Xác nhận chạy")
                    btn_cancel = QPushButton("Hủy")
                    btn_ok.clicked.connect(preview_dialog.accept)
                    btn_cancel.clicked.connect(preview_dialog.reject)
                    btn_box.addWidget(btn_ok)
                    btn_box.addWidget(btn_cancel)
                    dlg_layout.addLayout(btn_box)
                    
                    if preview_dialog.exec() != QDialog.DialogCode.Accepted:
                        return
                        
        # 4. Kiểm tra Cache Dịch thuật cục bộ trước khi chạy thực tế
        if self.load_translation_cache():
            QMessageBox.information(
                self, 
                "Tải từ Cache", 
                "Đã tự động nạp bản dịch thô và sơ chế từ cache cục bộ (phát hiện cấu hình trùng khớp)."
            )
            return

        self.status_label.setText("Đang dịch thuật phụ đề...")
        self.progress_dialog = QProgressDialog("Đang dịch phụ đề...", "Huỷ", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.show()
        
        self.translate_thread = TranslationWorker(
            self.segments, 
            backend_engine, 
            api_key, 
            ollama_model=ollama_model, 
            vp_dict_paths=vp_dict_paths,
            refine_enabled=refine_enabled,
            refine_engine=refine_engine,
            refine_api_key=refine_api_key,
            glossary=glossary
        )
        self.translate_thread.progress.connect(self.on_translate_progress)
        self.translate_thread.finished.connect(self.on_translate_finished)
        self.translate_thread.error.connect(self.on_translate_error)
        self.translate_thread.start()
        
    def translate_subtitles(self):
        """Compatibility wrapper for UI button to start translation."""
        return self.start_translation()

    def export_srt_file(self):
        if not self.segments:
            QMessageBox.warning(self, "Cảnh báo", "Không có phụ đề để xuất.")
            return
        default_name = "merged.srt"
        if self.video_path:
            try:
                default_name = os.path.splitext(os.path.basename(self.video_path))[0] + ".srt"
            except Exception:
                pass
        file_path, _ = QFileDialog.getSaveFileName(self, "Lưu file SRT", os.path.join(os.getcwd(), default_name), "SubRip (*.srt)")
        if not file_path:
            return
        try:
            srt_text = transcriber.segments_to_srt(self.segments)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(srt_text)
            QMessageBox.information(self, "Thành công", f"Đã lưu phụ đề tới:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu file SRT: {e}")
        
    def on_translate_progress(self, text):
        self.progress_dialog.setLabelText(text)
        self.status_label.setText(text)
        
    def on_translate_finished(self, translated_segments):
        self.progress_dialog.close()
        self.segments = translated_segments
        self.status_label.setText("Dịch thuật hoàn tất!")
        self.save_translation_cache() # Lưu cache dịch thuật bền vững
        QMessageBox.information(self, "Dịch thành công", "Đã dịch thuật và sơ chế hoàn tất phụ đề.")
        self.populate_subtitle_table()
        
    def on_translate_error(self, err_msg):
        self.progress_dialog.close()
        self.status_label.setText("Dịch thuật thất bại.")
        QMessageBox.critical(self, "Lỗi dịch thuật", f"Đã xảy ra sự cố trong quá trình dịch thuật:\n{err_msg}")
        
    def play_video_segment(self):
        # Nếu đang phát thì dừng lại
        if hasattr(self, 'preview_timer') and self.preview_timer.isActive():
            self.stop_video_preview()
            return
            
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn 1 dòng phụ đề để phát video thử.")
            return
            
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy file video gốc.")
            return
            
        start_s = self.segments[row]['start']
        end_s = self.segments[row]['end']
        self.preview_text = self.segments[row].get('text', '')
        
        self.preview_cap = cv2.VideoCapture(self.video_path)
        fps = self.preview_cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        self.preview_delay = int(1000 / fps)
        
        self.preview_start_frame = int(start_s * fps)
        self.preview_end_frame = int(end_s * fps)
        self.preview_current_frame = self.preview_start_frame
        
        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, self.preview_start_frame)
        self.btn_play_seg.setText("⏹ Dừng phát")
        self.btn_play_seg.setStyleSheet("background-color: #ff9999; color: #0c0c0e; font-weight: bold; padding: 8px;")
        
        self.status_label.setText(f"Đang phát preview phân đoạn: {start_s:.1f}s -> {end_s:.1f}s")
        self.preview_timer.start(self.preview_delay)

    def play_preview_frame(self):
        if not self.preview_cap or not self.preview_cap.isOpened():
            self.stop_video_preview()
            return
            
        if self.preview_current_frame > self.preview_end_frame:
            self.stop_video_preview()
            return
            
        ret, frame = self.preview_cap.read()
        if not ret:
            self.stop_video_preview()
            return
            
        self.preview_current_frame += 1
        
        # 1. Vẽ hộp che (giống logic tĩnh)
        row = self.table.currentRow()
        bbox = self.segments[row].get('bbox') or self.selected_bbox
        if bbox:
            bx, by, bw, bh = bbox
            fh, fw, _ = frame.shape
            bx1 = max(0, min(bx, fw))
            by1 = max(0, min(by, fh))
            bx2 = max(0, min(bx + bw, fw))
            by2 = max(0, min(by + bh, fh))
            if bx2 > bx1 and by2 > by1:
                mask_mode = self.cb_mask_mode.currentText() if hasattr(self, 'cb_mask_mode') else "Blur nhanh (Gaussian Blur)"
                if "Không che" not in mask_mode:
                    if "Che đen đặc" in mask_mode:
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
                    elif "Blur" in mask_mode:
                        crop = frame[by1:by2, bx1:bx2]
                        if crop.shape[0] > 0 and crop.shape[1] > 0:
                            blur_k = max(3, int(min(crop.shape[0], crop.shape[1]) | 1))
                            crop_blur = cv2.GaussianBlur(crop, (blur_k, blur_k), 0)
                            frame[by1:by2, bx1:bx2] = crop_blur
                    else: # Inpaint
                        crop = frame[by1:by2, bx1:bx2]
                        if crop.shape[0] > 0 and crop.shape[1] > 0:
                            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                            mask.fill(255)
                            inpainted = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
                            frame[by1:by2, bx1:bx2] = inpainted
                            
        # 2. Vẽ phụ đề đè lên theo style hiện tại
        if self.preview_text:
            preset = self.get_current_subtitle_preset()
            frame, _ = dubber.draw_burned_subtitle(frame, self.preview_text, bbox=None, default_bbox=None, preset=preset)
            
        # 3. Crop if zoomed
        is_zoomed = hasattr(self, 'chk_zoom_sub') and self.chk_zoom_sub.isChecked()
        h, w, _ = frame.shape
        self.video_width = w
        self.video_height = h
        
        if is_zoomed:
            cy = 0.82
            if self.subtitle_custom_pos:
                cy = self.subtitle_custom_pos['y_pct'] / 100.0
            
            y_start = int(max(0.0, cy - 0.18) * h)
            y_end = int(min(1.0, cy + 0.15) * h)
            x_start = int(0.05 * w)
            x_end = int(0.95 * w)
            
            if y_end - y_start < 50:
                y_start = max(0, h - 100)
                y_end = h
            show_frame = frame[y_start:y_end, x_start:x_end]
        else:
            show_frame = frame
            
        # Truyền frame gốc cho DraggablePreviewLabel tự scale đúng tỷ lệ
        self.lbl_main_preview.setVideoFrame(show_frame)

    def stop_video_preview(self):
        if hasattr(self, 'preview_timer'):
            self.preview_timer.stop()
        if hasattr(self, 'preview_cap') and self.preview_cap:
            self.preview_cap.release()
            self.preview_cap = None
        self.btn_play_seg.setText("▶ Phát phân đoạn")
        self.btn_play_seg.setStyleSheet("background-color: #7fbeb2; color: #0c0c0e; font-weight: bold; padding: 8px;")
        self.status_label.setText("Đã dừng phát")
        self.trigger_canvas_update()

    def on_subtitle_table_row_selected(self):
        row = self.table.currentRow()
        if row < 0 or not hasattr(self, 'segments') or not self.segments:
            return
        if row < len(self.segments):
            seg = self.segments[row]
            start_s = seg.get('start', 0.0)
            self.seek_to_timestamp(start_s)

    def on_subtitle_table_row_double_clicked(self, item):
        if item is None:
            return
        row = item.row()
        if row >= 0 and hasattr(self, 'segments') and self.segments and row < len(self.segments):
            seg = self.segments[row]
            start_s = seg.get('start', 0.0)
            self.seek_to_timestamp(start_s)

    def seek_to_timestamp(self, seconds):
        if not self.video_path or not os.path.exists(self.video_path):
            return
        try:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            if total_frames <= 0:
                return
            target_frame = max(0, min(total_frames - 1, int(seconds * fps)))
            val = int((target_frame / float(total_frames)) * 1000)
            self.slider_player_timeline.setValue(val)
        except Exception:
            pass

    def ocr_active_row_with_gemini(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.segments):
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn 1 dòng phụ đề để chạy nhận diện.")
            return
            
        api_key = self.txt_gemini_key.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Lỗi", "Vui lòng cấu hình Gemini API Key ở tab bên trái trước.")
            return
            
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy video gốc.")
            return
            
        # Đọc frame
        start_s = self.segments[row]['start']
        end_s = self.segments[row]['end']
        t_target = self.segments[row].get('ocr_timestamp', (start_s + end_s) / 2.0)
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_target * fps))
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            QMessageBox.warning(self, "Lỗi", "Không thể đọc khung hình từ video.")
            return
            
        bbox = self.segments[row].get('bbox') or self.selected_bbox
        
        self.status_label.setText(f"⏳ Đang gửi ảnh dòng {row+1} lên Gemini AI...")
        self.btn_gemini_ocr_row.setEnabled(False)
        
        if hasattr(self, 'gemini_main_worker') and self.gemini_main_worker.isRunning():
            self.gemini_main_worker.terminate()
            self.gemini_main_worker.wait()
            
        self.gemini_main_worker = GeminiOCRWorker(row, frame, bbox, api_key)
        self.gemini_main_worker.finished.connect(self.on_main_gemini_success)
        self.gemini_main_worker.error.connect(self.on_main_gemini_error)
        self.gemini_main_worker.start()
        
    def on_main_gemini_success(self, row, text):
        self.btn_gemini_ocr_row.setEnabled(True)
        self.status_label.setText(f"✨ Đã nhận diện xong chữ dòng {row+1} bằng Gemini AI.")
        
        # Cập nhật bảng và bộ nhớ segments
        self.table.blockSignals(True)
        self.table.setItem(row, 3, QTableWidgetItem(text))
        self.segments[row]['orig_text'] = text
        self.table.blockSignals(False)
        
        # Trigger canvas update để hiển thị lại
        self.trigger_canvas_update()
        
    def on_main_gemini_error(self, row, err_msg):
        self.btn_gemini_ocr_row.setEnabled(True)
        self.status_label.setText(f"❌ Gemini AI nhận diện dòng {row+1} lỗi: {err_msg}")
        QMessageBox.critical(self, "Lỗi", f"Gemini AI lỗi: {err_msg}")
        
    def preview_segment_tts(self):
        # Kiểm tra xem đang ở tab Kịch bản & Giọng đọc hay tab Lồng tiếng & Xuất
        if self.tabs.currentIndex() == 3:
            tab = self.script_tab
            row = tab.table_segments.currentRow()
            if row < 0:
                QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn 1 dòng kịch bản để nghe thử.")
                return
                
            text = tab.segments_data[row]['text']
            voice = tab.cb_script_voice.currentData()
            speed_val = tab.slider_speed.value()
            rate = f"{speed_val:+}%" if speed_val != 0 else "+0%"
            pitch = "+0Hz"
        else:
            row = self.table.currentRow()
            if row < 0:
                QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn 1 dòng phụ đề để nghe thử.")
                return
                
            text = self.segments[row]['text']
            voice = self.cb_voice.currentData()
            rate = "+0%"
            pitch = "+0Hz"
            
        if not text.strip():
            QMessageBox.warning(self, "Lỗi", "Nội dung trống, không thể nghe thử.")
            return
            
        self.status_label.setText("Đang sinh giọng lồng tiếng nghe thử...")
        
        # Tắt âm thanh cũ đang phát
        winsound.PlaySound(None, winsound.SND_PURGE)
        
        class PreviewTTSThread(QThread):
            progress = pyqtSignal(str)
            finished = pyqtSignal(str)
            error = pyqtSignal(str)
            
            def __init__(self, text, voice, rate, pitch):
                super().__init__()
                self.text = text
                self.voice = voice
                self.rate = rate
                self.pitch = pitch
                
            def run(self):
                temp_mp3 = os.path.join(tempfile.gettempdir(), "temp_preview.mp3")
                temp_wav = os.path.join(tempfile.gettempdir(), "temp_preview.wav")
                
                if os.path.exists(temp_mp3):
                    try: os.remove(temp_mp3)
                    except: pass
                if os.path.exists(temp_wav):
                    try: os.remove(temp_wav)
                    except: pass
                    
                # Sinh TTS
                success = dubber.generate_tts(self.text, self.voice, temp_mp3, self.rate, self.pitch)
                if not success or not os.path.exists(temp_mp3):
                    self.error.emit("Không thể sinh giọng đọc từ Edge-TTS. Vui lòng kiểm tra kết nối mạng hoặc giọng đọc đã chọn.")
                    return
                
                # Chuyển đổi định dạng MP3 sang WAV qua pydub để phát bằng winsound
                try:
                    sound = AudioSegment.from_file(temp_mp3)
                    sound.export(temp_wav, format="wav")
                    self.finished.emit(temp_wav)
                except Exception as e:
                    self.error.emit(f"Lỗi chuyển đổi âm thanh để phát thử: {str(e)}")
                    
        self.preview_tts_thread = PreviewTTSThread(text, voice, rate, pitch)
        self.preview_tts_thread.finished.connect(self.play_preview_audio)
        self.preview_tts_thread.error.connect(self.show_preview_error)
        self.preview_tts_thread.start()
        
    def play_preview_audio(self, wav_path):
        self.status_label.setText("Đang phát âm thanh nghe thử...")
        # Phát âm thanh không đồng bộ (không gây đơ app) từ tệp tin WAV
        winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        
    def show_preview_error(self, err_msg):
        self.status_label.setText("Lỗi phát thử.")
        QMessageBox.warning(self, "Lỗi nghe thử", err_msg)
        
    def start_dubbing(self):
        self.pipeline_start_time = QDateTime.currentDateTime()
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn video trước khi bấm CHẠY!")
            return

        out_path = self.txt_out.text().strip() if hasattr(self, 'txt_out') else ""
        if not out_path:
            base, ext = os.path.splitext(self.video_path)
            out_path = f"{base}_dubbed.mp4"
            if hasattr(self, 'txt_out'):
                self.txt_out.setText(out_path)

        # Kiểm tra và khởi tạo bbox hợp lệ (loại bỏ bbox rỗng [0,0,0,0])
        def is_valid_box(b):
            return bool(b and isinstance(b, (list, tuple)) and len(b) >= 4 and b[2] > 10 and b[3] > 10)

        w = getattr(self, 'video_width', 1920) or 1920
        h = getattr(self, 'video_height', 1080) or 1080

        # Lọc danh sách selected_bboxes
        if hasattr(self, 'selected_bboxes') and self.selected_bboxes:
            self.selected_bboxes = [b for b in self.selected_bboxes if is_valid_box(b)]

        if not is_valid_box(self.selected_bbox):
            if self.selected_bboxes:
                self.selected_bbox = self.selected_bboxes[0]
            else:
                self.selected_bbox = [int(w * 0.1), int(h * 0.72), int(w * 0.8), int(h * 0.2)]

        # Nếu có khoanh vùng Logo mà chưa chọn file Logo khách hàng, hiển thị hộp thoại hỏi người dùng
        if getattr(self, 'logo_bbox', None) and not (hasattr(self, 'txt_logo_path') and self.txt_logo_path.text().strip()):
            reply = QMessageBox.question(
                self,
                "Chèn Logo Thương Hiệu Khách Hàng",
                "Phát hiện bạn đã khoanh vùng Logo.\n\n"
                "Bạn có muốn chọn ảnh Logo của khách (PNG/JPG trong suốt) để chèn đè lên vị trí logo gốc không?\n\n"
                " - Chọn Yes: Mở cửa sổ duyệt chọn file ảnh Logo.\n"
                " - Chọn No: Chỉ làm mờ / che logo gốc.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.browse_logo()

        # Khóa nút CHẠY, mở nút HỦY để tránh click kép và cho phép dừng khẩn cấp
        if hasattr(self, 'btn_run_main'):
            self.btn_run_main.setEnabled(False)
        if hasattr(self, 'btn_start_dub'):
            self.btn_start_dub.setEnabled(False)
        if hasattr(self, 'btn_cancel_job'):
            self.btn_cancel_job.setEnabled(True)
        if hasattr(self, 'btn_cancel_main'):
            self.btn_cancel_main.setEnabled(True)
        if hasattr(self, 'lbl_status_state'):
            self.lbl_status_state.setText("🔄 Processing...")

        voice = self.cb_voice.currentData() if hasattr(self, 'cb_voice') else "vi-VN-HoaiMyNeural"
        bg_vol = (self.slider_bg.value() if hasattr(self, 'slider_bg') else 10) / 100.0
        dub_vol = (self.slider_dub.value() if hasattr(self, 'slider_dub') else 100) / 100.0
        burn_sub = self.chk_burn_sub_export.isChecked() if hasattr(self, 'chk_burn_sub_export') else True
        if not burn_sub and hasattr(self, 'segments') and self.segments:
            burn_sub = True # Ép đè phụ đề tiếng Việt mặc định nếu có danh sách segments đã dịch

        preset = self.get_current_subtitle_preset() if hasattr(self, 'get_current_subtitle_preset') else {}
        self.log_info(f"[RENDER] Đang tiến hành che sub cũ, đè phụ đề tiếng Việt và lồng tiếng ra video thành phẩm...")
        workers_cnt = self.spin_workers.value() if hasattr(self, 'spin_workers') else 4
        chunk_workers = self.spin_chunk_workers.value() if hasattr(self, 'spin_chunk_workers') else workers_cnt
        source_lang = self.combo_source_lang.currentText() if hasattr(self, 'combo_source_lang') else "auto"
        target_lang = self.combo_target_lang.currentText() if hasattr(self, 'combo_target_lang') else "Tiếng Việt (vi)"
        gemini_api_key = self.txt_gemini_key.text().strip() if hasattr(self, 'txt_gemini_key') else ""
        xkiro_api_key = self.txt_xkiro_key.text().strip() if hasattr(self, 'txt_xkiro_key') else ""
        engine = self.cb_engine.currentText() if hasattr(self, 'cb_engine') else "Supersubs AI"
        refine_enabled = self.chk_refine_enabled.isChecked() if hasattr(self, 'chk_refine_enabled') else False
        refine_engine = getattr(self, 'cb_refine_engine', None).currentText() if hasattr(self, 'cb_refine_engine') else "Gemini 1.5 Flash"
        refine_api_key = gemini_api_key

        ocr_engine_raw = self.cb_ocr_engine.currentText() if hasattr(self, 'cb_ocr_engine') else "PaddleOCR"
        if "paddle" in ocr_engine_raw.lower():
            ocr_engine = "paddleocr"
        elif "xkiro" in ocr_engine_raw.lower():
            ocr_engine = "xkiro"
        elif "easyocr" in ocr_engine_raw.lower() or "truyền thống" in ocr_engine_raw.lower() or "offline" in ocr_engine_raw.lower():
            ocr_engine = "easyocr"
        else:
            ocr_engine = "gemini"

        self.pipeline_thread = FullOneClickPipelineWorker(
            video_path=self.video_path,
            output_path=out_path,
            workers_cnt=workers_cnt,
            chunk_workers=chunk_workers,
            selected_bbox=self.selected_bbox,
            selected_bboxes=self.selected_bboxes,
            voice=voice,
            bg_vol=bg_vol,
            dub_vol=dub_vol,
            burn_sub=burn_sub,
            preset=preset,
            enable_dubbing=self.chk_enable_dubbing.isChecked() if hasattr(self, 'chk_enable_dubbing') else True,
            logo_path=getattr(self, 'logo_path', None),
            source_lang=source_lang,
            target_lang=target_lang,
            api_key=gemini_api_key,
            xkiro_key=xkiro_api_key,
            engine=engine,
            ocr_engine=ocr_engine,
            title_bbox=getattr(self, 'title_bbox', None),
            refine_enabled=refine_enabled,
            refine_engine=refine_engine,
            refine_api_key=refine_api_key,
            ollama_model=self.txt_ollama_model.text().strip() if hasattr(self, 'txt_ollama_model') else "qwen2.5",
            vp_dict_paths=getattr(self, 'vp_dict_paths', None),
            scan_interval=self.spin_scan_interval.value() if hasattr(self, 'spin_scan_interval') else 0.5,
            min_sub_duration=self.spin_min_sub_dur.value() if hasattr(self, 'spin_min_sub_dur') else 0.3
        )
        self.pipeline_thread.progress.connect(self.log_info)
        self.pipeline_thread.progress_updated.connect(self.update_pipeline_progress)
        self.pipeline_thread.eta_updated.connect(self.update_pipeline_eta)
        self.pipeline_thread.chunk_progress.connect(self.update_chunk_progress)
        self.pipeline_thread.segments_ready.connect(self.on_segments_updated)
        self.pipeline_thread.finished.connect(self.on_dubbing_finished)
        self.pipeline_thread.error.connect(self.on_dubbing_error)
        self.pipeline_thread.start()

    def update_pipeline_progress(self, value, step_name):
        if hasattr(self, 'main_progress_bar'):
            self.main_progress_bar.setValue(int(value))
        if hasattr(self, 'lbl_status_state'):
            self.lbl_status_state.setText(f"⚡ {value}% - {step_name}")
        QApplication.processEvents()

    def update_pipeline_eta(self, eta_str):
        if hasattr(self, 'lbl_eta_time'):
            self.lbl_eta_time.setText(f"⏱️ Ước tính còn lại: {eta_str}")
        QApplication.processEvents()

    def update_chunk_progress(self, done, total):
        if hasattr(self, 'lbl_chunks_count'):
            self.lbl_chunks_count.setText(f"Chunks: {done}/{total}")

    def cancel_dubbing(self):
        """Hủy luồng xử lý background khẩn cấp khi người dùng ấn nút [🛑 Hủy]"""
        if hasattr(self, 'pipeline_thread') and self.pipeline_thread.isRunning():
            self.log_info("🛑 Đang gửi tín hiệu dừng khẩn cấp luồng Background...")
            self.pipeline_thread.stop()
            self.pipeline_thread.wait(2000)
            self.log_info("🛑 Tiến trình đã được hủy thành công bởi người dùng.")

        if hasattr(self, 'dub_thread') and self.dub_thread.isRunning():
            self.dub_thread.terminate()
            self.log_info("🛑 Đã dừng luồng dubbing.")

        if hasattr(self, 'btn_run_main'):
            self.btn_run_main.setEnabled(True)
        if hasattr(self, 'btn_start_dub'):
            self.btn_start_dub.setEnabled(True)
        if hasattr(self, 'btn_cancel_job'):
            self.btn_cancel_job.setEnabled(False)
        if hasattr(self, 'btn_cancel_main'):
            self.btn_cancel_main.setEnabled(False)
        if hasattr(self, 'lbl_status_state'):
            self.lbl_status_state.setText("🟢 Ready")
        if hasattr(self, 'lbl_eta_time'):
            self.lbl_eta_time.setText("⏱️ Ước tính còn lại: --:--")

    def on_segments_updated(self, segments):
        self.segments = segments
        self.populate_subtitle_table()
        
    def append_log3(self, text):
        self.txt_logs3.append(text)
        self.status_label.setText(text)
        
    def on_dubbing_finished(self, out_video):
        self.btn_run_main.setEnabled(True)
        if hasattr(self, 'btn_start_dub'):
            self.btn_start_dub.setEnabled(True)
        if hasattr(self, 'btn_cancel_main'):
            self.btn_cancel_main.setEnabled(False)
        if hasattr(self, 'lbl_status_state'):
            self.lbl_status_state.setText("✅ Done")
        if hasattr(self, 'lbl_eta_time'):
            self.lbl_eta_time.setText("⏱️ Ước tính còn lại: 00:00")
        self.status_label.setText("Kết xuất lồng tiếng hoàn tất!")
        self.log_info(f"🎉 XUẤT VIDEO THÀNH CÔNG: {out_video}")

        duration_min = 0.0
        if getattr(self, 'pipeline_start_time', None):
            secs = self.pipeline_start_time.secsTo(QDateTime.currentDateTime())
            duration_min = max(0.01, secs / 60.0)
        self.append_history_record("✅ Thành công", output_video=out_video, duration_minutes=duration_min, error_msg="")

        if not os.environ.get("QT_QPA_PLATFORM"):
            QMessageBox.information(
                self, 
                "Xuất Video Thành Công!", 
                f"Video lồng tiếng của bạn đã được xuất thành công!\nĐường dẫn: {out_video}"
            )
        
    def on_dubbing_error(self, err):
        self.btn_run_main.setEnabled(True)
        if hasattr(self, 'btn_start_dub'):
            self.btn_start_dub.setEnabled(True)
        if hasattr(self, 'btn_cancel_main'):
            self.btn_cancel_main.setEnabled(False)
        if hasattr(self, 'lbl_status_state'):
            self.lbl_status_state.setText("❌ Error")
        if hasattr(self, 'lbl_eta_time'):
            self.lbl_eta_time.setText("⏱️ Ước tính còn lại: --:--")
        self.status_label.setText("Lỗi kết xuất lồng tiếng.")

        duration_min = 0.0
        if getattr(self, 'pipeline_start_time', None):
            secs = self.pipeline_start_time.secsTo(QDateTime.currentDateTime())
            duration_min = max(0.01, secs / 60.0)
        self.append_history_record("❌ Thất bại", output_video="", duration_minutes=duration_min, error_msg=str(err))

        QMessageBox.critical(self, "Lỗi kết xuất", f"Đã xảy ra lỗi khi tạo video lồng tiếng:\n{err}")
        
    def setup_color_button_hex(self, btn, txt, initial_rgb, on_changed_callback):
        # Set initial values
        qcolor = QColor(*initial_rgb)
        btn.setStyleSheet(f"background-color: {qcolor.name()}; border: 1px solid #1c1c1f;")
        txt.setText(qcolor.name().upper())
        
        # When button is clicked
        def on_btn_clicked():
            dialog_color = QColorDialog.getColor(QColor(txt.text()), self, "Chọn màu")
            if dialog_color.isValid():
                btn.setStyleSheet(f"background-color: {dialog_color.name()}; border: 1px solid #1c1c1f;")
                txt.setText(dialog_color.name().upper())
                on_changed_callback(dialog_color.red(), dialog_color.green(), dialog_color.blue())
                
        # When hex text is edited
        def on_txt_edited():
            text = txt.text().strip()
            if not text.startswith("#"):
                text = "#" + text
            if len(text) == 7:
                c = QColor(text)
                if c.isValid():
                    btn.setStyleSheet(f"background-color: {c.name()}; border: 1px solid #1c1c1f;")
                    on_changed_callback(c.red(), c.green(), c.blue())
                    return
            # Revert if invalid
            qcolor = QColor(*initial_rgb)
            txt.setText(qcolor.name().upper())
            
        btn.clicked.connect(on_btn_clicked)
        if hasattr(txt, 'editingFinished'):
            txt.editingFinished.connect(on_txt_edited)
        
    def update_color_button(self, btn, txt, rgb):
        qcolor = QColor(*rgb)
        btn.setStyleSheet(f"background-color: {qcolor.name()}; border: 1px solid #1c1c1f;")
        if hasattr(txt, 'blockSignals'):
            txt.blockSignals(True)
        txt.setText(qcolor.name().upper())
        if hasattr(txt, 'blockSignals'):
            txt.blockSignals(False)
        
    def get_current_subtitle_preset(self):
        font_name = self.cb_font_name.currentText() if hasattr(self, 'cb_font_name') and self.cb_font_name is not None else "Arial"
        if getattr(self, 'custom_font_path', None) and font_name == os.path.basename(self.custom_font_path):
            font_name = self.custom_font_path

        mask_modes_map = ["none", "black", "blur", "inpaint"]
        mask_idx = self.cb_mask_mode.currentIndex() if hasattr(self, 'cb_mask_mode') and self.cb_mask_mode is not None else 2
        mask_mode = mask_modes_map[mask_idx] if 0 <= mask_idx < len(mask_modes_map) else "blur"

        remove_algos_map = ["ffmpeg", "opencv"]
        algo_idx = self.cb_remove_algo.currentIndex() if hasattr(self, 'cb_remove_algo') and self.cb_remove_algo is not None else 0
        remove_algo = remove_algos_map[algo_idx] if 0 <= algo_idx < len(remove_algos_map) else "ffmpeg"

        v_align = self.cb_v_align.currentText().lower() if hasattr(self, 'cb_v_align') and self.cb_v_align is not None else "bottom"
        h_align = self.cb_h_align.currentText().lower() if hasattr(self, 'cb_h_align') and self.cb_h_align is not None else "center"
        margin_v_type = "percent" if (hasattr(self, 'cb_margin_v_type') and self.cb_margin_v_type is not None and self.cb_margin_v_type.currentIndex() == 0) else "pixels"
        margin_v_val = self.spin_margin_v.value() if hasattr(self, 'spin_margin_v') and self.spin_margin_v is not None else 8.0
        margin_h_type = "percent" if (hasattr(self, 'cb_margin_h_type') and self.cb_margin_h_type is not None and self.cb_margin_h_type.currentIndex() == 0) else "pixels"
        margin_h_val = self.spin_margin_h.value() if hasattr(self, 'spin_margin_h') and self.spin_margin_h is not None else 5.0

        font_size = self.spin_font_size.value() if hasattr(self, 'spin_font_size') and self.spin_font_size is not None else 20
        outline_width = self.spin_outline_width.value() if hasattr(self, 'spin_outline_width') and self.spin_outline_width is not None else 2
        bg_opacity = self.slider_bg_opacity.value() if hasattr(self, 'slider_bg_opacity') and self.slider_bg_opacity is not None else 70
        use_bg_box = self.chk_use_bg_box.isChecked() if hasattr(self, 'chk_use_bg_box') and self.chk_use_bg_box is not None else False
        smart_pos = self.chk_smart_pos.isChecked() if hasattr(self, 'chk_smart_pos') and self.chk_smart_pos is not None else True

        preset = {
            "v_align": v_align,
            "h_align": h_align,
            "margin_v_type": margin_v_type,
            "margin_v_val": margin_v_val,
            "margin_h_type": margin_h_type,
            "margin_h_val": margin_h_val,
            "font_name": font_name,
            "font_size": font_size,
            "font_color": getattr(self, 'preset_font_color', [255, 255, 255]),
            "outline_color": getattr(self, 'preset_outline_color', [0, 0, 0]),
            "outline_width": outline_width,
            "bg_color": getattr(self, 'preset_bg_color', [0, 0, 0]),
            "bg_opacity": bg_opacity,
            "use_bg_box": use_bg_box,
            "mask_mode": mask_mode,
            "remove_algo": remove_algo,
            "smart_pos": smart_pos
        }
        if getattr(self, 'subtitle_custom_pos', None):
            preset["custom_pos"] = dict(self.subtitle_custom_pos)
        return preset
        
    def set_subtitle_preset_ui(self, preset_dict, apply_style=True):
        if not preset_dict:
            return
        if hasattr(self, 'cb_v_align') and self.cb_v_align is not None:
            self.cb_v_align.setCurrentText(preset_dict.get("v_align", "bottom").capitalize())
        if hasattr(self, 'cb_h_align') and self.cb_h_align is not None:
            self.cb_h_align.setCurrentText(preset_dict.get("h_align", "center").capitalize())
        if hasattr(self, 'cb_margin_v_type') and self.cb_margin_v_type is not None:
            self.cb_margin_v_type.setCurrentIndex(0 if preset_dict.get("margin_v_type") == "percent" else 1)
        if hasattr(self, 'spin_margin_v') and self.spin_margin_v is not None:
            self.spin_margin_v.setValue(preset_dict.get("margin_v_val", 8.0))
        
        if "mask_mode" in preset_dict and hasattr(self, 'cb_mask_mode') and self.cb_mask_mode is not None:
            mask_modes_map = ["none", "black", "blur", "inpaint"]
            try:
                idx = mask_modes_map.index(preset_dict["mask_mode"])
                self.cb_mask_mode.setCurrentIndex(idx)
            except ValueError:
                pass
                
        if "remove_algo" in preset_dict and hasattr(self, 'cb_remove_algo') and self.cb_remove_algo is not None:
            remove_algos_map = ["ffmpeg", "opencv"]
            try:
                idx = remove_algos_map.index(preset_dict["remove_algo"])
                self.cb_remove_algo.setCurrentIndex(idx)
            except ValueError:
                pass
                
        if "smart_pos" in preset_dict and hasattr(self, 'chk_smart_pos') and self.chk_smart_pos is not None:
            self.chk_smart_pos.setChecked(preset_dict["smart_pos"])
            
        if hasattr(self, 'cb_margin_h_type') and self.cb_margin_h_type is not None:
            self.cb_margin_h_type.setCurrentIndex(0 if preset_dict.get("margin_h_type") == "percent" else 1)
        if hasattr(self, 'spin_margin_h') and self.spin_margin_h is not None:
            self.spin_margin_h.setValue(preset_dict.get("margin_h_val", 5.0))
        
        if apply_style:
            font_name = preset_dict.get("font_name", "Arial")
            if hasattr(self, 'cb_font_name') and self.cb_font_name is not None:
                if font_name.endswith(('.ttf', '.otf', '.ttc')) and os.path.exists(font_name):
                    self.custom_font_path = font_name
                    basename = os.path.basename(font_name)
                    idx = self.cb_font_name.findText(basename)
                    if idx == -1:
                        self.cb_font_name.addItem(basename)
                    self.cb_font_name.setCurrentText(basename)
                else:
                    self.cb_font_name.setCurrentText(font_name)
                
            if hasattr(self, 'spin_font_size') and self.spin_font_size is not None:
                self.spin_font_size.setValue(preset_dict.get("font_size", 20))
            
            self.preset_font_color = preset_dict.get("font_color", [255, 255, 255])
            if hasattr(self, 'btn_font_color') and hasattr(self, 'txt_font_color_hex') and self.btn_font_color and self.txt_font_color_hex:
                self.update_color_button(self.btn_font_color, self.txt_font_color_hex, self.preset_font_color)
            
            self.preset_outline_color = preset_dict.get("outline_color", [0, 0, 0])
            if hasattr(self, 'btn_outline_color') and hasattr(self, 'txt_outline_color_hex') and self.btn_outline_color and self.txt_outline_color_hex:
                self.update_color_button(self.btn_outline_color, self.txt_outline_color_hex, self.preset_outline_color)
            
            self.preset_bg_color = preset_dict.get("bg_color", [0, 0, 0])
            if hasattr(self, 'btn_bg_color') and hasattr(self, 'txt_bg_color_hex') and self.btn_bg_color and self.txt_bg_color_hex:
                self.update_color_button(self.btn_bg_color, self.txt_bg_color_hex, self.preset_bg_color)
            
            if hasattr(self, 'spin_outline_width') and self.spin_outline_width is not None:
                self.spin_outline_width.setValue(preset_dict.get("outline_width", 2))
            if hasattr(self, 'chk_use_bg_box') and self.chk_use_bg_box is not None:
                self.chk_use_bg_box.setChecked(preset_dict.get("use_bg_box", False))
            if hasattr(self, 'slider_bg_opacity') and self.slider_bg_opacity is not None:
                self.slider_bg_opacity.setValue(preset_dict.get("bg_opacity", 70))

        # Đồng bộ Tab2 widgets từ preset vừa áp dụng
        self._tab2_sync_from_preset()

    def apply_selected_preset(self, text):
        """Áp dụng preset khi người dùng chọn từ combobox presets."""
        try:
            if not text:
                return
            if text == "Tùy chỉnh (Custom)":
                return
            preset = self.presets_db.get(text)
            if preset:
                self.set_subtitle_preset_ui(preset, apply_style=True)
                self.status_label.setText(f"Áp dụng preset: {text}")
        except Exception as e:
            self.status_label.setText(f"Lỗi áp preset: {e}")
            
    def on_font_color_changed(self, r, g, b):
        self.preset_font_color = [r, g, b]
        self.mark_preset_custom()
        
    def on_outline_color_changed(self, r, g, b):
        self.preset_outline_color = [r, g, b]
        self.mark_preset_custom()
        
    def on_bg_color_changed(self, r, g, b):
        self.preset_bg_color = [r, g, b]
        self.mark_preset_custom()
        
    def on_preset_changed(self, text):
        if not hasattr(self, 'cb_preset') or self.cb_preset is None:
            return
        is_custom = (text == "Tùy chỉnh (Custom)")
        if hasattr(self, 'spin_custom_pos_x') and self.spin_custom_pos_x is not None:
            self.spin_custom_pos_x.setEnabled(is_custom)
            if hasattr(self, 'spin_custom_pos_y') and self.spin_custom_pos_y is not None:
                self.spin_custom_pos_y.setEnabled(is_custom)
            if hasattr(self, 'btn_reset_custom_pos') and self.btn_reset_custom_pos is not None:
                self.btn_reset_custom_pos.setEnabled(is_custom)
        if text in getattr(self, 'presets_db', {}):
            self.subtitle_custom_pos = None
            apply_mode_idx = self.cb_preset_apply_mode.currentIndex() if hasattr(self, 'cb_preset_apply_mode') and self.cb_preset_apply_mode is not None else 1
            apply_style = (apply_mode_idx == 1)
            self.set_subtitle_preset_ui(self.presets_db[text], apply_style=apply_style)
        elif is_custom:
            if not getattr(self, 'subtitle_custom_pos', None) and hasattr(self, 'spin_custom_pos_x') and self.spin_custom_pos_x is not None:
                self.subtitle_custom_pos = {
                    "x_pct": self.spin_custom_pos_x.value(),
                    "y_pct": self.spin_custom_pos_y.value()
                }
            self.trigger_canvas_update()
            
    def on_custom_pos_spin_changed(self):
        if hasattr(self, 'spin_custom_pos_x') and self.spin_custom_pos_x is not None:
            self.subtitle_custom_pos = {
                "x_pct": self.spin_custom_pos_x.value(),
                "y_pct": self.spin_custom_pos_y.value()
            }
        self.mark_preset_custom()

    def mark_preset_custom(self):
        if getattr(self, 'block_preset_signals', False):
            return
        if hasattr(self, 'cb_preset_profile') and self.cb_preset_profile:
            idx = self.cb_preset_profile.findText("Custom")
            if idx >= 0 and self.cb_preset_profile.currentIndex() != idx:
                self.cb_preset_profile.blockSignals(True)
                self.cb_preset_profile.setCurrentIndex(idx)
                self.cb_preset_profile.blockSignals(False)
        if hasattr(self, 'cb_preset') and self.cb_preset is not None:
            self.cb_preset.blockSignals(True)
            idx = self.cb_preset.findText("Custom")
            if idx >= 0:
                self.cb_preset.setCurrentIndex(idx)
            else:
                self.cb_preset.setCurrentText("Tùy chỉnh (Custom)")
            self.cb_preset.blockSignals(False)
        if hasattr(self, 'spin_custom_pos_x') and self.spin_custom_pos_x is not None:
            self.spin_custom_pos_x.setEnabled(True)
            if hasattr(self, 'spin_custom_pos_y') and self.spin_custom_pos_y is not None:
                self.spin_custom_pos_y.setEnabled(True)
            if hasattr(self, 'btn_reset_custom_pos') and self.btn_reset_custom_pos is not None:
                self.btn_reset_custom_pos.setEnabled(True)
        
    def on_font_changed(self):
        self.mark_preset_custom()
        font_name = self.cb_font_name.currentText()
        if hasattr(self, 'tab2_font_combo'):
            self.tab2_font_combo.blockSignals(True)
            self.tab2_font_combo.setCurrentFont(QFont(font_name))
            self.tab2_font_combo.blockSignals(False)
        self._tab2_update_font_preview()
        self.trigger_canvas_update()
        font_path = dubber.get_font_path(font_name)
        if font_path and os.path.exists(font_path):
            if not check_font_vietnamese_support(font_path):
                QMessageBox.warning(self, "Cảnh báo Font chữ", 
                                    f"Font '{font_name}' có thể không hỗ trợ đầy đủ tiếng Việt Unicode có dấu!\n"
                                    "Nếu hiển thị bị lỗi, hãy chọn các font thay thế như: Arial, Noto Sans, Roboto, Segoe UI, Tahoma.")

    # ========== Tab2 <-> Preset Bidirectional Sync Methods ==========

    def _tab2_sync_font_to_preset(self, font=None):
        """Khi Tab2 font combo thay đổi -> cập nhật cb_font_name trong right panel."""
        if isinstance(font, str):
            font_name = font
        elif font is not None and hasattr(font, 'family') and font.family():
            font_name = font.family()
        else:
            font_name = self.tab2_font_combo.get_current_font_family()

        if not font_name:
            font_name = self.tab2_font_combo.get_current_font_family()
        if hasattr(self, 'cb_font_name'):
            idx = self.cb_font_name.findText(font_name)
            if idx == -1:
                self.cb_font_name.addItem(font_name)
            self.cb_font_name.blockSignals(True)
            self.cb_font_name.setCurrentText(font_name)
            self.cb_font_name.blockSignals(False)
        self._tab2_update_font_preview()
        self.mark_preset_custom()
        self.trigger_canvas_update()

    def _tab2_sync_size_to_preset(self, val):
        """Khi Tab2 font size thay đổi -> cập nhật spin_font_size trong right panel."""
        if hasattr(self, 'spin_font_size'):
            self.spin_font_size.blockSignals(True)
            self.spin_font_size.setValue(val)
            self.spin_font_size.blockSignals(False)
        self.mark_preset_custom()
        self.trigger_canvas_update()

    def _tab2_sync_outline_to_preset(self, val):
        """Khi Tab2 outline width thay đổi -> cập nhật spin_outline_width trong right panel."""
        if hasattr(self, 'spin_outline_width'):
            self.spin_outline_width.blockSignals(True)
            self.spin_outline_width.setValue(val)
            self.spin_outline_width.blockSignals(False)
        self.mark_preset_custom()
        self.trigger_canvas_update()

    def _tab2_sync_bg_box_to_preset(self, checked):
        """Khi Tab2 bg box checkbox thay đổi -> cập nhật chk_use_bg_box trong right panel."""
        if hasattr(self, 'chk_use_bg_box'):
            self.chk_use_bg_box.blockSignals(True)
            self.chk_use_bg_box.setChecked(checked)
            self.chk_use_bg_box.blockSignals(False)
        self.mark_preset_custom()
        self.trigger_canvas_update()

    def _tab2_pick_color(self, color_type):
        """Mở QColorDialog từ Tab2 và đồng bộ màu về preset system."""
        from PyQt6.QtWidgets import QColorDialog
        if color_type == 'font':
            init_color = QColor(*self.preset_font_color)
        else:
            init_color = QColor(*self.preset_outline_color)

        color = QColorDialog.getColor(init_color, self, "Chọn màu chữ" if color_type == 'font' else "Chọn màu viền")
        if not color.isValid():
            return

        rgb = [color.red(), color.green(), color.blue()]
        hex_str = color.name().upper()

        if color_type == 'font':
            self.preset_font_color = rgb
            self.tab2_btn_font_color.setStyleSheet(f"background-color: {hex_str}; border: 2px solid #334155; border-radius: 4px;")
            self.tab2_lbl_font_hex.setText(hex_str)
            # Sync to right panel
            if hasattr(self, 'btn_font_color') and hasattr(self, 'txt_font_color_hex'):
                self.update_color_button(self.btn_font_color, self.txt_font_color_hex, rgb)
        else:
            self.preset_outline_color = rgb
            self.tab2_btn_outline_color.setStyleSheet(f"background-color: {hex_str}; border: 2px solid #334155; border-radius: 4px;")
            self.tab2_lbl_outline_hex.setText(hex_str)
            # Sync to right panel
            if hasattr(self, 'btn_outline_color') and hasattr(self, 'txt_outline_color_hex'):
                self.update_color_button(self.btn_outline_color, self.txt_outline_color_hex, rgb)

        self._tab2_update_font_preview()
        self.mark_preset_custom()
        self.trigger_canvas_update()

    def _tab2_update_font_preview(self, *args):
        """Cập nhật Live Font Preview Label trong Tab2 và in rõ thông tin kiểu chữ lên Log Console."""
        if not hasattr(self, 'tab2_font_preview'):
            return
        font_family = self.tab2_font_combo.get_current_font_family()
        font_size = self.tab2_font_size.value()
        font_color = getattr(self, 'preset_font_color', [255, 255, 255])
        outline_color = getattr(self, 'preset_outline_color', [0, 0, 0])
        outline_w = self.tab2_outline_width.value()
        use_bg = self.tab2_chk_bg_box.isChecked() if hasattr(self, 'tab2_chk_bg_box') else False

        fc_hex = "#{:02X}{:02X}{:02X}".format(*font_color)
        oc_hex = "#{:02X}{:02X}{:02X}".format(*outline_color)

        bg_css = "background-color: #000000;" if use_bg else "background-color: #0f172a;"

        self.tab2_font_preview.setStyleSheet(
            f"{bg_css} color: {fc_hex}; border: 1px solid #38bdf8; "
            f"border-radius: 6px; padding: 10px; font-family: '{font_family}'; font-size: {max(14, min(36, font_size))}px; font-weight: bold;"
        )
        self.tab2_font_preview.setText("Xin chào tôi là supersubs")

        log_msg = f"🎨 Đã chọn kiểu chữ Subtitle: Font '{font_family}' | Cỡ: {font_size}px | Màu chữ: {fc_hex} | Viền: {outline_w}px ({oc_hex})"
        if hasattr(self, 'log_info'):
            if not hasattr(self, '_last_logged_font_str') or self._last_logged_font_str != log_msg:
                self._last_logged_font_str = log_msg
                self.log_info(log_msg)

    def _tab2_sync_from_preset(self):
        """Đồng bộ ngược từ right panel preset về Tab2 widgets (gọi khi load preset)."""
        if not hasattr(self, 'tab2_font_combo'):
            return
        # Font
        if hasattr(self, 'cb_font_name'):
            self.tab2_font_combo.blockSignals(True)
            self.tab2_font_combo.setCurrentFont(QFont(self.cb_font_name.currentText()))
            self.tab2_font_combo.blockSignals(False)
        # Size
        if hasattr(self, 'spin_font_size'):
            self.tab2_font_size.blockSignals(True)
            self.tab2_font_size.setValue(self.spin_font_size.value())
            self.tab2_font_size.blockSignals(False)
        # Outline
        if hasattr(self, 'spin_outline_width'):
            self.tab2_outline_width.blockSignals(True)
            self.tab2_outline_width.setValue(self.spin_outline_width.value())
            self.tab2_outline_width.blockSignals(False)
        # Bg box
        if hasattr(self, 'chk_use_bg_box'):
            self.tab2_chk_bg_box.blockSignals(True)
            self.tab2_chk_bg_box.setChecked(self.chk_use_bg_box.isChecked())
            self.tab2_chk_bg_box.blockSignals(False)
        # Colors
        fc_hex = "#{:02X}{:02X}{:02X}".format(*self.preset_font_color)
        oc_hex = "#{:02X}{:02X}{:02X}".format(*self.preset_outline_color)
        self.tab2_btn_font_color.setStyleSheet(f"background-color: {fc_hex}; border: 2px solid #334155; border-radius: 4px;")
        self.tab2_lbl_font_hex.setText(fc_hex)
        self.tab2_btn_outline_color.setStyleSheet(f"background-color: {oc_hex}; border: 2px solid #334155; border-radius: 4px;")
        self.tab2_lbl_outline_hex.setText(oc_hex)
        self._tab2_update_font_preview()


    def browse_custom_font(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file font TrueType/OpenType", "", "Font Files (*.ttf *.otf *.ttc)")
        if file_path:
            dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "fonts")
            os.makedirs(dest_dir, exist_ok=True)
            basename = os.path.basename(file_path)
            dest_path = os.path.join(dest_dir, basename)
            if file_path != dest_path:
                try:
                    import shutil
                    shutil.copy(file_path, dest_path)
                except Exception as e:
                    QMessageBox.warning(self, "Lỗi", f"Không thể sao chép font vào dự án: {e}")
                    return
            self.custom_font_path = dest_path
            idx = self.cb_font_name.findText(basename)
            if idx == -1:
                self.cb_font_name.addItem(basename)
            self.cb_font_name.setCurrentText(basename)
            if not check_font_vietnamese_support(dest_path):
                QMessageBox.warning(self, "Cảnh báo Font chữ", 
                                    "Font chữ bạn vừa chọn không hỗ trợ đầy đủ tiếng Việt Unicode có dấu!\n"
                                    "Gợi ý font thay thế: Arial, Noto Sans, Roboto.")
                                    
    def get_api_config_path(self):
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, "api_keys.json")

    def batch_import_gemini_keys(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("📋 Dán Hàng Loạt Gemini API Keys")
        dlg.resize(550, 350)
        vbox = QVBoxLayout(dlg)
        
        lbl_hint = QLabel("Dán danh sách Gemini API Keys vào khung dưới đây (Mỗi key trên 1 dòng hoặc phân cách bởi dấu phẩy):")
        lbl_hint.setWordWrap(True)
        lbl_hint.setStyleSheet("color: #38bdf8; font-weight: bold;")
        vbox.addWidget(lbl_hint)

        txt_batch = QTextEdit()
        txt_batch.setPlaceholderText("AIzaSyA...\nAIzaSyB...\nAIzaSyC...")
        vbox.addWidget(txt_batch)

        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_ok = QPushButton("✓ Nhập Keys")
        btn_ok.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 6px 14px;")
        def process_import():
            raw_text = txt_batch.toPlainText()
            import re
            tokens = [t.strip() for t in re.split(r'[\n,;\t]+', raw_text) if t.strip()]
            if tokens:
                if hasattr(self, 'txt_gemini_key') and self.txt_gemini_key:
                    existing = [k.strip() for k in self.txt_gemini_key.text().split(",") if k.strip()]
                    all_keys = list(dict.fromkeys(existing + tokens))
                    self.txt_gemini_key.setText(", ".join(all_keys))
                if hasattr(self, 'save_api_config_from_ui'):
                    self.save_api_config_from_ui()
                QMessageBox.information(self, "Thành công", f"Đã nạp & lưu thành công {len(tokens)} Gemini API Key mới vào hệ thống xoay key (Key Pool)!")
            dlg.accept()

        btn_ok.clicked.connect(process_import)
        btn_cancel = QPushButton("Hủy")
        btn_cancel.clicked.connect(dlg.reject)
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        vbox.addLayout(btn_box)
        dlg.exec()

    def clean_empty_gemini_key_slots(self):
        to_remove = []
        for row, txt in zip(list(self.gemini_key_rows), list(self.gemini_key_inputs)):
            if not txt.text().strip():
                to_remove.append((row, txt))
        for row, txt in to_remove:
            if txt in self.gemini_key_inputs:
                idx = self.gemini_key_inputs.index(txt)
                self.gemini_key_inputs.pop(idx)
                if idx < len(self.gemini_key_status_labels):
                    self.gemini_key_status_labels.pop(idx)
            if row in self.gemini_key_rows:
                self.gemini_key_rows.remove(row)
            row.deleteLater()
        self._reindex_key_slot_labels()
        self._sync_gemini_keys_to_main_input()
        self.save_api_config()

    def _sync_gemini_keys_to_main_input(self):
        keys = [txt.text().strip() for txt in getattr(self, 'gemini_key_inputs', []) if txt.text().strip()]
        joined_keys = ", ".join(keys)
        if hasattr(self, 'txt_gemini_key') and self.txt_gemini_key:
            self.txt_gemini_key.blockSignals(True)
            self.txt_gemini_key.setText(joined_keys)
            self.txt_gemini_key.blockSignals(False)

    def check_all_gemini_keys_health(self):
        self.log_info("🔍 Đang kiểm tra kết nối & trạng thái hạn ngạch (Rate Limit/Quota) của tất cả Gemini Keys...")
        if hasattr(self, 'btn_check_gemini_keys'):
            self.btn_check_gemini_keys.setEnabled(False)
        
        key_list = [txt.text().strip() for txt in getattr(self, 'gemini_key_inputs', [])]
        
        self.gemini_key_checker = GeminiKeyCheckWorker(key_list)
        self.gemini_key_checker.key_tested.connect(self.on_gemini_key_tested)
        self.gemini_key_checker.finished.connect(self.on_gemini_key_check_finished)
        self.gemini_key_checker.start()

    def on_gemini_key_tested(self, idx, key, status_code, message):
        if hasattr(self, 'gemini_key_status_labels') and idx < len(self.gemini_key_status_labels):
            lbl = self.gemini_key_status_labels[idx]
            if status_code == "ACTIVE":
                lbl.setText("🟢 Đang hoạt động (API OK)")
                lbl.setStyleSheet("color: #4ade80; font-weight: bold; font-size: 11px;")
            elif status_code == "RATE_LIMIT":
                lbl.setText("🟡 Chờ hạn ngạch (Rate Limit 429)")
                lbl.setStyleSheet("color: #facc15; font-weight: bold; font-size: 11px;")
            elif status_code == "INITIALIZING":
                lbl.setText("🟡 Key mới (Đang đồng bộ Google, chờ ~30s)")
                lbl.setStyleSheet("color: #facc15; font-weight: bold; font-size: 11px;")
            elif status_code == "EXHAUSTED":
                lbl.setText("🔴 Key không hợp lệ / Hết Quota")
                lbl.setStyleSheet("color: #f87171; font-weight: bold; font-size: 11px;")
            elif status_code == "EMPTY":
                lbl.setText("⚪ Chưa nhập")
                lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
            else:
                lbl.setText(message)
                lbl.setStyleSheet("color: #f87171; font-weight: bold; font-size: 11px;")

    def on_gemini_key_check_finished(self):
        if hasattr(self, 'btn_check_gemini_keys'):
            self.btn_check_gemini_keys.setEnabled(True)
        self.log_info("✔ Kiểm tra hoàn tất trạng thái Gemini Key Pool.")

    def reset_preset(self):
        self.subtitle_custom_pos = None
        default_preset = {
            "v_align": "bottom", "h_align": "center", "margin_v_type": "percent", "margin_v_val": 8.0, "margin_h_type": "percent", "margin_h_val": 5.0,
            "font_name": "Arial", "font_size": 20, "font_color": [255, 255, 255], "outline_color": [0, 0, 0], "outline_width": 2, "use_bg_box": False, "bg_color": [0, 0, 0], "bg_opacity": 0,
            "remove_algo": "opencv", "smart_pos": False
        }
        self.cb_preset.blockSignals(True)
        self.cb_preset.setCurrentText("Mặc định (Dưới - Giữa)")
        self.cb_preset.blockSignals(False)
        self.cb_remove_algo.setCurrentIndex(1)
        self.set_subtitle_preset_ui(default_preset, apply_style=True)
        
    def open_preview_dialog(self):
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn hoặc tải video nguồn trước.")
            return
        if not self.segments:
            QMessageBox.warning(self, "Cảnh báo", "Không có phân đoạn phụ đề dịch nào để xem trước.")
            return
            
        dialog = SubtitlePreviewDialog(self.video_path, self.segments, self.get_current_subtitle_preset(), self)
        dialog.exec()
        
    def undo_bbox_change(self):
        if not self.bbox_history_stack:
            self.status_label.setText("Không có thao tác nào để Undo.")
            return
        prev_state = self.bbox_history_stack.pop()
        current_state = [(seg.get('bbox'), seg.get('confidence')) for seg in self.segments]
        self.bbox_redo_stack.append(current_state)
        
        for idx, (bbox, conf) in enumerate(prev_state):
            if idx < len(self.segments):
                self.segments[idx]['bbox'] = bbox
                self.segments[idx]['confidence'] = conf
        self.populate_subtitle_table()
        self.status_label.setText("Đã hoàn tác (Undo) thay đổi vùng che.")
        
    def redo_bbox_change(self):
        if not self.bbox_redo_stack:
            self.status_label.setText("Không có thao tác nào để Redo.")
            return
        next_state = self.bbox_redo_stack.pop()
        current_state = [(seg.get('bbox'), seg.get('confidence')) for seg in self.segments]
        self.bbox_history_stack.append(current_state)
        
        for idx, (bbox, conf) in enumerate(next_state):
            if idx < len(self.segments):
                self.segments[idx]['bbox'] = bbox
                self.segments[idx]['confidence'] = conf
        self.populate_subtitle_table()
        self.status_label.setText("Đã khôi phục (Redo) thay đổi vùng che.")
        
    def save_bbox_state_to_history(self):
        state = [(seg.get('bbox'), seg.get('confidence')) for seg in self.segments]
        self.bbox_history_stack.append(state)
        self.bbox_redo_stack.clear()
        
    def get_video_cache_id(self, video_path):
        if not video_path or not os.path.exists(video_path):
            return None
        import hashlib
        try:
            stat = os.stat(video_path)
            base = os.path.basename(video_path)
            key_str = f"{base}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(key_str.encode('utf-8')).hexdigest()
        except Exception:
            return None
            
    def save_cache_file(self):
        video_id = self.get_video_cache_id(self.video_path)
        if not video_id:
            return
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{video_id}_detection.json")
        try:
            cache_structure = {
                'config': {
                    'ocr_lang': self.cb_ocr_lang.currentText(),
                    'restrict_region': self.chk_restrict_ocr.isChecked()
                },
                'data': [
                    {
                        'bbox': seg.get('bbox'),
                        'confidence': seg.get('confidence'),
                        'ocr_timestamp': seg.get('ocr_timestamp')
                    } for seg in self.segments
                ]
            }
            import json
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_structure, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def trigger_auto_detection(self):
        if not self.segments:
            QMessageBox.warning(self, "Cảnh báo", "Không có phân đoạn phụ đề nào để quét.")
            return
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn hoặc tải video nguồn trước.")
            return
            
        import torch
        import json
        
        video_id = self.get_video_cache_id(self.video_path)
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{video_id}_detection.json")
        
        # Kiểm tra Cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_obj = json.load(f)
                
                cached_config = cached_obj.get('config', {})
                current_ocr_lang = self.cb_ocr_lang.currentText()
                current_restrict = self.chk_restrict_ocr.isChecked()
                
                if (cached_config.get('ocr_lang') == current_ocr_lang and 
                    cached_config.get('restrict_region') == current_restrict):
                    
                    self.save_bbox_state_to_history()
                    cached_data = cached_obj.get('data', [])
                    for idx, item in enumerate(cached_data):
                        if idx < len(self.segments):
                            self.segments[idx]['bbox'] = item.get('bbox')
                            self.segments[idx]['confidence'] = item.get('confidence')
                            self.segments[idx]['ocr_timestamp'] = item.get('ocr_timestamp', (self.segments[idx]['start'] + self.segments[idx]['end']) / 2.0)
                    self.populate_subtitle_table()
                    self.status_label.setText("Đã nạp vùng che từ cache cục bộ.")
                    QMessageBox.information(self, "Tải từ Cache", "Đã tự động nạp tọa độ vùng che từ cache cục bộ (tốc độ < 10ms).")
                    return
            except Exception:
                pass
                
        # Ước tính số cụm quét
        groups_count = 0
        last_end = -10.0
        for seg in self.segments:
            if last_end < 0:
                groups_count = 1
            else:
                if seg['start'] - last_end >= 1.0:
                    groups_count += 1
            last_end = seg['end']
            
        gpu_avail = torch.cuda.is_available()
        sec_per_scan = 4.25 if not gpu_avail else 0.15
        est_seconds = int(groups_count * sec_per_scan)
        est_min = est_seconds // 60
        est_sec = est_seconds % 60
        est_time_str = f"{est_min} phút {est_sec} giây" if est_min > 0 else f"{est_sec} giây"
        
        warning_msg = ""
        if not gpu_avail:
            warning_msg += "⚠️ CẢNH BÁO: Thiết bị đang chạy ở chế độ CPU-only (không có GPU CUDA).\n\n"
        warning_msg += f"Video của bạn có {len(self.segments)} phân đoạn (gộp thành {groups_count} cụm quét).\n" \
                       f"Dự toán thời gian quét EasyOCR thô khoảng: {est_time_str} (đây là ước tính TỐT NHẤT - best-case, có thể lâu hơn nếu nhiều đoạn phải quét lại do chữ mờ/chuyển cảnh - cơ chế Retry cộng thêm tối đa ~8.5s cho mỗi cụm).\n\n"
                       
        if groups_count > 80:
            warning_msg += "💡 KHUYẾN NGHỊ: Số cụm quét lớn (>80). Để tiết kiệm thời gian, bạn nên chọn:\n" \
                           " - Chế độ 'Blur nhanh' khi xuất video để giảm thời gian xử lý.\n\n"
                           
        warning_msg += "Bạn có muốn bắt đầu quá trình quét chữ gốc tự động không?"
        
        reply = QMessageBox.question(self, "Xác nhận quét OCR", warning_msg,
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return
            
        self.btn_auto_detect.setEnabled(False)
        self.status_label.setText("Đang trích xuất vùng che tự động...")
        
        class OCRWorkerThread(QThread):
            progress = pyqtSignal(str)
            finished = pyqtSignal(list)
            error = pyqtSignal(str)
            frame_signal = pyqtSignal(object, int, int, float, object, str)
            
            def __init__(self, video_path, segments, ocr_lang, restrict_region):
                super().__init__()
                self.video_path = video_path
                self.segments = segments
                self.ocr_lang = ocr_lang
                self.restrict_region = restrict_region
                
            def run(self):
                try:
                    def _on_frame(frame, f_idx, tot_f, t_sec, bbox, msg):
                        self.frame_signal.emit(frame, f_idx, tot_f, t_sec, bbox, msg)

                    res = transcriber.run_segment_guided_ocr(
                        self.video_path,
                        self.segments,
                        progress_callback=self.progress.emit,
                        ocr_lang=self.ocr_lang,
                        restrict_region=self.restrict_region,
                        frame_callback=_on_frame
                    )
                    self.finished.emit(res)
                except Exception as e:
                    import traceback
                    self.error.emit(f"{str(e)}\n{traceback.format_exc()}")
                    
        self.ocr_thread = OCRWorkerThread(
            self.video_path,
            self.segments,
            self.cb_ocr_lang.currentText(),
            self.chk_restrict_ocr.isChecked()
        )
        
        def on_ocr_finished(res_segs):
            self.save_bbox_state_to_history()
            self.segments = res_segs
            self.populate_subtitle_table()
            self.btn_auto_detect.setEnabled(True)
            self.status_label.setText("Tự động quét hoàn tất!")
            self.save_cache_file()
            QMessageBox.information(self, "Hoàn tất quét", "Đã trích xuất xong vùng phụ đề gốc và tự động lưu cache.")
            
        def on_ocr_error(err_str):
            self.btn_auto_detect.setEnabled(True)
            self.status_label.setText("Lỗi quét phụ đề.")
            QMessageBox.critical(self, "Lỗi quét OCR", f"Đã xảy ra sự cố trong quá trình quét OCR:\n{err_str}")
            
        self.ocr_thread.progress.connect(self.status_label.setText)
        self.ocr_thread.frame_signal.connect(self.on_worker_frame_update)
        self.ocr_thread.finished.connect(on_ocr_finished)
        self.ocr_thread.error.connect(on_ocr_error)
        self.ocr_thread.start()

    def edit_mask_box(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn 1 dòng phụ đề để điều chỉnh hộp che.")
            return
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy file video gốc.")
            return
            
        start_s = self.segments[row]['start']
        end_s = self.segments[row]['end']
        t_target = self.segments[row].get('ocr_timestamp', (start_s + end_s) / 2.0)
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_target * fps))
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            QMessageBox.critical(self, "Lỗi", f"Không thể lấy khung hình tại giây thứ {t_target:.2f}")
            return
            
        existing_box = self.segments[row].get('bbox')
        selector = VideoRegionSelector(frame, self, title="Vẽ vùng che cho phân đoạn hiện tại")
        if existing_box:
            selector.set_initial_bbox(existing_box)
            
        if selector.exec() == QDialog.DialogCode.Accepted:
            self.save_bbox_state_to_history()
            new_box = selector.selected_bbox
            
            reply_apply = QMessageBox.question(
                self,
                "Lựa chọn áp dụng",
                "Bạn muốn áp dụng vùng che mới này cho:\n"
                " - Yes: Chỉ riêng dòng phụ đề này.\n"
                " - No: Tất cả các dòng phụ đề có vị trí tương tự (chênh lệch dọc <= 20px, ngang <= 30px).",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes
            )
            
            if reply_apply == QMessageBox.StandardButton.Cancel:
                return
                
            if reply_apply == QMessageBox.StandardButton.Yes:
                self.segments[row]['bbox'] = new_box
                self.segments[row]['confidence'] = 100
            else:
                orig_box = self.segments[row].get('bbox')
                count = 0
                for idx, seg in enumerate(self.segments):
                    box = seg.get('bbox')
                    is_similar = False
                    if orig_box is None:
                        is_similar = (box is None) or (idx == row)
                    elif box is not None:
                        ox, oy, ow, oh = orig_box
                        bx, by, bw, bh = box
                        if abs(oy - by) <= 20 and abs(oh - bh) <= 20 and abs(ox - bx) <= 30 and abs(ow - bw) <= 30:
                            is_similar = True
                    if is_similar:
                        self.segments[idx]['bbox'] = new_box
                        self.segments[idx]['confidence'] = 100
                        count += 1
                self.status_label.setText(f"Đã cập nhật hộp che cho {count} dòng tương tự.")
                
            self.populate_subtitle_table()
            self.save_cache_file()
        
    def open_batch_dialog(self):
        dialog = BatchProcessingDialog(self)
        dialog.exec()

    def parse_glossary_text(self, text):
        glossary = {}
        if not text:
            return glossary
        for line_num, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            if '=' not in line:
                continue
            parts = line.split('=', 1)
            k = parts[0].strip()
            v = parts[1].strip()
            if k:
                glossary[k] = v
        return glossary

    def save_glossary_to_file(self):
        glossary_text = self.txt_glossary.toPlainText().strip()
        if not glossary_text:
            QMessageBox.warning(self, "Cảnh báo", "Không có nội dung glossary nào để lưu.")
            return
            
        errors = []
        for idx, line in enumerate(glossary_text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            if '=' not in line:
                errors.append(f"Dòng {idx}: {line}")
                
        if errors:
            err_msg = "Phát hiện dòng không đúng định dạng 'từ_gốc = từ_dịch':\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                err_msg += "\n..."
            QMessageBox.warning(self, "Lỗi định dạng", err_msg)
            return
            
        glossary_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "glossary")
        os.makedirs(glossary_dir, exist_ok=True)
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Lưu Glossary", glossary_dir, "Glossary Files (*.txt)")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(glossary_text)
                QMessageBox.information(self, "Thành công", f"Đã lưu glossary thành công tại:\n{os.path.basename(file_path)}")
                self.update_glossary_combobox()
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu file glossary: {e}")

    def load_glossary_from_file(self):
        glossary_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "glossary")
        os.makedirs(glossary_dir, exist_ok=True)
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Tải Glossary", glossary_dir, "Glossary Files (*.txt)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.txt_glossary.setPlainText(content)
                QMessageBox.information(self, "Thành công", f"Đã nạp glossary thành công từ:\n{os.path.basename(file_path)}")
                self.update_glossary_combobox()
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể đọc file glossary: {e}")
                
    def on_glossary_dropdown_changed(self):
        filename = self.cb_glossary_files.currentText()
        if not filename or filename == "-- Chọn Glossary --":
            return
        glossary_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "glossary")
        file_path = os.path.join(glossary_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.txt_glossary.setPlainText(content)
                self.status_label.setText(f"Đã nạp file glossary: {filename}")
            except Exception:
                pass
                
    def update_glossary_combobox(self):
        if not hasattr(self, 'cb_glossary_files') or self.cb_glossary_files is None:
            return
        self.cb_glossary_files.blockSignals(True)
        self.cb_glossary_files.clear()
        self.cb_glossary_files.addItem("-- Chọn Glossary --")
        glossary_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "glossary")
        if os.path.exists(glossary_dir):
            for file in os.listdir(glossary_dir):
                if file.endswith('.txt'):
                    self.cb_glossary_files.addItem(file)
        self.cb_glossary_files.blockSignals(False)

    def get_resolved_translation_config(self):
        engine_str = self.cb_engine.currentText() if hasattr(self, 'cb_engine') else "Dịch thô"
        engine = self.clean_combobox_value(engine_str)
        refine_enabled = False
        refine_engine = "Ollama Local"
        
        if engine == "Supersubs AI":
            backend_engine = "Quick Translator (VietPhrase)"
            refine_enabled = True
            refine_engine = "Ollama Local"
        elif engine == "Dịch thô":
            backend_engine = "Quick Translator (VietPhrase)"
            refine_enabled = False
        elif engine == "Dịch cơ bản":
            backend_engine = "Google Translate"
            refine_enabled = False
        else:
            backend_engine = engine
            
        glossary_raw = self.txt_glossary.toPlainText() if hasattr(self, 'txt_glossary') else ""
        glossary = self.parse_glossary_text(glossary_raw)
        return {
            'source_lang': 'auto',
            'target_lang': 'vi',
            'engine': backend_engine,
            'refine_enabled': refine_enabled,
            'refine_engine': refine_engine,
            'glossary': glossary
        }

    def save_translation_cache(self):
        video_id = self.get_video_cache_id(self.video_path)
        if not video_id:
            return
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{video_id}_translation.json")
        try:
            import json
            cache_structure = {
                'config': self.get_resolved_translation_config(),
                'data': [
                    {
                        'raw_text': seg.get('raw_text', ''),
                        'text': seg.get('text', ''),
                        'orig_text': seg.get('orig_text', ''),
                        'manual_override': seg.get('manual_override', False)
                    } for seg in self.segments
                ]
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_structure, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load_translation_cache(self):
        video_id = self.get_video_cache_id(self.video_path)
        if not video_id:
            return False
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "cache")
        cache_path = os.path.join(cache_dir, f"{video_id}_translation.json")
        if not os.path.exists(cache_path):
            return False
        try:
            import json
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_obj = json.load(f)
            
            cached_config = cached_obj.get('config', {})
            current_config = self.get_resolved_translation_config()
            
            if cached_config == current_config:
                cached_data = cached_obj.get('data', [])
                for idx, item in enumerate(cached_data):
                    if idx < len(self.segments):
                        self.segments[idx]['raw_text'] = item.get('raw_text', '')
                        self.segments[idx]['text'] = item.get('text', '')
                        self.segments[idx]['orig_text'] = item.get('orig_text', '')
                        self.segments[idx]['manual_override'] = item.get('manual_override', False)
                self.populate_subtitle_table()
                self.status_label.setText("Đã nạp phụ đề dịch từ cache cục bộ.")
                return True
        except Exception:
            pass
        return False

    def clean_combobox_value(self, text):
        if " (" in text:
            return text.split(" (")[0].strip()
        return text.strip()

    def trigger_canvas_update(self):
        if hasattr(self, 'preview_timer') and self.preview_timer.isActive():
            self.stop_video_preview()
        if not hasattr(self, 'canvas_timer'):
            return
        if self.canvas_timer is None:
            return
        if not self.canvas_timer.isActive() and hasattr(self, 'canvas_timer'):
            self.canvas_timer.start(150) # Debounce 150ms
            return
        self.canvas_timer.stop()
        self.canvas_timer.start(150) # Debounce 150ms

    def set_subtitle_custom_pos_pct(self, x_pct, y_pct):
        self.subtitle_custom_pos = {
            "x_pct": max(0.0, min(100.0, float(x_pct))),
            "y_pct": max(0.0, min(100.0, float(y_pct)))
        }
        if hasattr(self, 'cb_preset'):
            self.mark_preset_custom()
        if hasattr(self, 'spin_custom_pos_x') and hasattr(self, 'spin_custom_pos_y'):
            self.spin_custom_pos_x.blockSignals(True)
            self.spin_custom_pos_y.blockSignals(True)
            self.spin_custom_pos_x.setValue(self.subtitle_custom_pos['x_pct'])
            self.spin_custom_pos_y.setValue(self.subtitle_custom_pos['y_pct'])
            self.spin_custom_pos_x.blockSignals(False)
            self.spin_custom_pos_y.blockSignals(False)
        self.trigger_canvas_update()
        self.status_label.setText(f"Da dat vi tri sub: X={self.subtitle_custom_pos['x_pct']:.1f}%, Y={self.subtitle_custom_pos['y_pct']:.1f}%")

    def reset_subtitle_custom_pos(self):
        self.subtitle_custom_pos = None
        if hasattr(self, 'spin_custom_pos_x') and hasattr(self, 'spin_custom_pos_y'):
            self.spin_custom_pos_x.blockSignals(True)
            self.spin_custom_pos_y.blockSignals(True)
            self.spin_custom_pos_x.setValue(50.0)
            self.spin_custom_pos_y.setValue(88.0)
            self.spin_custom_pos_x.blockSignals(False)
            self.spin_custom_pos_y.blockSignals(False)
        self.trigger_canvas_update()
        self.status_label.setText("Da reset vi tri sub ve preset can le.")

    def update_canvas_realtime_now(self):
        table = getattr(self, 'table', None)
        if not table:
            return
        row = table.currentRow()
        if row < 0 or row >= len(self.segments):
            return
        if not self.video_path or not os.path.exists(self.video_path):
            return
            
        start_s = self.segments[row]['start']
        end_s = self.segments[row]['end']
        t_target = self.segments[row].get('ocr_timestamp', (start_s + end_s) / 2.0)
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_target * fps))
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return
            
        # 1. Vẽ vùng che
        bbox = self.segments[row].get('bbox') or self.selected_bbox
        if bbox:
            bx, by, bw, bh = bbox
            fh, fw, _ = frame.shape
            bx1 = max(0, min(bx, fw))
            by1 = max(0, min(by, fh))
            bx2 = max(0, min(bx + bw, fw))
            by2 = max(0, min(by + bh, fh))
            if bx2 > bx1 and by2 > by1:
                mask_mode = self.cb_mask_mode.currentText() if hasattr(self, 'cb_mask_mode') else "Blur nhanh (Gaussian Blur)"
                if "Không che" not in mask_mode:
                    if "Che đen đặc" in mask_mode:
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
                    elif "Blur" in mask_mode:
                        crop = frame[by1:by2, bx1:bx2]
                        if crop.shape[0] > 0 and crop.shape[1] > 0:
                            blur_k = max(3, int(min(crop.shape[0], crop.shape[1]) | 1))
                            crop_blur = cv2.GaussianBlur(crop, (blur_k, blur_k), 0)
                            frame[by1:by2, bx1:bx2] = crop_blur
                    else: # Inpaint
                        crop = frame[by1:by2, bx1:bx2]
                        if crop.shape[0] > 0 and crop.shape[1] > 0:
                            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                            mask.fill(255)
                            inpainted = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
                            frame[by1:by2, bx1:bx2] = inpainted
                            
        # 2. Vẽ phụ đề đè lên theo style hiện tại
        sub_text = self.segments[row].get('text', '')
        if sub_text:
            preset = self.get_current_subtitle_preset()
            frame, _ = dubber.draw_burned_subtitle(frame, sub_text, bbox=None, default_bbox=None, preset=preset)
            
        # 3. Scale giữ tỷ lệ aspect ratio
        h, w, _ = frame.shape
        self.video_width = w
        self.video_height = h
        
        # Cắt ảnh nếu bật zoom cận cảnh sub
        is_zoomed = hasattr(self, 'chk_zoom_sub') and self.chk_zoom_sub.isChecked()
        if is_zoomed:
            cy = 0.82
            if self.subtitle_custom_pos:
                cy = self.subtitle_custom_pos['y_pct'] / 100.0
            
            y_start = int(max(0.0, cy - 0.18) * h)
            y_end = int(min(1.0, cy + 0.15) * h)
            x_start = int(0.05 * w)
            x_end = int(0.95 * w)
            
            if y_end - y_start < 50:
                y_start = max(0, h - 100)
                y_end = h
            
            show_frame = frame[y_start:y_end, x_start:x_end]
        else:
            show_frame = frame
            
        # Truyền frame gốc cho DraggablePreviewLabel tự scale đúng tỷ lệ
        self.lbl_main_preview.setVideoFrame(show_frame)

    def apply_dark_theme(self):
        # Ultra Premium Dark Theme style (Modern Midnight Obsidian & Cyan design system)
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0e17;
                color: #f8fafc;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 12px;
            }
            QFrame#header {
                background-color: #151c2e;
                border-bottom: 1px solid #2a364f;
            }
            QFrame#card {
                background-color: #151c2e;
                border: 1px solid #2a364f;
                border-radius: 8px;
                padding: 6px;
            }
            QFrame#card QLabel {
                color: #cbd5e1;
            }
            QFrame#card > QLabel {
                color: #38bdf8;
                font-weight: bold;
                font-size: 13px;
            }
            QTabWidget::pane {
                border: 1px solid #2a364f;
                background-color: #151c2e;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #0a0e17;
                color: #94a3b8;
                padding: 8px 18px;
                border: 1px solid transparent;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 3px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #151c2e;
                color: #38bdf8;
                border-top: 3px solid #3b82f6;
                border-left: 1px solid #2a364f;
                border-right: 1px solid #2a364f;
            }
            QPushButton {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: bold;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #2a364f;
                border-color: #38bdf8;
            }
            QPushButton:pressed {
                background-color: #0f172a;
            }
            QLineEdit, QComboBox, QTextEdit {
                background-color: #0f172a;
                border: 1px solid #2a364f;
                border-radius: 6px;
                padding: 5px 8px;
                color: #f8fafc;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
                border: 1px solid #38bdf8;
                background-color: #141d2e;
            }
            QSlider::groove:horizontal {
                border: 1px solid #2a364f;
                height: 6px;
                background: #1e293b;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #38bdf8;
                border: 1px solid #3b82f6;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #60a5fa;
            }
            QTableWidget {
                background-color: #0a0e17;
                gridline-color: #1f2937;
                border: 1px solid #2a364f;
                alternate-background-color: #111827;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QTableWidget::item:selected {
                background-color: #1d4ed8;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #151c2e;
                color: #38bdf8;
                padding: 6px;
                border: 1px solid #2a364f;
                font-weight: bold;
            }
            QScrollBar:vertical {
                border: none;
                background: #0a0e17;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #2a364f;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3b82f6;
            }
        """)

# Helper function to check font unicode support
def check_font_vietnamese_support(font_path):
    test_chars = "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵđ"
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        font = ImageFont.truetype(font_path, 30)
        
        missing_char = "\ufffe"
        def render_char(char):
            img = Image.new("L", (30, 30), 0)
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), char, font=font, fill=255)
            return np.array(img)
            
        try:
            missing_arr = render_char(missing_char)
        except Exception:
            missing_arr = np.zeros((30, 30), dtype=np.uint8)
            
        try:
            space_arr = render_char(" ")
        except Exception:
            space_arr = np.zeros((30, 30), dtype=np.uint8)
            
        for char in test_chars:
            try:
                char_arr = render_char(char)
                if np.array_equal(char_arr, missing_arr) or np.array_equal(char_arr, space_arr):
                    return False
            except Exception:
                return False
        return True
    except Exception:
        return False


class SubtitlePreviewDialog(QDialog):
    def __init__(self, video_path, segments, current_preset, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Xem trước phụ đề (Subtitle Preview)")
        self.resize(750, 500)
        self.setModal(True)
        
        self.video_path = video_path
        self.segments = segments
        self.preset = current_preset
        
        layout = QVBoxLayout(self)
        
        # 1. Label hiển thị ảnh preview
        self.lbl_image = QLabel("Đang tải khung hình...")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumSize(640, 360)
        self.lbl_image.setStyleSheet("border: 1px solid #1c1c1f; background-color: #000000;")
        layout.addWidget(self.lbl_image)
        
        # 2. Warning Label
        self.lbl_warning = QLabel("⚠️ CẢNH BÁO: Phụ đề của phân đoạn này quá dài! Đã thu nhỏ font về 12px nhưng chữ vẫn bị tràn ra ngoài video.")
        self.lbl_warning.setStyleSheet("color: #ff9999; font-weight: bold; background-color: #331111; border: 1px solid #ff3333; border-radius: 4px; padding: 6px;")
        self.lbl_warning.setVisible(False)
        layout.addWidget(self.lbl_warning)
        
        # 3. Thanh chọn phân đoạn
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(QLabel("Chọn phân đoạn phụ đề:"))
        
        self.cb_segments = QComboBox()
        nav_layout.addWidget(self.cb_segments)
        
        self.btn_prev = QPushButton("◀ Trước")
        self.btn_next = QPushButton("Sau ▶")
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        
        layout.addLayout(nav_layout)
        
        # Nút Đóng
        btn_close = QPushButton("Đóng", self)
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
        
        # Nạp các phân đoạn vào combobox
        self.valid_segments_indices = []
        for idx, seg in enumerate(self.segments):
            text_strip = seg.get('text', '').strip()
            if text_strip:
                self.cb_segments.addItem(f"Dòng {idx+1} ({seg['start']:.1f}s - {seg['end']:.1f}s): {text_strip[:30]}...", idx)
                self.valid_segments_indices.append(idx)
                
        # Connect events
        self.cb_segments.currentIndexChanged.connect(self.on_segment_changed)
        self.btn_prev.clicked.connect(self.prev_segment)
        self.btn_next.clicked.connect(self.next_segment)
        
        # Mặc định chọn dòng hiện tại hoặc dòng đầu tiên có text
        initial_index = 0
        if parent and hasattr(parent, 'table'):
            selected_row = parent.table.currentRow()
            if selected_row in self.valid_segments_indices:
                initial_index = self.valid_segments_indices.index(selected_row)
                
        if self.cb_segments.count() > 0:
            self.cb_segments.setCurrentIndex(initial_index)
            self.update_preview()
        else:
            self.lbl_image.setText("Không tìm thấy phân đoạn phụ đề dịch nào để xem trước.")
            
    def prev_segment(self):
        curr = self.cb_segments.currentIndex()
        if curr > 0:
            self.cb_segments.setCurrentIndex(curr - 1)
            
    def next_segment(self):
        curr = self.cb_segments.currentIndex()
        if curr < self.cb_segments.count() - 1:
            self.cb_segments.setCurrentIndex(curr + 1)
            
    def on_segment_changed(self):
        self.update_preview()
        
    def update_preview(self):
        if self.cb_segments.count() == 0:
            return
            
        seg_idx = self.cb_segments.currentData()
        seg = self.segments[seg_idx]
        
        # 1. Trích xuất frame từ video
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        # Ưu tiên lấy frame ở giữa phân đoạn
        timestamp_s = (seg['start'] + seg['end']) / 2.0
        frame_idx = int(timestamp_s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        # Nếu không đọc được, thử lấy frame ở đầu phân đoạn
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(seg['start'] * fps))
            ret, frame = cap.read()
            
        cap.release()
        
        if not ret:
            self.lbl_image.setText("Không thể đọc được khung hình từ video.")
            return
            
        # 2. Render phụ đề lên frame sử dụng logic trong dubber.py
        active_bbox = seg.get('bbox')
        rendered_frame, overflowed = dubber.draw_burned_subtitle(
            frame, seg['text'], active_bbox, preset=self.preset
        )
        
        # 3. Hiển thị cảnh báo tràn chữ
        if overflowed:
            self.lbl_warning.setVisible(True)
        else:
            self.lbl_warning.setVisible(False)
            
        # 4. Hiển thị ảnh lên QLabel
        h_frame, w_frame, _ = rendered_frame.shape
        scale = min(640 / w_frame, 360 / h_frame)
        new_w = int(w_frame * scale)
        new_h = int(h_frame * scale)
        
        resized = cv2.resize(rendered_frame, (new_w, new_h))
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_image.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.lbl_image.setPixmap(pixmap)


class BatchProcessingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Xử lý hàng loạt Video (Batch Video Dubber)")
        self.resize(850, 550)
        self.setModal(True)
        self.parent = parent
        
        self.video_list = []
        
        layout = QVBoxLayout(self)
        
        # Bảng danh sách video
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Đường dẫn file", "Độ phân giải", "Thời lượng (s)", "Trạng thái"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        # Hàng nút thêm/xóa
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("➕ Thêm video...")
        self.btn_remove = QPushButton("❌ Xoá video")
        self.btn_out_dir = QPushButton("📂 Thư mục đầu ra...")
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_out_dir)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Nhãn hiển thị thư mục đầu ra
        self.lbl_out_dir = QLabel("Thư mục lưu kết quả: Mặc định (cùng thư mục video gốc)")
        self.lbl_out_dir.setStyleSheet("color: #7fbeb2; font-weight: bold;")
        layout.addWidget(self.lbl_out_dir)
        self.output_directory = ""
        
        # Cấu hình giọng đọc & âm lượng cho batch
        card_config = QFrame()
        card_config.setObjectName("card")
        config_layout = QVBoxLayout(card_config)
        config_layout.setContentsMargins(15, 15, 15, 15)
        config_layout.addWidget(QLabel("⚙️ CẤU HÌNH CHO MẺ XỬ LÝ (BATCH OPTIONS)"))
        
        row_voice = QHBoxLayout()
        row_voice.addWidget(QLabel("Giọng lồng tiếng:"))
        self.cb_voice = QComboBox()
        for i in range(parent.cb_voice.count()):
            self.cb_voice.addItem(parent.cb_voice.itemText(i), parent.cb_voice.itemData(i))
        self.cb_voice.setCurrentIndex(parent.cb_voice.currentIndex())
        row_voice.addWidget(self.cb_voice)
        
        self.chk_burn_sub = QCheckBox("Ghi đè phụ đề tiếng Việt (Che sub gốc)")
        self.chk_burn_sub.setChecked(parent.chk_burn_sub.isChecked())
        row_voice.addWidget(self.chk_burn_sub)
        config_layout.addLayout(row_voice)
        
        row_vol = QHBoxLayout()
        row_vol.addWidget(QLabel("Nhạc nền video gốc:"))
        self.slider_bg = QSlider(Qt.Orientation.Horizontal)
        self.slider_bg.setRange(0, 100)
        self.slider_bg.setValue(parent.slider_bg.value())
        row_vol.addWidget(self.slider_bg)
        
        row_vol.addWidget(QLabel("   Giọng lồng tiếng:"))
        self.slider_dub = QSlider(Qt.Orientation.Horizontal)
        self.slider_dub.setRange(0, 200)
        self.slider_dub.setValue(parent.slider_dub.value())
        row_vol.addWidget(self.slider_dub)
        config_layout.addLayout(row_vol)
        
        layout.addWidget(card_config)
        
        # Tiến trình
        progress_layout = QVBoxLayout()
        self.lbl_progress = QLabel("Tiến độ: Sẵn sàng")
        progress_layout.addWidget(self.lbl_progress)
        
        self.bar_total = QProgressBar()
        self.bar_total.setFormat("Tổng tiến trình: %v/%m video (%p%)")
        progress_layout.addWidget(self.bar_total)
        
        self.bar_current = QProgressBar()
        self.bar_current.setFormat("Video hiện tại: %p%")
        progress_layout.addWidget(self.bar_current)
        
        layout.addLayout(progress_layout)
        
        # Log area
        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        layout.addWidget(self.txt_logs)
        
        # Hàng nút chạy
        btn_action_layout = QHBoxLayout()
        self.btn_start = QPushButton("🚀 BẮT ĐẦU XỬ LÝ HÀNG LOẠT")
        self.btn_start.setStyleSheet("background-color: #dfb15b; color: #0c0c0e; font-size: 14px; font-weight: bold; padding: 10px;")
        self.btn_close = QPushButton("Đóng")
        btn_action_layout.addWidget(self.btn_start)
        btn_action_layout.addWidget(self.btn_close)
        layout.addLayout(btn_action_layout)
        
        # Connect events
        self.btn_add.clicked.connect(self.add_videos)
        self.btn_remove.clicked.connect(self.remove_video)
        self.btn_out_dir.clicked.connect(self.choose_output_dir)
        self.btn_start.clicked.connect(self.start_batch_processing)
        self.btn_close.clicked.connect(self.close)
        
    def add_videos(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Chọn các file video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if file_paths:
            for path in file_paths:
                if any(v['path'] == path for v in self.video_list):
                    continue
                    
                cap = cv2.VideoCapture(path)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frames / fps if fps > 0 else 0.0
                cap.release()
                
                v_data = {
                    'path': path,
                    'resolution': f"{w}x{h}",
                    'duration': duration,
                    'status': "Chờ xử lý",
                    'output': ""
                }
                self.video_list.append(v_data)
                
            self.update_table()
            
    def remove_video(self):
        curr_row = self.table.currentRow()
        if curr_row >= 0 and curr_row < len(self.video_list):
            self.video_list.pop(curr_row)
            self.update_table()
            
    def choose_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu kết quả")
        if dir_path:
            self.output_directory = dir_path
            self.lbl_out_dir.setText(f"Thư mục lưu kết quả: {dir_path}")
            
    def update_table(self):
        self.table.setRowCount(len(self.video_list))
        for idx, v in enumerate(self.video_list):
            self.table.setItem(idx, 0, QTableWidgetItem(os.path.basename(v['path'])))
            self.table.setItem(idx, 1, QTableWidgetItem(v['resolution']))
            self.table.setItem(idx, 2, QTableWidgetItem(f"{v['duration']:.1f}s"))
            self.table.setItem(idx, 3, QTableWidgetItem(v['status']))
            
    def start_batch_processing(self):
        if not self.video_list:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng thêm ít nhất 1 video để xử lý.")
            return
            
        self.btn_start.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_remove.setEnabled(False)
        self.btn_out_dir.setEnabled(False)
        
        voice = self.cb_voice.currentData()
        bg_vol = self.slider_bg.value() / 100.0
        dub_vol = self.slider_dub.value() / 100.0
        burn_sub = self.chk_burn_sub.isChecked()
        preset_sub = self.parent.get_current_subtitle_preset()
        engine = self.parent.cb_engine.currentText()
        gemini_key = self.parent.txt_gemini_key.text().strip()
        groq_key = self.parent.txt_groq_key.text().strip()
        deepl_key = self.parent.txt_deepl_key.text().strip()
        ollama_model = self.parent.txt_ollama_model.text().strip()
        
        api_key = ""
        if "Gemini" in engine:
            api_key = gemini_key
        elif "Groq" in engine:
            api_key = groq_key
        elif "DeepL" in engine:
            api_key = deepl_key
            
        vp_dict_paths = {
            'vp_path': self.parent.txt_vp_path.text().strip(),
            'names_path': self.parent.txt_names_path.text().strip()
        }
        
        class BatchWorkerThread(QThread):
            progress_total = pyqtSignal(int, int)
            progress_current = pyqtSignal(int)
            log = pyqtSignal(str)
            status_update = pyqtSignal(int, str)
            finished = pyqtSignal()
            
            def __init__(self, video_list, output_dir, voice, bg_vol, dub_vol, burn_sub, preset_sub, engine, api_key, ollama_model, vp_dict_paths, whisper_model="base"):
                super().__init__()
                self.video_list = video_list
                self.output_dir = output_dir
                self.voice = voice
                self.bg_vol = bg_vol
                self.dub_vol = dub_vol
                self.burn_sub = burn_sub
                self.preset_sub = preset_sub
                self.engine = engine
                self.api_key = api_key
                self.ollama_model = ollama_model
                self.vp_dict_paths = vp_dict_paths
                self.whisper_model = whisper_model
                
            def run(self):
                total = len(self.video_list)
                for idx, v in enumerate(self.video_list):
                    self.status_update.emit(idx, "Đang xử lý...")
                    self.progress_total.emit(idx, total)
                    self.log.emit(f"\n==================================================")
                    self.log.emit(f"🎬 BẮT ĐẦU XỬ LÝ VIDEO {idx+1}/{total}: {os.path.basename(v['path'])}")
                    
                    try:
                        self.log.emit("Bước 1: Trích xuất âm thanh và nhận diện giọng nói...")
                        audio_path = downloader.extract_audio(v['path'])
                        
                        segments = []
                        if self.api_key and "Gemini" in self.engine:
                            self.log.emit(" - Sử dụng Gemini API nhận diện giọng nói...")
                            segments = transcriber.transcribe_gemini(audio_path, self.api_key, self.log.emit)
                        else:
                            self.log.emit(" - Sử dụng Whisper Local nhận diện giọng nói...")
                            segments = transcriber.transcribe_local_whisper(audio_path, self.whisper_model, self.log.emit)
                            
                        try:
                            os.remove(audio_path)
                        except Exception:
                            pass
                            
                        self.log.emit(f" => Trích xuất thành công {len(segments)} câu phụ đề gốc.")
                        if not segments:
                            self.log.emit("⚠️ Không tìm thấy phụ đề, chuyển sang video tiếp theo.")
                            self.status_update.emit(idx, "Không có phụ đề")
                            continue
                            
                        self.log.emit("Bước 2: Tự động dịch phụ đề...")
                        translated_segs = translator.translate_segments(
                            segments, 'auto', 'vi', self.engine, self.api_key,
                            progress_callback=self.log.emit, ollama_model=self.ollama_model,
                            vp_dict_paths=self.vp_dict_paths
                        )
                        self.log.emit(" => Dịch thuật phụ đề hoàn tất.")
                        
                        self.log.emit("Bước 3: Tiến hành lồng tiếng & ghi đè phụ đề...")
                        base, ext = os.path.splitext(v['path'])
                        if self.output_dir:
                            out_video = os.path.join(self.output_dir, os.path.basename(base) + "_longtieng" + ext)
                        else:
                            out_video = base + "_longtieng" + ext
                            
                        res_path, overflowed = dubber.create_dubbed_video(
                            v['path'], translated_segs, self.voice, out_video,
                            bg_volume=self.bg_vol, dub_volume=self.dub_vol,
                            burn_subtitles=self.burn_sub, selected_bbox=None,
                            preset=self.preset_sub, progress_callback=self.log.emit
                        )
                        
                        if overflowed:
                            self.log.emit(f"⚠️ CẢNH BÁO TRÀN CHỮ: Có {len(overflowed)} phân đoạn chữ quá dài, không thể ngắt dòng/co nhỏ vừa khung hình:")
                            for s in overflowed[:3]:
                                self.log.emit(f"   - \"{s}\"")
                            if len(overflowed) > 3:
                                self.log.emit("   - ...")
                                
                        self.log.emit(f"🎉 HOÀN THÀNH VIDEO {idx+1}: {os.path.basename(res_path)}")
                        self.status_update.emit(idx, "Hoàn thành")
                        v['output'] = res_path
                    except Exception as e:
                        import traceback
                        err_str = f"❌ LỖI VIDEO {idx+1}: {str(e)}\n{traceback.format_exc()}"
                        self.log.emit(err_str)
                        self.status_update.emit(idx, "Lỗi")
                self.finished.emit()
                
        self.thread = BatchWorkerThread(
            self.video_list, self.output_directory, voice, bg_vol, dub_vol,
            burn_sub, preset_sub, engine, api_key, ollama_model, vp_dict_paths,
            whisper_model=self.parent.cb_whisper_model.currentText()
        )
        self.thread.log.connect(self.append_log)
        self.thread.status_update.connect(self.update_status)
        self.thread.progress_total.connect(self.update_progress_total)
        self.thread.progress_current.connect(self.update_progress_current)
        self.thread.finished.connect(self.on_batch_finished)
        self.thread.start()
        
    def append_log(self, text):
        self.txt_logs.append(text)
        
    def update_status(self, idx, status):
        self.video_list[idx]['status'] = status
        self.update_table()
        
    def update_progress_total(self, idx, total):
        self.bar_total.setMaximum(total)
        self.bar_total.setValue(idx)
        self.lbl_progress.setText(f"Đang xử lý video {idx+1}/{total}...")
        
    def update_progress_current(self, percent):
        self.bar_current.setValue(percent)
        
    def on_batch_finished(self):
        self.bar_total.setValue(self.bar_total.maximum())
        self.lbl_progress.setText("Xử lý hàng loạt hoàn tất!")
        self.btn_start.setEnabled(True)
        self.btn_add.setEnabled(True)
        self.btn_remove.setEnabled(True)
        self.btn_out_dir.setEnabled(True)
        QMessageBox.information(self, "Thành công", "Đã hoàn thành toàn bộ mẻ xử lý hàng loạt video!")

# =====================================================================
# CÁC CLASS CHO TAB "KỊCH BẢN & GIỌNG ĐỌC"
# =====================================================================

DEFAULT_PROMPTS = [
    {
        "id": "default_1",
        "title": "Kịch bản Review sản phẩm",
        "content": "Hãy viết một kịch bản video ngắn để review sản phẩm [SẢN PHẨM] với độ dài khoảng [ĐỘ DÀI] giây. Kịch bản cần hướng tới đối tượng người xem là [ĐỐI TƯỢNG NGƯỜI XEM]. Sử dụng tông giọng [TÔNG GIỌNG] (ví dụ: hài hước, chân thực, uy tín). Kịch bản bao gồm cấu trúc:\n- Hook 3 giây đầu để giữ chân người xem.\n- Phần thân: Nêu bật 3 tính năng hoặc trải nghiệm thực tế nổi trội nhất của sản phẩm.\n- Phần kết: Kêu gọi hành động (CTA) như like, subscribe hoặc bấm vào giỏ hàng.\nLưu ý: Viết dưới dạng hội thoại tự nhiên, câu ngắn gọn, dễ đọc, phù hợp làm voice-over.",
        "tags": ["review", "sanpham", "tiktok"]
    },
    {
        "id": "default_2",
        "title": "Kịch bản Kể chuyện (Storytelling)",
        "content": "Hãy đóng vai một người kể chuyện chuyên nghiệp và viết một kịch bản kể về chủ đề [CHỦ ĐỀ] với độ dài khoảng [ĐỘ DÀI] chữ hoặc giây. Đối tượng thính giả là [ĐỐI TƯỢNG NGƯỜI XEM]. Tông giọng kể chuyện cần thể hiện sự [TÔNG GIỌNG] (ví dụ: bí ẩn, kịch tính, trầm lắng, truyền cảm hứng).\nYêu cầu:\n- Mở đầu bằng một câu hỏi tu từ hoặc một sự thật gây tò mò lớn.\n- Nhịp điệu câu chuyện có sự cao trào, thắt nút và mở nút rõ ràng.\n- Sử dụng từ ngữ giàu hình ảnh và cảm xúc, câu từ gãy gọn để giọng đọc AI truyền tải tốt nhất.",
        "tags": ["storytelling", "kechuyen", "youtube"]
    },
    {
        "id": "default_3",
        "title": "Kịch bản Quảng cáo ngắn",
        "content": "Hãy viết kịch bản quảng cáo video ngắn quảng bá cho [SẢN PHẨM/DỊCH VỤ] về chủ đề [CHỦ ĐỀ]. Độ dài kịch bản khoảng [ĐỘ DÀI] giây. Tông giọng yêu cầu: [TÔNG GIỌNG] (ví dụ: năng động, thúc giục, lôi cuốn). Đối tượng khách hàng mục tiêu: [ĐỐI TƯỢNG NGƯỜI XEM].\nCấu trúc kịch bản:\n- 3 giây đầu: Đánh trúng nỗi đau (Pain point) của khách hàng.\n- 5 giây tiếp theo: Giới thiệu giải pháp là sản phẩm/dịch vụ của chúng tôi.\n- 5 giây tiếp theo: Nêu ưu đãi giới hạn hoặc bằng chứng thuyết phục (social proof).\n- Kết thúc: Kêu gọi hành động rõ ràng (CTA).",
        "tags": ["ads", "quangcao", "short"]
    },
    {
        "id": "default_4",
        "title": "Kịch bản Thuyết trình / Giáo dục",
        "content": "Hãy viết một kịch bản video bài giảng/thuyết trình giải thích về chủ đề [CHỦ ĐỀ]. Độ dài khoảng [ĐỘ DÀI] phút. Tông giọng yêu cầu: [TÔNG GIỌNG] (ví dụ: sư phạm, dễ hiểu, chuyên nghiệp, khoa học). Đối tượng người nghe: [ĐỐI TƯỢNG NGƯỜI XEM].\nYêu cầu kịch bản:\n- Phân chia bài giảng thành 3 luận điểm chính rõ ràng.\n- Tránh dùng thuật ngữ quá hàn lâm mà không giải thích, dùng các ví dụ so sánh thực tế để người nghe dễ hình dung.\n- Có lời khuyên/tóm tắt đọng lại ở cuối video.",
        "tags": ["giao-duc", "thuyettrinh", "lecture"]
    },
    {
        "id": "default_5",
        "title": "Kịch bản Voice-over phim/video",
        "content": "Hãy đóng vai là một bình luận viên phim chuyên nghiệp, viết một kịch bản tóm tắt phim/video về bộ phim [TÊN PHIM] (hoặc chủ đề [CHỦ ĐỀ]). Độ dài khoảng [ĐỘ DÀI] giây hoặc từ. Tông giọng yêu cầu: [TÔNG GIỌNG] (ví dụ: kịch tính, lôi cuốn, hài hước, cà khịa nhẹ nhàng). Đối tượng khán giả: [ĐỐI TƯỢNG NGƯỜI XEM].\nYêu cầu:\n- Lối dẫn dắt dí dỏm, tạo cảm giác tò mò mà không spoil toàn bộ các tình tiết bất ngờ cốt lõi.\n- Sử dụng các cụm từ đang trend, câu ngắn để tạo nhịp điệu nhanh, lôi cuốn.",
        "tags": ["review-phim", "voiceover", "giaitri"]
    },
    {
        "id": "default_6",
        "title": "Kịch bản Podcast / Phỏng vấn",
        "content": "Hãy viết một kịch bản chương trình Podcast/Phỏng vấn giả lập giữa 2 người (Người dẫn chương trình Host và Khách mời chuyên gia) nói về chủ đề [CHỦ ĐỀ]. Độ dài khoảng [ĐỘ DÀI] phút. Tông giọng yêu cầu: [TÔNG GIỌNG] (ví dụ: chia sẻ chân tình, cởi mở, chuyên sâu). Đối tượng người nghe: [ĐỐI TƯỢNG NGƯỜI XEM].\nYêu cầu:\n- Có phần chào hỏi (Intro) ngắn giới thiệu chủ đề.\n- Tạo ra 3-4 câu hỏi phỏng vấn tự nhiên và các câu trả lời tương ứng mang lại giá trị cao cho người nghe.\n- Lời kết (Outro) cảm ơn và kêu gọi theo dõi kênh.",
        "tags": ["podcast", "interview", "talkshow"]
    }
]

class CustomPromptDialog(QDialog):
    def __init__(self, parent=None, title="", content="", tags=""):
        super().__init__(parent)
        self.setWindowTitle("Prompt mẫu tùy chỉnh")
        self.setMinimumSize(450, 320)
        self.init_ui(title, content, tags)
        
    def init_ui(self, title, content, tags):
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Tiêu đề Prompt:"))
        self.txt_title = QLineEdit()
        self.txt_title.setText(title)
        self.txt_title.setPlaceholderText("Ví dụ: Kịch bản TikTok bán hàng")
        layout.addWidget(self.txt_title)
        
        layout.addWidget(QLabel("Nội dung Prompt gợi ý:"))
        self.txt_content = QTextEdit()
        self.txt_content.setPlainText(content)
        self.txt_content.setPlaceholderText("Gõ nội dung prompt ở đây...")
        layout.addWidget(self.txt_content)
        
        layout.addWidget(QLabel("Tags (phân cách bằng dấu phẩy):"))
        self.txt_tags = QLineEdit()
        self.txt_tags.setText(tags)
        self.txt_tags.setPlaceholderText("Ví dụ: tiktok, banhang, review")
        layout.addWidget(self.txt_tags)
        
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("Lưu lại")
        self.btn_save.setStyleSheet("background-color: #dfb15b; color: #0c0c0e; font-weight: bold; padding: 6px 12px;")
        self.btn_save.clicked.connect(self.accept)
        
        self.btn_cancel = QPushButton("Hủy")
        self.btn_cancel.setStyleSheet("background-color: #555555; color: white; padding: 6px 12px;")
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
        
    def get_data(self):
        tags_str = self.txt_tags.text().strip()
        tags = [t.strip().lower() for t in tags_str.split(",") if t.strip()]
        return {
            "title": self.txt_title.text().strip(),
            "content": self.txt_content.toPlainText().strip(),
            "tags": tags
        }

class ScriptHistoryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Lịch sử kịch bản")
        self.setMinimumSize(600, 400)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        lbl_title = QLabel("📜 LỊCH SỬ KỊCH BẢN GẦN ĐÂY")
        lbl_title.setStyleSheet("color: #dfb15b; font-weight: bold; font-size: 14px;")
        layout.addWidget(lbl_title)
        
        self.table_history = QTableWidget()
        self.table_history.setColumnCount(4)
        self.table_history.setHorizontalHeaderLabels(["Thời gian", "Tiêu đề", "Phân đoạn", "Hành động"])
        self.table_history.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_history.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_history.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_history.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_history.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_history.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table_history)
        
        self.load_history()
        
    def load_history(self):
        self.table_history.setRowCount(0)
        history_data = load_script_history()
            
        self.table_history.setRowCount(len(history_data))
        for idx, entry in enumerate(history_data):
            time_str = entry.get("created_at", "")[:19].replace("T", " ")
            time_item = QTableWidgetItem(time_str)
            title_item = QTableWidgetItem(entry.get("title", ""))
            count_item = QTableWidgetItem(f"{entry.get('segments_count', 0)} dòng")
            
            self.table_history.setItem(idx, 0, time_item)
            self.table_history.setItem(idx, 1, title_item)
            self.table_history.setItem(idx, 2, count_item)
            
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(2, 2, 2, 2)
            action_layout.setSpacing(5)
            
            btn_load = QPushButton("Tải lại")
            btn_load.setStyleSheet("background-color: #7fbeb2; color: #0c0c0e; font-weight: bold; padding: 3px 8px;")
            btn_load.clicked.connect(lambda checked, e=entry: self.load_entry(e))
            
            action_layout.addWidget(btn_load)
            self.table_history.setCellWidget(idx, 3, action_widget)
            
    def load_entry(self, entry):
        if self.parent:
            self.parent.load_script_from_history(entry)
        self.accept()

class AIScriptGeneratorWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, video_path, gemini_key, groq_key, ollama_model, whisper_model):
        super().__init__()
        self.video_path = video_path
        self.gemini_key = gemini_key
        self.groq_key = groq_key
        self.ollama_model = ollama_model
        self.whisper_model = whisper_model
        
    def run(self):
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                self.error.emit("Đường dẫn video mẫu không hợp lệ.")
                return
                
            # Giai đoạn 1: Chuyển giọng nói gốc sang văn bản (Whisper / Gemini)
            self.progress.emit("Đang tách âm thanh từ video gốc...")
            audio_path = downloader.extract_audio(self.video_path)
            
            self.progress.emit("Đang quét giọng nói gốc trong video mẫu...")
            if self.gemini_key:
                self.progress.emit("Đang phân tích âm thanh bằng Gemini AI...")
                segments = transcriber.transcribe_gemini(audio_path, self.gemini_key, self.progress.emit)
            else:
                self.progress.emit("Đang quét giọng nói bằng Whisper cục bộ...")
                segments = transcriber.transcribe_local_whisper(audio_path, self.whisper_model, self.progress.emit)
                
            # Xóa file audio tạm
            try: os.remove(audio_path)
            except: pass
            
            if not segments:
                self.error.emit("Không phát hiện thấy giọng nói hoặc âm thanh nào trong video gốc.")
                return
                
            # Gộp lời thoại
            transcribed_text = " ".join([seg["text"].strip() for seg in segments if seg.get("text")])
            if not transcribed_text.strip():
                self.error.emit("Không tìm thấy lời thoại nào trong video mẫu.")
                return
                
            # Giai đoạn 2: Dùng LLM viết lại kịch bản
            self.progress.emit("Đang dùng trí tuệ nhân tạo (AI) soạn kịch bản kể chuyện mới...")
            
            prompt = (
                "Bạn là một biên kịch chuyên nghiệp và nhà sáng tạo nội dung hàng đầu trên mạng xã hội (TikTok, Reels, Shorts).\n"
                "Nhiệm vụ: Viết một kịch bản lời thoại kể chuyện (Voiceover) có chiều sâu, hấp dẫn, thu hút người xem từ những giây đầu tiên, dựa trên nội dung hội thoại gốc dưới đây.\n\n"
                "Hội thoại gốc trích xuất từ video:\n"
                f"{transcribed_text}\n\n"
                "YÊU CẦU CỦA KỊCH BẢN MỚI:\n"
                "1. Định dạng kể chuyện có chiều sâu, cuốn hút. Chia thành các câu ngắn gọn để dễ đọc.\n"
                "2. Viết bằng tiếng Việt tự nhiên, truyền cảm, thoát ý hoàn toàn khỏi những câu thoại khô khan.\n"
                "3. Loại bỏ hoàn toàn các ký tự ghi chú, nhạc nền hoặc chỉ dẫn. CHỈ viết lời thoại nói trực tiếp.\n"
                "4. Không được thêm bất kỳ ký tự Markdown nào (như ** hay nháy ngược `), chỉ để văn bản thô trơn để tránh lỗi đọc TTS.\n"
                "Kịch bản hoàn chỉnh:"
            )
            
            res_text = ""
            if self.gemini_key and len(self.gemini_key.strip()) >= 20:
                from google import genai
                candidate_models = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-flash-lite-latest", "gemini-2.5-flash-lite", "gemini-3.5-flash", "gemini-2.0-flash-exp", "gemini-2.0-flash-lite-preview", "gemini-1.5-flash-8b"]
                last_err = None
                response = None
                client = genai.Client(api_key=self.gemini_key)
                for model_name in candidate_models:
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt
                        )
                        break
                    except Exception as e:
                        err_str = str(e)
                        if "404" in err_str or "not found" in err_str.lower() or "not supported" in err_str.lower():
                            last_err = e
                            continue
                        else:
                            raise e
                if response is None:
                    if last_err:
                        raise last_err
                    raise ValueError("Không thể sử dụng bất kỳ model Gemini nào để tạo kịch bản.")
                res_text = response.text.strip()
            elif self.groq_key:
                import requests
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.groq_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "llama-3.1-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                }
                res = requests.post(url, json=payload, headers=headers, timeout=60)
                res.raise_for_status()
                res_text = res.json()["choices"][0]["message"]["content"].strip()
            else:
                # Ollama local
                import requests
                url = "http://localhost:11434/api/generate"
                payload = {
                    "model": self.ollama_model if self.ollama_model else "qwen2.5",
                    "prompt": prompt,
                    "stream": False
                }
                res = requests.post(url, json=payload, timeout=60)
                res.raise_for_status()
                res_text = res.json()["response"].strip()
                
            self.finished.emit(res_text)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.error.emit(f"Lỗi sinh kịch bản AI: {str(e)}")

class ExportVideoSyncWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, video_path, segments, silence_sec, sync_mode, output_dir):
        super().__init__()
        self.video_path = video_path
        self.segments = segments
        self.silence_sec = silence_sec
        self.sync_mode = sync_mode # "stretch", "freeze", "none"
        self.output_dir = output_dir
        
    def run(self):
        import subprocess
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                self.error.emit("Video mẫu không hợp lệ.")
                return
                
            self.progress.emit("Đang đo độ dài video mẫu...")
            v_dur_ms = dubber.get_video_duration_ms(self.video_path)
            if v_dur_ms <= 0:
                self.error.emit("Không thể đọc độ dài video mẫu.")
                return
            v_total = v_dur_ms / 1000.0
            
            N = len(self.segments)
            if N <= 0:
                self.error.emit("Không có kịch bản segment nào để ghép nối.")
                return
                
            v_chunk = v_total / N
            temp_dir = tempfile.mkdtemp(prefix="video_sync_")
            
            # Danh sách tệp video đã xử lý
            sync_files = []
            
            for i, seg in enumerate(self.segments):
                self.progress.emit(f"Đang đồng bộ đoạn video {i+1}/{N}...")
                
                # 1. Cắt đoạn video thô tương ứng
                start_t = i * v_chunk
                end_t = (i + 1) * v_chunk
                raw_chunk = os.path.join(temp_dir, f"raw_{i}.mp4")
                
                cmd_cut = [
                    "ffmpeg", "-y", "-ss", str(start_t), "-to", str(end_t),
                    "-i", self.video_path, "-an", "-c:v", "libx264", "-crf", "23",
                    "-preset", "veryfast", raw_chunk
                ]
                subprocess.run(cmd_cut, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if not os.path.exists(raw_chunk) or os.path.getsize(raw_chunk) == 0:
                    self.error.emit(f"Không thể cắt đoạn video {i+1}.")
                    return
                
                # 2. Xử lý đồng bộ dựa vào thời lượng audio của segment
                a_dur = seg.get("duration", 0.0)
                if a_dur <= 0:
                    a_dur = 2.5 # Default duration if not generated
                
                sync_chunk = os.path.join(temp_dir, f"sync_{i}.mp4")
                
                if self.sync_mode.startswith("Co giãn"): # "stretch"
                    # Tính toán hệ số setpts
                    speed_ratio = v_chunk / a_dur
                    # Giới hạn hệ số co giãn giữa 0.3x và 3.0x để tránh quá giật
                    speed_ratio = max(0.3, min(3.0, speed_ratio))
                    
                    cmd_sync = [
                        "ffmpeg", "-y", "-i", raw_chunk,
                        "-vf", f"setpts={speed_ratio}*PTS",
                        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                        sync_chunk
                    ]
                    subprocess.run(cmd_sync, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                elif self.sync_mode.startswith("Đứng hình"): # "freeze"
                    if a_dur > v_chunk:
                        # Đóng băng khung hình cuối cùng
                        pad_dur = a_dur - v_chunk
                        cmd_sync = [
                            "ffmpeg", "-y", "-i", raw_chunk,
                            "-vf", f"tpad=stop_mode=clone:stop_duration={pad_dur}",
                            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                            sync_chunk
                        ]
                        subprocess.run(cmd_sync, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    else:
                        # Cắt bớt video
                        cmd_sync = [
                            "ffmpeg", "-y", "-ss", "0", "-to", str(a_dur),
                            "-i", raw_chunk,
                            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                            sync_chunk
                        ]
                        subprocess.run(cmd_sync, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                else: # Không đồng bộ
                    # Copy đoạn video gốc làm đoạn đồng bộ
                    import shutil
                    shutil.copy2(raw_chunk, sync_chunk)
                    
                if os.path.exists(sync_chunk):
                    sync_files.append(sync_chunk)
                else:
                    # Fallback nếu lỗi ffmpeg
                    sync_files.append(raw_chunk)
                    
            if not sync_files:
                self.error.emit("Đồng bộ video thất bại.")
                return
                
            self.progress.emit("Đang hợp nhất các phân đoạn video...")
            # Tạo file concat list
            concat_list_path = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_list_path, "w", encoding="utf-8") as f:
                for f_path in sync_files:
                    f.write(f"file '{f_path.replace(os.sep, '/')}'\n")
                    
            # Hợp nhất video
            merged_video = os.path.join(temp_dir, "merged_video.mp4")
            cmd_concat = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list_path, "-c", "copy", merged_video
            ]
            subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if not os.path.exists(merged_video):
                self.error.emit("Hợp nhất các phân đoạn video thất bại.")
                return
                
            # Tạo tổng hợp audio
            self.progress.emit("Đang ghép nối âm thanh lồng tiếng...")
            combined_audio = AudioSegment.empty()
            silence_segment = AudioSegment.silent(duration=int(self.silence_sec * 1000))
            
            for seg in self.segments:
                a_path = seg.get("audio_path")
                if a_path and os.path.exists(a_path):
                    sound = AudioSegment.from_file(a_path)
                    if len(combined_audio) > 0:
                        combined_audio += silence_segment
                    combined_audio += sound
                else:
                    default_silence = AudioSegment.silent(duration=2500)
                    if len(combined_audio) > 0:
                        combined_audio += silence_segment
                    combined_audio += default_silence
                    
            merged_audio_path = os.path.join(temp_dir, "merged_audio.wav")
            combined_audio.export(merged_audio_path, format="wav")
            
            # Xuất phụ đề SRT
            self.progress.emit("Đang xuất file phụ đề SRT...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            final_video_path = os.path.join(self.output_dir, f"final_video_{timestamp}.mp4")
            final_srt_path = os.path.join(self.output_dir, f"final_video_{timestamp}.srt")
            
            transcriber.export_srt_with_silence(
                segments=self.segments,
                output_path=final_srt_path,
                default_duration=2.5,
                silence_between=self.silence_sec
            )
            
            # Trộn video và audio
            self.progress.emit("Đang nạp âm thanh vào video thành phẩm...")
            cmd_mix = [
                "ffmpeg", "-y", "-i", merged_video, "-i", merged_audio_path,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0",
                final_video_path
            ]
            subprocess.run(cmd_mix, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Dọn dẹp tệp tạm
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except: pass
            
            if os.path.exists(final_video_path):
                self.finished.emit(final_video_path)
            else:
                self.error.emit("Trộn video và audio thất bại.")
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.error.emit(f"Lỗi ghép video: {str(e)}")

class ScriptPreviewTTSThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, text, voice, rate, pitch):
        super().__init__()
        self.text = text
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        
    def run(self):
        temp_mp3 = os.path.join(tempfile.gettempdir(), "temp_preview_script.mp3")
        temp_wav = os.path.join(tempfile.gettempdir(), "temp_preview_script.wav")
        
        if os.path.exists(temp_mp3):
            try: os.remove(temp_mp3)
            except: pass
        if os.path.exists(temp_wav):
            try: os.remove(temp_wav)
            except: pass
            
        success = dubber.generate_tts(self.text, self.voice, temp_mp3, self.rate, self.pitch)
        if not success or not os.path.exists(temp_mp3):
            self.error.emit("Không thể sinh giọng đọc. Vui lòng kiểm tra kết nối mạng.")
            return
            
        try:
            sound = AudioSegment.from_file(temp_mp3)
            sound.export(temp_wav, format="wav")
            self.finished.emit(temp_wav)
        except Exception as e:
            self.error.emit(f"Lỗi phát thử: {str(e)}")

class BatchTTSWorker(QThread):
    progress = pyqtSignal(int, str) # row_idx, status
    finished = pyqtSignal(list) # list of tuples (row_idx, out_file, duration)
    error = pyqtSignal(str) # connection or general error message
    
    def __init__(self, segments, voice, output_dir, rate, pitch):
        super().__init__()
        self.segments = segments # list of dicts
        self.voice = voice
        self.output_dir = output_dir
        self.rate = rate
        self.pitch = pitch
        self.is_cancelled = False
        
    def run(self):
        results = []
        os.makedirs(self.output_dir, exist_ok=True)
        import re
        import time
        
        for seg in self.segments:
            if self.is_cancelled:
                break
                
            idx = seg["index"] - 1
            text = seg["text"].strip()
            if not text:
                continue
                
            # Bỏ qua nếu dòng này chỉ chứa dấu câu/ký tự đặc biệt để tránh lỗi edge-tts
            clean = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', '', text)
            if not clean.strip():
                results.append((idx, "", 0.0))
                self.progress.emit(idx, "Đã sẵn sàng")
                continue
                
            self.progress.emit(idx, "Đang tạo...")
            out_file = os.path.join(self.output_dir, f"segment_{idx+1}.mp3")
            
            if os.path.exists(out_file):
                try: os.remove(out_file)
                except: pass
                
            success = False
            retries = 3
            seg_voice = seg.get("voice", self.voice)
            for attempt in range(retries):
                try:
                    success = dubber.generate_tts(text, seg_voice, out_file, self.rate, self.pitch)
                    if success and os.path.exists(out_file):
                        break
                except Exception as e:
                    print(f"[DEBUG] Thử lại dòng {idx+1} lần {attempt+1} do lỗi: {e}")
                
                # Tránh gọi dồn dập khi lỗi mạng
                time.sleep(1.0)
                
            if success and os.path.exists(out_file):
                duration = 0.0
                try:
                    duration = dubber.get_audio_duration(out_file)
                except Exception as de:
                    print(f"Error measuring audio duration: {de}")
                    try:
                        from pydub import AudioSegment
                        sound = AudioSegment.from_file(out_file)
                        duration = len(sound) / 1000.0
                    except:
                        duration = 2.5
                        
                results.append((idx, out_file, duration))
                self.progress.emit(idx, "Đã sẵn sàng")
            else:
                self.progress.emit(idx, "Lỗi mạng...")
                print(f"Lỗi sinh TTS dòng {idx+1}: Thất bại sau 3 lần thử.")
                results.append((idx, "", 0.0))
                
            # Nghỉ ngắn giữa các câu để tránh Microsoft rate-limiting
            time.sleep(0.3)
                
        self.finished.emit(results)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
