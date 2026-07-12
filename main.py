import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import tempfile
import cv2
import winsound
import time
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QSlider, QTextEdit, QDialog, QMessageBox, QProgressDialog,
    QAbstractItemView, QFrame, QCheckBox, QDoubleSpinBox, QSpinBox,
    QColorDialog, QProgressBar
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QRect, QPoint, QThread, pyqtSignal, QTimer

from pydub import AudioSegment
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
        if self.is_drawing or (not self.begin.isNull() and not self.end.isNull()):
            painter = QPainter(self)
            pen = QPen(QColor(127, 190, 178))
            pen.setWidth(2)
            painter.setPen(pen)
            rect = QRect(self.begin, self.end)
            painter.drawRect(rect)
            painter.end()

class DraggablePreviewLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dragging_subtitle = False

    def _pixmap_rect(self):
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return None
        x = (self.width() - pix.width()) // 2
        y = (self.height() - pix.height()) // 2
        return QRect(x, y, pix.width(), pix.height())

    def _update_custom_pos(self, event):
        rect = self._pixmap_rect()
        if rect is None:
            return
        p = event.position().toPoint()
        x = max(rect.left(), min(p.x(), rect.right()))
        y = max(rect.top(), min(p.y(), rect.bottom()))
        x_pct = ((x - rect.left()) / max(1, rect.width())) * 100.0
        y_pct = ((y - rect.top()) / max(1, rect.height())) * 100.0
        target = self.parent()
        if not hasattr(target, "set_subtitle_custom_pos_pct"):
            target = self.window()
        if hasattr(target, "set_subtitle_custom_pos_pct"):
            target.set_subtitle_custom_pos_pct(x_pct, y_pct)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap_rect() is not None:
            self.dragging_subtitle = True
            self._update_custom_pos(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging_subtitle:
            self._update_custom_pos(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.dragging_subtitle:
            self._update_custom_pos(event)
            self.dragging_subtitle = False
        super().mouseReleaseEvent(event)


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
                font-size: 12px;
                color: #dfb15b;
                border: none;
                padding: 4px 0px;
                background-color: transparent;
            }
            QPushButton:hover {
                color: #ffcc66;
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
    def __init__(self, frame, parent=None, title="Kéo chuột vẽ khung chọn vùng phụ đề (Hardsub)"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(1100, 500)
        self.parent = parent
        self.video_path = getattr(parent, 'video_path', None) if parent else None
        
        self.frame = frame
        self.h, self.w, _ = frame.shape
        
        # Khởi tạo Layout chính 2 cột
        main_layout = QHBoxLayout(self)
        
        # --- CỘT TRÁI: VẼ VÙNG QUÉT ---
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("<b>1. Vẽ vùng quét (Kéo thả chuột trên hình):</b>"))
        
        self.label = DrawingLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("border: 1px solid #1c1c1f; background-color: #0c0c0e;")
        left_layout.addWidget(self.label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Nút chức năng cột trái
        left_btn_layout = QHBoxLayout()
        self.btn_next_frame = QPushButton("🔄 Thử frame khác")
        self.btn_next_frame.setStyleSheet("background-color: #dfb15b; color: #0c0c0e; font-weight: bold; padding: 6px;")
        self.btn_next_frame.clicked.connect(self.get_another_frame)
        left_btn_layout.addWidget(self.btn_next_frame)
        left_btn_layout.addStretch()
        left_layout.addLayout(left_btn_layout)
        
        main_layout.addLayout(left_layout, 1)
        
        # --- CỘT PHẢI: PREVIEW & OCR FEEDBACK ---
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("<b>2. Kết quả nhận diện (OCR Feedback tức thì):</b>"))
        
        self.label_preview = QLabel(self)
        self.label_preview.setFixedSize(480, 270)
        self.label_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_preview.setStyleSheet("border: 1px solid #1c1c1f; background-color: #0c0c0e; color: #9c9c9f;")
        self.label_preview.setText("Vẽ khung chọn vùng quét bên trái để chạy thử OCR")
        right_layout.addWidget(self.label_preview)
        
        self.lbl_ocr_status = QLabel("Nhập vùng quét để chạy thử")
        self.lbl_ocr_status.setStyleSheet("color: #9c9c9f; font-weight: bold;")
        right_layout.addWidget(self.lbl_ocr_status)
        
        self.txt_ocr_result = QTextEdit()
        self.txt_ocr_result.setReadOnly(True)
        self.txt_ocr_result.setMaximumHeight(80)
        right_layout.addWidget(self.txt_ocr_result)
        
        # Hàng nút Xác nhận ở dưới cùng cột phải
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        self.btn_confirm = QPushButton("Xác nhận vùng quét")
        self.btn_confirm.setStyleSheet("background-color: #7fbeb2; color: #0c0c0e; font-weight: bold; padding: 8px 16px;")
        self.btn_confirm.clicked.connect(self.accept)
        btn_box.addWidget(self.btn_confirm)
        
        btn_cancel = QPushButton("Hủy")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)
        right_layout.addLayout(btn_box)
        
        main_layout.addLayout(right_layout, 1)
        
        # Load frame ban đầu
        self.init_display_frame()
        self.selected_bbox = None  # [x, y, w, h] theo độ phân giải gốc của video
        
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
            self.selected_bbox = [raw_x, raw_y, raw_w, raw_h]
            self.trigger_instant_ocr()
            
    def trigger_instant_ocr(self):
        if not self.selected_bbox:
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
        
        # Khởi chạy Worker Thread bất đồng bộ chống đơ
        if hasattr(self, 'ocr_worker') and self.ocr_worker.isRunning():
            self.ocr_worker.terminate()
            self.ocr_worker.wait()
            
        self.ocr_worker = InstantOCRWorker(self.frame, self.selected_bbox, ocr_lang)
        self.ocr_worker.finished.connect(self.on_ocr_success)
        self.ocr_worker.error.connect(self.on_ocr_error)
        self.ocr_worker.start()
        
    def on_ocr_success(self, results, summary):
        self.btn_confirm.setEnabled(True)
        self.btn_next_frame.setEnabled(True)
        self.lbl_ocr_status.setText("✅ QUÉT HOÀN TẤT")
        self.lbl_ocr_status.setStyleSheet("color: #7fbeb2; font-weight: bold;")
        self.txt_ocr_result.setPlainText(f"Kết quả nhận diện được:\n{summary}" if summary else "Không tìm thấy chữ nào.")
        self.draw_ocr_preview(results)
        
    def on_ocr_error(self, err_msg):
        self.btn_confirm.setEnabled(True)
        self.btn_next_frame.setEnabled(True)
        self.lbl_ocr_status.setText("❌ LỖI OCR")
        self.lbl_ocr_status.setStyleSheet("color: #ff9999; font-weight: bold;")
        self.txt_ocr_result.setPlainText(f"Lỗi: {err_msg}")
        
    def draw_ocr_preview(self, results):
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

# Worker cho luồng Trích xuất phụ đề
class TranscriptionWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list, str) # segments, video_path
    error = pyqtSignal(str)
    
    def __init__(self, video_path, mode, bbox, whisper_model, api_key, ocr_lang="auto"):
        super().__init__()
        self.video_path = video_path
        self.mode = mode
        self.bbox = bbox
        self.whisper_model = whisper_model
        self.api_key = api_key
        self.ocr_lang = ocr_lang
        
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
                segments = transcriber.run_hardsub_ocr(self.video_path, self.bbox, self.progress.emit, ocr_lang=self.ocr_lang)
                
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

# Worker cho luồng lồng tiếng và xuất video
class DubbingWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, video_path, segments, voice, output_path, bg_vol, voice_vol, burn_subtitles=False, selected_bbox=None, preset=None):
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
                progress_callback=self.progress.emit
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
        self.temp_preview_audio = None
        self.preset_font_color = [255, 255, 255]
        self.preset_outline_color = [0, 0, 0]
        self.preset_bg_color = [0, 0, 0]
        self.custom_font_path = None
        self.subtitle_custom_pos = None
        self.bbox_history_stack = []
        self.bbox_redo_stack = []
        
        from PyQt6.QtGui import QShortcut, QKeySequence
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo_bbox_change)
        self.shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo.activated.connect(self.redo_bbox_change)
        
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
        
        self.setup_ui()
        self.update_glossary_combobox()
        self.apply_dark_theme()
        
    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # 1. Custom Header Bar (Branding area)
        header_frame = QFrame()
        header_frame.setObjectName("header")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 12, 20, 12)
        
        title_layout = QVBoxLayout()
        lbl_title = QLabel("<span style='color: #dfb15b; font-size: 22px; font-weight: bold;'>supersubs v1</span> <span style='color: #7fbeb2; font-size: 11px; font-weight: bold;'>[ IMPECCABLE ]</span>")
        lbl_sub = QLabel("Công cụ tải video, trích xuất chữ cứng OCR, dịch thuật AI và lồng tiếng tự động co giãn")
        lbl_sub.setStyleSheet("color: #9c9c9f; font-size: 11px;")
        title_layout.addWidget(lbl_title)
        title_layout.addWidget(lbl_sub)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        lbl_brand_status = QLabel("🟢 SẴN SÀNG")
        lbl_brand_status.setStyleSheet("color: #7fbeb2; font-weight: bold; font-size: 11px; border: 1px solid #1c1c1f; border-radius: 4px; padding: 4px 8px; background-color: #141416;")
        header_layout.addWidget(lbl_brand_status)
        
        self.main_layout.addWidget(header_frame)
        
        # 2. Splitter for Left and Right Columns
        from PyQt6.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #1c1c1f; width: 4px; }")
        
        # Left Panel (Tabs)
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(450)
        splitter.addWidget(self.tabs)
        
        # Right Panel (Persistent Video Preview Canvas)
        self.right_preview_pane = QFrame()
        self.right_preview_pane.setObjectName("card")
        self.right_preview_pane.setStyleSheet("margin: 10px; border: 1px solid #1c1c1f; border-radius: 8px; background-color: #141416;")
        splitter.addWidget(self.right_preview_pane)
        
        right_layout = QVBoxLayout(self.right_preview_pane)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(12)
        
        lbl_preview_title = QLabel("📺 XEM TRƯỚC HÌNH ẢNH & PHỤ ĐỀ (PREVIEW)")
        lbl_preview_title.setStyleSheet("color: #dfb15b; font-weight: bold; font-size: 13px;")
        right_layout.addWidget(lbl_preview_title)
        
        self.lbl_main_preview = DraggablePreviewLabel(self)
        self.lbl_main_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_main_preview.setStyleSheet("border: 1px solid #1c1c1f; background-color: #0c0c0e; border-radius: 6px; color: #9c9c9f; font-size: 14px;")
        self.lbl_main_preview.setText("Vui lòng tải video và chọn phân đoạn phụ đề để xem trước")
        # Ensure label expands to fill empty space
        self.lbl_main_preview.setSizePolicy(
            self.lbl_main_preview.sizePolicy().horizontalPolicy(),
            self.lbl_main_preview.sizePolicy().verticalPolicy().Expanding
        )
        right_layout.addWidget(self.lbl_main_preview)

        lbl_drag_hint = QLabel("Keo truc tiep tren preview de dat vi tri sub.")
        lbl_drag_hint.setStyleSheet("color: #9c9c9f; font-style: italic;")
        right_layout.addWidget(lbl_drag_hint)
        
        # Canvas control bar
        controls_layout = QHBoxLayout()
        
        self.btn_play_seg = QPushButton("▶ Phát phân đoạn")
        self.btn_play_seg.setStyleSheet("background-color: #7fbeb2; color: #0c0c0e; font-weight: bold; padding: 8px;")
        self.btn_play_seg.clicked.connect(self.play_video_segment)
        controls_layout.addWidget(self.btn_play_seg)
        
        self.btn_preview_tts = QPushButton("🔊 Nghe thử giọng AI")
        self.btn_preview_tts.setStyleSheet("background-color: #7fbeb2; color: #0c0c0e; font-weight: bold; padding: 8px;")
        self.btn_preview_tts.clicked.connect(self.preview_segment_tts)
        controls_layout.addWidget(self.btn_preview_tts)
        
        self.btn_edit_mask = QPushButton("✏️ Chỉnh hộp che (Mask Box)")
        self.btn_edit_mask.setStyleSheet("background-color: #dfb15b; color: #0c0c0e; font-weight: bold; padding: 8px;")
        self.btn_edit_mask.clicked.connect(self.edit_mask_box)
        controls_layout.addWidget(self.btn_edit_mask)

        self.btn_reset_sub_pos = QPushButton("Reset vi tri sub")
        self.btn_reset_sub_pos.clicked.connect(self.reset_subtitle_custom_pos)
        controls_layout.addWidget(self.btn_reset_sub_pos)
        
        right_layout.addLayout(controls_layout)
        
        # Thêm splitter vào layout chính
        self.main_layout.addWidget(splitter)
        
        # Cấu hình các Tabs
        self.create_tab1_download()
        self.create_tab2_editor()
        self.create_tab3_dubbing()
        self.create_tab4_script()
        
        # Debounce QTimer cho vẽ canvas
        self.canvas_timer = QTimer(self)
        self.canvas_timer.setSingleShot(True)
        self.canvas_timer.timeout.connect(self.update_canvas_realtime_now)
        
        # Thanh trạng thái (Status Bar)
        self.status_label = QLabel("Sẵn sàng")
        self.statusBar().addPermanentWidget(self.status_label)
        
    def create_tab1_download(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # CARD 1: Nguồn Video
        card_source = CollapsibleCard("📥 NGUỒN VIDEO / URL")
        
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("Dán Link Video:"))
        self.txt_url = QLineEdit()
        self.txt_url.setPlaceholderText("https://www.youtube.com/watch?...")
        url_layout.addWidget(self.txt_url)
        card_source.addLayout(url_layout)
        
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Hoặc file trong máy:"))
        self.txt_file = QLineEdit()
        self.txt_file.setReadOnly(True)
        file_layout.addWidget(self.txt_file)
        btn_browse = QPushButton("Duyệt file...")
        btn_browse.clicked.connect(self.browse_video)
        file_layout.addWidget(btn_browse)
        card_source.addLayout(file_layout)
        layout.addWidget(card_source)
        
        # CARD 2: Cấu hình trích xuất
        # CARD 2: Cấu hình trích xuất
        card_extract = CollapsibleCard("⚙ CẤU HÌNH TRÍCH XUẤT PHỤ ĐỀ")
        
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(QLabel("Phương pháp:"))
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Nhận dạng giọng nói (Whisper/Gemini)", "Quét chữ cứng trên video (OCR)"])
        self.cb_mode.currentIndexChanged.connect(self.on_mode_changed)
        opt_layout.addWidget(self.cb_mode)
        
        self.btn_select_region = QPushButton("🔍 Vẽ vùng quét chữ trên Video")
        self.btn_select_region.setEnabled(False)
        self.btn_select_region.clicked.connect(self.select_ocr_region)
        opt_layout.addWidget(self.btn_select_region)
        
        self.lbl_bbox = QLabel("Vùng quét: Chưa chọn")
        self.lbl_bbox.setStyleSheet("color: #7fbeb2; font-weight: bold;")
        opt_layout.addWidget(self.lbl_bbox)
        card_extract.addLayout(opt_layout)
        
        # Whisper Settings Panel
        self.whisper_widget = QWidget()
        whisper_layout = QHBoxLayout(self.whisper_widget)
        whisper_layout.setContentsMargins(0, 0, 0, 0)
        whisper_layout.addWidget(QLabel("Model Whisper Local:"))
        self.cb_whisper_model = QComboBox()
        self.cb_whisper_model.addItems([
            "tiny (máy rất yếu)",
            "base (nhanh, nhẹ)",
            "small (CPU ổn)",
            "medium (cân bằng)",
            "large-v3 (GPU, chuẩn nhất)"
        ])
        self.cb_whisper_model.setCurrentText("base (nhanh, nhẹ)")
        whisper_layout.addWidget(self.cb_whisper_model)
        card_extract.addWidget(self.whisper_widget)

        # OCR Settings Panel
        self.ocr_widget = QWidget()
        ocr_settings_layout = QHBoxLayout(self.ocr_widget)
        ocr_settings_layout.setContentsMargins(0, 0, 0, 0)
        ocr_settings_layout.addWidget(QLabel("Ngôn ngữ chữ cứng (OCR):"))
        self.cb_ocr_lang = QComboBox()
        self.cb_ocr_lang.addItems([
            "Tự động (Trung, Anh)", 
            "Tiếng Trung Giản Thể (`ch_sim`)", 
            "Tiếng Trung Phồn Thể (`ch_tra`)", 
            "Tiếng Việt (`vi`)", 
            "Tiếng Anh (`en`)", 
            "Tiếng Nhật (`ja`)", 
            "Tiếng Hàn (`ko`)",
            "Tiếng Pháp (`fr`)",
            "Tiếng Đức (`de`)",
            "Tiếng Tây Ban Nha (`es`)",
            "Tiếng Nga (`ru`)",
            "Tiếng Thái (`th`)"
        ])
        self.cb_ocr_lang.setCurrentIndex(0)
        ocr_settings_layout.addWidget(self.cb_ocr_lang)
        card_extract.addWidget(self.ocr_widget)
        
        self.ocr_widget.setVisible(False)
        layout.addWidget(card_extract)
        
        # CARD 3: Cấu hình API Keys & Từ điển cục bộ
        card_config = CollapsibleCard("🔑 CẤU HÌNH API KEYS & TỪ ĐIỂN CỤC BỘ (TÙY CHỌN)")
        
        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel("Gemini Key:"))
        self.txt_gemini_key = QLineEdit()
        self.txt_gemini_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_gemini_key.setPlaceholderText("AIzaSy...")
        api_layout.addWidget(self.txt_gemini_key)
        
        api_layout.addWidget(QLabel("  Groq Key:"))
        self.txt_groq_key = QLineEdit()
        self.txt_groq_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_groq_key.setPlaceholderText("gsk_...")
        api_layout.addWidget(self.txt_groq_key)
        
        api_layout.addWidget(QLabel("  DeepL Key:"))
        self.txt_deepl_key = QLineEdit()
        self.txt_deepl_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.txt_deepl_key.setPlaceholderText("Tùy chọn cho DeepL...")
        api_layout.addWidget(self.txt_deepl_key)
        
        api_layout.addWidget(QLabel("  Model Ollama:"))
        self.txt_ollama_model = QLineEdit()
        self.txt_ollama_model.setText("qwen2.5")
        self.txt_ollama_model.setPlaceholderText("qwen2.5 / llama3.1")
        self.txt_ollama_model.setMaximumWidth(100)
        api_layout.addWidget(self.txt_ollama_model)
        card_config.addLayout(api_layout)
        
        dict_layout = QHBoxLayout()
        dict_layout.addWidget(QLabel("VietPhrase.txt:"))
        self.txt_vp_path = QLineEdit()
        self.txt_vp_path.setPlaceholderText("Để trống sẽ dùng tệp mặc định trong thư mục Data/")
        dict_layout.addWidget(self.txt_vp_path)
        btn_vp = QPushButton("Chọn...")
        btn_vp.clicked.connect(self.browse_vp_dict)
        dict_layout.addWidget(btn_vp)
        
        dict_layout.addWidget(QLabel("  Names.txt:"))
        self.txt_names_path = QLineEdit()
        self.txt_names_path.setPlaceholderText("Để trống dùng mặc định")
        dict_layout.addWidget(self.txt_names_path)
        btn_names = QPushButton("Chọn...")
        btn_names.clicked.connect(self.browse_names_dict)
        dict_layout.addWidget(btn_names)
        card_config.addLayout(dict_layout)
        layout.addWidget(card_config)
        card_config.hide()
        
        # Nút trích xuất chính
        self.btn_start_extract = QPushButton("🚀 BẮT ĐẦU TRÍCH XUẤT PHỤ ĐỀ GỐC")
        self.btn_start_extract.setStyleSheet("background-color: #dfb15b; color: #0c0c0e; font-size: 14px; font-weight: bold; padding: 12px; border-radius: 6px;")
        self.btn_start_extract.clicked.connect(self.start_extraction)
        layout.addWidget(self.btn_start_extract)
        
        # Log tiến độ
        self.txt_logs1 = QTextEdit()
        self.txt_logs1.setReadOnly(True)
        self.txt_logs1.setPlaceholderText("Tiến độ xử lý sẽ hiển thị ở đây...")
        layout.addWidget(self.txt_logs1)
        
        self.tabs.addTab(tab, "📥 Tải & Trích phụ đề")
        
    def create_tab2_editor(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # Thanh công cụ trên
        toolbar_frame = QFrame()
        toolbar_frame.setObjectName("card")
        toolbar_layout = QHBoxLayout(toolbar_frame)
        toolbar_layout.setContentsMargins(15, 10, 15, 10)
        toolbar_layout.setSpacing(10)
        
        toolbar_layout.addWidget(QLabel("🌐 DỊCH PHỤ ĐỀ:"))
        self.cb_engine = QComboBox()
        self.cb_engine.addItems([
            "Supersubs AI (Tối ưu: VietPhrase + Tinh chỉnh văn phong)",
            "Dịch thô (Chỉ dùng Quick Translator)",
            "Dịch cơ bản (Google Translate - Fallback)"
        ])
        toolbar_layout.addWidget(self.cb_engine)
        
        btn_translate = QPushButton("✨ Dịch tự động")
        btn_translate.setStyleSheet("background-color: #dfb15b; color: #0c0c0e; font-weight: bold;")
        btn_translate.clicked.connect(self.start_translation)
        toolbar_layout.addWidget(btn_translate)
        
        toolbar_layout.addStretch()
        layout.addWidget(toolbar_frame)
        
        # Card Cấu hình Sơ chế Dịch thuật (Phần 3)
        refine_frame = CollapsibleCard("🤖 CẤU HÌNH SƠ CHẾ & TINH CHỈNH DỊCH THUẬT (LLM)")
        refine_layout = refine_frame
        
        self.chk_refine_enabled = QCheckBox("Bật sơ chế dịch thuật (LLM Refinement)")
        self.chk_refine_enabled.setChecked(True)
        self.chk_refine_enabled.hide()
        
        lbl_refine_engine = QLabel("  Động cơ LLM:")
        self.cb_refine_engine = QComboBox()
        self.cb_refine_engine.addItems([
            "Gemini 1.5 Flash (Nhanh, Rẻ, Yêu cầu API Key)",
            "Gemini 1.5 Pro (Thông minh, Chậm hơn, Yêu cầu API Key)",
            "Gemini 2.0 Flash (Thế hệ mới, Cực nhanh, Yêu cầu API Key)",
            "Groq Llama 3.1 (70B) (Rất nhanh, Miễn phí/Rẻ, Yêu cầu Groq Key)",
            "Ollama Local (Chạy offline bảo mật, Tải model cục bộ)"
        ])
        lbl_refine_engine.hide()
        self.cb_refine_engine.hide()
        
        row1 = QGridLayout()
        row1.addWidget(QLabel("Tải/Chọn Glossary:"), 0, 0)
        self.cb_glossary_files = QComboBox()
        self.cb_glossary_files.addItem("-- Chọn Glossary --")
        self.cb_glossary_files.currentIndexChanged.connect(self.on_glossary_dropdown_changed)
        row1.addWidget(self.cb_glossary_files, 0, 1)
        
        self.btn_load_glossary = QPushButton("📁 Tải file...")
        self.btn_load_glossary.clicked.connect(self.load_glossary_from_file)
        row1.addWidget(self.btn_load_glossary, 1, 0)
        
        self.btn_save_glossary = QPushButton("💾 Lưu file...")
        self.btn_save_glossary.clicked.connect(self.save_glossary_to_file)
        row1.addWidget(self.btn_save_glossary, 1, 1)
        
        refine_frame.addLayout(row1)
        
        refine_layout.addWidget(QLabel("Bảng thuật ngữ Glossary (từ_gốc = từ_dịch, mỗi câu một dòng):"))
        self.txt_glossary = QTextEdit()
        self.txt_glossary.setPlaceholderText("Ví dụ:\nLlama = Lạc Đà\nAI = Trí tuệ Nhân tạo (AI)")
        self.txt_glossary.setMaximumHeight(80)
        refine_layout.addWidget(self.txt_glossary)
        
        layout.addWidget(refine_frame)
        
        # Bảng phụ đề (9 cột)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "Bắt đầu (s)", 
            "Kết thúc (s)", 
            "Thời lượng (s)",
            "Câu gốc", 
            "Dịch thô",
            "Bản hoàn chỉnh (Kích đúp để sửa)", 
            "Tốc độ đọc",
            "Độ tin cậy",
            "Hộp che (Bbox)"
        ])
        
        # Cấu hình bảng
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setDefaultSectionSize(36) # Row height 36px for breathing room
        self.table.cellChanged.connect(self.on_cell_changed)
        self.table.itemSelectionChanged.connect(self.trigger_canvas_update)
        
        layout.addWidget(self.table)
        
        # Hướng dẫn nhỏ
        lbl_hint = QLabel("💡 Mẹo: Nhấp đúp vào cột 'Bắt đầu', 'Kết thúc' hoặc 'Bản hoàn chỉnh' để tinh chỉnh. Nếu tốc độ đọc vượt quá ngưỡng cấu hình, cột tốc độ sẽ báo màu.")
        lbl_hint.setStyleSheet("color: #9c9c9f; font-style: italic;")
        layout.addWidget(lbl_hint)
        
        self.tabs.addTab(tab, "✏ Biên tập & Dịch thuật")
        
    def create_tab3_dubbing(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # CARD 1: Cấu hình giọng lồng tiếng
        card_voice = CollapsibleCard("🎙 CHỌN GIỌNG ĐỌC AI")
        
        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel("Giọng lồng tiếng:"))
        self.cb_voice = QComboBox()
        for v in dubber.get_supported_voices():
            self.cb_voice.addItem(v["desc"], v["name"])
        voice_row.addWidget(self.cb_voice)
        card_voice.addLayout(voice_row)
        layout.addWidget(card_voice)
        
        # CARD 2: Bộ trộn âm thanh
        card_mixer = CollapsibleCard("🎛 BỘ TRỘN ÂM LƯỢNG (MIXER)")
        
        vol_layout = QGridLayout()
        vol_layout.addWidget(QLabel("Nhạc nền video gốc:"), 0, 0)
        self.slider_bg = QSlider(Qt.Orientation.Horizontal)
        self.slider_bg.setRange(0, 100)
        self.slider_bg.setValue(10)
        vol_layout.addWidget(self.slider_bg, 0, 1)
        self.lbl_bg_vol = QLabel("10%")
        self.slider_bg.valueChanged.connect(lambda v: self.lbl_bg_vol.setText(f"{v}%"))
        vol_layout.addWidget(self.lbl_bg_vol, 0, 2)
        
        vol_layout.addWidget(QLabel("Giọng đọc lồng tiếng:"), 1, 0)
        self.slider_dub = QSlider(Qt.Orientation.Horizontal)
        self.slider_dub.setRange(0, 200)
        self.slider_dub.setValue(100)
        vol_layout.addWidget(self.slider_dub, 1, 1)
        self.lbl_dub_vol = QLabel("100%")
        self.slider_dub.valueChanged.connect(lambda v: self.lbl_dub_vol.setText(f"{v}%"))
        vol_layout.addWidget(self.lbl_dub_vol, 1, 2)
        card_mixer.addLayout(vol_layout)
        layout.addWidget(card_mixer)
        
        # CARD SUBTITLE (Cấu hình phụ đề) - Thêm mới
        self.card_subtitle = CollapsibleCard("🔤 CẤU HÌNH KIỂU CHỮ & VỊ TRÍ PHỤ ĐỀ GHI ĐÈ")
        self.card_subtitle.setVisible(False)
        sub_layout = self.card_subtitle
        
        # 1. Preset & Apply Mode row
        row_preset = QGridLayout()
        row_preset.addWidget(QLabel("Bộ mẫu (Preset):"), 0, 0)
        self.cb_preset = QComboBox()
        self.cb_preset.addItems([
            "Mặc định (Dưới - Giữa)",
            "Dưới - Trái",
            "Dưới - Phải",
            "Trên - Giữa",
            "Giữa - Giữa",
            "Tùy chỉnh (Custom)"
        ])
        row_preset.addWidget(self.cb_preset, 0, 1)
        
        self.btn_reset_preset = QPushButton("Reset về mặc định")
        row_preset.addWidget(self.btn_reset_preset, 0, 2)
        
        row_preset.addWidget(QLabel("Cách áp mẫu:"), 1, 0)
        self.cb_preset_apply_mode = QComboBox()
        self.cb_preset_apply_mode.addItems([
            "Chỉ áp VỊ TRÍ, giữ style gốc",
            "Áp dụng cả VỊ TRÍ & STYLE"
        ])
        row_preset.addWidget(self.cb_preset_apply_mode, 1, 1, 1, 2)
        sub_layout.addLayout(row_preset)
        
        # 2. Vị trí căn lề (Position & Align) row
        row_pos = QGridLayout()
        row_pos.addWidget(QLabel("Căn lề ngang:"), 0, 0)
        self.cb_h_align = QComboBox()
        self.cb_h_align.addItems(["Left", "Center", "Right"])
        self.cb_h_align.setCurrentText("Center")
        row_pos.addWidget(self.cb_h_align, 0, 1)
        
        row_pos.addWidget(QLabel("Căn dọc:"), 0, 2)
        self.cb_v_align = QComboBox()
        self.cb_v_align.addItems(["Top", "Middle", "Bottom"])
        self.cb_v_align.setCurrentText("Bottom")
        row_pos.addWidget(self.cb_v_align, 0, 3)
        
        row_pos.addWidget(QLabel("Margin Dọc:"), 1, 0)
        margin_v_layout = QHBoxLayout()
        self.spin_margin_v = QDoubleSpinBox()
        self.spin_margin_v.setRange(0, 500)
        self.spin_margin_v.setValue(8.0)
        self.spin_margin_v.setSingleStep(0.5)
        margin_v_layout.addWidget(self.spin_margin_v)
        self.cb_margin_v_type = QComboBox()
        self.cb_margin_v_type.addItems(["%", "px"])
        margin_v_layout.addWidget(self.cb_margin_v_type)
        row_pos.addLayout(margin_v_layout, 1, 1)
        
        row_pos.addWidget(QLabel("Margin Ngang:"), 1, 2)
        margin_h_layout = QHBoxLayout()
        self.spin_margin_h = QDoubleSpinBox()
        self.spin_margin_h.setRange(0, 500)
        self.spin_margin_h.setValue(5.0)
        self.spin_margin_h.setSingleStep(0.5)
        margin_h_layout.addWidget(self.spin_margin_h)
        self.cb_margin_h_type = QComboBox()
        self.cb_margin_h_type.addItems(["%", "px"])
        margin_h_layout.addWidget(self.cb_margin_h_type)
        row_pos.addLayout(margin_h_layout, 1, 3)
        sub_layout.addLayout(row_pos)
        
        # 2b. Tọa độ tùy chỉnh (Custom Position) row
        self.row_custom_pos = QHBoxLayout()
        self.row_custom_pos.addWidget(QLabel("Vị trí X (%):"))
        self.spin_custom_pos_x = QDoubleSpinBox()
        self.spin_custom_pos_x.setRange(0.0, 100.0)
        self.spin_custom_pos_x.setValue(50.0)
        self.spin_custom_pos_x.setSingleStep(1.0)
        self.spin_custom_pos_x.setEnabled(False)
        self.row_custom_pos.addWidget(self.spin_custom_pos_x)
        
        self.row_custom_pos.addWidget(QLabel("  Vị trí Y (%):"))
        self.spin_custom_pos_y = QDoubleSpinBox()
        self.spin_custom_pos_y.setRange(0.0, 100.0)
        self.spin_custom_pos_y.setValue(88.0)
        self.spin_custom_pos_y.setSingleStep(1.0)
        self.spin_custom_pos_y.setEnabled(False)
        self.row_custom_pos.addWidget(self.spin_custom_pos_y)
        
        self.btn_reset_custom_pos = QPushButton("Đặt lại tâm (50%, 88%)")
        self.btn_reset_custom_pos.setEnabled(False)
        self.row_custom_pos.addWidget(self.btn_reset_custom_pos)
        sub_layout.addLayout(self.row_custom_pos)
        
        # 3. Font & Size row
        row_font = QGridLayout()
        row_font.addWidget(QLabel("Font chữ:"), 0, 0)
        self.cb_font_name = QComboBox()
        self.cb_font_name.addItems([
            "Arial", "Calibri", "Segoe UI", "Times New Roman", "Tahoma", "Courier New", "Consolas"
        ])
        row_font.addWidget(self.cb_font_name, 0, 1)
        
        self.btn_browse_font = QPushButton("Duyệt font (.ttf)...")
        row_font.addWidget(self.btn_browse_font, 0, 2)
        
        row_font.addWidget(QLabel("Cỡ chữ:"), 1, 0)
        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(8, 120)
        self.spin_font_size.setValue(20)
        row_font.addWidget(self.spin_font_size, 1, 1)
        
        row_font.addWidget(QLabel("Độ dày viền:"), 1, 2)
        self.spin_outline_width = QSpinBox()
        self.spin_outline_width.setRange(0, 20)
        self.spin_outline_width.setValue(2)
        row_font.addWidget(self.spin_outline_width, 1, 3)
        sub_layout.addLayout(row_font)
        
        # 4. Color & HEX row
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
        
        row_color.addWidget(QLabel("Màu hộp nền:"), 1, 0)
        bg_c_layout = QHBoxLayout()
        self.btn_bg_color = QPushButton()
        self.btn_bg_color.setFixedSize(24, 24)
        bg_c_layout.addWidget(self.btn_bg_color)
        self.txt_bg_color_hex = QLineEdit()
        self.txt_bg_color_hex.setMaximumWidth(80)
        bg_c_layout.addWidget(self.txt_bg_color_hex)
        row_color.addLayout(bg_c_layout, 1, 1)
        
        self.chk_use_bg_box = QCheckBox("Bật nền")
        self.chk_use_bg_box.setChecked(False)
        row_color.addWidget(self.chk_use_bg_box, 1, 2)
        
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Độ mờ nền:"))
        self.slider_bg_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_bg_opacity.setRange(0, 255)
        self.slider_bg_opacity.setValue(0)
        opacity_layout.addWidget(self.slider_bg_opacity)
        row_color.addLayout(opacity_layout, 1, 3)
        sub_layout.addLayout(row_color)
        
        # 4b. Che phụ đề gốc & Cấu hình OCR (Phần 2)
        row_mask = QGridLayout()
        row_mask.addWidget(QLabel("Che sub gốc:"), 0, 0)
        self.cb_mask_mode = QComboBox()
        self.cb_mask_mode.addItems([
            "Không che (None)",
            "Che đen đặc (Black Box)",
            "Blur nhanh (Gaussian Blur)",
            "Inpaint chất lượng cao"
        ])
        self.cb_mask_mode.setCurrentIndex(2) # Mặc định là Blur
        row_mask.addWidget(self.cb_mask_mode, 0, 1)
        
        row_mask.addWidget(QLabel("Tốc độ tối đa:"), 0, 2)
        self.spin_speed_threshold = QSpinBox()
        self.spin_speed_threshold.setRange(5, 50)
        self.spin_speed_threshold.setValue(20)
        self.spin_speed_threshold.setSuffix(" ch/s")
        self.spin_speed_threshold.valueChanged.connect(self.populate_subtitle_table)
        row_mask.addWidget(self.spin_speed_threshold, 0, 3)
        
        row_mask.addWidget(QLabel("Cảnh báo < :"), 1, 0)
        self.spin_confidence_threshold = QSpinBox()
        self.spin_confidence_threshold.setRange(10, 100)
        self.spin_confidence_threshold.setValue(70)
        self.spin_confidence_threshold.setSuffix("%")
        row_mask.addWidget(self.spin_confidence_threshold, 1, 1)
        
        self.chk_restrict_ocr = QCheckBox("Giới hạn quét ngang 60%")
        self.chk_restrict_ocr.setChecked(True)
        row_mask.addWidget(self.chk_restrict_ocr, 1, 2)
        
        self.btn_auto_detect = QPushButton("🔍 Tự quét vùng sub gốc (OCR)")
        self.btn_auto_detect.setStyleSheet("background-color: #dfb15b; color: #0c0c0e; font-weight: bold;")
        self.btn_auto_detect.clicked.connect(self.trigger_auto_detection)
        row_mask.addWidget(self.btn_auto_detect, 1, 3)
        sub_layout.addLayout(row_mask)
        
        # 4c. Thuật toán xóa watermark
        row_algo = QGridLayout()
        row_algo.addWidget(QLabel("Thuật toán xóa:"), 0, 0)
        self.cb_remove_algo = QComboBox()
        self.cb_remove_algo.addItems([
            "Xóa cơ bản (FFmpeg) - Phù hợp máy yếu, xử lý siêu tốc",
            "Xóa AI (OpenCV) - Hiệu quả cao, xóa mượt mà (Render lâu hơn)"
        ])
        self.cb_remove_algo.setCurrentIndex(1) # Mặc định là OpenCV (Xóa AI)
        self.cb_remove_algo.currentIndexChanged.connect(self.trigger_canvas_update)
        row_algo.addWidget(self.cb_remove_algo, 0, 1)
        
        self.chk_smart_pos = QCheckBox("Tự động căn chỉnh phụ đề đè lên hộp che (Smart Pos)")
        self.chk_smart_pos.setChecked(False)
        self.chk_smart_pos.setStyleSheet("color: #dfb15b; font-weight: bold;")
        self.chk_smart_pos.stateChanged.connect(self.mark_preset_custom)
        self.chk_smart_pos.stateChanged.connect(self.trigger_canvas_update)
        row_algo.addWidget(self.chk_smart_pos, 1, 0, 1, 2)
        sub_layout.addLayout(row_algo)
        
        # 5. Buttons row
        row_actions = QHBoxLayout()
        self.btn_preview_sub = QPushButton("👁️ Xem trước phụ đề (Preview)...")
        self.btn_preview_sub.setStyleSheet("background-color: #7fbeb2; color: #0c0c0e; font-weight: bold;")
        row_actions.addWidget(self.btn_preview_sub)
        row_actions.addStretch()
        sub_layout.addLayout(row_actions)
        
        layout.addWidget(self.card_subtitle)
        
        # CARD 3: Xuất bản
        card_export = CollapsibleCard("💾 ĐƯỜNG DẪN XUẤT VIDEO")
        
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Lưu video thành phẩm tại:"))
        self.txt_out = QLineEdit()
        self.txt_out.setReadOnly(True)
        out_layout.addWidget(self.txt_out)
        btn_out_browse = QPushButton("Duyệt thư mục...")
        btn_out_browse.clicked.connect(self.browse_output)
        out_layout.addWidget(btn_out_browse)
        card_export.addLayout(out_layout)
        
        # Thêm checkbox chọn ghi đè phụ đề
        self.chk_burn_sub = QCheckBox("Ghi đè phụ đề tiếng Việt lên video (Che hoàn toàn phụ đề gốc)")
        self.chk_burn_sub.setChecked(False)
        self.chk_burn_sub.setStyleSheet("color: #dfb15b; font-weight: bold; margin-top: 5px;")
        card_export.addWidget(self.chk_burn_sub)
        
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
        
        self.tabs.addTab(tab, "🎙 Lồng tiếng & Xuất")
        
    def create_tab4_script(self):
        self.script_voice_tab = QWidget()
        self.tabs.addTab(self.script_voice_tab, "Kịch bản & Giọng đọc")
        self.build_script_voice_tab()

    def build_script_voice_tab(self):
        layout = QVBoxLayout(self.script_voice_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        self.script_tab = ScriptVoiceoverTab(self)
        layout.addWidget(self.script_tab)
        
        # Hook checkbox để hiển thị card cấu hình
        self.chk_burn_sub.toggled.connect(self.card_subtitle.setVisible)
        
        # Kết nối sự kiện cho các control màu sắc
        self.setup_color_button_hex(self.btn_font_color, self.txt_font_color_hex, self.preset_font_color, self.on_font_color_changed)
        self.setup_color_button_hex(self.btn_outline_color, self.txt_outline_color_hex, self.preset_outline_color, self.on_outline_color_changed)
        self.setup_color_button_hex(self.btn_bg_color, self.txt_bg_color_hex, self.preset_bg_color, self.on_bg_color_changed)
        
        # Kết nối sự kiện thay đổi giá trị để đánh dấu preset Custom
        self.cb_v_align.currentIndexChanged.connect(self.mark_preset_custom)
        self.cb_h_align.currentIndexChanged.connect(self.mark_preset_custom)
        self.spin_margin_v.valueChanged.connect(self.mark_preset_custom)
        self.cb_margin_v_type.currentIndexChanged.connect(self.mark_preset_custom)
        self.spin_margin_h.valueChanged.connect(self.mark_preset_custom)
        self.cb_margin_h_type.currentIndexChanged.connect(self.mark_preset_custom)
        self.spin_font_size.valueChanged.connect(self.mark_preset_custom)
        self.spin_outline_width.valueChanged.connect(self.mark_preset_custom)
        self.chk_use_bg_box.toggled.connect(self.mark_preset_custom)
        self.slider_bg_opacity.valueChanged.connect(self.mark_preset_custom)
        self.spin_confidence_threshold.valueChanged.connect(self.populate_subtitle_table)
        
        self.cb_font_name.currentIndexChanged.connect(self.on_font_changed)
        self.btn_browse_font.clicked.connect(self.browse_custom_font)
        self.cb_preset.currentTextChanged.connect(self.on_preset_changed)
        self.btn_reset_preset.clicked.connect(self.reset_preset)
        
        # Kết nối sự kiện tọa độ tùy chỉnh
        self.spin_custom_pos_x.valueChanged.connect(self.on_custom_pos_spin_changed)
        self.spin_custom_pos_y.valueChanged.connect(self.on_custom_pos_spin_changed)
        self.spin_custom_pos_x.valueChanged.connect(self.trigger_canvas_update)
        self.spin_custom_pos_y.valueChanged.connect(self.trigger_canvas_update)
        self.btn_reset_custom_pos.clicked.connect(self.reset_subtitle_custom_pos)
        
        # Kết nối sự kiện thay đổi style để vẽ canvas cập nhật thời gian thực
        self.cb_preset.currentIndexChanged.connect(self.trigger_canvas_update)
        self.cb_preset_apply_mode.currentIndexChanged.connect(self.trigger_canvas_update)
        self.cb_h_align.currentIndexChanged.connect(self.trigger_canvas_update)
        self.cb_v_align.currentIndexChanged.connect(self.trigger_canvas_update)
        self.spin_margin_v.valueChanged.connect(self.trigger_canvas_update)
        self.cb_margin_v_type.currentIndexChanged.connect(self.trigger_canvas_update)
        self.spin_margin_h.valueChanged.connect(self.trigger_canvas_update)
        self.cb_margin_h_type.currentIndexChanged.connect(self.trigger_canvas_update)
        self.cb_font_name.currentIndexChanged.connect(self.trigger_canvas_update)
        self.spin_font_size.valueChanged.connect(self.trigger_canvas_update)
        self.spin_outline_width.valueChanged.connect(self.trigger_canvas_update)
        self.txt_font_color_hex.textChanged.connect(self.trigger_canvas_update)
        self.txt_outline_color_hex.textChanged.connect(self.trigger_canvas_update)
        self.txt_bg_color_hex.textChanged.connect(self.trigger_canvas_update)
        self.chk_use_bg_box.stateChanged.connect(self.trigger_canvas_update)
        self.slider_bg_opacity.valueChanged.connect(self.trigger_canvas_update)
        self.cb_mask_mode.currentIndexChanged.connect(self.trigger_canvas_update)
        self.btn_preview_sub.clicked.connect(self.trigger_canvas_update)
        
        # Apply mặc định ban đầu cho màu nút
        self.update_color_button(self.btn_font_color, self.txt_font_color_hex, self.preset_font_color)
        self.update_color_button(self.btn_outline_color, self.txt_outline_color_hex, self.preset_outline_color)
        self.update_color_button(self.btn_bg_color, self.txt_bg_color_hex, self.preset_bg_color)
        
    def on_mode_changed(self, index):
        # Bật/tắt nút vẽ khung quét chữ tuỳ thuộc vào chế độ chọn
        is_ocr = (index == 1)
        self.btn_select_region.setEnabled(is_ocr)
        if not is_ocr:
            self.lbl_bbox.setText("Vùng quét: Không áp dụng")
            
        # Ẩn/Hiện widget tương ứng
        self.whisper_widget.setVisible(not is_ocr)
        self.ocr_widget.setVisible(is_ocr)
            
    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if file_path:
            self.video_path = file_path
            self.txt_file.setText(file_path)
            self.btn_select_region.setEnabled(self.cb_mode.currentIndex() == 1)
            return

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
        selector = VideoRegionSelector(frame, self)
        if self.selected_bbox:
            selector.set_initial_bbox(self.selected_bbox)
        if selector.exec() == QDialog.DialogCode.Accepted:
            if selector.selected_bbox:
                self.selected_bbox = selector.selected_bbox
                x, y, w, h = self.selected_bbox
                self.lbl_bbox.setText(f"Vùng quét: X={x}, Y={y}, W={w}, H={h}")
                self.status_label.setText("Đã chọn vùng quét phụ đề")
            else:
                QMessageBox.information(self, "Thông tin", "Không có vùng quét nào được chọn.")
                self.lbl_bbox.setText("Vùng quét: Chưa chọn")
                
    def browse_output(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Lưu Video Lồng Tiếng", "", "Video File (*.mp4)")
        if file_path:
            self.txt_out.setText(file_path)
            
    def browse_vp_dict(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn tệp VietPhrase.txt", "", "Text Files (*.txt)")
        if file_path:
            self.txt_vp_path.setText(file_path)
            
    def browse_names_dict(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn tệp Names.txt", "", "Text Files (*.txt)")
        if file_path:
            self.txt_names_path.setText(file_path)

    def start_extraction(self):
        video_path = self.video_path
        
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng chọn hoặc tải tệp video trước khi trích xuất phụ đề.")
            return
            
        mode = 'whisper' if self.cb_mode.currentIndex() == 0 else 'ocr'
        whisper_model = self.clean_combobox_value(self.cb_whisper_model.currentText())
        api_key = self.txt_gemini_key.text().strip()
        ocr_lang = self.cb_ocr_lang.currentText()
        
        self.txt_logs1.clear()
        self.btn_start_extract.setEnabled(False)
        self.status_label.setText("Đang trích xuất phụ đề...")
        
        worker_bbox = self.selected_bbox if mode == 'ocr' else None
        self.trans_thread = TranscriptionWorker(video_path, mode, worker_bbox, whisper_model, api_key, ocr_lang=ocr_lang)
        self.trans_thread.progress.connect(self.append_log1)
        self.trans_thread.finished.connect(self.on_extraction_finished)
        self.trans_thread.error.connect(self.on_extraction_error)
        self.trans_thread.start()
        
    def append_log1(self, text):
        self.txt_logs1.append(text)
        self.status_label.setText(text)
        
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
        
        # Bước 2: Gộp các segment có khoảng cách < 0.2s
        merged = []
        current = filtered[0]
        
        for next_seg in filtered[1:]:
            gap = next_seg['start'] - current['end']
            if gap < 0.2:
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
        return merged

    def on_extraction_finished(self, segments, video_path):
        self.btn_start_extract.setEnabled(True)
        self.video_path = video_path
        self.segments = self.preprocess_extracted_segments(segments)
        self.txt_file.setText(video_path)
        
        # Đề xuất đường dẫn đầu ra mặc định
        base, ext = os.path.splitext(video_path)
        self.txt_out.setText(base + "_longtieng" + ext)
        
        self.status_label.setText("Trích xuất phụ đề thành công!")
        QMessageBox.information(self, "Thành công", f"Đã trích xuất phụ đề thành công. Tìm thấy {len(segments)} câu phụ đề gốc.")
        
        # Hiển thị dữ liệu lên bảng ở Tab 3
        self.populate_subtitle_table()
        # Chuyển sang Tab biên tập
        self.tabs.setCurrentIndex(2)
        
    def on_extraction_error(self, err_msg):
        self.btn_start_extract.setEnabled(True)
        self.status_label.setText("Lỗi trích xuất.")
        QMessageBox.critical(self, "Lỗi hệ thống", f"Quá trình trích xuất gặp sự cố:\n{err_msg}")

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
        # Điền file kết quả mặc định
        base, ext = os.path.splitext(video_path)
        self.txt_out.setText(base + "_longtieng" + ext)
        
        self.status_label.setText("Tải video thành công!")
        QMessageBox.information(
            self, 
            "Tải thành công", 
            f"Đã tải xong video và lưu tại:\n{video_path}\n\nHệ thống sẽ chuyển bạn sang Tab 'Trích Phụ đề' để xử lý tiếp."
        )
        
        # Chuyển sang Tab 2 (Trích Phụ đề)
        self.tabs.setCurrentIndex(1)
        
    def on_download_error(self, err_msg):
        self.btn_download.setEnabled(True)
        self.status_label.setText("Lỗi tải video.")
        QMessageBox.critical(self, "Lỗi tải video", f"Quá trình tải video gặp sự cố:\n{err_msg}")
        
    def populate_subtitle_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.segments))
        
        for idx, seg in enumerate(self.segments):
            start = seg['start']
            end = seg['end']
            duration = end - start
            orig_text = seg.get('orig_text', seg.get('text') or "")
            raw_text = seg.get('raw_text') or ""
            trans_text = seg.get('text') or ""
            
            if not isinstance(raw_text, str):
                raw_text = str(raw_text)
            if not isinstance(trans_text, str):
                trans_text = str(trans_text)
            
            # Ước lượng thời gian TTS: 1 từ đọc khoảng 0.35 giây
            est_tts_dur = max(1.0, len(trans_text.split()) * 0.35)
            speed_factor = est_tts_dur / duration if duration > 0 else 1.0
            chars_per_sec = len(trans_text) / duration if duration > 0 else 0
            
            item_start = QTableWidgetItem(f"{start:.2f}")
            item_end = QTableWidgetItem(f"{end:.2f}")
            item_duration = QTableWidgetItem(f"{duration:.2f}")
            item_duration.setFlags(item_duration.flags() & ~Qt.ItemFlag.ItemIsEditable)
            
            item_orig = QTableWidgetItem(orig_text)
            item_orig.setFlags(item_orig.flags() & ~Qt.ItemFlag.ItemIsEditable)
            
            item_raw_trans = QTableWidgetItem(raw_text)
            item_raw_trans.setFlags(item_raw_trans.flags() & ~Qt.ItemFlag.ItemIsEditable)
            
            item_trans = QTableWidgetItem(trans_text)
            
            # Highlight nếu bản hoàn chỉnh khác với bản dịch thô (đã tinh chỉnh bởi LLM)
            if raw_text and trans_text and raw_text != trans_text:
                item_trans.setBackground(QColor(30, 60, 45)) # Teal/Green nền
                item_trans.setForeground(QColor(180, 240, 200)) # Green chữ
            
            item_factor = QTableWidgetItem(f"{speed_factor:.2f}x ({chars_per_sec:.1f} ch/s)")
            item_factor.setFlags(item_factor.flags() & ~Qt.ItemFlag.ItemIsEditable)
            
            # Cảnh báo dựa trên ngưỡng tốc độ đọc cấu hình động (mặc định 20 ký tự/s)
            speed_threshold = self.spin_speed_threshold.value() if hasattr(self, 'spin_speed_threshold') else 20
            if chars_per_sec > speed_threshold:
                item_factor.setBackground(QColor(255, 153, 153))
                item_factor.setForeground(QColor(102, 51, 0))
            elif speed_factor > 1.05:
                item_factor.setBackground(QColor(255, 204, 153))
                item_factor.setForeground(QColor(102, 51, 0))
                
            conf = seg.get('confidence')
            conf_str = f"{conf}%" if conf is not None else ""
            item_conf = QTableWidgetItem(conf_str)
            item_conf.setFlags(item_conf.flags() & ~Qt.ItemFlag.ItemIsEditable)
            
            # Cảnh báo confidence thấp bằng màu cam/vàng
            warn_threshold = self.spin_confidence_threshold.value()
            if conf is not None and conf < warn_threshold:
                item_conf.setText("⚠️ " + conf_str)
                item_conf.setBackground(QColor(255, 204, 153))
                item_conf.setForeground(QColor(102, 51, 0))
                
            bbox_val = seg.get('bbox')
            bbox_str = f"[{bbox_val[0]},{bbox_val[1]},{bbox_val[2]},{bbox_val[3]}]" if bbox_val else ""
            item_bbox = QTableWidgetItem(bbox_str)
            item_bbox.setFlags(item_bbox.flags() & ~Qt.ItemFlag.ItemIsEditable)
            
            self.table.setItem(idx, 0, item_start)
            self.table.setItem(idx, 1, item_end)
            self.table.setItem(idx, 2, item_duration)
            self.table.setItem(idx, 3, item_orig)
            self.table.setItem(idx, 4, item_raw_trans)
            self.table.setItem(idx, 5, item_trans)
            self.table.setItem(idx, 6, item_factor)
            self.table.setItem(idx, 7, item_conf)
            self.table.setItem(idx, 8, item_bbox)
            
        self.table.blockSignals(False)
        
    def on_cell_changed(self, row, column):
        if row < 0 or row >= len(self.segments):
            return
            
        # Khi sửa đổi bảng, cập nhật dữ liệu của segments
        self.table.blockSignals(True)
        try:
            if column == 0:  # Sửa Start
                val = float(self.table.item(row, 0).text())
                self.segments[row]['start'] = val
            elif column == 1:  # Sửa End
                val = float(self.table.item(row, 1).text())
                self.segments[row]['end'] = val
            elif column == 5:  # Sửa Bản hoàn chỉnh
                val = self.table.item(row, 5).text()
                self.segments[row]['text'] = val
                self.segments[row]['manual_override'] = True
                self.save_translation_cache()
                
            # Cập nhật lại thời lượng và Speed Factor
            start = self.segments[row]['start']
            end = self.segments[row]['end']
            duration = end - start
            trans_text = self.segments[row].get('text', '')
            
            # Cập nhật ô thời lượng
            self.table.item(row, 2).setText(f"{duration:.2f}")
            
            # Tính toán lại tốc độ đọc & giọng nói
            est_tts_dur = max(1.0, len(trans_text.split()) * 0.35)
            speed_factor = est_tts_dur / duration if duration > 0 else 1.0
            chars_per_sec = len(trans_text) / duration if duration > 0 else 0
            
            item_factor = self.table.item(row, 6)
            item_factor.setText(f"{speed_factor:.2f}x ({chars_per_sec:.1f} ch/s)")
            
            # Reset màu
            item_factor.setBackground(QColor(20, 20, 22)) # Background mặc định table (#141416)
            item_factor.setForeground(QColor(243, 242, 238)) # Champagne (#f3f2ee)
            
            # Cảnh báo dựa trên ngưỡng tốc độ đọc cấu hình động (mặc định 20 ký tự/s)
            speed_threshold = self.spin_speed_threshold.value() if hasattr(self, 'spin_speed_threshold') else 20
            if chars_per_sec > speed_threshold:
                item_factor.setBackground(QColor(255, 153, 153))
                item_factor.setForeground(QColor(102, 51, 0))
            elif speed_factor > 1.05:
                item_factor.setBackground(QColor(255, 204, 153))
                item_factor.setForeground(QColor(102, 51, 0))
                
        except ValueError:
            # Nếu người dùng nhập sai định dạng số
            pass
        finally:
            self.table.blockSignals(False)
            
    def start_translation(self):
        if not self.segments:
            QMessageBox.warning(self, "Cảnh báo", "Không có phụ đề nào để dịch.")
            return
            
        engine = self.clean_combobox_value(self.cb_engine.currentText())
        
        # Ánh xạ từ các tùy chọn tinh giản sang cấu hình backend thực tế
        refine_enabled = False
        refine_engine = "Ollama Local"
        refine_api_key = ""
        
        if engine == "Supersubs AI":
            # Giai đoạn 1: Dịch thô bằng Quick Translator (VietPhrase)
            backend_engine = "Quick Translator (VietPhrase)"
            # Giai đoạn 2: Sơ chế bằng LLM (Ollama Local)
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
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn 1 dòng phụ đề để phát video thử.")
            return
            
        if not self.video_path or not os.path.exists(self.video_path):
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy file video gốc.")
            return
            
        start_s = self.segments[row]['start']
        end_s = self.segments[row]['end']
        text = self.segments[row]['text'] # Chữ dịch tiếng Việt dùng làm phụ đề xem thử
        
        self.status_label.setText(f"Đang phát preview phân đoạn: {start_s}s -> {end_s}s")
        
        # Chạy trong luồng phụ để tránh đơ app
        class VideoPreviewThread(QThread):
            def __init__(self, video_path, start, end, sub_text):
                super().__init__()
                self.video_path = video_path
                self.start_s = start
                self.end_s = end
                self.sub_text = sub_text
                
            def run(self):
                cap = cv2.VideoCapture(self.video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.start_s * fps))
                delay = int(1000 / fps)
                while cap.isOpened():
                    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if current_frame > int(self.end_s * fps):
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_resized = cv2.resize(frame, (640, 360))
                    # Vẽ phụ đề tiếng Việt có viền nền đen mờ lên video
                    if self.sub_text:
                        frame_resized = draw_subtitle_on_frame(frame_resized, self.sub_text)
                        
                    cv2.imshow("Preview Video Segment (Bam Q de tat)", frame_resized)
                    if cv2.waitKey(delay) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
                
        self.preview_video_thread = VideoPreviewThread(self.video_path, start_s, end_s, text)
        self.preview_video_thread.start()
        
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
        if not self.segments:
            QMessageBox.warning(self, "Cảnh báo", "Không có phụ đề để lồng tiếng.")
            return
            
        out_path = self.txt_out.text().strip()
        if not out_path:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn đường dẫn lưu video kết quả.")
            return
            
        voice = self.cb_voice.currentData()
        bg_vol = self.slider_bg.value() / 100.0
        dub_vol = self.slider_dub.value() / 100.0
        
        self.txt_logs3.clear()
        self.btn_start_dub.setEnabled(False)
        self.status_label.setText("Bắt đầu kết xuất lồng tiếng...")
        
        burn_sub = self.chk_burn_sub.isChecked()
        selected_bbox = self.selected_bbox
        preset = self.get_current_subtitle_preset()
        
        self.dub_thread = DubbingWorker(
            self.video_path,
            self.segments,
            voice,
            out_path,
            bg_vol,
            dub_vol,
            burn_subtitles=burn_sub,
            selected_bbox=selected_bbox,
            preset=preset
        )
        self.dub_thread.progress.connect(self.append_log3)
        self.dub_thread.finished.connect(self.on_dubbing_finished)
        self.dub_thread.error.connect(self.on_dubbing_error)
        self.dub_thread.start()
        
    def append_log3(self, text):
        self.txt_logs3.append(text)
        self.status_label.setText(text)
        
    def on_dubbing_finished(self, out_video):
        self.btn_start_dub.setEnabled(True)
        self.status_label.setText("Kết xuất lồng tiếng hoàn tất!")
        QMessageBox.information(
            self, 
            "Xuất Video Thành Công!", 
            f"Video lồng tiếng của bạn đã được xuất thành công!\nĐường dẫn: {out_video}"
        )
        
    def on_dubbing_error(self, err):
        self.btn_start_dub.setEnabled(True)
        self.status_label.setText("Lỗi kết xuất lồng tiếng.")
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
        txt.editingFinished.connect(on_txt_edited)
        
    def update_color_button(self, btn, txt, rgb):
        qcolor = QColor(*rgb)
        btn.setStyleSheet(f"background-color: {qcolor.name()}; border: 1px solid #1c1c1f;")
        txt.blockSignals(True)
        txt.setText(qcolor.name().upper())
        txt.blockSignals(False)
        
    def get_current_subtitle_preset(self):
        font_name = self.cb_font_name.currentText()
        if self.custom_font_path and font_name == os.path.basename(self.custom_font_path):
            font_name = self.custom_font_path
            
        mask_modes_map = ["none", "black", "blur", "inpaint"]
        mask_mode = mask_modes_map[self.cb_mask_mode.currentIndex()]
        
        remove_algos_map = ["ffmpeg", "opencv"]
        remove_algo = remove_algos_map[self.cb_remove_algo.currentIndex()]
            
        preset = {
            "v_align": self.cb_v_align.currentText().lower(),
            "h_align": self.cb_h_align.currentText().lower(),
            "margin_v_type": "percent" if self.cb_margin_v_type.currentIndex() == 0 else "pixels",
            "margin_v_val": self.spin_margin_v.value(),
            "margin_h_type": "percent" if self.cb_margin_h_type.currentIndex() == 0 else "pixels",
            "margin_h_val": self.spin_margin_h.value(),
            "font_name": font_name,
            "font_size": self.spin_font_size.value(),
            "font_color": self.preset_font_color,
            "outline_color": self.preset_outline_color,
            "outline_width": self.spin_outline_width.value(),
            "bg_color": self.preset_bg_color,
            "bg_opacity": self.slider_bg_opacity.value(),
            "use_bg_box": self.chk_use_bg_box.isChecked(),
            "mask_mode": mask_mode,
            "remove_algo": remove_algo,
            "smart_pos": self.chk_smart_pos.isChecked()
        }
        if self.subtitle_custom_pos:
            preset["custom_pos"] = dict(self.subtitle_custom_pos)
        return preset
        
    def set_subtitle_preset_ui(self, preset_dict, apply_style=True):
        self.cb_v_align.setCurrentText(preset_dict["v_align"].capitalize())
        self.cb_h_align.setCurrentText(preset_dict["h_align"].capitalize())
        self.cb_margin_v_type.setCurrentIndex(0 if preset_dict["margin_v_type"] == "percent" else 1)
        self.spin_margin_v.setValue(preset_dict["margin_v_val"])
        
        if "mask_mode" in preset_dict:
            mask_modes_map = ["none", "black", "blur", "inpaint"]
            try:
                idx = mask_modes_map.index(preset_dict["mask_mode"])
                self.cb_mask_mode.setCurrentIndex(idx)
            except ValueError:
                pass
                
        if "remove_algo" in preset_dict:
            remove_algos_map = ["ffmpeg", "opencv"]
            try:
                idx = remove_algos_map.index(preset_dict["remove_algo"])
                self.cb_remove_algo.setCurrentIndex(idx)
            except ValueError:
                pass
                
        if "smart_pos" in preset_dict:
            self.chk_smart_pos.setChecked(preset_dict["smart_pos"])
            
        self.cb_margin_h_type.setCurrentIndex(0 if preset_dict["margin_h_type"] == "percent" else 1)
        self.spin_margin_h.setValue(preset_dict["margin_h_val"])
        
        if apply_style:
            font_name = preset_dict["font_name"]
            if font_name.endswith(('.ttf', '.otf', '.ttc')) and os.path.exists(font_name):
                self.custom_font_path = font_name
                basename = os.path.basename(font_name)
                idx = self.cb_font_name.findText(basename)
                if idx == -1:
                    self.cb_font_name.addItem(basename)
                self.cb_font_name.setCurrentText(basename)
            else:
                self.cb_font_name.setCurrentText(font_name)
                
            self.spin_font_size.setValue(preset_dict["font_size"])
            
            self.preset_font_color = preset_dict["font_color"]
            self.update_color_button(self.btn_font_color, self.txt_font_color_hex, self.preset_font_color)
            
            self.preset_outline_color = preset_dict["outline_color"]
            self.update_color_button(self.btn_outline_color, self.txt_outline_color_hex, self.preset_outline_color)
            
            self.preset_bg_color = preset_dict["bg_color"]
            self.update_color_button(self.btn_bg_color, self.txt_bg_color_hex, self.preset_bg_color)
            
            self.spin_outline_width.setValue(preset_dict["outline_width"])
            self.chk_use_bg_box.setChecked(preset_dict["use_bg_box"])
            self.slider_bg_opacity.setValue(preset_dict["bg_opacity"])
            
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
        is_custom = (text == "Tùy chỉnh (Custom)")
        if hasattr(self, 'spin_custom_pos_x'):
            self.spin_custom_pos_x.setEnabled(is_custom)
            self.spin_custom_pos_y.setEnabled(is_custom)
            self.btn_reset_custom_pos.setEnabled(is_custom)
        if text in self.presets_db:
            self.subtitle_custom_pos = None
            apply_style = self.cb_preset_apply_mode.currentIndex() == 1
            self.set_subtitle_preset_ui(self.presets_db[text], apply_style=apply_style)
        elif is_custom:
            if not self.subtitle_custom_pos:
                self.subtitle_custom_pos = {
                    "x_pct": self.spin_custom_pos_x.value(),
                    "y_pct": self.spin_custom_pos_y.value()
                }
            self.trigger_canvas_update()
            
    def on_custom_pos_spin_changed(self):
        self.subtitle_custom_pos = {
            "x_pct": self.spin_custom_pos_x.value(),
            "y_pct": self.spin_custom_pos_y.value()
        }
        self.mark_preset_custom()

    def mark_preset_custom(self):
        self.cb_preset.blockSignals(True)
        self.cb_preset.setCurrentText("Tùy chỉnh (Custom)")
        self.cb_preset.blockSignals(False)
        if hasattr(self, 'spin_custom_pos_x'):
            self.spin_custom_pos_x.setEnabled(True)
            self.spin_custom_pos_y.setEnabled(True)
            self.btn_reset_custom_pos.setEnabled(True)
        
    def on_font_changed(self):
        self.mark_preset_custom()
        font_name = self.cb_font_name.currentText()
        font_path = dubber.get_font_path(font_name)
        if font_path and os.path.exists(font_path):
            if not check_font_vietnamese_support(font_path):
                QMessageBox.warning(self, "Cảnh báo Font chữ", 
                                    f"Font '{font_name}' có thể không hỗ trợ đầy đủ tiếng Việt Unicode có dấu!\n"
                                    "Nếu hiển thị bị lỗi, hãy chọn các font thay thế như: Arial, Noto Sans, Roboto, Segoe UI, Tahoma.")
                                    
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
            
            def __init__(self, video_path, segments, ocr_lang, restrict_region):
                super().__init__()
                self.video_path = video_path
                self.segments = segments
                self.ocr_lang = ocr_lang
                self.restrict_region = restrict_region
                
            def run(self):
                try:
                    res = transcriber.run_segment_guided_ocr(
                        self.video_path,
                        self.segments,
                        progress_callback=self.progress.emit,
                        ocr_lang=self.ocr_lang,
                        restrict_region=self.restrict_region
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
        engine = self.clean_combobox_value(self.cb_engine.currentText())
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
            
        glossary = self.parse_glossary_text(self.txt_glossary.toPlainText())
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
        row = self.table.currentRow()
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
        bbox = self.segments[row].get('bbox')
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
        lbl_w = self.lbl_main_preview.width()
        lbl_h = self.lbl_main_preview.height()
        if lbl_w < 100 or lbl_h < 100:
            lbl_w, lbl_h = 640, 360
            
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_image.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        scaled_pixmap = pixmap.scaled(
            lbl_w, 
            lbl_h, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_main_preview.setPixmap(scaled_pixmap)

    def apply_dark_theme(self):
        # Premium Dark Theme style (Neo Kinpaku design system by Impeccable)
        self.setStyleSheet("""
            QWidget {
                background-color: #0c0c0e;
                color: #f3f2ee;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QFrame#header {
                background-color: #141416;
                border-bottom: 1px solid #1c1c1f;
            }
            QFrame#card {
                background-color: #141416;
                border: 1px solid #1c1c1f;
                border-radius: 8px;
            }
            QFrame#card QLabel {
                color: #f3f2ee;
            }
            /* Style first label in cards as card title */
            QFrame#card > QLabel {
                color: #dfb15b;
                font-weight: bold;
                font-size: 11px;
            }
            QTabWidget::pane {
                border: 1px solid #1c1c1f;
                background-color: #0c0c0e;
            }
            QTabBar::tab {
                background-color: #141416;
                color: #9c9c9f;
                padding: 8px 16px;
                border: 1px solid #1c1c1f;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0c0c0e;
                color: #dfb15b;
                border-bottom: 2px solid #dfb15b;
                font-weight: bold;
            }
            QPushButton {
                background-color: #141416;
                color: #f3f2ee;
                border: 1px solid #1c1c1f;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1c1c1f;
                border-color: #2e2e33;
            }
            QPushButton:pressed {
                background-color: #2e2e33;
            }
            QLineEdit, QComboBox, QTextEdit {
                background-color: #141416;
                border: 1px solid #1c1c1f;
                border-radius: 4px;
                padding: 5px;
                color: #f3f2ee;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
                border: 1px solid #dfb15b;
            }
            QSlider::groove:horizontal {
                border: 1px solid #1c1c1f;
                height: 8px;
                background: #141416;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #dfb15b;
                border: 1px solid #2e2e33;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QTableWidget {
                background-color: #141416;
                gridline-color: #1c1c1f;
                border: 1px solid #1c1c1f;
                alternate-background-color: #0c0c0e;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #1c1c1f;
                color: #dfb15b;
            }
            QHeaderView::section {
                background-color: #141416;
                color: #f3f2ee;
                padding: 5px;
                border: 1px solid #1c1c1f;
                font-weight: bold;
            }
            QScrollBar:vertical {
                border: none;
                background: #141416;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #1c1c1f;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #2e2e33;
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
            if self.gemini_key:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
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

from PyQt6.QtWidgets import QSplitter, QScrollArea, QListWidget, QListWidgetItem, QDoubleSpinBox, QProgressBar

class ScriptVoiceoverTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.segments_data = [] # list of dicts
        self.batch_worker = None
        self.supported_voices = dubber.get_supported_voices()
        
        self.init_ui()
        self.refresh_prompts()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 1. THƯ VIỆN PROMPT (Collapsible Card ở trên cùng)
        self.card_prompts = CollapsibleCard("💡 THƯ VIỆN PROMPT", "vertical")
        self.card_prompts.toggle_collapse() # Đóng mặc định để tiết kiệm diện tích
        
        # Ô tìm kiếm prompt
        self.txt_search_prompt = QLineEdit()
        self.txt_search_prompt.setPlaceholderText("Tìm kiếm prompt theo tiêu đề, nội dung, tags...")
        self.txt_search_prompt.textChanged.connect(self.refresh_prompts)
        self.card_prompts.addWidget(self.txt_search_prompt)
        
        # List hiển thị prompt
        self.list_prompts = QListWidget()
        self.list_prompts.setMaximumHeight(130)
        self.list_prompts.currentRowChanged.connect(self.on_prompt_selected)
        self.card_prompts.addWidget(self.list_prompts)
        
        # Preview prompt
        self.card_prompts.addWidget(QLabel("Xem trước Prompt:"))
        self.txt_prompt_preview = QTextEdit()
        self.txt_prompt_preview.setReadOnly(True)
        self.txt_prompt_preview.setMaximumHeight(80)
        self.card_prompts.addWidget(self.txt_prompt_preview)
        
        # Hàng nút hành động Prompt
        prompt_btn_layout = QHBoxLayout()
        self.btn_copy_prompt = QPushButton("Sao chép prompt")
        self.btn_copy_prompt.setStyleSheet("background-color: #7fbeb2; color: #0c0c0e; font-weight: bold; padding: 6px;")
        self.btn_copy_prompt.clicked.connect(self.copy_selected_prompt)
        prompt_btn_layout.addWidget(self.btn_copy_prompt)
        
        self.btn_add_prompt = QPushButton("Thêm prompt")
        self.btn_add_prompt.setStyleSheet("background-color: #dfb15b; color: #0c0c0e; font-weight: bold; padding: 6px;")
        self.btn_add_prompt.clicked.connect(self.add_new_prompt)
        prompt_btn_layout.addWidget(self.btn_add_prompt)
        
        self.btn_delete_prompt = QPushButton("Xóa prompt")
        self.btn_delete_prompt.setStyleSheet("background-color: #ff5555; color: white; font-weight: bold; padding: 6px;")
        self.btn_delete_prompt.clicked.connect(self.delete_selected_prompt)
        prompt_btn_layout.addWidget(self.btn_delete_prompt)
        
        self.card_prompts.addLayout(prompt_btn_layout)
        main_layout.addWidget(self.card_prompts)
        
        # 2. THANH CHIA KÉO GIÃN DỌC (Dành cho Kịch bản và Bảng đoạn dịch)
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)
        
        # Khối trên: Ô nhập kịch bản & nút tách & video mẫu
        upper_widget = QWidget()
        upper_layout = QVBoxLayout(upper_widget)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        upper_layout.setSpacing(6)
        
        lbl_right_title = QLabel("📝 Kịch bản đọc & tạo giọng nói")
        lbl_right_title.setStyleSheet("color: #dfb15b; font-weight: bold; font-size: 15px;")
        upper_layout.addWidget(lbl_right_title)
        
        # Hàng nạp Video mẫu đầu vào
        video_row = QHBoxLayout()
        video_row.addWidget(QLabel("Video mẫu (tùy chọn):"))
        self.txt_script_video = QLineEdit()
        self.txt_script_video.setReadOnly(True)
        self.txt_script_video.setPlaceholderText("Chọn video mẫu nếu muốn sinh kịch bản hoặc đồng bộ hình ảnh")
        video_row.addWidget(self.txt_script_video)
        
        self.btn_browse_script_video = QPushButton("Chọn video...")
        self.btn_browse_script_video.setStyleSheet("background-color: #141416; border: 1px solid #1c1c1f; color: #dfb15b; padding: 4px 10px;")
        self.btn_browse_script_video.clicked.connect(self.browse_script_video)
        video_row.addWidget(self.btn_browse_script_video)
        
        self.btn_ai_gen_script = QPushButton("🤖 AI sinh kịch bản từ video")
        self.btn_ai_gen_script.setStyleSheet("background-color: #dfb15b; color: #0c0c0e; font-weight: bold; padding: 4px 10px;")
        self.btn_ai_gen_script.clicked.connect(self.generate_script_from_video)
        video_row.addWidget(self.btn_ai_gen_script)
        upper_layout.addLayout(video_row)
        
        self.txt_script = QTextEdit()
        self.txt_script.setPlaceholderText("Dán hoặc viết lời đọc tại đây. Nên dùng câu ngắn, mỗi ý một nhịp để giọng đọc giống video mạng xã hội hơn.")
        self.txt_script.setMinimumHeight(120)
        upper_layout.addWidget(self.txt_script)
        
        script_btn_layout = QHBoxLayout()
        self.btn_split = QPushButton("Tách câu đọc")
        self.btn_split.setStyleSheet("background-color: #dfb15b; color: #0c0c0e; font-weight: bold; padding: 6px 12px;")
        self.btn_split.clicked.connect(self.split_script_segment)
        script_btn_layout.addWidget(self.btn_split)
        
        self.btn_history = QPushButton("Lịch sử")
        self.btn_history.setStyleSheet("background-color: #141416; border: 1px solid #1c1c1f; color: #dfb15b; padding: 6px 12px;")
        self.btn_history.clicked.connect(self.show_script_history_dialog)
        script_btn_layout.addWidget(self.btn_history)
        script_btn_layout.addStretch()
        upper_layout.addLayout(script_btn_layout)

        self.lbl_script_stats = QLabel("0 câu đọc • 0 ký tự")
        self.lbl_script_stats.setStyleSheet("color: #9c9c9f; font-size: 11px;")
        upper_layout.addWidget(self.lbl_script_stats)
        
        self.lbl_warning_notice = QLabel("Đã sẵn sàng")
        self.lbl_warning_notice.setStyleSheet("color: #7fbeb2; font-weight: bold; font-size: 11px; margin-top: 5px;")
        upper_layout.addWidget(self.lbl_warning_notice)
        
        splitter.addWidget(upper_widget)
        
        # Khối dưới: Cấu hình TTS + Bảng phân đoạn + Nút tạo + Card Xuất
        lower_widget = QScrollArea() # Sử dụng ScrollArea để tránh bị chèn ép nội dung khi thu hẹp
        lower_widget.setWidgetResizable(True)
        lower_widget.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        lower_content = QWidget()
        lower_layout = QVBoxLayout(lower_content)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        lower_layout.setSpacing(10)
        
        # Cấu hình giọng đọc (TTS Engine)
        card_tts = CollapsibleCard("CẤU HÌNH GIỌNG ĐỌC", "grid")
        
        v_layout = QVBoxLayout()
        v_layout.addWidget(QLabel("Giọng đọc mặc định:"))
        self.cb_script_voice = QComboBox()
        for v in self.supported_voices:
            self.cb_script_voice.addItem(v["desc"], v["name"])
        v_layout.addWidget(self.cb_script_voice)
        card_tts.addLayout(v_layout, 1, 0)
        
        s_layout = QVBoxLayout()
        s_layout.addWidget(QLabel("Khoảng lặng giữa câu:"))
        self.spin_silence = QDoubleSpinBox()
        self.spin_silence.setRange(0.0, 10.0)
        self.spin_silence.setSingleStep(0.05)
        self.spin_silence.setValue(0.35)
        self.spin_silence.setSuffix(" giây")
        s_layout.addWidget(self.spin_silence)
        card_tts.addLayout(s_layout, 1, 1)
        
        sp_layout = QHBoxLayout()
        sp_layout.addWidget(QLabel("Tốc độ đọc:"))
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setRange(-50, 100)
        self.slider_speed.setValue(0)
        self.lbl_speed_val = QLabel("0%")
        self.slider_speed.valueChanged.connect(lambda v: self.lbl_speed_val.setText(f"{v:+}%" if v != 0 else "0%"))
        sp_layout.addWidget(self.slider_speed)
        sp_layout.addWidget(self.lbl_speed_val)
        card_tts.addLayout(sp_layout, 2, 0, 1, 2)
        
        lower_layout.addWidget(card_tts)
        
        # Bảng segment
        self.table_segments = QTableWidget()
        self.table_segments.setColumnCount(6)
        self.table_segments.setHorizontalHeaderLabels(["STT", "Nội dung", "Giọng", "File audio", "Thời lượng", "Trạng thái"])
        self.table_segments.setWordWrap(True)
        self.table_segments.verticalHeader().setDefaultSectionSize(54)
        self.table_segments.setAlternatingRowColors(True)
        self.table_segments.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_segments.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_segments.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_segments.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_segments.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table_segments.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table_segments.itemChanged.connect(self.on_table_item_changed)
        self.table_segments.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_segments.customContextMenuRequested.connect(self.show_table_context_menu)
        self.cb_script_voice.currentIndexChanged.connect(self.on_global_voice_changed)
        self.table_segments.setMinimumHeight(220)
        lower_layout.addWidget(self.table_segments)
        
        # Hàng điều khiển Tạo giọng & Tiến trình
        batch_control_layout = QHBoxLayout()
        self.btn_generate_all = QPushButton("Tạo giọng cho các câu cần tạo")
        self.btn_generate_all.setStyleSheet("background-color: #7fbeb2; color: #0c0c0e; font-weight: bold; padding: 8px 16px;")
        self.btn_generate_all.clicked.connect(self.generate_batch_tts)
        batch_control_layout.addWidget(self.btn_generate_all)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        batch_control_layout.addWidget(self.progress_bar)
        lower_layout.addLayout(batch_control_layout)
        
        # Card Xuất kết quả
        card_export = CollapsibleCard("XUẤT AUDIO, PHỤ ĐỀ VÀ VIDEO", "vertical")
        
        out_dir_row = QHBoxLayout()
        out_dir_row.addWidget(QLabel("Thư mục đầu ra:"))
        self.txt_out_dir = QLineEdit()
        default_out = os.path.abspath(os.path.join(".", "Output_Script"))
        self.txt_out_dir.setText(default_out)
        out_dir_row.addWidget(self.txt_out_dir)
        
        btn_browse_out = QPushButton("Chọn...")
        btn_browse_out.setStyleSheet("background-color: #141416; border: 1px solid #1c1c1f; color: #dfb15b; padding: 4px 10px;")
        btn_browse_out.clicked.connect(self.browse_out_dir)
        out_dir_row.addWidget(btn_browse_out)
        card_export.addLayout(out_dir_row)
        
        sync_row = QHBoxLayout()
        sync_row.addWidget(QLabel("Đồng bộ video mẫu:"))
        self.cb_sync_mode = QComboBox()
        self.cb_sync_mode.addItems([
            "Co giãn tốc độ (Speed Stretch) - Khớp khít hình với tiếng",
            "Đứng hình chờ tiếng (Freeze Frame) - Đóng băng khung hình cuối",
            "Không đồng bộ (Chỉ ghép Audio + SRT)"
        ])
        sync_row.addWidget(self.cb_sync_mode)
        card_export.addLayout(sync_row)
        
        self.btn_export = QPushButton("⚡ Ghép audio và xuất phụ đề SRT")
        self.btn_export.setStyleSheet("background-color: #dfb15b; color: #0c0c0e; font-weight: bold; padding: 8px 16px; font-size: 13px;")
        self.btn_export.clicked.connect(self.export_final_results)
        card_export.addWidget(self.btn_export)
        
        self.btn_export_video = QPushButton("🎬 Ghép audio, đồng bộ và xuất video")
        self.btn_export_video.setStyleSheet("background-color: #7fbeb2; color: #0c0c0e; font-weight: bold; padding: 8px 16px; font-size: 13px; margin-top: 5px;")
        self.btn_export_video.clicked.connect(self.export_final_video)
        card_export.addWidget(self.btn_export_video)
        
        lower_layout.addWidget(card_export)
        
        lower_widget.setWidget(lower_content)
        splitter.addWidget(lower_widget)
        
        splitter.setSizes([230, 520])
        
    def refresh_prompts(self):
        self.list_prompts.clear()
        search_query = self.txt_search_prompt.text().strip().lower()
        
        self.loaded_prompts = DEFAULT_PROMPTS + load_custom_prompts()
        
        self.displayed_prompts = []
        for p in self.loaded_prompts:
            title = p.get("title", "")
            content = p.get("content", "")
            tags = p.get("tags", [])
            
            if search_query:
                match_title = search_query in title.lower()
                match_content = search_query in content.lower()
                match_tags = any(search_query in t.lower() for t in tags)
                if not (match_title or match_content or match_tags):
                    continue
            
            self.displayed_prompts.append(p)
            self.list_prompts.addItem(title)
            
        self.txt_prompt_preview.clear()
        
    def on_prompt_selected(self, index):
        if 0 <= index < len(self.displayed_prompts):
            p = self.displayed_prompts[index]
            self.txt_prompt_preview.setPlainText(p.get("content", ""))
            
    def copy_selected_prompt(self):
        row = self.list_prompts.currentRow()
        if 0 <= row < len(self.displayed_prompts):
            p = self.displayed_prompts[row]
            content = p.get("content", "")
            QApplication.clipboard().setText(content)
            if self.parent:
                self.parent.status_label.setText("Đã sao chép prompt vào Clipboard!")
        else:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn một prompt từ danh sách.")
            
    def add_new_prompt(self):
        dialog = CustomPromptDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            if not data["title"] or not data["content"]:
                QMessageBox.warning(self, "Lỗi", "Không được để trống Tiêu đề và Nội dung Prompt.")
                return
            add_custom_prompt(data["title"], data["content"], data["tags"])
            self.refresh_prompts()
            
    def delete_selected_prompt(self):
        row = self.list_prompts.currentRow()
        if 0 <= row < len(self.displayed_prompts):
            p = self.displayed_prompts[row]
            prompt_id = p.get("id", "")
            if prompt_id.startswith("default_"):
                QMessageBox.warning(self, "Lỗi", "Không thể xóa prompt mặc định!")
                return
            
            confirm = QMessageBox.question(
                self, "Xác nhận", f"Bạn có chắc muốn xóa prompt '{p.get('title')}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if confirm == QMessageBox.StandardButton.Yes:
                delete_custom_prompt(prompt_id)
                self.refresh_prompts()
        else:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn một prompt tùy chỉnh để xóa.")
            
    def browse_script_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn Video mẫu", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if file_path:
            self.txt_script_video.setText(os.path.abspath(file_path))
            if self.parent:
                self.parent.status_label.setText(f"Đã chọn video mẫu: {os.path.basename(file_path)}")
                
    def generate_script_from_video(self):
        video_path = self.txt_script_video.text().strip()
        if not video_path:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn video mẫu trước.")
            return
            
        gemini_key = ""
        groq_key = ""
        ollama_model = "qwen2.5"
        whisper_model = "base"
        
        # Lấy cấu hình từ main_window
        if self.parent:
            gemini_key = self.parent.txt_gemini_key.text().strip()
            groq_key = self.parent.txt_groq_key.text().strip()
            ollama_model = self.parent.txt_ollama_model.text().strip()
            whisper_model = self.parent.cb_whisper_model.currentText().strip()
            
        self.btn_ai_gen_script.setEnabled(False)
        self.btn_ai_gen_script.setText("🤖 Đang quét video...")
        
        self.ai_worker = AIScriptGeneratorWorker(
            video_path=video_path,
            gemini_key=gemini_key,
            groq_key=groq_key,
            ollama_model=ollama_model,
            whisper_model=whisper_model
        )
        self.ai_worker.progress.connect(self.on_ai_gen_progress)
        self.ai_worker.finished.connect(self.on_ai_gen_finished)
        self.ai_worker.error.connect(self.on_ai_gen_error)
        self.ai_worker.start()
        
    def on_ai_gen_progress(self, msg):
        self.lbl_warning_notice.setText(msg)
        self.lbl_warning_notice.setStyleSheet("color: #dfb15b; font-weight: bold;")
        if self.parent:
            self.parent.status_label.setText(msg)
            
    def on_ai_gen_finished(self, script_text):
        self.btn_ai_gen_script.setEnabled(True)
        self.btn_ai_gen_script.setText("🤖 AI sinh kịch bản từ video")
        self.txt_script.setPlainText(script_text)
        self.lbl_warning_notice.setText("Đã sinh kịch bản thành công!")
        self.lbl_warning_notice.setStyleSheet("color: #7fbeb2; font-weight: bold;")
        if self.parent:
            self.parent.status_label.setText("Đã sinh kịch bản thành công!")
            
    def on_ai_gen_error(self, err_msg):
        self.btn_ai_gen_script.setEnabled(True)
        self.btn_ai_gen_script.setText("🤖 AI sinh kịch bản từ video")
        QMessageBox.critical(self, "Lỗi sinh kịch bản", err_msg)
        self.lbl_warning_notice.setText("Sinh kịch bản thất bại.")
        self.lbl_warning_notice.setStyleSheet("color: #ff5555; font-weight: bold;")
        
    def export_final_video(self):
        video_path = self.txt_script_video.text().strip()
        if not video_path:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn video mẫu ở phần nhập kịch bản trước.")
            return
            
        # Kiểm tra xem đã tạo giọng đầy đủ chưa
        valid_segs = [seg for seg in self.segments_data if seg.get("audio_path") and os.path.exists(seg["audio_path"])]
        if not valid_segs:
            QMessageBox.warning(self, "Lỗi", "Vui lòng tạo giọng nói hàng loạt trước khi ghép video.")
            return
            
        out_dir = self.txt_out_dir.text().strip()
        os.makedirs(out_dir, exist_ok=True)
        
        silence_sec = self.spin_silence.value()
        sync_mode = self.cb_sync_mode.currentText().strip()
        
        self.btn_export_video.setEnabled(False)
        self.btn_export_video.setText("🎬 Đang đồng bộ video...")
        
        self.video_worker = ExportVideoSyncWorker(
            video_path=video_path,
            segments=self.segments_data,
            silence_sec=silence_sec,
            sync_mode=sync_mode,
            output_dir=out_dir
        )
        self.video_worker.progress.connect(self.on_export_video_progress)
        self.video_worker.finished.connect(self.on_export_video_finished)
        self.video_worker.error.connect(self.on_export_video_error)
        self.video_worker.start()
        
    def on_export_video_progress(self, msg):
        self.lbl_warning_notice.setText(msg)
        self.lbl_warning_notice.setStyleSheet("color: #dfb15b; font-weight: bold;")
        if self.parent:
            self.parent.status_label.setText(msg)
            
    def on_export_video_finished(self, out_video):
        self.btn_export_video.setEnabled(True)
        self.btn_export_video.setText("🎬 Ghép audio, đồng bộ và xuất video")
        self.lbl_warning_notice.setText("Xuất video thành công!")
        self.lbl_warning_notice.setStyleSheet("color: #7fbeb2; font-weight: bold;")
        
        # Mở thư mục kết quả
        try: os.startfile(os.path.dirname(out_video))
        except: pass
        
        QMessageBox.information(
            self, "Xuất Video thành công",
            f"Đã xuất video thành công!\n\n🎬 Video: {out_video}"
        )
        
    def on_export_video_error(self, err_msg):
        self.btn_export_video.setEnabled(True)
        self.btn_export_video.setText("🎬 Ghép audio, đồng bộ và xuất video")
        QMessageBox.critical(self, "Lỗi xuất video", err_msg)
        self.lbl_warning_notice.setText("Xuất video thất bại.")
        self.lbl_warning_notice.setStyleSheet("color: #ff5555; font-weight: bold;")

    def split_script_segment(self):
        script_text = self.txt_script.toPlainText().strip()
        if not script_text:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập hoặc dán kịch bản.")
            return
            
        segments = split_script_to_sentences(script_text)
        if not segments:
            QMessageBox.warning(self, "Lỗi", "Không thể tách đoạn từ kịch bản này.")
            return
            
        default_voice = self.cb_script_voice.currentData()
        for seg in segments:
            seg["voice"] = default_voice
            
        title = script_text[:60]
        add_script_history(title, script_text, len(segments))
        
        self.segments_data = segments
        self.populate_segments_table()
        self.update_script_stats()
        self.lbl_warning_notice.setText(f"Đã tách {len(segments)} câu đọc. Có thể tạo giọng ngay.")
        self.lbl_warning_notice.setStyleSheet("color: #7fbeb2; font-weight: bold;")
        
    def populate_segments_table(self):
        self.table_segments.blockSignals(True)
        self.table_segments.setRowCount(0)
        
        self.table_segments.setRowCount(len(self.segments_data))
        voice_desc = self.cb_script_voice.currentText()
        
        for idx, seg in enumerate(self.segments_data):
            stt_item = QTableWidgetItem(str(seg["index"]))
            stt_item.setFlags(stt_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_segments.setItem(idx, 0, stt_item)
            
            content_item = QTableWidgetItem(seg["text"])
            self.table_segments.setItem(idx, 1, content_item)
            
            # Dropdown chọn giọng cho dòng này
            voice_combo = QComboBox()
            for v in self.supported_voices:
                voice_combo.addItem(v["desc"], v["name"])
            
            seg_voice = seg.get("voice", self.cb_script_voice.currentData())
            idx_v = voice_combo.findData(seg_voice)
            if idx_v >= 0:
                voice_combo.setCurrentIndex(idx_v)
            else:
                voice_combo.setCurrentIndex(0)
                
            voice_combo.currentIndexChanged.connect(lambda _, r=idx, cb=voice_combo: self.on_row_voice_changed(r, cb))
            self.table_segments.setCellWidget(idx, 2, voice_combo)
            
            audio_item = QTableWidgetItem(seg.get("audio_path", ""))
            audio_item.setFlags(audio_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_segments.setItem(idx, 3, audio_item)
            
            dur_val = seg.get("duration", 0.0)
            duration_item = QTableWidgetItem(f"{dur_val:.2f}s" if dur_val > 0 else "")
            duration_item.setFlags(duration_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_segments.setItem(idx, 4, duration_item)
            
            status_text = "Cần tạo lại" if seg.get("needs_regen", True) else "Đã sẵn sàng"
            status_item = QTableWidgetItem(status_text)
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            
            if seg.get("needs_regen", True):
                status_item.setBackground(QColor(180, 110, 0)) # Cam
                status_item.setForeground(QColor(255, 255, 255))
            else:
                status_item.setBackground(QColor(0, 120, 0)) # Xanh lá
                status_item.setForeground(QColor(255, 255, 255))
                
            self.table_segments.setItem(idx, 5, status_item)
            
        self.table_segments.blockSignals(False)
        self.table_segments.resizeRowsToContents()
        self.update_script_stats()

    def update_script_stats(self):
        total_chars = sum(len(seg.get("text", "").strip()) for seg in self.segments_data)
        total_duration = sum(float(seg.get("duration", 0.0) or 0.0) for seg in self.segments_data)
        if total_duration > 0:
            self.lbl_script_stats.setText(f"{len(self.segments_data)} câu đọc • {total_chars} ký tự • {total_duration:.1f} giây audio")
        else:
            self.lbl_script_stats.setText(f"{len(self.segments_data)} câu đọc • {total_chars} ký tự")
        
    def on_table_item_changed(self, item):
        if item.column() == 1:
            row = item.row()
            new_text = item.text().strip()
            if row < len(self.segments_data):
                old_text = self.segments_data[row]["text"]
                if old_text != new_text:
                    self.segments_data[row]["text"] = new_text
                    self.segments_data[row]["needs_regen"] = True
                    self.update_script_stats()
                    
                    self.table_segments.blockSignals(True)
                    status_item = self.table_segments.item(row, 5)
                    status_item.setText("Cần tạo lại")
                    status_item.setBackground(QColor(180, 110, 0))
                    status_item.setForeground(QColor(255, 255, 255))
                    self.table_segments.blockSignals(False)
                    
                    self.lbl_warning_notice.setText("⚠️ Trạng thái: Cần tạo lại")
                    self.lbl_warning_notice.setStyleSheet("color: #dfb15b; font-weight: bold;")
                    
    def on_row_voice_changed(self, row, combo):
        if row < len(self.segments_data):
            new_voice = combo.currentData()
            if self.segments_data[row].get("voice") != new_voice:
                self.segments_data[row]["voice"] = new_voice
                self.segments_data[row]["needs_regen"] = True
                
                self.table_segments.blockSignals(True)
                status_item = self.table_segments.item(row, 5)
                if status_item:
                    status_item.setText("Cần tạo lại")
                    status_item.setBackground(QColor(180, 110, 0))
                self.table_segments.blockSignals(False)
                self.lbl_warning_notice.setText("⚠️ Trạng thái: Cần tạo lại")
                self.lbl_warning_notice.setStyleSheet("color: #dfb15b; font-weight: bold;")

    def on_global_voice_changed(self):
        global_voice = self.cb_script_voice.currentData()
        self.table_segments.blockSignals(True)
        for idx in range(self.table_segments.rowCount()):
            widget = self.table_segments.cellWidget(idx, 2)
            if isinstance(widget, QComboBox):
                idx_v = widget.findData(global_voice)
                if idx_v >= 0:
                    widget.setCurrentIndex(idx_v)
            if idx < len(self.segments_data):
                if self.segments_data[idx].get("voice") != global_voice:
                    self.segments_data[idx]["voice"] = global_voice
                    self.segments_data[idx]["needs_regen"] = True
                    status_item = self.table_segments.item(idx, 5)
                    if status_item:
                        status_item.setText("Cần tạo lại")
                        status_item.setBackground(QColor(180, 110, 0))
        self.table_segments.blockSignals(False)
        self.lbl_warning_notice.setText("⚠️ Trạng thái: Cần tạo lại")
        self.lbl_warning_notice.setStyleSheet("color: #dfb15b; font-weight: bold;")

    def show_table_context_menu(self, pos):
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        
        action_preview = menu.addAction("🔊 Nghe thử dòng này")
        action_insert = menu.addAction("➕ Chèn dòng trống phía dưới")
        action_delete = menu.addAction("❌ Xóa dòng này")
        menu.addSeparator()
        action_move_up = menu.addAction("⬆ Di chuyển lên")
        action_move_down = menu.addAction("⬇ Di chuyển xuống")
        
        action = menu.exec(self.table_segments.mapToGlobal(pos))
        if not action:
            return
            
        row = self.table_segments.currentRow()
        if row < 0:
            return
            
        if action == action_preview:
            # Gọi trực tiếp qua parent MainWindow
            if self.parent:
                self.parent.preview_segment_tts()
        elif action == action_insert:
            self.insert_script_row(row)
        elif action == action_delete:
            self.delete_script_row(row)
        elif action == action_move_up:
            self.move_script_row(row, -1)
        elif action == action_move_down:
            self.move_script_row(row, 1)

    def insert_script_row(self, row):
        new_seg = {
            "index": row + 2,
            "text": "Dòng mới...",
            "voice": self.cb_script_voice.currentData(),
            "needs_regen": True,
            "audio_path": "",
            "duration": 0.0
        }
        self.segments_data.insert(row + 1, new_seg)
        self.reindex_segments()
        self.populate_segments_table()
        self.lbl_warning_notice.setText("⚠️ Trạng thái: Cần tạo lại")
        self.lbl_warning_notice.setStyleSheet("color: #dfb15b; font-weight: bold;")
        
    def delete_script_row(self, row):
        if row < len(self.segments_data):
            self.segments_data.pop(row)
            self.reindex_segments()
            self.populate_segments_table()
            self.lbl_warning_notice.setText("⚠️ Trạng thái: Cần tạo lại")
            self.lbl_warning_notice.setStyleSheet("color: #dfb15b; font-weight: bold;")
            
    def move_script_row(self, row, direction):
        target = row + direction
        if 0 <= target < len(self.segments_data):
            self.segments_data[row], self.segments_data[target] = self.segments_data[target], self.segments_data[row]
            self.reindex_segments()
            self.populate_segments_table()
            self.table_segments.setCurrentCell(target, 1)
            self.lbl_warning_notice.setText("⚠️ Trạng thái: Cần tạo lại")
            self.lbl_warning_notice.setStyleSheet("color: #dfb15b; font-weight: bold;")
            
    def reindex_segments(self):
        for idx, seg in enumerate(self.segments_data):
            seg["index"] = idx + 1
        self.update_script_stats()
            
    def show_script_history_dialog(self):
        dialog = ScriptHistoryDialog(self)
        dialog.exec()
        
    def load_script_from_history(self, entry):
        self.txt_script.setPlainText(entry.get("script", ""))
        self.split_script_segment()
        
    def generate_batch_tts(self):
        to_generate = [seg for seg in self.segments_data if seg.get("needs_regen", True)]
        if not to_generate:
            QMessageBox.information(self, "Thông báo", "Tất cả các dòng đều đã sẵn sàng, không cần tạo lại.")
            return
            
        voice = self.cb_script_voice.currentData()
        
        speed_val = self.slider_speed.value()
        rate_str = f"{speed_val:+}%"
        pitch_str = "+0Hz"
        
        temp_dir = os.path.abspath(os.path.join(".", "temp_dub", "script_integration"))
        os.makedirs(temp_dir, exist_ok=True)
        
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(to_generate))
        self.progress_bar.setVisible(True)
        self.btn_generate_all.setEnabled(False)
        self.lbl_warning_notice.setText("Đang tạo...")
        self.lbl_warning_notice.setStyleSheet("color: #dfb15b; font-weight: bold;")
        
        self.batch_worker = BatchTTSWorker(to_generate, voice, temp_dir, rate_str, pitch_str)
        self.batch_worker.progress.connect(self.on_batch_progress)
        self.batch_worker.finished.connect(self.on_batch_finished)
        self.batch_worker.start()
        
    def on_batch_progress(self, row_idx, status):
        self.table_segments.blockSignals(True)
        status_item = self.table_segments.item(row_idx, 5)
        status_item.setText(status)
        if status == "Đang tạo...":
            status_item.setBackground(QColor(100, 100, 100))
        elif status == "Đã sẵn sàng":
            status_item.setBackground(QColor(0, 120, 0))
            status_item.setForeground(QColor(255, 255, 255))
            self.progress_bar.setValue(self.progress_bar.value() + 1)
        elif status == "Lỗi mạng...":
            status_item.setBackground(QColor(180, 0, 0))
            status_item.setForeground(QColor(255, 255, 255))
        self.table_segments.blockSignals(False)
        
    def on_batch_finished(self, results):
        self.btn_generate_all.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        has_error = False
        
        for row_idx, file_path, duration in results:
            if row_idx < len(self.segments_data):
                if file_path:
                    self.segments_data[row_idx]['audio_path'] = file_path
                    self.segments_data[row_idx]['duration'] = duration
                    self.segments_data[row_idx]['needs_regen'] = False
                else:
                    self.segments_data[row_idx]['needs_regen'] = True
                    has_error = True
                    
        self.populate_segments_table()
        self.update_script_stats()
        
        if has_error:
            self.lbl_warning_notice.setText("⚠️ Lỗi mạng...")
            self.lbl_warning_notice.setStyleSheet("color: #ff5555; font-weight: bold;")
            QMessageBox.warning(self, "Hoàn tất tạo TTS", "Đã hoàn thành tạo giọng nhưng một số đoạn gặp lỗi kết nối.")
        else:
            self.lbl_warning_notice.setText("Đã sẵn sàng")
            self.lbl_warning_notice.setStyleSheet("color: #7fbeb2; font-weight: bold;")
            QMessageBox.information(self, "Thành công", "Đã tạo giọng nói thành công cho toàn bộ kịch bản!")
            
    def browse_out_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục đầu ra")
        if dir_path:
            self.txt_out_dir.setText(dir_path)
            
    def export_final_results(self):
        valid_segs = [seg for seg in self.segments_data if seg.get("audio_path") and os.path.exists(seg["audio_path"])]
        if not valid_segs:
            QMessageBox.warning(self, "Lỗi xuất", "Không có file âm thanh nào hợp lệ để ghép nối. Hãy chạy Tạo giọng nói hàng loạt trước.")
            return
            
        out_dir = self.txt_out_dir.text().strip()
        os.makedirs(out_dir, exist_ok=True)
        
        silence_sec = self.spin_silence.value()
        silence_ms = int(silence_sec * 1000)
        
        combined = AudioSegment.empty()
        silence_segment = AudioSegment.silent(duration=silence_ms)
        
        for seg in valid_segs:
            try:
                sound = AudioSegment.from_file(seg["audio_path"])
                if len(combined) > 0:
                    combined += silence_segment
                combined += sound
            except Exception as e:
                print(f"Error reading segment audio: {e}")
                
        if len(combined) == 0:
            QMessageBox.warning(self, "Lỗi", "Không thể ghép nối file âm thanh nào.")
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_audio_path = os.path.join(out_dir, f"final_voice_{timestamp}.wav")
        final_srt_path = os.path.join(out_dir, f"final_voice_{timestamp}.srt")
        
        try:
            combined.export(final_audio_path, format="wav")
            
            export_srt_with_silence(
                segments=self.segments_data,
                output_path=final_srt_path,
                default_duration=2.5,
                silence_between=silence_sec
            )
            
            QMessageBox.information(
                self, "Xuất kết quả thành công",
                f"Đã xuất thành công kết quả kịch bản!\n\n🔊 Audio: {final_audio_path}\n📝 Subtitle SRT: {final_srt_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Lỗi xuất", f"Không thể lưu kết quả xuất file: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
