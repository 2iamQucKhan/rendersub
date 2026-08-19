import os
import cv2
import numpy as np
import threading

class PaddleOCRWrapper:
    """
    Wrapper chuyên nghiệp cho PaddleOCR với hỗ trợ đa ngôn ngữ (Tiếng Việt, Tiếng Trung, Tiếng Anh...).
    Tự động fallback an toàn sang EasyOCR nếu chưa cài thư viện paddleocr.
    """
    
    # Map mã ngôn ngữ tiêu chuẩn của PaddleOCR
    LANG_MAP = {
        'vi': 'vi',           # Tiếng Việt
        'vietnamese': 'vi',
        'en': 'en',           # Tiếng Anh
        'english': 'en',
        'zh': 'ch',           # Tiếng Trung (Giản thể & Phồn thể)
        'chinese': 'ch',
        'ch': 'ch',
        'zh_sim': 'ch',
        'zh_tra': 'ch',
        'ja': 'japan',        # Tiếng Nhật
        'japanese': 'japan',
        'ko': 'korean',       # Tiếng Hàn
        'korean': 'korean',
        'fr': 'french',       # Tiếng Pháp
        'french': 'french',
        'de': 'german',       # Tiếng Đức
        'german': 'german',
        'es': 'spanish',      # Tiếng Tây Ban Nha
        'spanish': 'spanish',
        'ru': 'cyrillic',     # Tiếng Nga
        'russian': 'cyrillic'
    }
    
    def __init__(self, lang='vi'):
        self.lang = lang
        self.ocr = None
        self._available = None
        self._lock = threading.Lock()
        self._init_ocr()
    
    def is_available(self):
        if self._available is None:
            try:
                import paddleocr
                self._available = True
            except Exception:
                self._available = False
        return self._available
    
    def _init_ocr(self):
        """Khởi tạo PaddleOCR với ngôn ngữ đã chọn (nếu có thư viện)."""
        if not self.is_available():
            return
        if self.ocr is None:
            with self._lock:
                if self.ocr is None:
                    try:
                        from paddleocr import PaddleOCR
                        lang_code = self.LANG_MAP.get(self.lang.lower(), 'ch')
                        self.ocr = PaddleOCR(
                            use_angle_cls=True,
                            lang=lang_code,
                            det_db_thresh=0.3,
                            det_db_box_thresh=0.5,
                            show_log=False
                        )
                    except Exception as e:
                        print(f"⚠️ [PaddleOCR] Lỗi khởi tạo: {e}. Đang chuyển sang chế độ EasyOCR fallback...")
                        self._available = False
                        self.ocr = None
    
    def set_lang(self, lang):
        """Thay đổi ngôn ngữ nhận dạng."""
        if lang != self.lang:
            self.lang = lang
            self.ocr = None
            self._init_ocr()
    
    def ocr_frame(self, frame, bbox):
        """OCR trên frame với vùng bbox và trả về chuỗi văn bản nhận diện được."""
        if frame is None or bbox is None:
            return None
        
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x1 = max(0, min(x, fw - 1))
        y1 = max(0, min(y, fh - 1))
        x2 = max(x1 + 1, min(x + w, fw))
        y2 = max(y1 + 1, min(y + h, fh))
        
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return None
        
        if not self.is_available() or self.ocr is None:
            # Fallback sang EasyOCR
            try:
                import transcriber
                reader = transcriber.get_easyocr_reader(['ch_sim', 'en'] if self.lang in ['zh', 'ch'] else ['vi', 'en'])
                res = reader.readtext(cropped)
                texts = [r[1].strip() for r in res if len(r) >= 2 and r[1].strip()]
                return ' '.join(texts) if texts else None
            except Exception:
                return None
        
        try:
            result = self.ocr.ocr(cropped, cls=True)
            if result and result[0]:
                texts = []
                for line in result[0]:
                    if line and len(line) >= 2:
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text, confidence = text_info[0], float(text_info[1])
                        else:
                            text, confidence = str(text_info), 1.0
                        if confidence > 0.45 and text and text.strip():
                            texts.append(text.strip())
                if texts:
                    return ' '.join(texts)
        except Exception as e:
            print(f"⚠️ [PaddleOCR] Lỗi OCR frame: {e}")
        
        return None
    
    def ocr_frame_with_boxes(self, frame, bbox):
        """OCR và trả về cả text + bounding boxes theo tọa độ frame gốc."""
        if frame is None or bbox is None:
            return []
        
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x1 = max(0, min(x, fw - 1))
        y1 = max(0, min(y, fh - 1))
        x2 = max(x1 + 1, min(x + w, fw))
        y2 = max(y1 + 1, min(y + h, fh))
        
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return []
        
        if not self.is_available() or self.ocr is None:
            try:
                import transcriber
                reader = transcriber.get_easyocr_reader(['ch_sim', 'en'] if self.lang in ['zh', 'ch'] else ['vi', 'en'])
                res = reader.readtext(cropped)
                items = []
                for r in res:
                    box, text, conf = r
                    box_orig = [[pt[0] + x1, pt[1] + y1] for pt in box]
                    items.append({'text': text, 'confidence': conf, 'box': box_orig})
                return items
            except Exception:
                return []
        
        try:
            result = self.ocr.ocr(cropped, cls=True)
            if result and result[0]:
                items = []
                for line in result[0]:
                    if line and len(line) >= 2:
                        box = line[0]
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text, confidence = text_info[0], float(text_info[1])
                        else:
                            text, confidence = str(text_info), 1.0
                        if confidence > 0.45 and text and text.strip():
                            box_orig = [[pt[0] + x1, pt[1] + y1] for pt in box]
                            items.append({
                                'text': text.strip(),
                                'confidence': confidence,
                                'box': box_orig
                            })
                return items
        except Exception as e:
            print(f"⚠️ [PaddleOCR] Lỗi: {e}")
        
        return []

    def readtext(self, image_np, min_confidence=0.45):
        """Định dạng tương thích 100% với EasyOCR: list of (box, text, conf)."""
        if not self.is_available() or self.ocr is None:
            import transcriber
            reader = transcriber.get_easyocr_reader(['ch_sim', 'en'] if self.lang in ['zh', 'ch'] else ['vi', 'en'])
            return reader.readtext(image_np)

        try:
            result = self.ocr.ocr(image_np, cls=True)
            formatted = []
            if result and result[0]:
                for line in result[0]:
                    box = line[0]
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        txt, conf = text_info[0], float(text_info[1])
                    else:
                        txt, conf = str(text_info), 1.0
                    if conf >= min_confidence and txt and str(txt).strip():
                        formatted.append((box, str(txt).strip(), conf))
            return formatted
        except Exception as e:
            print(f"⚠️ [PaddleOCR] Lỗi nhận diện: {e}. Fallback sang EasyOCR...")
            import transcriber
            reader = transcriber.get_easyocr_reader(['ch_sim', 'en'] if self.lang in ['zh', 'ch'] else ['vi', 'en'])
            return reader.readtext(image_np)

# Singleton instances pool theo ngôn ngữ
_ocr_instances = {}
_instances_lock = threading.Lock()

def get_paddle_ocr(lang='vi'):
    """Lấy instance PaddleOCR (Singleton thread-safe theo ngôn ngữ)."""
    lang_key = str(lang).lower().strip()
    if lang_key not in _ocr_instances:
        with _instances_lock:
            if lang_key not in _ocr_instances:
                _ocr_instances[lang_key] = PaddleOCRWrapper(lang=lang_key)
    return _ocr_instances[lang_key]
