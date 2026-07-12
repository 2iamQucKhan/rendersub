import os
import sys
import re
import requests
import json
import threading
from pathlib import Path
from deep_translator import GoogleTranslator, MyMemoryTranslator

# ----------------- HỆ THỐNG CACHE DỊCH THUẬT TOÀN CỤC -----------------
class TranslationCacheManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(TranslationCacheManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.lock = threading.Lock()
        base_dir = Path(__file__).resolve().parent
        self.cache_dir = base_dir / "Data" / "cache"
        self.cache_path = self.cache_dir / "global_translation_cache.json"
        self.cache_data = {}
        self.load_cache()
        self._initialized = True

    def load_cache(self):
        with self.lock:
            if self.cache_path.exists():
                try:
                    with self.cache_path.open('r', encoding='utf-8-sig') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            self.cache_data = data
                except Exception as e:
                    print(f"Lỗi tải cache dịch thuật toàn cục: {e}")

    def save_cache(self):
        with self.lock:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
                with tmp_path.open('w', encoding='utf-8') as f:
                    json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
                tmp_path.replace(self.cache_path)
            except Exception as e:
                print(f"Lỗi lưu cache dịch thuật toàn cục: {e}")

    def get(self, text, engine, src_lang, tgt_lang):
        key = f"{src_lang}->{tgt_lang}:{engine}:{text}"
        with self.lock:
            val = self.cache_data.get(key)
            if val is not None:
                if not val.strip() or val == text:
                    return None
            return val

    def set(self, text, engine, src_lang, tgt_lang, translated_text):
        key = f"{src_lang}->{tgt_lang}:{engine}:{text}"
        with self.lock:
            self.cache_data[key] = translated_text
        self.save_cache()

global_translation_cache = TranslationCacheManager()


# ----------------- HỆ THỐNG DỊCH VIETPHRASE (QUICK TRANSLATOR) -----------------
class VietPhraseTranslator:
    def __init__(self, vp_path=None, names_path=None, pa_path=None):
        self.vp_path = vp_path
        self.names_path = names_path
        self.pa_path = pa_path
        self.dict_data = {}
        self.shortcuts = {}
        self.loaded = False
        
    def load(self, progress_callback=None):
        if self.loaded:
            return
            
        if progress_callback:
            progress_callback("Đang nạp toàn bộ từ điển Hán Việt, VietPhrase, Lạc Việt và Babylon (vui lòng đợi)...")
            
        # Tìm đường dẫn mặc định trong thư mục Data của dự án
        def get_default_path(filename):
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))
            return os.path.join(base_path, "Data", filename)
            
        vp = self.vp_path or get_default_path("VietPhrase.txt")
        names = self.names_path or get_default_path("Names.txt")
        pa = self.pa_path or get_default_path("ChinesePhienAmWords.txt")
        
        # Lấy thư mục mẹ của VietPhrase để tự động load các tệp cùng thư mục
        parent_dir = os.path.dirname(vp) if vp else ""
        
        tc = os.path.join(parent_dir, "ThieuChuu.txt") if parent_dir else get_default_path("ThieuChuu.txt")
        baby = os.path.join(parent_dir, "Babylon.txt") if parent_dir else get_default_path("Babylon.txt")
        lv = os.path.join(parent_dir, "LacViet.txt") if parent_dir else get_default_path("LacViet.txt")
        shortcuts = os.path.join(parent_dir, "Shortcuts.txt") if parent_dir else get_default_path("Shortcuts.txt")
        modern = os.path.join(parent_dir, "ChineseModern.txt") if parent_dir else get_default_path("ChineseModern.txt")
        
        # Nạp theo thứ tự ngược độ ưu tiên:
        # Babylon -> LacViet -> ThieuChuu -> PhienAm -> VietPhrase -> ChineseModern -> Names
        
        # 1. Nạp Babylon (Từ điển Anh - Trung)
        if os.path.exists(baby):
            try:
                with open(baby, 'r', encoding='utf-8-sig') as f:
                    for line in f:
                        line = line.strip()
                        if not line or '=' not in line:
                            continue
                        parts = line.split('=', 1)
                        key = parts[0].strip()
                        val = parts[1].strip()
                        if ';' in val:
                            val = val.split(';')[0].strip()
                        self.dict_data[key] = val
            except Exception as e:
                print(f"Lỗi nạp Babylon: {e}")
                
        # 2. Nạp Lạc Việt (Từ điển Trung - Việt chi tiết)
        if os.path.exists(lv):
            try:
                with open(lv, 'r', encoding='utf-8-sig') as f:
                    for line in f:
                        line = line.strip()
                        if not line or '=' not in line:
                            continue
                        parts = line.split('=', 1)
                        key = parts[0].strip()
                        val = parts[1].strip()
                        # Làm sạch định dạng của Lạc Việt (lọc pinyin ✚[...] và tách nghĩa đầu tiên)
                        clean_val = re.sub(r'✚\[.*?\]', '', val).strip()
                        clean_val = re.sub(r'\\n\\t\d+\.', ';', clean_val)
                        clean_val = clean_val.replace('\\n', ';').replace('\\t', ';')
                        sub_parts = [p.strip() for p in re.split(r'[;,]', clean_val) if p.strip()]
                        if sub_parts:
                            self.dict_data[key] = sub_parts[0]
            except Exception as e:
                print(f"Lỗi nạp LacViet: {e}")
                
        # 3. Nạp Thiều Chửu (Giải nghĩa từ)
        if os.path.exists(tc):
            try:
                with open(tc, 'r', encoding='utf-8-sig') as f:
                    for line in f:
                        line = line.strip()
                        if not line or '=' not in line:
                            continue
                        parts = line.split('=', 1)
                        key = parts[0].strip()
                        val = parts[1].strip()
                        # Lấy âm đọc Hán Việt đầu tiên trước ngoặc vuông pinyin [..]
                        clean_val = val.split('\\n')[0].split('\n')[0].strip()
                        if '[' in clean_val:
                            clean_val = clean_val.split('[')[0].strip()
                        sub_parts = [p.strip() for p in clean_val.split(',') if p.strip()]
                        if sub_parts:
                            self.dict_data[key] = sub_parts[0]
            except Exception as e:
                print(f"Lỗi nạp ThieuChuu: {e}")
                
        # 4. Nạp Phiên Âm tự động (Ưu tiên đè lên các ký tự đơn lẻ từ điển lớn)
        if os.path.exists(pa):
            try:
                with open(pa, 'r', encoding='utf-16') as f:
                    for line in f:
                        line = line.strip()
                        if not line or '=' not in line:
                            continue
                        parts = line.split('=', 1)
                        key = parts[0].strip()
                        val = parts[1].strip()
                        if '|' in val:
                            val = val.split('|')[0]
                        self.dict_data[key] = val
            except Exception as e:
                print(f"Lỗi nạp PhienAm: {e}")
                
        # 5. Nạp VietPhrase chính
        if os.path.exists(vp):
            try:
                with open(vp, 'r', encoding='utf-8-sig') as f:
                    for line in f:
                        line = line.strip()
                        if not line or '=' not in line:
                            continue
                        parts = line.split('=', 1)
                        key = parts[0].strip()
                        val = parts[1].strip()
                        if '/' in val:
                            val = val.split('/')[-1].strip()
                        elif '|' in val:
                            val = val.split('|')[-1].strip()
                        self.dict_data[key] = val
            except Exception as e:
                print(f"Lỗi nạp VietPhrase: {e}")
                
        # 5.5. Nạp từ điển tiếng Trung hiện đại / từ lóng (ChineseModern.txt)
        if os.path.exists(modern):
            try:
                with open(modern, 'r', encoding='utf-8-sig') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        parts = line.split('=', 1)
                        key = parts[0].strip()
                        val = parts[1].strip()
                        if '/' in val:
                            val = val.split('/')[-1].strip()
                        elif '|' in val:
                            val = val.split('|')[-1].strip()
                        self.dict_data[key] = val
            except Exception as e:
                print(f"Lỗi nạp ChineseModern: {e}")
                
        # 6. Nạp Names (Cao nhất)
        if os.path.exists(names):
            try:
                with open(names, 'r', encoding='utf-16') as f:
                    for line in f:
                        line = line.strip()
                        if not line or '=' not in line:
                            continue
                        parts = line.split('=', 1)
                        key = parts[0].strip()
                        val = parts[1].strip()
                        if '/' in val:
                            val = val.split('/')[-1].strip()
                        elif '|' in val:
                            val = val.split('|')[-1].strip()
                        self.dict_data[key] = val
            except Exception as e:
                print(f"Lỗi nạp Names: {e}")
                
        # 7. Nạp các Shortcuts viết tắt
        if os.path.exists(shortcuts):
            try:
                with open(shortcuts, 'r', encoding='utf-8-sig') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        parts = line.split('=', 1)
                        self.shortcuts[parts[0].strip()] = parts[1].strip()
            except Exception as e:
                print(f"Lỗi nạp Shortcuts: {e}")
                
        # 8. Nạp danh sách từ chửi thề kiểm duyệt (Censor.txt)
        censor_path = os.path.join(parent_dir, "Censor.txt") if parent_dir else get_default_path("Censor.txt")
        self.censor_words = []
        if os.path.exists(censor_path):
            try:
                with open(censor_path, 'r', encoding='utf-8-sig') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        self.censor_words.append(line)
            except Exception as e:
                print(f"Lỗi nạp Censor list: {e}")
                
        # Fallback khẩn cấp nếu hoàn toàn rỗng
        if not self.dict_data:
            fallback_dict = {
                "你好": "xin chào", "谢谢": "cảm ơn", "再见": "tạm biệt", "什么": "cái gì",
                "越南": "Việt Nam", "中国": "Trung Quốc", "我": "ta/tôi", "你": "ngươi/bạn",
                "他": "hắn/anh ấy", "她": "nàng/cô ấy", "是": "là", "不": "không",
                "的": "của", "在": "tại/ở", "了": "rồi", "要": "muốn/cần", "去": "đi"
            }
            self.dict_data.update(fallback_dict)
            
        self.loaded = True
        if progress_callback:
            progress_callback(f"Đã nạp {len(self.dict_data)} từ khóa từ điển và {len(self.shortcuts)} tắt!")
            
    def translate(self, text, progress_callback=None):
        self.load(progress_callback)
        if not self.dict_data:
            return text
            
        result = []
        i = 0
        n = len(text)
        max_len = 50
        
        while i < n:
            char = text[i]
            if char.isspace() or char in ",.!?()[]{}:;\"'<>~@#$%^&*_-+=/\\|":
                result.append(char)
                i += 1
                continue
                
            # Gom nhóm ký tự Latin để không bị tách rời từ tiếng Anh viết tắt
            if re.match(r'[a-zA-Z0-9]', char):
                latin_word = ""
                while i < n and re.match(r'[a-zA-Z0-9]', text[i]):
                    latin_word += text[i]
                    i += 1
                result.append(latin_word)
                continue
                
            matched = False
            for length in range(min(max_len, n - i), 0, -1):
                sub = text[i:i+length]
                if sub in self.dict_data:
                    result.append(self.dict_data[sub])
                    i += length
                    matched = True
                    break
            if not matched:
                result.append(char)
                i += 1
                
        raw_res = " ".join(result)
        raw_res = re.sub(r'\s+([,.\!?])', r'\1', raw_res)
        raw_res = re.sub(r'\s+', ' ', raw_res).strip()
        
        # Áp dụng các quy tắc viết tắt (Shortcuts)
        for abbrev, full_word in self.shortcuts.items():
            pattern = re.compile(rf'\b{re.escape(abbrev)}\b', re.IGNORECASE)
            raw_res = pattern.sub(full_word, raw_res)
            
        # Áp dụng kiểm duyệt từ chửi thề (Censor)
        if hasattr(self, 'censor_words') and self.censor_words:
            sorted_censor = sorted(self.censor_words, key=len, reverse=True)
            for word in sorted_censor:
                if re.search(r'[\u4e00-\u9fff]', word):
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                else:
                    pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
                raw_res = pattern.sub('***', raw_res)
                
        return raw_res

# Hộp thoại đọc base_url cho 9router
def get_custom_provider_base_url():
    default_url = "http://localhost:20127/v1"
    local_appdata = os.environ.get("LOCALAPPDATA")
    config_path = Path(local_appdata) / "hermes" / "config.yaml" if local_appdata else Path.home() / "AppData" / "Local" / "hermes" / "config.yaml"
    if config_path.exists():
        try:
            with config_path.open('r', encoding='utf-8') as f:
                for line in f:
                    if 'base_url:' in line:
                        parts = line.split('base_url:', 1)
                        if len(parts) > 1:
                            url = parts[1].strip().strip('"').strip("'")
                            if url:
                                return url
        except Exception as e:
            print(f"Khong the doc config provider tuy chinh: {e}")
    return default_url

_capitalization_names = None

def load_names_for_capitalization():
    global _capitalization_names
    if _capitalization_names is not None:
        return _capitalization_names
    _capitalization_names = []
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    names_path = os.path.join(base_path, "Data", "Names.txt")
    if os.path.exists(names_path):
        try:
            with open(names_path, 'r', encoding='utf-16') as f:
                for line in f:
                    line = line.strip()
                    if not line or '=' not in line:
                        continue
                    parts = line.split('=', 1)
                    val = parts[1].strip()
                    if '/' in val:
                        val = val.split('/')[-1].strip()
                    elif '|' in val:
                        val = val.split('|')[-1].strip()
                    if val and val not in _capitalization_names:
                        _capitalization_names.append(val)
        except Exception as e:
            print(f"Lỗi nạp Names để viết hoa: {e}")
    _capitalization_names.sort(key=len, reverse=True)
    return _capitalization_names

def capitalize_proper_names(text):
    if not text:
        return text
    names = load_names_for_capitalization()
    for name in names:
        pattern = re.compile(rf'\b{re.escape(name)}\b', re.IGNORECASE)
        text = pattern.sub(name, text)
    return text

# ----------------- HÀM BIÊN DỊCH PHÂN ĐOẠN PHỤ ĐỀ CHÍNH -----------------
def translate_segments(segments, source_lang='auto', target_lang='vi', engine='Google Translate', api_key=None, progress_callback=None, ollama_model='qwen2.5', vp_dict_paths=None):
    """
    Dịch danh sách các câu phụ đề [{'start', 'end', 'text'}] sang ngôn ngữ đích.
    Hỗ trợ:
      - Google Translate / MyMemory (Miễn phí)
      - Quick Translator (VietPhrase) [Cục bộ, miễn phí, chuyên tiếng Trung]
      - Ollama Local (Mô hình AI cục bộ không cần Key)
      - Gemini 1.5 Flash / Gemini 1.5 Pro (Cần Gemini API Key)
      - Groq Llama 3.1 (70B) (Cần Groq API Key)
      - DeepL Translate (Cần DeepL API Key)
    """
    if not segments:
        return []
        
    lang_map = {
        'Vietnamese': 'vi',
        'English': 'en',
        'Chinese': 'zh-CN',
        'Japanese': 'ja',
        'Korean': 'ko',
        'French': 'fr',
        'German': 'de',
        'Spanish': 'es',
        'auto': 'auto'
    }
    
    src = lang_map.get(source_lang, source_lang)
    tgt = lang_map.get(target_lang, target_lang)
    
    texts = [seg['text'] for seg in segments]
    
    # --- KIỂM TRA CACHE DỊCH THUẬT TOÀN CỤC ---
    uncached_indices = []
    uncached_texts = []
    cached_results = [None] * len(texts)
    
    for idx, text in enumerate(texts):
        cached_val = global_translation_cache.get(text, engine, src, tgt)
        if cached_val is not None:
            cached_results[idx] = cached_val
        else:
            uncached_indices.append(idx)
            uncached_texts.append(text)
            
    # Nếu toàn bộ đã có trong cache, dựng lại và trả về ngay lập tức
    if not uncached_texts:
        if progress_callback:
            progress_callback(f"Đã nạp toàn bộ {len(texts)} dòng dịch từ Cache toàn cục!")
        translated_segments = []
        for idx, seg in enumerate(segments):
            trans_text = cached_results[idx]
            has_override = seg.get('manual_override', False)
            final_text = seg.get('text', trans_text) if has_override else trans_text
            new_seg = {
                'start': seg['start'],
                'end': seg['end'],
                'raw_text': trans_text,
                'text': final_text,
                'orig_text': seg.get('orig_text', seg.get('text', '')),
                'manual_override': has_override
            }
            if 'bbox' in seg:
                new_seg['bbox'] = seg['bbox']
            if 'confidence' in seg:
                new_seg['confidence'] = seg['confidence']
            translated_segments.append(new_seg)
        return translated_segments

    # Trỏ texts sang danh sách chưa dịch để các engine chạy tiếp
    original_texts = list(texts)
    texts = uncached_texts
    translated_texts = []
    
    if progress_callback:
        progress_callback(f"Đang chuẩn bị dịch {len(texts)} dòng bằng {engine} (đã nạp {len(original_texts) - len(texts)} dòng từ cache)...")
        
    # --- 1. ENGINE: QUICK TRANSLATOR (VIETPHRASE) ---
    if "Quick Translator" in engine:
        vp_paths = vp_dict_paths or {}
        translator = VietPhraseTranslator(
            vp_path=vp_paths.get('vp_path'),
            names_path=vp_paths.get('names_path'),
            pa_path=vp_paths.get('pa_path')
        )
        # Nạp từ điển trước để hiển thị trạng thái
        translator.load(progress_callback)
        
        for idx, text in enumerate(texts):
            translated_texts.append(translator.translate(text))
            if progress_callback and (idx + 1) % 10 == 0:
                percent = int(((idx + 1) / len(texts)) * 100)
                progress_callback(f"Đang dịch bằng VietPhrase... {percent}% ({idx+1}/{len(texts)} câu)")
                
    # --- 2. ENGINE: OLLAMA LOCAL (AI CỤC BỘ KHÔNG CẦN KEY) ---
    elif engine == 'Ollama Local':
        url = "http://localhost:11434/api/chat"
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            prompt = (
                f"Bạn là một biên dịch viên phim và video chuyên nghiệp, am hiểu ngôn ngữ giao tiếp tự nhiên.\n"
                f"Nhiệm vụ: Dịch các câu thoại video sau đây từ ngôn ngữ '{src}' sang ngôn ngữ '{tgt}'.\n\n"
                f"YÊU CẦU QUAN TRỌNG:\n"
                f"1. Dịch THOÁT Ý, tự nhiên và trôi chảy theo văn phong hội thoại giao tiếp hàng ngày của người Việt. Tuyệt đối tránh dịch word-by-word (sát nghĩa đen từng từ) gây cảm giác gượng ép, máy móc.\n"
                f"2. Giữ nguyên cấu trúc dòng: Dịch đúng thứ tự từng dòng, CHỈ xuất ra kết quả dịch, mỗi dòng tương ứng với một dòng gốc. Không tự thêm số thứ tự, chú thích hay giải thích nào.\n\n"
                f"Nội dung cần dịch:\n" + "\n".join(batch)
            )
            payload = {
                "model": ollama_model if ollama_model else "qwen2.5",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
            try:
                res = requests.post(url, json=payload, timeout=60)
                res.raise_for_status()
                res_text = res.json()["message"]["content"].strip()
                
                # Loại bỏ định dạng mã code block nếu có
                if res_text.startswith("```"):
                    lines_split = res_text.splitlines()
                    if lines_split[0].startswith("```"):
                        lines_split = lines_split[1:]
                    if lines_split[-1].startswith("```"):
                        lines_split = lines_split[:-1]
                    res_text = "\n".join(lines_split).strip()
                    
                lines = [line.strip() for line in res_text.splitlines() if line.strip()]
                
                if len(lines) == len(batch):
                    translated_texts.extend(lines)
                else:
                    # Dự phòng sang Google Dịch
                    gt = GoogleTranslator(source=src, target=tgt)
                    res_b = gt.translate_batch(batch)
                    if res_b and isinstance(res_b, list):
                        translated_texts.extend(res_b)
                    else:
                        translated_texts.extend(batch)
            except Exception as e:
                print(f"Lỗi kết nối Ollama: {e}. Chuyển sang Google Dịch dự phòng.")
                try:
                    gt = GoogleTranslator(source=src, target=tgt)
                    res_b = gt.translate_batch(batch)
                    if res_b and isinstance(res_b, list):
                        translated_texts.extend(res_b)
                    else:
                        translated_texts.extend(batch)
                except Exception:
                    translated_texts.extend(batch)
                    
            if progress_callback:
                percent = int(((i + len(batch)) / len(texts)) * 100)
                progress_callback(f"Dịch bằng Ollama... {percent}% (Đã dịch {min(i + batch_size, len(texts))}/{len(texts)} câu)")

    # --- 3. ENGINE: GEMINI + QUICK TRANSLATOR (VIETPHRASE) HYBRID ---
    elif "Hybrid" in engine and api_key:
        # Khởi tạo và nạp từ điển VietPhrase
        vp_paths = vp_dict_paths or {}
        vp_translator = VietPhraseTranslator(
            vp_path=vp_paths.get('vp_path'),
            names_path=vp_paths.get('names_path'),
            pa_path=vp_paths.get('pa_path')
        )
        vp_translator.load(progress_callback)
        
        # Xác định model name tương ứng
        model_name = "gemini-1.5-flash" # Mặc định
        if "1.5 Pro" in engine:
            model_name = "gemini-1.5-pro"
        elif "2.0 Flash" in engine:
            model_name = "gemini-2.0-flash"
        elif "1.5 Flash" in engine:
            model_name = "gemini-1.5-flash"
            
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Dịch VietPhrase thô trước cho toàn bộ batch
            batch_vp = []
            for t in batch:
                batch_vp.append(vp_translator.translate(t))
                
            # Cố gắng dịch theo lô (batch) để tăng tốc độ
            lines_prompt = []
            for idx, (orig, vp_res) in enumerate(zip(batch, batch_vp), 1):
                lines_prompt.append(f"Dòng {idx}: {orig} <Gợi ý VietPhrase: {vp_res}>")
                
            prompt = (
                "Bạn là một dịch giả phim chuyên nghiệp dịch từ tiếng Trung sang tiếng Việt.\n"
                "Nhiệm vụ: Hãy dịch các câu thoại tiếng Trung sau sang tiếng Việt cực kỳ tự nhiên, trôi chảy, đúng văn phong phim ảnh, nói chuyện hàng ngày.\n"
                "Chúng tôi cung cấp gợi ý VietPhrase trong dấu <Gợi ý VietPhrase: ...> để bạn biết đúng tên nhân vật, địa danh, hoặc từ Hán Việt cổ trang.\n\n"
                "YÊU CẦU QUAN TRỌNG:\n"
                "1. TUYỆT ĐỐI KHÔNG dịch sát nghĩa đen từng từ (word-by-word) kiểu VietPhrase. Hãy viết lại thành câu nói tự nhiên nhất của người Việt.\n"
                "2. Giữ đúng tên riêng, địa danh, hoặc từ Hán Việt cổ trang từ gợi ý VietPhrase.\n"
                "3. Chỉ trả về kết quả dịch cuối cùng cho từng dòng, mỗi câu trên một dòng mới. Không ghi số thứ tự 'Dòng X:', không thêm giải thích hay chú thích nào.\n\n"
                "Ví dụ mẫu:\n"
                "Dòng 1: 师兄，啊怎么搞那么狼狈 <Gợi ý VietPhrase: Sư huynh, a thế nào làm như vậy chật vật>\n"
                "Kết quả 1: Sư huynh, sao huynh lại thê thảm thế này?\n\n"
                "Dòng 2: 这一世，我不会再让你受苦 <Gợi ý VietPhrase: Đời này, ta sẽ không lại để ngươi chịu khổ>\n"
                "Kết quả 2: Kiếp này, ta sẽ không để nàng phải chịu khổ nữa.\n\n"
                "Nội dung cần dịch:\n" + "\n".join(lines_prompt)
            )
            
            batch_success = False
            batch_results = []
            
            try:
                if api_key.startswith("sk-"):
                    base_url = get_custom_provider_base_url()
                    url = f"{base_url}/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3
                    }
                    res = requests.post(url, json=payload, headers=headers, timeout=60)
                    res.raise_for_status()
                    res_text = res.json()["choices"][0]["message"]["content"].strip()
                else:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    res_text = response.text.strip()
                
                if res_text.startswith("```"):
                    lines_split = res_text.splitlines()
                    if lines_split[0].startswith("```"):
                        lines_split = lines_split[1:]
                    if lines_split[-1].startswith("```"):
                        lines_split = lines_split[:-1]
                    res_text = "\n".join(lines_split).strip()
                    
                lines = [line.strip() for line in res_text.splitlines() if line.strip()]
                
                cleaned_lines = []
                for line in lines:
                    cleaned = re.sub(r'^(?:kết quả|Kết quả|dòng|Dòng|ket qua|Ket qua|dong|Dong|Kết quả dòng|Kết quả dòng|Kết quả Dòng|Kết quả Dòng)\s*\d+\s*:\s*', '', line).strip()
                    cleaned_lines.append(cleaned)
                    
                if len(cleaned_lines) == len(batch):
                    batch_results = cleaned_lines
                    batch_success = True
            except Exception as e:
                print(f"Lỗi dịch lô: {e}")
                
            # Nếu dịch lô thất bại hoặc lệch dòng, chuyển sang dịch từng dòng đơn để đảm bảo 100% dùng Gemini thoát ý
            if not batch_success:
                print(f"Số dòng lệch hoặc lỗi lô ở đoạn {i}-{i+len(batch)}. Đang tự động chuyển sang dịch từng câu đơn...")
                single_results = []
                for idx_s, (orig, vp_res) in enumerate(zip(batch, batch_vp)):
                    line_prompt = (
                        "Bạn là một dịch giả phim chuyên nghiệp dịch từ tiếng Trung sang tiếng Việt.\n"
                        "Hãy dịch câu thoại tiếng Trung sau sang tiếng Việt cực kỳ tự nhiên, trôi chảy, đúng văn phong hội thoại phim ảnh.\n"
                        f"Gốc: {orig}\n"
                        f"Gợi ý VietPhrase: {vp_res}\n\n"
                        "YÊU CẦU:\n"
                        "1. Dịch thoát ý tự nhiên, không dịch từng từ thô cứng.\n"
                        "2. Giữ tên riêng, địa danh hợp lý.\n"
                        "3. Chỉ xuất duy nhất câu dịch tiếng Việt cuối cùng, không thêm giải thích.\n"
                        "Dịch:"
                    )
                    try:
                        if api_key.startswith("sk-"):
                            base_url = get_custom_provider_base_url()
                            url = f"{base_url}/chat/completions"
                            headers = {
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json"
                            }
                            payload = {
                                "model": model_name,
                                "messages": [{"role": "user", "content": line_prompt}],
                                "temperature": 0.3
                            }
                            res = requests.post(url, json=payload, headers=headers, timeout=20)
                            res.raise_for_status()
                            single_res = res.json()["choices"][0]["message"]["content"].strip()
                        else:
                            import google.generativeai as genai
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel(model_name)
                            response = model.generate_content(line_prompt)
                            single_res = response.text.strip()
                        single_results.append(single_res)
                    except Exception as e_s:
                        print(f"Lỗi dịch dòng đơn {i+idx_s}: {e_s}")
                        single_results.append(vp_res) # Cuối cùng mới dùng VietPhrase làm dự phòng
                batch_results = single_results
                
            translated_texts.extend(batch_results)
                
            if progress_callback:
                percent = int(((i + len(batch)) / len(texts)) * 100)
                progress_callback(f"Dịch Hybrid (Gemini + VietPhrase)... {percent}% (Đã dịch {min(i + batch_size, len(texts))}/{len(texts)} câu)")
 
    # --- 4. ENGINE: GEMINI AI (FLASH HOẶC PRO) ---
    elif ("Gemini" in engine or engine == 'Gemini AI') and api_key:
        model_name = "gemini-1.5-pro" if "Pro" in engine else "gemini-1.5-flash"
        
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            prompt = (
                f"Bạn là một biên dịch viên phim và video chuyên nghiệp, am hiểu ngôn ngữ giao tiếp tự nhiên.\n"
                f"Nhiệm vụ: Dịch các câu thoại video sau đây từ ngôn ngữ '{src}' sang ngôn ngữ '{tgt}'.\n\n"
                f"YÊU CẦU QUAN TRỌNG:\n"
                f"1. Dịch THOÁT Ý, tự nhiên và trôi chảy theo văn phong hội thoại giao tiếp hàng ngày của người Việt. Tuyệt đối tránh dịch word-by-word (sát nghĩa đen từng từ) gây cảm giác gượng ép, máy móc.\n"
                f"2. Giữ nguyên cấu trúc dòng: Dịch đúng thứ tự từng dòng, CHỈ xuất ra kết quả dịch, mỗi dòng tương ứng với một dòng gốc. Không tự thêm số thứ tự, chú thích hay giải thích nào.\n\n"
                f"Nội dung cần dịch:\n" + "\n".join(batch)
            )
            try:
                if api_key.startswith("sk-"):
                    base_url = get_custom_provider_base_url()
                    url = f"{base_url}/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3
                    }
                    res = requests.post(url, json=payload, headers=headers, timeout=60)
                    res.raise_for_status()
                    res_text = res.json()["choices"][0]["message"]["content"].strip()
                else:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    res_text = response.text.strip()
                
                if res_text.startswith("```"):
                    lines_split = res_text.splitlines()
                    if lines_split[0].startswith("```"):
                        lines_split = lines_split[1:]
                    if lines_split[-1].startswith("```"):
                        lines_split = lines_split[:-1]
                    res_text = "\n".join(lines_split).strip()
                    
                lines = [line.strip() for line in res_text.splitlines() if line.strip()]
                
                if len(lines) == len(batch):
                    translated_texts.extend(lines)
                else:
                    gt = GoogleTranslator(source=src, target=tgt)
                    res_b = gt.translate_batch(batch)
                    if res_b and isinstance(res_b, list):
                        translated_texts.extend(res_b)
                    else:
                        translated_texts.extend(batch)
            except Exception as e:
                print(f"Lỗi Gemini: {e}")
                try:
                    gt = GoogleTranslator(source=src, target=tgt)
                    res_b = gt.translate_batch(batch)
                    if res_b and isinstance(res_b, list):
                        translated_texts.extend(res_b)
                    else:
                        translated_texts.extend(batch)
                except Exception:
                    translated_texts.extend(batch)
            
            if progress_callback:
                percent = int(((i + len(batch)) / len(texts)) * 100)
                progress_callback(f"Dịch bằng Gemini ({model_name})... {percent}% (Đã dịch {min(i + batch_size, len(texts))}/{len(texts)} câu)")

    # --- 4. ENGINE: GROQ CLOUD AI (LLAMA 3.1 70B) ---
    elif "Groq" in engine and api_key:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            prompt = (
                f"Bạn là một biên dịch viên phim và video chuyên nghiệp, am hiểu ngôn ngữ giao tiếp tự nhiên.\n"
                f"Nhiệm vụ: Dịch các câu thoại video sau đây từ ngôn ngữ '{src}' sang ngôn ngữ '{tgt}'.\n\n"
                f"YÊU CẦU QUAN TRỌNG:\n"
                f"1. Dịch THOÁT Ý, tự nhiên và trôi chảy theo văn phong hội thoại giao tiếp hàng ngày của người Việt. Tuyệt đối tránh dịch word-by-word (sát nghĩa đen từng từ) gây cảm giác gượng ép, máy móc.\n"
                f"2. Giữ nguyên cấu trúc dòng: Dịch đúng thứ tự từng dòng, CHỈ xuất ra kết quả dịch, mỗi dòng tương ứng với một dòng gốc. Không tự thêm số thứ tự, chú thích hay giải thích nào.\n\n"
                f"Nội dung cần dịch:\n" + "\n".join(batch)
            )
            payload = {
                "model": "llama-3.1-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=30)
                res.raise_for_status()
                res_text = res.json()["choices"][0]["message"]["content"].strip()
                
                if res_text.startswith("```"):
                    lines_split = res_text.splitlines()
                    if lines_split[0].startswith("```"):
                        lines_split = lines_split[1:]
                    if lines_split[-1].startswith("```"):
                        lines_split = lines_split[:-1]
                    res_text = "\n".join(lines_split).strip()
                    
                lines = [line.strip() for line in res_text.splitlines() if line.strip()]
                
                if len(lines) == len(batch):
                    translated_texts.extend(lines)
                else:
                    gt = GoogleTranslator(source=src, target=tgt)
                    res_b = gt.translate_batch(batch)
                    if res_b and isinstance(res_b, list):
                        translated_texts.extend(res_b)
                    else:
                        translated_texts.extend(batch)
            except Exception as e:
                print(f"Lỗi Groq: {e}")
                try:
                    gt = GoogleTranslator(source=src, target=tgt)
                    res_b = gt.translate_batch(batch)
                    if res_b and isinstance(res_b, list):
                        translated_texts.extend(res_b)
                    else:
                        translated_texts.extend(batch)
                except Exception:
                    translated_texts.extend(batch)
                    
            if progress_callback:
                percent = int(((i + len(batch)) / len(texts)) * 100)
                progress_callback(f"Dịch bằng Groq (Llama 3.1)... {percent}% (Đã dịch {min(i + batch_size, len(texts))}/{len(texts)} câu)")

    # --- 5. ENGINE: DEEPL TRANSLATE ---
    elif "DeepL" in engine and api_key:
        is_free = api_key.endswith(":fx")
        url = "https://api-free.deepl.com/v2/translate" if is_free else "https://api.deepl.com/v2/translate"
        headers = {
            "Authorization": f"DeepL-Auth-Key {api_key}",
            "Content-Type": "application/json"
        }
        
        # DeepL hỗ trợ dịch danh sách mảng văn bản
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            payload = {
                "text": batch,
                "target_lang": tgt.upper() if tgt.upper() != 'ZH-CN' else 'ZH'
            }
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=30)
                res.raise_for_status()
                lines = [item["text"] for item in res.json()["translations"]]
                
                if len(lines) == len(batch):
                    translated_texts.extend(lines)
                else:
                    gt = GoogleTranslator(source=src, target=tgt)
                    res_b = gt.translate_batch(batch)
                    if res_b and isinstance(res_b, list):
                        translated_texts.extend(res_b)
                    else:
                        translated_texts.extend(batch)
            except Exception as e:
                print(f"Lỗi DeepL: {e}")
                try:
                    gt = GoogleTranslator(source=src, target=tgt)
                    res_b = gt.translate_batch(batch)
                    if res_b and isinstance(res_b, list):
                        translated_texts.extend(res_b)
                    else:
                        translated_texts.extend(batch)
                except Exception:
                    translated_texts.extend(batch)
                    
            if progress_callback:
                percent = int(((i + len(batch)) / len(texts)) * 100)
                progress_callback(f"Dịch bằng DeepL... {percent}% (Đã dịch {min(i + batch_size, len(texts))}/{len(texts)} câu)")

    # --- 6. CÁC ENGINE MIỄN PHÍ DỰ PHÒNG (GOOGLE / MYMEMORY) ---
    else:
        if engine == 'MyMemory':
            translator = MyMemoryTranslator(source=src, target=tgt)
        else:
            translator = GoogleTranslator(source=src, target=tgt)
            
        try:
            translated_texts = translator.translate_batch(texts)
            if not translated_texts or not isinstance(translated_texts, list):
                raise ValueError("translate_batch returned None or invalid type")
        except Exception:
            translated_texts = []
            for idx, text in enumerate(texts):
                try:
                    res_t = translator.translate(text)
                    if res_t is None:
                        res_t = text
                    translated_texts.append(res_t)
                except Exception:
                    translated_texts.append(text)
                if progress_callback:
                    percent = int(((idx + 1) / len(texts)) * 100)
                    progress_callback(f"Dịch từng dòng dự phòng... {percent}% ({idx+1}/{len(texts)})")
                    
    # Map kết quả dịch mới trở lại vị trí gốc và cập nhật vào cache toàn cục
    final_translated = []
    uncached_idx_in_translated = 0
    
    for idx, orig_text in enumerate(original_texts):
        if cached_results[idx] is not None:
            final_translated.append(cached_results[idx])
        else:
            if translated_texts and uncached_idx_in_translated < len(translated_texts):
                trans_text = translated_texts[uncached_idx_in_translated]
                uncached_idx_in_translated += 1
            else:
                trans_text = orig_text
            final_translated.append(trans_text)
            # Lưu lại vào cache toàn cục nếu dịch thành công
            if trans_text and trans_text.strip() and trans_text != orig_text:
                global_translation_cache.set(orig_text, engine, src, tgt, trans_text)
            
    translated_texts = final_translated

    # Tái thiết lập kết quả phụ đề
    translated_segments = []
    for idx, seg in enumerate(segments):
        trans_text = translated_texts[idx] if idx < len(translated_texts) else seg.get('text', '')
        
        # Nếu dòng đã sửa tay (manual_override), giữ lại bản text đã sửa tay đó
        has_override = seg.get('manual_override', False)
        final_text = seg.get('text', trans_text) if has_override else trans_text
        
        if not has_override:
            final_text = capitalize_proper_names(final_text)
            trans_text = capitalize_proper_names(trans_text)
            
        new_seg = {
            'start': seg['start'],
            'end': seg['end'],
            'raw_text': trans_text,
            'text': final_text,
            'orig_text': seg.get('orig_text', seg.get('text', '')),
            'manual_override': has_override
        }
        if 'bbox' in seg:
            new_seg['bbox'] = seg['bbox']
        if 'confidence' in seg:
            new_seg['confidence'] = seg['confidence']
        translated_segments.append(new_seg)
        
    return translated_segments

def refine_translated_segments(segments, glossary=None, api_key=None, engine='Gemini 1.5 Flash', progress_callback=None, ollama_model='qwen2.5'):
    """
    Sơ chế/Tinh chỉnh bản dịch thô (Giai đoạn 2) dùng LLM hỗ trợ Glossary và Ngữ cảnh.
    """
    if not segments:
        return segments
        
    import time
    import json
    import requests
    import re
    
    # --- KIỂM TRA CACHE SƠ CHẾ TOÀN CỤC ---
    cached_refinements = {}
    uncached_segments = []
    
    for idx, seg in enumerate(segments):
        orig = seg.get('orig_text', seg.get('text', ''))
        raw = seg.get('raw_text', seg.get('text', ''))
        
        # Đảm bảo cache hết hạn khi thay đổi glossary hoặc engine/model
        glossary_key = json.dumps(glossary, sort_keys=True) if glossary else ""
        model_key = f"{engine}:{ollama_model}" if engine == "Ollama Local" else engine
        cache_key = f"refine:{model_key}:{glossary_key}:{orig}->{raw}"
        
        cached_val = global_translation_cache.cache_data.get(cache_key)
        if cached_val is not None and cached_val.strip() and cached_val != orig:
            cached_refinements[idx] = cached_val
            if not seg.get('manual_override', False):
                seg['text'] = cached_val
        else:
            uncached_segments.append((idx, seg))
            
    # Nếu toàn bộ đã có trong cache sơ chế, trả về ngay lập tức
    if not uncached_segments:
        if progress_callback:
            progress_callback(f"Đã nạp toàn bộ {len(segments)} dòng sơ chế từ Cache toàn cục!")
        return segments
        
    # 1. Xây dựng Glossary Rules dạng text
    glossary_rules = ""
    if glossary:
        rules_list = []
        for k, v in glossary.items():
            if k.strip() and v.strip():
                rules_list.append(f"- '{k.strip()}' -> '{v.strip()}'")
        if rules_list:
            glossary_rules = "\n".join(rules_list)
            
    # 2. Phân chia lô động (Dynamic Batching) dựa trên tổng số ký tự (giới hạn 2500 ký tự)
    batches = []
    current_batch = []
    current_chars = 0
    
    for idx, seg in uncached_segments:
        orig = seg.get('orig_text', seg.get('text', ''))
        raw = seg.get('raw_text', seg.get('text', ''))
        
        entry = {
            'index': idx,
            'orig_text': orig,
            'raw_text': raw
        }
        # Tính độ dài ký tự của dòng này
        line_len = len(orig) + len(raw)
        
        # Nếu lô hiện tại vượt 2500 ký tự và không rỗng, chốt lô cũ và tạo lô mới
        if current_chars + line_len > 2500 and current_batch:
            batches.append(current_batch)
            current_batch = [entry]
            current_chars = line_len
        else:
            current_batch.append(entry)
            current_chars += line_len
            
    if current_batch:
        batches.append(current_batch)
        
    if progress_callback:
        progress_callback(f"Đang chuẩn bị sơ chế {len(segments)} dòng (chia làm {len(batches)} lô động)...")
        
    # Xác định endpoint và header của custom API hoặc OpenRouter
    model_name = "google/gemini-1.5-flash"
    if "1.5 Pro" in engine:
        model_name = "google/gemini-1.5-pro"
    elif "2.0 Flash" in engine:
        model_name = "google/gemini-2.0-flash"
    elif "Llama 3.1" in engine:
        model_name = "meta-llama/llama-3.1-70b-instruct"
    elif engine == "Ollama Local":
        model_name = ollama_model if ollama_model else "qwen2.5"
        
    if engine == "Ollama Local":
        base_url = "http://localhost:11434/v1"
    else:
        base_url = get_custom_provider_base_url()
    key = api_key or os.environ.get("CUSTOM_API_KEY", "")
    
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    # 3. Chạy refine cho từng lô
    for b_idx, batch in enumerate(batches):
        if progress_callback:
            percent = int((b_idx / len(batches)) * 100)
            progress_callback(f"Đang sơ chế dịch thuật (LLM Refinement): {percent}% (Lô {b_idx+1}/{len(batches)})...")
            
        batch_lines_prompt = []
        for item in batch:
            batch_lines_prompt.append(f"Line {item['index']}: {item['orig_text']} <Raw: {item['raw_text']}>")
            
        prompt = (
            "Bạn là một chuyên gia bản địa hóa video Douyin/Trung Quốc sang tiếng Việt, đặc biệt am hiểu ngôn ngữ mạng và các từ lóng bắt trend của giới trẻ/Gen Z Việt Nam.\n"
            "Nhiệm vụ của bạn là: Đọc các câu thoại tiếng Trung gốc và bản dịch thô trong danh sách dưới đây, giải mã nghĩa gốc của ngữ cảnh và dịch lại/sơ chế thành tiếng Việt cực kỳ tự nhiên, trôi chảy, bắt trend giới trẻ Việt Nam.\n\n"
            "QUY TẮC CẦN TUÂN THỦ:\n"
            "1. Dịch siêu mượt, bắt trend: Thoát hoàn toàn khỏi văn văn phong Hán Việt cứng nhắc. Hãy dùng ngôn ngữ giao tiếp hàng ngày, dí dỏm, hài hước. Nếu có từ lóng Trung Quốc, hãy chuyển đổi linh hoạt sang từ lóng tương đương của giới trẻ Việt Nam (ví dụ: 'vô ngữ' -> cạn lời/chằm zn; 'trà xanh' -> trà xanh/pick me girl; 'nữ cường' -> ngầu lòi/nữ cường; 'đả kích' -> cà khịa/dìm hàng...).\n"
            "2. Áp dụng bảng thuật ngữ Glossary dưới đây một cách nghiêm ngặt (nếu có):\n"
            f"{glossary_rules if glossary_rules else '(Không có)'}\n\n"
            "YÊU CẦU ĐỊNH DẠNG ĐẦU RA:\n"
            "1. Chỉ trả về duy nhất một danh sách mảng JSON (JSON list of strings) chứa các câu đã sơ chế xong, khớp chính xác số lượng và thứ tự dòng đầu vào. Tuyệt đối không gộp câu, không tự ý ngắt dòng.\n"
            "2. Tuyệt đối không giải thích, không chào hỏi, không thêm bất kỳ ký tự nào khác bên ngoài mảng JSON.\n\n"
            "Ví dụ định dạng đầu ra:\n"
            "[\n"
            "  \"Dòng dịch đã sơ chế 1\",\n"
            "  \"Dòng dịch đã sơ chế 2\"\n"
            "]\n\n"
            "DANH SÁCH CÁC CÂU CẦN SƠ CHẾ:\n" + "\n".join(batch_lines_prompt)
        )
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"} if "gemini" in model_name or "gpt" in model_name else None
        }
        
        # Cơ chế Retry (3 lần, exponential backoff)
        success = False
        refined_list = []
        retry_delay = 2.0
        
        for attempt in range(3):
            try:
                # Nếu API key rỗng và không chạy Ollama, cảnh báo nhưng không crash
                if not key and engine != "Ollama Local":
                    raise ValueError("Thiếu API Key cho động cơ dịch thuật LLM.")
                    
                res = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=30)
                res.raise_for_status()
                res_text = res.json()["choices"][0]["message"]["content"].strip()
                
                # Bóc tách JSON an toàn bằng Regex
                json_match = re.search(r'\[\s*".*?"\s*\]', res_text, re.DOTALL) or re.search(r'\[[\s\S]*?\]', res_text)
                if json_match:
                    res_text = json_match.group(0)
                    
                parsed_json = json.loads(res_text)
                if isinstance(parsed_json, list) and len(parsed_json) == len(batch):
                    refined_list = parsed_json
                    success = True
                    break
                else:
                    raise ValueError("Số lượng dòng phản hồi từ LLM không khớp với yêu cầu.")
            except Exception as e:
                print(f"Loi refine lo {b_idx} (Lan thu {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    
        # Áp dụng kết quả hoặc fallback
        for offset, item in enumerate(batch):
            idx = item['index']
            refined_val = refined_list[offset] if (success and offset < len(refined_list)) else item['raw_text']
            
            # Khớp Glossary thủ công bằng ranh giới từ để bảo hiểm 100% (word boundary \b)
            if glossary:
                for gk, gv in glossary.items():
                    if gk.strip() and gv.strip():
                        # Compile với ranh giới từ Unicode-aware
                        pattern = re.compile(rf'\b{re.escape(gk.strip())}\b', re.IGNORECASE)
                        refined_val = pattern.sub(gv.strip(), refined_val)
            
            # Ghi đè vào cột text nếu segment không bị manual_override
            if not segments[idx].get('manual_override', False):
                refined_val = capitalize_proper_names(refined_val)
                segments[idx]['text'] = refined_val

            # Lưu lại vào cache toàn cục
            orig = segments[idx].get('orig_text', segments[idx].get('text', ''))
            raw = segments[idx].get('raw_text', segments[idx].get('text', ''))
            glossary_key = json.dumps(glossary, sort_keys=True) if glossary else ""
            model_key = f"{engine}:{ollama_model}" if engine == "Ollama Local" else engine
            cache_key = f"refine:{model_key}:{glossary_key}:{orig}->{raw}"
            if success and refined_val and refined_val.strip() and refined_val != orig:
                global_translation_cache.cache_data[cache_key] = refined_val
            
        global_translation_cache.save_cache()
                
    if progress_callback:
        progress_callback("Hoàn tất sơ chế dịch thuật (LLM Refinement)!")
        
    return segments
