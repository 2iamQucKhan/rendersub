import os
import json
import cv2
from PIL import Image
from google import genai

def rotate_log_file_if_needed(log_file_path, max_bytes=1024*1024):
    """
    Xoay vòng file log đơn giản: Nếu file log vượt quá max_bytes (1MB),
    tự động giữ lại 500 dòng mới nhất và cắt phần log cũ.
    """
    if os.path.exists(log_file_path):
        try:
            if os.path.getsize(log_file_path) > max_bytes:
                with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                recent_lines = lines[-500:] if len(lines) > 500 else lines
                with open(log_file_path, "w", encoding="utf-8") as f:
                    f.write(f"--- LOG ROTATED AT MAXIMUM SIZE (1MB) ---\n")
                    f.writelines(recent_lines)
        except Exception:
            pass

def log_gemini_error(msg: str):
    print(f"[Gemini API Error] {msg}")
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "gemini_api_errors.log")
        rotate_log_file_if_needed(log_file)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")
    except Exception:
        pass

def parse_retry_delay(error_obj_or_str, default_sec=15.0):
    """
    Trích xuất số giây retryDelay từ phản hồi lỗi 429 RESOURCE_EXHAUSTED của Gemini API.
    Ví dụ: 'Please retry in 17.818800071s.' hoặc {'retryDelay': '17s'}
    """
    import re
    try:
        err_str = str(error_obj_or_str)
        m = re.search(r"retry in ([\d\.]+)s", err_str, re.IGNORECASE)
        if m:
            return min(60.0, max(2.0, float(m.group(1))))
        m2 = re.search(r"['\"]?retryDelay['\"]?\s*:\s*['\"]?([\d\.]+)s?['\"]?", err_str, re.IGNORECASE)
        if m2:
            return min(60.0, max(2.0, float(m2.group(1))))
    except Exception:
        pass
    return default_sec

def is_gemini_key(key: str) -> bool:
    """Kiểm tra xem key có đúng định dạng của Google Gemini không (không phải sk- key của xKiro/OpenAI)."""
    if not key or not isinstance(key, str):
        return False
    k = key.strip()
    if k.startswith("sk-"):
        return False
    return len(k) >= 20 and " " not in k

def load_gemini_keys():
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "api_keys.json"),
        os.path.join(os.getcwd(), "config", "api_keys.json"),
        "config/api_keys.json"
    ]
    valid_keys = []
    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    keys = data.get("gemini_keys", [])
                    if isinstance(keys, str):
                        keys = [k.strip() for k in keys.split(",") if k.strip()]
                    for idx, k in enumerate(keys):
                        if k and isinstance(k, str):
                            k_clean = k.strip()
                            if is_gemini_key(k_clean):
                                valid_keys.append(k_clean)
                            else:
                                if k_clean.startswith("sk-"):
                                    log_gemini_error(f"⚠️ Bỏ qua key #{idx+1} '{k_clean[:10]}...' trong gemini_keys vì đây là xKiro/OpenAI key (prefix 'sk-').")
                                else:
                                    log_gemini_error(f"⚠️ API Key #{idx+1} '{k_clean[:10]}...' bị bỏ qua (Độ dài tối thiểu 20 ký tự, không chứa khoảng trắng).")
                    if valid_keys:
                        return valid_keys
            except Exception as e:
                log_gemini_error(f"Lỗi khi đọc file key {p}: {e}")
    if not valid_keys:
        log_gemini_error("⚠️ Không tìm thấy Gemini API Key trong config/api_keys.json.")
    return valid_keys

class GeminiKeyManager:
    """Quản lý danh sách Gemini API Keys và tự động xoay vòng khi gặp lỗi 429/503/404."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.keys = []
            cls._instance.current_index = 0
            cls._instance.key_status = {}  # key -> "active" | "rate_limited" | "expired"
            cls._instance.load_keys()
        return cls._instance

    def load_keys(self, custom_keys=None):
        """Load danh sách keys từ config hoặc custom_keys."""
        if custom_keys:
            if isinstance(custom_keys, str):
                self.keys = [k.strip() for k in custom_keys.split(",") if k.strip() and len(k.strip()) >= 20]
            else:
                self.keys = [str(k).strip() for k in custom_keys if str(k).strip() and len(str(k).strip()) >= 20]
        else:
            self.keys = load_gemini_keys()
        
        for k in self.keys:
            if k not in self.key_status:
                self.key_status[k] = "active"

    def get_next_key(self):
        """Lấy key tiếp theo, bỏ qua key đang lỗi."""
        if not self.keys:
            self.load_keys()
        if not self.keys:
            return None
        
        # 1. Ưu tiên key active
        for i in range(len(self.keys)):
            idx = (self.current_index + i) % len(self.keys)
            k = self.keys[idx]
            if self.key_status.get(k) == "active":
                self.current_index = (idx + 1) % len(self.keys)
                return k
                
        # 2. Nếu không có active, thử rate_limited
        for i in range(len(self.keys)):
            idx = (self.current_index + i) % len(self.keys)
            k = self.keys[idx]
            if self.key_status.get(k) != "expired":
                self.current_index = (idx + 1) % len(self.keys)
                return k

        # 3. Fallback: xoay vòng bình thường
        self.current_index = (self.current_index + 1) % len(self.keys)
        return self.keys[self.current_index]

    def mark_key_rate_limited(self, key):
        """Đánh dấu key bị rate limit (429 / RESOURCE_EXHAUSTED)."""
        self.key_status[key] = "rate_limited"
        masked = key[:10] + "..." if len(key) >= 10 else key
        log_gemini_error(f"⚠️ Key {masked} bị Rate Limit (429), đang tự động chuyển sang key khác...")

    def mark_key_expired(self, key):
        """Đánh dấu key đã lỗi vĩnh viễn (404/503/Invalid)."""
        self.key_status[key] = "expired"
        masked = key[:10] + "..." if len(key) >= 10 else key
        log_gemini_error(f"⚠️ Key {masked} đã bị lỗi, đang tự động chuyển sang key khác...")

    def mark_key_active(self, key):
        self.key_status[key] = "active"

    def test_key(self, key):
        """Kiểm tra key có hợp lệ không."""
        import requests
        try:
            resp = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": key},
                timeout=8
            )
            return resp.status_code == 200
        except Exception:
            return False

    def test_all_keys(self):
        """Kiểm tra tất cả keys và trả về kết quả."""
        results = {}
        for k in self.keys:
            ok = self.test_key(k)
            self.key_status[k] = "active" if ok else "expired"
            results[k] = self.key_status[k]
        return results

    def get_summary(self):
        active_cnt = sum(1 for s in self.key_status.values() if s == "active")
        expired_cnt = sum(1 for s in self.key_status.values() if s == "expired")
        total_cnt = len(self.keys)
        return active_cnt, expired_cnt, total_cnt

gemini_key_manager = GeminiKeyManager()

GEMINI_ACTIVE_MODELS = [
    "gemini-flash-latest",            # ⭐ Best & luôn active
    "gemini-flash-lite-latest",       # 💨 Siêu nhẹ & nhanh nhất
    "gemini-3.5-flash",               # ⚡ Thế hệ mới
    "gemini-3.7-flash",               # ⚡ Model mạnh nhất
    "gemini-2.5-flash-lite",          # ⚡ Tiết kiệm quota
    "gemini-2.0-flash-exp",           # 🧪 Experimental
    "gemini-2.0-flash-lite-preview",  # 🧪 Lite Preview
    "gemini-1.5-flash-8b",            # 🧪 8B
    "gemini-2.5-pro",                 # 🏆 Pro
    "gemini-pro-latest"               # 🏆 Pro Latest
]
GEMINI_VISION_MODELS = GEMINI_ACTIVE_MODELS
GEMINI_AVAILABLE_MODELS = GEMINI_ACTIVE_MODELS

def get_configured_gemini_models(primary_model=None):
    """
    Trả về danh sách các model Gemini ưu tiên để thử theo thứ tự fallback.
    """
    settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "app_settings.json")
    auto_fallback = True
    if not primary_model:
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    primary_model = cfg.get("gemini_model", "gemini-flash-latest")
                    auto_fallback = cfg.get("gemini_auto_fallback_model", True)
            except Exception:
                primary_model = "gemini-flash-latest"
        else:
            primary_model = "gemini-flash-latest"

    clean_primary = str(primary_model).split()[0].replace("models/", "") if primary_model else "gemini-flash-latest"
    if clean_primary == "Auto" or clean_primary == "Tự" or "Auto" in str(primary_model):
        clean_primary = "gemini-flash-latest"

    models_order = [clean_primary]
    if auto_fallback or "Auto" in str(primary_model):
        for m in GEMINI_ACTIVE_MODELS:
            if m not in models_order:
                models_order.append(m)
    return models_order

def test_gemini_model_status(model_name="gemini-flash-latest", api_key=None):
    """
    Kiểm tra xem model Gemini cụ thể có hoạt động hay không.
    Trả về (is_ok: bool, message: str)
    """
    from google import genai
    keys = [api_key] if api_key else load_gemini_keys()
    if not keys:
        return False, "Không tìm thấy Gemini API Key hợp lệ trong config."

    clean_model = str(model_name).split()[0].replace("models/", "")
    if clean_model == "Auto" or "Auto" in str(model_name):
        clean_model = "gemini-flash-latest"

    last_err = None
    for k in keys:
        try:
            client = genai.Client(api_key=k)
            res = client.models.generate_content(
                model=clean_model,
                contents="Hello, reply with 1 word 'OK'."
            )
            txt = res.text.strip() if res and hasattr(res, 'text') and res.text else "OK"
            return True, f"✅ Model '{clean_model}' hoạt động tốt! (Response: {txt[:30]})"
        except Exception as e:
            last_err = e
            continue
    return False, f"❌ Model '{clean_model}' không khả dụng: {last_err}"

def list_available_gemini_models(api_key=None):
    """
    Gọi Gemini API để lấy danh sách models đang active hỗ trợ generateContent.
    """
    import requests
    keys = [api_key] if api_key else load_gemini_keys()
    if not keys:
        return []
    
    for k in keys:
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models"
            params = {"key": k}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                raw_models = resp.json().get("models", [])
                vision_models = []
                for m in raw_models:
                    m_name = m.get("name", "").replace("models/", "")
                    methods = m.get("supportedGenerationMethods", []) or m.get("supportedMethods", [])
                    if "generateContent" in methods:
                        vision_models.append(m_name)
                if vision_models:
                    return vision_models
        except Exception:
            pass

    try:
        from google import genai
        client = genai.Client(api_key=keys[0])
        models = []
        for m in client.models.list():
            m_name = m.name.replace("models/", "")
            models.append(m_name)
        if models:
            return models
    except Exception as e:
        log_gemini_error(f"Lỗi khi lấy danh sách model: {e}")

    return GEMINI_ACTIVE_MODELS

get_gemini_active_models_list = list_available_gemini_models

def extract_subtitles_with_gemini_vision(frame_rgb, model_name=None):
    """
    Sử dụng Gemini Vision API (SDK google.genai) để quét chữ, dịch thuật và trả về tọa độ Bounding Box [ymin, xmin, ymax, xmax].
    Tự động fallback mượt mà qua các model khả dụng khi gặp lỗi 404 / 503 / 429 và xoay vòng Key.
    """
    keys = load_gemini_keys()
    if not keys:
        log_gemini_error("⚠️ Gemini API lỗi: Không có API Key trong config/api_keys.json. Đang chuyển sang OCR truyền thống...")
        return []

    pil_img = Image.fromarray(frame_rgb)

    prompt = (
        "Hãy phân tích khung hình video này:\n"
        "1. Nhận diện toàn bộ chữ xuất hiện trong ảnh (kể cả chữ nghệ thuật, chữ ở mép trên, mép dưới).\n"
        "2. Dịch toàn bộ nội dung đó sang tiếng Việt khẩu ngữ tự nhiên, mượt mà, hóm hỉnh.\n"
        "3. Trả về tọa độ Bounding Box [ymin, xmin, ymax, xmax] theo tỷ lệ 0 - 1000 chuẩn của khung hình.\n\n"
        "BẮT BỘC trả về KẾT QUẢ dưới dạng duy nhất là 1 JSON ARRAY thuần (JSON valid, KHÔNG kèm markdown formatting hay giải thích thêm):\n"
        "[\n"
        "  {\n"
        "    \"original_text\": \"chu_goc_tieng_trung\",\n"
        "    \"translated_text\": \"ban_dich_tieng_viet\",\n"
        "    \"box_2d\": [ymin, xmin, ymax, xmax]\n"
        "  }\n"
        "]"
    )

    fallback_models = get_configured_gemini_models(model_name)

    for idx, key in enumerate(keys):
        try:
            client = genai.Client(api_key=key)
            for m in fallback_models:
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=[prompt, pil_img]
                    )
                    raw_text = response.text.strip() if response and response.text else ""
                    if raw_text.startswith("```json"):
                        raw_text = raw_text[7:]
                    if raw_text.startswith("```"):
                        raw_text = raw_text[3:]
                    if raw_text.endswith("```"):
                        raw_text = raw_text[:-3]
                    raw_text = raw_text.strip()

                    parsed = json.loads(raw_text)
                    if isinstance(parsed, list):
                        if len(parsed) > 0:
                            print(f"✅ [Gemini Vision] Quét thành công với Model '{m}' & API Key #{idx + 1}: {len(parsed)} vùng chữ")
                        return parsed
                except Exception as ex_m:
                    err_str = str(ex_m)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                        if idx < len(keys) - 1:
                            log_gemini_error(f"⚠️ Key #{idx + 1} chạm hạn ngạch 429. Đang chuyển sang Key #{idx + 2}...")
                            break  # Chuyển sang key tiếp theo
                        else:
                            retry_delay = parse_retry_delay(ex_m, default_sec=16.0)
                            log_gemini_error(f"⏳ Tất cả key Gemini chạm hạn mức 429 ({m}). Đang đợi {retry_delay:.1f}s để hồi phục quota...")
                            time.sleep(retry_delay + 1.5)
                            try:
                                response = client.models.generate_content(model=m, contents=[prompt, pil_img])
                                raw_text = response.text.strip() if response and response.text else ""
                                if raw_text.startswith("```json"): raw_text = raw_text[7:]
                                if raw_text.startswith("```"): raw_text = raw_text[3:]
                                if raw_text.endswith("```"): raw_text = raw_text[:-3]
                                parsed = json.loads(raw_text.strip())
                                if isinstance(parsed, list) and len(parsed) > 0:
                                    print(f"✅ [Gemini Vision] Quét thành công với Model '{m}' sau khi đợi quota: {len(parsed)} vùng chữ")
                                    return parsed
                            except Exception:
                                break
                    elif "404" in err_str or "NOT_FOUND" in err_str:
                        log_gemini_error(f"⚠️ Model '{m}' không khả dụng (404), đang chuyển sang model tiếp theo...")
                        continue
                    elif "503" in err_str or "UNAVAILABLE" in err_str:
                        log_gemini_error(f"⚠️ Model '{m}' đang quá tải (503), đang chuyển sang model tiếp theo...")
                        continue
                    else:
                        log_gemini_error(f"⚠️ Gemini API lỗi với Model '{m}': {ex_m}")
                        continue
        except Exception as e:
            log_gemini_error(f"⚠️ Gemini API lỗi với Key #{idx + 1}: {e}")
            continue

    log_gemini_error("⚠️ Gemini API thất bại trên tất cả keys/models. Đang chuyển sang OCR truyền thống...")
    return []

def scan_video_frames_with_gemini(video_path, sample_interval_sec=0.5, api_keys=None, model_name=None, progress_callback=None):
    """
    Quét toàn bộ video bằng Gemini Vision API theo mốc thời gian (interval sampling):
    - Nhận diện chữ nghệ thuật, logo, tiêu đề và sub.
    - Dịch tự nhiên sang Tiếng Việt.
    - Trả về danh sách segments chứa timecode (start/end), text, orig_text và bounding box pixel.
    - Tự động fallback qua các model mới và xoay key khi gặp lỗi Quota/429/404/503.
    """
    if not os.path.exists(video_path):
        print(f"[Gemini Vision] File video khong ton tai: {video_path}")
        return []

    if not api_keys:
        api_keys = load_gemini_keys()
    if not api_keys:
        log_gemini_error("⚠️ Gemini API lỗi: Không tìm thấy API Key hợp lệ (bắt đầu bằng 'AIzaSy') để quét video. Đang chuyển sang OCR truyền thống...")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Gemini Vision] Khong the mo video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_step = int(fps * sample_interval_sec)
    if frame_step <= 0:
        frame_step = max(1, int(fps * 0.5))

    raw_detections = []

    current_frame_idx = 0
    key_index = 0
    num_keys = len(api_keys)

    fallback_models = get_configured_gemini_models(model_name)
    print(f"[Gemini Vision] Bat dau quet video: {duration_sec:.1f}s | Sampling moi {sample_interval_sec}s | So luong Keys: {num_keys} | Model khoi dau: {fallback_models[0]}")

    consecutive_quota_errors = 0
    while current_frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        timestamp_sec = current_frame_idx / fps
        pct = int(min(100, (current_frame_idx / total_frames) * 100))
        if progress_callback:
            progress_callback(f"Đang quét video (Gemini Vision {fallback_models[0]}): {pct}% (tại {timestamp_sec:.1f}s)...")

        # Chuẩn bị ảnh
        success, encoded_image = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            current_frame_idx += frame_step
            continue

        pil_img = Image.open(io.BytesIO(encoded_image.tobytes()))

        prompt = (
            "Phát hiện tất cả các vùng chữ phụ đề hoặc tiêu đề có trong bức ảnh này.\n"
            "Đối với mỗi vùng phát hiện được, hãy trích xuất nội dung gốc và dịch ngay sang Tiếng Việt chuẩn theo ngữ cảnh video.\n"
            "Trả về kết quả dưới định dạng JSON mảng các đối tượng chính xác như sau:\n"
            "[\n"
            "  {\n"
            "    \"original_text\": \"chữ gốc trong ảnh\",\n"
            "    \"translated_text\": \"bản dịch tiếng Việt chuẩn\",\n"
            "    \"box_2d\": [ymin, xmin, ymax, xmax]\n"
            "  }\n"
            "]"
        )

        success = False
        attempts = 0
        max_attempts = num_keys * len(fallback_models)

        while not success and attempts < max_attempts:
            curr_key = api_keys[key_index % num_keys]
            try:
                client = genai.Client(api_key=curr_key)
                for m in fallback_models:
                    try:
                        response = client.models.generate_content(
                            model=m,
                            contents=[prompt, pil_img]
                        )
                        raw_text = response.text.strip() if response and response.text else ""

                        if raw_text.startswith("```json"):
                            raw_text = raw_text[7:]
                        if raw_text.startswith("```"):
                            raw_text = raw_text[3:]
                        if raw_text.endswith("```"):
                            raw_text = raw_text[:-3]
                        raw_text = raw_text.strip()

                        parsed = json.loads(raw_text)
                        if isinstance(parsed, list):
                            for item in parsed:
                                orig = item.get("original_text", "").strip()
                                trans = item.get("translated_text", "").strip()
                                box = item.get("box_2d", [0, 0, 0, 0])
                                if orig and box:
                                    ymin, xmin, ymax, xmax = box
                                    y1 = int(ymin * height / 1000.0)
                                    x1 = int(xmin * width / 1000.0)
                                    y2 = int(ymax * height / 1000.0)
                                    x2 = int(xmax * width / 1000.0)
                                    w_box = max(10, x2 - x1)
                                    h_box = max(10, y2 - y1)
                                    pixel_bbox = [x1, y1, w_box, h_box]

                                    raw_detections.append({
                                        "timestamp": timestamp_sec,
                                        "orig_text": orig,
                                        "translated_text": trans,
                                        "bbox": pixel_bbox
                                    })
                            success = True
                            consecutive_quota_errors = 0
                            break
                    except Exception as ex_m:
                        err_str = str(ex_m)
                        if "404" in err_str or "NOT_FOUND" in err_str:
                            log_gemini_error(f"⚠️ Model '{m}' không khả dụng (404), chuyển sang model tiếp...")
                        elif "503" in err_str or "UNAVAILABLE" in err_str:
                            log_gemini_error(f"⚠️ Model '{m}' quá tải (503), chuyển sang model tiếp...")
                        elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                            consecutive_quota_errors += 1
                            retry_delay = parse_retry_delay(ex_m, default_sec=16.0)
                            
                            # Nếu còn các key khác, thử key khác trước
                            if num_keys > 1 and key_index % num_keys != (num_keys - 1):
                                log_gemini_error(f"⚠️ Key #{key_index % num_keys + 1} hết quota 429 ({m}). Đang đổi sang Key tiếp theo...")
                                break
                            
                            # Nếu đã qua hết key hoặc chỉ có 1 key: Tạm dừng đợi quota hồi phục thay vì spam đổi model
                            msg_wait = f"⏳ Gemini chạm hạn mức 429 ({m}). Đang tạm dừng {retry_delay:.1f}s để hồi phục quota trước khi thử lại..."
                            log_gemini_error(msg_wait)
                            if progress_callback:
                                progress_callback(msg_wait)
                            time.sleep(retry_delay + 1.5)
                            
                            # Thử lại 1 lần với model và key hiện tại sau khi đã đợi
                            try:
                                response = client.models.generate_content(model=m, contents=[prompt, pil_img])
                                raw_text = response.text.strip() if response and response.text else ""
                                if raw_text.startswith("```json"): raw_text = raw_text[7:]
                                if raw_text.startswith("```"): raw_text = raw_text[3:]
                                if raw_text.endswith("```"): raw_text = raw_text[:-3]
                                parsed = json.loads(raw_text.strip())
                                if isinstance(parsed, list):
                                    for item in parsed:
                                        orig = item.get("original_text", "").strip()
                                        trans = item.get("translated_text", "").strip()
                                        box = item.get("box_2d", [0, 0, 0, 0])
                                        if orig and box:
                                            ymin, xmin, ymax, xmax = box
                                            y1 = int(ymin * height / 1000.0)
                                            x1 = int(xmin * width / 1000.0)
                                            y2 = int(ymax * height / 1000.0)
                                            x2 = int(xmax * width / 1000.0)
                                            raw_detections.append({
                                                "timestamp": timestamp_sec,
                                                "orig_text": orig,
                                                "translated_text": trans,
                                                "bbox": [x1, y1, max(10, x2 - x1), max(10, y2 - y1)]
                                            })
                                    success = True
                                    consecutive_quota_errors = 0
                                    break
                            except Exception:
                                break
                        else:
                            log_gemini_error(f"⚠️ Gemini API lỗi khi quét frame với Model '{m}': {ex_m}")
                        continue
            except Exception as e_key:
                log_gemini_error(f"⚠️ Lỗi khởi tạo client với Key #{key_index % num_keys + 1}: {e_key}")

            if not success:
                key_index += 1
                attempts += 1

        current_frame_idx += frame_step

    cap.release()

    if not raw_detections:
        print("[Gemini Vision] Khong tim thay phu de/chu tren toan bo video.")
        return []

    # Ghép và lọc trùng lặp các phân đoạn phụ đề liên tiếp
    raw_detections.sort(key=lambda x: x['timestamp'])
    segments = []
    current_seg = None

    for item in raw_detections:
        ts = item['timestamp']
        orig = item['orig_text']
        trans = item['translated_text']
        bbox = item['bbox']

        if current_seg is None:
            current_seg = {
                "start": max(0.0, ts - 0.2),
                "end": ts + sample_interval_sec,
                "text": trans,
                "orig_text": orig,
                "bbox": bbox
            }
        else:
            # Nếu chữ tương tự và thời gian nối tiếp nhau < 2s
            if current_seg['orig_text'] == orig or current_seg['text'] == trans:
                current_seg['end'] = ts + sample_interval_sec
            else:
                segments.append(current_seg)
                current_seg = {
                    "start": max(0.0, ts - 0.2),
                    "end": ts + sample_interval_sec,
                    "text": trans,
                    "orig_text": orig,
                    "bbox": bbox
                }

    if current_seg:
        segments.append(current_seg)

    print(f"[Gemini Vision] Da tao {len(segments)} phan doan sub hoan chinh voi ca ban dich & Bounding Box.")
    return segments

if __name__ == "__main__":
    video_path = "videos/宿舍空调哥舍友夏天不开空调_哔哩哔哩_bilibili.mp4"
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 50)  # frame 50
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            results = extract_subtitles_with_gemini_vision(frame_rgb)
            print("\n=== KET QUA GEMINI VISION API ===")
            for item in results:
                orig = item.get("original_text", "")
                trans = item.get("translated_text", "")
                box = item.get("box_2d", [0,0,0,0])
                ymin, xmin, ymax, xmax = box
                # Convert 0-1000 to pixels
                y1, x1, y2, x2 = int(ymin * h / 1000.0), int(xmin * w / 1000.0), int(ymax * h / 1000.0), int(xmax * w / 1000.0)
                orig_str = orig.encode('ascii', errors='backslashreplace').decode('ascii')
                trans_str = trans.encode('ascii', errors='backslashreplace').decode('ascii')
                print(f"[Gemini Vision] Chu goc: {orig_str} | Dich: {trans_str}")
                print(f"[Gemini Vision] Toa do Pixel: Y={y1}->{y2}px, X={x1}->{x2}px (Khung hinh {w}x{h})\n")
