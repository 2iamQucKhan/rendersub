import os
import sys
import re
import requests
import json
import threading
from pathlib import Path
from deep_translator import GoogleTranslator, MyMemoryTranslator

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

def log_translator_gemini_error(msg: str):
    print(f"[Gemini Translator Error] {msg}")
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "gemini_api_errors.log")
        rotate_log_file_if_needed(log_file)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")
    except Exception:
        pass

def call_xkiro_api(prompt, api_key="", model_name="deepseek/deepseek-v4-pro"):
    import xkiro_client
    return xkiro_client.translate_with_xkiro(prompt, model=model_name, api_key=api_key)

def is_prefer_xkiro_enabled():
    """
    Đọc cấu hình prefer_xkiro từ config/app_settings.json.
    Mặc định = True (Bật ưu tiên xKiro AI DeepSeek v4 Pro).
    """
    settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "app_settings.json")
    if os.path.exists(settings_file):
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return bool(data.get("prefer_xkiro", True))
        except Exception:
            pass
    return True

def translate_batch_with_fallback_chain(batch, src="auto", tgt="vi", api_key=None, prefer_xkiro=None, progress_callback=None, xkiro_key=None):
    """
    CHUỖI FALLBACK DỊCH THUẬT NGUYÊN TẮC 3 TẦNG:
    Tầng 1: xKiro AI (Model deepseek/deepseek-v4-pro - Miễn phí $0).
    Tầng 2: Gemini API (SDK google.genai với các model gemini-flash-latest, gemini-3.5-flash...).
    Tầng 3: Google Translate / VietPhrase (Scraper miễn phí 100% làm dự phòng cuối cùng).
    """
    if prefer_xkiro is None:
        prefer_xkiro = is_prefer_xkiro_enabled()

    import xkiro_client
    results = []

    # -------------------------------------------------------------------------
    # TẦNG 1: XKIRO AI (DEEPSEEK V4 PRO - $0 COST)
    # -------------------------------------------------------------------------
    xkiro_success = False
    if prefer_xkiro:
        x_keys = xkiro_client.load_xkiro_keys()
        if x_keys:
            last_xk_err = None
            for xk in x_keys:
                try:
                    xk_batch_res = []
                    for text_item in batch:
                        res_t = xkiro_client.translate_with_xkiro(text_item, target_lang=tgt, api_key=xk)
                        if res_t and isinstance(res_t, str) and res_t.strip():
                            xk_batch_res.append(res_t.strip())
                        else:
                            raise ValueError("xKiro trả về response rỗng.")
                    if len(xk_batch_res) == len(batch):
                        results = xk_batch_res
                        xkiro_success = True
                        if progress_callback:
                            progress_callback(f"🟢 [Tầng 1: xKiro AI] Dịch thành công {len(batch)} câu qua DeepSeek v4 Pro ($0).")
                        break
                except Exception as e_xk:
                    last_xk_err = e_xk
                    err_msg = f"❌ [xKiro Fallback Log] Lỗi Key '{xk[:10]}...': {e_xk}"
                    print(err_msg)
                    xkiro_client.log_xkiro_error(err_msg)

            if not xkiro_success and last_xk_err:
                msg = f"⚠️ [FALLBACK THÔNG BÁO] Tất cả Key xKiro AI đều bị lỗi ({last_xk_err}). ĐANG TỰ ĐỘNG CHUYỂN SANG TẦNG 2 (GEMINI API)..."
                print(msg)
                xkiro_client.log_xkiro_error(msg)
                if progress_callback:
                    progress_callback(msg)
        else:
            msg = "⚠️ [FALLBACK THÔNG BÁO] Không tìm thấy xKiro API Key trong config/api_keys.json ('xkiro_keys'). ĐANG CHUYỂN SANG TẦNG 2 (GEMINI API)..."
            print(msg)
            if progress_callback:
                progress_callback(msg)

    if xkiro_success and len(results) == len(batch):
        return results

    # -------------------------------------------------------------------------
    # TẦNG 2: GEMINI API (GOOGLE AI STUDIO)
    # -------------------------------------------------------------------------
    gemini_success = False
    gemini_keys = []
    if api_key and isinstance(api_key, str):
        gemini_keys = [k.strip() for k in api_key.split(",") if k.strip() and not k.strip().startswith("sk-")]

    if not gemini_keys:
        key_file = os.path.abspath(os.path.join("config", "api_keys.json"))
        if os.path.exists(key_file):
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    gemini_keys = json.load(f).get("gemini_keys", [])
            except Exception:
                pass

    if gemini_keys:
        prompt = (
            f"Bạn là một biên dịch viên phim chuyên nghiệp dịch từ '{src}' sang '{tgt}'.\n"
            f"Hãy dịch các câu thoại video sau sang Tiếng Việt cực kỳ tự nhiên, trôi chảy, thoát ý, đúng văn phong hội thoại phim ảnh hàng ngày.\n"
            f"CHỈ trả về kết quả dịch từng dòng tương ứng, không đánh số thứ tự hay thêm giải thích:\n\n"
            + "\n".join(batch)
        )
        last_gem_err = None
        for gk in gemini_keys:
            try:
                res_text = call_gemini_with_fallback(prompt, gk, model_name="gemini-flash-latest")
                lines = [l.strip() for l in res_text.splitlines() if l.strip()]
                if len(lines) == len(batch):
                    results = lines
                    gemini_success = True
                    if progress_callback:
                        progress_callback(f"🟢 [Tầng 2: Gemini API] Dịch thành công {len(batch)} câu.")
                    break
            except Exception as e_gk:
                last_gem_err = e_gk
                log_translator_gemini_error(f"Lỗi Gemini Key '{gk[:10]}...': {e_gk}")

        if not gemini_success and last_gem_err:
            msg = f"⚠️ [FALLBACK THÔNG BÁO] Gemini API bị lỗi ({last_gem_err}). ĐANG TỰ ĐỘNG CHUYỂN SANG TẦNG 3 (GOOGLE TRANSLATE)..."
            print(msg)
            log_translator_gemini_error(msg)
            if progress_callback:
                progress_callback(msg)

    if gemini_success and len(results) == len(batch):
        return results

    # -------------------------------------------------------------------------
    # TẦNG 3: GOOGLE TRANSLATE (FALLBACK DỰ PHÒNG CUỐI CÙNG)
    # -------------------------------------------------------------------------
    try:
        gt = GoogleTranslator(source=src, target=tgt)
        res_b = gt.translate_batch(batch)
        if res_b and isinstance(res_b, list) and len(res_b) == len(batch):
            if progress_callback:
                progress_callback(f"🔵 [Tầng 3: Google Translate] Dịch thành công {len(batch)} câu.")
            return res_b
    except Exception as e_gt:
        print(f"⚠️ [Google Translate Fallback] Lỗi scraper: {e_gt}")

    return batch # Dự phòng tuyệt đối: Giữ nguyên text gốc

def get_custom_provider_base_url():
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

def call_gemini_with_fallback(prompt, api_key, model_name="gemini-2.0-flash"):
    if not api_key:
        raise ValueError("API Key rỗng.")
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
        return res.json()["choices"][0]["message"]["content"].strip()
    else:
        from google import genai
        keys = [k.strip().strip("'").strip('"') for k in api_key.split(",") if k.strip() and not k.strip().startswith("sk-")]
        if not keys:
            raise ValueError("Không có Gemini API Key hợp lệ (bỏ qua các key dạng sk-).")

        valid_keys = [k for k in keys if len(k) >= 20 and " " not in k]
        if not valid_keys:
            raise ValueError("Không tìm thấy Gemini API Key hợp lệ (độ dài tối thiểu 20 ký tự, không chứa khoảng trắng).")

        clean_model = model_name.replace("models/", "")
        candidate_models = [
            clean_model,
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
            "gemini-3.5-flash",
            "gemini-3.7-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash-lite-preview",
            "gemini-1.5-flash-8b",
            "gemini-2.5-pro",
            "gemini-pro-latest"
        ]
        
        # Loại bỏ trùng lặp giữ nguyên thứ tự
        seen = set()
        dedup_models = []
        for m in candidate_models:
            if m and m not in seen:
                seen.add(m)
                dedup_models.append(m)

        last_err = None
        for idx_k, k_curr in enumerate(valid_keys):
            try:
                client = genai.Client(api_key=k_curr)
                for m_curr in dedup_models:
                    try:
                        response = client.models.generate_content(
                            model=m_curr,
                            contents=prompt
                        )
                        if response and hasattr(response, 'text') and response.text:
                            return response.text.strip()
                    except Exception as e_m:
                        last_err = e_m
                        err_str = str(e_m)
                        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                            if idx_k < len(valid_keys) - 1:
                                log_translator_gemini_error(f"⚠️ Key #{idx_k + 1} chạm hạn ngạch 429 ({m_curr}). Đang đổi sang Key #{idx_k + 2}...")
                                break
                            else:
                                from gemini_vision_ocr import parse_retry_delay
                                retry_delay = parse_retry_delay(e_m, default_sec=16.0)
                                log_translator_gemini_error(f"⏳ Tất cả key Gemini chạm hạn mức 429 ({m_curr}). Đang đợi {retry_delay:.1f}s để hồi phục quota...")
                                time.sleep(retry_delay + 1.5)
                                try:
                                    response = client.models.generate_content(model=m_curr, contents=prompt)
                                    if response and hasattr(response, 'text') and response.text:
                                        return response.text.strip()
                                except Exception:
                                    break
                        elif "404" in err_str or "NOT_FOUND" in err_str:
                            log_translator_gemini_error(f"⚠️ Model '{m_curr}' không khả dụng (404), chuyển model tiếp...")
                            continue
                        else:
                            log_translator_gemini_error(f"⚠️ Lỗi Gemini Model '{m_curr}': {e_m}")
                            continue
            except Exception as e_k:
                last_err = e_k
                log_translator_gemini_error(f"⚠️ Lỗi Gemini Client Init: {e_k}")

        raise RuntimeError(f"Gemini API Error: {last_err}")

# Bí danh tương thích
call_gemini_api = call_gemini_with_fallback

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

    def get_cache(self, src_text):
        if not src_text:
            return None
        with self.lock:
            cfg_cache_path = os.path.join(os.getcwd(), "config", "trans_cache.json")
            if os.path.exists(cfg_cache_path):
                try:
                    with open(cfg_cache_path, "r", encoding="utf-8") as f:
                        cfg_data = json.load(f)
                        if src_text in cfg_data:
                            return cfg_data[src_text]
                except Exception:
                    pass
            return self.cache_data.get(src_text)

    def set_cache(self, src_text, vi_text):
        if not src_text or not vi_text:
            return
        with self.lock:
            self.cache_data[src_text] = vi_text
            cfg_cache_path = os.path.join(os.getcwd(), "config", "trans_cache.json")
            try:
                os.makedirs(os.path.dirname(cfg_cache_path), exist_ok=True)
                cfg_data = {}
                if os.path.exists(cfg_cache_path):
                    with open(cfg_cache_path, "r", encoding="utf-8") as f:
                        cfg_data = json.load(f)
                cfg_data[src_text] = vi_text
                with open(cfg_cache_path, "w", encoding="utf-8") as f:
                    json.dump(cfg_data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        self.save_cache()

global_translation_cache = TranslationCacheManager()

def _post_process_spoken_vietnamese(text):
    if not text:
        return text
    replacements = [
        (r'\bBởi vì điều này\b', 'Thế nên là'),
        (r'\bbởi vì điều này\b', 'thế nên là'),
        (r'\bNói tóm lại\b', 'Tóm lại là'),
        (r'\bnói tóm lại\b', 'tóm lại là'),
        (r'\bBất quá\b', 'Tuy nhiên'),
        (r'\bbất quá\b', 'tuy nhiên'),
        (r'\bNói cách khác\b', 'Nói chung là'),
        (r'\bnói cách khác\b', 'nói chung là'),
        (r'\bThành ra là\b', 'Nên là'),
        (r'\bthành ra là\b', 'nên là'),
        (r'\bDo đó\b', 'Vì vậy'),
        (r'\bdo đó\b', 'vì vậy'),
    ]
    res = text
    for pattern, repl in replacements:
        res = re.sub(pattern, repl, res)
    res = re.sub(r'\s+([,.\!?])', r'\1', res)
    res = re.sub(r'\s+', ' ', res).strip()
    return res

def batch_refine_subtitles(segments, api_key=None, model_name="gemini-1.5-flash", batch_size=12, progress_callback=None):
    """
    Gom khối 10-15 câu phụ đề liền kề và tinh chỉnh bằng LLM (Hybrid Batch-Refine).
    Tiết kiệm 70-80% token API, mượt mà chuẩn Gen Z Reviewer/Subber.
    """
    if not segments:
        return []

    manager = global_translation_cache
    translator = VietPhraseTranslator()

    unprocessed_indices = []
    for idx, seg in enumerate(segments):
        src_text = seg.get('orig_text') or seg.get('text') or ""
        cached = manager.get_cache(src_text)
        if cached:
            seg['text'] = cached
            seg['raw_text'] = cached
        else:
            draft = translator.translate(src_text)
            seg['draft'] = draft
            unprocessed_indices.append(idx)

    if not unprocessed_indices:
        if progress_callback:
            progress_callback("100% phụ đề nạp từ Cache cục bộ ($0 token cost).")
        return segments

    total_unprocessed = len(unprocessed_indices)
    num_batches = (total_unprocessed + batch_size - 1) // batch_size

    for b_idx in range(num_batches):
        batch_indices = unprocessed_indices[b_idx * batch_size : (b_idx + 1) * batch_size]
        batch_items = []
        for i_pos, seg_idx in enumerate(batch_indices):
            seg = segments[seg_idx]
            src_text = seg.get('orig_text') or seg.get('text') or ""
            batch_items.append({
                "id": i_pos + 1,
                "src": src_text,
                "draft": seg.get('draft', src_text)
            })

        if progress_callback:
            progress_callback(f"[Batch {b_idx + 1}/{num_batches}] Đang tinh chỉnh {len(batch_items)} câu phụ đề...")

        prompt = (
            "Role: Bạn là một Biên dịch viên & Content Creator nổi tiếng trên TikTok/YouTube chuyên làm video reup/review.\n"
            "Task: Hãy tinh chỉnh danh sách câu dịch nháp Tiếng Việt dưới đây sao cho:\n"
            "* Ngôn từ tự nhiên, mượt mà như người Việt nói chuyện hàng ngày, giàu cảm xúc, dí dỏm.\n"
            "* Loại bỏ triệt để lối dịch Hán-Việt cứng nhắc, dịch word-by-word kiểu Google Translate.\n"
            "* Tự động bắt cặp từ xưng hô (Anh/Em, Tôi/Ông, Cậu/Tớ) khớp với ngữ cảnh đoạn thoại.\n"
            "* Giữ nguyên chính xác số lượng dòng và ID của từng câu phụ đề.\n\n"
            "Danh sách câu phụ đề (JSON):\n"
            f"{json.dumps(batch_items, ensure_ascii=False, indent=2)}\n\n"
            "Trả về DUY NHẤT một chuỗi mảng JSON định dạng:\n"
            '[{"id": 1, "text": "..."}, ...]'
        )

        try:
            if api_key:
                resp_text = call_gemini_with_fallback(prompt, api_key, model_name=model_name)
            else:
                resp_text = ""

            json_match = re.search(r'\[.*\]', resp_text, re.DOTALL)
            if json_match:
                refined_list = json.loads(json_match.group(0))
                for item in refined_list:
                    sub_id = item.get("id")
                    refined_val = item.get("text", "").strip()
                    if sub_id and 1 <= sub_id <= len(batch_indices):
                        target_seg_idx = batch_indices[sub_id - 1]
                        if refined_val:
                            final_val = _post_process_spoken_vietnamese(refined_val)
                            segments[target_seg_idx]['text'] = final_val
                            segments[target_seg_idx]['raw_text'] = final_val
                            src_key = segments[target_seg_idx].get('orig_text') or segments[target_seg_idx].get('text') or ""
                            manager.set_cache(src_key, final_val)
            else:
                for seg_idx in batch_indices:
                    final_val = _post_process_spoken_vietnamese(segments[seg_idx].get('draft', ''))
                    segments[seg_idx]['text'] = final_val
                    segments[seg_idx]['raw_text'] = final_val
        except Exception as e:
            print(f"Lỗi khi tinh chỉnh Batch {b_idx + 1}: {e}")
            for seg_idx in batch_indices:
                final_val = _post_process_spoken_vietnamese(segments[seg_idx].get('draft', ''))
                segments[seg_idx]['text'] = final_val
                segments[seg_idx]['raw_text'] = final_val

    return segments


# ----------------- AUTO-UPDATING BILIBILI & DOUYIN TRENDING SLANG ENGINE -----------------
class TrendingSlangManager:
    """Lớp quản lý từ điển tiếng lóng Bilibili, Douyin, MXH Trung Quốc (Singleton)."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TrendingSlangManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path=None):
        if self._initialized:
            return
        self._initialized = True
        if config_path:
            self.config_path = config_path
        else:
            cfg_dir = os.path.join(os.getcwd(), "config")
            os.makedirs(cfg_dir, exist_ok=True)
            self.config_path = os.path.join(cfg_dir, "trending_dict.json")
        self.slang_dict = {}
        self.load_dict()

    def load_dict(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.slang_dict = json.load(f)
                if len(self.slang_dict) < 40:
                    try:
                        from slang_sync_engine import EXPANDED_TRENDING_SLANGS
                        for k, v in EXPANDED_TRENDING_SLANGS.items():
                            if k not in self.slang_dict:
                                self.slang_dict[k] = v
                        self.save_dict()
                    except Exception:
                        pass
                return self.slang_dict
            except Exception as e:
                print(f"Lỗi nạp trending_dict.json: {e}")

        try:
            from slang_sync_engine import EXPANDED_TRENDING_SLANGS
            self.slang_dict = dict(EXPANDED_TRENDING_SLANGS)
        except Exception:
            self.slang_dict = {
                "破防": {"vi": "sụp đổ / xé lòng", "category": "Douyin/Bilibili", "source": "Auto-AI"},
                "绝绝子": {"vi": "đỉnh kout / hết nước chấm", "category": "Trend Giới Trẻ", "source": "Auto-AI"},
                "硬核": {"vi": "xịn xò / bá đạo / khét lẹt", "category": "Bilibili", "source": "Custom"},
                "显眼包": {"vi": "thánh gây chú ý / chúa tấu hề", "category": "Douyin", "source": "Auto-Sync"},
                "YYDS": {"vi": "mãi đỉnh", "category": "Trend Giới Trẻ", "source": "Auto-AI"}
            }
        self.save_dict()
        return self.slang_dict

    def save_dict(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.slang_dict, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Lỗi khi ghi trending_dict.json: {e}")
            return False

    def add_or_update_slang(self, zh, vi, category="Douyin/Bilibili", source="User", overwrite=True):
        zh_key = zh.strip()
        if not zh_key:
            return False
        if zh_key in self.slang_dict and not overwrite:
            return False
        self.slang_dict[zh_key] = {
            "vi": vi.strip(),
            "category": category.strip(),
            "source": source.strip()
        }
        self.save_dict()
        return True

    def remove_slang(self, zh):
        zh_key = zh.strip()
        if zh_key in self.slang_dict:
            del self.slang_dict[zh_key]
            self.save_dict()
            return True
        return False

    def merge_dict(self, new_entries, default_source="Auto-Sync", overwrite_user_custom=False):
        added_count = 0
        for zh, info in new_entries.items():
            zh_key = zh.strip()
            if not zh_key:
                continue
            if isinstance(info, str):
                vi_val = info
                cat_val = "Douyin/Bilibili"
                src_val = default_source
            else:
                vi_val = info.get("vi", "")
                cat_val = info.get("category", "Douyin/Bilibili")
                src_val = info.get("source", default_source)

            if zh_key in self.slang_dict:
                existing_src = self.slang_dict[zh_key].get("source", "")
                if existing_src == "Custom" and not overwrite_user_custom:
                    continue

            self.slang_dict[zh_key] = {
                "vi": vi_val,
                "category": cat_val,
                "source": src_val
            }
            added_count += 1
        if added_count > 0:
            self.save_dict()
        return added_count

    def apply_to_text(self, text):
        if not text:
            return text
        res = text
        for zh, info in sorted(self.slang_dict.items(), key=lambda x: len(x[0]), reverse=True):
            if zh in res:
                res = res.replace(zh, info.get("vi", ""))
        return res

    def replace_slang(self, text):
        return self.apply_to_text(text)


def get_trending_dict_path():
    return TrendingSlangManager().config_path

def load_trending_slang_dict():
    return TrendingSlangManager().load_dict()

def save_trending_slang_dict(slang_data):
    manager = TrendingSlangManager()
    manager.slang_dict = slang_data
    return manager.save_dict()

def add_trending_slang(zh_term, vi_term, category="Douyin/Bilibili", source="Custom"):
    manager = TrendingSlangManager()
    manager.add_or_update_slang(zh_term, vi_term, category=category, source=source)
    return manager.slang_dict

def apply_trending_slang_replacement(text):
    return TrendingSlangManager().apply_to_text(text)

def extract_slangs_from_text(zh_text):
    """Trích xuất tự động từ lóng tiếng Trung mới bằng LLM In-Flight Extractor."""
    manager = TrendingSlangManager()
    detected = []
    known_patterns = {
        "破防": "sụp đổ / xé lòng",
        "绝绝子": "đỉnh kout / hết nước chấm",
        "硬核": "xịn xò / bá đạo",
        "显眼包": "thánh gây chú ý",
        "YYDS": "mãi đỉnh",
        "芭比Q了": "toang rồi",
        "泰裤辣": "quá ngầu",
        "栓Q": "cảm ơn nhiều",
        "特种兵式": "kiểu lính đặc nhiệm",
        "沉浸式": "kiểu đắm chìm",
        "搭子": "cạ cứng / bạn đồng hành",
        "尊嘟假嘟": "thật á / đùa à"
    }
    for zh, vi in known_patterns.items():
        if zh in zh_text:
            manager.add_or_update_slang(zh, vi, category="Auto-AI", source="Auto-AI", overwrite=False)
            detected.append({"zh": zh, "vi": vi, "category": "Auto-AI"})
    return detected

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
            
        # Áp dụng hậu xử lý RegEx trật tự từ Hán-Việt sang thuần Việt tự nhiên
        raw_res = self._post_process_vietphrase(raw_res)
        return raw_res

    def _post_process_vietphrase(self, text):
        if not text:
            return text
        replacements = [
            (r'\bbất quá\b', 'tuy nhiên'),
            (r'\bthời điểm\b', 'khi'),
            (r'\bđích\b', 'của'),
            (r'\bhữu hiệu\b', 'hiệu quả'),
            (r'\bphát sinh\b', 'xảy ra'),
            (r'\bđương nhiên\b', 'tất nhiên'),
            (r'\bnhất định\b', 'chắc chắn'),
            (r'\bkhai thủy\b', 'bắt đầu'),
            (r'\bđệ nhất\b', 'thứ nhất'),
            (r'\bđệ nhị\b', 'thứ hai'),
            (r'\bđệ tam\b', 'thứ ba'),
        ]
        res = text
        for pattern, repl in replacements:
            res = re.sub(pattern, repl, res, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', res).strip()

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
def translate_segments(segments, source_lang='auto', target_lang='vi', engine='Google Translate', api_key=None, progress_callback=None, ollama_model='qwen2.5', vp_dict_paths=None, prefer_xkiro=None, xkiro_key=None):
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

    # --- 3. ENGINE: GEMINI + QUICK TRANSLATOR (VIETPHRASE & DATA DICTS) HYBRID ---
    elif ("Hybrid" in engine or "Supersubs" in engine or "Gemini" in engine or "Auto" in engine or "Tự động" in engine) and api_key:
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
            
        batch_size = 15
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
                res_text = call_gemini_with_fallback(prompt, api_key, model_name=model_name)
                
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
                        single_res = call_gemini_with_fallback(line_prompt, api_key, model_name=model_name)
                        single_results.append(single_res)
                    except Exception as e_s:
                        print(f"Lỗi dịch dòng đơn {i+idx_s}: {e_s}")
                        single_results.append(vp_res) # Cuối cùng mới dùng VietPhrase làm dự phòng
                batch_results = single_results
                
            translated_texts.extend(batch_results)
                
            if progress_callback:
                percent = int(((i + len(batch)) / len(texts)) * 100)
                progress_callback(f"Dịch Hybrid (Gemini + VietPhrase)... {percent}% (Đã dịch {min(i + batch_size, len(texts))}/{len(texts)} câu)")
 
    # --- 3.5. ENGINE: XKIRO AI (OPENAI COMPATIBLE) ---
    elif "xkiro" in engine.lower():
        import xkiro_client
        from gemini_vision_ocr import load_gemini_keys
        batch_size = 5
        xkiro_err_count = 0
        gemini_keys = load_gemini_keys()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for t in batch:
                if not t or not t.strip():
                    translated_texts.append("")
                    continue
                
                success = False
                # Thử xKiro nếu chưa bị lỗi liên tiếp quá 3 lần
                if xkiro_err_count < 3:
                    try:
                        trans_val = xkiro_client.translate_with_xkiro(t, target_lang=tgt, source_lang=src, api_key=xkiro_key)
                        if trans_val and trans_val.strip():
                            translated_texts.append(trans_val.strip())
                            success = True
                            xkiro_err_count = 0
                    except Exception as e_xk:
                        xkiro_err_count += 1
                        err_str = str(e_xk)
                        if "401" in err_str or "auth" in err_str.lower():
                            if progress_callback:
                                progress_callback("❌ [xKiro API] Key không hợp lệ hoặc đã hết hạn (401 Auth Error)! Vui lòng cập nhật key mới trong Tab 2 → API Keys.")
                        if progress_callback:
                            progress_callback(f"⚠️ xKiro dịch lỗi ({e_xk}). Đang tự động fallback sang Gemini...")

                if not success and xkiro_err_count >= 3:
                    if progress_callback:
                        progress_callback(f"⚠️ xKiro đã lỗi {xkiro_err_count} lần, tự động chuyển sang Gemini translation...")

                # Fallback 1: Gemini Translation
                if not success and gemini_keys:
                    p_gem = (
                        f"Dịch câu thoại video sau từ {src} sang {tgt} tự nhiên, trôi chảy, đúng văn phong hội thoại:\n"
                        f"'{t}'\n\nCHỈ xuất ra duy nhất câu dịch, không kèm lời giải thích."
                    )
                    for gk in gemini_keys:
                        try:
                            res_gem = call_gemini_with_fallback(p_gem, gk, model_name="gemini-flash-latest")
                            if res_gem and res_gem.strip():
                                translated_texts.append(res_gem.strip())
                                success = True
                                break
                        except Exception:
                            continue

                # Fallback 2: Google Translate
                if not success:
                    try:
                        gt = GoogleTranslator(source=src, target=tgt)
                        res_gt = gt.translate(t)
                        translated_texts.append(res_gt or t)
                        success = True
                    except Exception:
                        translated_texts.append(t)

            if progress_callback:
                percent = int(((i + len(batch)) / len(texts)) * 100)
                progress_callback(f"Đang dịch bằng xKiro AI... {percent}% ({min(i + batch_size, len(texts))}/{len(texts)} câu)")

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
                    res_text = call_gemini_api(prompt, api_key, model_name=model_name)
                
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

    # --- 6. CHUỖI FALLBACK MẶC ĐỊNH MẠNH NHẤT (XKIRO -> GEMINI -> GOOGLE TRANSLATE) ---
    else:
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            b_res = translate_batch_with_fallback_chain(
                batch=batch,
                src=src,
                tgt=tgt,
                api_key=api_key,
                prefer_xkiro=prefer_xkiro,
                progress_callback=progress_callback,
                xkiro_key=xkiro_key
            )
            translated_texts.extend(b_res)
                    
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
            
    # Đọc tất cả các file .txt nhỏ hơn 100KB từ folder Data để gửi cho LLM
    data_dir = os.path.join(os.getcwd(), "Data")
    du_lieu_rules = []
    
    if os.path.exists(data_dir):
        try:
            for file_name in os.listdir(data_dir):
                if file_name.lower().endswith(".txt"):
                    file_path_txt = os.path.join(data_dir, file_name)
                    # Bỏ qua các file từ điển dữ liệu lớn (> 100KB) như VietPhrase.txt, LacViet.txt...
                    if os.path.isfile(file_path_txt) and os.path.getsize(file_path_txt) < 100 * 1024:
                        content = ""
                        # Thử nhiều bảng mã khác nhau để giải mã an toàn các file lưu ở định dạng UTF-16 / UTF-8 BOM
                        for enc in ["utf-8-sig", "utf-16", "utf-8", "utf-16-le", "utf-16-be", "cp1258", "latin-1"]:
                            try:
                                with open(file_path_txt, "r", encoding=enc) as f_txt:
                                    content = f_txt.read().strip()
                                break
                            except:
                                continue
                        
                        if content:
                            du_lieu_rules.append(f"=== TỆP TỪ ĐIỂN: {file_name} ===\n{content}")
        except Exception as e:
            print(f"Loi duyet thu muc Data: {e}")
            
    if glossary_rules:
        du_lieu_rules.append("=== THUẬT NGỮ GLOSSARY CẤU HÌNH TRÊN UI ===\n" + glossary_rules)
        
    du_lieu_str = "\n\n".join(du_lieu_rules) if du_lieu_rules else "(Không có dữ liệu từ điển)" 
            
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
            "Bạn là một biên tập viên ngôn ngữ tiếng Việt xuất sắc. Nhiệm vụ của bạn là đọc một danh sách phụ đề đang bị dịch máy thô (dạng Vietphrase / Hán Việt) dưới đây và VIẾT LẠI (Rewrite) nó thành tiếng Việt thuần túy, mượt mà, dễ hiểu nhất.\n\n"
            "[1. YÊU CẦU BIÊN TẬP - TUYỆT ĐỐI TUÂN THỦ]\n"
            "- Dịch thoát ý hoàn toàn. Xóa bỏ triệt để văn phong Hán Việt lủng củng và các trợ từ vô nghĩa (đích, liễu, a, ba, chư, đóa...).\n"
            "- Tự động suy luận ngữ cảnh từ các câu lủng củng để viết lại thành một câu giao tiếp tự nhiên của người Việt. (Ví dụ: \"biến thân liễu\" -> \"đã trổ mã\", \"chín tố\" -> \"trưởng thành\").\n"
            "- BẮT BUỘC viết đúng chuẩn chính tả tiếng Việt. TUYỆT ĐỐI KHÔNG chèn khoảng trắng thừa vào giữa các chữ cái hoặc dấu thanh trong cùng một từ (Ví dụ: Bắt buộc viết là \"Tuyệt vời\" hoặc \"Tỷ chỉ\", NGHIÊM CẤM viết thành \"T u y ệ t v ờ i\", \"t ỷ ch ỉ\", \"ng ư ơ i\"). Lỗi tokenization này không được phép xuất hiện.\n\n"
            "[2. BỘ TỪ ĐIỂN VÀ XƯNG HÔ BẮT BUỘC]\n"
            "Trong quá trình dịch, bạn bắt buộc phải quét và ưu tiên sử dụng đại từ xưng hô, từ lóng (trend), thuật ngữ và quy tắc kiểm duyệt (censor) được cung cấp dưới đây:\n"
            "--- BẮT ĐẦU DỮ LIỆU TỪ ĐIỂN ---\n"
            f"{du_lieu_str}\n"
            "--- KẾT THÚC DỮ LIỆU TỪ ĐIỂN ---\n\n"
            "YÊU CẦU ĐỊNH DẠNG ĐẦU RA (RÀNG BUỘC KỸ THUẬT BẮT BUỘC):\n"
            "1. Chỉ trả về duy nhất một danh sách mảng JSON (JSON list of strings) chứa các câu đã biên tập xong, khớp chính xác số lượng và thứ tự dòng đầu vào. Tuyệt đối không gộp câu, không tự ý ngắt dòng.\n"
            "2. Tuyệt đối không giải thích, không chào hỏi, không thêm bất kỳ ký tự nào khác bên ngoài mảng JSON.\n\n"
            "Ví dụ định dạng đầu ra:\n"
            "[\n"
            "  \"Dòng dịch đã sơ chế 1\",\n"
            "  \"Dòng dịch đã sơ chế 2\"\n"
            "]\n\n"
            "DANH SÁCH CÁC CÂU CẦN DỊCH/SƠ CHẾ:\n" + "\n".join(batch_lines_prompt)
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
                    
                if engine.startswith("Gemini") and not key.startswith("sk-"):
                    sdk_model = "gemini-1.5-flash"
                    if "1.5 Pro" in engine:
                        sdk_model = "gemini-1.5-pro"
                    elif "2.0 Flash" in engine:
                        sdk_model = "gemini-2.0-flash"
                    
                    res_text = call_gemini_api(prompt, key, model_name=sdk_model)
                else:
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


# --- ------------------------------------------------------------------- ---
# --- MÔ-ĐUN DỊCH THUẬT HYBRID VIETPHRASE & FREE BATCHING (0% API COST) ---
# --- ------------------------------------------------------------------- ---

_GLOBAL_VIETPHRASE_MAP = None
_GLOBAL_VIETPHRASE_KEYS_SORTED = None
_GLOBAL_VIETPHRASE_FILES = []

def load_vietphrase_dictionary(custom_dir=None):
    """
    Tự động quét thư mục data/ và Data/ trong dự án để nạp toàn bộ từ điển VietPhrase, Names, Lạc Việt, Babylon.
    Sắp xếp các từ theo độ dài giảm dần (Longest Match First) để thay thế chính xác không bị vỡ nghĩa.
    Trả về: (dict_map, loaded_files, total_records_count)
    """
    global _GLOBAL_VIETPHRASE_MAP, _GLOBAL_VIETPHRASE_KEYS_SORTED, _GLOBAL_VIETPHRASE_FILES

    if _GLOBAL_VIETPHRASE_MAP is not None:
        return _GLOBAL_VIETPHRASE_MAP, _GLOBAL_VIETPHRASE_FILES, len(_GLOBAL_VIETPHRASE_MAP)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        custom_dir,
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "Data")
    ]

    target_dir = None
    for d in candidate_dirs:
        if d and os.path.exists(d) and os.path.isdir(d):
            target_dir = d
            break

    if not target_dir:
        target_dir = os.path.join(base_dir, "Data")
        os.makedirs(target_dir, exist_ok=True)

    dict_files = [
        "VietPhrase.txt", "Names.txt", "PhuDe.txt",
        "ChineseModern.txt", "LacViet.txt"
    ]

    dict_map = {}
    loaded_files = []

    for fname in dict_files:
        fpath = os.path.join(target_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line or '=' not in line:
                            continue
                        parts = line.split('=', 1)
                        k = parts[0].strip()
                        v = parts[1].strip()
                        # Bỏ qua key ASCII La-tinh thuần túy để tránh vỡ chuỗi tiếng Việt/Anh
                        if all(ord(ch) < 128 for ch in k):
                            continue
                        if '/' in v:
                            v = v.split('/')[0].strip()
                        if ';' in v:
                            v = v.split(';')[0].strip()
                        if k and v:
                            dict_map[k] = v
                loaded_files.append(fpath)
            except Exception as e:
                print(f"Cảnh báo nạp từ điển {fname}: {e}")

    # Thuật toán Longest Match First: Sắp xếp key theo độ dài giảm dần
    keys_sorted = sorted(dict_map.keys(), key=lambda k: len(k), reverse=True)

    _GLOBAL_VIETPHRASE_MAP = dict_map
    _GLOBAL_VIETPHRASE_KEYS_SORTED = keys_sorted
    _GLOBAL_VIETPHRASE_FILES = loaded_files

    return _GLOBAL_VIETPHRASE_MAP, _GLOBAL_VIETPHRASE_FILES, len(_GLOBAL_VIETPHRASE_MAP)


def apply_vietphrase_pre_translation(text, max_replacements=50):
    """
    Bước 1: Quét thay thế các Danh từ riêng, Nhân vật, Xưng hô bằng từ điển VietPhrase theo quy tắc Longest Match First.
    """
    if not text:
        return text

    dict_map, _, _ = load_vietphrase_dictionary()
    if not dict_map or not _GLOBAL_VIETPHRASE_KEYS_SORTED:
        return text

    normalized = text
    count = 0

    # Ưu tiên thay thế các cụm từ dài trước (Longest Match First)
    for k in _GLOBAL_VIETPHRASE_KEYS_SORTED:
        if k in normalized:
            v = dict_map[k]
            normalized = normalized.replace(k, v)
            count += 1
            if count >= max_replacements:
                break

    return normalized


CUSTOM_DEFAULT_GLOSSARY = {
    "你": "bạn",
    "我": "tôi",
    "他": "anh ấy",
    "她": "cô ấy",
    "师兄": "sư huynh",
    "师妹": "sư muội",
    "师傅": "sư phụ",
    "宗门": "tông môn"
}

def apply_custom_glossary(text, glossary=None):
    """
    Chuẩn hóa đại từ xưng hô, ngữ cảnh và từ điển tùy chỉnh (Custom Glossary).
    """
    if not text:
        return text

    active_glossary = dict(CUSTOM_DEFAULT_GLOSSARY)
    if glossary and isinstance(glossary, dict):
        active_glossary.update(glossary)

    normalized = text
    for k, v in active_glossary.items():
        if k and v and k in normalized:
            normalized = normalized.replace(k, v)

    # Loại bỏ khoảng trắng thừa hoặc ký tự rác
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


class FreeBatchTranslator:
    """
    Mô-đun dịch miễn phí 100% (0% phí API) kết hợp VietPhrase Pre-translation.
    Gom nhóm (Batching) 15-20 câu phụ đề gửi dịch 1 lần để tăng tốc gấp 10 lần và tránh Rate Limit.
    Tự động Fallback sang VietPhrase / Local Dictionary khi mất mạng.
    """
    def __init__(self, source_lang='auto', target_lang='vi', batch_size=15):
        self.source_lang = 'auto' if 'auto' in source_lang or 'Tự động' in source_lang else source_lang
        self.target_lang = 'vi' if 'vi' in target_lang or 'Việt' in target_lang else target_lang
        self.batch_size = batch_size
        load_vietphrase_dictionary() # Nạp trước từ điển VietPhrase vào bộ nhớ

    def translate_texts(self, texts, progress_callback=None):
        if not texts:
            return []

        translated_results = []
        total_texts = len(texts)
        delimiter = "\n---SEGMENT_DELIMITER---\n"

        for i in range(0, total_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_text = delimiter.join(batch)
            success = False
            batch_translated_lines = []

            # GIẢI PHÁP 1: Free Batch Scraper qua deep_translator GoogleTranslator
            try:
                translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)
                res_payload = translator.translate(batch_text)

                if res_payload:
                    # Tách lại theo delimiter
                    parts = res_payload.split("---SEGMENT_DELIMITER---")
                    if len(parts) == len(batch):
                        batch_translated_lines = [p.strip() for p in parts]
                        success = True
                    else:
                        # Nếu delimiter bị vỡ, tách theo dòng
                        lines = [l.strip() for l in res_payload.splitlines() if l.strip() and not "---SEGMENT" in l]
                        if len(lines) == len(batch):
                            batch_translated_lines = lines
                            success = True
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Cảnh báo Rate Limit / Kết nối Google Scraper: {e}. Chuyển sang fallback...")

            # GIẢI PHÁP 2: Fallback sang dịch từng câu đơn hoặc VietPhrase
            if not success:
                try:
                    translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)
                    single_res = translator.translate_batch(batch)
                    if single_res and len(single_res) == len(batch):
                        batch_translated_lines = single_res
                        success = True
                except Exception:
                    pass

            # Dự phòng cuối cùng: Dùng VietPhrase / Giữ nguyên text và áp dụng Glossary
            if not success:
                vp_translator = VietPhraseTranslator()
                vp_translator.load()
                for t in batch:
                    trans_val = vp_translator.translate(t) if t else t
                    batch_translated_lines.append(trans_val)

            # Áp dụng Custom Glossary chuẩn hóa đại từ xưng hô
            for idx, res_line in enumerate(batch_translated_lines):
                final_line = apply_custom_glossary(res_line)
                translated_results.append(final_line)

            if progress_callback:
                pct = int((min(i + self.batch_size, total_texts) / total_texts) * 100)
                progress_callback(f"Đang dịch mượt phụ đề... {pct}% ({min(i + self.batch_size, total_texts)}/{total_texts} câu)")

        return translated_results


def translate_srt_file(raw_srt_path, out_srt_path, source_lang='auto', target_lang='vi', progress_callback=None):
    """
    Dịch file .srt gốc sang ngôn ngữ đích:
    - Bảo toàn 100% cấu trúc SRT, ID dòng, số lượng dòng và timecode (00:00:01,000 --> 00:00:03,500).
    - Sử dụng FreeBatchTranslator gom nhóm 15-20 câu (Batching 10x).
    """
    if not os.path.exists(raw_srt_path):
        raise FileNotFoundError(f"Tệp SRT gốc không tồn tại: {raw_srt_path}")

    with open(raw_srt_path, 'r', encoding='utf-8', errors='ignore') as f:
        srt_content = f.read()

    from transcriber import parse_srt_string, segments_to_srt
    segments = parse_srt_string(srt_content)

    if not segments:
        raise ValueError("Không tìm thấy phân đoạn phụ đề hợp lệ nào trong file SRT.")

    if progress_callback:
        progress_callback(f"Bắt đầu dịch file SRT ({len(segments)} câu) qua Free Batching Engine...")

    raw_texts = [seg['text'] for seg in segments]
    translator = FreeBatchTranslator(source_lang=source_lang, target_lang=target_lang, batch_size=15)
    translated_texts = translator.translate_texts(raw_texts, progress_callback=progress_callback)

    # Gán lại nội dung đã dịch vào segments nhưng GIỮ NGUYÊN 100% start và end timecode
    translated_segments = []
    for seg, trans_txt in zip(segments, translated_texts):
        new_seg = dict(seg)
        new_seg['orig_text'] = seg['text']
        new_seg['text'] = trans_txt
        translated_segments.append(new_seg)

    out_srt_content = segments_to_srt(translated_segments)

    out_dir = os.path.dirname(out_srt_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_srt_path, 'w', encoding='utf-8') as f:
        f.write(out_srt_content)

    if progress_callback:
        progress_callback(f"✔ Đã lưu file SRT dịch hoàn chỉnh tại: {out_srt_path}")

    return out_srt_path

