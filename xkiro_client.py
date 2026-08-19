import os
import sys
import json

# Đảm bảo in UTF-8 trên Windows Console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

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

def log_xkiro_error(msg):
    """
    Ghi log lỗi xKiro API ra console và file logs/xkiro_api_errors.log
    """
    print(f"⚠️ [xKiro API] {msg}")
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "xkiro_api_errors.log")
        rotate_log_file_if_needed(log_file)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")
    except Exception:
        pass

def is_xkiro_key(key: str) -> bool:
    """
    Kiểm tra xem key có phải định dạng hợp lệ của xKiro (hoặc OpenAI compatible) hay không.
    Key xKiro bắt buộc phải bắt đầu bằng 'sk-' (ví dụ: sk-xt-...) và có độ dài >= 15 ký tự.
    """
    if not key or not isinstance(key, str):
        return False
    k = key.strip()
    return k.startswith("sk-") and len(k) >= 15 and " " not in k

def load_xkiro_keys():
    """
    Đọc mảng xkiro_keys từ file config/api_keys.json hoặc biến môi trường XKIRO_API_KEY.
    Chỉ trả về danh sách các key hợp lệ bắt đầu bằng 'sk-' (loại bỏ nhầm lẫn với Gemini key).
    """
    keys = []
    env_key = os.environ.get("XKIRO_API_KEY", "").strip()
    if env_key and is_xkiro_key(env_key):
        keys.append(env_key)

    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "api_keys.json"),
        os.path.join(os.getcwd(), "config", "api_keys.json"),
        "config/api_keys.json"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    x_keys = data.get("xkiro_keys", [])
                    if isinstance(x_keys, str):
                        x_keys = [k.strip() for k in x_keys.split(",") if k.strip()]
                    for k in x_keys:
                        if k and isinstance(k, str):
                            k_clean = k.strip()
                            if is_xkiro_key(k_clean):
                                if k_clean not in keys:
                                    keys.append(k_clean)
                            else:
                                if k_clean.startswith("AQ.") or k_clean.startswith("AIzaSy"):
                                    log_xkiro_error(f"⚠️ Bỏ qua key '{k_clean[:10]}...' trong xkiro_keys vì đây là Gemini key, không phải xKiro key (yêu cầu prefix 'sk-').")
            except Exception as e:
                log_xkiro_error(f"Lỗi khi đọc file config {p}: {e}")

    return keys

def get_xkiro_openai_client(api_key=None):
    """
    Khởi tạo OpenAI client trỏ về endpoint https://api.xkiro.com/v1
    Tự động lọc và đảm bảo chỉ dùng key định dạng 'sk-'.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("Chưa cài đặt thư viện 'openai'. Hãy chạy 'pip install openai'.")

    key_to_use = None
    if api_key and is_xkiro_key(api_key):
        key_to_use = api_key.strip()
    else:
        loaded = load_xkiro_keys()
        if loaded:
            key_to_use = loaded[0]

    if not key_to_use or not is_xkiro_key(key_to_use):
        raise ValueError("Thiếu xKiro API Key hợp lệ trong config/api_keys.json ('xkiro_keys' phải bắt đầu bằng 'sk-').")

    return OpenAI(
        base_url="https://api.xkiro.com/v1",
        api_key=key_to_use
    )

def test_xkiro_key_status(api_key=None):
    """
    Kiểm tra xem xKiro API Key có hợp lệ và hoạt động hay không.
    Trả về (is_ok: bool, message: str)
    """
    key_to_test = None
    if api_key and is_xkiro_key(api_key):
        key_to_test = api_key.strip()
    else:
        loaded = load_xkiro_keys()
        if loaded:
            key_to_test = loaded[0]

    if not key_to_test:
        if api_key and (api_key.startswith("AQ.") or api_key.startswith("AIzaSy")):
            return False, f"❌ Key '{api_key[:10]}...' là Gemini key (không thể dùng cho xKiro). xKiro key phải bắt đầu bằng 'sk-xt-'."
        return False, "❌ Chưa nhập xKiro API Key hợp lệ trong Tab 2 → API Keys (yêu cầu định dạng bắt đầu bằng 'sk-')."

    try:
        client = get_xkiro_openai_client(key_to_test)
        res = client.chat.completions.create(
            model="deepseek/deepseek-v4-pro",
            messages=[{"role": "user", "content": "Hello, reply with 1 word OK"}],
            max_tokens=10,
            temperature=0.1
        )
        if res and res.choices and res.choices[0].message and res.choices[0].message.content:
            txt = res.choices[0].message.content.strip()
            return True, f"✅ xKiro Key hợp lệ! (Phản hồi: {txt[:30]})"
        return False, "❌ xKiro không trả về nội dung hợp lệ."
    except Exception as e:
        err_str = str(e)
        if "401" in err_str or "authentication" in err_str.lower() or "invalid" in err_str.lower() or "disabled" in err_str.lower():
            log_xkiro_error(f"❌ xKiro API Key '{key_to_test[:10]}...' bị từ chối (401 Auth Error): {e}")
            return False, "❌ xKiro API Key không hợp lệ hoặc đã hết hạn (401 Auth Error)! Vui lòng cập nhật key mới trong Tab 2 → API Keys."
        log_xkiro_error(f"Lỗi test xKiro key: {e}")
        return False, f"❌ Lỗi kết nối xKiro API: {e}"

def translate_with_xkiro(text, target_lang="vi", source_lang="auto", model="deepseek/deepseek-v4-pro", api_key=None, prompt_template=None, max_tokens=1000, temperature=0.3, context="", tone="conversational"):
    """
    Dịch 1 đoạn văn bản thoại video bằng xKiro AI (Model miễn phí $0: deepseek/deepseek-v4-pro).
    Hỗ trợ custom prompt template và các tham số sinh ngôn ngữ.
    """
    text_clean = (text or "").strip()
    if not text_clean:
        return ""

    keys_to_try = []
    if api_key and is_xkiro_key(api_key):
        keys_to_try.append(api_key.strip())
    for k in load_xkiro_keys():
        if k not in keys_to_try:
            keys_to_try.append(k)

    if not keys_to_try:
        raise ValueError("Thiếu xKiro API Key hợp lệ trong config/api_keys.json (Key phải bắt đầu bằng 'sk-').")

    if not prompt_template:
        p_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "xkiro_prompt_template.json")
        if os.path.exists(p_path):
            try:
                with open(p_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    prompt_template = cfg.get("template")
                    max_tokens = cfg.get("max_tokens", max_tokens)
                    temperature = cfg.get("temperature", temperature)
            except Exception:
                pass

    if prompt_template and "{text}" in prompt_template:
        try:
            prompt = prompt_template.format(
                source_lang=source_lang,
                target_lang=target_lang,
                text=text_clean,
                context=context or "Hội thoại video",
                tone=tone or "conversational"
            )
        except Exception:
            prompt = prompt_template.replace("{text}", text_clean).replace("{source_lang}", str(source_lang)).replace("{target_lang}", str(target_lang))
    else:
        prompt = (
            f"Bạn là một dịch giả phim chuyên nghiệp dịch từ {source_lang} sang {target_lang}.\n"
            f"Nhiệm vụ: Dịch câu thoại video sau đây tự nhiên, trôi chảy, đúng văn phong hội thoại phim ảnh hàng ngày:\n"
            f"'{text_clean}'\n\n"
            f"CHỈ xuất ra duy nhất câu dịch {target_lang} cuối cùng, KHÔNG kèm lời giải thích hay ký tự thừa nào khác."
        )

    last_err = None
    for k_curr in keys_to_try:
        try:
            client = get_xkiro_openai_client(k_curr)
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            if res and res.choices and res.choices[0].message and res.choices[0].message.content:
                return res.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "401" in err_str or "auth" in err_str.lower():
                log_xkiro_error(f"❌ xKiro Key '{k_curr[:10]}...' lỗi 401 Auth: Key không hợp lệ hoặc hết hạn. Đang thử key tiếp theo...")
            else:
                log_xkiro_error(f"Lỗi dịch thuật xKiro ({model}) với Key '{k_curr[:10]}...': {e}")
            continue

    if last_err:
        raise last_err
    return text_clean
