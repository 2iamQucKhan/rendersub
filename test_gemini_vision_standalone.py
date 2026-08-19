import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Đảm bảo in tiếng Việt không lỗi mã hóa trên Windows Console
sys.stdout.reconfigure(encoding='utf-8')

def test_gemini_vision_api():
    print("=" * 70)
    print(" 🧪 KIỂM THỬ ĐỘC LẬP GEMINI VISION API (SDK MOI google.genai)")
    print("=" * 70)

    # 1. Đọc và kiểm tra Gemini API Key từ config/api_keys.json hoặc biến môi trường
    key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "api_keys.json")
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key and os.path.exists(key_path):
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                keys = data.get("gemini_keys", [])
                if keys:
                    api_key = keys[0].strip()
        except Exception as e:
            print(f"❌ Lỗi đọc file config/api_keys.json: {e}")

    print(f"\n1. API Key đọc từ hệ thống: '{api_key[:10]}...' (Độ dài: {len(api_key)})")
    
    # Sơ bộ kiểm tra độ dài và định dạng cơ bản của API Key
    if not api_key or len(api_key) < 20 or " " in api_key:
        print("\n❌ KIỂM TRA ĐỊNH DẠNG KEY SƠ BỘ: THẤT BẠI!")
        print("⚠️ API Key rỗng, quá ngắn (< 20 ký tự) hoặc chứa khoảng trắng.")
        print("💡 HƯỚNG DẪN: Hãy mở file 'config/api_keys.json' và nhập Gemini API Key thật của bạn.")
        return False

    print("✅ KIỂM TRA SƠ BỘ: HỢP LỆ! (Tiến hành gọi API thực tế...)")

    # 2. Tạo hình ảnh mẫu thử nghiệm chứa chữ tiếng Trung / tiếng Anh
    print("\n2. Tạo hình ảnh thử nghiệm...")
    img = Image.new('RGB', (640, 360), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 200, 590, 300], fill=(0, 0, 0))
    draw.text((80, 230), "Hello World - Testing Gemini 2.0 Flash Vision API", fill=(255, 255, 255))
    print("✅ Đã tạo ảnh mẫu 640x360 px thành công.")

    # 3. Kết nối Gemini API với SDK mới google.genai
    print("\n3. Kết nối đến Gemini API qua SDK google.genai...")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        prompt = (
            "Hãy phân tích ảnh này: Đọc nội dung chữ trong ảnh, dịch sang tiếng Việt tự nhiên "
            "và trả về Bounding Box [ymin, xmin, ymax, xmax] tỷ lệ 0-1000 dưới dạng JSON ARRAY."
        )

        candidate_models = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.7-flash", "gemini-flash-lite-latest"]
        last_err = None
        for model_name in candidate_models:
            print(f"📡 Đang thử kết nối Model: '{model_name}'...")
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, img]
                )
                if response and hasattr(response, 'text') and response.text:
                    res_text = response.text.strip()
                    print("\n" + "=" * 70)
                    print(f"🎉 PHẢN HỒI THÀNH CÔNG TỪ GEMINI VISION API (Model: {model_name}):")
                    print("=" * 70)
                    print(res_text)
                    print("=" * 70)
                    return True
            except Exception as e:
                last_err = e
                print(f"   ⚠️ Model '{model_name}' báo lỗi: {e}")

        if last_err:
            raise last_err

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ GEMINI API GẶP LỖI THỰC TẾ:")
        print(f"   Chi tiết lỗi: {e}")
        print("=" * 70)
        
        # Log lỗi vào file riêng
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "gemini_api_errors.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[Test Standalone Error] {e}\n")
        print(f"📝 Đã ghi chi tiết lỗi vào: {log_file}")
        return False

if __name__ == "__main__":
    test_gemini_vision_api()
