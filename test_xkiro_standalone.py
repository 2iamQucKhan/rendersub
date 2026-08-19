import os
import sys
import json

# Đảm bảo in UTF-8 trên Windows Console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def test_xkiro_standalone():
    print("=" * 75)
    print(" 🧪 KIỂM THỬ ĐỘC LẬP XKIRO AI TRANSLATION ENGINE (deepseek/deepseek-v4-pro)")
    print("=" * 75)

    # 1. Đọc xKiro API Key từ config/api_keys.json hoặc biến môi trường
    key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "api_keys.json")
    api_key = os.environ.get("XKIRO_API_KEY", "").strip()

    if not api_key and os.path.exists(key_path):
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                keys = data.get("xkiro_keys", [])
                if keys and isinstance(keys, list):
                    api_key = str(keys[0]).strip()
        except Exception as e:
            print(f"❌ Lỗi đọc file config/api_keys.json: {e}")

    print(f"\n1. API Key xKiro đọc từ hệ thống: '{api_key[:10] if api_key else ''}...' (Độ dài: {len(api_key)})")

    if not api_key or len(api_key) < 10 or " " in api_key:
        print("\n❌ KIỂM TRA SƠ BỘ KEY: THẤT BẠI - CHƯA CÓ XKIRO API KEY HỢP LỆ!")
        print("\n📌 HƯỚNG DẪN TẠO VÀ NHẬP KEY XKIRO (MIỄN PHÍ):")
        print("   Bước 1: Truy cập trang chủ xKiro tại: https://api.xkiro.com (hoặc xkiro.com)")
        print("   Bước 2: Đăng ký/Đăng nhập tài khoản và tạo API Key trong mục 'API Keys'.")
        print("   Bước 3: Mở file 'config/api_keys.json' trong project và thêm key vào mảng 'xkiro_keys':")
        print('   {\n     "gemini_keys": [...],\n     "xkiro_keys": [\n       "xk-YourXKiroApiKeyHere..."\n     ]\n   }')
        return False

    print("✅ KIỂM TRA SƠ BỘ KEY: HỢP LỆ! (Đang gửi request dịch thử nghiệm tới https://api.xkiro.com/v1...)")

    import xkiro_client

    # 2. Kiểm thử dịch câu mẫu tiếng Trung & Tiếng Anh
    test_sentences = [
        "当你从空调房出来，外面天气太热了！",
        "Hello everyone, welcome back to another video testing xKiro translation."
    ]

    print("\n2. Tiến hành dịch thử nghiệm với Model 'deepseek/deepseek-v4-pro':")
    for idx, sentence in enumerate(test_sentences, 1):
        print(f"\n   [Mẫu {idx}] Dòng gốc: '{sentence}'")
        try:
            translated = xkiro_client.translate_with_xkiro(sentence, target_lang="vi", api_key=api_key)
            print(f"   🟢 DỊCH THÀNH CÔNG: '{translated}'")
        except Exception as e:
            print(f"   ❌ LỖI DỊCH THUẬT: {e}")
            return False

    print("\n" + "=" * 75)
    print("🎉 TẤT CẢ PHẢN HỒI TỪ XKIRO AI DỊCH THUẬT ĐỀU THÀNH CÔNG RỰC RỠ!")
    print("=" * 75)
    return True

if __name__ == "__main__":
    test_xkiro_standalone()
