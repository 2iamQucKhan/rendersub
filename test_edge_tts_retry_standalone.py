import os
import sys

# Đảm bảo UTF-8 cho Windows console
sys.stdout.reconfigure(encoding='utf-8')

def test_tts_generation():
    print("=" * 70)
    print(" 🧪 KIỂM THỬ TÍNH NĂNG RETRY & FALLBACK CỦA EDGE-TTS & GOOGLE TTS")
    print("=" * 70)

    import dubber
    output_dir = os.path.join("output", "test_tts_demo")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Thử sinh giọng đọc Edge-TTS chuẩn
    test_file_edge = os.path.join(output_dir, "test_edge.mp3")
    print("1. Kiểm thử Edge-TTS với câu mẫu...")
    success_edge = dubber.generate_tts("Xin chào, đây là kiểm thử Edge-TTS tự động.", "vi-VN-HoaiMyNeural", test_file_edge)
    if success_edge and os.path.exists(test_file_edge):
        print(f"✅ Sinh Edge-TTS THÀNH CÔNG! Dung lượng file: {os.path.getsize(test_file_edge)} bytes.")
    else:
        print("❌ Sinh Edge-TTS thất bại.")

    # 2. Thử trực tiếp Google TTS fallback
    test_file_google = os.path.join(output_dir, "test_google.mp3")
    print("\n2. Kiểm thử Google TTS Fallback...")
    success_google = dubber.download_google_tts("Xin chào, đây là kiểm thử Google TTS dự phòng.", test_file_google)
    if success_google and os.path.exists(test_file_google):
        print(f"🟢 Sinh Google TTS Fallback THÀNH CÔNG! Dung lượng file: {os.path.getsize(test_file_google)} bytes.")
    else:
        print("❌ Sinh Google TTS Fallback thất bại.")

    return success_edge and success_google

if __name__ == "__main__":
    test_tts_generation()
