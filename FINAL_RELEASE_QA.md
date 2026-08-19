# FINAL RELEASE GATE & ADVERSARIAL QA AUDIT REPORT
**Target Commit:** `c72b5a8`  
**Repository:** `2iamQucKhan/rendersub`  
**Platform:** Windows 11 (AMD64 / x64)

---

## 1. PHẦN 1 — KIỂM TRA TRẠNG THÁI GIT & BẢO MẬT (PHASE 1)
- **Git Status:** Working tree clean (`c72b5a8` committed & pushed to `origin/main`).
- **Commit History:**
  - `c72b5a8`: `feat(packaging): add standalone Windows distribution`
  - `79caa7c`: `docs(qa): add release and clean-machine packaging QA report`
- **Quét Khóa & Bí Mật (Secret Scan):**
  - Quét chuỗi `AIzaSy`: Chỉ xuất hiện trong placeholder UI (`Dán Gemini API Key (AIzaSy...)...`) và bộ unit test bảo mật. Không có key thật.
  - Quét chuỗi `sk-`: Chỉ xuất hiện trong logic validate format xKiro API Key (`sk-xt-...`). Không có key thật.
  - Quét hardcoded developer paths: Không phát hiện đường dẫn cá nhân trong source code. Toàn bộ đường dẫn phụ thuộc được chuyển sang [`resource_utils.py`](resource_utils.py).

---

## 2. PHẦN 2 — KIỂM TRA FILE THỰC THI CHÍNH (PHASE 2)
- **Tập tin:** `dist/RenderSub/RenderSub.exe`
- **Kiến trúc:** PE32+ (x64 / AMD64)
- **Dung lượng:** 72,337,366 bytes (68.99 MB)
- **SHA256:** `0CE4ED7079AF1C2530F79695B0E3A4AB9171EBB7EE54B66E7FAFE5B46C3ED9A1`

---

## 3. PHẦN 3 — KIỂM TRA BUNDLED FFMPEG & FFPROBE (PHASE 3)
- **`dist/RenderSub/bin/ffmpeg.exe`**:
  - Dung lượng: 223,360,000 bytes (223.36 MB)
  - Phiên bản: `ffmpeg version 8.1-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers`
  - SHA256: `d1e2a156261ecc675081943197a85f08f2868784a0af499171ede89353edad31`
- **`dist/RenderSub/bin/ffprobe.exe`**:
  - Dung lượng: 223,153,664 bytes (223.15 MB)
  - Phiên bản: `ffprobe version 8.1-full_build-www.gyan.dev Copyright (c) 2007-2026 the FFmpeg developers`
  - SHA256: `70872c3ffbc43d0b2c570f9837f54d6e9a832f4ca25463e9735b6a3ec0621478`

---

## 4. PHẦN 4 & 5 — CÔ LẬP MÔI TRƯỜNG & KIỂM THỬ THIẾU BINARY (PHASE 4 & 5)
- **Thử nghiệm Cô lập PATH:**
  - Thiết lập `PATH=C:\Windows\System32;C:\Windows`, loại bỏ hoàn toàn Python, Git, Conda, WinGet.
  - Khởi chạy trực tiếp `RenderSub.exe` → Ứng dụng nạp hoàn chỉnh các DLLs của Qt6, PyTorch, OpenCV mà không yêu cầu môi trường dev bên ngoài.
- **Thử nghiệm Cố tình làm mất Binary (Kill FFmpeg/FFprobe Test):**
  - Tạm thời đổi tên `dist/RenderSub/bin/ffmpeg.exe` trong môi trường cô lập → Hệ thống đưa ra lỗi có kiểm soát rõ ràng (`RuntimeError: Không tìm thấy công cụ FFmpeg...`).
  - Khôi phục binary → Hoạt động trở lại bình thường.

---

## 5. PHẦN 6, 7, 8 — REAL PIPELINE, NO-AUDIO & UNICODE TEST (PHASE 6, 7, 8)
- **Test Video Không Audio (No-Audio Stream):**
  - Input: `pure_video_no_audio.mp4` (2.0s, 50 frames, không có stream âm thanh).
  - Output: `pure_video_no_audio_output.mp4` (13,393 bytes, 640x360, 25 FPS, duration 2.0s).
  - Kết quả: **PASS** (Không phát sinh lỗi stream specifier matching).
- **Test Đường Dẫn Đa Ngữ / Unicode Phức Tạp (Unicode Real Test):**
  - Thư mục: `C:\RenderSub Final QA\中文测试\日本語\Русский\`
  - Input: `中文视频_测试_日本語.mp4`
  - Output: `最终_测试_日本語.mp4` (19,647 bytes, 640x360, 25 FPS, duration 2.0s, has_audio=True).
  - Kết quả: **PASS** (Đọc, ghi, OCR, render và validate thành công 100%).

---

## 6. PHẦN 9 & 10 — USER DATA ISOLATION & FIRST RUN (PHASE 9 & 10)
- **Thư mục Dữ liệu Người dùng:** `%LOCALAPPDATA%\RenderSub\`
  - `config/`: Chứa `app_settings.json`
  - `logs/`: Chứa nhật ký xử lý
  - `output/`: Thư mục xuất mặc định
- **Thử nghiệm First Run:** Xóa thư mục `%LOCALAPPDATA%\RenderSub\` và khởi chạy lại → Tự động tạo cấu trúc thư mục mới, tải default templates từ bundle tĩnh, không có lỗi phân quyền hay ghi đè tài nguyên tĩnh.

---

## 7. PHẦN 11 — OCR MODEL INITIALIZATION (PHASE 11)
- **Phân loại First-Run Internet:**
  - `FIRST_RUN_REQUIRES_INTERNET = YES`
  - *Giải trình:* Lần đầu tiên chạy EasyOCR với một bộ ngôn ngữ mới (chưa có trong `~/.EasyOCR/model/`), EasyOCR sẽ tải model weights (`craft_mlt_25k.pth`, `latin.pth`, `chinese_sim.pth`) từ server chính thức. Khi đã tải về một lần, EasyOCR hoạt động 100% offline hoàn toàn không cần Internet.

---

## 8. PHẦN 14, 15, 16 — FAILURE HANDLING, BATCH & RESOURCE LEAK (PHASE 14, 15, 16)
- **Xử lý Input lỗi / Corrupt MP4:**
  - File không tồn tại → Nâng ngoại lệ `ValueError` có kiểm soát, trạng thái báo `FAILED`.
  - File MP4 hỏng header → Nâng ngoại lệ `ValueError: Không thể mở video hoặc định dạng video không hợp lệ`, không tạo output giả.
- **Batch Processing Test:**
  - Xử lý liên tiếp 3 video trong hàng đợi batch.
  - Video 1: 11.85s (khởi tạo EasyOCR model)
  - Video 2: 0.31s (tái sử dụng model bộ nhớ đệm)
  - Video 3: 0.31s
  - Mỗi video có output độc lập, không va chạm tên file.
- **Resource Leak Test:**
  - Bộ nhớ RAM ổn định giữa các chu kỳ batch, không có process FFmpeg zombie tồn đọng sau khi kết thúc.

---

## 9. PHẦN 17 & 18 — SECURITY & DEPENDENCY AUDIT (PHASE 17 & 18)
- **Danh mục phiên bản linh kiện:**
  - Python: `3.14.3`
  - PyQt6: `6.8.0`
  - PyTorch: `2.7.0.dev20250220+cpu`
  - TorchVision: `0.22.0.dev20250220+cpu`
  - EasyOCR: `1.7.2`
  - OpenCV: `4.11.0.86`
  - FFmpeg / FFprobe: `8.1-full_build-www.gyan.dev` (Gyan Static Build)
  - Pillow: `11.1.0`
  - pydub: `0.25.1`
  - edge-tts: `7.0.0`
  - deep-translator: `1.11.4`
- **Audit Bản quyền FFmpeg:** Phân loại: `LICENSE_REVIEW_REQUIRED` (Bản build Gyan Full chứa các thư viện GPL/LGPL, cần rà soát chính sách phân phối khi phát hành thương mại).

---

## 10. PHẦN 19 — CLEAN MACHINE STATUS (PHASE 19)
- Do phiên kiểm thử hiện tại được thực thi trên môi trường local host với PATH isolation (không có máy ảo VM thứ hai tách biệt hoàn toàn), theo nguyên tắc nghiêm ngặt của Release Gate:
  - Phân loại: `REAL CLEAN MACHINE = NOT VERIFIED`

---

## 11. PHẦN 20 — BẢNG MA TRẬN PHÂN LOẠI TỔNG THỂ (FINAL VERDICT MATRIX)

| Hạng mục kiểm tra | Phân loại kết quả |
| :--- | :---: |
| **CORE REGRESSION** | **PASS** |
| **STANDALONE EXE** | **PASS** |
| **BUNDLED FFMPEG** | **PASS** |
| **BUNDLED FFPROBE** | **PASS** |
| **REAL EXE PIPELINE** | **PASS** |
| **UNICODE** | **PASS** |
| **NO-AUDIO VIDEO** | **PASS** |
| **SETTINGS** | **PASS** |
| **CANCELLATION** | **PASS** |
| **FAILURE HANDLING** | **PASS** |
| **BATCH** | **PASS** |
| **RESOURCE CLEANUP** | **PASS** |
| **SECRET AUDIT** | **PASS** |
| **LICENSE REVIEW** | **REVIEW REQUIRED** |
| **REAL CLEAN MACHINE** | **NOT VERIFIED** |

---

## KẾT LUẬN CUỐI CÙNG (FINAL VERDICT)

```text
RELEASE CANDIDATE — CLEAN MACHINE NOT VERIFIED
```
*(Ghi chú: Toàn bộ 13 hạng mục tính năng, bảo mật, pipeline, standalone executable và PATH isolation đều đạt chuẩn PASS 100%. Trạng thái Release Candidate được gắn do tính trung thực trong việc chưa có máy ảo độc lập thứ hai và cần rà soát bản quyền phân phối FFmpeg GPL).*
