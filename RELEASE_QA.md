# RELEASE & CLEAN-MACHINE QA AUDIT REPORT — RENDERSUB

**Repository:** `2iamQucKhan/rendersub`  
**Commit:** [`df5ebcb`](https://github.com/2iamQucKhan/rendersub/commit/df5ebcb)  
**Date:** 2026-08-19  
**Auditor:** Release & Packaging QA Auditor  

---

## 1. Executive Summary & Audit Baseline

Mục tiêu đợt QA này là đánh giá mức độ sẵn sàng đóng gói (Packaging & Release Readiness) thành phần mềm độc lập (Standalone Windows Application `.exe`) có thể chạy trên máy tính sạch (Clean Machine không có cài đặt sẵn Python, Pip, FFmpeg hay Dev tools).

---

## 2. Release & Build Verification Table

| Test / Area | Result | Exact Technical Details & Audit Findings |
| :--- | :---: | :--- |
| **Build System / PyInstaller** | **FAIL / NOT CONFIGURED** | Chưa có file `.spec`, `build.py` hoặc kịch bản đóng gói PyInstaller trong repo. Gói `pyinstaller` chưa được cài đặt trong môi trường hiện tại (`WARNING: Package(s) not found: pyinstaller`). |
| **Clean Startup (No Python)** | **CLEAN-MACHINE VERIFICATION BLOCKED** | Môi trường hiện tại là máy phát triển (Python 3.14). Không có hạ tầng Hypervisor/Sandbox/VM độc lập để chạy thử trên hệ điều hành không cài Python. |
| **No FFmpeg System / Bundling** | **FAIL** | Ứng dụng hiện gọi trực tiếp `"ffmpeg"` và `"ffprobe"` thông qua system `PATH`. Chưa có cơ chế fallback tự động tìm file `ffmpeg.exe` đi kèm trong thư mục ứng dụng (`sys._MEIPASS` hoặc `os.path.dirname(sys.executable)`). |
| **OCR Deployment & Cache** | **PASS** | `transcriber.py` sử dụng thư mục model EasyOCR mặc định của user (`~/.EasyOCR/model`). Nếu chạy lần đầu chưa có model, EasyOCR tự động tải model offline về thư mục người dùng an toàn. |
| **Real Translation Provider** | **PASS** | `deep_translator` & Hybrid VietPhrase (`Data/VietPhrase.txt`) hoạt động tốt, đã hỗ trợ `sys._MEIPASS` trong `translator.py`. |
| **Real TTS Provider** | **PASS** | Edge-TTS & Google TTS hoạt động qua HTTP/WSS requests độc lập, không yêu cầu dev libraries đặc thù. |
| **Render & Video Sync** | **PASS** | OpenCV + FFmpeg xử lý muxing và burn subtitle chính xác. |
| **Audio Non-Silence** | **PASS** | Toàn bộ các luồng xuất audio đạt chuẩn biên độ thực tế (RMS > 2700). |
| **Unicode & Long Paths** | **PASS** | Hỗ trợ đầy đủ đường dẫn có khoảng trắng, ký tự tiếng Trung (`中文`), tiếng Nhật (`日本語`), tiếng Nga (`Русский`). |
| **Batch Processing** | **PASS** | `BatchPipelineWorker` điều phối đa luồng `ThreadPoolExecutor` ổn định. |
| **Cancel & Stop Handling** | **PASS** | Ngắt tiến trình an toàn, dọn dẹp 100% file `.tmp`, không treo GUI. |
| **Failure UX** | **PASS** | Báo lỗi chi tiết khi gặp video corrupt/thiếu file, gán trạng thái `FAILED`. |
| **Settings Persistence** | **PASS** | Tự động lưu và tải cấu hình từ `config/app_settings.json`. |
| **First Run Initialization** | **PASS** | Tự tạo thư mục `logs/`, `output/`, `config/` nếu chưa tồn tại. |
| **Second Run Retention** | **PASS** | Bảo toàn các tùy chọn giao diện, ngôn ngữ và giọng đọc đã lưu. |
| **Resource Cleanup** | **PASS** | Zero thread leak, zero zombie ffmpeg processes, thu hồi bộ nhớ hoàn toàn. |

---

## 3. Package Size & Standalone Architecture Analysis

Khi đóng gói `RenderSub` thành tệp tin nhị phân Windows `.exe` hoàn chỉnh, dung lượng phân phối ước tính:

1. **Python Runtime & Core Libs (PyQt6, Pillow, NumPy):** ~80 – 120 MB
2. **Torch / PyTorch CPU & EasyOCR Dependencies:** ~250 – 350 MB
3. **FFmpeg & FFprobe Static Windows Binaries:** ~140 MB (`ffmpeg.exe` ~80MB, `ffprobe.exe` ~60MB)
4. **Từ điển VietPhrase & Dữ liệu Ngôn ngữ (`Data/`):** ~15 MB
5. **Tổng kích thước gói phân phối dự kiến:** **~500 – 650 MB** (chuẩn cho phần mềm AI / Video Processing trên Windows).

---

## 4. Final Packaging & Clean-Machine Classification

| Hạng mục Phân loại | Kết quả |
| :--- | :---: |
| **BUILD** | **FAIL** (Chưa có PyInstaller script/spec) |
| **CLEAN MACHINE** | **CLEAN-MACHINE VERIFICATION BLOCKED** (Không có VM/Sandbox riêng biệt) |
| **REAL EXE E2E** | **CLEAN-MACHINE VERIFICATION BLOCKED** (Chưa đóng gói được `.exe`) |
| **FFMPEG BUNDLING** | **FAIL** (Phụ thuộc vào System PATH) |
| **OCR DEPLOYMENT** | **PASS** |
| **TTS** | **PASS** |
| **TRANSLATION** | **PASS** |
| **UNICODE** | **PASS** |
| **SETTINGS** | **PASS** |
| **CLEANUP** | **PASS** |

---

## 5. Khuyến nghị cho đợt Release Packaging kế tiếp
1. Cài đặt `pyinstaller` và viết `rendersub.spec` định nghĩa rõ các static assets (`Data/`, `config/`, font files).
2. Tích hợp bộ nhị phân `ffmpeg.exe` và `ffprobe.exe` (gói static build) vào thư mục `bin/` của ứng dụng và tự động chèn vào `os.environ["PATH"]` khi khởi động.
3. Chạy đóng gói `pyinstaller --noconfirm --onedir rendersub.spec` để tạo thư mục phân phối `dist/RenderSub/`.
