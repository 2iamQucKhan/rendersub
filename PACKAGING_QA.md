# BÁO CÁO QA/QC ĐÓNG GÓI BẢN PHÁT HÀNH STANDALONE (PACKAGING QA REPORT)

## 1. TỔNG QUAN CHIẾN LƯỢC ĐÓNG GÓI (PACKAGING STRATEGY)
- **Kiến trúc phân phối**: Standalone Windows Onedir Distribution (`dist/RenderSub/`)
- **Tập tin thực thi chính**: `RenderSub.exe` (68.99 MB)
- **Tổng dung lượng phân phối**: ~1.95 GB (bao gồm PyTorch, torchvision, EasyOCR model dependencies, PyQt6 runtime, and static FFmpeg 8.1 binaries)
- **Yêu cầu môi trường phía người dùng cuối**: **HOÀN TOÀN KHÔNG CẦN CÀI ĐẶT**:
  - Không cần Python
  - Không cần pip / PyQt6 / OpenCV / PyTorch
  - Không cần FFmpeg / FFprobe cài đặt trên hệ thống (PATH)

---

## 2. KIẾN TRÚC ĐỊNH VỊ TÀI NGUYÊN (CENTRALIZED RESOURCE RESOLUTION)
Toàn bộ logic truy xuất đường dẫn và binaries được tập trung hóa tuyệt đối trong module [`resource_utils.py`](resource_utils.py):
1. **`get_base_dir()`**:
   - Khi chạy ở dạng đóng gói (`sys.frozen == True`): Trỏ về thư mục chứa file thực thi (`dist/RenderSub`).
   - Khi chạy ở môi trường phát triển: Trỏ về thư mục gốc của dự án.
2. **`get_resource_path(relative_path)`**:
   - Ưu tiên tìm trong `_MEIPASS` (nếu có), thư mục gốc bundle `base_dir`, hoặc `base_dir/_internal`.
3. **`get_ffmpeg_path()` & `get_ffprobe_path()`**:
   - **Ưu tiên 1**: `dist/RenderSub/bin/ffmpeg.exe` (bundled static binary).
   - **Ưu tiên 2**: `<sys._MEIPASS>/bin/ffmpeg.exe`.
   - **Ưu tiên 3**: Hệ thống PATH (`shutil.which`).
   - **Cơ chế phòng thủ**: Nâng lỗi rõ ràng kèm hướng dẫn nếu không tìm thấy binary nào.
4. **`get_user_data_dir()`**:
   - Dữ liệu người dùng cấu hình (`app_settings.json`), lịch sử xuất sub, logs được ghi độc quyền vào `%LOCALAPPDATA%/RenderSub/` (hoặc `~/.rendersub/`), đảm bảo tương thích 100% với các thư mục cài đặt chỉ đọc (Read-only / Program Files).

---

## 3. DANH MỤC TÀI NGUYÊN BUNDLE (BUNDLED BINARIES & ASSETS)
| Thành phần | Đường dẫn trong bản phân phối | Dung lượng / Ghi chú |
| :--- | :--- | :--- |
| **Main Executable** | `dist/RenderSub/RenderSub.exe` | 68.99 MB |
| **FFmpeg Binary** | `dist/RenderSub/bin/ffmpeg.exe` | 223.36 MB (Gyan Static Build 8.1) |
| **FFprobe Binary** | `dist/RenderSub/bin/ffprobe.exe` | 223.15 MB (Gyan Static Build 8.1) |
| **Từ điển VietPhrase** | `dist/RenderSub/Data/VietPhrase.txt` | 4.37 MB (115,368 cụm từ) |
| **Từ điển Lạc Việt** | `dist/RenderSub/Data/LacViet.txt` | 5.68 MB |
| **Prompt Template** | `dist/RenderSub/config/xkiro_prompt_template.json` | 751 B |
| **Trending Dict** | `dist/RenderSub/config/trending_dict.json` | 6.98 KB |

---

## 4. MA TRẬN KIỂM ĐỊNH MÁY SẠCH (CLEAN-MACHINE ISOLATION MATRIX)
Đã thực hiện kiểm thử tự động độc lập qua [`tests/test_standalone_distribution.py`](tests/test_standalone_distribution.py) với môi trường bị cô lập hoàn toàn (`PATH` chỉ chứa `System32` và `dist/RenderSub/bin`, `PYTHONPATH=""`, `PYTHONHOME=""`):

| Test Case ID | Mục tiêu kiểm định | Kết quả | Ghi chú |
| :--- | :--- | :---: | :--- |
| **TEST-01** | Cấu trúc phân phối `dist/RenderSub` | **PASS** | `RenderSub.exe` tồn tại, dung lượng 68.99 MB |
| **TEST-02** | `bin/ffmpeg.exe` & `bin/ffprobe.exe` | **PASS** | Thực thi `ffmpeg -version` & `ffprobe -version` thành công từ binary nội bộ |
| **TEST-03** | Tài nguyên từ điển & config templates | **PASS** | Đầy đủ dữ liệu từ điển VietPhrase, Names, LacViet, templates |
| **TEST-04** | Bảo mật API Key & Secrets | **PASS** | Không có Google API Key hoặc secret dev bị rò rỉ vào bundle |
| **TEST-05** | Khởi chạy máy sạch cô lập (Clean-Machine Launch) | **PASS** | `RenderSub.exe` khởi động trơn tru, nạp đầy đủ Qt6, PyTorch, Matplotlib, EasyOCR không có lỗi DLL hay thiếu module |

---

## 5. KẾT QUẢ HỆ THỐNG KIỂM THỬ TỔNG THỂ (FULL REGRESSION SUITE)
Chạy toàn bộ 35 unit & E2E tests:
```text
python -m unittest tests/test_output_validation.py tests/test_ffmpeg_writer.py tests/test_parallel_processor.py tests/test_ocr_schema.py tests/test_translation.py tests/test_tts.py tests/test_unicode_paths.py tests/test_batch_pipeline.py tests/test_gui_state.py tests/test_real_e2e_pipeline.py tests/test_standalone_distribution.py
----------------------------------------------------------------------
Ran 35 tests in 35.763s

OK
```

## 6. KẾT LUẬN
Bản build standalone Windows cho RenderSub đã hoàn thành đóng gói thành công 100%, đáp ứng đầy đủ tất cả các tiêu chí độc lập môi trường và sẵn sàng phát hành.
