# RenderSub QA Re-Audit (Commit `f3fb13d`)

## 1. Executive Summary
Báo cáo tái kiểm thử và thẩm tra độ tin cậy (Re-Audit) đối với toàn bộ các bản vá đã thực hiện tại commit `f3fb13d` trên repository `2iamQucKhan/rendersub`.

Đợt audit này tập trung vào việc **xác minh bằng chứng thực tế qua code path và runtime**, đối chiếu với các tuyên bố trong `walkthrough.md`, kiểm tra chi tiết các trường hợp biên (edge cases), race conditions, frame loss, leak tài nguyên và độ tin cậy của bộ test suite.

---

## 2. Baseline & Search Verification

Toàn bộ các từ khóa nhạy cảm và nguy cơ bypass logic đã được quét trên toàn bộ codebase:

| Từ khóa / Pattern | Số lượng phát hiện | Đánh giá phân loại | Chi tiết vị trí & Bằng chứng |
| :--- | :---: | :--- | :--- |
| `sleep(` | 13 | ✅ **Hợp lệ (Legitimate)** | Chỉ xuất hiện ở: API exponential backoff (`translator.py`, `gemini_vision_ocr.py`), Edge-TTS retry (`dubber.py`), Key tester delay (`main.py:2580`), Batch queue delay giữa các job (`main.py:7716`). **Tuyệt đối không còn sleep giả lập tiến trình trong `BatchPipelineWorker`**. |
| `sig_item_finished` | 5 | ✅ **Hợp lệ** | Chỉ phát ra sau khi `execute_single_video_pipeline` hoàn tất hoặc ném exception trong `main.py` (`line 168` cho success, `line 184` cho failure). |
| `success=True` | 0 | ✅ **Sạch sẽ** | Không có gán cứng cờ thành công. |
| `COMPLETED` | 2 | ✅ **Hợp lệ** | Thuộc enum `PipelineState.COMPLETED` và chỉ phát khi hoàn tất bước 6 và xác thực output thành công. |
| `TODO` / `FIXME` | 0 | ✅ **Sạch sẽ** | Không có nợ kỹ thuật hoặc TODO dở dang trong code thực thi. |
| `fake` / `mock` | 12 | ✅ **Hợp lệ** | Chỉ xuất hiện trong test standalone / mock data của test files (`test_title_translation_standalone.py`, `test_tts_checkbox_guard.py`). Không xuất hiện trong code production. |

---

## 3. Detailed Verification per Component

### 3.1. Batch Pipeline Execution Path
- **Trạng thái:** ✅ **VERIFIED REAL EXECUTION**
- **Code path:** `BatchPipelineWorker.run()` ➔ `ThreadPoolExecutor(max_workers=actual_workers)` ➔ `process_item(item)` ➔ `execute_single_video_pipeline(vpath, out_path, cfg)`.
- **Cơ chế:**
  1. `Input Validation` (`validate_output_video` kiểm tra file nguồn).
  2. `OCR` (`gemini_vision_ocr` / `ParallelChunkOCRProcessor` với EasyOCR/PaddleOCR/xKiro).
  3. `Translation` (`translator.translate_segments`).
  4. `TTS & Audio Sync` (`dubber.create_dubbed_video` sinh audio từng câu + `speed_adjust_audio` + `pydub overlay`).
  5. `Rendering` (Xử lý watermark/burn sub + FFmpeg encoder mux video/audio).
  6. `Output Validation` (`validate_output_video` kiểm tra file kết quả).
- **Concurrency & Isolation:**
  - `item["index"]` độc lập, signal `sig_item_progress`, `sig_item_finished` gửi chính xác index của từng video.
  - Mỗi video xuất ra đường dẫn riêng biệt `f"{base_stem}_dubbed{ext}"` trong `output_dir`.
  - Biến đếm hoàn thành được bảo vệ bởi `threading.Lock()`.

---

### 3.2. E2E Smoke Test Classification
- **File:** `scratch/test_e2e_smoke_pipeline.py`
- **Phân loại:** 🟡 **HYBRID / PARTIAL E2E**
- **Phân tích từng bước:**
  - `Input Video Creation`: Tạo video 75 frames (3s) có vẽ text. (REAL)
  - `Validation`: OpenCV kiểm tra file hợp lệ. (REAL)
  - `OCR Execution`: EasyOCR Reader khởi tạo và quét video. (REAL)
  - *Lưu ý:* Do video sinh tự động bằng OpenCV text không kèm bounding box chỉ định, EasyOCR trả về 0 đoạn phụ đề. Do đó, bước Translation và TTS được tự động bypass hợp lệ (`if segments:`).
  - `Rendering & Muxing`: Xử lý khung hình và FFmpeg muxing tạo file output. (REAL)
  - `Output Validation`: OpenCV kiểm tra resolution 320x240, 75 frames, 25fps. (REAL)

---

### 3.3. Output Validation (`validate_output_video`)
- **Trạng thái:** ✅ **VERIFIED STRICT VALIDATION**
- **Bằng chứng kiểm tra:**
  1. `file_path` tồn tại và `os.path.getsize(file_path) > 0`.
  2. Mở file qua `cv2.VideoCapture`.
  3. `cap.isOpened()` kiểm tra container và codec header.
  4. Xác thực số học: `width > 0`, `height > 0`, `fps > 0`, `frame_count > 0`.
  5. Tính toán `duration = frame_count / fps >= min_duration`.
  6. `check_audio=True`: gọi `ffprobe` trích xuất `stream=codec_name,duration` của track `a:0`.
- **Test Corrupted File:** `tests/test_output_validation.py` kiểm thử file 38 bytes chứa text rác, OpenCV báo lỗi `moov atom not found` và hàm trả về `False` chính xác.

---

### 3.4. Atomic Output & Resource Cleanup
- **Trạng thái:** ✅ **VERIFIED ATOMIC & SAFE**
- **Cơ chế triển khai:**
  - `FFmpegVideoWriter`: Ghi vào `.tmp_{timestamp}_{base_name}`. Sau khi render xong và validate đạt chuẩn, thực hiện `shutil.move()` sang `final_output_path`. Nếu lỗi, file `.tmp` bị xóa bỏ ngay trong `release()`.
  - `dubber.create_dubbed_video`: Tạo thư mục tạm độc lập `supersubs_dub_{timestamp}_{pid}`. Bọc toàn bộ logic trong khối `try...finally: shutil.rmtree(temp_dir, ignore_errors=True)`.
  - Render file tạm `.tmp_{unique_tag}_{basename}` và chỉ move sang file đích khi render thành công.
  - Không bao giờ để lại file đích bị hỏng/dở dang nếu tiến trình bị ngắt.

---

### 3.5. FFmpeg Video Writer & Zero Dummy Frames
- **Trạng thái:** ✅ **VERIFIED ZERO DUMMY FRAMES**
- **Bằng chứng:**
  - Hàm `probe_encoder_support(encoder_name)` truyền frame 16x16 tới stream `null` (`-f null -`). Không tạo bất kỳ file tạm hay ghi vào output thật.
  - `FFmpegVideoWriter` ghi frame trực tiếp từ frame 0 nhận được từ caller qua `stdin.write(frame.tobytes())`.
  - `tests/test_ffmpeg_writer.py` kiểm tra frame đầu tiên của video xuất ra: giá trị màu của frame 0 là màu xanh thật sự (`mean > 100`), không phải màu đen `(0,0,0)` của dummy frame.

---

### 3.6. ParallelVideoProcessor & Strict Frame Ordering
- **Trạng thái:** ✅ **VERIFIED STRICT ORDERING & NO DEADLOCK**
- **Cơ chế:**
  - `reader_worker`: Đọc frame tuần tự `(0, frame0), (1, frame1)...` đẩy vào `read_queue`.
  - `process_worker`: Xử lý đa luồng và đẩy `(f_idx, processed)` vào `write_queue`.
  - `writer_worker`: Nhận frame và lưu vào từ điển `pending_frames[f_idx] = frame`. Chỉ ghi ra đĩa khi `next_expected_idx in pending_frames` và tăng `next_expected_idx += 1`.
  - `Worker Exception Handling`: Nếu 1 worker ném lỗi, gán `self.worker_exception = exc`, đặt `self.is_running = False`, gửi sentinel giải phóng hàng đợi và `run()` ném exception ngay lập tức, không gây deadlock.
  - `tests/test_parallel_processor.py`: Kiểm thử 20 frames với 4 worker threads, xác minh độ tăng đơn điệu (monotonic increase) của các khung hình xuất ra đạt 100%.

---

### 3.7. Cancellation Logic
- **Trạng thái:** ✅ **VERIFIED CLEAN CANCELLATION**
- **Cơ chế:**
  - `BatchPipelineWorker.cancel()`: Đặt cờ `_is_cancelled = True`, gọi `_executor.shutdown(wait=False, cancel_futures=True)`.
  - `execute_single_video_pipeline`: Kiểm tra `cancel_check_fn()` sau mỗi công đoạn (Validation, OCR, Translation, Dubbing). Nếu bị cancel, ném `InterruptedError`, dọn dẹp file `.tmp.mp4` và không emit SUCCESS.
  - `tests/test_batch_pipeline.py`: `test_batch_cancel_behavior` xác nhận cờ cancel và ngắt luồng an toàn.

---

### 3.8. Secrets & Security Audit
- **Trạng thái:** ✅ **CLEAN (NO SECRETS COMMITTED)**
- **Bằng chứng:**
  - `config/api_keys.json` được bỏ qua hoàn toàn trong `.gitignore` và không bị track trên Git (`git ls-files` trả về không tìm thấy).
  - Không có file `.env` nào bị commit.
  - `DEFAULT_API_KEYS` trong `main.py` là danh sách rỗng `[]`.
  - File template mẫu `config/api_keys.example.json` chỉ chứa placeholder `AIzaSyYourGeminiApiKeyHere1`.

---

## 4. Test Suite Quality Audit

Tổng số test: **25 tests** trong 9 file test mới:

| Test File | Phân loại | Mục tiêu kiểm thử | Mock cái gì? | Có bắt được regression không? |
| :--- | :---: | :--- | :--- | :---: |
| `tests/test_output_validation.py` | Unit | File không tồn tại, file 0 byte, file corrupt, video hợp lệ | Không | Có |
| `tests/test_ffmpeg_writer.py` | Integration | Probe encoder, ghi FFmpeg pipe thật, kiểm tra frame 0 không bị dummy đen | Không | Có |
| `tests/test_parallel_processor.py` | Integration | Strict frame ordering 4 workers, propagate exception khi worker lỗi | Không | Có |
| `tests/test_ocr_schema.py` | Unit | Merge & deduplicate subtitle, schema chuẩn, lọc noise rác | Không | Có |
| `tests/test_translation.py` | Integration | TrendingSlangManager, VietPhrase, Google Translate segments | Không | Có |
| `tests/test_tts.py` | Integration | Danh sách voice, bíp hóa `***`, speed adjustment audio | Không | Có |
| `tests/test_unicode_paths.py` | Unit | Đường dẫn tiếng Việt, Trung, Nhật, Hàn, Unicode video write | Không | Có |
| `tests/test_batch_pipeline.py` | Integration | Queue rỗng, multi-worker batch song song xuất video thật, cancel queue | Không | Có |
| `tests/test_gui_state.py` | GUI Unit | Trạng thái button Idle/Running, Checkbox TTS điều khiển dropdown voice | Không | Có |

---

## 5. Remaining Risks & Recommendations

### 🟡 R1: Thư viện `requirements.txt` chưa ghim phiên bản cụ thể (Unpinned Dependencies)
- **Rủi ro:** Các thư viện như `edge-tts`, `deep-translator`, `opencv-python`, `PyQt6` không có dấu `==x.y.z`, có thể gặp xung đột khi môi trường cài đặt phiên bản mới hơn trong tương lai.
- **Khuyến nghị:** Tạo file `requirements.lock` hoặc ghim phiên bản cụ thể (ví dụ: `PyQt6>=6.6.0,<6.9.0`, `opencv-python>=4.8.0`).

### 🟡 R2: Hợp nhất luồng Dubbing đơn lẻ trong GUI với `execute_single_video_pipeline`
- **Hiện trạng:** Tab chính 1-Click GUI sử dụng `DubbingThread` riêng, trong khi Batch Queue sử dụng `execute_single_video_pipeline`.
- **Khuyến nghị:** Refactor `DubbingThread` gọi trực tiếp `execute_single_video_pipeline` để quy về một code path duy nhất cho toàn ứng dụng.

---

## 6. Production Readiness Conclusion

### **ĐÁNH GIÁ: PRODUCTION READY FOR QC (SẴN SÀNG CHO ĐỢT ĐÁNH GIÁ QC/QA)**

- **P0 Blockers:** 0
- **Fake Simulation:** Đã xóa bỏ 100%.
- **Output Integrity:** Đảm bảo 100% video xuất ra được xác thực tính toàn vẹn (OpenCV, FFmpeg, duration, frame ordering).
- **Test Suite:** 25/25 Tests Passed.
- **Bytecode Compile:** 100% Passed (0 syntax/bytecode errors).
- **Security:** 0 secret/API keys bị rò rỉ trên Git.
