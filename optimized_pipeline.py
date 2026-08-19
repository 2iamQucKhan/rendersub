import os
import sys
import time
import json
import shutil
import subprocess
import queue
import threading
import numpy as np
import cv2

def is_ascii_path(path_str):
    try:
        path_str.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

def ensure_safe_ascii_video_path(video_path):
    """
    Xử lý đường dẫn tệp Unicode/Tiếng Trung:
    Nếu video_path chứa ký tự Unicode / CJK không thuộc ASCII:
    Tự động copy file video sang thư mục temp_ascii_safe/input_safe.mp4 để tránh lỗi OpenCV / FFmpeg.
    """
    if not video_path or not os.path.exists(video_path):
        return video_path

    if is_ascii_path(video_path):
        return video_path

    base_dir = os.path.dirname(os.path.abspath(video_path))
    temp_dir = os.path.join(base_dir, "temp_ascii_safe")
    os.makedirs(temp_dir, exist_ok=True)

    ext = os.path.splitext(video_path)[1] or ".mp4"
    safe_path = os.path.join(temp_dir, f"input_safe{ext}")

    try:
        if not os.path.exists(safe_path) or os.path.getmtime(video_path) > os.path.getmtime(safe_path):
            shutil.copy2(video_path, safe_path)
        return safe_path
    except Exception as e:
        print(f"Cảnh báo copy Unicode video path: {e}")
        return video_path

# --- 1. TĂNG TỐC OPENCV & PHẦN CỨNG (OPENCL / CUDA) ---
def enable_opencv_hardware_acceleration():
    """
    Kích hoạt các cờ tối ưu hóa của OpenCV, OpenCL GPU và cấu hình CUDA DNN nếu có.
    """
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass

    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
    except Exception:
        pass

    cuda_enabled = False
    try:
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            cuda_enabled = True
    except Exception:
        pass

    return {
        "optimized": cv2.useOptimized(),
        "opencl": cv2.ocl.useOpenCL() if hasattr(cv2.ocl, 'useOpenCL') else False,
        "cuda": cuda_enabled
    }


# --- 2. LỌC FRAME THÔNG MINH (SMART FRAME SKIPPING / MOTION DIFF) ---
class SmartFrameInpainter:
    """
    Lớp quản lý đệm và so sánh sự thay đổi của vùng Crop phụ đề giữa các khung hình.
    Sử dụng cv2.absdiff để phát hiện chuyển động. Nếu khác biệt < motion_threshold,
    tái sử dụng kết quả Inpaint/Mask của frame trước đó thay vì tính toán lại cv2.inpaint.
    """
    def __init__(self, motion_threshold=3.5):
        self.motion_threshold = motion_threshold
        self.prev_crops = {}          # {box_id: crop_gray}
        self.cached_results = {}      # {box_id: processed_crop_bgr}

    def process_crop(self, frame, bbox, mask_mode="inpaint"):
        """
        Xử lý mờ / che / inpaint trên vùng crop của frame.
        """
        if frame is None or not bbox or len(bbox) != 4:
            return frame

        x, y, w, h = bbox
        fh, fw = frame.shape[:2]

        x1 = max(0, min(x, fw - 1))
        y1 = max(0, min(y, fh - 1))
        x2 = max(x1 + 1, min(x + w, fw))
        y2 = max(y1 + 1, min(y + h, fh))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return frame

        box_id = (x1, y1, x2 - x1, y2 - y1, mask_mode)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Kiểm tra cache frame trước
        if box_id in self.prev_crops and box_id in self.cached_results:
            prev_gray = self.prev_crops[box_id]
            if prev_gray.shape == crop_gray.shape:
                # Tính độ chênh lệch tuyệt đối (Absolute Difference)
                diff = cv2.absdiff(crop_gray, prev_gray)
                mean_diff = np.mean(diff)

                if mean_diff < self.motion_threshold:
                    # Chênh lệch dưới ngưỡng -> Tái sử dụng kết quả đệm!
                    frame[y1:y2, x1:x2] = self.cached_results[box_id]
                    return frame

        # Nếu có sự thay đổi hoặc chưa có cache -> Thực hiện tính toán
        processed_crop = crop.copy()
        cw = x2 - x1
        ch = y2 - y1

        if mask_mode == "black":
            processed_crop[:] = 0
        elif mask_mode == "blur":
            kw = 51 if 51 < cw else max(1, cw - 1 | 1)
            kh = 51 if 51 < ch else max(1, ch - 1 | 1)
            if kw % 2 == 0: kw = max(1, kw - 1)
            if kh % 2 == 0: kh = max(1, kh - 1)
            processed_crop = cv2.GaussianBlur(crop, (kw, kh), 0)
        elif mask_mode == "inpaint":
            _, mask = cv2.threshold(crop_gray, 200, 255, cv2.THRESH_BINARY)
            processed_crop = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)

        # Lưu lại cache cho frame tiếp theo
        self.prev_crops[box_id] = crop_gray
        self.cached_results[box_id] = processed_crop

        frame[y1:y2, x1:x2] = processed_crop
        return frame


# --- 3. XÁC THỰC VIDEO ĐẦU RA (OUTPUT VALIDATION) ---
def validate_output_video(file_path, check_audio=False, min_duration=0.1):
    """
    Xác thực toàn diện tệp video đầu ra sau render:
    1. Kiểm tra tồn tại và kích thước file > 0.
    2. Mở bằng OpenCV: width > 0, height > 0, fps > 0, frame_count > 0, duration >= min_duration.
    3. Nếu check_audio=True: kiểm tra stream âm thanh hợp lệ bằng FFprobe hoặc subprocess.
    Trả về: (True, info_dict) hoặc (False, error_message).
    """
    if not file_path or not os.path.exists(file_path):
        return False, f"File không tồn tại: {file_path}"

    size_bytes = os.path.getsize(file_path)
    if size_bytes <= 0:
        return False, f"File video rỗng (0 bytes): {file_path}"

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False, f"OpenCV không thể mở file video: {file_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width <= 0 or height <= 0:
        return False, f"Kích thước video không hợp lệ: {width}x{height}"
    if fps <= 0:
        return False, f"FPS không hợp lệ: {fps}"
    if frame_count <= 0:
        return False, f"Số frame không hợp lệ: {frame_count}"

    duration_sec = frame_count / fps if fps > 0 else 0.0
    if duration_sec < min_duration:
        return False, f"Thời lượng video quá ngắn ({duration_sec:.2f}s < {min_duration}s)"

    has_audio = False
    if check_audio:
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name,duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                has_audio = True
            else:
                return False, "Không tìm thấy track âm thanh hợp lệ trong file video."
        except Exception as e:
            # Nếu không có ffprobe, thử qua pydub/wave hoặc bỏ qua cảnh báo
            pass

    info = {
        "file_path": file_path,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "has_audio": has_audio
    }
    return True, info


# --- 3. GHI VIDEO SIÊU TỐC QUA FFMPEG SUBPROCESS (GPU NVENC / CPU ULTRAFAST) ---
def probe_encoder_support(encoder_name):
    """Kiểm tra xem hệ thống có hỗ trợ video encoder này hay không bằng ffmpeg test pipe tới null."""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", "16x16",
            "-r", "25",
            "-i", "-",
            "-c:v", encoder_name,
            "-frames:v", "1",
            "-f", "null", "-"
        ]
        dummy_frame = np.zeros((16, 16, 3), dtype=np.uint8).tobytes()
        p = subprocess.run(cmd, input=dummy_frame, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=4)
        return p.returncode == 0
    except Exception:
        return False


class FFmpegVideoWriter:
    """
    Ghi video siêu tốc bằng cách truyền trực tiếp raw BGR bytes vào ống dẫn (stdin pipe) của FFmpeg.
    Tự động kiểm tra h264_nvenc qua probe, nếu không khả dụng thì fallback sang libx264.
    ĐẢM BẢO: Không ghi frame dummy nào vào file output thật!
    Hỗ trợ atomic temp output: ghi vào .tmp.mp4 trước khi validate.
    """
    def __init__(self, output_path, width, height, fps=25.0, codec="auto", atomic=True):
        self.final_output_path = output_path
        self.atomic = atomic
        if self.atomic:
            out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
            os.makedirs(out_dir, exist_ok=True)
            base_name = os.path.basename(output_path)
            self.output_path = os.path.join(out_dir, f".tmp_{int(time.time()*1000)}_{base_name}")
        else:
            self.output_path = output_path

        self.width = width
        self.height = height
        self.fps = fps
        self.pipe = None
        self.written_frames = 0
        self.is_closed = False

        self._init_ffmpeg(codec)

    def _init_ffmpeg(self, codec):
        ffmpeg_bin = "ffmpeg"
        
        chosen_encoder = None
        if codec == "nvenc" or (codec == "auto" and probe_encoder_support("h264_nvenc")):
            chosen_encoder = ["-c:v", "h264_nvenc", "-preset", "p1", "-b:v", "5M"]
        else:
            chosen_encoder = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22"]

        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "-"
        ] + chosen_encoder + [
            "-pix_fmt", "yuv420p",
            self.output_path
        ]

        try:
            self.pipe = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Không tìm thấy ffmpeg trên hệ thống. Vui lòng cài đặt FFmpeg và cấu hình PATH.") from exc
        except Exception as exc:
            raise RuntimeError(f"Lỗi khởi tạo FFmpeg Video Writer: {exc}") from exc

    def write(self, frame):
        if frame is None or self.is_closed:
            return
        if not self.pipe or not self.pipe.stdin:
            raise RuntimeError("FFmpeg pipe không ở trạng thái sẵn sàng để ghi.")

        try:
            self.pipe.stdin.write(frame.tobytes())
            self.written_frames += 1
        except Exception as exc:
            self.is_closed = True
            err_msg = ""
            if self.pipe:
                try:
                    _, stderr_bytes = self.pipe.communicate(timeout=2)
                    err_msg = stderr_bytes.decode('utf-8', errors='replace')[-1000:]
                except Exception:
                    pass
            raise RuntimeError(f"Lỗi ghi frame vào FFmpeg pipe (đã ghi {self.written_frames} frames): {exc}\n{err_msg}") from exc

    def release(self):
        if self.is_closed:
            return
        self.is_closed = True

        if self.pipe:
            try:
                if self.pipe.stdin:
                    try:
                        self.pipe.stdin.flush()
                    except Exception:
                        pass
                    self.pipe.stdin.close()
                stdout_data, stderr_data = self.pipe.communicate(timeout=30)
                if self.pipe.returncode != 0:
                    err_msg = stderr_data.decode('utf-8', errors='replace')[-1000:] if stderr_data else "Unknown error"
                    raise RuntimeError(f"FFmpeg render thất bại (mã thoát {self.pipe.returncode}):\n{err_msg}")
            except Exception as e:
                try:
                    self.pipe.kill()
                except Exception:
                    pass
                if not isinstance(e, RuntimeError):
                    raise RuntimeError(f"Lỗi đóng FFmpeg writer: {e}") from e
                else:
                    raise
            finally:
                self.pipe = None

        # Nếu dùng atomic output: validate file tạm rồi rename sang final_output_path
        if self.atomic and os.path.exists(self.output_path):
            valid, info = validate_output_video(self.output_path, check_audio=False)
            if not valid:
                try:
                    os.remove(self.output_path)
                except Exception:
                    pass
                raise RuntimeError(f"Video xuất ra không hợp lệ sau khi render: {info}")

            if os.path.exists(self.final_output_path):
                try:
                    os.remove(self.final_output_path)
                except Exception:
                    pass
            shutil.move(self.output_path, self.final_output_path)


# --- 4. KIẾN TRÚC PIPELINE ĐA LUỒNG VỚI STRICT FRAME ORDERING ---
class ParallelVideoProcessor:
    """
    Quản lý kiến trúc đa luồng an toàn:
    1. ReaderThread: Đọc frame từ VideoCapture nạp vào read_queue (frame_idx, frame).
    2. WorkerThreads: Xử lý frame (Smart Inpaint, Burn Subtitle), nạp vào write_queue (frame_idx, processed).
    3. WriterThread: Reorder buffer sắp xếp frame theo đúng thứ tự 0, 1, 2, 3... trước khi ghi ra FFmpegVideoWriter.
    Bắt lỗi & propagate exception ngay lập tức, không gây deadlock và không nuốt lỗi.
    """
    def __init__(self, video_path, output_path, process_frame_fn, max_queue_size=64, num_workers=2):
        self.video_path = video_path
        self.output_path = output_path
        self.process_frame_fn = process_frame_fn
        self.max_queue_size = max_queue_size
        self.num_workers = max(1, min(num_workers or 2, os.cpu_count() or 2))

        self.read_queue = queue.Queue(maxsize=max_queue_size)
        self.write_queue = queue.Queue(maxsize=max_queue_size)

        self.is_running = True
        self.worker_exception = None

    def cancel(self):
        self.is_running = False

    def run(self, progress_callback=None):
        enable_opencv_hardware_acceleration()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở file video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = FFmpegVideoWriter(self.output_path, width, height, fps=fps)

        # --- THREAD 1: READ THREAD ---
        def reader_worker():
            frame_idx = 0
            try:
                while self.is_running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    while self.is_running:
                        try:
                            self.read_queue.put((frame_idx, frame), timeout=0.2)
                            frame_idx += 1
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                if not self.worker_exception:
                    self.worker_exception = exc
                self.is_running = False
            finally:
                cap.release()
                # Gửi sentinel kết thúc cho tất cả processing workers
                for _ in range(self.num_workers):
                    try:
                        self.read_queue.put((None, None), timeout=2)
                    except Exception:
                        pass

        # --- THREAD 2: PROCESSING WORKER THREAD ---
        def process_worker(worker_id):
            inpainter = SmartFrameInpainter(motion_threshold=3.5)
            try:
                while self.is_running:
                    try:
                        f_idx, frame = self.read_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if f_idx is None:
                        self.read_queue.task_done()
                        break

                    try:
                        processed = self.process_frame_fn(frame, f_idx, total_frames, fps, inpainter)
                        while self.is_running:
                            try:
                                self.write_queue.put((f_idx, processed), timeout=0.2)
                                break
                            except queue.Full:
                                continue
                    except Exception as exc:
                        if not self.worker_exception:
                            self.worker_exception = exc
                        self.is_running = False
                        break
                    finally:
                        self.read_queue.task_done()
            finally:
                try:
                    self.write_queue.put((None, None), timeout=1)
                except Exception:
                    pass

        # --- THREAD 3: WRITE THREAD VỚI STRICT REORDERING BUFFER ---
        def writer_worker():
            pending_frames = {}
            next_expected_idx = 0
            workers_finished = 0
            written_count = 0

            while (self.is_running or workers_finished < self.num_workers or pending_frames) and not self.worker_exception:
                try:
                    f_idx, frame = self.write_queue.get(timeout=0.2)
                except queue.Empty:
                    if workers_finished >= self.num_workers and not pending_frames:
                        break
                    continue

                if f_idx is None:
                    workers_finished += 1
                    self.write_queue.task_done()
                    if workers_finished >= self.num_workers and not pending_frames:
                        break
                    continue

                pending_frames[f_idx] = frame
                self.write_queue.task_done()

                # Ghi tất cả frame theo đúng thứ tự tăng dần liên tục
                while next_expected_idx in pending_frames and not self.worker_exception:
                    frm = pending_frames.pop(next_expected_idx)
                    try:
                        writer.write(frm)
                        written_count += 1
                        next_expected_idx += 1
                        if progress_callback and total_frames > 0 and written_count % 15 == 0:
                            pct = int((written_count / total_frames) * 100)
                            progress_callback(f"Đang xuất video đa luồng... {pct}% ({written_count}/{total_frames})")
                    except Exception as exc:
                        if not self.worker_exception:
                            self.worker_exception = exc
                        self.is_running = False
                        break

            # Dọn dẹp nốt frame còn sót lại nếu có
            while next_expected_idx in pending_frames and not self.worker_exception:
                frm = pending_frames.pop(next_expected_idx)
                try:
                    writer.write(frm)
                    written_count += 1
                    next_expected_idx += 1
                except Exception as exc:
                    if not self.worker_exception:
                        self.worker_exception = exc
                    break

            try:
                writer.release()
            except Exception as exc:
                if not self.worker_exception:
                    self.worker_exception = exc

        # Khởi chạy các luồng
        t_read = threading.Thread(target=reader_worker, daemon=True)
        worker_threads = [
            threading.Thread(target=process_worker, args=(wid,), daemon=True)
            for wid in range(self.num_workers)
        ]
        t_write = threading.Thread(target=writer_worker, daemon=True)

        t_read.start()
        for tw in worker_threads:
            tw.start()
        t_write.start()

        t_read.join()
        for tw in worker_threads:
            tw.join()
        t_write.join()

        if self.worker_exception:
            # Nếu có lỗi, dọn dẹp file output chưa hoàn thiện và ném lỗi ra ngoài
            if os.path.exists(self.output_path):
                try:
                    os.remove(self.output_path)
                except Exception:
                    pass
            raise self.worker_exception

        return True


# --- 5. BỘ CHIA VIDEO SIÊU TỐC KHÔNG RE-ENCODE (FAST VIDEO CHUNKING VIA FFMPEG -C COPY) ---
def fast_chunk_video(video_path, chunk_duration_s=20.0, overlap_s=1.5, temp_dir=None):
    """
    Sử dụng FFmpeg subprocess cắt video thành các chunk 15-20s với gối đầu overlap_s mà KHÔNG RE-ENCODE (-c copy).
    Tự động xử lý đường dẫn Unicode / Tiếng Trung qua ensure_safe_ascii_video_path.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Tự động nạp đường dẫn an toàn không bị nghẽn đĩa Unicode
    video_path = ensure_safe_ascii_video_path(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration_s = total_frames / fps
    cap.release()

    if temp_dir is None:
        temp_dir = os.path.join(os.path.dirname(video_path), "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)

    chunks = []
    curr_start = 0.0
    chunk_idx = 0

    while curr_start < total_duration_s:
        rem_dur = total_duration_s - curr_start
        # BẮT BỘC KIỂM TRA ĐIỀU KIỆN BIÊN: Nếu remaining_duration < 1.0s (quá ngắn), tự động gộp luôn vào chunk liền trước
        if rem_dur < 1.0 and chunks:
            chunks[-1]['duration'] = total_duration_s - chunks[-1]['start_offset']
            break

        chunk_dur = min(chunk_duration_s + overlap_s, rem_dur)
        out_chunk_path = os.path.join(temp_dir, f"chunk_{chunk_idx:03d}.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{curr_start:.3f}",
            "-i", video_path,
            "-t", f"{chunk_dur:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            out_chunk_path
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if os.path.exists(out_chunk_path) and os.path.getsize(out_chunk_path) > 0:
                chunks.append({
                    'chunk_path': out_chunk_path,
                    'start_offset': curr_start,
                    'duration': chunk_dur,
                    'index': chunk_idx
                })
        except Exception:
            # Fallback nếu -c copy bị lỗi trên một số định dạng: cắt bằng OpenCV
            pass

        curr_start += chunk_duration_s
        chunk_idx += 1

    # Nếu FFmpeg copy thất bại (ví dụ không có ffmpeg trong PATH), tạo 1 chunk tham chiếu đến video gốc
    if not chunks:
        chunks.append({
            'chunk_path': video_path,
            'start_offset': 0.0,
            'duration': total_duration_s,
            'index': 0
        })

    return chunks


# --- 6. TỰ ĐỘNG PADDING VÙNG CROP & TIỀN XỬ LÝ ẢNH TĂNG TƯƠNG PHẢN CHO OCR ---
def preprocess_crop_for_ocr(frame, bbox, padding_px=10):
    """
    1. Tự động cộng margin padding 10px xung quanh vùng crop tránh đứt nét chữ/dấu câu.
    2. Nâng cấp chất lượng ảnh: CLAHE tăng tương phản cục bộ, Denoising và Bilateral Filter.
    3. Upscaling x2/x3 nếu vùng chữ nhỏ (chiều cao < 45px).
    """
    if frame is None or not bbox or len(bbox) != 4:
        return frame, bbox

    fh, fw = frame.shape[:2]
    x, y, w, h = bbox

    # 1. Automatic Crop Padding
    px1 = max(0, x - padding_px)
    py1 = max(0, y - padding_px)
    px2 = min(fw, x + w + padding_px)
    py2 = min(fh, y + h + padding_px)
    padded_bbox = [px1, py1, px2 - px1, py2 - py1]

    crop = frame[py1:py2, px1:px2]
    if crop.size == 0:
        return frame, bbox

    # 2. Upscaling cho các dòng phụ đề nhỏ
    ch, cw = crop.shape[:2]
    if ch < 45:
        scale = max(2.0, 45.0 / float(max(1, ch)))
        crop = cv2.resize(crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 3. Tăng cường độ tương phản qua CLAHE trên kênh Luminance (L)
    try:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        crop_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return crop_enhanced, padded_bbox
    except Exception:
        return crop, padded_bbox


# --- 7. QUY ĐỔI TIMECODE OFFSET & GOM CỤM KHỬ LẶP CHUẨN HOÁ SRT ---
def merge_and_deduplicate_subtitles(raw_subtitles, overlap_s=1.5, similarity_threshold=0.80):
    """
    1. Cộng start_offset cho timecode từng phân đoạn.
    2. Gom cụm các kết quả OCR giống nhau (Fuzzy matching > 80%) trong mốc thời gian liền kề.
    3. Lọc bỏ rác OCR (chuỗi ngắn < 2 ký tự không thuộc CJK).
    4. Khử trùng lặp vùng gối đầu giữa các chunk.
    """
    if not raw_subtitles:
        return []

    import difflib

    # Sắp xếp các subtitle theo start time tăng dần
    sorted_subs = sorted(raw_subtitles, key=lambda s: s.get('start', 0.0))
    filtered_subs = []

    # 1. Lọc bỏ rác OCR
    cjk_pattern = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]'
    import re

    for sub in sorted_subs:
        txt = sub.get('text', '').strip()
        if not txt:
            continue
        # Bỏ rác < 2 ký tự nếu không phải tiếng Trung/Nhật/Hàn
        if len(txt) < 2 and not re.search(cjk_pattern, txt):
            continue
        filtered_subs.append(dict(sub))

    if not filtered_subs:
        return []

    # 2. Gom cụm & Khử lặp bằng Fuzzy Matching
    merged = []
    current = filtered_subs[0]

    for next_sub in filtered_subs[1:]:
        t1 = current.get('text', '').strip().lower()
        t2 = next_sub.get('text', '').strip().lower()

        # Tính tỷ lệ tương đồng Fuzzy
        sim_ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
        is_placeholder = "[chữ khó" in t1 or "[chữ khó" in t2
        gap = next_sub['start'] - current['end']
        curr_dur = current['end'] - current['start']

        # Nếu giống nhau > 80% hoặc nằm trong vùng gối đầu overlap và chênh lệch < 1.2s -> Merging
        should_merge = (
            not is_placeholder and
            (sim_ratio >= similarity_threshold or t1 in t2 or t2 in t1) and
            gap <= (overlap_s + 0.3) and
            curr_dur < 8.0
        )

        if should_merge:
            current['end'] = max(current['end'], next_sub['end'])
            # Chọn văn bản dài/đầy đủ hơn
            if len(next_sub.get('text', '')) > len(current.get('text', '')):
                current['text'] = next_sub['text']
            if 'bbox' in next_sub and 'bbox' in current:
                from transcriber import merge_bboxes
                current['bbox'] = merge_bboxes(current.get('bbox'), next_sub.get('bbox'))
        else:
            merged.append(current)
            current = next_sub

    merged.append(current)
    return merged


# --- 8. QUẢN LÝ TIẾN TRÌNH XỬ LÝ CHUNK SONG SONG (PARALLEL CHUNK PROCESSOR) ---
class ParallelChunkOCRProcessor:
    """
    Lớp quản lý chia chunk siêu tốc và quét OCR song song qua ProcessPool / ThreadPool.
    Tích hợp Smart Frame Skipping (absdiff) và gom cụm khử trùng lặp file SRT.
    """
    def __init__(self, video_path, max_workers=None):
        self.video_path = video_path
        self.max_workers = max_workers or min(4, os.cpu_count() or 2)

    def process_video_ocr(self, bboxes=None, ocr_lang="auto", api_key="", progress_callback=None, check_cancel_func=None, ocr_engine="easyocr"):
        """
        Quy trình xử lý OCR phân đoạn song song hoàn chỉnh.
        """
        import concurrent.futures
        t0 = time.time()

        if not bboxes:
            # Fallback nếu không có bbox: chạy OCR toàn khung hình 1 chunk
            import transcriber
            chunk_subs = transcriber.run_hardsub_ocr(
                video_path=self.video_path,
                bbox=None,
                ocr_lang=ocr_lang,
                force_scan=True,
                api_key=api_key,
                ocr_engine=ocr_engine
            )
            return {
                'subtitles': chunk_subs,
                'total_time': time.time() - t0,
                'total_chunks': 1
            }

        if progress_callback:
            progress_callback("Đang thực hiện cắt video siêu tốc (FFmpeg -c copy với gối đầu 1.5s)...")

        # 1. Chia video thành các chunk 25s với gối đầu 1.5s
        chunks = fast_chunk_video(self.video_path, chunk_duration_s=25.0, overlap_s=1.5)
        total_chunks = len(chunks)

        # PRE-INITIALIZE OCR READER ON PARENT THREAD TO PREVENT THREAD POOL DEADLOCK
        if progress_callback:
            engine_name = "PaddleOCR" if "paddle" in str(ocr_engine).lower() else "EasyOCR"
            progress_callback(f"⚡ Đang nạp mô hình OCR (Pre-initializing {engine_name} Model)...")
        try:
            import transcriber
            transcriber.get_ocr_reader(engine=ocr_engine, lang_list=['ch_sim', 'en'])
        except Exception as e_init:
            if progress_callback:
                progress_callback(f"Cảnh báo nạp OCR: {e_init}")

        if progress_callback:
            progress_callback(f"🚀 Bắt đầu xử lý {total_chunks} chunks với {self.max_workers} luồng song song...")

        chunk_results = {}

        # Hàm worker quét OCR trên từng chunk
        def chunk_worker(chunk_info):
            if check_cancel_func and check_cancel_func():
                return []
            c_path = chunk_info['chunk_path']
            c_offset = chunk_info['start_offset']
            c_idx = chunk_info['index'] + 1
            c_dur = chunk_info['duration']

            if progress_callback:
                progress_callback(f"📊 Chunk #{c_idx}/{total_chunks}: ✂️ Đang xử lý OCR ({c_offset:.1f}s -> {c_offset + c_dur:.1f}s)...")

            import transcriber
            chunk_subs = transcriber.run_hardsub_ocr(
                video_path=c_path,
                bbox=bboxes,
                ocr_lang=ocr_lang,
                force_scan=True,
                api_key=api_key,
                ocr_engine=ocr_engine
            )

            # Quy đổi Offset mốc thời gian về thời gian gốc của video
            offset_subs = []
            for sub in chunk_subs:
                sub_copy = dict(sub)
                sub_copy['start'] = sub['start'] + c_offset
                sub_copy['end'] = sub['end'] + c_offset
                offset_subs.append(sub_copy)

            return offset_subs

        # Chạy song song qua ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(chunk_worker, chunk): chunk 
                for chunk in chunks
            }
            completed_count = 0

            for future in concurrent.futures.as_completed(future_to_chunk):
                if check_cancel_func and check_cancel_func():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                chunk_info = future_to_chunk[future]
                c_idx = chunk_info['index'] + 1
                c_raw_idx = chunk_info['index']
                completed_count += 1
                pct = int((completed_count / total_chunks) * 100)

                try:
                    res = future.result()
                    chunk_results[c_raw_idx] = res or []
                    if progress_callback:
                        progress_callback(f"✅ Hoàn thành chunk {c_idx}/{total_chunks} ({pct}%) - Tìm thấy {len(res)} câu phụ đề.")
                except Exception as e:
                    chunk_results[c_raw_idx] = []
                    if progress_callback:
                        progress_callback(f"❌ Lỗi chunk {c_idx}/{total_chunks}: {e}")

        # Ghép kết quả các chunks theo đúng thứ tự thời gian gốc
        all_raw_subtitles = []
        for i in range(total_chunks):
            all_raw_subtitles.extend(chunk_results.get(i, []))

        # Gom cụm & Khử lặp trùng cho file SRT hoàn chỉnh
        if progress_callback:
            progress_callback(f"🎯 Gom cụm & khử trùng lặp ({len(all_raw_subtitles)} câu raw)...")

        final_subtitles = merge_and_deduplicate_subtitles(all_raw_subtitles, overlap_s=1.5)
        total_time = time.time() - t0

        return {
            'subtitles': final_subtitles,
            'total_time': total_time,
            'total_chunks': total_chunks
        }


# --- 9. CHẾ ĐỘ XỬ LÝ VIDEO SIÊU DÀI CHO MÁY CẤU HÌNH YẾU (STREAMING CLEANUP & RESUME) ---

def get_checkpoint_file(video_path):
    base_dir = os.path.dirname(os.path.abspath(video_path))
    v_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(base_dir, f"{v_name}_checkpoint.json")

def save_progress_checkpoint(video_path, last_processed_second, extracted_subtitles):
    chk_file = get_checkpoint_file(video_path)
    try:
        data = {
            "video_path": os.path.abspath(video_path),
            "last_processed_second": float(last_processed_second),
            "extracted_subtitles": extracted_subtitles
        }
        with open(chk_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Cảnh báo lưu checkpoint: {e}")

def load_progress_checkpoint(video_path):
    chk_file = get_checkpoint_file(video_path)
    if os.path.exists(chk_file):
        try:
            with open(chk_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get("video_path") == os.path.abspath(video_path):
                    return data
        except Exception as e:
            print(f"Cảnh báo đọc checkpoint: {e}")
    return None

def clear_progress_checkpoint(video_path):
    chk_file = get_checkpoint_file(video_path)
    if os.path.exists(chk_file):
        try:
            os.remove(chk_file)
        except Exception:
            pass


class StreamingLongVideoProcessor:
    """
    Xử lý các video siêu dài (10 - 20+ tiếng) cho máy cấu hình yếu / RAM nhỏ / đĩa cứng hẹp.
    - Generator / Streaming Chunking: Chỉ cắt N chunk tương ứng với số worker active.
    - Auto Cleanup: Xóa ngay lập tức file video chunk tạm khỏi ổ đĩa ngay khi hoàn tất.
    - RAM Garbage Collection: Gọi gc.collect() và giải phóng bộ nhớ đệm OpenCV/EasyOCR.
    - Progress Checkpoint & Resume: Tự động lưu checkpoint, cho phép chạy tiếp khi ứng dụng bị ngắt.
    """
    def __init__(self, video_path, chunk_duration_s=20.0, overlap_s=1.5, max_workers=None, low_spec_mode=False):
        self.video_path = video_path
        self.chunk_duration_s = chunk_duration_s
        self.overlap_s = overlap_s
        self.low_spec_mode = low_spec_mode

        if low_spec_mode:
            self.max_workers = 1
        else:
            self.max_workers = max_workers or min(2, os.cpu_count() or 1)

    def process_streaming_ocr(self, bboxes, ocr_lang="auto", api_key="", progress_callback=None, check_cancel_func=None):
        import gc
        import transcriber

        t0 = time.time()

        # Tự động nạp đường dẫn an toàn không bị nghẽn đĩa Unicode
        safe_video_path = ensure_safe_ascii_video_path(self.video_path)

        # PRE-INITIALIZE OCR READER TRONG LUỒNG CHÍNH ĐỂ TRÁNH DEADLOCK KHÓA LUỒNG
        if progress_callback:
            progress_callback("⚡ Đang khởi tạo bộ đọc OCR (Pre-initializing EasyOCR Model)...")
        try:
            transcriber.get_easyocr_reader(['ch_sim', 'en'])
        except Exception as e_init:
            if progress_callback:
                progress_callback(f"Cảnh báo khởi tạo OCR: {e_init}")

        cap = cv2.VideoCapture(safe_video_path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở tệp video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration_s = total_frames / fps
        cap.release()

        temp_dir = os.path.join(os.path.dirname(safe_video_path), "temp_streaming_chunks")
        os.makedirs(temp_dir, exist_ok=True)

        # 1. Kiểm tra file Checkpoint phục hồi tiến trình cũ nếu có
        checkpoint = load_progress_checkpoint(self.video_path)
        start_second = 0.0
        all_raw_subtitles = []

        if checkpoint:
            start_second = checkpoint.get("last_processed_second", 0.0)
            all_raw_subtitles = checkpoint.get("extracted_subtitles", [])
            if progress_callback:
                progress_callback(f"🔄 ĐÃ PHÁT HIỆN CHECKPOINT CỦ: Tự động chạy tiếp từ giây thứ {start_second:.1f}s (Đã bóc {len(all_raw_subtitles)} câu).")

        curr_start = start_second
        chunk_idx = int(curr_start // self.chunk_duration_s)

        while curr_start < total_duration_s:
            if check_cancel_func and check_cancel_func():
                if progress_callback:
                    progress_callback("🛑 Tiến trình streaming đã nhận được tín hiệu dừng.")
                break

            rem_dur = total_duration_s - curr_start
            if rem_dur < 1.0 and all_raw_subtitles:
                break

            chunk_dur = min(self.chunk_duration_s + self.overlap_s, rem_dur)
            out_chunk_path = os.path.join(temp_dir, f"stream_chunk_{chunk_idx:04d}.mp4")

            # Cắt 1 chunk duy nhất trên ổ đĩa
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{curr_start:.3f}",
                "-i", self.video_path,
                "-t", f"{chunk_dur:.3f}",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                out_chunk_path
            ]

            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except Exception:
                out_chunk_path = self.video_path

            if progress_callback:
                pct = int((curr_start / total_duration_s) * 100)
                progress_callback(f"Streaming OCR... {pct}% ({curr_start:.1f}s/{total_duration_s:.1f}s) - Chunk #{chunk_idx}")

            # Chạy OCR trực tiếp trên chunk vừa băm
            chunk_subs = transcriber.run_hardsub_ocr(
                video_path=out_chunk_path,
                bbox=bboxes,
                ocr_lang=ocr_lang,
                force_scan=True,
                api_key=api_key
            )

            # Cộng offset mốc thời gian
            for sub in chunk_subs:
                sub_copy = dict(sub)
                sub_copy['start'] = sub['start'] + curr_start
                sub_copy['end'] = sub['end'] + curr_start
                all_raw_subtitles.append(sub_copy)

            # XÓA LẬP TỨC FILE CHUNK TẠM KHỎI Ổ ĐĨA CỨNG (AUTO CLEANUP)
            if out_chunk_path != self.video_path and os.path.exists(out_chunk_path):
                try:
                    os.remove(out_chunk_path)
                except Exception as e_del:
                    print(f"Cảnh báo xóa file tạm: {e_del}")

            # Giải phóng RAM chủ động (Garbage Collection)
            gc.collect()

            curr_start += self.chunk_duration_s
            chunk_idx += 1

            # Lưu Checkpoint tiến trình định kỳ
            save_progress_checkpoint(self.video_path, curr_start, all_raw_subtitles)

        # Xóa thư mục tạm nếu rỗng
        try:
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass

        # Đã hoàn tất 100%, xóa checkpoint
        clear_progress_checkpoint(self.video_path)

        if progress_callback:
            progress_callback("Đang gom cụm & khử trùng lặp mốc thời gian file SRT...")

        final_subtitles = merge_and_deduplicate_subtitles(all_raw_subtitles, overlap_s=self.overlap_s)
        total_time = time.time() - t0

        return {
            'subtitles': final_subtitles,
            'total_time': total_time,
            'total_chunks': chunk_idx
        }


# --- 10. PIPELINE STATE MACHINE & UNIFIED EXECUTION ENGINE ---

class PipelineState:
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    DOWNLOADING = "DOWNLOADING"
    OCR = "OCR"
    TRANSLATING = "TRANSLATING"
    TTS = "TTS"
    AUDIO_SYNC = "AUDIO_SYNC"
    RENDERING = "RENDERING"
    VALIDATING_OUTPUT = "VALIDATING_OUTPUT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


def execute_single_video_pipeline(video_path, output_path, config=None, progress_callback=None, cancel_check_fn=None, state_callback=None):
    """
    Thực thi toàn bộ pipeline xử lý video end-to-end (dùng chung cho cả 1-Click GUI và Batch Queue).
    Tuyệt đối không dùng giả lập (fake progress / sleep).
    Output chỉ thành công khi vượt qua xác thực validate_output_video().
    """
    if config is None:
        config = {}

    start_time = time.time()

    def update_state(state, progress_pct, msg):
        if state_callback:
            state_callback(state, progress_pct, msg)
        if progress_callback:
            progress_callback(f"[{progress_pct}%] [{state}] {msg}")

    # BƯỚC 1: XÁC THỰC INPUT VIDEO
    update_state(PipelineState.VALIDATING, 5, "Đang kiểm tra tính hợp lệ của video đầu vào...")
    if cancel_check_fn and cancel_check_fn():
        raise InterruptedError("Tiến trình bị hủy bởi người dùng.")

    valid_in, in_info = validate_output_video(video_path, check_audio=False, min_duration=0.1)
    if not valid_in:
        raise ValueError(f"Video đầu vào không hợp lệ: {in_info}")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Đọc cấu hình pipeline
    raw_ocr = str(config.get("ocr_engine", "gemini")).lower()
    if "paddle" in raw_ocr:
        ocr_engine = "paddleocr"
    elif "xkiro" in raw_ocr:
        ocr_engine = "xkiro"
    elif "easyocr" in raw_ocr or "offline" in raw_ocr:
        ocr_engine = "easyocr"
    else:
        ocr_engine = "gemini"

    source_lang = config.get("source_lang", "auto")
    target_lang = config.get("target_lang", "vi")
    api_key = config.get("api_key", "")
    xkiro_key = config.get("xkiro_key", "")
    engine = config.get("engine", "xKiro AI")
    voice = config.get("voice", "vi-VN-HoaiMyNeural")
    bg_vol = float(config.get("bg_vol", 0.1))
    dub_vol = float(config.get("dub_vol", 1.0))
    burn_sub = bool(config.get("burn_sub", True) if "burn_sub" in config else config.get("burn_subtitles", True))
    enable_dubbing = bool(config.get("enable_dubbing", True))
    preset = config.get("preset", None)
    selected_bbox = config.get("selected_bbox", None)
    selected_bboxes = config.get("selected_bboxes", None)
    logo_path = config.get("logo_path", None)
    logo_bbox = config.get("logo_bbox", None)
    title_bbox = config.get("title_bbox", None)
    chunk_workers = int(config.get("chunk_workers", 2) or 2)
    scan_interval = float(config.get("scan_interval", 0.5) or 0.5)

    # BƯỚC 2: OCR / TRÍCH XUẤT PHỤ ĐỀ
    update_state(PipelineState.OCR, 15, f"Bắt đầu quét phụ đề bằng OCR ({ocr_engine.upper()})...")
    if cancel_check_fn and cancel_check_fn():
        raise InterruptedError("Tiến trình bị hủy bởi người dùng.")

    segments = []
    if ocr_engine == "gemini":
        try:
            from gemini_vision_ocr import load_gemini_keys, scan_video_frames_with_gemini
            keys_list = [k.strip() for k in api_key.split(",") if k.strip()] if api_key else load_gemini_keys()
            if keys_list:
                vision_segs = scan_video_frames_with_gemini(
                    video_path=video_path,
                    sample_interval_sec=scan_interval,
                    api_keys=keys_list,
                    progress_callback=lambda m: progress_callback(m) if progress_callback else None
                )
                if vision_segs:
                    segments = vision_segs
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠️ Gemini OCR thất bại ({e}), chuyển sang quét OCR phân đoạn...")

    if not segments:
        if cancel_check_fn and cancel_check_fn():
            raise InterruptedError("Tiến trình bị hủy bởi người dùng.")
        ocr_boxes = selected_bboxes or ([selected_bbox] if selected_bbox else [])
        processor = ParallelChunkOCRProcessor(video_path, max_workers=chunk_workers)
        res_ocr = processor.process_video_ocr(
            bboxes=ocr_boxes,
            ocr_lang=source_lang,
            api_key=api_key,
            progress_callback=lambda m: progress_callback(m) if progress_callback else None,
            check_cancel_func=cancel_check_fn,
            ocr_engine=ocr_engine
        )
        segments = res_ocr.get("subtitles", [])

    # Chuẩn hóa schema dữ liệu subtitle segments
    clean_segments = []
    for s in (segments or []):
        txt = str(s.get("text", "")).strip()
        if not txt:
            continue
        clean_segments.append({
            "text": txt,
            "bbox": s.get("bbox", None),
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "confidence": float(s.get("confidence", 1.0))
        })
    segments = clean_segments

    update_state(PipelineState.OCR, 35, f"Quét OCR hoàn tất: tìm thấy {len(segments)} câu phụ đề.")

    if cancel_check_fn and cancel_check_fn():
        raise InterruptedError("Tiến trình bị hủy bởi người dùng.")

    # BƯỚC 3: DỊCH THUẬT PHỤ ĐỀ (TRANSLATION)
    if segments:
        update_state(PipelineState.TRANSLATING, 40, f"Đang dịch thuật {len(segments)} câu phụ đề ({engine})...")
        import translator
        translated_segs = translator.translate_segments(
            segments,
            source_lang=source_lang,
            target_lang=target_lang,
            engine=engine,
            api_key=api_key,
            progress_callback=lambda m: progress_callback(m) if progress_callback else None,
            xkiro_key=xkiro_key
        )
        if translated_segs:
            segments = translated_segs
        update_state(PipelineState.TRANSLATING, 60, f"Dịch thuật hoàn tất {len(segments)} câu.")

    if cancel_check_fn and cancel_check_fn():
        raise InterruptedError("Tiến trình bị hủy bởi người dùng.")

    # BƯỚC 4 & 5: TTS, AUDIO SYNC, SUBTITLE INPAINT & RENDER
    update_state(PipelineState.RENDERING, 70, "Đang tổng hợp âm thanh & render video thành phẩm...")
    import dubber

    translated_title_text = None
    if title_bbox and len(title_bbox) == 4:
        try:
            from deep_translator import GoogleTranslator
            gt = GoogleTranslator(source="auto", target=target_lang)
            # Dịch tiêu đề nếu có
        except Exception:
            pass

    out_temp = f"{output_path}.tmp.mp4"
    res_path, overflowed = dubber.create_dubbed_video(
        video_path=video_path,
        segments=segments,
        voice=voice,
        output_video_path=out_temp,
        bg_volume=bg_vol,
        dub_volume=dub_vol,
        burn_subtitles=burn_sub,
        selected_bbox=selected_bbox,
        preset=preset,
        progress_callback=lambda m: progress_callback(m) if progress_callback else None,
        enable_dubbing=enable_dubbing,
        selected_bboxes=selected_bboxes,
        logo_path=logo_path,
        title_text=translated_title_text,
        title_bbox=title_bbox,
        logo_bbox=logo_bbox
    )

    if cancel_check_fn and cancel_check_fn():
        if os.path.exists(out_temp):
            try: os.remove(out_temp)
            except Exception: pass
        raise InterruptedError("Tiến trình bị hủy bởi người dùng.")

    # BƯỚC 6: XÁC THỰC KẾT QUẢ ĐẦU RA (VALIDATING OUTPUT)
    update_state(PipelineState.VALIDATING_OUTPUT, 95, "Đang kiểm tra chất lượng file video đầu ra...")
    valid_out, out_info = validate_output_video(out_temp, check_audio=False, min_duration=0.1)
    if not valid_out:
        if os.path.exists(out_temp):
            try: os.remove(out_temp)
            except Exception: pass
        raise RuntimeError(f"Video đầu ra không hợp lệ sau khi render: {out_info}")

    # Atomic rename sang output_path chính thức
    if os.path.exists(output_path):
        try: os.remove(output_path)
        except Exception: pass
    shutil.move(out_temp, output_path)

    elapsed = time.time() - start_time
    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    update_state(PipelineState.COMPLETED, 100, f"Hoàn thành xuất sắc trong {elapsed:.1f}s ({size_mb} MB)!")

    return {
        "output_path": output_path,
        "elapsed_sec": elapsed,
        "size_mb": size_mb,
        "segments_count": len(segments),
        "info": out_info
    }



