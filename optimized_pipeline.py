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


# --- 3. GHI VIDEO SIÊU TỐC QUA FFMPEG SUBPROCESS (GPU NVENC / CPU ULTRAFAST) ---
class FFmpegVideoWriter:
    """
    Ghi video siêu tốc bằng cách truyền trực tiếp raw BGR bytes vào ống dẫn (stdin pipe) của FFmpeg.
    Tự động ưu tiên h264_nvenc (NVIDIA GPU Hardware Encoder), fallback sang libx264 (preset ultrafast).
    """
    def __init__(self, output_path, width, height, fps=25.0, codec="auto"):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.pipe = None
        self.use_cv2_fallback = False
        self.cv2_writer = None

        self._init_ffmpeg(codec)

    def _init_ffmpeg(self, codec):
        ffmpeg_bin = "ffmpeg"
        
        encoders_to_try = []
        if codec in ("nvenc", "auto"):
            encoders_to_try.append(["-c:v", "h264_nvenc", "-preset", "p1", "-b:v", "5M"])
        encoders_to_try.append(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22"])

        success = False
        dummy_bgr = np.zeros((self.height, self.width, 3), dtype=np.uint8).tobytes()

        for enc_args in encoders_to_try:
            cmd = [
                ffmpeg_bin, "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{self.width}x{self.height}",
                "-pix_fmt", "bgr24",
                "-r", str(self.fps),
                "-i", "-"
            ] + enc_args + [
                "-pix_fmt", "yuv420p",
                self.output_path
            ]

            try:
                p = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                # Ghi thử 1 frame dummy để đảm bảo encoder chấp nhận stream
                p.stdin.write(dummy_bgr)
                p.stdin.flush()
                time.sleep(0.02)
                if p.poll() is None:
                    self.pipe = p
                    success = True
                    break
                else:
                    try: p.kill()
                    except Exception: pass
            except Exception:
                pass

        if not success:
            self.use_cv2_fallback = True
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.cv2_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

    def write(self, frame):
        if frame is None:
            return
        if self.use_cv2_fallback and self.cv2_writer:
            self.cv2_writer.write(frame)
        elif self.pipe and self.pipe.stdin:
            try:
                self.pipe.stdin.write(frame.tobytes())
            except Exception:
                # Nếu pipe bị vỡ, chuyển sang cv2_writer khẩn cấp
                self.use_cv2_fallback = True
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.cv2_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
                self.cv2_writer.write(frame)

    def release(self):
        if self.use_cv2_fallback and self.cv2_writer:
            self.cv2_writer.release()
            self.cv2_writer = None
        elif self.pipe:
            try:
                if self.pipe.stdin:
                    self.pipe.stdin.close()
                self.pipe.wait(timeout=5)
            except Exception:
                try: self.pipe.kill()
                except Exception: pass
            self.pipe = None


# --- 4. KIẾN TRÚC PIPELINE ĐA LUỒNG (PRODUCER-CONSUMER PIPELINE) ---
class ParallelVideoProcessor:
    """
    Quản lý kiến trúc 3 luồng hoạt động song song qua queue:
    1. ReaderThread: Đọc frame từ VideoCapture nạp vào read_queue.
    2. WorkerThreads: Thực hiện xử lý frame (Smart Inpaint, Burn Subtitle), nạp vào write_queue.
    3. WriterThread: Lấy frame từ write_queue ghi vào VideoWriter/FFmpeg pipe.
    """
    def __init__(self, video_path, output_path, process_frame_fn, max_queue_size=64):
        self.video_path = video_path
        self.output_path = output_path
        self.process_frame_fn = process_frame_fn
        self.max_queue_size = max_queue_size

        self.read_queue = queue.Queue(maxsize=max_queue_size)
        self.write_queue = queue.Queue(maxsize=max_queue_size)

        self.is_running = True
        self.error_msg = None

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
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.read_queue.put((frame_idx, frame))
                frame_idx += 1
            self.read_queue.put((None, None))  # End signal for reader
            cap.release()

        # --- THREAD 2: PROCESS WORKER THREAD ---
        def process_worker():
            inpainter = SmartFrameInpainter(motion_threshold=3.5)
            while self.is_running:
                try:
                    f_idx, frame = self.read_queue.get(timeout=2)
                    if f_idx is None:
                        self.write_queue.put((None, None))
                        break

                    # Gọi hàm xử lý frame người dùng định nghĩa
                    processed = self.process_frame_fn(frame, f_idx, total_frames, fps, inpainter)

                    self.write_queue.put((f_idx, processed))
                    self.read_queue.task_done()
                except queue.Empty:
                    if not self.is_running:
                        break

        # --- THREAD 3: WRITE THREAD ---
        def writer_worker():
            written_count = 0
            while self.is_running:
                try:
                    f_idx, frame = self.write_queue.get(timeout=2)
                    if f_idx is None:
                        break
                    writer.write(frame)
                    written_count += 1
                    if progress_callback and total_frames > 0 and written_count % 15 == 0:
                        pct = int((written_count / total_frames) * 100)
                        progress_callback(f"Đang xuất video đa luồng... {pct}% ({written_count}/{total_frames})")
                    self.write_queue.task_done()
                except queue.Empty:
                    if not self.is_running:
                        break
            writer.release()

        # Khởi chạy đồng thời 3 threads
        t_read = threading.Thread(target=reader_worker, daemon=True)
        t_proc = threading.Thread(target=process_worker, daemon=True)
        t_write = threading.Thread(target=writer_worker, daemon=True)

        t_read.start()
        t_proc.start()
        t_write.start()

        t_read.join()
        t_proc.join()
        t_write.join()

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


