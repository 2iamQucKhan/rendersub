import cv2
import numpy as np
import time

def draw_visual_feedback_overlay(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    timestamp_s: float,
    status_text: str = "PROCESSING...",
    active_bbox=None,
    scanline_state: int = 0
) -> np.ndarray:
    """
    Vẽ các lớp phản hồi trực quan (Visual Feedback Overlays) đè lên OpenCV BGR frame:
    - Bounding Box / Crop Region nhấp nháy phát sáng (Pulsing glowing rect)
    - Tia laser quét Scanline di chuyển trong vùng Crop
    - Banner trạng thái mờ (Dark semi-transparent banner) với Progress %, Frame Index, Timestamp
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Chuẩn hóa active_bbox thành danh sách các box [(x, y, w, h)]
    bboxes = []
    if active_bbox:
        if isinstance(active_bbox, (list, tuple)):
            if len(active_bbox) == 4 and isinstance(active_bbox[0], (int, float)):
                bboxes = [active_bbox]
            else:
                bboxes = [b for b in active_bbox if isinstance(b, (list, tuple)) and len(b) == 4]

    # 1. Vẽ Bounding Box & Hiệu ứng Scanline trên từng vùng Crop
    for idx, (bx, by, bw, bh) in enumerate(bboxes):
        bx = int(max(0, min(bx, w - 1)))
        by = int(max(0, min(by, h - 1)))
        bw = int(max(1, min(bw, w - bx)))
        bh = int(max(1, min(bh, h - by)))

        # Màu Bounding Box phát sáng nhấp nháy (Xanh Neon / Cyan)
        pulse = (np.sin(time.time() * 8.0 + idx) + 1.0) / 2.0  # Range 0.0 - 1.0
        r_val = int(30 + 50 * pulse)
        g_val = int(220 + 35 * pulse)
        b_val = int(255)
        pulse_color = (b_val, g_val, r_val)

        # Vẽ hình chữ nhật vùng Crop
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), pulse_color, 2)

        # Vẽ 4 góc định vị (Target Corners) dày dặn
        c_len = max(8, min(20, min(bw, bh) // 4))
        # Top-Left
        cv2.line(annotated, (bx, by), (bx + c_len, by), pulse_color, 3)
        cv2.line(annotated, (bx, by), (bx, by + c_len), pulse_color, 3)
        # Top-Right
        cv2.line(annotated, (bx + bw, by), (bx + bw - c_len, by), pulse_color, 3)
        cv2.line(annotated, (bx + bw, by), (bx + bw, by + c_len), pulse_color, 3)
        # Bottom-Left
        cv2.line(annotated, (bx, by + bh), (bx + c_len, by + bh), pulse_color, 3)
        cv2.line(annotated, (bx, by + bh), (bx, by + bh - c_len), pulse_color, 3)
        # Bottom-Right
        cv2.line(annotated, (bx + bw, by + bh), (bx + bw - c_len, by + bh), pulse_color, 3)
        cv2.line(annotated, (bx + bw, by + bh), (bx + bw, by + bh - c_len), pulse_color, 3)

        # Hiệu ứng tia laser Scanline chạy dọc vùng Crop
        scan_pos = (int(scanline_state * 6 + idx * 10) % bh)
        scan_y = by + scan_pos
        cv2.line(annotated, (bx + 2, scan_y), (bx + bw - 2, scan_y), (0, 84, 255), 2)
        # Bóng laser nhẹ
        if scan_y + 1 < by + bh:
            cv2.line(annotated, (bx + 2, scan_y + 1), (bx + bw - 2, scan_y + 1), (50, 180, 255), 1)

    # 2. Vẽ Banner Trạng Thái Bán Trong Suốt (Status Overlay Banner) ở góc trên video
    overlay_h = 70
    overlay_w = min(420, w - 20)
    if overlay_w > 150 and h > 80:
        box_x1, box_y1 = 10, 10
        box_x2, box_y2 = box_x1 + overlay_w, box_y1 + overlay_h

        # Lớp overlay đen mờ
        overlay_mask = annotated.copy()
        cv2.rectangle(overlay_mask, (box_x1, box_y1), (box_x2, box_y2), (18, 24, 33), -1)
        cv2.addWeighted(overlay_mask, 0.75, annotated, 0.25, 0, annotated)

        # Viền sắc nét màu xanh cyan cho banner
        cv2.rectangle(annotated, (box_x1, box_y1), (box_x2, box_y2), (0, 180, 240), 1)
        # Vệt màu nhấn bên trái banner
        cv2.rectangle(annotated, (box_x1, box_y1), (box_x1 + 5, box_y2), (0, 220, 255), -1)

        # Tính phần trăm tiến trình %
        pct = 0.0
        if total_frames > 0:
            pct = min(100.0, max(0.0, (frame_idx / float(total_frames)) * 100.0))

        # Định dạng thời gian
        mins = int(timestamp_s // 60)
        secs = int(timestamp_s % 60)
        ms = int((timestamp_s % 1.0) * 100)
        time_str = f"{mins:02d}:{secs:02d}.{ms:02d}"

        # Text 1: Tiêu đề trạng thái
        clean_status = status_text[:40] if status_text else "PROCESSING..."
        cv2.putText(
            annotated,
            clean_status,
            (box_x1 + 15, box_y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Text 2: Tiến trình % & Frame / Timestamp
        info_str = f"Progress: {pct:5.1f}%  |  Frame: {frame_idx}/{total_frames}  |  {time_str}"
        cv2.putText(
            annotated,
            info_str,
            (box_x1 + 15, box_y1 + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (180, 220, 255),
            1,
            cv2.LINE_AA
        )

        # Thanh Progress Bar nhỏ bên dưới banner
        pb_x1, pb_y1 = box_x1 + 15, box_y1 + 52
        pb_w = overlay_w - 30
        pb_h = 6
        cv2.rectangle(annotated, (pb_x1, pb_y1), (pb_x1 + pb_w, pb_y1 + pb_h), (40, 50, 65), -1)
        fill_w = int(pb_w * (pct / 100.0))
        if fill_w > 0:
            cv2.rectangle(annotated, (pb_x1, pb_y1), (pb_x1 + fill_w, pb_y1 + pb_h), (0, 210, 255), -1)

        # Chấm nhấp nháy báo động đang chạy (Active Recording / Scan Indicator)
        if int(time.time() * 3) % 2 == 0:
            cv2.circle(annotated, (box_x2 - 18, box_y1 + 20), 5, (0, 0, 255), -1)

    return annotated
