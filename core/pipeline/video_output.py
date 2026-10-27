"""
輸出標註影片: 開影片檔、畫偵測框 / 測速區
"""

import cv2
import numpy as np


def open_video_writer(path, fps, size):
    """優先用 H.264 (瀏覽器可直接播放),這台電腦的 OpenCV 不支援就退回 mp4v"""
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    return None


def write_clip(video_path, path, first_raw, last_raw, boxes, speed_zone=None, max_gap=1, max_w=1280, max_fps=30):
    """
    從原始影片剪出第 first_raw ~ last_raw 幀 (跟 processor 的 raw_frame_idx 一樣從 1 開始) 存成片段。
    boxes = {raw_frame_idx: (box, 文字)} 這台車在處理過的幀的位置; 跳過的幀沿用前一個框, 超過 max_gap 幀沒看到就不畫
    畫面先縮小再畫框 (4K 影片在原尺寸畫框, 縮小後線條和文字會細到看不清楚); 60fps 影片只取一半的幀, 檔案小一半
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = min(max_w, w)
    out_h = int(h * out_w / w)
    scale = out_w / w
    step = max(1, round(fps / max_fps))
    writer = open_video_writer(path, fps / step, (out_w, out_h))
    if writer is None:
        cap.release()
        return False
    zone = speed_zone * scale if speed_zone is not None else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_raw - 1)
    last_box, last_seen = None, None
    for raw in range(first_raw, last_raw + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if raw in boxes:
            last_box, last_seen = boxes[raw], raw
        if (raw - first_raw) % step:
            continue
        frame = cv2.resize(frame, (out_w, out_h))
        drawn = []
        if last_box and raw - last_seen <= max_gap:
            box = tuple(int(v * scale) for v in last_box[0])
            drawn.append((box, (0, 0, 255), last_box[1]))
        draw_annotations(frame, drawn, zone)
        writer.write(frame)
    cap.release()
    writer.release()
    return True


def draw_annotations(frame, drawn, speed_zone=None):
    """drawn = [(box, 顏色, 文字), ...]; 有測速區就畫黃色多邊形"""
    if speed_zone is not None:
        cv2.polylines(frame, [speed_zone.astype(np.int32)], True, (0, 255, 255), 2)
    for (bx1, by1, bx2, by2), color, text in drawn:
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(frame, text, (bx1, max(by1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
