"""
各頁面共用的路徑和工具函式: 影片讀取、JSON 檔、CSV 匯出、從網址取得選取的實驗
"""

import os
import io
import csv
import json

import cv2
from flask import Response, abort, request

from core import db
from core.pipeline import BASE_DIR

VIDEO_DIR = os.path.join(BASE_DIR, "test_videos")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
STATUS_TEXT = {"queued": "排隊中", "running": "執行中", "done": "完成", "stopped": "已停止", "failed": "失敗"}


# ===== 影片 =====
def list_videos():
    if not os.path.isdir(VIDEO_DIR):
        return []
    return sorted(f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(VIDEO_EXTS))


def video_path(name):
    """test_videos/ 裡影片的完整路徑; 只取檔名 (擋掉 ../ 之類的路徑), 找不到回傳 None"""
    name = os.path.basename(name or "")
    path = os.path.join(VIDEO_DIR, name)
    return path if name and os.path.isfile(path) else None


def video_meta(path):
    """{"fps", "width", "height", "frames"}; 讀不到 fps 時是 0"""
    cap = cv2.VideoCapture(path)
    meta = {"fps": cap.get(cv2.CAP_PROP_FPS) or 0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)}
    cap.release()
    return meta


def read_frame(path, frame_idx=None, time_sec=None):
    """讀一幀畫面: frame_idx 從 1 開始 (跟 processor 的 raw_frame_idx 一樣), 或用秒數; 讀不到回傳 None"""
    cap = cv2.VideoCapture(path)
    if frame_idx is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec or 0.0) * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def video_info(filename):
    m = video_meta(os.path.join(VIDEO_DIR, filename))
    fps = m["fps"]
    return {"fps": round(fps, 1), "frames": m["frames"], "duration": round(m["frames"] / fps, 1) if fps else None,
            "size": f"{m['width']}×{m['height']}"}


# ===== 檔案 =====
def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data, exclusive=False):
    """exclusive=True: 檔案已存在就報錯, 絕不覆蓋; 否則先寫暫存檔再換掉, 寫到一半當掉也不會留下壞檔"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if exclusive:
        with open(path, "x", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)


def remove_file(path):
    try:
        os.remove(path)
    except OSError:
        pass


# ===== 實驗 =====
def get_experiment_or_404(exp_id):
    exp = db.get_experiment(exp_id)
    if exp is None:
        abort(404)
    return exp


def selected_ids():
    ids = request.args.getlist("id") or [s for s in request.args.get("ids", "").split(",") if s]
    return [int(i) for i in ids if str(i).isdigit()]


def selected_experiments(default_all=False):
    """網址上 ?id= / ?ids= / ?batch= 選取的實驗 (新的在前); 什麼都沒選時 default_all 決定回傳全部還是空的"""
    ids, batch = selected_ids(), request.args.get("batch")
    if ids or batch:
        return db.list_experiments(ids=ids, batch=batch)
    return db.list_experiments() if default_all else []


def csv_response(filename, headers, rows):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerows(rows)
    return Response("﻿" + buf.getvalue(), mimetype="text/csv",   # 加 BOM 讓 Excel 正確顯示中文
                    headers={"Content-Disposition": f"attachment; filename={filename}"})
