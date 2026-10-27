"""
影片與真值: 上傳影片、填每種違規的實際數量 (vt), 暫停影片框出違規存成標註畫面 (img)
"""

import os

import cv2
from flask import render_template, request, redirect, url_for, jsonify, flash, send_from_directory

from core import db
from core.pipeline import VIOLATION_TYPES

from . import app
from .common import VIDEO_DIR, VIDEO_EXTS, list_videos, video_info, video_path, video_meta, read_frame, remove_file

GT_IMG_DIR = os.path.join(db.DATA_DIR, "gt_imgs")   # 每張標註畫面的裁切圖 <iid>.jpg


@app.route("/videos")
def videos_page():
    counts = db.img_counts()
    items = [{"name": v, **video_info(v), "gt": db.get_video_gt(v), "imgs": counts.get(v, 0)} for v in list_videos()]
    return render_template("videos.html", videos=items)


@app.route("/videos/file/<path:filename>")
def video_file(filename):
    """test_videos/ 裡的原始影片, 給網頁預覽用"""
    return send_from_directory(VIDEO_DIR, filename, conditional=True)


@app.route("/videos/gt", methods=["POST"])
def save_video_gt():
    video = request.form["video"]
    for vt in VIOLATION_TYPES:
        raw = request.form.get(vt, "").strip()
        db.set_video_gt(video, vt, int(raw) if raw.isdigit() else None)
    flash(f"已儲存 {video} 的真值", "ok")
    return redirect(url_for("videos_page"))


@app.route("/videos/upload", methods=["POST"])
def upload_video():
    f = request.files.get("file")
    filename = os.path.basename((f.filename if f else "").replace("\\", "/"))
    if not filename or not filename.lower().endswith(VIDEO_EXTS):
        flash("請選擇影片檔 (" + ", ".join(VIDEO_EXTS) + ")", "error")
        return redirect(url_for("videos_page"))
    os.makedirs(VIDEO_DIR, exist_ok=True)
    f.save(os.path.join(VIDEO_DIR, filename))
    flash(f"已上傳 {filename}", "ok")
    return redirect(url_for("videos_page"))


# ===== 標註畫面 (img 真值): 暫停影片框出違規, 存時間 + 座標 + 違規類型 =====
def _error(msg, code=400):
    return jsonify({"error": msg}), code


def _save_gt_crop(frame, bbox, path):
    """存裁切圖 (框外多留 1/3 範圍, 畫上標註框) 給列表顯示"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    ox, oy = max(0, x1 - (x2 - x1) // 3), max(0, y1 - (y2 - y1) // 3)
    crop = frame[oy:min(h, y2 + (y2 - y1) // 3), ox:min(w, x2 + (x2 - x1) // 3)].copy()
    cv2.rectangle(crop, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy), (0, 0, 255), 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, crop)


@app.route("/api/gt-imgs")
def api_list_gt_imgs():
    return jsonify(db.list_imgs(request.args.get("video", "")))


@app.route("/api/gt-imgs", methods=["POST"])
def api_add_gt_img():
    data = request.get_json(force=True) or {}
    video = os.path.basename(str(data.get("video", "")))
    path = video_path(video)
    if not path:
        return _error("找不到影片", 404)
    if data.get("gt_ans") not in VIOLATION_TYPES:
        return _error("違規類型錯誤")
    try:
        t = float(data["time"])
        x1, y1, x2, y2 = [int(round(float(v))) for v in data["bbox"]]
    except (KeyError, TypeError, ValueError):
        return _error("時間或座標格式錯誤")

    meta = video_meta(path)
    fps, w, h = meta["fps"] or 30, meta["width"], meta["height"]
    x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
    y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return _error("框太小, 請在畫面上拖曳框出違規的人/車")
    frame_idx = int(t * fps + 1e-6) + 1   # 跟 processor 的 raw_frame_idx 一樣從 1 開始
    frame = read_frame(path, frame_idx=frame_idx)
    if frame is None:
        return _error("讀不到這一幀")

    bbox = [x1, y1, x2, y2]
    iid = db.add_img(video, frame_idx, bbox, data["gt_ans"], fps=fps)
    _save_gt_crop(frame, bbox, os.path.join(GT_IMG_DIR, f"{iid}.jpg"))
    return jsonify(db.get_img(iid))


@app.route("/api/gt-imgs/<int:iid>/delete", methods=["POST"])
def api_delete_gt_img(iid):
    db.delete_img(iid)
    remove_file(os.path.join(GT_IMG_DIR, f"{iid}.jpg"))
    return jsonify({"ok": True})


@app.route("/gt-imgs/<int:iid>.jpg")
def gt_img_file(iid):
    return send_from_directory(GT_IMG_DIR, f"{iid}.jpg")
