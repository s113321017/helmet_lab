"""
相機標定 (Homography): 在參考圖上點標定點算矩陣, 再點兩點測距離檢查準確度
每次標定存成新檔 data/homography/H_<時間>.json + 參考圖 .jpg, 不會覆蓋舊的結果
"""

import os
import re
import glob
import base64
from datetime import datetime

import cv2
import numpy as np
from flask import render_template, request, jsonify, abort, send_from_directory, Response

from core import db
from core.config import CONFIG_DIR
from core.pipeline import DEFAULT_PARAMS

from . import app
from .common import list_videos, video_path, read_frame, read_json, write_json
from .settings import load_file_settings, save_file_setting

HOMO_DIR = os.path.join(db.DATA_DIR, "homography")
LEGACY_HOMOGRAPHY = os.path.join(CONFIG_DIR, "homography_matrix.json")
_CALIB_NAME = re.compile(r"^H_[0-9A-Za-z_-]+$")


def _calib_path(name, ext=".json"):
    if not _CALIB_NAME.match(name or ""):
        abort(404)
    return os.path.join(HOMO_DIR, name + ext)


def _existing_calib_path(name, ext=".json"):
    path = _calib_path(name, ext)
    if not os.path.isfile(path):
        abort(404)
    return path


def load_calibration(name):
    data = read_json(_existing_calib_path(name))
    if data is None:
        abort(404)
    return data


def list_calibrations():
    """所有標定結果, 新的在前; 舊版 homography_matrix.json 放最後"""
    items = []
    for path in sorted(glob.glob(os.path.join(HOMO_DIR, "H_*.json")), reverse=True):
        d = read_json(path)
        if d is None:
            continue
        items.append({"name": os.path.splitext(os.path.basename(path))[0], "path": path,
                      "created_at": d.get("created_at"), "source": d.get("source"), "note": d.get("note"),
                      "points": len(d.get("image_points", [])), "tests": d.get("tests", []),
                      "rms_m": d.get("rms_m"), "has_image": bool(d.get("image"))})
    if os.path.isfile(LEGACY_HOMOGRAPHY):
        items.append({"name": None, "path": LEGACY_HOMOGRAPHY, "created_at": None,
                      "source": "calibrate_camera.py (舊版)", "note": None, "points": None, "tests": [],
                      "rms_m": None, "has_image": False})
    return items


def _unique_calib_name():
    base = "H_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    name, n = base, 2
    while os.path.exists(_calib_path(name)):
        name, n = f"{base}_{n}", n + 1
    return name


@app.route("/calibration")
def calibration_page():
    current = load_file_settings().get("homography_path", DEFAULT_PARAMS["homography_path"])
    return render_template("calibration.html", videos=list_videos(), calibrations=list_calibrations(),
                           current=os.path.normcase(os.path.abspath(current)),
                           open_name=request.args.get("open", ""))


@app.route("/calibration/frame")
def calibration_frame():
    """從測試影片截一張畫面當標定參考圖"""
    path = video_path(request.args.get("video", ""))
    frame = read_frame(path, time_sec=request.args.get("t", 0.0, type=float)) if path else None
    if frame is None:
        abort(404)
    return Response(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), mimetype="image/jpeg")


@app.route("/calibration/file/<name>.<ext>")
def calibration_file(name, ext):
    if ext not in ("json", "jpg"):
        abort(404)
    return send_from_directory(HOMO_DIR, os.path.basename(_existing_calib_path(name, "." + ext)))


@app.route("/api/calibration/<name>")
def calibration_get(name):
    return jsonify({"name": name, **load_calibration(name)})


def _decode_image(data_url):
    """網頁傳來的 data:image/...;base64 圖片 → OpenCV 畫面; 格式不對回傳 None"""
    m = re.match(r"^data:image/(?:jpeg|png);base64,(.+)$", data_url or "", re.S)
    if not m:
        return None
    return cv2.imdecode(np.frombuffer(base64.b64decode(m.group(1)), np.uint8), cv2.IMREAD_COLOR)


@app.route("/api/calibration", methods=["POST"])
def calibration_create():
    """算 homography 矩陣並存成新檔 (不覆蓋)"""
    data = request.get_json(force=True)
    try:
        img_pts = np.array(data["image_points"], dtype=np.float64)
        world_pts = np.array(data["world_points"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "標定點格式錯誤"}), 400
    if img_pts.ndim != 2 or img_pts.shape[1] != 2 or len(img_pts) < 4 or img_pts.shape != world_pts.shape:
        return jsonify({"error": "至少要 4 個點, 而且每個點都要有畫面座標和真實座標"}), 400
    matrix, _ = cv2.findHomography(img_pts, world_pts)
    if matrix is None:
        return jsonify({"error": "算不出矩陣, 請確認 4 個點不在同一條直線上"}), 400

    back = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = np.linalg.norm(back - world_pts, axis=1)
    os.makedirs(HOMO_DIR, exist_ok=True)
    name = _unique_calib_name()

    image_file = None
    frame = _decode_image(data.get("image"))
    if frame is not None:
        image_file = name + ".jpg"
        cv2.imwrite(os.path.join(HOMO_DIR, image_file), frame)

    record = {
        "matrix": matrix.tolist(),
        "image_points": img_pts.tolist(), "world_points": world_pts.tolist(),
        "point_labels": [str(s) for s in data.get("labels", [])][:len(img_pts)],
        "reprojected": back.round(4).tolist(), "errors_m": errors.round(4).tolist(),
        "rms_m": round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "image": image_file, "image_size": data.get("image_size"),
        "source": str(data.get("source") or ""), "note": str(data.get("note") or ""),
        "created_at": db.now(), "tests": [],
    }
    write_json(_calib_path(name), record, exclusive=True)
    return jsonify({"name": name, "path": _calib_path(name), **record})


@app.route("/api/calibration/<name>/tests", methods=["POST"])
def calibration_add_test(name):
    """把一筆兩點距離測試附加到標定紀錄 (只新增, 不改原本的矩陣和舊測試)"""
    record = load_calibration(name)
    data = request.get_json(force=True)
    try:
        p1, p2 = [[float(v) for v in p] for p in (data["p1"], data["p2"])]
        dist = float(data["distance_m"])
        actual = float(data["actual_m"]) if data.get("actual_m") not in (None, "") else None
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "測試資料格式錯誤"}), 400
    test = {"p1": p1, "p2": p2, "distance_m": round(dist, 3), "actual_m": actual,
            "error_m": round(dist - actual, 3) if actual is not None else None,
            "error_pct": round((dist - actual) / actual * 100, 2) if actual else None,
            "note": str(data.get("note") or ""), "created_at": db.now()}
    record.setdefault("tests", []).append(test)
    write_json(_calib_path(name), record)
    return jsonify({"tests": record["tests"]})


@app.route("/api/calibration/default", methods=["POST"])
def calibration_set_default():
    """設成建立實驗時預設的標定矩陣"""
    path = request.form.get("path", "")
    if os.path.normcase(os.path.abspath(path)) not in {os.path.normcase(c["path"]) for c in list_calibrations()}:
        abort(400)
    return jsonify({"saved": save_file_setting("homography_path", path)})
