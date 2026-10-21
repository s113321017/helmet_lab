"""
實驗平台網頁介面
------------------------------------------------
啟動:
    conda activate helmet
    python app.py
然後打開瀏覽器 http://127.0.0.1:5000

功能:
    - 選影片、調參數建立實驗 (數值欄位輸入 "0.2,0.3,0.4" 會自動展開成多個實驗做參數掃描)
    - 實驗在背景依序執行 (GPU 一次只跑一個),結果存進 PostgreSQL (DATABASE_URL)
    - 在實驗頁面標註每筆違規 正確/誤報/重複、填正確車牌 → 自動算 precision / recall / 車牌準確率
    - 影片頁面填每支影片的實際違規數 (真值) → 算 recall
    - 比較多個實驗的參數差異與指標,匯出 CSV 做圖表
"""

import os
import io
import csv
import sys
import glob
import json
import queue
import re
import base64
import shutil
import subprocess
import itertools
import threading
import traceback
from datetime import datetime

import cv2
import numpy as np
from flask import (Flask, render_template, request, redirect, url_for, jsonify, flash,
                   send_from_directory, Response, abort)

from core import db
from core.config import CONFIG_DIR, env
from core.pipeline import (BASE_DIR, DEFAULT_PARAMS, PARAM_SPECS, VIOLATION_TYPES, normalize_params,
                           plate_status, process_video)

VIDEO_DIR = os.path.join(BASE_DIR, "test_videos")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
MAX_EXPERIMENTS_PER_SUBMIT = 100
FIXED_PARAMS = {"no_helmet_class", "motorbike_class", "tracker"}   # 固定不變的參數, CSV 匯出不放
STATUS_TEXT = {"queued": "排隊中", "running": "執行中", "done": "完成", "stopped": "已停止", "failed": "失敗"}

app = Flask(__name__)
app.secret_key = env("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("找不到 FLASK_SECRET_KEY, 請在 .env 設定一串隨機字串, 例如用 "
                       "python -c \"import secrets; print(secrets.token_hex(32))\" 產生")
app.jinja_env.globals.update(VIOLATION_TYPES=VIOLATION_TYPES, STATUS_TEXT=STATUS_TEXT, LABELS=db.LABELS)


@app.template_filter("fmt")
def fmt(v, digits=3):
    if v is None:
        return "–"
    return round(v, digits) if isinstance(v, float) else v


@app.template_filter("pct")
def pct(v):
    return "–" if v is None else f"{v * 100:.1f}%"


# ===== 背景執行 =====
_job_queue = queue.Queue()
_stop_requests = set()


def _run_experiment(exp_id):
    exp = db.get_experiment(exp_id)
    if exp is None or exp["status"] != "queued":
        return   # 排隊期間被停止或刪除

    out_dir = db.experiment_dir(exp_id)
    os.makedirs(out_dir, exist_ok=True)
    db.update_experiment(exp_id, status="running", started_at=db.now(), progress=0)

    with open(os.path.join(out_dir, "log.txt"), "a", encoding="utf-8") as log_file:
        def log(msg):
            print(f"[實驗#{exp_id}] {msg}")
            log_file.write(msg + "\n")
            log_file.flush()

        try:
            summary = process_video(
                os.path.join(VIDEO_DIR, exp["video"]), exp["params"], out_dir,
                on_violation=lambda rec: db.add_violation(exp_id, rec),
                on_progress=lambda prog, n: db.update_experiment(exp_id, progress=prog, processed_frames=n),
                should_stop=lambda: exp_id in _stop_requests,
                log=log,
            )
            db.add_tracks(exp_id, summary["tracks"])
            db.update_experiment(
                exp_id, status="stopped" if summary["stopped"] else "done",
                progress=1 if not summary["stopped"] else db.get_experiment(exp_id)["progress"],
                finished_at=db.now(), fps=summary["fps"], total_frames=summary["total_frames"],
                processed_frames=summary["processed_frames"], elapsed_sec=summary["elapsed_sec"],
                avg_ms_per_frame=summary["avg_ms_per_frame"], avg_track_ms=summary["avg_track_ms"],
                avg_plate_ms=summary["avg_plate_ms"], avg_ocr_ms=summary["avg_ocr_ms"],
                output_video=summary["output_video"],
            )
        except Exception as e:
            log(traceback.format_exc())
            db.update_experiment(exp_id, status="failed", error=str(e), finished_at=db.now())
        finally:
            _stop_requests.discard(exp_id)


def _worker():
    while True:
        exp_id = _job_queue.get()
        try:
            _run_experiment(exp_id)
        except Exception:
            traceback.print_exc()


def start_worker():
    # 伺服器上次被關掉時還在跑的實驗標成失敗,還在排隊的重新排入
    for exp in reversed(db.list_experiments()):
        if exp["status"] == "running":
            db.update_experiment(exp["id"], status="failed", error="伺服器中斷,請重新執行")
        elif exp["status"] == "queued":
            _job_queue.put(exp["id"])
    threading.Thread(target=_worker, daemon=True).start()


# ===== 工具函式 =====
def list_videos():
    if not os.path.isdir(VIDEO_DIR):
        return []
    return sorted(f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(VIDEO_EXTS))


def video_info(filename):
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, filename))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    info = {"fps": round(fps, 1), "frames": frames, "duration": round(frames / fps, 1) if fps else None,
            "size": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"}
    cap.release()
    return info


def model_choices():
    found = glob.glob(os.path.join(BASE_DIR, "runs", "detect", "*", "weights", "*.pt"))
    return sorted(set(found + [DEFAULT_PARAMS["helmet_model_path"], DEFAULT_PARAMS["plate_model_path"]]))


def parse_param_form(form):
    """
    讀網頁表單的參數。回傳 (固定參數, 掃描參數 {key: [值...]})
    int / float 欄位輸入逗號分隔的多個值就會變成掃描參數。
    """
    fixed, sweep = {}, {}
    for key, label, typ, _group, _help in PARAM_SPECS:
        if typ == "bool":
            fixed[key] = key in form
            continue
        raw = form.get(key, "").strip()
        if raw == "":
            fixed[key] = DEFAULT_PARAMS[key]
            continue
        if typ in ("int", "float"):
            conv = (lambda s: int(float(s))) if typ == "int" else float
            try:
                values = [conv(s.strip()) for s in raw.split(",") if s.strip()]
            except ValueError:
                raise ValueError(f"「{label}」的值無法解析: {raw}")
            if len(values) > 1:
                sweep[key] = values
            else:
                fixed[key] = values[0]
        else:
            ext = FILE_KEYS.get(key)
            # 測速沒開就用不到標定矩陣, 不檢查
            if ext and not (key == "homography_path" and "speed_enabled" not in form):
                if not raw.lower().endswith(ext):
                    raise ValueError(f"「{label}」只能選 {ext} 檔: {raw}")
                if not os.path.isfile(raw):
                    raise ValueError(f"「{label}」找不到檔案: {raw}")
            fixed[key] = raw
    return fixed, sweep


def params_diff_keys(exps):
    """找出這些實驗之間有差異的參數"""
    return [k for k in DEFAULT_PARAMS if len({json.dumps(e["params"].get(k)) for e in exps}) > 1]


def selected_ids():
    ids = request.args.getlist("id") or [s for s in request.args.get("ids", "").split(",") if s]
    return [int(i) for i in ids if str(i).isdigit()]


def csv_response(filename, headers, rows):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerows(rows)
    return Response("﻿" + buf.getvalue(), mimetype="text/csv",   # 加 BOM 讓 Excel 正確顯示中文
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


# ===== 檔案路徑設定 (模型、標定矩陣; 改了就存檔, 重新整理頁面或刪除實驗後不會跳回預設) =====
SETTINGS_PATH = os.path.join(db.DATA_DIR, "model_settings.json")
FILE_KEYS = {"helmet_model_path": ".pt", "plate_model_path": ".pt", "homography_path": ".json"}
FILE_TYPES = {".pt": ("選擇模型檔 (.pt)", "PyTorch 模型 (*.pt)"), ".json": ("選擇標定矩陣檔 (.json)", "標定矩陣 (*.json)")}


def load_file_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in saved.items()
            if k in FILE_KEYS and isinstance(v, str) and v.lower().endswith(FILE_KEYS[k]) and os.path.isfile(v)}


def save_file_setting(key, path):
    """只存副檔名正確、而且存在的檔案"""
    if key not in FILE_KEYS or not path.lower().endswith(FILE_KEYS[key]) or not os.path.isfile(path):
        return False
    saved = load_file_settings()
    saved[key] = path
    os.makedirs(db.DATA_DIR, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(saved, f, ensure_ascii=False, indent=2)
    return True


@app.route("/api/file-setting", methods=["POST"])
def file_setting():
    return jsonify({"saved": save_file_setting(request.form.get("key", ""), request.form.get("value", "").strip())})


# ===== 頁面 =====
@app.route("/")
def index():
    prefill = {**DEFAULT_PARAMS, **load_file_settings()}
    prefill_videos = []
    src = request.args.get("from", type=int)
    if src and (src_exp := db.get_experiment(src)):
        prefill.update(src_exp["params"])
        prefill_videos = [src_exp["video"]]
    return render_template(
        "index.html", experiments=db.list_experiments(), counts=db.violation_counts(),
        videos=list_videos(), specs=PARAM_SPECS, prefill=prefill, prefill_videos=prefill_videos,
        model_choices=model_choices(), homography_choices=[c["path"] for c in list_calibrations()],
        file_keys=FILE_KEYS)


_PICK_SCRIPT = """
import sys, tkinter as tk
from tkinter import filedialog
start, title, type_name, pattern = sys.argv[1:5]
r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)
print(filedialog.askopenfilename(title=title, initialdir=start, filetypes=[(type_name, pattern)]))
"""


@app.route("/api/pick-file", methods=["POST"])
def pick_file():
    """在本機跳出檔案總管選檔, 只能選指定副檔名 (網頁只跑在 127.0.0.1, 所以可以直接開視窗)"""
    ext = request.form.get("ext", "")
    if ext not in FILE_TYPES:
        abort(400)
    title, type_name = FILE_TYPES[ext]
    current = request.form.get("current", "")
    start = os.path.dirname(current) if os.path.isfile(current) else BASE_DIR
    out = subprocess.run([sys.executable, "-c", _PICK_SCRIPT, start, title, type_name, "*" + ext],
                         capture_output=True, text=True, encoding="utf-8",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout.strip()
    path = os.path.normpath(out) if out else ""
    return jsonify({"path": path if path.lower().endswith(ext) else ""})


@app.route("/experiments", methods=["POST"])
def create_experiments():
    videos = request.form.getlist("videos")
    if not videos:
        flash("請至少選一支影片", "error")
        return redirect(url_for("index"))
    try:
        fixed, sweep = parse_param_form(request.form)
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("index"))

    for k in FILE_KEYS:
        if isinstance(fixed.get(k), str):
            save_file_setting(k, fixed[k])

    sweep_keys = list(sweep)
    combos = list(itertools.product(*sweep.values())) if sweep else [()]
    total = len(combos) * len(videos)
    if total > MAX_EXPERIMENTS_PER_SUBMIT:
        flash(f"一次會建立 {total} 個實驗,超過上限 {MAX_EXPERIMENTS_PER_SUBMIT},請減少掃描值", "error")
        return redirect(url_for("index"))

    base_name = request.form.get("name", "").strip() or datetime.now().strftime("實驗 %m/%d %H:%M")
    batch = datetime.now().strftime("B%Y%m%d-%H%M%S") if total > 1 else None
    for video in videos:
        for combo in combos:
            params = normalize_params({**fixed, **dict(zip(sweep_keys, combo))})
            name = base_name
            if sweep_keys:
                name += " [" + ", ".join(f"{k}={v}" for k, v in zip(sweep_keys, combo)) + "]"
            exp_id = db.create_experiment(name, video, params, batch)
            _job_queue.put(exp_id)
    flash(f"已建立 {total} 個實驗,排入背景執行", "ok")
    return redirect(url_for("index"))


@app.route("/experiments/<int:exp_id>")
def experiment_detail(exp_id):
    exp = db.get_experiment(exp_id)
    if exp is None:
        abort(404)
    violations = db.list_violations(exp_id)
    for v in violations:
        v["status_text"] = plate_status(v["is_clean"], v["plate_conf"])
    log_path = os.path.join(db.experiment_dir(exp_id), "log.txt")
    log_text = ""
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            log_text = "".join(f.readlines()[-300:])
    return render_template(
        "experiment.html", exp=exp, violations=violations, tracks=db.list_tracks(exp_id),
        metrics=db.compute_metrics(exp, violations), specs=PARAM_SPECS, defaults=DEFAULT_PARAMS,
        log_text=log_text)


@app.route("/experiments/<int:exp_id>/stop", methods=["POST"])
def stop_experiment(exp_id):
    exp = db.get_experiment(exp_id)
    if exp and exp["status"] == "queued":
        db.update_experiment(exp_id, status="stopped", finished_at=db.now())
    elif exp and exp["status"] == "running":
        _stop_requests.add(exp_id)
    return redirect(request.referrer or url_for("index"))


@app.route("/experiments/<int:exp_id>/rerun", methods=["POST"])
def rerun_experiment(exp_id):
    exp = db.get_experiment(exp_id)
    if exp is None:
        abort(404)
    new_id = db.create_experiment(exp["name"] + " (重跑)", exp["video"], exp["params"], exp["batch"])
    _job_queue.put(new_id)
    flash(f"已重新排入實驗 #{new_id}", "ok")
    return redirect(url_for("index"))


@app.route("/experiments/delete", methods=["POST"])
def delete_experiments():
    deleted = 0
    for exp_id in [int(i) for i in request.form.getlist("id")]:
        exp = db.get_experiment(exp_id)
        if exp is None or exp["status"] == "running":
            continue
        db.delete_experiment(exp_id)
        shutil.rmtree(db.experiment_dir(exp_id), ignore_errors=True)
        deleted += 1
    flash(f"已刪除 {deleted} 個實驗 (執行中的不會刪除)", "ok")
    return redirect(url_for("index"))


@app.route("/experiments/<int:exp_id>/notes", methods=["POST"])
def save_notes(exp_id):
    db.update_experiment(exp_id, notes=request.form.get("notes", ""), name=request.form.get("name", ""))
    return redirect(url_for("experiment_detail", exp_id=exp_id))


@app.route("/files/<int:exp_id>/<path:filename>")
def experiment_file(exp_id, filename):
    return send_from_directory(db.experiment_dir(exp_id), filename)


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 1) if values else None


def chart_data(rows, diff_keys):
    """比較頁圖表用的資料: 每個實驗的違規數、車牌、車速"""
    out = []
    for e, m in rows:
        vs, tracks = db.list_violations(e["id"]), db.list_tracks(e["id"])
        max_speeds = [t["max_speed_kmh"] for t in tracks if t["max_speed_kmh"] is not None]
        out.append({
            "id": e["id"], "label": f"#{e['id']} {e['name'] or ''}".strip(), "video": e["video"],
            "diff": {k: e["params"].get(k) for k in diff_keys},
            "violations": {vt: m["per_type"][vt]["detected"] for vt in VIOLATION_TYPES},
            "plate": {"clean": sum(bool(v["is_clean"]) for v in vs),
                      "unclean": sum(not v["is_clean"] for v in vs),
                      "list": [{"type": VIOLATION_TYPES[v["violation_type"]], "fixed": v["plate_fixed"],
                                "raw": v["plate_raw"], "conf": v["plate_conf"], "clean": bool(v["is_clean"])}
                               for v in vs if v["plate_fixed"] or v["plate_raw"]]},
            "speed": {"max": round(max(max_speeds), 1) if max_speeds else None,
                      "avg": _mean(t["avg_speed_kmh"] for t in tracks), "tracks": len(tracks)},
        })
    return out


@app.route("/compare")
def compare():
    batch = request.args.get("batch")
    ids = selected_ids()
    exps = db.list_experiments(ids=ids, batch=batch) if (ids or batch) else []
    rows = [(e, db.compute_metrics(e)) for e in exps]
    diff_keys = params_diff_keys(exps)
    return render_template("compare.html", rows=rows, diff_keys=diff_keys,
                           ids=",".join(str(e["id"]) for e in exps),
                           chart=chart_data(rows, diff_keys),
                           chart_types=[{"key": k, "name": n} for k, n in VIOLATION_TYPES.items()])


@app.route("/videos")
def videos_page():
    items = [{"name": v, **video_info(v), "gt": db.get_video_gt(v)} for v in list_videos()]
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


# ===== 相機標定 (Homography) =====
# 每次標定存成新檔 data/homography/H_<時間>.json + 參考圖 .jpg, 不會覆蓋舊的結果
HOMO_DIR = os.path.join(db.DATA_DIR, "homography")
LEGACY_HOMOGRAPHY = os.path.join(CONFIG_DIR, "homography_matrix.json")
_CALIB_NAME = re.compile(r"^H_[0-9A-Za-z_-]+$")


def _calib_path(name, ext=".json"):
    if not _CALIB_NAME.match(name or ""):
        abort(404)
    return os.path.join(HOMO_DIR, name + ext)


def load_calibration(name):
    path = _calib_path(name)
    if not os.path.isfile(path):
        abort(404)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_calibrations():
    """所有標定結果, 新的在前; 舊版 homography_matrix.json 放最後"""
    items = []
    for path in sorted(glob.glob(os.path.join(HOMO_DIR, "H_*.json")), reverse=True):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
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
    video = os.path.basename(request.args.get("video", ""))
    path = os.path.join(VIDEO_DIR, video)
    if not video or not os.path.isfile(path):
        abort(404)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, request.args.get("t", 0.0, type=float)) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        abort(404)
    return Response(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), mimetype="image/jpeg")


@app.route("/calibration/file/<name>.<ext>")
def calibration_file(name, ext):
    if ext not in ("json", "jpg"):
        abort(404)
    path = _calib_path(name, "." + ext)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(HOMO_DIR, os.path.basename(path))


@app.route("/api/calibration/<name>")
def calibration_get(name):
    return jsonify({"name": name, **load_calibration(name)})


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
    m = re.match(r"^data:image/(?:jpeg|png);base64,(.+)$", data.get("image") or "", re.S)
    if m:
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(m.group(1)), np.uint8), cv2.IMREAD_COLOR)
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
    with open(_calib_path(name), "x", encoding="utf-8") as f:   # "x": 檔案已存在就報錯, 絕不覆蓋
        json.dump(record, f, ensure_ascii=False, indent=2)
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
    path = _calib_path(name)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    os.replace(path + ".tmp", path)
    return jsonify({"tests": record["tests"]})


@app.route("/api/calibration/default", methods=["POST"])
def calibration_set_default():
    """設成建立實驗時預設的標定矩陣"""
    path = request.form.get("path", "")
    if os.path.normcase(os.path.abspath(path)) not in {os.path.normcase(c["path"]) for c in list_calibrations()}:
        abort(400)
    return jsonify({"saved": save_file_setting("homography_path", path)})


# ===== JSON API =====
@app.route("/api/experiments")
def api_experiments():
    counts = db.violation_counts()
    ids = selected_ids()
    return jsonify([{
        "id": e["id"], "status": e["status"], "status_text": STATUS_TEXT.get(e["status"], e["status"]),
        "progress": e["progress"] or 0, "counts": counts.get(e["id"], {}),
        "elapsed_sec": e["elapsed_sec"], "error": e["error"],
    } for e in db.list_experiments(ids=ids)])


@app.route("/api/violations/<int:vid>", methods=["POST"])
def api_update_violation(vid):
    db.update_violation(vid, request.get_json(force=True) or {})
    return jsonify({"ok": True})


# ===== CSV 匯出 =====
@app.route("/export/violations.csv")
def export_violations():
    ids = selected_ids()
    exps = db.list_experiments(ids=ids) if ids else db.list_experiments()
    param_keys = [k for k in DEFAULT_PARAMS if k not in FIXED_PARAMS]
    headers = (["experiment_id", "experiment_name", "batch", "video"] + [f"p_{k}" for k in param_keys]
               + ["time_sec", "frame_idx", "violation_type", "track_id", "det_conf", "plate_text_raw",
                  "plate_text_fixed", "plate_conf", "speed_kmh", "status", "warnings",
                  "label", "plate_gt", "note"])
    rows = []
    for e in reversed(exps):
        for v in db.list_violations(e["id"]):
            rows.append([e["id"], e["name"], e["batch"], e["video"]] + [e["params"].get(k) for k in param_keys]
                        + [v["time_sec"], v["frame_idx"], v["violation_type"], v["track_id"], v["det_conf"],
                           v["plate_raw"], v["plate_fixed"], v["plate_conf"], v["speed_kmh"],
                           plate_status(v["is_clean"], v["plate_conf"]), v["warnings"],
                           v["label"], v["plate_gt"], v["note"]])
    return csv_response("violations.csv", headers, rows)


@app.route("/export/tracks.csv")
def export_tracks():
    ids = selected_ids()
    exps = db.list_experiments(ids=ids) if ids else db.list_experiments()
    headers = ["experiment_id", "experiment_name", "video", "track_id", "class_name", "first_sec", "last_sec",
               "frames", "avg_speed_kmh", "max_speed_kmh"]
    rows = []
    for e in reversed(exps):
        for t in db.list_tracks(e["id"]):
            rows.append([e["id"], e["name"], e["video"], t["track_id"], t["class_name"], t["first_sec"],
                         t["last_sec"], t["frames"], t["avg_speed_kmh"], t["max_speed_kmh"]])
    return csv_response("tracks.csv", headers, rows)


@app.route("/export/compare.csv")
def export_compare():
    ids = selected_ids()
    batch = request.args.get("batch")
    exps = list(reversed(db.list_experiments(ids=ids, batch=batch))) if (ids or batch) else []
    param_keys = [k for k in DEFAULT_PARAMS if k not in FIXED_PARAMS]
    metric_keys = ["detected", "tp", "fp", "dup", "unlabeled", "gt", "precision", "recall", "f1"]
    plate_keys = ["labeled", "fixed_acc", "raw_acc", "char_acc", "clean_rate", "avg_conf"]
    headers = (["experiment_id", "name", "batch", "video", "status"] + [f"p_{k}" for k in param_keys]
               + [f"{vt}_{m}" for vt in VIOLATION_TYPES for m in metric_keys]
               + [f"plate_{m}" for m in plate_keys]
               + ["processed_frames", "elapsed_sec", "avg_ms_per_frame", "avg_track_ms", "avg_plate_ms",
                  "avg_ocr_ms"])
    rows = []
    for e in exps:
        m = db.compute_metrics(e)
        rows.append([e["id"], e["name"], e["batch"], e["video"], e["status"]]
                    + [e["params"].get(k) for k in param_keys]
                    + [m["per_type"][vt][k] for vt in VIOLATION_TYPES for k in metric_keys]
                    + [m["plate"][k] for k in plate_keys]
                    + [e["processed_frames"], e["elapsed_sec"], e["avg_ms_per_frame"], e["avg_track_ms"],
                       e["avg_plate_ms"], e["avg_ocr_ms"]])
    return csv_response("compare.csv", headers, rows)


if __name__ == "__main__":
    try:
        import torch, ultralytics  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"缺少套件 ({e})。目前的 Python: {sys.executable}\n"
                         "請先執行 conda activate helmet,或直接雙擊 run.bat")
    db.init_db()
    start_worker()
    host, port = env("HOST", "127.0.0.1"), int(env("PORT", "5000"))
    print(f"實驗平台已啟動: http://{host}:{port}")
    # use_reloader=False: 避免 debug 模式啟動兩個背景執行緒同時搶 GPU
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
