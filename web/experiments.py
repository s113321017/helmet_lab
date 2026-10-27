"""
實驗: 建立 (含參數掃描)、檢視、停止、重跑、刪除, 違規紀錄的人工標註
"""

import os
import shutil
import itertools
from datetime import datetime

from flask import render_template, request, redirect, url_for, jsonify, flash, send_from_directory

from core import db
from core.pipeline import DEFAULT_PARAMS, PARAM_SPECS, normalize_params, plate_status

from . import app, jobs
from .common import STATUS_TEXT, list_videos, get_experiment_or_404, selected_ids
from .settings import FILE_KEYS, MULTI_KEYS, load_file_settings, save_file_setting, model_choices
from .calibration import list_calibrations

MAX_EXPERIMENTS_PER_SUBMIT = 100


def parse_param_form(form):
    """
    讀網頁表單的參數。回傳 (固定參數, 掃描參數 {key: [值...]})
    int / float 欄位輸入逗號分隔的多個值就會變成掃描參數; 安全帽模型可以選多個檔案。
    """
    fixed, sweep = {}, {}
    for key, label, typ, _group, _help in PARAM_SPECS:
        if typ == "bool":
            fixed[key] = key in form
            continue
        raws = [s.strip() for s in (form.getlist(key) if key in MULTI_KEYS else [form.get(key, "")])]
        raws = list(dict.fromkeys(r for r in raws if r))   # 去掉空白和重複的
        if not raws:
            fixed[key] = DEFAULT_PARAMS[key]
            continue
        if typ in ("int", "float"):
            conv = (lambda s: int(float(s))) if typ == "int" else float
            try:
                values = [conv(s.strip()) for s in raws[0].split(",") if s.strip()]
            except ValueError:
                raise ValueError(f"「{label}」的值無法解析: {raws[0]}")
        else:
            values = raws
            ext = FILE_KEYS.get(key)
            # 測速沒開就用不到標定矩陣, 不檢查
            if ext and not (key == "homography_path" and "speed_enabled" not in form):
                for r in raws:
                    if not r.lower().endswith(ext):
                        raise ValueError(f"「{label}」只能選 {ext} 檔: {r}")
                    if not os.path.isfile(r):
                        raise ValueError(f"「{label}」找不到檔案: {r}")
        if len(values) > 1:
            sweep[key] = values
        else:
            fixed[key] = values[0]
    return fixed, sweep


def sweep_label(key, value):
    """實驗名稱裡顯示的掃描值; 模型路徑太長, 只顯示名稱 (runs/detect/train_v2/weights/best.pt → train_v2)"""
    if key in FILE_KEYS:
        parent, filename = os.path.split(os.path.normpath(value))
        if os.path.basename(parent) == "weights":
            return os.path.basename(os.path.dirname(parent))
        return filename
    return value


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
        file_keys=FILE_KEYS, multi_keys=MULTI_KEYS)


def _back_with_error(msg):
    flash(msg, "error")
    return redirect(url_for("index"))


@app.route("/experiments", methods=["POST"])
def create_experiments():
    videos = request.form.getlist("videos")
    if not videos:
        return _back_with_error("請至少選一支影片")
    try:
        fixed, sweep = parse_param_form(request.form)
    except ValueError as e:
        return _back_with_error(str(e))

    for k in FILE_KEYS:
        if isinstance(fixed.get(k), str):
            save_file_setting(k, fixed[k])

    sweep_keys = list(sweep)
    combos = list(itertools.product(*sweep.values())) if sweep else [()]
    total = len(combos) * len(videos)
    if total > MAX_EXPERIMENTS_PER_SUBMIT:
        return _back_with_error(f"一次會建立 {total} 個實驗,超過上限 {MAX_EXPERIMENTS_PER_SUBMIT},請減少掃描值")

    base_name = request.form.get("name", "").strip() or datetime.now().strftime("實驗 %m/%d %H:%M")
    # 有填批次名稱就用 (只建一個實驗也歸進去, 可以補跑到舊批次); 沒填而且建多個實驗才自動用日期時間
    batch = request.form.get("batch", "").strip() or (datetime.now().strftime("%Y-%m-%d %H:%M") if total > 1 else None)
    for video in videos:
        for combo in combos:
            params = normalize_params({**fixed, **dict(zip(sweep_keys, combo))})
            name = base_name
            if sweep_keys:
                name += " [" + ", ".join(f"{k}={sweep_label(k, v)}" for k, v in zip(sweep_keys, combo)) + "]"
            jobs.enqueue(db.create_experiment(name, video, params, batch))
    flash(f"已建立 {total} 個實驗,排入背景執行", "ok")
    return redirect(url_for("index"))


@app.route("/experiments/<int:exp_id>")
def experiment_detail(exp_id):
    exp = get_experiment_or_404(exp_id)
    violations = db.list_violations(exp_id)
    for v in violations:
        v["status_text"] = plate_status(v["is_clean"], v["plate_conf"])
        # 片段在整支影片跑完才剪, 執行中或失敗的實驗可能還沒有檔案
        if v["clip_path"] and not os.path.isfile(os.path.join(db.experiment_dir(exp_id), v["clip_path"])):
            v["clip_path"] = None
    log_text = ""
    if exp["log"] and os.path.exists(exp["log"]):
        with open(exp["log"], encoding="utf-8") as f:
            log_text = "".join(f.readlines()[-300:])
    return render_template(
        "experiment.html", exp=exp, violations=violations, tracks=db.list_tracks(exp_id),
        metrics=db.compute_metrics(exp, violations), predictions=db.list_predictions(exp_id),
        specs=PARAM_SPECS, defaults=DEFAULT_PARAMS, log_text=log_text)


@app.route("/experiments/<int:exp_id>/stop", methods=["POST"])
def stop_experiment(exp_id):
    exp = db.get_experiment(exp_id)
    if exp and exp["status"] == "queued":
        db.update_experiment(exp_id, status="stopped", finished_at=db.now())
    elif exp and exp["status"] == "running":
        jobs.request_stop(exp_id)
    return redirect(request.referrer or url_for("index"))


@app.route("/experiments/<int:exp_id>/rerun", methods=["POST"])
def rerun_experiment(exp_id):
    exp = get_experiment_or_404(exp_id)
    new_id = db.create_experiment(exp["name"] + " (重跑)", exp["video"], exp["params"], exp["batch"])
    jobs.enqueue(new_id)
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


@app.route("/api/experiments")
def api_experiments():
    counts = db.violation_counts()
    return jsonify([{
        "id": e["id"], "status": e["status"], "status_text": STATUS_TEXT.get(e["status"], e["status"]),
        "progress": e["progress"] or 0, "counts": counts.get(e["id"], {}),
        "elapsed_sec": e["elapsed_sec"], "error": e["error"],
    } for e in db.list_experiments(ids=selected_ids())])


@app.route("/api/violations/<int:vid>", methods=["POST"])
def api_update_violation(vid):
    db.update_violation(vid, request.get_json(force=True) or {})
    return jsonify({"ok": True})
