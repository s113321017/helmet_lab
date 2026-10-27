"""
比較多個實驗 (參數差異、指標、圖表) + CSV 匯出
"""

import json

from flask import render_template

from core import db
from core.pipeline import DEFAULT_PARAMS, VIOLATION_TYPES, plate_status

from . import app
from .common import selected_experiments, csv_response

FIXED_PARAMS = {"no_helmet_class", "motorbike_class", "tracker"}   # 固定不變的參數, CSV 匯出不放
EXPORT_PARAM_KEYS = [k for k in DEFAULT_PARAMS if k not in FIXED_PARAMS]
PARAM_HEADERS = [f"p_{k}" for k in EXPORT_PARAM_KEYS]


def export_params(e):
    return [e["params"].get(k) for k in EXPORT_PARAM_KEYS]


def params_diff_keys(exps):
    """找出這些實驗之間有差異的參數"""
    return [k for k in DEFAULT_PARAMS if len({json.dumps(e["params"].get(k)) for e in exps}) > 1]


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
    exps = selected_experiments()
    rows = [(e, db.compute_metrics(e)) for e in exps]
    diff_keys = params_diff_keys(exps)
    return render_template("compare.html", rows=rows, diff_keys=diff_keys,
                           ids=",".join(str(e["id"]) for e in exps),
                           chart=chart_data(rows, diff_keys),
                           chart_types=[{"key": k, "name": n} for k, n in VIOLATION_TYPES.items()])


# ===== CSV 匯出 (舊的實驗在前) =====
@app.route("/export/violations.csv")
def export_violations():
    headers = (["experiment_id", "experiment_name", "batch", "video"] + PARAM_HEADERS
               + ["time_sec", "frame_idx", "violation_type", "track_id", "det_conf", "plate_text_raw",
                  "plate_text_fixed", "plate_conf", "speed_kmh", "status", "warnings",
                  "label", "plate_gt", "note"])
    rows = [[e["id"], e["name"], e["batch"], e["video"]] + export_params(e)
            + [v["time_sec"], v["frame_idx"], v["violation_type"], v["track_id"], v["det_conf"],
               v["plate_raw"], v["plate_fixed"], v["plate_conf"], v["speed_kmh"],
               plate_status(v["is_clean"], v["plate_conf"]), v["warnings"],
               v["label"], v["plate_gt"], v["note"]]
            for e in reversed(selected_experiments(default_all=True)) for v in db.list_violations(e["id"])]
    return csv_response("violations.csv", headers, rows)


@app.route("/export/tracks.csv")
def export_tracks():
    headers = ["experiment_id", "experiment_name", "video", "track_id", "class_name", "first_sec", "last_sec",
               "frames", "avg_speed_kmh", "max_speed_kmh"]
    rows = [[e["id"], e["name"], e["video"], t["track_id"], t["class_name"], t["first_sec"],
             t["last_sec"], t["frames"], t["avg_speed_kmh"], t["max_speed_kmh"]]
            for e in reversed(selected_experiments(default_all=True)) for t in db.list_tracks(e["id"])]
    return csv_response("tracks.csv", headers, rows)


METRIC_KEYS = ["detected", "tp", "fp", "dup", "unlabeled", "gt", "precision", "recall", "f1"]
PLATE_KEYS = ["labeled", "fixed_acc", "raw_acc", "char_acc", "clean_rate", "avg_conf"]
IMG_KEYS = ["judged", "hit", "detected", "missed", "hit_rate"]
TIMING_KEYS = ["processed_frames", "elapsed_sec", "avg_ms_per_frame", "avg_track_ms", "avg_plate_ms", "avg_ocr_ms"]


@app.route("/export/compare.csv")
def export_compare():
    headers = (["experiment_id", "name", "batch", "video", "status"] + PARAM_HEADERS
               + [f"{vt}_{m}" for vt in VIOLATION_TYPES for m in METRIC_KEYS]
               + [f"plate_{m}" for m in PLATE_KEYS] + [f"img_{m}" for m in IMG_KEYS] + TIMING_KEYS)
    rows = []
    for e in reversed(selected_experiments()):
        m = db.compute_metrics(e)
        rows.append([e["id"], e["name"], e["batch"], e["video"], e["status"]] + export_params(e)
                    + [m["per_type"][vt][k] for vt in VIOLATION_TYPES for k in METRIC_KEYS]
                    + [m["plate"][k] for k in PLATE_KEYS] + [m["img"][k] for k in IMG_KEYS]
                    + [e[k] for k in TIMING_KEYS])
    return csv_response("compare.csv", headers, rows)
