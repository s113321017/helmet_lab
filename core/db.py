"""
PostgreSQL 資料庫: 每次實驗的參數、結果、違規紀錄、人工標註都存在資料庫 helmet_lab
連線字串從 .env (或環境變數) 的 DATABASE_URL 讀取, 例如 postgresql://postgres:密碼@localhost:5432/helmet_lab

資料表:
    experiments  每一次實驗 (一支影片 × 一組參數)
    violations   實驗偵測到的違規 + 人工標註 (label: tp 正確 / fp 誤報 / dup 重複計數, plate_gt 正確車牌)
    tracks       實驗裡每個 track id 的統計 (出現時間、幀數、平均/最高車速)
    video_gt     每支影片實際有幾台違規 (人工數的真值),用來算 recall
"""

import os
import threading
from datetime import datetime

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .config import BASE_DIR, env
from .pipeline import VIOLATION_TYPES

DATA_DIR = os.path.join(BASE_DIR, "data")
EXPERIMENTS_DIR = os.path.join(DATA_DIR, "experiments")

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    name TEXT,
    batch TEXT,
    video TEXT NOT NULL,
    params JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',   -- queued / running / done / stopped / failed
    progress DOUBLE PRECISION DEFAULT 0,
    created_at TEXT, started_at TEXT, finished_at TEXT,
    fps DOUBLE PRECISION, total_frames INTEGER, processed_frames INTEGER,
    elapsed_sec DOUBLE PRECISION, avg_ms_per_frame DOUBLE PRECISION, avg_track_ms DOUBLE PRECISION,
    avg_plate_ms DOUBLE PRECISION, avg_ocr_ms DOUBLE PRECISION,
    output_video TEXT,
    error TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS violations (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    time_sec DOUBLE PRECISION, frame_idx INTEGER, violation_type TEXT, track_id INTEGER,
    det_conf DOUBLE PRECISION,
    plate_raw TEXT, plate_fixed TEXT, plate_conf DOUBLE PRECISION, is_clean INTEGER, warnings TEXT,
    speed_kmh DOUBLE PRECISION, vehicle_img TEXT, plate_img TEXT,
    label TEXT,        -- NULL 未標註 / tp 正確 / fp 誤報 / dup 同一台車重複計數
    plate_gt TEXT,     -- 人工確認的正確車牌
    note TEXT
);
CREATE TABLE IF NOT EXISTS tracks (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    track_id INTEGER, class_name TEXT, first_sec DOUBLE PRECISION, last_sec DOUBLE PRECISION, frames INTEGER,
    avg_speed_kmh DOUBLE PRECISION, max_speed_kmh DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS video_gt (
    video TEXT NOT NULL,
    violation_type TEXT NOT NULL,
    gt_count INTEGER,
    PRIMARY KEY (video, violation_type)
);
CREATE INDEX IF NOT EXISTS idx_violations_exp ON violations(experiment_id);
CREATE INDEX IF NOT EXISTS idx_tracks_exp ON tracks(experiment_id);
"""

LABELS = {"tp": "正確", "fp": "誤報", "dup": "重複"}


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_pool = None
_pool_lock = threading.Lock()


def connect():
    """從連線池取一條連線; 用法 `with connect() as conn:`, 區塊結束自動 commit 並還給連線池"""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                url = env("DATABASE_URL")
                if not url:
                    raise RuntimeError("找不到 DATABASE_URL, 請在 .env 設定, 例如 "
                                       "postgresql://postgres:密碼@localhost:5432/helmet_lab")
                _pool = ConnectionPool(url, min_size=1, max_size=10, open=True,
                                       kwargs={"row_factory": dict_row})
    return _pool.connection()


def init_db():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    with connect() as conn:
        conn.execute(SCHEMA)


def experiment_dir(exp_id):
    return os.path.join(EXPERIMENTS_DIR, str(exp_id))


def _exp_dict(row):
    if row is None:
        return None
    return dict(row)


# ===== experiments =====
def create_experiment(name, video, params, batch=None):
    with connect() as conn:
        row = conn.execute(
            "INSERT INTO experiments (name, batch, video, params, status, created_at) "
            "VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
            (name, batch, video, Jsonb(params), "queued", now())).fetchone()
        return row["id"]


def get_experiment(exp_id):
    with connect() as conn:
        return _exp_dict(conn.execute("SELECT * FROM experiments WHERE id=%s", (exp_id,)).fetchone())


def list_experiments(ids=None, batch=None):
    sql, args = "SELECT * FROM experiments", []
    if ids:
        sql += " WHERE id = ANY(%s)"
        args = [list(ids)]
    elif batch:
        sql += " WHERE batch=%s"
        args = [batch]
    with connect() as conn:
        rows = conn.execute(sql + " ORDER BY id DESC", args).fetchall()
    return [_exp_dict(r) for r in rows]


def update_experiment(exp_id, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k}=%s" for k in fields)
    values = [Jsonb(v) if k == "params" else v for k, v in fields.items()]
    with connect() as conn:
        conn.execute(f"UPDATE experiments SET {cols} WHERE id=%s", (*values, exp_id))


def delete_experiment(exp_id):
    with connect() as conn:
        conn.execute("DELETE FROM experiments WHERE id=%s", (exp_id,))


def violation_counts():
    """每個實驗各違規類型的數量 {exp_id: {violation_type: n}}"""
    with connect() as conn:
        rows = conn.execute(
            "SELECT experiment_id, violation_type, COUNT(*) n FROM violations GROUP BY 1, 2").fetchall()
    out = {}
    for r in rows:
        out.setdefault(r["experiment_id"], {})[r["violation_type"]] = r["n"]
    return out


# ===== violations / tracks =====
_VIOLATION_COLS = ["time_sec", "frame_idx", "violation_type", "track_id", "det_conf", "plate_raw", "plate_fixed",
                   "plate_conf", "is_clean", "warnings", "speed_kmh", "vehicle_img", "plate_img"]


def add_violation(exp_id, rec):
    values = [rec.get(c) for c in _VIOLATION_COLS]
    values = [int(v) if isinstance(v, bool) else v for v in values]   # is_clean 欄位是 INTEGER
    with connect() as conn:
        conn.execute(
            f"INSERT INTO violations (experiment_id, {', '.join(_VIOLATION_COLS)}) "
            f"VALUES (%s{', %s' * len(_VIOLATION_COLS)})",
            (exp_id, *values))


def list_violations(exp_id):
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM violations WHERE experiment_id=%s ORDER BY time_sec, id", (exp_id,)).fetchall()]


def update_violation(vid, fields):
    allowed = {k: (v if v != "" else None) for k, v in fields.items() if k in ("label", "plate_gt", "note")}
    if "plate_gt" in allowed and allowed["plate_gt"]:
        allowed["plate_gt"] = allowed["plate_gt"].strip().upper()
    if not allowed:
        return
    cols = ", ".join(f"{k}=%s" for k in allowed)
    with connect() as conn:
        conn.execute(f"UPDATE violations SET {cols} WHERE id=%s", (*allowed.values(), vid))


def add_tracks(exp_id, track_rows):
    with connect() as conn:
        conn.cursor().executemany(
            "INSERT INTO tracks (experiment_id, track_id, class_name, first_sec, last_sec, frames, "
            "avg_speed_kmh, max_speed_kmh) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            [(exp_id, t["track_id"], t["class_name"], t["first_sec"], t["last_sec"], t["frames"],
              t["avg_speed_kmh"], t["max_speed_kmh"]) for t in track_rows])


def list_tracks(exp_id):
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM tracks WHERE experiment_id=%s ORDER BY track_id", (exp_id,)).fetchall()]


# ===== 影片真值 =====
def get_video_gt(video):
    with connect() as conn:
        rows = conn.execute("SELECT violation_type, gt_count FROM video_gt WHERE video=%s", (video,)).fetchall()
    return {r["violation_type"]: r["gt_count"] for r in rows}


def set_video_gt(video, vtype, count):
    with connect() as conn:
        if count is None:
            conn.execute("DELETE FROM video_gt WHERE video=%s AND violation_type=%s", (video, vtype))
        else:
            conn.execute("INSERT INTO video_gt (video, violation_type, gt_count) VALUES (%s,%s,%s) "
                         "ON CONFLICT (video, violation_type) DO UPDATE SET gt_count = EXCLUDED.gt_count",
                         (video, vtype, count))


# ===== 指標計算 =====
def _norm_plate(s):
    return "".join(ch for ch in (s or "").upper() if ch.isalnum())


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _ratio(a, b):
    return round(a / b, 3) if b else None


def compute_metrics(exp, violations=None):
    """
    依人工標註算出實驗指標:
        每種違規: 偵測數、tp/fp/dup/未標註、precision、recall(需要影片真值)、F1
        車牌: 有填正確車牌的筆數、修正後完全正確率、原始OCR完全正確率、字元正確率、格式正常率、平均信心值
    """
    if violations is None:
        violations = list_violations(exp["id"])
    gt = get_video_gt(exp["video"])

    per_type = {}
    for vt in VIOLATION_TYPES:
        vs = [v for v in violations if v["violation_type"] == vt]
        tp = sum(v["label"] == "tp" for v in vs)
        fp = sum(v["label"] == "fp" for v in vs)
        dup = sum(v["label"] == "dup" for v in vs)
        precision = _ratio(tp, tp + fp + dup)
        recall = _ratio(tp, gt[vt]) if gt.get(vt) else None
        f1 = round(2 * precision * recall / (precision + recall), 3) if precision and recall else None
        per_type[vt] = {"detected": len(vs), "tp": tp, "fp": fp, "dup": dup,
                        "unlabeled": len(vs) - tp - fp - dup, "gt": gt.get(vt),
                        "precision": precision, "recall": recall, "f1": f1}

    with_gt = [v for v in violations if v["plate_gt"]]
    fixed_ok = sum(_norm_plate(v["plate_fixed"]) == _norm_plate(v["plate_gt"]) for v in with_gt)
    raw_ok = sum(_norm_plate(v["plate_raw"]) == _norm_plate(v["plate_gt"]) for v in with_gt)
    char_total = sum(len(_norm_plate(v["plate_gt"])) for v in with_gt)
    char_err = sum(_edit_distance(_norm_plate(v["plate_fixed"]), _norm_plate(v["plate_gt"])) for v in with_gt)
    confs = [v["plate_conf"] for v in violations if v["plate_conf"] is not None]
    plate = {
        "labeled": len(with_gt),
        "fixed_acc": _ratio(fixed_ok, len(with_gt)),
        "raw_acc": _ratio(raw_ok, len(with_gt)),
        "char_acc": round(max(0.0, 1 - char_err / char_total), 3) if char_total else None,
        "clean_rate": _ratio(sum(bool(v["is_clean"]) for v in violations), len(violations)),
        "avg_conf": round(sum(confs) / len(confs), 3) if confs else None,
    }
    return {"per_type": per_type, "plate": plate, "total": len(violations)}
