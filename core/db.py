"""
PostgreSQL 資料庫: 每次實驗的參數、結果、違規紀錄、人工標註都存在資料庫 helmet_lab
連線字串從 .env (或環境變數) 的 DATABASE_URL 讀取, 例如 postgresql://postgres:密碼@localhost:5432/helmet_lab

資料表 (依 ER Model v8):
    batch    批次 (一次送出多個實驗)
    model    安全帽模型版本 (權重路徑)
    video    影片 (test_videos/ 底下的檔名 + fps)
    type     違規類型 (no_helmet / illegal_motorcycle_entry / speeding)
    run      每一次實驗 (一支影片 × 一組參數), 屬於 video / model / batch
    track    run 裡的每台車 (tid = ByteTrack 編號, 每個 run 從 1 開始, 所以 (run_id, tid) 才唯一)
    record   違規紀錄, 掛在 track 底下 + 人工標註 (gt_label: tp 正確 / fp 誤報 / dup 重複計數, gt_plate 正確車牌)
    vt       每支影片每種違規實際有幾台 (人工數的真值),用來算 recall
    img      從影片切出的單人裁切圖 + 人工真值 gt_ans
    ri       run 對 img 的預測; 多存 vid 用複合外鍵保證 run 和 img 是同一支影片

對外函式回傳的欄位名稱維持舊版 (experiments / violations / tracks) 的命名, app.py 和網頁模板不用改。
舊版資料表還在的話, init_db() 會自動把資料搬進新表 (舊表保留不刪)。
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
CREATE TABLE IF NOT EXISTS batch (
    bid SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT,
    res JSONB                       -- 批次彙總結果
);
CREATE TABLE IF NOT EXISTS model (
    mid SERIAL PRIMARY KEY,
    weights TEXT NOT NULL UNIQUE,   -- 安全帽模型權重路徑
    note TEXT
);
CREATE TABLE IF NOT EXISTS video (
    vid SERIAL PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,      -- test_videos/ 底下的檔名
    fps DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS type (
    cid SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS run (
    run_id SERIAL PRIMARY KEY,
    vid INTEGER NOT NULL REFERENCES video(vid),
    mid INTEGER REFERENCES model(mid),
    bid INTEGER REFERENCES batch(bid) ON DELETE SET NULL,
    name TEXT,
    params JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',   -- queued / running / done / stopped / failed
    progress DOUBLE PRECISION DEFAULT 0,
    created_at TEXT, started_at TEXT, finished_at TEXT,
    total_frames INTEGER, processed_frames INTEGER,
    elapsed_sec DOUBLE PRECISION, avg_ms_per_frame DOUBLE PRECISION, avg_track_ms DOUBLE PRECISION,
    avg_plate_ms DOUBLE PRECISION, avg_ocr_ms DOUBLE PRECISION,
    output_video TEXT,
    log TEXT,                        -- 執行紀錄檔路徑
    error TEXT,
    notes TEXT,
    UNIQUE (run_id, vid)             -- 給 ri 的複合外鍵參照
);
CREATE TABLE IF NOT EXISTS track (
    track_id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
    tid INTEGER NOT NULL,            -- ByteTrack 編號, 每個 run 從 1 開始
    class_name TEXT, first_sec DOUBLE PRECISION, last_sec DOUBLE PRECISION, frames INTEGER,
    avg_speed_kmh DOUBLE PRECISION, max_speed_kmh DOUBLE PRECISION,
    UNIQUE (run_id, tid)
);
CREATE TABLE IF NOT EXISTS record (
    rid SERIAL PRIMARY KEY,
    track_id INTEGER NOT NULL REFERENCES track(track_id) ON DELETE CASCADE,
    cid INTEGER NOT NULL REFERENCES type(cid),
    time_sec DOUBLE PRECISION, frame_idx INTEGER,
    confidence DOUBLE PRECISION,     -- 偵測信心值
    crop_path TEXT, plate_img TEXT,  -- 違規車輛截圖、車牌放大截圖
    clip_path TEXT,                  -- 超速影片片段 (經過測速區的時段前後)
    plate_raw TEXT, plate_fixed TEXT, plate_conf DOUBLE PRECISION, is_clean BOOLEAN, warnings TEXT,
    speed_kmh DOUBLE PRECISION,
    gt_label TEXT,     -- NULL 未標註 / tp 正確 / fp 誤報 / dup 同一台車重複計數
    gt_plate TEXT,     -- 人工確認的正確車牌
    note TEXT
);
CREATE TABLE IF NOT EXISTS vt (
    vid INTEGER NOT NULL REFERENCES video(vid) ON DELETE CASCADE,
    cid INTEGER NOT NULL REFERENCES type(cid),
    gt_count INTEGER NOT NULL,
    PRIMARY KEY (vid, cid)
);
CREATE TABLE IF NOT EXISTS img (
    iid SERIAL PRIMARY KEY,
    vid INTEGER NOT NULL REFERENCES video(vid) ON DELETE CASCADE,
    frame_idx INTEGER NOT NULL,
    bbox JSONB,
    gt_ans TEXT,
    UNIQUE (iid, vid)                -- 給 ri 的複合外鍵參照
);
CREATE TABLE IF NOT EXISTS ri (
    run_id INTEGER NOT NULL,
    iid INTEGER NOT NULL,
    vid INTEGER NOT NULL,
    pred JSONB,
    PRIMARY KEY (run_id, iid),
    FOREIGN KEY (run_id, vid) REFERENCES run(run_id, vid) ON DELETE CASCADE,
    FOREIGN KEY (iid, vid) REFERENCES img(iid, vid) ON DELETE CASCADE
);
ALTER TABLE record ADD COLUMN IF NOT EXISTS clip_path TEXT;
CREATE INDEX IF NOT EXISTS idx_run_bid ON run(bid);
CREATE INDEX IF NOT EXISTS idx_record_track ON record(track_id);
CREATE INDEX IF NOT EXISTS idx_img_vid ON img(vid);
"""

# 舊版資料表 → 新表; 保留原本的實驗編號 / 違規編號, data/experiments/<id> 和網址都不用改
_MIGRATE = """
INSERT INTO batch (name, created_at)
    SELECT batch, MIN(created_at) FROM experiments WHERE batch IS NOT NULL GROUP BY batch
    ON CONFLICT (name) DO NOTHING;
INSERT INTO model (weights)
    SELECT DISTINCT params->>'helmet_model_path' FROM experiments WHERE params->>'helmet_model_path' IS NOT NULL
    ON CONFLICT (weights) DO NOTHING;
INSERT INTO video (path, fps)
    SELECT video, MAX(fps) FROM (SELECT video, fps FROM experiments
                                 UNION ALL SELECT video, NULL FROM video_gt) s GROUP BY video
    ON CONFLICT (path) DO NOTHING;
INSERT INTO type (name)
    SELECT violation_type FROM violations WHERE violation_type IS NOT NULL
    UNION SELECT violation_type FROM video_gt
    ON CONFLICT (name) DO NOTHING;
INSERT INTO run (run_id, vid, mid, bid, name, params, status, progress, created_at, started_at, finished_at,
                 total_frames, processed_frames, elapsed_sec, avg_ms_per_frame, avg_track_ms, avg_plate_ms,
                 avg_ocr_ms, output_video, error, notes)
    SELECT e.id, v.vid, m.mid, b.bid, e.name, e.params, e.status, e.progress, e.created_at, e.started_at,
           e.finished_at, e.total_frames, e.processed_frames, e.elapsed_sec, e.avg_ms_per_frame, e.avg_track_ms,
           e.avg_plate_ms, e.avg_ocr_ms, e.output_video, e.error, e.notes
    FROM experiments e
    JOIN video v ON v.path = e.video
    LEFT JOIN model m ON m.weights = e.params->>'helmet_model_path'
    LEFT JOIN batch b ON b.name = e.batch;
INSERT INTO track (run_id, tid, class_name, first_sec, last_sec, frames, avg_speed_kmh, max_speed_kmh)
    SELECT experiment_id, track_id, class_name, first_sec, last_sec, frames, avg_speed_kmh, max_speed_kmh
    FROM tracks WHERE track_id IS NOT NULL
    ON CONFLICT (run_id, tid) DO NOTHING;
INSERT INTO track (run_id, tid)
    SELECT DISTINCT experiment_id, track_id FROM violations WHERE track_id IS NOT NULL
    ON CONFLICT (run_id, tid) DO NOTHING;
INSERT INTO record (rid, track_id, cid, time_sec, frame_idx, confidence, crop_path, plate_img, plate_raw,
                    plate_fixed, plate_conf, is_clean, warnings, speed_kmh, gt_label, gt_plate, note)
    SELECT o.id, t.track_id, c.cid, o.time_sec, o.frame_idx, o.det_conf, o.vehicle_img, o.plate_img, o.plate_raw,
           o.plate_fixed, o.plate_conf, o.is_clean <> 0, o.warnings, o.speed_kmh, o.label, o.plate_gt, o.note
    FROM violations o
    JOIN track t ON t.run_id = o.experiment_id AND t.tid = o.track_id
    JOIN type c ON c.name = o.violation_type;
INSERT INTO vt (vid, cid, gt_count)
    SELECT v.vid, c.cid, g.gt_count FROM video_gt g
    JOIN video v ON v.path = g.video
    JOIN type c ON c.name = g.violation_type
    WHERE g.gt_count IS NOT NULL;
SELECT setval(pg_get_serial_sequence('run', 'run_id'), GREATEST((SELECT MAX(run_id) FROM run), 1));
SELECT setval(pg_get_serial_sequence('record', 'rid'), GREATEST((SELECT MAX(rid) FROM record), 1));
"""

# 回傳舊版欄位名稱, app.py / 網頁模板不用改
_RUN_SELECT = """
SELECT r.run_id AS id, r.name, b.name AS batch, v.path AS video, v.fps, m.weights AS model,
       r.params, r.status, r.progress, r.created_at, r.started_at, r.finished_at,
       r.total_frames, r.processed_frames, r.elapsed_sec, r.avg_ms_per_frame, r.avg_track_ms,
       r.avg_plate_ms, r.avg_ocr_ms, r.output_video, r.log, r.error, r.notes
FROM run r
JOIN video v ON v.vid = r.vid
LEFT JOIN model m ON m.mid = r.mid
LEFT JOIN batch b ON b.bid = r.bid
"""

_RECORD_SELECT = """
SELECT rec.rid AS id, t.run_id AS experiment_id, rec.time_sec, rec.frame_idx, c.name AS violation_type,
       t.tid AS track_id, rec.confidence AS det_conf, rec.plate_raw, rec.plate_fixed, rec.plate_conf,
       rec.is_clean, rec.warnings, rec.speed_kmh, rec.crop_path AS vehicle_img, rec.plate_img, rec.clip_path,
       rec.gt_label AS label, rec.gt_plate AS plate_gt, rec.note
FROM record rec
JOIN track t ON t.track_id = rec.track_id
JOIN type c ON c.cid = rec.cid
"""

_RUN_COLS = {"name", "params", "status", "progress", "created_at", "started_at", "finished_at", "total_frames",
             "processed_frames", "elapsed_sec", "avg_ms_per_frame", "avg_track_ms", "avg_plate_ms", "avg_ocr_ms",
             "output_video", "log", "error", "notes"}

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
        conn.cursor().executemany("INSERT INTO type (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
                                  [(vt,) for vt in VIOLATION_TYPES])
        # 只在新表還沒用過時搬一次舊資料。不能用「run 是空的」判斷: 實驗全部刪掉後 run 也是空的,
        # 會把刪掉的實驗又搬回來。video 會在建立實驗 / 填真值 / 標註畫面時新增, 而且程式從不刪除,
        # 所以 video 是空的才代表新表從來沒用過
        has_legacy = conn.execute("SELECT to_regclass('experiments') IS NOT NULL ok").fetchone()["ok"]
        if has_legacy and not conn.execute("SELECT EXISTS (SELECT 1 FROM video) ok").fetchone()["ok"]:
            conn.execute(_MIGRATE)
            for r in conn.execute("SELECT run_id FROM run").fetchall():
                conn.execute("UPDATE run SET log=%s WHERE run_id=%s", (_log_path(r["run_id"]), r["run_id"]))


def experiment_dir(exp_id):
    return os.path.join(EXPERIMENTS_DIR, str(exp_id))


def _log_path(exp_id):
    return os.path.join(experiment_dir(exp_id), "log.txt")


def _exp_dict(row):
    if row is None:
        return None
    return dict(row)


def _upsert_id(conn, table, key_col, key, id_col, **extra):
    """依唯一欄位找出一列的編號, 沒有就新增 (extra 只在新增時寫入)"""
    cols = [key_col, *extra]
    row = conn.execute(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
        f"ON CONFLICT ({key_col}) DO UPDATE SET {key_col} = EXCLUDED.{key_col} RETURNING {id_col}",
        (key, *extra.values())).fetchone()
    return row[id_col]


def _video_id(conn, video):
    return _upsert_id(conn, "video", "path", video, "vid")


# ===== experiments (run) =====
def create_experiment(name, video, params, batch=None):
    with connect() as conn:
        vid = _video_id(conn, video)
        weights = params.get("helmet_model_path")
        mid = _upsert_id(conn, "model", "weights", weights, "mid") if weights else None
        bid = _upsert_id(conn, "batch", "name", batch, "bid", created_at=now()) if batch else None
        run_id = conn.execute(
            "INSERT INTO run (vid, mid, bid, name, params, status, created_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING run_id",
            (vid, mid, bid, name, Jsonb(params), "queued", now())).fetchone()["run_id"]
        conn.execute("UPDATE run SET log=%s WHERE run_id=%s", (_log_path(run_id), run_id))
        return run_id


def get_experiment(exp_id):
    with connect() as conn:
        return _exp_dict(conn.execute(_RUN_SELECT + " WHERE r.run_id=%s", (exp_id,)).fetchone())


def list_experiments(ids=None, batch=None):
    sql, args = _RUN_SELECT, []
    if ids:
        sql += " WHERE r.run_id = ANY(%s)"
        args = [list(ids)]
    elif batch:
        sql += " WHERE b.name=%s"
        args = [batch]
    with connect() as conn:
        rows = conn.execute(sql + " ORDER BY r.run_id DESC", args).fetchall()
    return [_exp_dict(r) for r in rows]


def update_experiment(exp_id, **fields):
    fps = fields.pop("fps", None)   # fps 是影片的屬性, 存在 video 表
    unknown = set(fields) - _RUN_COLS
    if unknown:
        raise ValueError(f"run 沒有這些欄位: {', '.join(sorted(unknown))}")
    with connect() as conn:
        if fields:
            cols = ", ".join(f"{k}=%s" for k in fields)
            values = [Jsonb(v) if k == "params" else v for k, v in fields.items()]
            conn.execute(f"UPDATE run SET {cols} WHERE run_id=%s", (*values, exp_id))
        if fps is not None:
            conn.execute("UPDATE video SET fps=%s WHERE vid=(SELECT vid FROM run WHERE run_id=%s)", (fps, exp_id))


def delete_experiment(exp_id):
    with connect() as conn:
        conn.execute("DELETE FROM run WHERE run_id=%s", (exp_id,))


def violation_counts():
    """每個實驗各違規類型的數量 {exp_id: {violation_type: n}}"""
    with connect() as conn:
        rows = conn.execute(
            "SELECT t.run_id, c.name, COUNT(*) n FROM record rec "
            "JOIN track t ON t.track_id = rec.track_id JOIN type c ON c.cid = rec.cid GROUP BY 1, 2").fetchall()
    out = {}
    for r in rows:
        out.setdefault(r["run_id"], {})[r["name"]] = r["n"]
    return out


# ===== violations (record) / tracks =====
def _track_id(conn, exp_id, tid):
    """(run_id, tid) 對應的 track 編號; 違規在 run 跑完 (add_tracks) 之前就會寫入, 所以先建一列空的 track"""
    return conn.execute(
        "INSERT INTO track (run_id, tid) VALUES (%s,%s) "
        "ON CONFLICT (run_id, tid) DO UPDATE SET tid = EXCLUDED.tid RETURNING track_id",
        (exp_id, tid)).fetchone()["track_id"]


def add_violation(exp_id, rec):
    with connect() as conn:
        conn.execute(
            "INSERT INTO record (track_id, cid, time_sec, frame_idx, confidence, crop_path, plate_img, clip_path, "
            "plate_raw, plate_fixed, plate_conf, is_clean, warnings, speed_kmh) "
            "VALUES (%s,(SELECT cid FROM type WHERE name=%s),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (_track_id(conn, exp_id, rec["track_id"]), rec["violation_type"], rec.get("time_sec"),
             rec.get("frame_idx"), rec.get("det_conf"), rec.get("vehicle_img"), rec.get("plate_img"),
             rec.get("clip_path"),
             rec.get("plate_raw"), rec.get("plate_fixed"), rec.get("plate_conf"), rec.get("is_clean"),
             rec.get("warnings"), rec.get("speed_kmh")))


def list_violations(exp_id):
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            _RECORD_SELECT + " WHERE t.run_id=%s ORDER BY rec.time_sec, rec.rid", (exp_id,)).fetchall()]


_LABEL_COLS = {"label": "gt_label", "plate_gt": "gt_plate", "note": "note"}


def update_violation(vid, fields):
    allowed = {k: (v if v != "" else None) for k, v in fields.items() if k in _LABEL_COLS}
    if "plate_gt" in allowed and allowed["plate_gt"]:
        allowed["plate_gt"] = allowed["plate_gt"].strip().upper()
    if not allowed:
        return
    cols = ", ".join(f"{_LABEL_COLS[k]}=%s" for k in allowed)
    with connect() as conn:
        conn.execute(f"UPDATE record SET {cols} WHERE rid=%s", (*allowed.values(), vid))


def add_tracks(exp_id, track_rows):
    with connect() as conn:
        conn.cursor().executemany(
            "INSERT INTO track (run_id, tid, class_name, first_sec, last_sec, frames, avg_speed_kmh, max_speed_kmh) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (run_id, tid) DO UPDATE SET "
            "class_name=EXCLUDED.class_name, first_sec=EXCLUDED.first_sec, last_sec=EXCLUDED.last_sec, "
            "frames=EXCLUDED.frames, avg_speed_kmh=EXCLUDED.avg_speed_kmh, max_speed_kmh=EXCLUDED.max_speed_kmh",
            [(exp_id, t["track_id"], t["class_name"], t["first_sec"], t["last_sec"], t["frames"],
              t["avg_speed_kmh"], t["max_speed_kmh"]) for t in track_rows])


def list_tracks(exp_id):
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT track_id AS id, run_id AS experiment_id, tid AS track_id, class_name, first_sec, last_sec, "
            "frames, avg_speed_kmh, max_speed_kmh FROM track WHERE run_id=%s ORDER BY tid", (exp_id,)).fetchall()]


# ===== 影片真值 (vt) =====
def get_video_gt(video):
    with connect() as conn:
        rows = conn.execute(
            "SELECT c.name, vt.gt_count FROM vt JOIN video v ON v.vid = vt.vid JOIN type c ON c.cid = vt.cid "
            "WHERE v.path=%s", (video,)).fetchall()
    return {r["name"]: r["gt_count"] for r in rows}


def set_video_gt(video, vtype, count):
    with connect() as conn:
        if count is None:
            conn.execute("DELETE FROM vt USING video v, type c "
                         "WHERE vt.vid = v.vid AND vt.cid = c.cid AND v.path=%s AND c.name=%s", (video, vtype))
        else:
            conn.execute("INSERT INTO vt (vid, cid, gt_count) VALUES (%s,(SELECT cid FROM type WHERE name=%s),%s) "
                         "ON CONFLICT (vid, cid) DO UPDATE SET gt_count = EXCLUDED.gt_count",
                         (_video_id(conn, video), vtype, count))


# ===== 裁切圖 (img) / 預測 (ri) =====
# frame_idx 跟 processor 的 raw_frame_idx 一樣從 1 開始, 時間 = frame_idx / fps
_IMG_SELECT = """
SELECT i.iid, v.path AS video, i.frame_idx, i.bbox, i.gt_ans, ROUND((i.frame_idx / v.fps)::numeric, 2)::float AS time_sec
FROM img i JOIN video v ON v.vid = i.vid
"""


def add_img(video, frame_idx, bbox=None, gt_ans=None, fps=None):
    with connect() as conn:
        vid = _video_id(conn, video)
        if fps:
            conn.execute("UPDATE video SET fps=%s WHERE vid=%s", (fps, vid))
        return conn.execute(
            "INSERT INTO img (vid, frame_idx, bbox, gt_ans) VALUES (%s,%s,%s,%s) RETURNING iid",
            (vid, frame_idx, Jsonb(bbox) if bbox is not None else None, gt_ans)
        ).fetchone()["iid"]


def get_img(iid):
    with connect() as conn:
        row = conn.execute(_IMG_SELECT + " WHERE i.iid=%s", (iid,)).fetchone()
    return dict(row) if row else None


def list_imgs(video):
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            _IMG_SELECT + " WHERE v.path=%s ORDER BY i.frame_idx, i.iid", (video,)).fetchall()]


def img_counts():
    """每支影片有幾張標註畫面 {video: n}"""
    with connect() as conn:
        rows = conn.execute("SELECT v.path, COUNT(*) n FROM img i JOIN video v ON v.vid = i.vid GROUP BY 1").fetchall()
    return {r["path"]: r["n"] for r in rows}


def delete_img(iid):
    with connect() as conn:
        conn.execute("DELETE FROM img WHERE iid=%s", (iid,))


def list_predictions(exp_id):
    """這個實驗的影片的所有標註畫面 + 比對結果 (pred 是 None = 實驗跑完之後才標註的)"""
    with connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT i.iid, i.frame_idx, i.bbox, i.gt_ans, ri.pred, "
            "ROUND((i.frame_idx / v.fps)::numeric, 2)::float AS time_sec "
            "FROM run r JOIN video v ON v.vid = r.vid JOIN img i ON i.vid = r.vid "
            "LEFT JOIN ri ON ri.iid = i.iid AND ri.run_id = r.run_id "
            "WHERE r.run_id=%s ORDER BY i.frame_idx, i.iid", (exp_id,)).fetchall()]


def add_prediction(exp_id, iid, pred):
    """vid 取 run 的影片; img 不是同一支影片的話複合外鍵會擋下來 (psycopg.errors.ForeignKeyViolation)"""
    with connect() as conn:
        conn.execute(
            "INSERT INTO ri (run_id, iid, vid, pred) SELECT run_id, %s, vid, %s FROM run WHERE run_id=%s "
            "ON CONFLICT (run_id, iid) DO UPDATE SET pred = EXCLUDED.pred",
            (iid, Jsonb(pred), exp_id))


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


def img_metrics(predictions):
    """標註畫面比對: 抓到 / 有偵測沒判違規 / 沒偵測到; 抓到率 = 抓到 / (三者合計), 沒跑到和實驗後才標註的不算"""
    counts = {"hit": 0, "detected": 0, "missed": 0, "skipped": 0, "pending": 0}
    for p in predictions:
        counts[p["pred"]["status"] if p["pred"] else "pending"] += 1
    judged = counts["hit"] + counts["detected"] + counts["missed"]
    return {**counts, "judged": judged, "hit_rate": _ratio(counts["hit"], judged)}


def compute_metrics(exp, violations=None):
    """
    依人工標註算出實驗指標:
        每種違規: 偵測數、tp/fp/dup/未標註、precision、recall(需要影片真值)、F1
        車牌: 有填正確車牌的筆數、修正後完全正確率、原始OCR完全正確率、字元正確率、格式正常率、平均信心值
        標註畫面: 影片上框出的違規有幾個被抓到
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
    return {"per_type": per_type, "plate": plate, "img": img_metrics(list_predictions(exp["id"])),
            "total": len(violations)}
