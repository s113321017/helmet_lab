"""
背景執行實驗: 一個工作執行緒依序跑排隊中的實驗 (GPU 一次只跑一個)
"""

import os
import queue
import threading
import traceback

from core import db
from core.pipeline import process_video

from .common import VIDEO_DIR

_job_queue = queue.Queue()
_stop_requests = set()


def enqueue(exp_id):
    _job_queue.put(exp_id)


def request_stop(exp_id):
    """執行中的實驗: 下一幀處理前停止 (排隊中的直接把狀態改成已停止就好, 不用呼叫這個)"""
    _stop_requests.add(exp_id)


def run_experiment(exp_id):
    exp = db.get_experiment(exp_id)
    if exp is None or exp["status"] != "queued":
        return   # 排隊期間被停止或刪除

    out_dir = db.experiment_dir(exp_id)
    os.makedirs(out_dir, exist_ok=True)
    db.update_experiment(exp_id, status="running", started_at=db.now(), progress=0)

    with open(exp["log"], "a", encoding="utf-8") as log_file:
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
                gt_imgs=db.list_imgs(exp["video"]),
            )
            db.add_tracks(exp_id, summary["tracks"])
            for r in summary["gt_results"]:
                db.add_prediction(exp_id, r["iid"], r["pred"])
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
            run_experiment(exp_id)
        except Exception:
            traceback.print_exc()


def start_worker():
    # 伺服器上次被關掉時還在跑的實驗標成失敗,還在排隊的重新排入
    for exp in reversed(db.list_experiments()):
        if exp["status"] == "running":
            db.update_experiment(exp["id"], status="failed", error="伺服器中斷,請重新執行")
        elif exp["status"] == "queued":
            enqueue(exp["id"])
    threading.Thread(target=_worker, daemon=True).start()
