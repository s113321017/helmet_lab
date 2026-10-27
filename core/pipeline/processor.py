"""
處理一支影片: 串起 安全帽/機車偵測 → 違規判斷 (未戴安全帽 / 機車禁行 / 超速) → 車牌定位 + OCR → 存檔
"""

import os
import time

import cv2

from .params import normalize_params
from .helmet import HelmetDetector
from .plate import get_plate_model, locate_and_read_plate
from .calibration import load_homography_matrix
from .speed import SpeedEstimator
from .violations import VIOLATION_TYPES, FILE_TAG, ViolationCounter, crop_vehicle
from .video_output import open_video_writer, draw_annotations, write_clip
from .gt_match import GtMatcher


def process_video(video_path, params, out_dir, on_violation=None, on_progress=None,
                  should_stop=None, log=print, gt_imgs=None):
    """
    處理一支影片。
        params       : 要覆蓋 DEFAULT_PARAMS 的參數
        out_dir      : 截圖存到 out_dir/violations/,標註影片存到 out_dir/
        on_violation : 每確認一筆違規就呼叫 on_violation(record)
        on_progress  : on_progress(進度0~1, 已處理幀數)
        should_stop  : 回傳 True 就提前結束
        gt_imgs      : 人工標註畫面 (img 真值), 跑的時候比對那個時間、位置偵測到什麼 (見 gt_match.py)
    回傳摘要 dict (含 records 違規清單、tracks 每台車統計、計時資訊、gt_results 真值比對結果)
    """
    p = normalize_params(params)
    name = os.path.splitext(os.path.basename(video_path))[0]
    vio_dir = os.path.join(out_dir, "violations")
    os.makedirs(vio_dir, exist_ok=True)

    detector = HelmetDetector(p)
    plate_model = get_plate_model(p["plate_model_path"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"無法開啟影片: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    max_raw = int(p["max_seconds"] * fps) if p["max_seconds"] > 0 else None
    if max_raw and total_frames:
        total_frames = min(total_frames, max_raw)

    # ----- 測速 (需要相機標定矩陣) -----
    homography_matrix, speed_zone = (load_homography_matrix(p["homography_path"], (w, h), log)
                                     if p["speed_enabled"] else (None, None))
    if not p["speed_zone_enabled"]:
        speed_zone = None
    speed = (SpeedEstimator(homography_matrix, speed_zone, p["speed_history_len"])
             if homography_matrix is not None else None)
    log(f"速度估算: {'啟用' if speed else '停用'}"
        + (f" (速限 {p['speed_limit_kmh']} km/h, {'只在測速區內' if speed_zone is not None else '整個畫面'})"
           if speed else ""))

    if (p["motorcycle_restricted"] or speed) and detector.motorbike_id is None:
        log(f"⚠️ 找不到類別名稱 '{p['motorbike_class']}',機車禁行/速度估算會失效。模型類別: {detector.class_names}")

    # ----- 標註影片 -----
    writer, out_w, out_h, output_video = None, 0, 0, None
    if p["save_video"]:
        out_w = min(1280, w)
        out_h = int(h * out_w / w)
        output_video = f"detected_{name}.mp4"
        writer = open_video_writer(os.path.join(out_dir, output_video), fps / p["vid_stride"], (out_w, out_h))
        if writer is None:
            log("⚠️ 無法建立輸出影片,略過影片輸出")
            output_video = None

    counters = {
        "no_helmet": ViolationCounter(p["confirm_frames"]),
        "illegal_motorcycle_entry": ViolationCounter(p["confirm_frames"]),
        "speeding": ViolationCounter(p["speed_confirm_frames"]),
    }
    pending_speeding = {}   # 已確認超速、還沒找到車牌的 track_id -> (確認當下車速, 那次經過測速區的時段)
    zone_spans = {}         # track_id -> 這台車最近一次經過測速區的時段 {"start", "end", "inside"} (秒)
    track_boxes = {}        # track_id -> {raw_frame_idx: (box, 文字)} 機車每幀的位置, 剪超速片段時畫框用
    clips = []              # 要剪的超速片段 (track_id, 時段, 檔案路徑)
    # 標註時間前後 0.2 秒 (至少一個跳幀間隔) 內的偵測都拿來比
    gt_matcher = GtMatcher(gt_imgs or [], max(p["vid_stride"], int(fps * 0.2)),
                           {"no_helmet": detector.no_helmet_id, "illegal_motorcycle_entry": detector.motorbike_id,
                            "speeding": detector.motorbike_id})
    tracks = {}             # track_id -> 統計 (出現時間、幀數、車速)
    records = []
    timing = {"track_ms": 0.0, "plate_ms": 0.0, "ocr_ms": 0.0, "plate_calls": 0, "ocr_calls": 0}

    def try_record(vtype, tid, box, det_conf, frame, time_sec, frame_idx, speed_kmh=None, span=None):
        """確認違規後定位車牌並存檔,有找到車牌才算數 (回傳 True); span = 超速那次經過測速區的時段"""
        vehicle_crop = crop_vehicle(frame, box, vtype)
        base = f"{name}_{FILE_TAG[vtype]}_track{tid}"
        plate = locate_and_read_plate(plate_model, vehicle_crop, os.path.join(vio_dir, base + "_plate.jpg"),
                                      p, timing, log)
        if plate is None:
            return False
        cv2.imwrite(os.path.join(vio_dir, base + "_vehicle.jpg"), vehicle_crop)

        rec = {
            "video": name, "time_sec": round(time_sec, 2), "frame_idx": frame_idx,
            "violation_type": vtype, "track_id": tid, "det_conf": round(det_conf, 4),
            "plate_raw": plate["plate_raw"], "plate_fixed": plate["plate_fixed"],
            "plate_conf": plate["plate_conf"], "is_clean": plate["is_clean"],
            "warnings": "; ".join(plate["warnings"]),
            "speed_kmh": round(speed_kmh, 1) if speed_kmh is not None else None,
            "vehicle_img": f"violations/{base}_vehicle.jpg", "plate_img": f"violations/{base}_plate.jpg",
            "clip_path": f"violations/{base}_clip.mp4" if span else None,
        }
        if span:
            # 車子可能還在測速區裡, 時段還沒結束, 等整支影片跑完再剪
            clips.append((tid, span, os.path.join(out_dir, rec["clip_path"])))
        records.append(rec)
        if on_violation:
            on_violation(rec)
        counters[vtype].saved.add(tid)
        return True

    raw_frame_idx = 0    # 原始影片的幀編號(沒有跳過的)
    processed_idx = 0    # 實際處理過的幀數
    stopped = False
    start = time.time()

    while True:
        if max_raw and raw_frame_idx >= max_raw:
            break
        ok = cap.grab()   # 只解碼到內部緩衝區,還沒完整轉出畫面,比read()快
        if not ok:
            break
        raw_frame_idx += 1
        if (raw_frame_idx - 1) % p["vid_stride"] != 0:
            continue   # 跳過這幀,不做完整解碼(vid_stride的手動實作)

        ok, frame = cap.retrieve()   # 這裡才真正把畫面解碼出來
        if not ok:
            break
        processed_idx += 1
        time_sec = raw_frame_idx / fps

        if should_stop and should_stop():
            stopped = True
            break

        t0 = time.time()
        detections = detector.track(frame)
        timing["track_ms"] += (time.time() - t0) * 1000
        gt_matcher.update(raw_frame_idx, detections, time_sec)

        drawn = []   # 要畫在輸出影片上的框 (box, 顏色, 文字)
        for box, tid, cls_id, det_conf in detections:
            t = tracks.setdefault(tid, {"class_name": detector.class_names[cls_id], "first_sec": time_sec,
                                        "frames": 0, "speeds": []})
            t["last_sec"] = time_sec
            t["frames"] += 1

            if cls_id == detector.no_helmet_id:
                # --- 未戴安全帽 ---
                counter = counters["no_helmet"]
                counter.hit(tid)
                if counter.confirmed(tid):
                    log(f"  🚨 track#{tid} 未戴安全帽違規確認,嘗試定位車牌...")
                    try_record("no_helmet", tid, box, det_conf, frame, time_sec, raw_frame_idx)
                drawn.append((box, (0, 0, 255), f"#{tid} no helmet"))

            elif cls_id == detector.motorbike_id:
                label = f"#{tid} motorbike"

                # --- 機車禁行 ---
                if p["motorcycle_restricted"]:
                    counter = counters["illegal_motorcycle_entry"]
                    counter.hit(tid)
                    if counter.confirmed(tid):
                        log(f"  🚨 track#{tid} 機車禁行違規確認,嘗試定位車牌...")
                        try_record("illegal_motorcycle_entry", tid, box, det_conf, frame, time_sec, raw_frame_idx)

                # --- 超速 ---
                if speed:
                    inside, speed_kmh = speed.update(tid, box, time_sec)
                    span = zone_spans.get(tid)
                    if inside:
                        if span is None or not span["inside"]:   # 新的一次經過測速區
                            span = zone_spans[tid] = {"start": time_sec, "inside": True}
                        span["end"] = time_sec
                    elif span is not None:
                        span["inside"] = False
                    if inside:
                        counter = counters["speeding"]
                        is_speeding = speed_kmh is not None and speed_kmh > p["speed_limit_kmh"]
                        if is_speeding:
                            counter.hit(tid)
                        else:
                            counter.miss(tid)
                        if speed_kmh is not None:
                            t["speeds"].append(speed_kmh)
                            label += f" {speed_kmh:.1f}km/h"
                        if is_speeding and counter.confirmed(tid) and tid not in pending_speeding:
                            log(f"  🚨 track#{tid} 超速違規確認 (約 {speed_kmh:.1f} km/h),嘗試定位車牌...")
                            pending_speeding[tid] = (speed_kmh, span)

                    # 已確認超速但還沒拍到車牌: 出了測速區也繼續找車牌 (車速用確認當下的)
                    if tid in pending_speeding:
                        if try_record("speeding", tid, box, det_conf, frame, time_sec,
                                      raw_frame_idx, *pending_speeding[tid]):
                            del pending_speeding[tid]
                    track_boxes.setdefault(tid, {})[raw_frame_idx] = (box, label)

                drawn.append((box, (255, 128, 0), label))

        if writer is not None:
            draw_annotations(frame, drawn, speed_zone if speed else None)
            writer.write(cv2.resize(frame, (out_w, out_h)))

        if processed_idx % 10 == 0 and on_progress:
            on_progress(raw_frame_idx / total_frames if total_frames else 0, processed_idx)
        if processed_idx % 100 == 0:
            elapsed = time.time() - start
            log(f"  [計時] 已處理 {processed_idx} 幀,累計耗時 {elapsed:.1f} 秒,"
                f"平均每幀 {elapsed / processed_idx * 1000:.1f} ms")

    cap.release()
    if writer is not None:
        writer.release()

    for tid, span, clip_path in clips:
        first_raw = max(1, int((span["start"] - p["clip_pad_sec"]) * fps))
        last_raw = int((span["end"] + p["clip_pad_sec"]) * fps)
        if write_clip(video_path, clip_path, first_raw, last_raw, track_boxes.get(tid, {}), speed_zone,
                      max_gap=p["vid_stride"]):
            log(f"  🎬 track#{tid} 超速片段 {span['start']:.1f}~{span['end']:.1f} 秒 → {os.path.basename(clip_path)}")
        else:
            log(f"  ⚠️ track#{tid} 無法建立超速片段")

    gt_results = gt_matcher.results(detector.class_names, {vt: c.saved for vt, c in counters.items()})
    if gt_results:
        n = GtMatcher.summary(gt_results)
        log(f"  標註畫面比對: 共 {len(gt_results)} 張, 抓到 {n.get('hit', 0)}, 有偵測沒判違規 {n.get('detected', 0)}, "
            f"沒偵測到 {n.get('missed', 0)}, 沒跑到 {n.get('skipped', 0)}")

    elapsed = time.time() - start
    counts = {vt: len(c.saved) for vt, c in counters.items()}
    log(f"  完成: {name} (未戴安全帽 {counts['no_helmet']} 台, 機車禁行 {counts['illegal_motorcycle_entry']} 台, "
        f"超速 {counts['speeding']} 台),耗時 {elapsed:.1f} 秒")

    track_rows = []
    for tid, t in sorted(tracks.items()):
        speeds = t["speeds"]
        track_rows.append({
            "track_id": tid, "class_name": t["class_name"],
            "first_sec": round(t["first_sec"], 2), "last_sec": round(t["last_sec"], 2), "frames": t["frames"],
            "avg_speed_kmh": round(sum(speeds) / len(speeds), 1) if speeds else None,
            "max_speed_kmh": round(max(speeds), 1) if speeds else None,
        })

    return {
        "video": name, "fps": fps, "width": w, "height": h,
        "total_frames": total_frames, "processed_frames": processed_idx,
        "elapsed_sec": round(elapsed, 2),
        "avg_ms_per_frame": round(elapsed / max(processed_idx, 1) * 1000, 1),
        "avg_track_ms": round(timing["track_ms"] / max(processed_idx, 1), 1),
        "avg_plate_ms": round(timing["plate_ms"] / timing["plate_calls"], 1) if timing["plate_calls"] else None,
        "avg_ocr_ms": round(timing["ocr_ms"] / timing["ocr_calls"], 1) if timing["ocr_calls"] else None,
        "stopped": stopped, "output_video": output_video,
        "records": records, "tracks": track_rows, "gt_results": gt_results,
    }
