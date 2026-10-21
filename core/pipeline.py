"""
安全帽偵測 + 機車禁行判斷 + 車牌定位 + OCR + 速度估算 核心 (main.py 命令列 與 app.py 網頁介面 共用)
------------------------------------------------
邏輯:
    每一幀畫面丟給安全帽模型,單一次 track() 同時追蹤「沒戴安全帽」跟「機車」兩個類別:
    1. 「沒戴安全帽」連續幾幀確認 → 觸發「未戴安全帽」事件
    2. 「機車」如果這段路是機車禁行,連續幾幀偵測到 → 觸發「機車禁行」事件
    3. 「機車」在測速區 (標定點圍成的路面) 內套用相機標定矩陣估算真實車速 → 連續幾幀超速 → 觸發「超速」事件
    事件觸發時存違規車輛截圖 + 車牌放大截圖 + OCR讀出車牌文字
    (每台車、每種違規只存一次;沒找到車牌就下一幀再試)

所有可調整的參數都在 DEFAULT_PARAMS,呼叫 process_video() 時傳入要覆蓋的值即可。
"""

import os
import json
import time
import tempfile
from collections import deque

import cv2
import numpy as np


from .config import BASE_DIR, CONFIG_DIR, env

# ===== 預設參數 (網頁表單、命令列都從這裡取預設值; 模型路徑可在 .env 設定) =====
DEFAULT_PARAMS = {
    # 模型
    "helmet_model_path": env("HELMET_MODEL_PATH",
                             os.path.join(BASE_DIR, "runs", "detect", "train_v2", "weights", "best.pt")),
    "plate_model_path": env("PLATE_MODEL_PATH", os.path.join(BASE_DIR, "weights", "license_plate_detector.pt")),
    "no_helmet_class": "DNoHelmet",
    "motorbike_class": "motorbike",
    "ocr_model": "cct-s-v2-global-model",
    # 偵測 / 追蹤
    "conf": 0.2,
    "imgsz": 960,
    "vid_stride": 2,
    "confirm_frames": 20,          # 連續幾幀都判斷違規,才算確認觸發事件
    "tracker": "bytetrack.yaml",   # motorbike 追蹤常斷可改 custom_bytetrack.yaml
    "plate_conf": 0.25,
    # 違規規則
    "motorcycle_restricted": False,  # 這段路是不是機車禁行
    "speed_enabled": True,
    "homography_path": os.path.join(CONFIG_DIR, "homography_matrix.json"),
    "speed_limit_kmh": 10.0,
    "speed_history_len": 10,       # 用最近幾筆位置做速度平滑,避免單幀跳動
    "speed_zone_enabled": True,    # 只在測速區 (標定點圍成的範圍) 內測速,區外 homography 外插誤差大
    "speed_confirm_frames": 5,     # 測速區內連續幾幀超速才觸發 (測速區不長,車子待不了 confirm_frames 那麼多幀)
    # 執行
    "max_seconds": 0.0,            # 只跑影片前幾秒 (0 = 整支),快速測試用
    "save_video": False,           # 輸出畫好框線的影片
}

# 網頁表單用的參數說明: (key, 顯示名稱, 型別, 分組, 說明)
# 型別 int / float 的欄位在網頁可輸入逗號分隔多個值做參數掃描
PARAM_SPECS = [
    ("helmet_model_path", "安全帽模型", "str", "模型", "YOLO 權重路徑 (.pt)"),
    ("plate_model_path", "車牌模型", "str", "模型", "車牌偵測 YOLO 權重路徑"),
    ("ocr_model", "OCR 模型", "str", "模型", "fast-plate-ocr 模型名稱"),
    ("conf", "偵測信心門檻", "float", "偵測 / 追蹤", "YOLO conf"),
    ("imgsz", "輸入尺寸", "int", "偵測 / 追蹤", "YOLO imgsz"),
    ("vid_stride", "跳幀", "int", "偵測 / 追蹤", "每 N 幀處理 1 幀"),
    ("confirm_frames", "確認幀數", "int", "偵測 / 追蹤", "連續幾幀違規才觸發事件"),
    ("plate_conf", "車牌偵測信心門檻", "float", "偵測 / 追蹤", ""),
    ("motorcycle_restricted", "機車禁行路段", "bool", "違規規則", "勾選 = 偵測到機車即違規"),
    ("speed_enabled", "啟用測速", "bool", "違規規則", "需要相機標定矩陣"),
    ("homography_path", "標定矩陣檔", "str", "違規規則", "「相機標定」頁產生的 json"),
    ("speed_limit_kmh", "速限 (km/h)", "float", "違規規則", ""),
    ("speed_history_len", "速度平滑筆數", "int", "違規規則", "用最近幾筆位置算速度"),
    ("speed_zone_enabled", "只在測速區內測速", "bool", "違規規則", "測速區 = 標定點圍成的路面範圍"),
    ("speed_confirm_frames", "超速確認幀數", "int", "違規規則", "測速區內連續幾幀超速才觸發事件"),
    ("max_seconds", "只跑前幾秒", "float", "執行", "0 = 整支影片"),
    ("save_video", "輸出標註影片", "bool", "執行", "畫框線 + ID + 車速"),
]

VIOLATION_TYPES = {
    "no_helmet": "未戴安全帽",
    "illegal_motorcycle_entry": "機車禁行",
    "speeding": "超速",
}
_FILE_TAG = {"no_helmet": "nohelmet", "illegal_motorcycle_entry": "restricted", "speeding": "speeding"}
# 違規車輛截圖要從偵測框往外擴多少 (左右, 往上, 往下),都是框寬/框高的倍數:
# 沒戴安全帽的框只框到頭,車牌在頭下面好幾個頭高的位置 (車尾燈下方),所以主要往下擴
_CROP_PAD = {"no_helmet": (1.5, 0.5, 5.0), "illegal_motorcycle_entry": (0.3, 0.3, 0.3), "speeding": (0.3, 0.3, 0.3)}
PLATE_ZOOM_SIZE = (400, 150)  # 車牌放大輸出的固定尺寸(寬,高)


def normalize_params(params):
    """補上預設值並把型別轉正確 (網頁表單送進來的都是字串)"""
    p = dict(DEFAULT_PARAMS)
    for k, v in (params or {}).items():
        if k not in DEFAULT_PARAMS:
            continue
        default = DEFAULT_PARAMS[k]
        if isinstance(default, bool):
            p[k] = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")
        elif isinstance(default, int):
            p[k] = int(float(v))
        elif isinstance(default, float):
            p[k] = float(v)
        else:
            p[k] = str(v)
    return p


# ===== 台灣車牌格式規則: LLL-DDDD =====
# 前3碼:英文字母,排除 I、O(容易跟數字1、0搞混)
# 後4碼:阿拉伯數字,排除 4
LETTER_CONFUSION = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}  # 數字→形狀相似的字母
DIGIT_CONFUSION = {"O": "0", "I": "1", "Z": "2", "S": "5", "G": "6", "B": "8", "Q": "0", "D": "0"}  # 字母→形狀相似的數字
INVALID_LETTERS = {"I", "O"}
INVALID_DIGITS = {"4"}


def fix_plate_format(raw_text):
    """
    依台灣車牌格式 LLL-DDDD 修正OCR結果。
    回傳 (修正後文字, 是否完全符合格式沒有警告, 警告訊息list)
    """
    if not raw_text:
        return None, False, ["空字串"]

    cleaned = "".join(ch for ch in raw_text.upper() if ch.isalnum())

    if len(cleaned) != 7:
        return raw_text, False, [f"長度不是7碼(目前{len(cleaned)}碼),可能裁切或辨識有缺漏"]

    warnings = []
    chars = list(cleaned)

    for i in range(3):  # 前3碼應該是字母
        c = chars[i]
        if c.isdigit():
            guess = LETTER_CONFUSION.get(c)
            if guess and guess not in INVALID_LETTERS:
                chars[i] = guess
            else:
                warnings.append(f"第{i+1}碼讀到數字'{c}',無法安全對應到允許的字母,請人工確認")
        elif c in INVALID_LETTERS:
            warnings.append(f"第{i+1}碼讀到排除字母'{c}',請人工確認")

    for i in range(3, 7):  # 後4碼應該是數字
        c = chars[i]
        if c.isalpha():
            guess = DIGIT_CONFUSION.get(c)
            if guess and guess not in INVALID_DIGITS:
                chars[i] = guess
            else:
                warnings.append(f"第{i+1}碼讀到字母'{c}',無法安全對應到允許的數字,請人工確認")
        elif c in INVALID_DIGITS:
            warnings.append(f"第{i+1}碼讀到排除數字'{c}',實際車牌不會出現,可能誤判")

    fixed = "".join(chars[:3]) + "-" + "".join(chars[3:])
    is_clean = (len(warnings) == 0)
    return fixed, is_clean, warnings


def plate_status(is_clean, confidence):
    """車牌格式不合規定→🔺;信心值低於0.5→🟡"""
    if not is_clean:
        return "🔺 格式異常"
    if confidence is not None and confidence < 0.5:
        return "🟡 信心偏低"
    return "✅ 正常"


# ===== 模型快取 (車牌模型 / OCR 沒有狀態,可以跨實驗共用) =====
_ocr_models = {}
_plate_models = {}


def get_ocr_model(name):
    if name not in _ocr_models:
        from fast_plate_ocr import LicensePlateRecognizer
        print(f"正在載入車牌 OCR 模型 (fast-plate-ocr: {name})...")
        _ocr_models[name] = LicensePlateRecognizer(name)
    return _ocr_models[name]


def get_plate_model(path):
    if path not in _plate_models:
        from ultralytics import YOLO
        _plate_models[path] = YOLO(path)
    return _plate_models[path]


def read_plate_text(plate_image_bgr, ocr_model_name):
    """對車牌小圖做文字辨識,回傳(文字, 信心值),辨識失敗回傳(None, None)"""
    if plate_image_bgr.size == 0:
        return None, None
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, plate_image_bgr)
        preds = get_ocr_model(ocr_model_name).run(tmp_path, return_confidence=True)
    finally:
        os.remove(tmp_path)

    if not preds:
        return None, None
    pred = preds[0]

    # fast_plate_ocr 回傳 PlatePrediction 物件: .plate=文字, .char_probs=每字元信心值陣列
    text = getattr(pred, "plate", None)
    conf = None
    char_probs = getattr(pred, "char_probs", None)
    if text and char_probs is not None:
        relevant = char_probs[:len(text)]
        if len(relevant) > 0:
            conf = float(sum(relevant) / len(relevant))

    if text:
        text = str(text).upper().replace(" ", "").replace("_", "")
    return text, conf


def locate_and_read_plate(plate_model, vehicle_crop, save_path, p, timing, log=print):
    """
    在裁切出來的車輛區域內找車牌、存放大截圖、跑OCR。
    沒找到車牌回傳 None,找到回傳 dict(plate_raw, plate_fixed, plate_conf, is_clean, warnings)
    """
    t0 = time.time()
    pr = plate_model.predict(vehicle_crop, conf=p["plate_conf"], verbose=False)[0]
    timing["plate_ms"] += (time.time() - t0) * 1000
    timing["plate_calls"] += 1

    if len(pr.boxes) == 0:
        log("    ⚠️ 沒找到車牌位置,這次不記錄,下一幀再試")
        return None

    best = int(pr.boxes.conf.argmax())
    px1, py1, px2, py2 = [int(v) for v in pr.boxes.xyxy[best].tolist()]
    plate_crop = vehicle_crop[py1:py2, px1:px2]
    if plate_crop.size == 0:
        log("    ⚠️ 車牌裁切失敗,這次不記錄,下一幀再試")
        return None

    plate_zoom = cv2.resize(plate_crop, PLATE_ZOOM_SIZE, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(save_path, plate_zoom)

    t0 = time.time()
    plate_text, plate_conf = read_plate_text(plate_crop, p["ocr_model"])
    timing["ocr_ms"] += (time.time() - t0) * 1000
    timing["ocr_calls"] += 1

    plate_fixed, is_clean, warnings = fix_plate_format(plate_text)
    conf_display = f"{plate_conf:.3f}" if plate_conf is not None else "-"
    log(f"    車牌原始: {plate_text or '(讀取失敗)'} → 修正後: {plate_fixed or '-'} (信心值 {conf_display})")
    for w in (warnings if not is_clean else []):
        log(f"      ⚠️ {w}")

    return {"plate_raw": plate_text, "plate_fixed": plate_fixed, "plate_conf": plate_conf,
            "is_clean": is_clean, "warnings": warnings}


# ===== 速度估算 =====
def load_homography_matrix(path, frame_size, log=print):
    """
    讀取相機標定檔,回傳 (矩陣, 測速區多邊形),失敗回傳 (None, None)(速度估算會自動停用)。
    測速區 = 標定點在畫面上圍成的範圍 (homography 只在這塊路面附近準,越往外外插誤差越大);
    舊版只存矩陣的檔案沒有標定點,測速區回傳 None (整個畫面都測速)。
    """
    if not os.path.exists(path):
        log(f"⚠️ 找不到相機標定矩陣檔案: {path},速度估算功能停用")
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 舊版 calibrate_camera.py 只存矩陣本身; 網頁標定存成 {"matrix": ..., 標定點, 測試紀錄...}
        matrix = np.array(data["matrix"] if isinstance(data, dict) else data, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"矩陣形狀應該是 3x3, 實際是 {matrix.shape}")
        log(f"已載入相機標定矩陣: {path}")
    except Exception as e:
        log(f"⚠️ 讀取矩陣檔案失敗: {e},速度估算功能停用")
        return None, None

    points = data.get("image_points") if isinstance(data, dict) else None
    if not points or len(points) < 3:
        log("⚠️ 標定檔沒有標定點,無法建立測速區,整個畫面都會測速")
        return matrix, None
    pts = np.array(points, dtype=np.float32)
    calib_size = data.get("image_size")
    if calib_size and tuple(calib_size) != tuple(frame_size):
        # 標定畫面跟影片解析度不同時,標定點等比例換到影片座標 (矩陣本身沒換,這種情況車速本來就不準)
        log(f"⚠️ 標定畫面 {calib_size[0]}x{calib_size[1]} 跟影片 {frame_size[0]}x{frame_size[1]} 解析度不同,"
            "車速可能不準,請用同解析度的畫面重新標定")
        pts *= np.array([frame_size[0] / calib_size[0], frame_size[1] / calib_size[1]], dtype=np.float32)
    zone = cv2.convexHull(pts)   # 標定點不一定照順序點,取凸包當多邊形
    return matrix, zone


def in_zone(zone, px, py):
    """點是否在測速區內 (沒有測速區 = 整個畫面都算)"""
    return zone is None or cv2.pointPolygonTest(zone, (float(px), float(py)), False) >= 0


def pixel_to_world(matrix, px, py):
    """用homography矩陣把像素座標換算成真實世界座標(公尺)"""
    point = np.array([[[px, py]]], dtype=np.float32)
    world = cv2.perspectiveTransform(point, matrix.astype(np.float32))
    return float(world[0][0][0]), float(world[0][0][1])


def compute_speed_kmh(position_history):
    """
    用位置歷史(deque裡存 (time_sec, world_x, world_y))算出平均速度(km/h)。
    至少要有2筆記錄才能算,資料不夠回傳None。
    """
    if len(position_history) < 2:
        return None
    t0, x0, y0 = position_history[0]
    t1, x1, y1 = position_history[-1]
    dt = t1 - t0
    if dt <= 0:
        return None
    dist_m = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return dist_m / dt * 3.6   # m/s 轉 km/h


def _open_video_writer(path, fps, size):
    """優先用 H.264 (瀏覽器可直接播放),這台電腦的 OpenCV 不支援就退回 mp4v"""
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    return None


def process_video(video_path, params, out_dir, on_violation=None, on_progress=None,
                  should_stop=None, log=print):
    """
    處理一支影片。
        params       : 要覆蓋 DEFAULT_PARAMS 的參數
        out_dir      : 截圖存到 out_dir/violations/,標註影片存到 out_dir/
        on_violation : 每確認一筆違規就呼叫 on_violation(record)
        on_progress  : on_progress(進度0~1, 已處理幀數)
        should_stop  : 回傳 True 就提前結束
    回傳摘要 dict (含 records 違規清單、tracks 每台車統計、計時資訊)
    """
    from ultralytics import YOLO

    p = normalize_params(params)
    name = os.path.splitext(os.path.basename(video_path))[0]
    vio_dir = os.path.join(out_dir, "violations")
    os.makedirs(vio_dir, exist_ok=True)

    # 安全帽模型每支影片重新載入,確保追蹤器狀態 (track id) 不會延續到下一支影片
    helmet_model = YOLO(p["helmet_model_path"])
    # 自訂 tracker 設定放在 configs/, 找不到就交給 ultralytics 用內建的 (例如 bytetrack.yaml)
    custom_tracker = os.path.join(CONFIG_DIR, p["tracker"])
    tracker = custom_tracker if os.path.isfile(custom_tracker) else p["tracker"]
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

    homography_matrix, speed_zone = (load_homography_matrix(p["homography_path"], (w, h), log)
                                     if p["speed_enabled"] else (None, None))
    speed_enabled = homography_matrix is not None
    if not p["speed_zone_enabled"]:
        speed_zone = None
    log(f"速度估算: {'啟用' if speed_enabled else '停用'}"
        + (f" (速限 {p['speed_limit_kmh']} km/h, {'只在測速區內' if speed_zone is not None else '整個畫面'})"
           if speed_enabled else ""))

    class_names = helmet_model.names
    no_helmet_id = next((cid for cid, n in class_names.items() if n == p["no_helmet_class"]), None)
    motorbike_id = next((cid for cid, n in class_names.items() if n == p["motorbike_class"]), None)
    if no_helmet_id is None:
        cap.release()
        raise ValueError(f"找不到類別名稱 '{p['no_helmet_class']}',模型類別: {class_names}")
    if (p["motorcycle_restricted"] or speed_enabled) and motorbike_id is None:
        log(f"⚠️ 找不到類別名稱 '{p['motorbike_class']}',機車禁行/速度估算會失效。模型類別: {class_names}")
    track_classes = [no_helmet_id] + ([motorbike_id] if motorbike_id is not None else [])

    writer, out_w, out_h, output_video = None, 0, 0, None
    if p["save_video"]:
        out_w = min(1280, w)
        out_h = int(h * out_w / w)
        output_video = f"detected_{name}.mp4"
        writer = _open_video_writer(os.path.join(out_dir, output_video), fps / p["vid_stride"], (out_w, out_h))
        if writer is None:
            log("⚠️ 無法建立輸出影片,略過影片輸出")
            output_video = None

    confirm_count = {vt: {} for vt in VIOLATION_TYPES}   # 違規類型 -> {track_id: 連續違規幀數}
    saved = {vt: set() for vt in VIOLATION_TYPES}        # 違規類型 -> 已存過截圖的 track_id
    position_history = {}   # track_id -> deque[(time_sec, world_x, world_y)],只存測速區內的位置
    pending_speeding = {}   # 已確認超速、還沒找到車牌的 track_id -> 確認當下車速
    tracks = {}             # track_id -> 統計 (出現時間、幀數、車速)
    records = []
    timing = {"track_ms": 0.0, "plate_ms": 0.0, "ocr_ms": 0.0, "plate_calls": 0, "ocr_calls": 0}

    def try_record(vtype, tid, box, det_conf, frame, time_sec, frame_idx, speed_kmh=None):
        """確認違規後定位車牌並存檔,有找到車牌才算數 (回傳 True)"""
        x1, y1, x2, y2 = box
        pad_w, pad_up, pad_down = _CROP_PAD[vtype]
        pad_x = int((x2 - x1) * pad_w)
        vx1, vy1 = max(x1 - pad_x, 0), max(y1 - int((y2 - y1) * pad_up), 0)
        vx2, vy2 = min(x2 + pad_x, frame.shape[1]), min(y2 + int((y2 - y1) * pad_down), frame.shape[0])
        vehicle_crop = frame[vy1:vy2, vx1:vx2]

        base = f"{name}_{_FILE_TAG[vtype]}_track{tid}"
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
        }
        records.append(rec)
        if on_violation:
            on_violation(rec)
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
        result = helmet_model.track(
            frame, classes=track_classes, conf=p["conf"], imgsz=p["imgsz"],
            tracker=tracker, persist=True, verbose=False,
        )[0]
        timing["track_ms"] += (time.time() - t0) * 1000

        drawn = []   # 要畫在輸出影片上的框 (box, 顏色, 文字)
        if result.boxes is not None and result.boxes.id is not None:
            for box, tid, cls_id, det_conf in zip(
                result.boxes.xyxy.tolist(), result.boxes.id.tolist(),
                result.boxes.cls.tolist(), result.boxes.conf.tolist()
            ):
                box = [int(v) for v in box]
                tid, cls_id = int(tid), int(cls_id)
                x1, y1, x2, y2 = box

                t = tracks.setdefault(tid, {"class_name": class_names[cls_id], "first_sec": time_sec,
                                            "frames": 0, "speeds": []})
                t["last_sec"] = time_sec
                t["frames"] += 1

                if cls_id == no_helmet_id:
                    cnt = confirm_count["no_helmet"]
                    cnt[tid] = cnt.get(tid, 0) + 1
                    if cnt[tid] >= p["confirm_frames"] and tid not in saved["no_helmet"]:
                        log(f"  🚨 track#{tid} 未戴安全帽違規確認,嘗試定位車牌...")
                        if try_record("no_helmet", tid, box, det_conf, frame, time_sec, raw_frame_idx):
                            saved["no_helmet"].add(tid)
                    drawn.append((box, (0, 0, 255), f"#{tid} no helmet"))

                elif cls_id == motorbike_id:
                    label = f"#{tid} motorbike"

                    # --- 機車禁行判斷 ---
                    if p["motorcycle_restricted"]:
                        cnt = confirm_count["illegal_motorcycle_entry"]
                        cnt[tid] = cnt.get(tid, 0) + 1
                        if cnt[tid] >= p["confirm_frames"] and tid not in saved["illegal_motorcycle_entry"]:
                            log(f"  🚨 track#{tid} 機車禁行違規確認,嘗試定位車牌...")
                            if try_record("illegal_motorcycle_entry", tid, box, det_conf,
                                          frame, time_sec, raw_frame_idx):
                                saved["illegal_motorcycle_entry"].add(tid)

                    # --- 速度估算 + 超速判斷 ---
                    if speed_enabled:
                        # 用車輛框「底邊中點」當代表點,因為這是車輛跟路面實際接觸的位置,
                        # homography矩陣是針對路面這個平面算的,框中心點(浮在半空中)代入會不準
                        bx, by = (x1 + x2) / 2, y2
                        if in_zone(speed_zone, bx, by):
                            wx, wy = pixel_to_world(homography_matrix, bx, by)
                            hist = position_history.setdefault(tid, deque(maxlen=p["speed_history_len"]))
                            hist.append((time_sec, wx, wy))
                            speed_kmh = compute_speed_kmh(hist)

                            cnt = confirm_count["speeding"]
                            is_speeding = speed_kmh is not None and speed_kmh > p["speed_limit_kmh"]
                            if is_speeding:
                                cnt[tid] = cnt.get(tid, 0) + 1
                            else:
                                cnt[tid] = max(0, cnt.get(tid, 0) - 1)
                            if speed_kmh is not None:
                                t["speeds"].append(speed_kmh)
                                label += f" {speed_kmh:.1f}km/h"
                            if (is_speeding and cnt[tid] >= p["speed_confirm_frames"]
                                    and tid not in saved["speeding"] and tid not in pending_speeding):
                                log(f"  🚨 track#{tid} 超速違規確認 (約 {speed_kmh:.1f} km/h),嘗試定位車牌...")
                                pending_speeding[tid] = speed_kmh
                        else:
                            # 離開測速區: 清掉位置紀錄,下次再進來重新算,不跟區外的位置接在一起
                            position_history.pop(tid, None)

                        # 已確認超速但還沒拍到車牌: 出了測速區也繼續找車牌 (車速用確認當下的)
                        if tid in pending_speeding:
                            if try_record("speeding", tid, box, det_conf, frame, time_sec,
                                          raw_frame_idx, pending_speeding[tid]):
                                saved["speeding"].add(tid)
                                del pending_speeding[tid]

                    drawn.append((box, (255, 128, 0), label))

        if writer is not None:
            if speed_enabled and speed_zone is not None:
                cv2.polylines(frame, [speed_zone.astype(np.int32)], True, (0, 255, 255), 2)
            for (bx1, by1, bx2, by2), color, text in drawn:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(frame, text, (bx1, max(by1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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

    elapsed = time.time() - start
    counts = {vt: len(s) for vt, s in saved.items()}
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
        "records": records, "tracks": track_rows,
    }
