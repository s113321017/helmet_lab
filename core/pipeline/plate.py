"""
車牌定位 (YOLO) + 車牌文字辨識 (fast-plate-ocr)
"""

import os
import time
import tempfile

import cv2

from .plate_format import fix_plate_format

PLATE_ZOOM_SIZE = (400, 150)  # 車牌放大輸出的固定尺寸(寬,高)

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
