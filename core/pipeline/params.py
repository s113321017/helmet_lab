"""
所有可調整的參數 + 網頁表單的參數說明
"""

import os

from ..config import BASE_DIR, CONFIG_DIR, env

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
    "clip_pad_sec": 1.0,           # 超速影片片段: 經過測速區的時段前後各多留幾秒
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
    ("clip_pad_sec", "超速片段前後秒數", "float", "違規規則", "超速影片片段在經過測速區的時段前後多留幾秒"),
    ("max_seconds", "只跑前幾秒", "float", "執行", "0 = 整支影片"),
    ("save_video", "輸出標註影片", "bool", "執行", "畫框線 + ID + 車速"),
]


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
