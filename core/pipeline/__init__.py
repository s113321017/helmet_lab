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

檔案:
    params.py        所有可調整的參數 (DEFAULT_PARAMS) + 網頁表單說明
    helmet.py        安全帽 / 機車偵測 + 追蹤
    violations.py    違規類型、確認計數、違規車輛裁切
    calibration.py   相機標定 (homography 矩陣、測速區、像素 → 公尺)
    speed.py         測速
    plate.py         車牌定位 + OCR
    plate_format.py  台灣車牌格式修正
    video_output.py  輸出標註影片
    processor.py     process_video(): 把上面串起來處理一支影片
"""

from ..config import BASE_DIR
from .params import DEFAULT_PARAMS, PARAM_SPECS, normalize_params
from .violations import VIOLATION_TYPES
from .plate_format import plate_status
from .processor import process_video
