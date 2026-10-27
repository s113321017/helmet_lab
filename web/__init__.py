"""
實驗平台網頁介面 (Flask)。啟動入口是專案根目錄的 app.py, 各頁面的路由分在這個資料夾:
    common.py        共用的路徑、工具函式 (影片讀取、JSON 檔、CSV 匯出、選取實驗)
    jobs.py          背景執行實驗 (GPU 一次只跑一個)
    settings.py      模型 / 標定矩陣路徑設定、本機選檔視窗
    experiments.py   建立 / 檢視 / 停止 / 重跑 / 刪除實驗、違規標註
    compare.py       比較多個實驗、CSV 匯出
    videos.py        影片與真值、標註違規畫面 (img)
    calibration.py   相機標定
"""

from flask import Flask

from core import db
from core.config import env
from core.pipeline import BASE_DIR, VIOLATION_TYPES

from .common import STATUS_TEXT

# root_path 設成專案根目錄, templates/ 和 static/ 照舊放在根目錄
app = Flask(__name__, root_path=BASE_DIR)
app.secret_key = env("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("找不到 FLASK_SECRET_KEY, 請在 .env 設定一串隨機字串, 例如用 "
                       "python -c \"import secrets; print(secrets.token_hex(32))\" 產生")
app.jinja_env.globals.update(VIOLATION_TYPES=VIOLATION_TYPES, STATUS_TEXT=STATUS_TEXT, LABELS=db.LABELS)


@app.template_filter("fmt")
def fmt(v, digits=3):
    if v is None:
        return "–"
    return round(v, digits) if isinstance(v, float) else v


@app.template_filter("pct")
def pct(v):
    return "–" if v is None else f"{v * 100:.1f}%"


# 載入各頁面的路由 (放在最後, 這些模組會 import 上面的 app)
from . import settings, calibration, experiments, compare, videos  # noqa: E402,F401
