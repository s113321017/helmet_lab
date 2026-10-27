"""
實驗平台網頁介面
------------------------------------------------
啟動:
    conda activate helmet
    python app.py
然後打開瀏覽器 http://127.0.0.1:5000

功能:
    - 選影片、調參數建立實驗 (數值欄位輸入 "0.2,0.3,0.4" 會自動展開成多個實驗做參數掃描)
    - 實驗在背景依序執行 (GPU 一次只跑一個),結果存進 PostgreSQL (DATABASE_URL)
    - 在實驗頁面標註每筆違規 正確/誤報/重複、填正確車牌 → 自動算 precision / recall / 車牌準確率
    - 影片頁面填每支影片的實際違規數 (真值) → 算 recall; 暫停影片框出違規畫面 → 檢查實驗有沒有抓到
    - 比較多個實驗的參數差異與指標,匯出 CSV 做圖表

各頁面的程式在 web/ 資料夾 (說明見 web/__init__.py)
"""

import sys

from core import db
from core.config import env
from web import app
from web.jobs import start_worker

if __name__ == "__main__":
    try:
        import torch, ultralytics  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"缺少套件 ({e})。目前的 Python: {sys.executable}\n"
                         "請先執行 conda activate helmet,或直接雙擊 run.bat")
    db.init_db()
    start_worker()
    host, port = env("HOST", "127.0.0.1"), int(env("PORT", "5000"))
    print(f"實驗平台已啟動: http://{host}:{port}")
    # use_reloader=False: 避免 debug 模式啟動兩個背景執行緒同時搶 GPU
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
