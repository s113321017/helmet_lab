"""
專案路徑 + 讀取 .env 設定 (不需要額外套件)
已經存在的系統環境變數優先, .env 只補上沒設定的值
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 專案根目錄
CONFIG_DIR = os.path.join(BASE_DIR, "configs")


def load_env(path=os.path.join(BASE_DIR, ".env")):
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()


def env(key, default=None):
    """讀環境變數, 空字串視為沒設定"""
    return os.environ.get(key) or default
