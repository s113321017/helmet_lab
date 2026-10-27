"""
檔案路徑設定 (模型、標定矩陣): 改了就存檔, 重新整理頁面或刪除實驗後不會跳回預設; 本機選檔視窗
"""

import os
import sys
import glob
import subprocess

from flask import request, jsonify, abort

from core import db
from core.pipeline import BASE_DIR, DEFAULT_PARAMS

from . import app
from .common import read_json, write_json

SETTINGS_PATH = os.path.join(db.DATA_DIR, "model_settings.json")
FILE_KEYS = {"helmet_model_path": ".pt", "plate_model_path": ".pt", "homography_path": ".json"}
MULTI_KEYS = {"helmet_model_path"}   # 可以選多個, 每個各建一個實驗 (同一批次比較)
FILE_TYPES = {".pt": ("選擇模型檔 (.pt)", "PyTorch 模型 (*.pt)"), ".json": ("選擇標定矩陣檔 (.json)", "標定矩陣 (*.json)")}


def valid_file(key, path):
    """欄位 key 可以用這個檔案嗎 (副檔名正確而且存在)"""
    return key in FILE_KEYS and isinstance(path, str) and path.lower().endswith(FILE_KEYS[key]) and os.path.isfile(path)


def load_file_settings():
    return {k: v for k, v in read_json(SETTINGS_PATH, {}).items() if valid_file(k, v)}


def save_file_setting(key, path):
    """只存副檔名正確、而且存在的檔案"""
    if not valid_file(key, path):
        return False
    write_json(SETTINGS_PATH, {**load_file_settings(), key: path})
    return True


def model_choices():
    found = glob.glob(os.path.join(BASE_DIR, "runs", "detect", "*", "weights", "*.pt"))
    return sorted(set(found + [DEFAULT_PARAMS["helmet_model_path"], DEFAULT_PARAMS["plate_model_path"]]))


@app.route("/api/file-setting", methods=["POST"])
def file_setting():
    return jsonify({"saved": save_file_setting(request.form.get("key", ""), request.form.get("value", "").strip())})


_PICK_SCRIPT = """
import sys, tkinter as tk
from tkinter import filedialog
start, title, type_name, pattern = sys.argv[1:5]
r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)
print(filedialog.askopenfilename(title=title, initialdir=start, filetypes=[(type_name, pattern)]))
"""


@app.route("/api/pick-file", methods=["POST"])
def pick_file():
    """在本機跳出檔案總管選檔, 只能選指定副檔名 (網頁只跑在 127.0.0.1, 所以可以直接開視窗)"""
    ext = request.form.get("ext", "")
    if ext not in FILE_TYPES:
        abort(400)
    title, type_name = FILE_TYPES[ext]
    current = request.form.get("current", "")
    start = os.path.dirname(current) if os.path.isfile(current) else BASE_DIR
    out = subprocess.run([sys.executable, "-c", _PICK_SCRIPT, start, title, type_name, "*" + ext],
                         capture_output=True, text=True, encoding="utf-8",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout.strip()
    path = os.path.normpath(out) if out else ""
    return jsonify({"path": path if path.lower().endswith(ext) else ""})
