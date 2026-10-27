"""
相機標定: 讀取 homography 矩陣、測速區, 像素座標 → 真實世界座標 (公尺)
"""

import os
import json

import cv2
import numpy as np


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
