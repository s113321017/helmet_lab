"""
命令列版本:批次處理 test_videos/ 裡所有影片,結果輸出到 output_results/
(偵測邏輯在 core/pipeline.py,跟網頁實驗平台 app.py 共用;想記錄/比較實驗請用 python app.py)

輸出:
    output_results/detected_xxx.mp4                  → 標註影片 (SAVE_VIDEO=True 時)
    output_results/violations/xxx_..._vehicle.jpg    → 違規車輛截圖
    output_results/violations/xxx_..._plate.jpg      → 車牌放大截圖
    output_results/violations.csv                     → 違規行為、車牌文字、時間標記、信心值、車速總表

注意:pip install 記得也裝 fast-plate-ocr:
    pip install fast-plate-ocr
"""

import os
import csv
import glob

from core.pipeline import BASE_DIR, DEFAULT_PARAMS, process_video, plate_status


# ===== 設定區 (沒寫的參數用 core/pipeline.py 的 DEFAULT_PARAMS; 模型路徑在 .env 設定) =====
PARAMS = {
    **DEFAULT_PARAMS,
    "motorcycle_restricted": False,   # 這支測試影片是不是機車禁行路段
    "speed_limit_kmh": 10,
    "save_video": True,
}

INPUT_DIR = os.path.join(BASE_DIR, "test_videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_results")
CSV_PATH = os.path.join(OUTPUT_DIR, "violations.csv")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    videos = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    if not videos:
        print(f"⚠️ 在 '{INPUT_DIR}' 找不到影片")
        return

    all_records = []
    for video_path in videos:
        print(f"\n處理中: {os.path.basename(video_path)} (機車禁行: {PARAMS['motorcycle_restricted']})")
        summary = process_video(video_path, PARAMS, OUTPUT_DIR)
        all_records.extend(summary["records"])
        print(f"  [計時] 共 {summary['processed_frames']} 幀,平均每幀 {summary['avg_ms_per_frame']} ms")

    print("\n正在寫入 CSV 結果...")
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["video", "time_sec", "violation_type", "track_id",
                             "plate_text_raw", "plate_text_fixed", "confidence", "speed_kmh", "status"])
        for rec in all_records:
            csv_writer.writerow([
                rec["video"], rec["time_sec"], rec["violation_type"], rec["track_id"],
                rec["plate_raw"], rec["plate_fixed"], rec["plate_conf"], rec["speed_kmh"],
                plate_status(rec["is_clean"], rec["plate_conf"]),
            ])

    print(f"\n全部完成,違規截圖在: {os.path.join(OUTPUT_DIR, 'violations')}")
    print(f"違規總表: {CSV_PATH}")


if __name__ == "__main__":
    main()
