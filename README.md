# Helmet Lab：機車違規偵測實驗平台

用 YOLOv8 和 ByteTrack 偵測交通影片裡的機車違規，並自動截下車牌、用 OCR 讀出車牌號碼。
每次實驗的參數、偵測結果和人工標註都存在 PostgreSQL，可以在網頁上比較不同參數或不同模型的 precision、recall 與車牌準確率。

## 功能

- **偵測三種違規**：未戴安全帽、機車禁行、超速（以相機標定的 homography 估算車速）
- **車牌辨識**：用車牌偵測模型裁出車牌，再交給 [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) 讀出文字
- **網頁實驗平台**（`app.py`）
  - 選影片、調參數建立實驗。數值欄位輸入 `0.2,0.3,0.4` 會自動展開成參數掃描
  - 實驗排隊依序在背景執行（GPU 一次只跑一個）
  - 逐筆標註違規（正確 / 誤報 / 重複）和正確車牌，系統自動計算指標
  - 比較多個實驗，結果可匯出成 CSV
  - 相機標定頁面：在畫面上點選路面參考點，產生 homography 矩陣
- **命令列批次處理**（`main.py`）：一次處理 `test_videos/` 裡的所有影片，輸出截圖和 CSV

## 專案結構

```
helmet/
├── app.py                  # 網頁實驗平台 (Flask) 入口
├── main.py                 # 命令列批次處理入口
├── run.bat                 # Windows 一鍵啟動 (讀 .env → 啟動 PostgreSQL → 開網頁)
├── core/
│   ├── config.py           # 專案路徑、讀取 .env
│   ├── pipeline/           # 偵測 / 追蹤 / 車牌 OCR / 測速核心
│   │   ├── params.py       #   所有可調整的參數
│   │   ├── helmet.py       #   安全帽 / 機車偵測 + 追蹤
│   │   ├── violations.py   #   違規類型、確認計數、違規車輛裁切
│   │   ├── calibration.py  #   相機標定 (homography、測速區)
│   │   ├── speed.py        #   測速
│   │   ├── plate.py        #   車牌定位 + OCR
│   │   ├── plate_format.py #   台灣車牌格式修正
│   │   ├── video_output.py #   輸出標註影片、剪超速片段
│   │   ├── gt_match.py     #   標註畫面 (img 真值) 比對
│   │   └── processor.py    #   process_video(): 串起整個流程
│   └── db.py               # PostgreSQL 資料表與存取
├── web/                    # 網頁各頁面的路由
│   ├── common.py           #   共用工具 (影片讀取、JSON 檔、CSV、選取實驗)
│   ├── jobs.py             #   背景執行實驗
│   ├── settings.py         #   模型 / 標定矩陣路徑設定、選檔視窗
│   ├── experiments.py      #   建立 / 檢視 / 標註實驗
│   ├── compare.py          #   比較實驗、CSV 匯出
│   ├── videos.py           #   影片與真值、標註違規畫面
│   └── calibration.py      #   相機標定
├── templates/              # 網頁 HTML 模板
├── static/                 # CSS
├── configs/
│   └── custom_bytetrack.yaml   # 自訂 ByteTrack 參數
├── requirements.txt
└── .env.example            # 環境變數範本
```

以下資料夾不會上傳到 GitHub，需要自己準備：

| 資料夾 | 內容 |
|---|---|
| `runs/` | YOLO 訓練結果與權重（`runs/detect/train_v2/weights/best.pt`） |
| `weights/` | 其他權重，例如車牌偵測模型 `license_plate_detector.pt` |
| `test_videos/` | 要測試的影片 |
| `data/` | 實驗輸出（截圖、影片）、相機標定結果 |
| `configs/homography_matrix.json` | 預設相機標定矩陣（每台相機不同，請在網頁「相機標定」頁重新產生） |
| `AI city Challenge.v1i.yolov8/` | 訓練用資料集（Roboflow 匯出，YOLOv8 格式） |

## 安裝

### 1. Python 環境

```bash
conda create -n helmet python=3.10
conda activate helmet

# PyTorch 依照自己的 CUDA 版本安裝, 例如 CUDA 12.1:
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### 2. PostgreSQL

安裝 PostgreSQL 後，建立資料庫：

```sql
CREATE DATABASE helmet_lab;
```

資料表會在第一次啟動 `app.py` 時自動建立。

### 3. 環境變數

```bash
cp .env.example .env
```

打開 `.env` 填入自己的設定：

| 變數 | 說明 |
|---|---|
| `DATABASE_URL` | PostgreSQL 連線字串（必填） |
| `HELMET_MODEL_PATH` | 安全帽模型權重，留空則使用 `runs/detect/train_v2/weights/best.pt` |
| `PLATE_MODEL_PATH` | 車牌偵測模型權重，留空則使用 `weights/license_plate_detector.pt` |
| `HOST` / `PORT` | 網頁伺服器位址，預設 `127.0.0.1:5000` |
| `FLASK_SECRET_KEY` | Flask session 金鑰（必填，用 `python -c "import secrets; print(secrets.token_hex(32))"` 產生） |
| `PYTHON_EXE` / `PGBIN` / `PGDATA` | 只有 `run.bat` 會用到 |

## 使用方式

### 網頁實驗平台

```bash
conda activate helmet
python app.py
```

啟動後打開 <http://127.0.0.1:5000>。

在 Windows 上也可以直接雙擊 `run.bat`，它會先啟動 PostgreSQL，再開啟網頁。

### 命令列批次處理

```bash
python main.py
```

程式會處理 `test_videos/*.mp4`，結果輸出到 `output_results/`：

- `violations/`：違規車輛截圖、車牌放大截圖
- `violations.csv`：違規總表，包含違規類型、車牌文字、時間、信心值和車速
- `detected_*.mp4`：畫好偵測框的影片

### 訓練模型

資料集的類別：`DHelmet`、`DNoHelmet`、`motorbike`。

```python
from ultralytics import YOLO
YOLO("yolov8s.pt").train(data="AI city Challenge.v1i.yolov8/data.yaml", epochs=30, imgsz=960, batch=8, name="train_v2")
```

## 資料庫

依 ER Model v8 設計：

| 資料表 | 內容 |
|---|---|
| `batch` | 批次（一次送出多個實驗） |
| `model` | 安全帽模型版本（權重路徑） |
| `video` | 影片檔名與 fps |
| `type` | 違規類型（`no_helmet` / `illegal_motorcycle_entry` / `speeding`） |
| `run` | 每一次實驗（一支影片搭配一組參數）的參數、狀態、計時，屬於 video / model / batch |
| `track` | run 裡每台車的出現時間、幀數、平均與最高車速（`tid` 每個 run 從 1 開始，`(run_id, tid)` 唯一） |
| `record` | 偵測到的違規，掛在 track 底下，和人工標註（`gt_label`：tp / fp / dup，`gt_plate`：正確車牌） |
| `vt` | 每支影片每種違規的實際數量（人工計數的真值），用來計算 recall |
| `img` | 從影片切出的單人裁切圖與人工真值 `gt_ans` |
| `ri` | run 對 img 的預測；用複合外鍵確保 run 和 img 屬於同一支影片 |

舊版的 `experiments` / `violations` / `tracks` / `video_gt` 資料表還在的話，啟動時會自動把資料搬進新表（舊表保留不刪，實驗編號不變）。

