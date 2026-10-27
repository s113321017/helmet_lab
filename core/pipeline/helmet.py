"""
安全帽 / 機車偵測 + 追蹤 (YOLO + ByteTrack)
單一次 track() 同時追蹤「沒戴安全帽」跟「機車」兩個類別
"""

import os

from ..config import CONFIG_DIR


def resolve_tracker(name):
    """自訂 tracker 設定放在 configs/, 找不到就交給 ultralytics 用內建的 (例如 bytetrack.yaml)"""
    custom = os.path.join(CONFIG_DIR, name)
    return custom if os.path.isfile(custom) else name


class HelmetDetector:
    def __init__(self, p):
        from ultralytics import YOLO

        # 安全帽模型每支影片重新建立,確保追蹤器狀態 (track id) 不會延續到下一支影片
        self.model = YOLO(p["helmet_model_path"])
        self.tracker = resolve_tracker(p["tracker"])
        self.conf, self.imgsz = p["conf"], p["imgsz"]

        self.class_names = self.model.names
        self.no_helmet_id = self._class_id(p["no_helmet_class"])
        self.motorbike_id = self._class_id(p["motorbike_class"])
        if self.no_helmet_id is None:
            raise ValueError(f"找不到類別名稱 '{p['no_helmet_class']}',模型類別: {self.class_names}")
        self.track_classes = [self.no_helmet_id] + ([self.motorbike_id] if self.motorbike_id is not None else [])

    def _class_id(self, name):
        return next((cid for cid, n in self.class_names.items() if n == name), None)

    def track(self, frame):
        """追蹤一幀, 回傳 [(box[x1,y1,x2,y2], track_id, class_id, 信心值), ...] (沒有 track id 的框不回傳)"""
        result = self.model.track(
            frame, classes=self.track_classes, conf=self.conf, imgsz=self.imgsz,
            tracker=self.tracker, persist=True, verbose=False,
        )[0]
        if result.boxes is None or result.boxes.id is None:
            return []
        return [([int(v) for v in box], int(tid), int(cls_id), det_conf)
                for box, tid, cls_id, det_conf in zip(
                    result.boxes.xyxy.tolist(), result.boxes.id.tolist(),
                    result.boxes.cls.tolist(), result.boxes.conf.tolist())]
