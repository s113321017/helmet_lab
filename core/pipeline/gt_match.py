"""
人工標註畫面 (img 真值) 比對:
    在「影片與真值」頁暫停影片、框出違規的人/車存成真值 (時間 + 座標 + 違規類型)。
    實驗執行時, 在標註時間前後幾幀找跟標註框重疊的偵測結果, 跑完再看那台車有沒有被判定成同一種違規:
        hit       抓到: 有偵測到, 而且那台車被判定成這種違規
        detected  有偵測到東西, 但沒判定成這種違規 (類別不對 / 連續幀數不夠) → 漏報
        missed    標註的位置附近什麼都沒偵測到 → 漏報
        skipped   實驗沒跑到這個時間 (只跑前幾秒 / 中途停止), 不算
"""

IOU_MIN = 0.3   # 偵測框中心不在標註框內時, IoU 至少要這麼多才算同一個目標


def _overlap(box, gt):
    """回傳 (偵測框中心是否在標註框內, IoU)"""
    x1, y1, x2, y2 = box
    g1, h1, g2, h2 = gt
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    inside = g1 <= cx <= g2 and h1 <= cy <= h2
    inter = max(0, min(x2, g2) - max(x1, g1)) * max(0, min(y2, h2) - max(y1, h1))
    union = (x2 - x1) * (y2 - y1) + (g2 - g1) * (h2 - h1) - inter
    return inside, inter / union if union > 0 else 0.0


class GtMatcher:
    def __init__(self, gt_imgs, window, relevant_class):
        """
        gt_imgs        : [{"iid", "frame_idx", "bbox": [x1, y1, x2, y2], "gt_ans": 違規類型}, ...]
        window         : 標註時間前後幾幀內的偵測都拿來比
        relevant_class : {違規類型: 應該偵測到的類別 id} (未戴安全帽 → DNoHelmet, 其他 → motorbike)
        """
        self.gt_imgs = gt_imgs
        self.window = window
        self.relevant_class = relevant_class
        self.best = {}          # iid -> (排序用的 key, 偵測結果)
        self.last_frame = 0

    def update(self, raw_frame_idx, detections, time_sec):
        self.last_frame = raw_frame_idx
        for g in self.gt_imgs:
            dist = abs(raw_frame_idx - g["frame_idx"])
            if dist > self.window:
                continue
            for box, tid, cls_id, det_conf in detections:
                inside, iou = _overlap(box, g["bbox"])
                if not inside and iou < IOU_MIN:
                    continue
                # 優先: 類別對 > 中心在框內 > IoU 大 > 時間近
                key = (cls_id == self.relevant_class.get(g["gt_ans"]), inside, iou, -dist)
                if g["iid"] not in self.best or key > self.best[g["iid"]][0]:
                    self.best[g["iid"]] = (key, {"tid": tid, "class_id": cls_id, "conf": round(det_conf, 3),
                                                 "iou": round(iou, 3), "time_sec": round(time_sec, 2),
                                                 "box": list(box)})

    def results(self, class_names, saved):
        """saved = {違規類型: 被判定違規 (有存紀錄) 的 track id 集合}; 回傳 [{"iid", "pred"}, ...]"""
        out = []
        for g in self.gt_imgs:
            if g["frame_idx"] - self.window > self.last_frame:
                pred = {"status": "skipped"}
            elif g["iid"] not in self.best:
                pred = {"status": "missed"}
            else:
                d = dict(self.best[g["iid"]][1])
                d["class"] = class_names[d.pop("class_id")]
                hit = d["tid"] in saved.get(g["gt_ans"], ())
                pred = {"status": "hit" if hit else "detected", **d}
            out.append({"iid": g["iid"], "pred": pred})
        return out

    @staticmethod
    def summary(results):
        """{status: 數量}"""
        counts = {}
        for r in results:
            counts[r["pred"]["status"]] = counts.get(r["pred"]["status"], 0) + 1
        return counts
