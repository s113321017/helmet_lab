"""
測速: 在測速區內記錄每台車的真實世界位置, 用最近幾筆位置算平均車速
"""

from collections import deque

from .calibration import in_zone, pixel_to_world


def compute_speed_kmh(position_history):
    """
    用位置歷史(deque裡存 (time_sec, world_x, world_y))算出平均速度(km/h)。
    至少要有2筆記錄才能算,資料不夠回傳None。
    """
    if len(position_history) < 2:
        return None
    t0, x0, y0 = position_history[0]
    t1, x1, y1 = position_history[-1]
    dt = t1 - t0
    if dt <= 0:
        return None
    dist_m = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return dist_m / dt * 3.6   # m/s 轉 km/h


class SpeedEstimator:
    def __init__(self, matrix, zone, history_len):
        self.matrix = matrix
        self.zone = zone                # None = 整個畫面都測速
        self.history_len = history_len
        self.history = {}               # track_id -> deque[(time_sec, world_x, world_y)],只存測速區內的位置

    def update(self, tid, box, time_sec):
        """
        記錄這台車這一幀的位置。回傳 (是否在測速區內, 車速 km/h 或 None)
        """
        x1, y1, x2, y2 = box
        # 用車輛框「底邊中點」當代表點,因為這是車輛跟路面實際接觸的位置,
        # homography矩陣是針對路面這個平面算的,框中心點(浮在半空中)代入會不準
        bx, by = (x1 + x2) / 2, y2
        if not in_zone(self.zone, bx, by):
            # 離開測速區: 清掉位置紀錄,下次再進來重新算,不跟區外的位置接在一起
            self.history.pop(tid, None)
            return False, None
        wx, wy = pixel_to_world(self.matrix, bx, by)
        hist = self.history.setdefault(tid, deque(maxlen=self.history_len))
        hist.append((time_sec, wx, wy))
        return True, compute_speed_kmh(hist)
