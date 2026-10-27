"""
違規類型定義 + 違規確認計數 + 違規車輛裁切
"""

VIOLATION_TYPES = {
    "no_helmet": "未戴安全帽",
    "illegal_motorcycle_entry": "機車禁行",
    "speeding": "超速",
}
FILE_TAG = {"no_helmet": "nohelmet", "illegal_motorcycle_entry": "restricted", "speeding": "speeding"}
# 違規車輛截圖要從偵測框往外擴多少 (左右, 往上, 往下),都是框寬/框高的倍數:
# 沒戴安全帽的框只框到頭,車牌在頭下面好幾個頭高的位置 (車尾燈下方),所以主要往下擴
CROP_PAD = {"no_helmet": (1.5, 0.5, 5.0), "illegal_motorcycle_entry": (0.3, 0.3, 0.3), "speeding": (0.3, 0.3, 0.3)}


class ViolationCounter:
    """記錄每台車連續違規幾幀、哪些車已經存過截圖 (每台車、每種違規只存一次)"""

    def __init__(self, confirm_frames):
        self.confirm_frames = confirm_frames
        self.counts = {}    # track_id -> 連續違規幀數
        self.saved = set()  # 已存過截圖的 track_id

    def hit(self, tid):
        self.counts[tid] = self.counts.get(tid, 0) + 1

    def miss(self, tid):
        self.counts[tid] = max(0, self.counts.get(tid, 0) - 1)

    def confirmed(self, tid):
        """連續違規幀數夠了而且還沒存過"""
        return self.counts.get(tid, 0) >= self.confirm_frames and tid not in self.saved


def crop_vehicle(frame, box, vtype):
    """依違規類型把偵測框往外擴, 裁出整台車 (要包含車牌)"""
    x1, y1, x2, y2 = box
    pad_w, pad_up, pad_down = CROP_PAD[vtype]
    pad_x = int((x2 - x1) * pad_w)
    vx1, vy1 = max(x1 - pad_x, 0), max(y1 - int((y2 - y1) * pad_up), 0)
    vx2, vy2 = min(x2 + pad_x, frame.shape[1]), min(y2 + int((y2 - y1) * pad_down), frame.shape[0])
    return frame[vy1:vy2, vx1:vx2]
