"""
台灣車牌格式規則: LLL-DDDD
前3碼:英文字母,排除 I、O(容易跟數字1、0搞混)
後4碼:阿拉伯數字,排除 4
"""

LETTER_CONFUSION = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}  # 數字→形狀相似的字母
DIGIT_CONFUSION = {"O": "0", "I": "1", "Z": "2", "S": "5", "G": "6", "B": "8", "Q": "0", "D": "0"}  # 字母→形狀相似的數字
INVALID_LETTERS = {"I", "O"}
INVALID_DIGITS = {"4"}


def fix_plate_format(raw_text):
    """
    依台灣車牌格式 LLL-DDDD 修正OCR結果。
    回傳 (修正後文字, 是否完全符合格式沒有警告, 警告訊息list)
    """
    if not raw_text:
        return None, False, ["空字串"]

    cleaned = "".join(ch for ch in raw_text.upper() if ch.isalnum())

    if len(cleaned) != 7:
        return raw_text, False, [f"長度不是7碼(目前{len(cleaned)}碼),可能裁切或辨識有缺漏"]

    warnings = []
    chars = list(cleaned)

    for i in range(3):  # 前3碼應該是字母
        c = chars[i]
        if c.isdigit():
            guess = LETTER_CONFUSION.get(c)
            if guess and guess not in INVALID_LETTERS:
                chars[i] = guess
            else:
                warnings.append(f"第{i+1}碼讀到數字'{c}',無法安全對應到允許的字母,請人工確認")
        elif c in INVALID_LETTERS:
            warnings.append(f"第{i+1}碼讀到排除字母'{c}',請人工確認")

    for i in range(3, 7):  # 後4碼應該是數字
        c = chars[i]
        if c.isalpha():
            guess = DIGIT_CONFUSION.get(c)
            if guess and guess not in INVALID_DIGITS:
                chars[i] = guess
            else:
                warnings.append(f"第{i+1}碼讀到字母'{c}',無法安全對應到允許的數字,請人工確認")
        elif c in INVALID_DIGITS:
            warnings.append(f"第{i+1}碼讀到排除數字'{c}',實際車牌不會出現,可能誤判")

    fixed = "".join(chars[:3]) + "-" + "".join(chars[3:])
    is_clean = (len(warnings) == 0)
    return fixed, is_clean, warnings


def plate_status(is_clean, confidence):
    """車牌格式不合規定→🔺;信心值低於0.5→🟡"""
    if not is_clean:
        return "🔺 格式異常"
    if confidence is not None and confidence < 0.5:
        return "🟡 信心偏低"
    return "✅ 正常"
