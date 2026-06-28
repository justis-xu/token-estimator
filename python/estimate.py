"""
启发式 token 估算器参考实现，与 go/estimator.go 保持同步。

calculate_discount.py 用此模块在 discount=1.0 下计算启发式估算值，
然后与真实 token 数对比得到折扣系数。
字符分类边界、Unicode 范围、各类权重必须与 estimator.go 完全一致。
"""

import math
import unicodedata

CJK_START = 0x4E00
CJK_END   = 0x9FFF

# 默认权重——与 go/estimator.go defaultWeights() 严格对齐。
# 可通过 calculate_weights.py 校准后写入 config.json 的 "weights" 段，
# 由 Go 和 Python 侧共同读取。
DEFAULT_WEIGHTS: dict[str, float] = {
    "default_cjk":     1.5,  # 无词表时 CJK 兜底除数：1/1.5≈0.667 token/字（需非零）
    "latin_divisor":   4.0,  # 拉丁字母：ceil(词长 / 4)
    "hiragana":        1.0,  # 平假名 / 片假名
    "korean":          1.5,  # 韩文音节
    "digit":           0.5,  # 数字（连续段，每位）
    "newline":         0.5,  # 换行符
    "tab":             0.8,  # Tab
    "ascii_space":     0.2,  # ASCII 空格
    "cjk_punctuation": 1.0,  # 中文标点 / 全角符号
    "ascii_punct":     0.7,  # ASCII 标点
    "other":           3.0,  # 其余（emoji、罕见符号）
}


def _is_latin(cp: int) -> bool:
    return (0x61 <= cp <= 0x7A) or (0x41 <= cp <= 0x5A)


def _is_nd_digit(ch: str) -> bool:
    # Unicode Nd 类，与 Go unicode.IsDigit 行为一致
    return unicodedata.category(ch) == "Nd"


def classify_text(text: str) -> str:
    """按 CJK 字符占比返回 'zh'、'en' 或 'mixed'。"""
    if not text:
        return "mixed"
    cjk = sum(1 for c in text if CJK_START <= ord(c) <= CJK_END)
    ratio = cjk / len(text)
    if ratio >= 0.6:
        return "zh"
    if ratio <= 0.1:
        return "en"
    return "mixed"


def _scan(
    table: bytes | None,
    text: str,
    bigrams: dict[str, int] | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[float, float]:
    """返回 (bigram_tokens, heuristic_tokens)，不应用 discount。

    bigram_tokens  — 高频词表精确命中的 token 数。
    heuristic_tokens — 单字表 + 启发式规则累计的 token 数，供 discount 缩放。
    """
    w = DEFAULT_WEIGHTS if weights is None else {**DEFAULT_WEIGHTS, **weights}
    runes = list(text)
    n = len(runes)
    bigram_t = 0.0
    heuristic_t = 0.0

    i = 0
    while i < n:
        ch = runes[i]
        cp = ord(ch)

        # CJK 基本区：优先查高频词表，再查单字表
        if CJK_START <= cp <= CJK_END:
            if bigrams and i + 1 < n:
                next_cp = ord(runes[i + 1])
                if CJK_START <= next_cp <= CJK_END:
                    pair = ch + runes[i + 1]
                    if pair in bigrams:
                        bigram_t += bigrams[pair]
                        i += 2
                        continue
            if table is not None:
                heuristic_t += table[cp - CJK_START]
            else:
                heuristic_t += 1.0 / w["default_cjk"]  # 除数：1/1.5≈0.667 token/字
            i += 1

        # CJK 扩展区 / 兼容汉字：UTF-8 三字节，BPE 按字节切分 → 固定 3 token/字
        elif (0x3400 <= cp <= 0x4DBF) or (0xF900 <= cp <= 0xFAFF):
            heuristic_t += 3.0
            i += 1

        # 拉丁字母连续段：按词长分桶
        elif _is_latin(cp):
            j = i + 1
            while j < n and _is_latin(ord(runes[j])):
                j += 1
            heuristic_t += math.ceil((j - i) / w["latin_divisor"])
            i = j

        # 平假名 / 片假名
        elif (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF):
            heuristic_t += w["hiragana"]
            i += 1

        # 韩文音节
        elif 0xAC00 <= cp <= 0xD7AF:
            heuristic_t += w["korean"]
            i += 1

        # 数字连续段（Unicode Nd 类）
        elif _is_nd_digit(ch):
            j = i + 1
            while j < n and _is_nd_digit(runes[j]):
                j += 1
            heuristic_t += (j - i) * w["digit"]
            i = j

        # 换行符
        elif ch == "\n" or ch == "\r":
            heuristic_t += w["newline"]
            i += 1

        # Tab（代码/JSON 缩进）
        elif ch == "\t":
            heuristic_t += w["tab"]
            i += 1

        # ASCII 空格
        elif ch == " ":
            heuristic_t += w["ascii_space"]
            i += 1

        # 中文标点 / 全角符号（，。、！？等）
        elif (0x2000 <= cp <= 0x206F) or (0x3000 <= cp <= 0x303F) or (0xFF00 <= cp <= 0xFFEF):
            heuristic_t += w["cjk_punctuation"]
            i += 1

        # ASCII 标点（可打印非字母数字）
        elif 0x21 <= cp <= 0x7E and not ch.isalnum():
            heuristic_t += w["ascii_punct"]
            i += 1

        # 其余：emoji、罕见符号等
        else:
            heuristic_t += w["other"]
            i += 1

    return bigram_t, heuristic_t


def estimate(
    table: bytes | None,
    text: str,
    discount: float | dict = 1.0,
    bigrams: dict[str, int] | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """返回估算 token 数：bigram_tokens + heuristic_tokens × discount。"""
    if isinstance(discount, dict):
        cat = classify_text(text)
        discount = discount.get(cat, discount.get("mixed", 1.0))
    bigram_t, heuristic_t = _scan(table, text, bigrams, weights)
    return bigram_t + heuristic_t * discount


def estimate_split(
    table: bytes | None,
    text: str,
    bigrams: dict[str, int] | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[float, float]:
    """返回 (bigram_tokens, heuristic_tokens)，供校准使用。"""
    return _scan(table, text, bigrams, weights)
