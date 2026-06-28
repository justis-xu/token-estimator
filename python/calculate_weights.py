"""
从真实分词结果校准各字符类的启发式权重系数。

方法：对每类字符单独构造探针文本（仅含该类字符），
通过本地分词器取得真实 token 数，计算 weight = real_tokens / unit_count。
各字符类互相隔离，避免相互干扰。

结果写入 tables/config.json 的 "weights" 段，被 Go 估算器直接读取；
Python 侧 estimate.py 也从同一配置加载，保持双端同步。

执行顺序（generate_tables.py 和 generate_bigrams.py 之后）：
    python calculate_weights.py   # 校准权重
    python calculate_discount.py  # 用新权重重新校准 discount
    python generate_golden.py     # 重新生成评测集

用法：
    HF_TOKEN=... python calculate_weights.py
"""

import json
import os

from config import (
    HF_MODELS, TIKTOKEN_MODELS, API_MODELS,
    TABLES_DIR, HF_TOKEN,
)
from estimate import DEFAULT_WEIGHTS

# ---------------------------------------------------------------------------
# 探针定义
# ---------------------------------------------------------------------------
# 每个探针：(字符类名称, 重复单元, 重复次数)
# 原则：仅包含目标类字符，unit_count = 实际计量单位数。
#
# 注：
# - 拉丁字母探针用无空格连续字母，模拟单词内部（避免空格干扰）
# - 数字探针用连续数字串（一整段），测试数字段权重
# - emoji 用单码点 emoji（U+1F600），一个 emoji = 一个 rune
# ---------------------------------------------------------------------------

LATIN_UNIT_LEN = 4   # 每个计量单位的字母数（对应 latin_divisor 的校准基准）
REPS           = 200 # 探针重复次数（越大噪声越小）

_LATIN_PROBE    = "abcd" * REPS                  # 800 个拉丁字母，共 REPS 个 4 字母单元
_DIGIT_PROBE    = "1234567890" * (REPS // 10)    # 200 位连续数字
_HIRAGANA_PROBE = "あいうえお" * (REPS // 5)    # 200 个平假名
_KOREAN_PROBE   = "가나다라마" * (REPS // 5)    # 200 个韩文音节

# 不参与探针校准的 key（直接沿用 DEFAULT_WEIGHTS）：
#   default_cjk   — 无词表时的除数，语义是"N字=1token"；Extension A 已在
#                   estimator.go/estimate.py 中硬编码 3.0（UTF-8字节级BPE），不需校准。
#   newline/tab/ascii_space — 孤立探针中 BPE 批量合并空白，给出失真的极低比值；
#                   在真实混合文本里这三类字符的贡献已被 discount 吸收，保留经验值。
PROBES: list[tuple[str, str, int]] = [
    # (weight_key, probe_text, unit_count)
    ("latin_divisor",  _LATIN_PROBE,              REPS),               # REPS 个 4 字母单元
    ("hiragana",       _HIRAGANA_PROBE,            len(_HIRAGANA_PROBE)),
    ("korean",         _KOREAN_PROBE,              len(_KOREAN_PROBE)),
    ("digit",          _DIGIT_PROBE,               len(_DIGIT_PROBE)),
    ("cjk_punctuation","，。！？、；：" * (REPS // 7), (REPS // 7) * 7),
    ("ascii_punct",    "!@#$%^&*" * (REPS // 8),  (REPS // 8) * 8),
    ("other",          "😀" * (REPS // 2),         REPS // 2),         # 单码点 emoji
]

# ---------------------------------------------------------------------------
# 分词后端（复用 calculate_discount.py 的实现）
# ---------------------------------------------------------------------------

from calculate_discount import real_count_hf, real_count_tiktoken


def _real_count(model_key: str, text: str) -> int:
    if model_key in HF_MODELS:
        return real_count_hf(model_key, text)
    if model_key in TIKTOKEN_MODELS:
        return real_count_tiktoken(model_key, text)
    # API 模型不参与本地权重校准
    return -1


# ---------------------------------------------------------------------------
# 校准逻辑
# ---------------------------------------------------------------------------

def calibrate_model(model_key: str) -> dict[str, float] | None:
    """校准单个模型的字符类权重，返回权重字典（与 DEFAULT_WEIGHTS 同结构）。

    对每类字符：weight = real_tokens / unit_count。
    latin_divisor 特殊处理：divisor = unit_len × unit_count / real_tokens。
    """
    print(f"\n[{model_key}] 开始校准权重 ...")
    weights: dict[str, float] = {}

    for key, probe, unit_count in PROBES:
        try:
            real = _real_count(model_key, probe)
        except Exception as e:
            print(f"  [{model_key}] {key}: 分词失败 ({e})，跳过")
            continue

        if unit_count == 0:
            continue

        if key == "latin_divisor":
            # 探针 = REPS 个 LATIN_UNIT_LEN 字母单元（无空格连续串）
            # 真实分词得到 real tokens，推导 divisor = 总字母数 / real
            total_letters = LATIN_UNIT_LEN * unit_count
            fitted = round(total_letters / max(real, 1), 4) if real > 0 else DEFAULT_WEIGHTS[key]
            weights[key] = fitted
            print(f"  {key}: {total_letters} 字母 → {real} tokens → divisor={fitted:.4f}")
        else:
            fitted = round(real / unit_count, 4)
            weights[key] = fitted
            print(f"  {key}: {unit_count} 单位 → {real} tokens → weight={fitted:.4f}")

    if not weights:
        return None

    # 未校准的 key 保留默认值
    for k, v in DEFAULT_WEIGHTS.items():
        if k not in weights:
            weights[k] = v

    return weights


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    cfg_path = os.path.join(TABLES_DIR, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"{cfg_path} 不存在，请先运行 calculate_discount.py")

    with open(cfg_path, encoding="utf-8") as f:
        cfg: dict = json.load(f)

    # 只校准本地可分词的模型（HF + tiktoken），跳过 API 模型
    calibrated: dict[str, dict[str, float]] = {}
    for model_key in list(HF_MODELS) + list(TIKTOKEN_MODELS):
        result = calibrate_model(model_key)
        if result:
            calibrated[model_key] = result

    if not calibrated:
        print("无可校准模型（需要 HF_TOKEN 或 tiktoken），退出")
        return

    # 计算跨模型均值作为 "default" 权重。
    # default_cjk 例外：各模型的 3.0 是"扩展区汉字 + 已知 table"场景下的测量值，
    # 而 "default" 段在 table=nil（未知模型）时会被用于全量普通汉字，
    # 平均约 1.5 token/字，必须保留经验值，不能用探针均值覆盖。
    SKIP_FOR_DEFAULT = {"default_cjk"}
    default_weights: dict[str, float] = {}
    for k in DEFAULT_WEIGHTS:
        if k in SKIP_FOR_DEFAULT:
            default_weights[k] = DEFAULT_WEIGHTS[k]
            continue
        vals = [m[k] for m in calibrated.values() if k in m]
        default_weights[k] = round(sum(vals) / len(vals), 4) if vals else DEFAULT_WEIGHTS[k]

    cfg["weights"] = {**calibrated, "default": default_weights}

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"\nweights → {cfg_path}")
    print("默认权重（跨模型均值）：")
    for k, v in default_weights.items():
        ref = DEFAULT_WEIGHTS.get(k, "?")
        diff = f"（原 {ref}）" if v != ref else ""
        print(f"  {k}: {v} {diff}")

    print("\n完成。请重新运行 calculate_discount.py 和 generate_golden.py 使新权重生效。")


if __name__ == "__main__":
    main()
