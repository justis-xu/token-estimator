"""
从语料计算各模型、各文本类别的分段 discount 系数。

文本按 CJK 字符占比分三类：
    zh    — CJK 比例 ≥ 0.6（纯中文）
    mixed — 0.1 < CJK 比例 < 0.6
    en    — CJK 比例 ≤ 0.1（纯英文 / 代码）

对每个（模型 × 类别）组合，收集比值：
    r_i = (real_tokens_i - bigram_tokens_i) / heuristic_tokens_i

取第 55 百分位作为 discount（略高于均值，使估算值在统计上偏高，
避免 context-compression 场景的低估）。

输出：tables/config.json
    {"qwen": {"zh": 0.71, "mixed": 0.76, "en": 0.82}, ...}

用法：
    ARK_API_KEY=... HF_TOKEN=... python calculate_discount.py
"""

import json
import os
import time
import requests
from config import (
    HF_MODELS, TIKTOKEN_MODELS, API_MODELS, ALL_MODELS,
    OUTPUT_DIR, TABLES_DIR, HF_TOKEN, ARK_API_KEY,
)
from estimate import estimate_split, classify_text, DEFAULT_WEIGHTS
from hf_data import ensure_output_file

CATEGORIES = ("zh", "mixed", "en")

# ---------------------------------------------------------------------------
# 加载词表
# ---------------------------------------------------------------------------

def load_table(model_key: str) -> bytes:
    path = os.path.join(TABLES_DIR, f"{model_key}.bin")
    with open(path, "rb") as f:
        return f.read()


def load_bigrams(model_key: str) -> dict[str, int] | None:
    path = os.path.join(TABLES_DIR, f"{model_key}.bigram")
    if not os.path.exists(path):
        return None
    import struct
    with open(path, "rb") as f:
        data = f.read()
    from config import CJK_START
    n = struct.unpack(">I", data[:4])[0]
    result = {}
    for i in range(n):
        off = 4 + i * 5
        off1, off2, count = struct.unpack(">HHB", data[off:off + 5])
        result[chr(CJK_START + off1) + chr(CJK_START + off2)] = count
    return result


# ---------------------------------------------------------------------------
# 各模型真实 token 数
# ---------------------------------------------------------------------------

_hf_tokenizers: dict = {}
_tiktoken_encs: dict = {}


def real_count_hf(model_key: str, text: str) -> int:
    if model_key not in _hf_tokenizers:
        from transformers import AutoTokenizer
        kwargs = {"trust_remote_code": True}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        _hf_tokenizers[model_key] = AutoTokenizer.from_pretrained(HF_MODELS[model_key], **kwargs)
    return len(_hf_tokenizers[model_key].encode(text, add_special_tokens=False))


def real_count_tiktoken(model_key: str, text: str) -> int:
    if model_key not in _tiktoken_encs:
        import tiktoken
        _tiktoken_encs[model_key] = tiktoken.get_encoding(TIKTOKEN_MODELS[model_key])
    return len(_tiktoken_encs[model_key].encode(text))


def real_count_volc(cfg: dict, text: str) -> int:
    headers = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type":  "application/json",
    }
    resp = requests.post(
        cfg["endpoint"], headers=headers,
        json={"model": cfg["model"], "text": text}, timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["total_tokens"]


def real_count(model_key: str, text: str) -> int:
    if model_key in HF_MODELS:
        return real_count_hf(model_key, text)
    if model_key in TIKTOKEN_MODELS:
        return real_count_tiktoken(model_key, text)
    cfg = API_MODELS[model_key]
    if cfg["type"] == "volc":
        return real_count_volc(cfg, text)
    raise ValueError(f"未知模型类型: {model_key}")


# ---------------------------------------------------------------------------
# Discount 校准
# ---------------------------------------------------------------------------

# 55 百分位：略高于均值，使估算值在统计上偏高，
# 为 context-compression 场景提供安全边际
DISCOUNT_PERCENTILE = 55


def _percentile(data: list[float], p: int) -> float:
    if not data:
        return 1.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def calc_discount(
    model_key: str,
    corpus: list[dict],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """返回 {"zh": float, "mixed": float, "en": float} 分段 discount。

    weights 来自 config.json 的 "weights" 段（若已校准），
    与 estimate_split 保持一致。
    """
    table = load_table(model_key)
    cfg   = API_MODELS.get(model_key, {})
    interval = 1.0 / cfg["rps"] if cfg else 0.0

    bigrams = load_bigrams(model_key)
    if bigrams:
        print(f"  [{model_key}] 加载 {len(bigrams)} 个高频词对")

    ratios: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    for entry in corpus:
        text = entry["text"]
        cat  = classify_text(text)
        # discount 只作用于启发式部分；高频词命中不参与缩放。
        # 校准公式：discount = (real - bigram_tokens) / heuristic_tokens
        bigram_t, heuristic_t = estimate_split(table, text, bigrams=bigrams, weights=weights)
        if heuristic_t <= 0:
            continue  # 文本完全被高频词覆盖，discount 无意义
        real = real_count(model_key, text)
        ratios[cat].append((real - bigram_t) / heuristic_t)
        if interval:
            time.sleep(interval)

    result: dict[str, float] = {}
    for cat in CATEGORIES:
        rs = ratios[cat]
        result[cat] = round(_percentile(rs, DISCOUNT_PERCENTILE), 4) if rs else 1.0
        mean_val = (sum(rs) / len(rs)) if rs else 0.0
        print(f"    {cat}: mean={mean_val:.4f}  p{DISCOUNT_PERCENTILE}={result[cat]:.4f}  n={len(rs)}")
    return result


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    corpus_path = ensure_output_file("corpus.jsonl")

    corpus = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))
    print(f"语料: {len(corpus)} 条")

    # 从 config.json 加载已校准权重（若存在），保持与 Go 侧一致
    existing_path = os.path.join(TABLES_DIR, "config.json")
    existing: dict[str, dict] = {}
    weights: dict[str, float] | None = None

    if os.path.exists(existing_path):
        try:
            raw = json.loads(open(existing_path).read())
            for k, v in raw.items():
                if k == "weights":
                    # 取 default 权重供 Python 侧 estimate_split 使用
                    if isinstance(v, dict) and "default" in v:
                        weights = v["default"]
                    elif isinstance(v, dict):
                        weights = next(iter(v.values()), None)
                    print(f"已加载权重配置（{'已校准' if weights else '默认'}）")
                elif isinstance(v, dict) and set(v.keys()) >= {"zh", "mixed", "en"}:
                    existing[k] = v
            if existing:
                print(f"已加载 {len(existing)} 个模型的 discount 基准")
        except Exception:
            pass

    def _api_key_missing(model_key: str) -> bool:
        cfg = API_MODELS.get(model_key, {})
        if cfg.get("type") == "volc" and not ARK_API_KEY:
            return True
        return False

    discounts: dict[str, dict] = {}
    skipped: list[str] = []
    for key in ALL_MODELS:
        if _api_key_missing(key):
            if key in existing:
                discounts[key] = existing[key]
                print(f"\n[{key}] 跳过（无 API Key）— 保留已有值 {existing[key]}")
            else:
                print(f"\n[{key}] 跳过（无 API Key）— 无已有值，由 default 兜底")
            skipped.append(key)
            continue
        print(f"\n[{key}] 计算 discount ...")
        discounts[key] = calc_discount(key, corpus, weights=weights)
        print(f"  {key}: {discounts[key]}")

    if skipped:
        print(f"\n跳过模型（无 API Key）: {skipped}")

    # default = 已计算模型中各类别最大值（最保守）
    computed = [v for k, v in discounts.items() if k not in skipped]
    if not computed:
        computed = list(discounts.values())
    if computed:
        discounts["default"] = {
            cat: max(v[cat] for v in computed if cat in v)
            for cat in CATEGORIES
        }
    else:
        discounts["default"] = {cat: 1.0 for cat in CATEGORIES}
    print(f"\ndefault（最保守）: {discounts['default']}")

    # 保留 config.json 中的 weights 段，只更新 discount 值
    out_path = os.path.join(TABLES_DIR, "config.json")
    existing_cfg: dict = {}
    if os.path.exists(out_path):
        try:
            existing_cfg = json.loads(open(out_path).read())
        except Exception:
            pass
    existing_cfg.update(discounts)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing_cfg, f, indent=2, ensure_ascii=False)
    print(f"\nconfig.json → {out_path}")


if __name__ == "__main__":
    main()
