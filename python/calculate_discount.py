"""
Calculate per-model, per-category discount coefficients from corpus.jsonl.

Each corpus entry is classified into one of three text categories based on
the fraction of CJK characters in the text:

    zh    — CJK ratio >= 0.6   (mostly Chinese)
    mixed — 0.1 < CJK ratio < 0.6
    en    — CJK ratio <= 0.1   (mostly English / code)

For each (model, category) pair we collect ratios:

    r_i = real_tokens_i / estimate(table, text_i, discount=1.0)

and set:

    discount = percentile(ratios, DISCOUNT_PERCENTILE)

Output: output/config.json  —  schema:
    {"qwen": {"zh": 0.71, "mixed": 0.76, "en": 0.82}, ...}

Usage:
  ARK_API_KEY=... ANTHROPIC_API_KEY=... HF_TOKEN=... python calculate_discount.py
"""

import json
import os
import time
import requests
from config import (
    HF_MODELS, TIKTOKEN_MODELS, API_MODELS, ALL_MODELS,
    OUTPUT_DIR, HF_TOKEN, ARK_API_KEY, ANTHROPIC_API_KEY,
)
from estimate import estimate, classify_text

CATEGORIES = ("zh", "mixed", "en")

# ---------------------------------------------------------------------------
# Load tables
# ---------------------------------------------------------------------------

def load_table(model_key: str) -> bytes:
    path = os.path.join(OUTPUT_DIR, f"{model_key}.bin")
    with open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Real token count — one backend per model type
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
    return resp.json()["total_tokens"]


_claude_client = None
_claude_overhead: int | None = None


def _get_claude_client():
    global _claude_client
    if _claude_client is None:
        import anthropic
        _claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _claude_client


def _claude_raw_count(model: str, text: str) -> int:
    resp = _get_claude_client().beta.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return resp.input_tokens


def _get_claude_overhead(model: str) -> int:
    """Message-structure overhead. Measured via a known 1-token reference char
    ('a' is a single token in any tokenizer), avoiding empty content which the
    API may reject."""
    global _claude_overhead
    if _claude_overhead is None:
        _claude_overhead = _claude_raw_count(model, "a") - 1
        print(f"  [claude] overhead = {_claude_overhead}")
    return _claude_overhead


def real_count_claude(cfg: dict, text: str) -> int:
    overhead = _get_claude_overhead(cfg["model"])
    return max(_claude_raw_count(cfg["model"], text) - overhead, 1)


def real_count(model_key: str, text: str) -> int:
    if model_key in HF_MODELS:
        return real_count_hf(model_key, text)
    if model_key in TIKTOKEN_MODELS:
        return real_count_tiktoken(model_key, text)
    cfg = API_MODELS[model_key]
    if cfg["type"] == "volc":
        return real_count_volc(cfg, text)
    if cfg["type"] == "anthropic":
        return real_count_claude(cfg, text)
    raise ValueError(f"unknown type for {model_key}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# 55th percentile: slightly above mean, biases toward over-estimation while
# keeping MAE within the 15% threshold for context-compression safety.
DISCOUNT_PERCENTILE = 55


def _percentile(data: list[float], p: int) -> float:
    """Return the p-th percentile of data (0–100)."""
    if not data:
        return 1.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def calc_discount(model_key: str, corpus: list[dict]) -> dict[str, float]:
    """Return {"zh": float, "mixed": float, "en": float} discount per text category."""
    table = load_table(model_key)
    cfg   = API_MODELS.get(model_key, {})
    interval = 1.0 / cfg["rps"] if cfg else 0.0

    ratios: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    for entry in corpus:
        text = entry["text"]
        cat  = classify_text(text)
        est  = estimate(table, text, discount=1.0)
        if est <= 0:
            continue
        real = real_count(model_key, text)
        ratios[cat].append(real / est)
        if interval:
            time.sleep(interval)

    result: dict[str, float] = {}
    for cat in CATEGORIES:
        rs = ratios[cat]
        result[cat] = round(_percentile(rs, DISCOUNT_PERCENTILE), 4) if rs else 1.0
        mean_val = (sum(rs) / len(rs)) if rs else 0.0
        print(f"    {cat}: mean={mean_val:.4f}  p{DISCOUNT_PERCENTILE}={result[cat]:.4f}  n={len(rs)}")
    return result


def main():
    corpus_path = os.path.join(OUTPUT_DIR, "corpus.jsonl")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"{corpus_path} not found — run scrape_corpus.py first")

    corpus = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))
    print(f"corpus: {len(corpus)} entries")

    # Load existing segmented discounts to preserve values for skipped models.
    existing: dict[str, dict] = {}
    existing_path = os.path.join(OUTPUT_DIR, "config.json")
    if os.path.exists(existing_path):
        try:
            raw = json.loads(open(existing_path).read())
            for k, v in raw.items():
                if isinstance(v, dict) and set(v.keys()) >= {"zh", "mixed", "en"}:
                    existing[k] = v
            if existing:
                print(f"loaded {len(existing)} existing segmented discounts from config.json")
        except Exception:
            pass

    def _api_key_missing(model_key: str) -> bool:
        cfg = API_MODELS.get(model_key, {})
        if cfg.get("type") == "anthropic" and not ANTHROPIC_API_KEY:
            return True
        if cfg.get("type") == "volc" and not ARK_API_KEY:
            return True
        return False

    discounts: dict[str, dict] = {}
    skipped: list[str] = []
    for key in ALL_MODELS:
        if _api_key_missing(key):
            if key in existing:
                discounts[key] = existing[key]
                print(f"\n[{key}] skipped (no API key) — kept existing {existing[key]}")
            else:
                print(f"\n[{key}] skipped (no API key) — no existing value, will be covered by default")
            skipped.append(key)
            continue
        print(f"\n[{key}] calculating discount ...")
        discounts[key] = calc_discount(key, corpus)
        print(f"  {key}: {discounts[key]}")

    if skipped:
        print(f"\nskipped models (no API key): {skipped}")

    # default = most conservative (largest per category) among computed models.
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
    print(f"\ndefault (most conservative): {discounts['default']}")

    out_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(discounts, f, indent=2, ensure_ascii=False)
    print(f"\nconfig.json → {out_path}")


if __name__ == "__main__":
    main()
