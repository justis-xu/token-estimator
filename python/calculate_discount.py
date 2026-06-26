"""
Calculate per-model discount coefficients from corpus.jsonl.

GLOBAL multiplier approach: discount corrects the *whole* heuristic estimate,
not just the CJK part. For each model:

    discount = mean( real_tokens / estimate(table, text, discount=1.0) )

where estimate() mirrors go/estimator.go exactly (see estimate.py).
Output: output/config.json

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
from estimate import estimate

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

def calc_discount(model_key: str, corpus: list[dict]) -> float:
    table = load_table(model_key)
    cfg   = API_MODELS.get(model_key, {})
    interval = 1.0 / cfg["rps"] if cfg else 0.0

    ratios = []
    for entry in corpus:
        text = entry["text"]
        est = estimate(table, text, discount=1.0)
        if est <= 0:
            continue
        real = real_count(model_key, text)
        ratios.append(real / est)
        if interval:
            time.sleep(interval)

    if not ratios:
        return 1.0
    return sum(ratios) / len(ratios)


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

    discounts: dict[str, float] = {}
    for key in ALL_MODELS:
        print(f"\n[{key}] calculating discount ...")
        d = calc_discount(key, corpus)
        discounts[key] = round(d, 4)
        print(f"  {key}: {discounts[key]:.4f}")

    # default = most conservative (largest) so unknown models never undercount
    discounts["default"] = max(discounts.values())
    print(f"\ndefault (most conservative): {discounts['default']:.4f}")

    out_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(discounts, f, indent=2, ensure_ascii=False)
    print(f"\nconfig.json → {out_path}")


if __name__ == "__main__":
    main()
