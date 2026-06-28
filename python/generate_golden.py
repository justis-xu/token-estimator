"""
Generate golden.jsonl — ground-truth token counts for the Go accuracy test.

For each (corpus entry × model), record real token count.
Output: output/golden.jsonl  (calibration data, not needed for Go runtime)
  {"text":"...","model":"qwen","tokens":42}

Usage:
  ARK_API_KEY=... HF_TOKEN=... python generate_golden.py

This script reuses the real_count() helpers from calculate_discount.py.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from calculate_discount import real_count
from config import ALL_MODELS, API_MODELS, OUTPUT_DIR
from hf_data import ensure_output_file


def _rate_limit_interval(model_key: str) -> float:
    cfg = API_MODELS.get(model_key)
    if not cfg:
        return 0.0
    return 1.0 / cfg["rps"]


def write_golden(
    corpus: list[dict],
    model_keys: list[str],
    out_path: str,
    real_count_fn=real_count,
    sleep_fn=time.sleep,
) -> int:
    total = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for model_key in model_keys:
            print(f"[{model_key}] ...")
            interval = _rate_limit_interval(model_key)
            for entry in corpus:
                text = entry["text"]
                try:
                    tokens = real_count_fn(model_key, text)
                except Exception as e:
                    print(f"  skip: {e}")
                    continue
                out.write(json.dumps(
                    {"text": text, "model": model_key, "tokens": tokens},
                    ensure_ascii=False,
                ) + "\n")
                total += 1
                if interval:
                    sleep_fn(interval)
    return total


def _api_key_missing(model_key: str) -> bool:
    from config import API_MODELS, ARK_API_KEY
    cfg = API_MODELS.get(model_key, {})
    if cfg.get("type") == "volc" and not ARK_API_KEY:
        return True
    return False


def main():
    corpus_path = ensure_output_file("corpus.jsonl")

    corpus = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))

    active_models = []
    for key in ALL_MODELS:
        if _api_key_missing(key):
            print(f"[{key}] skipped — no API key")
        else:
            active_models.append(key)

    out_path = os.path.join(OUTPUT_DIR, "golden.jsonl")
    total = write_golden(corpus, active_models, out_path)

    print(f"\ngolden.jsonl: {total} entries → {out_path}")


if __name__ == "__main__":
    main()
