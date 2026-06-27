"""
Generate per-model bigram token tables.

For the top-N most frequent adjacent CJK character pairs found in the corpus,
records how many tokens the model produces for that 2-character string.

Output: output/{key}.bigram   (binary)
Format: big-endian uint32 N, then N × (uint16 offset1, uint16 offset2, uint8 count)
        sorted by (offset1<<16|offset2) for fast lookup in Go.

Usage:
    ARK_API_KEY=... HF_TOKEN=... python generate_bigrams.py
"""

import json
import os
import struct
import time
import requests
from collections import Counter
from config import (
    HF_MODELS, TIKTOKEN_MODELS, API_MODELS, ALL_MODELS,
    CJK_START, CJK_END,
    OUTPUT_DIR, HF_TOKEN, ARK_API_KEY, ANTHROPIC_API_KEY,
)

TOP_N = 5000  # most-frequent bigrams to cover


# ---------------------------------------------------------------------------
# Corpus extraction
# ---------------------------------------------------------------------------

def extract_top_bigrams(corpus_path: str, top_n: int = TOP_N) -> list[str]:
    counter: Counter = Counter()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            prev: str | None = None
            for ch in text:
                cp = ord(ch)
                if CJK_START <= cp <= CJK_END:
                    if prev is not None:
                        counter[prev + ch] += 1
                    prev = ch
                else:
                    prev = None
    return [bg for bg, _ in counter.most_common(top_n)]


# ---------------------------------------------------------------------------
# Binary I/O
# ---------------------------------------------------------------------------

def write_bigram_bin(result: dict[str, int], out_path: str) -> None:
    records = []
    for bg, count in result.items():
        off1 = ord(bg[0]) - CJK_START
        off2 = ord(bg[1]) - CJK_START
        records.append((off1, off2, min(count, 255)))
    records.sort(key=lambda r: (r[0] << 16) | r[1])
    with open(out_path, "wb") as f:
        f.write(struct.pack(">I", len(records)))
        for off1, off2, count in records:
            f.write(struct.pack(">HHB", off1, off2, count))
    print(f"  wrote {len(records)} bigrams → {out_path}")


def read_bigram_bin(path: str) -> dict[str, int]:
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack(">I", data[:4])[0]
    result = {}
    for i in range(n):
        off = 4 + i * 5
        off1, off2, count = struct.unpack(">HHB", data[off:off + 5])
        result[chr(CJK_START + off1) + chr(CJK_START + off2)] = count
    return result


# ---------------------------------------------------------------------------
# Per-model builders
# ---------------------------------------------------------------------------

def build_bigram_hf(model_key: str, repo: str, bigrams: list[str]) -> dict[str, int]:
    from transformers import AutoTokenizer
    print(f"[{model_key}] loading tokenizer: {repo}")
    kwargs: dict = {"trust_remote_code": True}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    tok = AutoTokenizer.from_pretrained(repo, **kwargs)
    result = {}
    for i, bg in enumerate(bigrams):
        ids = tok.encode(bg, add_special_tokens=False)
        result[bg] = len(ids)
        if i % 500 == 0:
            print(f"  {model_key}: {i}/{len(bigrams)}")
    return result


def build_bigram_tiktoken(model_key: str, encoding_name: str, bigrams: list[str]) -> dict[str, int]:
    import tiktoken
    print(f"[{model_key}] loading tiktoken encoding: {encoding_name}")
    enc = tiktoken.get_encoding(encoding_name)
    return {bg: len(enc.encode(bg)) for bg in bigrams}


def build_bigram_volc(model_key: str, cfg: dict, bigrams: list[str]) -> dict[str, int]:
    if not ARK_API_KEY:
        raise EnvironmentError("ARK_API_KEY is not set")
    endpoint = cfg["endpoint"]
    model    = cfg["model"]
    interval = 1.0 / cfg["rps"]
    headers  = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type":  "application/json",
    }
    result = {}
    backoff = interval
    for i, bg in enumerate(bigrams):
        while True:
            try:
                resp = requests.post(
                    endpoint, headers=headers,
                    json={"model": model, "text": bg}, timeout=10,
                )
                if resp.status_code == 429:
                    print(f"  {model_key}: 429 at {i}, backing off {backoff:.2f}s", flush=True)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue
                resp.raise_for_status()
                backoff = interval
                break
            except Exception as e:
                print(f"  {model_key}: error at {i} ({e}), retrying {backoff:.1f}s", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
        result[bg] = resp.json()["data"][0]["total_tokens"]
        if (i + 1) % 200 == 0:
            print(f"  {model_key}: {i + 1}/{len(bigrams)}", flush=True)
        time.sleep(interval)
    return result


def _api_key_missing(model_key: str) -> bool:
    cfg = API_MODELS.get(model_key, {})
    if cfg.get("type") == "anthropic" and not ANTHROPIC_API_KEY:
        return True
    if cfg.get("type") == "volc" and not ARK_API_KEY:
        return True
    return False


def build_bigram(model_key: str, bigrams: list[str]) -> dict[str, int]:
    if model_key in HF_MODELS:
        return build_bigram_hf(model_key, HF_MODELS[model_key], bigrams)
    if model_key in TIKTOKEN_MODELS:
        return build_bigram_tiktoken(model_key, TIKTOKEN_MODELS[model_key], bigrams)
    cfg = API_MODELS[model_key]
    if cfg["type"] == "volc":
        return build_bigram_volc(model_key, cfg, bigrams)
    raise ValueError(f"unsupported model type for {model_key}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    corpus_path = os.path.join(OUTPUT_DIR, "corpus.jsonl")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError("corpus.jsonl not found — run scrape_corpus.py first")

    print(f"Extracting top {TOP_N} bigrams from corpus ...")
    bigrams = extract_top_bigrams(corpus_path, TOP_N)
    print(f"  found {len(bigrams)} unique bigrams")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for key in ALL_MODELS:
        out_path = os.path.join(OUTPUT_DIR, f"{key}.bigram")
        if os.path.exists(out_path):
            print(f"[{key}] already exists, skipping")
            continue
        if _api_key_missing(key):
            print(f"[{key}] skipped — no API key")
            continue
        print(f"\n[{key}] building bigram table ({len(bigrams)} pairs) ...")
        result = build_bigram(key, bigrams)
        write_bigram_bin(result, out_path)


if __name__ == "__main__":
    main()
