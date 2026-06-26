"""
Generate per-model CJK single-character token tables.

For each character U+4E00..U+9FFF, records how many tokens the model produces
for that character (capped at 255). Output: output/{key}.bin, 20992 bytes.

Usage:
  HF_TOKEN=...  ANTHROPIC_API_KEY=...  ARK_API_KEY=...  python generate_tables.py

HF_TOKEN is needed for gated repos (Kimi-K2, DeepSeek-V3 etc.).
Accept each model's license on HuggingFace before running.
"""

import os
import time
import requests
from config import (
    HF_MODELS, TIKTOKEN_MODELS, API_MODELS,
    CJK_START, CJK_END, CJK_COUNT,
    OUTPUT_DIR, HF_TOKEN, ARK_API_KEY, ANTHROPIC_API_KEY,
)


# ---------------------------------------------------------------------------
# HuggingFace path
# ---------------------------------------------------------------------------

def build_hf_table(model_key: str, repo: str) -> bytes:
    from transformers import AutoTokenizer
    print(f"[{model_key}] loading tokenizer: {repo}")
    kwargs = {"trust_remote_code": True}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    tok = AutoTokenizer.from_pretrained(repo, **kwargs)

    table = bytearray(CJK_COUNT)
    for i, cp in enumerate(range(CJK_START, CJK_END + 1)):
        ids = tok.encode(chr(cp), add_special_tokens=False)
        table[i] = min(len(ids), 255)
        if i % 2000 == 0:
            print(f"  {model_key}: {i}/{CJK_COUNT}")
    return bytes(table)


# ---------------------------------------------------------------------------
# tiktoken path
# ---------------------------------------------------------------------------

def build_tiktoken_table(model_key: str, encoding_name: str) -> bytes:
    import tiktoken
    print(f"[{model_key}] loading tiktoken encoding: {encoding_name}")
    enc = tiktoken.get_encoding(encoding_name)

    table = bytearray(CJK_COUNT)
    for i, cp in enumerate(range(CJK_START, CJK_END + 1)):
        ids = enc.encode(chr(cp))
        table[i] = min(len(ids), 255)
        if i % 2000 == 0:
            print(f"  {model_key}: {i}/{CJK_COUNT}")
    return bytes(table)


# ---------------------------------------------------------------------------
# API path — Volcano Engine (doubao)
# ---------------------------------------------------------------------------

def build_volc_table(model_key: str, cfg: dict) -> bytes:
    if not ARK_API_KEY:
        raise EnvironmentError("ARK_API_KEY is not set")

    endpoint = cfg["endpoint"]
    model    = cfg["model"]
    interval = 1.0 / cfg["rps"]
    headers  = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type":  "application/json",
    }

    table = bytearray(CJK_COUNT)
    for i, cp in enumerate(range(CJK_START, CJK_END + 1)):
        resp = requests.post(
            endpoint,
            headers=headers,
            json={"model": model, "text": chr(cp)},
            timeout=10,
        )
        resp.raise_for_status()
        total = resp.json()["total_tokens"]
        table[i] = min(total, 255)
        if i % 500 == 0:
            print(f"  {model_key}: {i}/{CJK_COUNT}")
        time.sleep(interval)
    return bytes(table)


# ---------------------------------------------------------------------------
# API path — Anthropic (claude)
# Baseline-subtract to remove messages-structure overhead.
# ---------------------------------------------------------------------------

def _claude_count(client, model: str, text: str) -> int:
    import anthropic
    resp = client.beta.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return resp.input_tokens


def build_claude_table(model_key: str, cfg: dict) -> bytes:
    import anthropic
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")

    client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    model    = cfg["model"]
    interval = 1.0 / cfg["rps"]

    # Message-structure overhead via a known 1-token reference char ('a').
    # Avoids empty content, which the API may reject.
    overhead = _claude_count(client, model, "a") - 1
    print(f"[{model_key}] overhead = {overhead} tokens")
    time.sleep(interval)

    table = bytearray(CJK_COUNT)
    for i, cp in enumerate(range(CJK_START, CJK_END + 1)):
        raw = _claude_count(client, model, chr(cp))
        table[i] = min(max(raw - overhead, 1), 255)
        if i % 500 == 0:
            print(f"  {model_key}: {i}/{CJK_COUNT}")
        time.sleep(interval)
    return bytes(table)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def build_table(model_key: str) -> bytes:
    if model_key in HF_MODELS:
        return build_hf_table(model_key, HF_MODELS[model_key])
    if model_key in TIKTOKEN_MODELS:
        return build_tiktoken_table(model_key, TIKTOKEN_MODELS[model_key])
    cfg = API_MODELS[model_key]
    if cfg["type"] == "volc":
        return build_volc_table(model_key, cfg)
    if cfg["type"] == "anthropic":
        return build_claude_table(model_key, cfg)
    raise ValueError(f"unknown model type for {model_key}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    from config import ALL_MODELS

    for key in ALL_MODELS:
        out_path = os.path.join(OUTPUT_DIR, f"{key}.bin")
        if os.path.exists(out_path):
            print(f"[{key}] already exists, skipping")
            continue
        table = build_table(key)
        with open(out_path, "wb") as f:
            f.write(table)
        print(f"[{key}] written {len(table)} bytes → {out_path}")


if __name__ == "__main__":
    main()
