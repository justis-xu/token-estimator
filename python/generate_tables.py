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

def build_volc_table(model_key: str, cfg: dict, out_path: str = "") -> bytes:
    if not ARK_API_KEY:
        raise EnvironmentError("ARK_API_KEY is not set")

    endpoint = cfg["endpoint"]
    model    = cfg["model"]
    interval = 1.0 / cfg["rps"]
    headers  = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type":  "application/json",
    }

    # Resume from checkpoint if it exists.
    checkpoint_path = out_path + ".ckpt"
    table = bytearray(CJK_COUNT)
    start = 0
    if os.path.exists(checkpoint_path):
        ckpt = open(checkpoint_path, "rb").read()
        n = len(ckpt)
        table[:n] = ckpt
        start = n
        print(f"  {model_key}: resuming from {start}/{CJK_COUNT}")

    backoff = interval
    for i, cp in enumerate(range(CJK_START + start, CJK_END + 1), start=start):
        while True:
            try:
                resp = requests.post(
                    endpoint,
                    headers=headers,
                    json={"model": model, "text": chr(cp)},
                    timeout=10,
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
                print(f"  {model_key}: error at {i} ({e}), retrying in {backoff:.1f}s", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

        total = resp.json()["data"][0]["total_tokens"]
        table[i] = min(total, 255)
        if (i + 1) % 500 == 0:
            # Save checkpoint every 500 chars.
            open(checkpoint_path, "wb").write(bytes(table[:i + 1]))
            print(f"  {model_key}: {i + 1}/{CJK_COUNT}", flush=True)
        time.sleep(interval)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
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

def build_table(model_key: str, out_path: str = "") -> bytes:
    if model_key in HF_MODELS:
        return build_hf_table(model_key, HF_MODELS[model_key])
    if model_key in TIKTOKEN_MODELS:
        return build_tiktoken_table(model_key, TIKTOKEN_MODELS[model_key])
    cfg = API_MODELS[model_key]
    if cfg["type"] == "volc":
        return build_volc_table(model_key, cfg, out_path)
    if cfg["type"] == "anthropic":
        return build_claude_table(model_key, cfg)
    raise ValueError(f"unknown model type for {model_key}")


def _api_key_missing(model_key: str) -> bool:
    cfg = API_MODELS.get(model_key, {})
    if cfg.get("type") == "anthropic" and not ANTHROPIC_API_KEY:
        return True
    if cfg.get("type") == "volc" and not ARK_API_KEY:
        return True
    return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    from config import ALL_MODELS

    for key in ALL_MODELS:
        out_path = os.path.join(OUTPUT_DIR, f"{key}.bin")
        if os.path.exists(out_path):
            print(f"[{key}] already exists, skipping")
            continue
        if _api_key_missing(key):
            print(f"[{key}] skipped — no API key")
            continue
        table = build_table(key, out_path)
        with open(out_path, "wb") as f:
            f.write(table)
        print(f"[{key}] written {len(table)} bytes → {out_path}")


if __name__ == "__main__":
    main()
