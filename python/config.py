import os

# HuggingFace models — tokenizer loaded locally via transformers.
# Repos verified latest via web search on 2026-06-26.
# from_pretrained only downloads tokenizer files (a few MB), not model weights,
# so picking a large variant in the series is fine.
HF_MODELS = {
    "qwen":     "Qwen/Qwen3.6-27B",        # Qwen3.6 (latest open; 3.7-Max not open-weight)
    "qwen2":    "Qwen/Qwen2.5-72B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-V4-Pro",  # DeepSeek-V4 (2026-04)
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "glm":      "zai-org/GLM-5.2",          # NOTE org moved THUDM -> zai-org
    "glm4":     "zai-org/glm-4-9b-chat",
    "minimax":  "MiniMaxAI/MiniMax-M2.7",   # MiniMax-M2.7 (latest)
    "kimi":     "moonshotai/Kimi-K2.6",     # Kimi K2.6 (K2.7-Code is a coding variant)
}

# tiktoken encodings — local, no API needed.
# Verified current via web search 2026-06-26: o200k_base still covers the
# newest OpenAI models (gpt-4o, o1, o3, o4, gpt-5, gpt-5.1).
TIKTOKEN_MODELS = {
    "gpt-4o": "o200k_base",   # gpt-4o, gpt-4o-mini, o1, o3, o4, gpt-5, gpt-5.1
    "gpt-4":  "cl100k_base",  # gpt-4, gpt-3.5-turbo, gpt-4-turbo
}

# API-based models
API_MODELS = {
    "doubao": {
        "type":     "volc",
        "endpoint": "https://ark.cn-beijing.volces.com/api/v3/tokenization",
        "model":    "doubao-seed-1-6-250615",
        "rps":      10,
    },
    "claude": {
        "type":    "anthropic",
        "model":   "claude-opus-4-8",
        "rps":     5,
    },
}

ALL_MODELS = list(HF_MODELS) + list(TIKTOKEN_MODELS) + list(API_MODELS)

CJK_START = 0x4E00
CJK_END   = 0x9FFF
CJK_COUNT = CJK_END - CJK_START + 1  # 20992

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
ARK_API_KEY = os.environ.get("ARK_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
