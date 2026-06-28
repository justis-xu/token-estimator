import os

# HuggingFace 模型——通过 transformers 本地加载分词器。
# 只需下载分词器配置（几 MB），无需模型权重。
# 2026-06-26 核对为当前最新版本。
HF_MODELS = {
    "qwen":        "Qwen/Qwen3.6-27B",           # Qwen3.6（最新开源；3.7-Max 未开源）
    "qwen2":       "Qwen/Qwen2.5-72B-Instruct",
    "deepseek":    "deepseek-ai/DeepSeek-V4-Pro", # DeepSeek-V4（2026-04）
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "glm":         "zai-org/GLM-5.2",             # 注意：组织从 THUDM 改为 zai-org
    "glm4":        "zai-org/glm-4-9b-chat",
    "minimax":     "MiniMaxAI/MiniMax-M2.7",      # MiniMax-M2.7（当前最新）
    "kimi":        "moonshotai/Kimi-K2.6",        # K2.7-Code 是代码专用变体
}

# tiktoken 编码——本地离线，无需 API。
# 2026-06-26 核对：o200k_base 覆盖最新 OpenAI 模型。
TIKTOKEN_MODELS = {
    "gpt-4o": "o200k_base",  # gpt-4o, gpt-4o-mini, o1, o3, o4, gpt-5, gpt-5.1
    "gpt-4":  "cl100k_base", # gpt-4, gpt-3.5-turbo, gpt-4-turbo
    "claude": "o200k_base",  # 近似值；Claude 无开源分词器
}

# API 模型——需要调用外部接口获取 token 数
API_MODELS = {
    "doubao": {
        "type":     "volc",
        "endpoint": "https://ark.cn-beijing.volces.com/api/v3/tokenization",
        "model":    "doubao-seed-1-6-250615",
        "rps":      40,
    },
}

ALL_MODELS = list(HF_MODELS) + list(TIKTOKEN_MODELS) + list(API_MODELS)

CJK_START = 0x4E00
CJK_END   = 0x9FFF
CJK_COUNT = CJK_END - CJK_START + 1  # 20992 个汉字

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables")
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
ARK_API_KEY = os.environ.get("ARK_API_KEY", "")
HF_DATASET_REPO = "justis-xu/token-estimator-data"
HF_DATASET_REVISION = os.environ.get("HF_DATASET_REVISION", "main")
HF_DATASET_SUBDIR = ""
