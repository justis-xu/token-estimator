# token-estimator

面向中文场景的轻量级 token 数量估算器。无 API 调用，无网络依赖，Go 内存查表，单次估算延迟在微秒级。

---

## 为什么需要它

大语言模型按 token 计费，也按 token 限制上下文长度。上下文引擎在每轮对话前都需要知道当前 prompt 占多少 token，才能决定是否截断或压缩历史。

调分词 API 是精确的，但有代价：每次额外一次网络请求（几十毫秒延迟）、接口有调用频次限制、API 不可用时上下文引擎跟着挂。

本地精确分词（如 tiktoken）只覆盖 GPT 系列，对 Qwen、DeepSeek、Claude 等模型估算结果是错的。

token-estimator 的目标：**本地、零网络、延迟接近零、误差 ≤ 15%**，覆盖主流中文模型。

---

## 方案对比

| 方案 | 精度 | 延迟 | 支持模型 | 无网络 |
|------|------|------|---------|--------|
| 字符数粗估 | 差 | 极低 | 任意 | ✅ |
| CJK-aware 固定系数 | 一般 | 极低 | 任意 | ✅ |
| tiktoken 本地分词 | 高 | 低 | OpenAI 系列 | ✅ |
| 直接调 API | 极高 | 高 | 任意 | ❌ |
| **本方案（单字表 + 高频词表 + 分段 discount）** | **高** | **极低** | **多模型** | **✅** |

---

## 核心思路

**两阶段设计**：把重计算推到离线，在线路径只剩查表。

### 离线阶段（一次性）

**单字表**：对 CJK 字符块（U+4E00–U+9FFF，20,992 个汉字）逐一调用各模型分词器，记录每字对应的 token 数，存成 20KB 的二进制文件。

**高频词表**：提取语料库中出现频率最高的 5000 个相邻汉字对（最常用的双字词），记录每个词的实际 token 数。这是对 BPE 合并现象的精确修正——"中国"在多数模型里是 1 个 token，但单字表会估为 2 个。

**分段 discount**：按文本中的中文字符比例将文本分为三类：
- `zh`（中文比例 ≥ 60%）：纯中文场景，BPE 合并最多
- `mixed`（10%–60%）：中英混排，常见于技术文章
- `en`（≤ 10%）：纯英文 / 代码

对每类文本单独用真实语料校准 discount 系数（real_tokens / heuristic_estimate 的第 55 百分位），令估算值在统计上略微高于真实值，避免 context-compression 场景的低估。

### 在线阶段（每次调用）

单字表和高频词表在服务启动时一次性加载（约 10 模型共 ~340KB）。每次估算：

1. 判断文本类别（zh / mixed / en），选对应 discount
2. 单遍扫描：CJK 相邻两字优先查高频词表，命中则跳 2 字；否则查单字表；英文按 `ceil(词长/4)`；其余字符按类别加权
3. 累加结果乘 discount 取整

整个过程纯内存，无 IO，无网络，延迟 < 10μs。

---

## 支持的模型

| 模型关键词 | 词表来源 | 编码方案 |
|-----------|---------|---------|
| `qwen` | HuggingFace tokenizer | Qwen3 |
| `qwen2` | HuggingFace tokenizer | Qwen2.5 |
| `deepseek` | HuggingFace tokenizer | DeepSeek-V4 |
| `deepseek-v3` | HuggingFace tokenizer | DeepSeek-V3 |
| `glm` | HuggingFace tokenizer | GLM-5 |
| `glm4` | HuggingFace tokenizer | GLM-4 |
| `minimax` | HuggingFace tokenizer | MiniMax-M2 |
| `kimi` / `moonshot` | HuggingFace tokenizer | Kimi-K2 |
| `gpt-4o` / `o1` / `o3` / `o4` / `gpt-5` | tiktoken | o200k_base |
| `gpt-4` / `gpt-3.5` | tiktoken | cl100k_base |
| `doubao` | Volcano Engine API | 豆包 Seed-1.6 |
| `claude` | Anthropic API | claude-opus-4-8 |

未匹配的模型优先使用豆包词表（粒度偏细，不会低估），再回退到 1.5 token/字。

---

## 快速开始

### 1. 生成词表（离线，一次性）

```bash
cd python
pip install -r requirements.txt

# HuggingFace 类模型需要 HF Token（在 huggingface.co 接受各模型 license 后获取）
export HF_TOKEN=...

# 豆包需要火山引擎 API Key
export ARK_API_KEY=...

# Claude 需要 Anthropic API Key（可选）
export ANTHROPIC_API_KEY=...

python generate_tables.py    # 生成 output/*.bin（单字表，每模型 20KB）
python scrape_corpus.py      # 抓取校准语料 output/corpus.jsonl（~1000 万字）
python generate_bigrams.py   # 生成 output/*.bigram（top-5000 高频词表）
python calculate_discount.py # 生成 output/config.json（分段 discount 系数）
```

### 2. 在 Go 服务里使用（在线）

```go
import (
    "log"
    "os"

    estimator "github.com/justis-xu/token-estimator/go"
)

// 服务启动时加载（TOKEN_TABLES_DIR 指向上一步生成的 output/ 目录）
if err := estimator.Init(os.Getenv("TOKEN_TABLES_DIR")); err != nil {
    log.Fatal(err)
}

// 每次估算
tokens, err := estimator.Estimate(text, "qwen")
if err != nil {
    return err
}
```

### 3. 运行测试

```bash
cd go
go test ./...

# 如果已经生成 python/output，可额外跑真实 golden 精度测试
TOKEN_TABLES_DIR=../python/output go test -v -run TestEstimateAccuracy ./...
```

---

## 精度说明

目标 MAE ≤ 15%。因使用 55 百分位 discount，统计上会略微高估（1–5%），适合 context-compression 场景：**低估会导致 context 溢出，高估只是多截一点**。

校准语料：~3600 条维基百科全文 + 中文新闻 + 技术文章 + 手写样本，约 1000 万字，zh/mixed/en 三类均有覆盖。

精确精度数据在完成最新一轮校准（`generate_golden.py` + `TestEstimateAccuracy`）后更新。

以下场景建议直接调分词 API：
- **精确计费**：按 token 数核算费用，误差不可接受
- **接近窗口上限**：上下文已用到 95% 以上，几百 token 的误差可能导致超限
- **纯英文 / 代码密集内容**：启发式精度不足，tiktoken（GPT）或对应模型 API 更合适

---

## 文件结构

```
python/
  generate_tables.py     # 生成各模型 CJK 单字词表（output/*.bin）
  generate_bigrams.py    # 生成高频词表（output/*.bigram）
  scrape_corpus.py       # 抓取校准语料（output/corpus.jsonl）
  calculate_discount.py  # 计算分段 discount 系数（output/config.json）
  generate_golden.py     # 生成精度验证集（output/golden.jsonl）
  estimate.py            # Python 版估算逻辑（与 Go 保持同步，供校准使用）
  config.py              # 模型配置

  output/                # 生成产物（*.bin、*.bigram、config.json、corpus/golden）

go/
  estimator.go           # 在线估算核心逻辑
  estimator_test.go      # 单元测试 + 精度测试
  go.mod
```

---

## 持续校准

上线后记录 API 返回的 `usage.prompt_tokens`（真实值）和当时的估算值，积累 500 条以上后重跑 `calculate_discount.py`，替换 `config.json` 并重启服务即可生效，不需要改代码。
