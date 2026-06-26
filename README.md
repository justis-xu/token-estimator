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
| **本方案（查表 + discount）** | **高** | **极低** | **多模型** | **✅** |

---

## 核心思路

**两阶段设计**：把重计算推到离线，在线路径只剩查表。

**离线阶段（一次性）**：用各模型真实分词器处理 CJK 字符表里 20,992 个汉字，记录每字对应的 token 数，存成二进制文件（每模型 20KB）。同时从真实语料校准一个全局 discount 系数，用于修正 BPE 跨字合并带来的系统性高估。

**在线阶段（每次调用）**：词表在服务启动时一次性加载入内存（当前 10 个模型合计约 240KB）。估算时单遍扫描文本，CJK 字符查表累加，英文按 `ceil(词长/4)` 估算，最后乘 discount 系数取整。整个过程纯内存操作，无 IO，无网络。

词表和系数文件可以独立替换，替换后重启生效，不需要重新编译。启动时要求词表目录至少包含一个 `.bin` 文件和 `config.json`；单个模型没有专属词表时，在线估算会自动 fallback，不会让请求直接失败。`config.json` 里同时保存每个模型的 `discount` 和 `weights`，字符级权重会优先按模型读取，没有对应模型时再回退到 `weights.default`。

---

## 支持的模型

| 模型关键词 | 词表来源 | 编码方案 |
|-----------|---------|---------|
| `qwen` | HuggingFace tokenizer | Qwen3 |
| `deepseek` | HuggingFace tokenizer | DeepSeek-V4 |
| `glm` | HuggingFace tokenizer | GLM-5 |
| `minimax` | HuggingFace tokenizer | MiniMax-M2 |
| `kimi` / `moonshot` | HuggingFace tokenizer | Kimi-K2 |
| `gpt-4o` / `o1` / `o3` / `o4` / `gpt-5` | tiktoken | o200k_base |
| `gpt-4` / `gpt-3.5` | tiktoken | cl100k_base |
| `qwen2` | HuggingFace tokenizer | Qwen2.5 |
| `deepseek-v3` | HuggingFace tokenizer | DeepSeek-V3 |
| `glm4` | HuggingFace tokenizer | GLM-4 |
| `doubao` | Volcano Engine API | 豆包 |
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

# Claude 需要 Anthropic API Key
export ANTHROPIC_API_KEY=...

python generate_tables.py     # 生成 output/*.bin
python scrape_corpus.py       # 抓取校准语料 output/corpus.jsonl
python calculate_discount.py  # 生成 output/config.json
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
TOKEN_TABLES_DIR=../python/output go test -v ./...
```

---

## 精度说明

在中文为主的混合文本（维基百科 + 中英混合技术文章）上，平均绝对误差（MAE）目标 ≤ 15%。正式中文文本通常 5%~8%，中英混合约 10%~15%。

当前这批验证集是 65 条语料、约 1.9 万字符，覆盖中文/英文 Wikipedia、V2EX、阮一峰博客、Python/Go/MDN 文档片段和手写消息样本。最近一次跑出来的 MAE 如下：

- `gpt-4o`: `9.7%`
- `gpt-4`: `10.0%`
- `deepseek` / `deepseek-v3`: `9.7%`
- `glm` / `glm4`: `11.3%` 到 `11.4%`
- `qwen`: `11.4%`
- `qwen2`: `12.0%`
- `minimax`: `11.4%`
- `kimi`: `11.9%`

以下场景建议直接调分词 API：

- **精确计费**：按 token 数核算费用，误差不可接受
- **接近窗口上限**：上下文已用到 95% 以上，几百 token 的误差可能导致超限
- **纯英文 / 代码密集内容**：启发式精度不足，tiktoken（GPT）或对应模型 API 更合适

---

## 文件结构

```
python/
  generate_tables.py    # 生成各模型 CJK 词表
  scrape_corpus.py      # 抓取校准语料
  calculate_discount.py # 计算 BPE discount 系数
  estimate.py           # Python 版估算逻辑（与 Go 保持同步，供校准使用）
  config.py             # 模型配置
  output/               # 生成产物（.bin × 10 + config.json + corpus/golden）

go/
  estimator.go          # 在线估算核心逻辑
  estimator_test.go     # 单元测试 + 精度测试
  go.mod
```

---

## 持续校准

上线后记录 API 返回的 `usage.prompt_tokens`（真实值）和当时的估算值，积累 500 条以上后重跑 `calculate_discount.py`，替换 `config.json` 并重启服务即可生效，不需要改代码。
