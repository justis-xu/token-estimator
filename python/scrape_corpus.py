"""
Scrape a mixed Chinese corpus for BPE discount calibration.

Sources:
  1. Chinese Wikipedia random summaries (150 articles) — formal Chinese prose
  2. English Wikipedia random summaries (50 articles) — calibrates English estimation
  3. Tech text: 少数派 RSS → V2EX → 阮一峰 (cascade fallback, 40 entries target)
  4. Hand-written message samples (30 entries) — realistic prompt/response formats

Output: output/corpus.jsonl
"""

import json
import os
import re
import time
import xml.etree.ElementTree as ET
import requests
from config import OUTPUT_DIR

V2EX_API     = "https://www.v2ex.com/api/topics/hot.json"
RUANYF_ATOM  = "https://www.ruanyifeng.com/blog/atom.xml"
SSPAI_RSS    = "https://sspai.com/feed"

# Wikipedia main API — batch 10 random extracts per request, much friendlier on rate limits.
WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
WIKI_BATCH = 10  # grnlimit max = 10


# ---------------------------------------------------------------------------
# Source 1 & 2: Wikipedia (Chinese + English) — batch via action=query
# ---------------------------------------------------------------------------

def fetch_wikipedia(n: int, lang: str = "zh") -> list[dict]:
    source = "zh_wikipedia" if lang == "zh" else "en_wikipedia"
    url = WIKI_API.format(lang=lang)
    params = {
        "action":      "query",
        "generator":   "random",
        "grnnamespace": 0,
        "grnlimit":    WIKI_BATCH,
        "prop":        "extracts",
        "exintro":     True,
        "explaintext": True,
        "exlimit":     WIKI_BATCH,
        "format":      "json",
    }
    headers = {"User-Agent": "token-estimator/1.0 (calibration corpus)"}
    results = []
    backoff = 1.0
    while len(results) < n:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 429:
                print(f"  {source}: 429, backing off {backoff:.0f}s ...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            r.raise_for_status()
            backoff = 1.0
            pages = r.json().get("query", {}).get("pages", {})
            for page in pages.values():
                text = (page.get("extract") or "").strip()
                if text and len(text) > 80:
                    results.append({"source": source, "text": text[:1200]})
            print(f"  {source}: {len(results)}/{n}")
        except Exception as e:
            print(f"  {source}: error {e}")
        time.sleep(1.5)
    return results[:n]


# ---------------------------------------------------------------------------
# Source 3: Tech text cascade — 少数派 → V2EX → 阮一峰
# ---------------------------------------------------------------------------

def fetch_sspai(n: int) -> list[dict]:
    """少数派 RSS — reliable Chinese tech/lifestyle articles."""
    results = []
    try:
        r = requests.get(SSPAI_RSS, timeout=15,
                         headers={"User-Agent": "token-estimator/1.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for item in root.findall(".//item")[:n]:
            title   = item.findtext("title") or ""
            content = item.findtext("description") or item.findtext("content:encoded") or ""
            content = re.sub(r"<[^>]+>", "", content)
            content = re.sub(r"\s{3,}", "\n\n", content)
            text = (title.strip() + "\n\n" + content.strip()).strip()
            if text and len(text) > 80:
                results.append({"source": "sspai", "text": text[:1000]})
    except Exception as e:
        print(f"  sspai error: {e}")
    return results


def fetch_v2ex(n: int) -> list[dict]:
    results = []
    try:
        r = requests.get(
            V2EX_API, timeout=15,
            headers={"User-Agent": "token-estimator/1.0",
                     "Referer": "https://www.v2ex.com/"},
        )
        r.raise_for_status()
        for topic in r.json()[:n]:
            text = (topic.get("title", "") + "\n" + topic.get("content", "")).strip()
            if text:
                results.append({"source": "v2ex", "text": text[:800]})
    except Exception as e:
        print(f"  v2ex error: {e}")
    return results


def fetch_ruanyifeng(n: int) -> list[dict]:
    results = []
    try:
        r = requests.get(RUANYF_ATOM, timeout=15,
                         headers={"User-Agent": "token-estimator/1.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall(".//a:entry", ns)[:n]:
            title   = entry.findtext("a:title", default="", namespaces=ns)
            content = entry.findtext("a:content", default="", namespaces=ns) \
                or entry.findtext("a:summary", default="", namespaces=ns)
            content = re.sub(r"<[^>]+>", "", content or "")
            text = (title + "\n" + content).strip()
            if text:
                results.append({"source": "ruanyifeng", "text": text[:800]})
    except Exception as e:
        print(f"  ruanyifeng error: {e}")
    return results


def fetch_tech(n: int = 40) -> list[dict]:
    """少数派 → V2EX → 阮一峰 cascade; warn if all fail."""
    results = fetch_sspai(n)
    if len(results) < n:
        results.extend(fetch_v2ex(n - len(results)))
    if len(results) < n:
        results.extend(fetch_ruanyifeng(n - len(results)))
    if not results:
        print("  WARNING: all tech sources failed; corpus relies on Wikipedia + manual only.")
    return results[:n]


# ---------------------------------------------------------------------------
# Source 4: Hand-written message-format samples
# Covers realistic prompt patterns: system, user, tool output, code, JSON,
# markdown, long prompts — weighted toward the formats seen in real LLM usage.
# ---------------------------------------------------------------------------

MANUAL_SAMPLES: list[dict] = [
    # --- system prompts (pure Chinese) ---
    {"source": "manual", "type": "system",
     "text": "你是一个专业的代码助手，帮助用户解决编程问题。请用简洁清晰的语言回答，不要添加多余的解释。"},
    {"source": "manual", "type": "system",
     "text": "你是一个数据分析专家，擅长使用 Python 处理和分析结构化数据。回答时优先提供可运行的代码示例。"},
    {"source": "manual", "type": "system",
     "text": "你是客服助手，负责回答用户关于订单、物流和退款的问题。语气友好，用词简短，不超过 200 字。"},
    {"source": "manual", "type": "system",
     "text": "你是一名资深后端工程师，熟悉 Go、Rust 和分布式系统设计。回答时优先考虑性能和可维护性，指出代码中潜在的并发问题。"},
    {"source": "manual", "type": "system",
     "text": "你是一个医学信息助手，只能提供一般性健康科普，不能替代执业医师的诊断意见。每次回答结尾需提醒用户咨询专业医生。"},

    # --- system prompts (Chinese/English mixed) ---
    {"source": "manual", "type": "system",
     "text": "You are an AI assistant. 请用中文回答所有问题，代码块使用 markdown 格式，变量名保持英文。"},
    {"source": "manual", "type": "system",
     "text": "你是一个 Kubernetes 运维专家。当用户提到 pod、deployment、service 等资源时，优先给出 kubectl 命令。"},
    {"source": "manual", "type": "system",
     "text": "系统提示：用户输入的 context 来自上游 RAG 检索，可能包含噪声。请结合 context 回答，若无关联则说明。"},
    {"source": "manual", "type": "system",
     "text": "You are a senior software architect. 在给出技术方案前，先列出 trade-off 分析，并说明你选择该方案的核心理由。不要超过 500 字。"},

    # --- user questions ---
    {"source": "manual", "type": "user",
     "text": "请帮我解释一下 Transformer 架构中 attention 机制的核心原理，最好用类比的方式说明。"},
    {"source": "manual", "type": "user",
     "text": "我的 MySQL 查询很慢，EXPLAIN 显示全表扫描，应该怎么优化？表大概有 500 万行。"},
    {"source": "manual", "type": "user",
     "text": "用 Python 写一个函数，输入一个字符串列表，返回出现频率最高的前 5 个词。"},
    {"source": "manual", "type": "user",
     "text": "Go 里面 goroutine 泄漏的常见原因有哪些？怎么用 pprof 检测？"},
    {"source": "manual", "type": "user",
     "text": "我需要设计一个支持百万级并发的消息推送系统，请给出架构方案和技术选型。"},
    {"source": "manual", "type": "user",
     "text": "帮我写一个 Dockerfile，基于 python:3.12-slim，安装 requirements.txt 依赖，"
             "支持非 root 用户运行，镜像大小要尽量小。"},
    {"source": "manual", "type": "user",
     "text": "解释一下 vector database 和传统关系型数据库在存储和检索上的核心差异，"
             "什么场景下应该优先选 Milvus 或 Pinecone？"},
    {"source": "manual", "type": "user",
     "text": "把下面这段代码从 Python 2 迁移到 Python 3，注意处理 unicode 兼容问题。"},
    {"source": "manual", "type": "user",
     "text": "这个 React 组件渲染性能很差，请用 useMemo 和 useCallback 优化，并解释为什么这样改。"},
    {"source": "manual", "type": "user",
     "text": "Kafka 和 RabbitMQ 的核心区别是什么？我们的场景是日志收集 + 实时计算，选哪个更合适？"},
    {"source": "manual", "type": "user",
     "text": "帮我审查这段 SQL，找出潜在的 N+1 查询问题，并给出优化建议。"},

    # --- long user prompts (context-compression target) ---
    {"source": "manual", "type": "user_long",
     "text": (
         "我正在开发一个多轮对话的 AI 助手，需要实现上下文压缩功能。"
         "当前对话历史如下：\n\n"
         "用户：帮我分析一下这段 Python 代码的性能瓶颈。\n"
         "助手：这段代码主要有三个问题：1) 内层循环重复计算；2) 没有使用批量操作；3) GC 压力大。\n"
         "用户：第一个问题怎么修？\n"
         "助手：可以把重复计算提到循环外，用临时变量缓存结果。\n"
         "用户：改完后还是慢，能用 numpy 优化吗？\n\n"
         "现在用户又发来了一段 500 行的代码，请帮我决定应该保留哪些历史上下文，"
         "以及如何在 4096 token 限制内塞入尽可能多的有效信息。"
     )},
    {"source": "manual", "type": "user_long",
     "text": (
         "请帮我写一份完整的技术方案文档，主题是「基于 LLM 的智能客服系统架构设计」。"
         "要求包括：\n"
         "1. 系统整体架构图（文字描述）\n"
         "2. 核心模块设计（意图识别、知识检索、回答生成、人工介入）\n"
         "3. 关键技术选型及理由（向量数据库、LLM 选型、部署方案）\n"
         "4. 性能指标：首字延迟 < 500ms，P99 < 2s，并发 QPS > 200\n"
         "5. 数据安全与合规要求（用户隐私、敏感词过滤）\n"
         "6. 灰度发布与回滚方案\n\n"
         "文档面向技术评审委员会，需要专业且严谨，预计 2000 字左右。"
     )},

    # --- tool outputs (JSON) ---
    {"source": "manual", "type": "tool",
     "text": json.dumps({
         "status": "success",
         "data": {"用户名": "张三", "订单号": "ORD-20240115-8823",
                  "商品": "MacBook Pro 16寸", "数量": 1, "金额": 19999.00,
                  "收货地址": "北京市朝阳区望京街道10号院3号楼502室"},
     }, ensure_ascii=False)},
    {"source": "manual", "type": "tool",
     "text": json.dumps({
         "tool": "search_documents",
         "results": [
             {"title": "分布式系统一致性协议综述", "score": 0.92,
              "snippet": "Raft 协议通过 leader 选举和日志复制保证强一致性..."},
             {"title": "CAP 定理的工程实践", "score": 0.87,
              "snippet": "在网络分区发生时，系统必须在一致性和可用性之间做出取舍..."},
         ]
     }, ensure_ascii=False)},
    {"source": "manual", "type": "tool",
     "text": json.dumps({
         "function": "execute_sql",
         "query": "SELECT user_id, COUNT(*) as 订单数 FROM orders WHERE created_at > '2024-01-01' GROUP BY user_id",
         "rows_returned": 1523,
         "execution_time_ms": 48,
         "warning": "索引缺失，建议在 created_at 字段添加索引"
     }, ensure_ascii=False)},
    {"source": "manual", "type": "tool",
     "text": json.dumps({
         "action": "send_notification",
         "recipients": ["user_001", "user_002"],
         "message": "您的申请已审批通过，请在3个工作日内完成材料提交。",
         "channel": "站内信",
         "status": "已发送"
     }, ensure_ascii=False)},
    {"source": "manual", "type": "tool",
     "text": json.dumps({
         "tool": "web_search",
         "query": "Go context cancellation best practices",
         "results": [
             {"url": "https://pkg.go.dev/context", "title": "context package - Go", "snippet":
              "Package context defines the Context type, which carries deadlines, cancellation signals..."},
             {"url": "https://blog.golang.org/context", "title": "Go Concurrency Patterns: Context",
              "snippet": "In Go servers, each incoming request is handled in its own goroutine..."},
         ]
     }, ensure_ascii=False)},

    # --- formatted text (JSON pretty / Markdown / code) ---
    {"source": "manual", "type": "json_pretty",
     "text": """{
  "model": "qwen-plus",
  "messages": [
    {
      "role": "system",
      "content": "你是一个代码审查助手，优先指出会导致线上事故的问题。"
    },
    {
      "role": "user",
      "content": "请审查下面的 diff，并按严重程度排序。"
    }
  ],
  "temperature": 0.2,
  "stream": false
}"""},
    {"source": "manual", "type": "markdown",
     "text": """# 发布检查清单

| 指标 | 当前值 | 阈值 | 结论 |
|------|--------|------|------|
| p95 延迟 | 128ms | 200ms | 通过 |
| 错误率 | 0.03% | 0.10% | 通过 |
| token 估算误差 | 11.8% | 15.0% | 通过 |

- 确认 `TOKEN_TABLES_DIR` 指向最新 output 目录
- 确认 config.json 已随词表一起发布
- 灰度期间记录真实 usage 和估算值
- 上线后连续观察 3 天误差曲线"""},
    {"source": "manual", "type": "markdown",
     "text": """## API 设计规范

### 请求格式
所有接口使用 `Content-Type: application/json`，认证通过 `Authorization: Bearer <token>` 传递。

### 错误码规范
| 错误码 | 含义 | 处理建议 |
|--------|------|---------|
| 4001 | token 无效或过期 | 重新登录获取新 token |
| 4003 | 权限不足 | 检查用户角色配置 |
| 4290 | 触发限流 | 指数退避后重试 |
| 5001 | 内部服务错误 | 上报监控，勿重试 |

### 分页约定
统一使用 `cursor` 分页，禁止使用 `offset/limit` 在大数据集上分页。"""},
    {"source": "manual", "type": "code_go",
     "text": """```go
func BuildPrompt(ctx context.Context, messages []Message, maxTokens int) (string, error) {
    est, err := estimator.Estimate(ctx, messages)
    if err != nil {
        return "", fmt.Errorf("estimate tokens: %w", err)
    }
    if est > maxTokens {
        messages = truncateMessages(messages, maxTokens)
    }
    var b strings.Builder
    for _, msg := range messages {
        fmt.Fprintf(&b, "%s: %s\\n\\n", msg.Role, msg.Content)
    }
    return b.String(), nil
}
```"""},
    {"source": "manual", "type": "code_python",
     "text": """```python
def chunk_text(text: str, max_tokens: int, model: str = "qwen") -> list[str]:
    \"\"\"Split text into chunks, each fitting within max_tokens.\"\"\"
    words = text.split()
    chunks, current, count = [], [], 0
    for word in words:
        t = estimate_tokens(word, model)
        if count + t > max_tokens and current:
            chunks.append(" ".join(current))
            current, count = [word], t
        else:
            current.append(word)
            count += t
    if current:
        chunks.append(" ".join(current))
    return chunks
```"""},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "corpus.jsonl")

    corpus: list[dict] = []

    print("Fetching Chinese Wikipedia (150 articles) ...")
    zh_wiki = fetch_wikipedia(150, lang="zh")
    corpus.extend(zh_wiki)
    print(f"  got {len(zh_wiki)} zh-wikipedia articles")

    print("Fetching English Wikipedia (50 articles) ...")
    en_wiki = fetch_wikipedia(50, lang="en")
    corpus.extend(en_wiki)
    print(f"  got {len(en_wiki)} en-wikipedia articles")

    print("Fetching tech text (少数派 → V2EX → 阮一峰, target 40) ...")
    tech = fetch_tech(40)
    corpus.extend(tech)
    by_src = {}
    for x in tech:
        by_src[x["source"]] = by_src.get(x["source"], 0) + 1
    print(f"  got {len(tech)} entries: {by_src}")

    corpus.extend(MANUAL_SAMPLES)
    print(f"  added {len(MANUAL_SAMPLES)} manual samples")

    total_chars = sum(len(x["text"]) for x in corpus)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\ncorpus.jsonl: {len(corpus)} entries, {total_chars:,} chars → {out_path}")


if __name__ == "__main__":
    main()
