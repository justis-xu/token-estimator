"""
Scrape a mixed Chinese corpus for BPE discount calibration.

Sources:
  1. Chinese Wikipedia random summaries (50 articles) — formal Chinese prose
  2. V2EX hot topics (30 entries) — informal tech discussion, Chinese/English mixed
  3. Hand-written message samples (20 entries) — system/user/tool message formats

Output: output/corpus.jsonl
"""

import json
import os
import time
import xml.etree.ElementTree as ET
import requests
from config import OUTPUT_DIR

WIKI_API     = "https://zh.wikipedia.org/api/rest_v1/page/random/summary"
V2EX_API     = "https://www.v2ex.com/api/topics/hot.json"
RUANYF_ATOM  = "https://www.ruanyifeng.com/blog/atom.xml"  # 阮一峰博客, Chinese tech prose


# ---------------------------------------------------------------------------
# Source 1: Chinese Wikipedia
# ---------------------------------------------------------------------------

def fetch_wikipedia(n: int = 50) -> list[dict]:
    results = []
    for i in range(n):
        try:
            r = requests.get(
                WIKI_API, timeout=10,
                headers={"Accept": "application/json",
                         "User-Agent": "token-estimator/1.0"},
            )
            r.raise_for_status()
            text = r.json().get("extract", "").strip()
            if text:
                results.append({"source": "wikipedia", "text": text})
        except Exception as e:
            print(f"  wiki #{i}: {e}")
        time.sleep(0.3)
        if (i + 1) % 10 == 0:
            print(f"  wikipedia: {i+1}/{n}")
    return results


# ---------------------------------------------------------------------------
# Source 2: tech text — V2EX hot topics, with 阮一峰博客 Atom as fallback.
# V2EX's legacy API 1.0 may be offline; if it yields nothing we fall back.
# ---------------------------------------------------------------------------

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
            # strip rough HTML tags
            import re
            content = re.sub(r"<[^>]+>", "", content or "")
            text = (title + "\n" + content).strip()
            if text:
                results.append({"source": "ruanyifeng", "text": text[:800]})
    except Exception as e:
        print(f"  ruanyifeng error: {e}")
    return results


def fetch_tech(n: int = 30) -> list[dict]:
    results = fetch_v2ex(n)
    if len(results) < n:
        results.extend(fetch_ruanyifeng(n - len(results)))
    if not results:
        print("  WARNING: no tech-source text fetched (V2EX + 阮一峰 both failed); "
              "corpus will rely on Wikipedia + manual samples only.")
    return results


# ---------------------------------------------------------------------------
# Source 3: Hand-written message-format samples
# ---------------------------------------------------------------------------

MANUAL_SAMPLES: list[dict] = [
    # system — pure Chinese
    {"source": "manual", "type": "system",
     "text": "你是一个专业的代码助手，帮助用户解决编程问题。请用简洁清晰的语言回答，不要添加多余的解释。"},
    {"source": "manual", "type": "system",
     "text": "你是一个数据分析专家，擅长使用 Python 处理和分析结构化数据。回答时优先提供可运行的代码示例。"},
    {"source": "manual", "type": "system",
     "text": "你是客服助手，负责回答用户关于订单、物流和退款的问题。语气友好，用词简短，不超过 200 字。"},
    # system — Chinese/English mixed
    {"source": "manual", "type": "system",
     "text": "You are an AI assistant. 请用中文回答所有问题，代码块使用 markdown 格式，变量名保持英文。"},
    {"source": "manual", "type": "system",
     "text": "你是一个 Kubernetes 运维专家。当用户提到 pod、deployment、service 等资源时，优先给出 kubectl 命令。"},
    {"source": "manual", "type": "system",
     "text": "系统提示：用户输入的 context 来自上游 RAG 检索，可能包含噪声。请结合 context 回答，若无关联则说明。"},
    # user — questions
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
    # user — instructions
    {"source": "manual", "type": "user",
     "text": "把下面这段代码从 Python 2 迁移到 Python 3，注意处理 unicode 兼容问题。"},
    {"source": "manual", "type": "user",
     "text": "帮我审查这段 SQL，找出潜在的 N+1 查询问题，并给出优化建议。"},
    {"source": "manual", "type": "user",
     "text": "这个 React 组件渲染性能很差，请用 useMemo 和 useCallback 优化，并解释为什么这样改。"},
    # tool_output — JSON with Chinese fields
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
    # whitespace-heavy formats — JSON / Markdown / code
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
- 灰度期间记录真实 usage 和估算值"""},
    {"source": "manual", "type": "code",
     "text": """```go
func BuildPrompt(messages []Message) string {
    var b strings.Builder
    for _, msg := range messages {
        b.WriteString(msg.Role)
        b.WriteString(": ")
        b.WriteString(msg.Content)
        b.WriteString("\n\n")
    }
    return b.String()
}
```"""},
    {"source": "manual", "type": "user",
     "text": "帮我写一个 Dockerfile，基于 python:3.12-slim，安装 requirements.txt 依赖，"
             "支持非 root 用户运行，镜像大小要尽量小。"},
    {"source": "manual", "type": "user",
     "text": "解释一下 vector database 和传统关系型数据库在存储和检索上的核心差异，"
             "什么场景下应该优先选 Milvus 或 Pinecone？"},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "corpus.jsonl")

    corpus: list[dict] = []

    print("Fetching Wikipedia ...")
    corpus.extend(fetch_wikipedia(50))
    print(f"  got {sum(1 for x in corpus if x['source'] == 'wikipedia')} articles")

    print("Fetching tech text (V2EX → 阮一峰 fallback) ...")
    corpus.extend(fetch_tech(30))
    print(f"  got {sum(1 for x in corpus if x['source'] in ('v2ex', 'ruanyifeng'))} entries")

    corpus.extend(MANUAL_SAMPLES)
    print(f"  added {len(MANUAL_SAMPLES)} manual samples")

    with open(out_path, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\ncorpus.jsonl: {len(corpus)} entries → {out_path}")


if __name__ == "__main__":
    main()
