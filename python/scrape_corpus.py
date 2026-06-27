"""
Scrape a mixed Chinese corpus for BPE discount calibration.

Sources (all parallel):
  Wikipedia: zh (1000 quality articles ≥500 chars) + en (50)
  News RSS:  人民日报(2) 中新社 环球时报 凤凰 新浪 光明日报 财新
  Tech RSS:  少数派 虎嗅 36kr 爱范儿 极客公园 InfoQ开源中国 V2EX 阮一峰
  Manual:    ~52 hand-written prompt/response samples

Output: output/corpus.jsonl  (incremental, resumable)
"""

import json
import os
import re
import time
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from config import OUTPUT_DIR

ZH_TEXT_CAP  = 8000
EN_TEXT_CAP  = 2000
WIKI_MIN_LEN = 500

WIKI_API   = "https://{lang}.wikipedia.org/w/api.php"
WIKI_BATCH = 10
V2EX_API   = "https://www.v2ex.com/api/topics/hot.json"
RUANYF_ATOM = "https://www.ruanyifeng.com/blog/atom.xml"

# High-quality HF datasets for Chinese/English text
# (dataset_name, config_name, split, source_name, n_items)
HF_DATASETS = [
    ("wangrui6/Zhihu-KOL", "default", "train", "hf_zhihu", 1000),                 # Chinese Q&A
    ("Skylion007/openwebtext", "default", "train", "hf_openwebtext", 300),        # English web text
]


# ---------------------------------------------------------------------------
# Source 1 & 2: Wikipedia (Chinese + English) — batch via action=query
# ---------------------------------------------------------------------------

def fetch_hf_dataset(dataset_name: str, config_name: str, split: str, source: str, n: int, writer=None) -> list[dict]:
    """Fetch n items from a Hugging Face dataset via streaming."""
    results = []
    try:
        from datasets import load_dataset
        print(f"  {source}: loading {dataset_name} ({split})...")
        # Load dataset dynamically based on whether it needs a config name
        if config_name and config_name != "default":
            ds = load_dataset(dataset_name, config_name, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)
        
        for row in ds:
            # Handle different column names (text vs INSTRUCTION/RESPONSE)
            text = row.get("text", "")
            if not text and "INSTRUCTION" in row and "RESPONSE" in row:
                text = row["INSTRUCTION"] + "\n\n" + row["RESPONSE"]
                
            # basic clean up and length filter
            text = re.sub(r"\s{3,}", "\n\n", text).strip()
            if len(text) > 200:
                cap = EN_TEXT_CAP if "en" in source or "openwebtext" in source else ZH_TEXT_CAP
                item = {"source": source, "text": text[:cap]}
                results.append(item)
                if writer:
                    writer(item)
            if len(results) >= n:
                break
        print(f"  {source}: {len(results)}/{n}")
    except Exception as e:
        print(f"  {source}: error {e}")
    return results

def fetch_wikipedia(n: int, lang: str = "zh", writer=None) -> list[dict]:
    """Fetch Wikipedia using HF datasets instead of Wikimedia API."""
    source = "zh_wikipedia" if lang == "zh" else "en_wikipedia"
    config = "20231101.zh" if lang == "zh" else "20231101.en"
    return fetch_hf_dataset("wikimedia/wikipedia", config, "train", source, n, writer)


# ---------------------------------------------------------------------------
# Generic RSS fetcher
# ---------------------------------------------------------------------------

def fetch_rss(url: str, source: str, n: int) -> list[dict]:
    """Fetch up to n items from an RSS/Atom feed."""
    results = []
    try:
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
        # RSS <item> or Atom <entry>
        items = root.findall(".//item") or root.findall(
            ".//{http://www.w3.org/2005/Atom}entry")
        for item in items[:n]:
            title = (item.findtext("title") or
                     item.findtext("{http://www.w3.org/2005/Atom}title") or "")
            body  = (item.findtext("description") or
                     item.findtext("{http://www.w3.org/2005/Atom}content") or
                     item.findtext("{http://www.w3.org/2005/Atom}summary") or "")
            body  = re.sub(r"<[^>]+>", "", body)
            body  = re.sub(r"\s{3,}", "\n\n", body)
            text  = (title.strip() + "\n\n" + body.strip()).strip()
            if text and len(text) > 80:
                results.append({"source": source, "text": text[:ZH_TEXT_CAP]})
        print(f"  {source}: {len(results)} entries")
    except Exception as e:
        print(f"  {source} ({url}): {e}")
    return results


def fetch_v2ex(n: int) -> list[dict]:
    results = []
    try:
        r = requests.get(V2EX_API, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                                  "Referer": "https://www.v2ex.com/"})
        r.raise_for_status()
        for topic in r.json()[:n]:
            text = (topic.get("title", "") + "\n" + topic.get("content", "")).strip()
            if text:
                results.append({"source": "v2ex", "text": text[:ZH_TEXT_CAP]})
        print(f"  v2ex: {len(results)} entries")
    except Exception as e:
        print(f"  v2ex: {e}")
    return results


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

    # --- additional pure-Chinese samples for zh calibration ---
    {"source": "manual", "type": "zh_prose",
     "text": "人工智能技术的快速发展正在深刻改变各行各业的运作方式。从制造业的智能化生产到医疗领域的辅助诊断，再到金融行业的风险控制，大模型的应用已经渗透到经济社会的方方面面。然而，与此同时，关于数据安全、隐私保护和算法偏见的担忧也日益凸显，如何在技术创新与伦理监管之间找到平衡，成为各国政府和企业面临的共同挑战。"},
    {"source": "manual", "type": "zh_prose",
     "text": "在城市化进程不断加快的背景下，交通拥堵问题愈发严峻。以北京、上海为代表的特大城市，每逢早晚高峰，主干道上车辆绵延数公里，通勤时间动辄超过一小时。为此，各地相继推出限号出行、错峰上下班等措施，同时加大对公共交通基础设施的投入，力图从源头上缓解道路压力。"},
    {"source": "manual", "type": "zh_prose",
     "text": "中医药文化是中华民族几千年来与疾病斗争的智慧结晶。针灸、推拿、中药方剂等传统疗法，经过历代医家的实践积累与理论总结，形成了一套独特的理论体系。近年来，随着国际社会对传统医学的重视程度不断提升，中医药正走向世界舞台，在全球卫生治理中发挥越来越重要的作用。"},
    {"source": "manual", "type": "zh_prose",
     "text": "量子计算是当前信息技术领域最受关注的前沿方向之一。不同于经典计算机依赖比特的0和1进行运算，量子计算机利用量子叠加和量子纠缠等特性，能够在特定问题上实现指数级的加速。目前，谷歌、IBM、百度等科技巨头均已投入大量资源开展量子计算研究，预计在未来十年内将在密码学、药物研发等领域取得突破性进展。"},
    {"source": "manual", "type": "zh_prose",
     "text": "乡村振兴战略实施以来，农村面貌发生了翻天覆地的变化。硬化道路通到了家门口，自来水和天然气进入了千家万户，电商平台让农产品走出了大山。越来越多的年轻人选择返乡创业，带回了资金、技术和新理念，为农村经济注入了新的活力。与此同时，乡村旅游、民宿经济等新业态蓬勃兴起，成为农民增收的重要渠道。"},
    {"source": "manual", "type": "zh_dialogue",
     "text": "用户：我的孩子今年上小学三年级，最近数学成绩一直在下滑，不知道该怎么办。\n\n助手：小学三年级数学成绩下滑是很常见的情况，这个阶段数学从简单的加减法过渡到了乘除和分数，对很多孩子来说是一个重要的分水岭。建议您先和孩子沟通，了解他在哪些具体知识点上感到困难，是乘法口诀、应用题还是几何图形。找准薄弱点之后，可以每天拿出30分钟针对性练习，配合生活中的实际场景帮助理解，比如买东西找零、分蛋糕等。"},
    {"source": "manual", "type": "zh_dialogue",
     "text": "用户：公司要求我们用大模型做一个内部知识库问答系统，但我不知道从哪里开始。\n\n助手：搭建内部知识库问答系统一般分四个步骤。第一步是文档处理，把公司的 PDF、Word、Wiki 等文档解析成纯文本，按段落或页面切分成适当大小的块。第二步是向量化，用嵌入模型（比如 text-embedding-3-small 或 bge-m3）把文本块转成向量，存入向量数据库如 Milvus 或 Chroma。第三步是检索，用户提问时先向量化问题，检索最相似的 K 个文本块作为上下文。第四步是生成，把问题和检索到的上下文拼成 prompt，调用大模型生成回答。"},
    {"source": "manual", "type": "zh_news",
     "text": "国家统计局今日发布数据显示，上半年全国居民消费价格指数（CPI）同比上涨0.3%，工业生产者出厂价格指数（PPI）同比下降2.1%。专家分析认为，当前物价总体保持平稳，但内需偏弱的问题仍需关注。国家发展改革委表示，下半年将继续实施积极的财政政策和稳健的货币政策，综合运用多种手段提振居民消费信心，推动经济持续回升向好。"},
    {"source": "manual", "type": "zh_news",
     "text": "本市今日召开城市更新工作推进会议，会议部署了老旧小区改造、背街小巷整治、历史文化街区保护等重点任务。市住房和城乡建设局局长在会上表示，今年计划完成800个老旧小区的改造工作，惠及居民约15万户。改造内容包括屋顶防水、外墙保温、加装电梯、完善停车设施及绿化提升等，确保改造后居民生活质量得到切实提升。"},
    {"source": "manual", "type": "zh_technical",
     "text": "微服务架构在提升系统可扩展性的同时，也带来了服务治理的复杂性。在高并发场景下，服务间的调用链路可能涉及数十个微服务，一旦某个服务出现延迟或故障，容易引发连锁反应，导致整个系统雪崩。因此，在设计微服务架构时，必须重视熔断机制、限流降级和链路追踪等可靠性手段。常用的解决方案包括 Sentinel、Hystrix 等熔断框架，以及 SkyWalking、Jaeger 等分布式追踪系统。"},
    {"source": "manual", "type": "zh_technical",
     "text": "数据库索引是提升查询性能的核心手段，但索引并非越多越好。每个索引都会占用额外的磁盘空间，并在数据写入时带来维护开销。合理的索引策略应基于实际的查询模式：对于高频的等值查询，B树索引是首选；对于范围查询，需要考虑索引的选择性；对于多列查询，联合索引的列顺序至关重要，应将选择性高的列放在前面。此外，定期分析慢查询日志，找出缺失索引的高耗时SQL，是数据库调优的重要环节。"},
    {"source": "manual", "type": "zh_prose",
     "text": '深海探测是人类认识地球的重要途径。海洋覆盖地球表面约71%的面积，平均深度超过3600米，但至今人类直接探索过的海底面积不足5%。深海环境极端恶劣，高压、低温、黑暗使得探测设备的研发面临巨大挑战。近年来，我国自主研发的"奋斗者"号全海深载人潜水器成功坐底马里亚纳海沟，刷新了中国载人深潜记录，标志着我国在深海探测领域取得了重大突破。'},
    {"source": "manual", "type": "zh_prose",
     "text": '阅读是获取知识、拓展视野的重要方式，但在信息爆炸的时代，如何高效阅读成为一项值得学习的技能。主题阅读法是一种高效的策略：围绕某一主题，同时阅读多本相关书籍，在对比和综合中形成系统性认知。笔记法同样重要，费曼技术建议用自己的话解释所学内容，通过"教学"来检验真正的理解程度。此外，番茄工作法可以帮助保持专注，避免长时间阅读带来的疲劳和注意力分散。'},
    {"source": "manual", "type": "zh_technical",
     "text": "大语言模型的推理效率是制约其大规模部署的关键瓶颈。KV Cache 技术通过缓存注意力层的键值矩阵，避免重复计算，显著降低了自回归解码的延迟。量化技术（INT8、INT4）通过降低权重精度来压缩模型大小和加速推理，但可能带来一定的精度损失。投机解码（Speculative Decoding）利用小模型预生成候选序列，再由大模型批量验证，在保证输出质量的前提下大幅提升吞吐量。这些优化技术的组合使用，使得在消费级硬件上部署百亿参数模型成为可能。"},
    {"source": "manual", "type": "zh_dialogue",
     "text": "用户：最近看新闻说美联储要降息，这对我们普通人有什么影响？\n\n助手：美联储降息对普通人的影响主要体现在几个方面。首先是贷款成本，降息后银行贷款利率通常会跟着下降，如果你有房贷或车贷，再融资的成本会降低；如果你计划购房，月供压力也会有所减轻。其次是储蓄收益，存款利率会随之下降，存在银行的钱收益会减少，这可能促使部分人把资金转向股市或其他投资渠道。第三是汇率，降息通常会导致美元走弱，如果你有美元资产或涉及跨境业务，需要关注汇率变动带来的风险。"},
    {"source": "manual", "type": "zh_prose",
     "text": "食品安全是关系人民群众身体健康和生命安全的重大民生问题。近年来，我国持续完善食品安全法律法规体系，建立了从农田到餐桌的全链条监管机制。农业农村部加强农药残留、兽药残留的监测抽查，市场监管总局对食品生产加工环节实施严格的许可制度，海关对进口食品实行批批检验。与此同时，社会共治的理念逐步深入人心，消费者投诉举报渠道不断畅通，企业主体责任意识不断增强。"},
    {"source": "manual", "type": "zh_technical",
     "text": "容器化技术的普及极大地简化了软件的部署和运维工作。Docker 通过将应用程序及其依赖打包成镜像，实现了「一次构建，到处运行」的目标，解决了不同环境之间的兼容性问题。Kubernetes 则在此基础上提供了容器编排能力，自动处理容器的调度、扩缩容、健康检查和故障恢复。在实际生产环境中，通常还需要配合 Helm 进行包管理、Prometheus 进行监控告警、Istio 进行服务网格治理，共同构成完整的云原生技术栈。"},
    {"source": "manual", "type": "zh_prose",
     "text": "非物质文化遗产是一个民族文化身份和历史记忆的重要载体。剪纸、京剧、昆曲、皮影戏……这些凝聚着先人智慧和审美情趣的艺术形式，正面临着传承断层的困境。如何在现代社会中激活传统文化的生命力，是摆在我们面前的重要课题。一些成功的探索令人振奋：故宫文创将传统元素与现代设计相融合，让古老的文物以崭新的面貌走进年轻人的日常生活；一些非遗传承人借助短视频平台，让更多人了解和爱上了传统手工艺。"},
    {"source": "manual", "type": "zh_dialogue",
     "text": "用户：我想用Python写一个爬虫，抓取某电商平台的商品价格，但不知道从哪里开始。\n\n助手：写一个基础的价格爬虫，可以分三步走。第一步，分析目标页面。用浏览器打开商品页，按F12打开开发者工具，切到Network标签，刷新页面，看价格数据是直接在HTML里，还是通过XHR接口返回的JSON。第二步，根据情况选择工具。如果是静态HTML，用requests+BeautifulSoup就够了；如果是动态渲染的JavaScript页面，需要用Selenium或Playwright模拟浏览器。第三步，解析数据并存储。找到价格元素的CSS选择器或XPath，提取数值后存入CSV或数据库。注意要设置合理的请求间隔，避免被封IP。"},
    {"source": "manual", "type": "zh_prose",
     "text": "碳达峰碳中和是我国应对气候变化、推动绿色发展的重大战略决策。实现双碳目标，需要能源结构、产业结构、交通运输、城乡建设等多个领域的系统性变革。可再生能源是替代化石燃料的核心路径，风电、光伏装机规模的持续扩大，正在重塑我国的能源版图。储能技术的突破至关重要，只有解决了可再生能源间歇性、波动性的问题，才能真正实现对传统电力系统的替代。此外，碳捕获与封存技术、氢能的规模化应用也是实现深度脱碳不可或缺的手段。"},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "corpus.jsonl")

    # --- resume: count existing entries per source ---
    have: dict[str, int] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    src = json.loads(line).get("source", "?")
                    have[src] = have.get(src, 0) + 1
                except Exception:
                    pass
        if have:
            print(f"Resuming — found {sum(have.values())} existing entries: {have}")

    # thread-safe incremental writer
    _lock = threading.Lock()
    _fh   = open(out_path, "a", encoding="utf-8")

    def write(item: dict) -> None:
        with _lock:
            _fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            _fh.flush()

    def write_all(items: list[dict]) -> None:
        for item in items:
            write(item)

    # build task list
    tasks: list[tuple] = []  # (fn, *args)

    # Wikipedia zh (2000, min 500 chars)
    zh_need = max(0, 2000 - have.get("zh_wikipedia", 0))
    if zh_need > 0:
        tasks.append(("wiki_zh", zh_need))
    # Wikipedia en (200)
    en_need = max(0, 200 - have.get("en_wikipedia", 0))
    if en_need > 0:
        tasks.append(("wiki_en", en_need))

    # HF additional datasets
    for ds_name, cfg, split, source, n in HF_DATASETS:
        need = max(0, n - have.get(source, 0))
        if need > 0:
            tasks.append(("hf_ds", ds_name, cfg, split, source, need))

    # V2EX (JSON API)
    if have.get("v2ex", 0) < 30:
        tasks.append(("v2ex", 30 - have.get("v2ex", 0)))

    # 阮一峰 Atom
    if have.get("ruanyifeng", 0) < 20:
        tasks.append(("ruanyifeng", 20 - have.get("ruanyifeng", 0)))

    # manual samples (one-shot)
    if have.get("manual", 0) == 0:
        write_all(MANUAL_SAMPLES)
        print(f"  manual: wrote {len(MANUAL_SAMPLES)} samples")

    print(f"\nRunning {len(tasks)} source tasks in parallel ...\n")

    def run_task(task):
        kind = task[0]
        if kind == "wiki_zh":
            fetch_wikipedia(task[1], "zh", write)
        elif kind == "wiki_en":
            fetch_wikipedia(task[1], "en", write)
        elif kind == "hf_ds":
            _, ds_name, cfg, split, source, n = task
            fetch_hf_dataset(ds_name, cfg, split, source, n, write)
        elif kind == "v2ex":
            write_all(fetch_v2ex(task[1]))
        elif kind == "ruanyifeng":
            write_all(fetch_rss(RUANYF_ATOM, "ruanyifeng", task[1]))

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(run_task, t): t for t in tasks}
        for f in as_completed(futures):
            if f.exception():
                print(f"  task {futures[f]} error: {f.exception()}")

    _fh.close()

    # final stats
    count, total = 0, 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                count += 1
                total += len(json.loads(line).get("text", ""))
            except Exception:
                pass
    print(f"\ncorpus.jsonl: {count} entries, {total:,} chars → {out_path}")


if __name__ == "__main__":
    main()
