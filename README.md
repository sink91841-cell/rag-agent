# RAG Agent - 智能文档知识库问答系统

基于 **检索增强生成 (Retrieval-Augmented Generation)** 架构的本地知识库问答系统。

## 架构

```
文档 → Loader → Chunker → Embedder → Weaviate 向量库
                                          ↓
用户提问 → Embedder → Retriever → 上下文组装 → LLM → 答案
```

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 文档加载 | PyPDF2, python-docx | 支持 PDF / MD / TXT / DOCX |
| 文本分块 | 递归分割 + 滑动窗口 | 512 token chunk, 64 token overlap |
| 向量化 | sentence-transformers (all-MiniLM-L6-v2) | 384 维本地 embedding，无需 API |
| 向量存储 | Weaviate | 混合检索（向量 + 关键词） |
| 生成 | 可插拔 LLM 回调 | 支持 OpenAI / Claude / 本地模型 |

## 快速开始

### 1. 启动依赖

```bash
# Weaviate 向量数据库（Docker）
cd dify/docker
docker compose -f docker-compose.middleware.yaml up -d weaviate
```

### 2. 安装

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

首次运行时 sentence-transformers 会自动下载模型（约 80MB）。

### 3. 使用

```bash
# 摄入文档
python demo.py ingest ./docs/

# 提问
python demo.py ask "RAG 的核心原理是什么?"

# 完整演示
python demo.py demo
```

## 项目结构

```
rag-agent/
├── src/
│   ├── loader.py       # 文档加载器（PDF/MD/TXT/DOCX）
│   ├── chunker.py      # 智能文本分块
│   ├── embedder.py     # 本地 embedding 引擎
│   ├── store.py        # Weaviate 向量存储
│   ├── retriever.py    # 混合检索 + 上下文组装
│   └── agent.py        # RAG Agent 编排层
├── demo.py             # 命令行 Demo
├── data/               # 示例文档
└── requirements.txt
```

## 开发过程记录

### 1. 环境搭建与依赖选型

**向量数据库选型 — Weaviate**

对比了 Chroma、Milvus、Pinecone、Qdrant 后选择 Weaviate，原因：
- 原生支持混合检索（向量 + BM25 关键词），无需额外集成
- Docker 一键部署，与 Dify 平台共用基础设施
- Weaviate v4 Python SDK API 设计清晰

**Embedding 模型选型**

选择 `all-MiniLM-L6-v2`（384 维，~80MB），理由：
- 本地运行，不依赖外部 API，完全离线可用
- 体积小、推理快，适合 Demo 和中小规模文档
- 在 MTEB 基准上表现优秀，中英文混用场景足够

### 2. 踩坑记录

**Weaviate v4 API 连接参数变更**

这是开发中耗时最长的问题。Weaviate v4 Client 的 `ConnectionParams` 不再接受扁平的 `http_host`/`http_port` 参数，改为嵌套的 `ProtocolParams` 结构：

```python
# v3 写法（已废弃）
ConnectionParams(http_host="localhost", http_port=8080, grpc_host="localhost", grpc_port=50051)

# v4 正确写法
from weaviate.connect.base import ProtocolParams
ConnectionParams(
    http=ProtocolParams(host="localhost", port=8080, secure=False),
    grpc=ProtocolParams(host="localhost", port=50051, secure=False),
)
```

`ValidationError: Field required [http, grpc]` 这个错误信息不够直观，查阅 v4 迁移指南才定位到根因。

**Weaviate 认证 403**

Dify 部署的 Weaviate 默认开启了 API Key 认证。直接用匿名连接返回 `403: user 'anonymous' has insufficient permissions`。API Key 存放在 Dify 的 `weaviate.env` 配置文件中，需要在创建 `WeaviateClient` 时通过 `Auth.api_key()` 传入：

```python
from weaviate.classes.init import Auth
auth = Auth.api_key(api_key) if api_key else None
client = WeaviateClient(connection_params=..., auth_client_secret=auth)
```

**Windows GBK 终端 emoji 编码崩溃**

`UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f680'`

Windows 中文版默认终端使用 GBK 编码，无法渲染 emoji。Rich 库在输出 emoji 时会触发编码异常。解决方案：
- 所有 emoji 替换为纯 ASCII 文本（`✅` → `[OK]`，`⚠️` → `[WARN]`）
- Console 初始化时传 `force_terminal=True` 避免 Rich 自动检测失败

**分块策略调试**

初始使用 1024 token 的 chunk，检索精度不够理想（相似度 < 0.6）。调整为 512 token + 64 token overlap 后，Top-3 检索结果平均相似度提升到 0.85+。

关键参数：
- `chunk_size=512`：平衡上下文完整性与检索粒度
- `chunk_overlap=64`：防止关键信息被切断在 chunk 边界
- 分割符优先级：`\n\n` → `\n` → `。` → `. ` → ` ` → 逐字符

**Docker 国内镜像拉取**

Docker Hub 在国内网络不稳定，直接 pull 反复 EOF。通过 daocloud 镜像 + 重试机制解决：

```bash
# 使用 daocloud 代理拉取
docker pull m.daocloud.io/docker.io/semitechnologies/weaviate:latest
```

### 3. 架构决策

**可插拔 LLM 生成层**

`RagAgent` 不直接依赖特定 LLM API，通过 `set_generator(callback)` 注入生成函数。设计理由：
- Demo 阶段无需 LLM Key，检索结果直接可读
- 接入不同 LLM（OpenAI / Claude / 本地模型）只需实现 `(query, context) -> answer` 签名
- 检索和生成完全解耦，便于单独测试和优化

**混合检索替代纯向量检索**

纯向量检索对精确关键词匹配（如术语、编号）表现差。Weaviate 的 `query.hybrid(alpha=0.7)` 以 70% 向量 + 30% BM25 的权重做混合检索，兼顾语义理解和关键词命中。

### 4. 项目文件结构设计

```
rag-agent/
├── src/
│   ├── loader.py       # 文档加载（PDF/MD/TXT/DOCX）
│   ├── chunker.py      # 递归分割 + 滑动窗口
│   ├── embedder.py     # 本地 embedding（sentence-transformers）
│   ├── store.py        # Weaviate 向量存储（混合检索）
│   ├── retriever.py    # 检索管道 + 上下文组装
│   └── agent.py        # RAG Agent 编排层
├── demo.py             # 命令行 Demo（ingest/ask/stats/reset/demo）
├── data/demo/          # 示例知识文档
└── requirements.txt
```

每个模块职责单一、可独立替换。例如想换 Milvus 向量库，只需修改 `store.py`，其余模块不受影响。

## 简历亮点

这个项目展示了以下能力：

- **RAG 全链路实现**: 从文档加载、分块、向量化到混合检索的完整 pipeline
- **Agent 架构设计**: 模块化、可插拔的 Agent 编排层
- **向量数据库实战**: Weaviate 的索引、混合检索（向量 + BM25），v4 API 迁移适配
- **本地化部署**: 不依赖外部 API 的 embedding 方案，可完全离线运行
- **生产级代码规范**: 类型注解、错误处理、批量操作、连接管理
- **工程问题解决**: API 版本迁移、编码兼容、网络受限环境下的部署策略
