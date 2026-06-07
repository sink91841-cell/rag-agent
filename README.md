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

## 简历亮点

这个项目展示了以下能力：

- **RAG 全链路实现**: 从文档加载、分块、向量化到混合检索的完整 pipeline
- **Agent 架构设计**: 模块化、可插拔的 Agent 编排层
- **向量数据库实战**: Weaviate 的索引、混合检索（向量 + BM25）
- **本地化部署**: 不依赖外部 API 的 embedding 方案，可完全离线运行
- **生产级代码规范**: 类型注解、错误处理、批量操作、连接管理
