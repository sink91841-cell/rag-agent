"""RAG Agent - 完整的检索增强生成管道.

这是整个系统的编排层，负责：
1. 文档摄入（Load → Chunk → Embed → Store）
2. 查询问答（Query → Retrieve → Rerank → Context → Generate）
3. 知识库管理（统计、清空）

不依赖特定 LLM API，通过回调函数实现可插拔的生成层。
"""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from .loader import DocumentLoader
from .chunker import TextChunker, estimate_token_count
from .embedder import Embedder
from .store import VectorStore
from .retriever import Retriever


@dataclass
class AgentStats:
    """知识库统计信息."""

    document_count: int
    chunk_count: int
    total_tokens: int


class RagAgent:
    """RAG 知识库问答智能体.

    使用方式:
        agent = RagAgent()
        agent.ingest("docs/")                        # 摄入文档
        answer = agent.ask("什么是 RAG?")              # 检索（本地）
        # agent.ask 返回上下文，由外部 LLM 做最终生成
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        weaviate_host: str = "localhost",
        weaviate_port: int = 8080,
        weaviate_api_key: str | None = None,
    ):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = Embedder(model_name=model_name)
        self.store = VectorStore(http_host=weaviate_host, http_port=weaviate_port, api_key=weaviate_api_key)
        self.retriever = Retriever(embedder=self.embedder, store=self.store, top_k=top_k)

        self._generator: Callable[[str, str], str] | None = None
        self._connected = False

    def ensure_connected(self) -> None:
        """确保 Weaviate 连接就绪."""
        if not self._connected:
            self.store.connect()
            self._connected = True

    def set_generator(self, func: Callable[[str, str], str]) -> None:
        """设置 LLM 生成回调.

        Args:
            func: (query, context) -> answer 的回调函数
        """
        self._generator = func

    def ingest_file(self, file_path: str) -> int:
        """摄入单个文件."""
        self.ensure_connected()

        docs = self.loader.load(file_path)
        if not docs:
            return 0
        return self._ingest_docs(docs)

    def ingest_directory(self, dir_path: str) -> int:
        """批量摄入目录下所有文档."""
        self.ensure_connected()

        docs = self.loader.load_directory(dir_path)
        if not docs:
            print("未找到支持的文档文件")
            return 0
        return self._ingest_docs(docs)

    def _ingest_docs(self, docs: list) -> int:
        """内部摄入管道: docs → chunks → embed → store."""
        chunks = self.chunker.split_documents(docs)
        if not chunks:
            return 0

        chunks_with_vectors = self.embedder.embed_chunks(chunks)
        count = self.store.index_chunks(chunks_with_vectors)
        return count

    def retrieve(self, query: str) -> list[dict]:
        """执行检索，返回原始结果列表."""
        self.ensure_connected()
        return self.retriever.retrieve(query)

    def ask(self, query: str) -> str:
        """RAG 问答——检索后调用生成器给出答案.

        如果未设置 generator，返回组装好的上下文供外部使用.
        """
        self.ensure_connected()

        results = self.retrieve(query)
        context = self.retriever.build_context(results)

        if self._generator is not None:
            return self._generator(query, context)

        # 没有 LLM 时返回检索上下文 + 来源
        sources = "\n".join(
            f"  {i}. {r['source']} (相似度: {r['score']:.3f})"
            for i, r in enumerate(results, 1)
        )
        return f"📚 检索到 {len(results)} 个相关片段:\n\n{sources}\n\n--- 上下文 ---\n{context[:2000]}"

    def stats(self) -> AgentStats:
        """获取知识库统计."""
        self.ensure_connected()
        return AgentStats(
            document_count=0,  # 需额外追踪
            chunk_count=self.store.count(),
            total_tokens=0,
        )

    def reset(self) -> None:
        """清空知识库."""
        self.ensure_connected()
        self.store.clear()

    def close(self) -> None:
        self.store.disconnect()
        self._connected = False
