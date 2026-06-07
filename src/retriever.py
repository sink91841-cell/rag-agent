"""检索器 - 混合搜索 + 上下文组装."""

from .embedder import Embedder
from .store import VectorStore


class Retriever:
    """检索管道：Query → Embed → Search → Context 组装."""

    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = 5):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        """执行检索，返回最相关的文档片段."""
        query_vector = self.embedder.embed_query(query)
        results = self.store.search(query_vector, top_k=self.top_k)
        return results

    def build_context(self, results: list[dict], max_tokens: int = 3000) -> str:
        """将检索结果组装成 LLM 可用的上下文."""
        parts: list[str] = []
        token_estimate = 0

        for i, result in enumerate(results, 1):
            snippet = f"[来源{i}] {result['source']} (片段{result['chunk_index']})\n{result['content']}"
            # 粗略估算 token
            snippet_tokens = len(snippet) // 2
            if token_estimate + snippet_tokens > max_tokens and parts:
                break
            parts.append(snippet)
            token_estimate += snippet_tokens

        return "\n\n---\n\n".join(parts)
