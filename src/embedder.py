"""向量化引擎 - 使用本地 sentence-transformers 模型做 embedding."""

from typing import Sequence

from .chunker import Chunk


class Embedder:
    """本地 Embedding 引擎，无需外部 API."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量将文本转为向量."""
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """将查询文本转为向量."""
        embedding = self.model.encode([query], normalize_embeddings=True)
        return embedding[0].tolist()

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[tuple[Chunk, list[float]]]:
        """一次性对所有 chunk 做 embedding，返回 (chunk, vector) 对."""
        texts = [chunk.text for chunk in chunks]
        vectors = self.embed_texts(texts)
        return list(zip(chunks, vectors))
