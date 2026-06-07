"""Weaviate 向量存储 - 文档索引与相似度检索."""

import uuid
from collections.abc import Sequence

from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from weaviate.connect.base import ProtocolParams
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType

from .chunker import Chunk

COLLECTION_NAME = "RagDocuments"


class VectorStore:
    """封装 Weaviate 的写入和检索操作."""

    def __init__(
        self,
        http_host: str = "localhost",
        http_port: int = 8080,
        grpc_port: int = 50051,
        api_key: str | None = None,
    ):
        self.client: WeaviateClient | None = None
        auth = Auth.api_key(api_key) if api_key else None
        self._connect_params = ConnectionParams(
            http=ProtocolParams(host=http_host, port=http_port, secure=False),
            grpc=ProtocolParams(host=http_host, port=grpc_port, secure=False),
        )
        self._auth = auth

    def connect(self) -> WeaviateClient:
        """建立连接并确保集合存在."""
        if self.client is not None:
            return self.client

        self.client = WeaviateClient(
            connection_params=self._connect_params,
            auth_client_secret=self._auth,
        )
        self.client.connect()
        self._ensure_collection()
        return self.client

    def disconnect(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def _ensure_collection(self) -> None:
        client = self._require_client()
        if not client.collections.exists(COLLECTION_NAME):
            client.collections.create(
                name=COLLECTION_NAME,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="page_num", data_type=DataType.INT),
                ],
            )

    def index_chunks(self, chunks_with_vectors: Sequence[tuple[Chunk, list[float]]]) -> int:
        """批量写入 chunk + 向量."""
        client = self._require_client()
        collection = client.collections.get(COLLECTION_NAME)

        with collection.batch.fixed_size(batch_size=100) as batch:
            for chunk, vector in chunks_with_vectors:
                obj_uuid = uuid.uuid4()
                batch.add_object(
                    uuid=obj_uuid,
                    properties={
                        "content": chunk.text,
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                        "page_num": chunk.page_num or 0,
                    },
                    vector=vector,
                )

        if batch.number_errors > 0:
            raise RuntimeError(f"批量写入出错: {batch.number_errors} 条失败")

        return len(chunks_with_vectors)

    def search(self, query_vector: list[float], top_k: int = 5, alpha: float = 0.7) -> list[dict]:
        """混合检索（向量 + 关键词），alpha 控制向量权重."""
        client = self._require_client()
        collection = client.collections.get(COLLECTION_NAME)

        response = collection.query.hybrid(
            query="",
            vector=query_vector,
            alpha=alpha,
            limit=top_k,
            return_metadata=["score"],
        )

        return [
            {
                "content": obj.properties["content"],
                "source": obj.properties["source"],
                "chunk_index": obj.properties["chunk_index"],
                "page_num": obj.properties.get("page_num"),
                "score": obj.metadata.score,
            }
            for obj in response.objects
        ]

    def clear(self) -> None:
        """清空集合，方便调试."""
        client = self._require_client()
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)
            self._ensure_collection()

    def count(self) -> int:
        client = self._require_client()
        if not client.collections.exists(COLLECTION_NAME):
            return 0
        return client.collections.get(COLLECTION_NAME).aggregate.over_all(total_count=True).total_count

    def _require_client(self) -> WeaviateClient:
        if self.client is None:
            raise RuntimeError("未连接 Weaviate，请先调用 connect()")
        return self.client
