"""智能文本分块器 - 递归分块 + 滑动窗口重叠."""

from dataclasses import dataclass
from typing import Sequence

from .loader import Document


@dataclass
class Chunk:
    """分块后的文本片段，保留溯源信息."""

    text: str
    source: str
    chunk_index: int
    page_num: int | None = None
    metadata: dict | None = None


class TextChunker:
    """递归字符分块器，优先在自然断点处切分."""

    SEPARATORS = ["\n\n", "\n", "。", ". ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: Sequence[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            doc_chunks = self._split_text(doc.content)
            chunks.extend(
                Chunk(
                    text=chunk_text.strip(),
                    source=doc.source,
                    chunk_index=i,
                    page_num=doc.page_num,
                    metadata=doc.metadata,
                )
                for i, chunk_text in enumerate(doc_chunks)
                if chunk_text.strip()
            )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        """递归分割——先从大断点切，切不动再降级."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for sep in self.SEPARATORS:
            if sep == "":
                return self._force_split(text)
            splits = text.split(sep)
            if len(splits) > 1:
                return self._merge_splits(splits, sep)
        return self._force_split(text)

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """贪心合并，保持 chunk_size 上限 + overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for split in splits:
            piece = split + (separator if split != splits[-1] else "")
            piece_len = len(piece)

            if current_len + piece_len > self.chunk_size and current:
                chunks.append("".join(current))
                # 滑动窗口 overlap：保留最后一段作为下一个块的起点
                overlap_text = "".join(current)
                overlap_start = max(0, len(overlap_text) - self.chunk_overlap)
                current = [overlap_text[overlap_start:]]
                current_len = len(current[0])

            current.append(piece)
            current_len += piece_len

        if current:
            chunks.append("".join(current))
        return chunks

    def _force_split(self, text: str) -> list[str]:
        """最终兜底：按字符数硬切."""
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]


def estimate_token_count(text: str) -> int:
    """估算 token 数，辅助调参."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 2
