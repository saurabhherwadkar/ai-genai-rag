import logging
import re

from rank_bm25 import BM25Okapi

from rag_pipeline.models.chunk import Chunk
from rag_pipeline.models.query_result import QueryResult
from rag_pipeline.vectorstore.vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Performs BM25 keyword-based retrieval over the stored chunks."""

    def __init__(self, vector_store: VectorStoreManager, top_k: int = 50) -> None:
        self._vector_store = vector_store
        self._top_k = top_k
        self._logger = logger
        self._corpus_tokens: list[list[str]] = []
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def build_index(self) -> None:
        """Build the BM25 index from all chunks currently in the vector store."""
        self._vector_store._ensure_initialized()
        collection = self._vector_store._collection
        count = collection.count()
        if count == 0:
            self._logger.warning("No chunks in vector store to build BM25 index")
            return

        results = collection.get(include=["documents", "metadatas"])
        ids = results["ids"]
        documents = results["documents"]
        metadatas = results["metadatas"]

        self._chunks = []
        self._corpus_tokens = []
        for chunk_id, doc, meta in zip(ids, documents, metadatas, strict=True):
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=meta.get("document_id", ""),
                content=doc,
                chunk_index=meta.get("chunk_index", 0),
                start_char=meta.get("start_char", 0),
                end_char=meta.get("end_char", 0),
                metadata=meta,
            )
            self._chunks.append(chunk)
            self._corpus_tokens.append(self._tokenize(doc))

        self._bm25 = BM25Okapi(self._corpus_tokens)
        self._logger.info("BM25 index built with %d documents", len(self._chunks))

    def retrieve(self, query: str) -> list[QueryResult]:
        """Retrieve top-k chunks using BM25 scoring."""
        if self._bm25 is None or not self._chunks:
            self._logger.warning("BM25 index not built, returning empty results")
            return []

        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        scored_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: self._top_k]

        results = []
        for rank, idx in enumerate(scored_indices, start=1):
            if scores[idx] <= 0:
                break
            # Normalize BM25 score to 0-1 range using the max score
            max_score = scores[scored_indices[0]] if scores[scored_indices[0]] > 0 else 1.0
            normalized_score = scores[idx] / max_score

            result = QueryResult(
                chunk=self._chunks[idx],
                similarity_score=normalized_score,
                rank=rank,
            )
            results.append(result)

        self._logger.debug("BM25 retrieved %d results for query", len(results))
        return results

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        return re.findall(r"\w+", text.lower())
