import logging

from rag_pipeline.models.pipeline_config import RetrievalConfig
from rag_pipeline.models.query_result import QueryResult
from rag_pipeline.query.bm25_retriever import BM25Retriever
from rag_pipeline.query.retriever import Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines semantic (vector) and syntactic (BM25) retrieval using RRF.

    Retrieves top-N from each source independently, then merges results
    using Reciprocal Rank Fusion to produce the final ranked list.
    """

    RRF_K = 60  # Standard RRF constant to prevent high-ranked items from dominating

    def __init__(
        self,
        semantic_retriever: Retriever,
        bm25_retriever: BM25Retriever,
        config: RetrievalConfig,
    ) -> None:
        self._semantic_retriever = semantic_retriever
        self._bm25_retriever = bm25_retriever
        self._config = config
        self._logger = logger

    def retrieve(self, processed_query: str) -> list[QueryResult]:
        """Retrieve using both semantic and BM25, fuse with RRF.

        Fetches top-50 from each retriever, merges via RRF, returns top-k.
        """
        semantic_results = self._semantic_retriever.retrieve(
            processed_query, top_k_override=self._config.semantic_top_k
        )
        bm25_results = self._bm25_retriever.retrieve(processed_query)

        self._logger.info(
            "Hybrid retrieval: %d semantic, %d BM25 results",
            len(semantic_results),
            len(bm25_results),
        )

        fused = self._reciprocal_rank_fusion(semantic_results, bm25_results)

        top_k = self._config.top_k
        final = fused[:top_k]

        for rank, result in enumerate(final, start=1):
            result.rank = rank

        self._logger.info("RRF fusion produced %d final results", len(final))
        return final

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[QueryResult],
        bm25_results: list[QueryResult],
    ) -> list[QueryResult]:
        """Merge two ranked lists using Reciprocal Rank Fusion.

        RRF score = sum over lists of: 1 / (k + rank_in_list)
        where k is a constant (typically 60).
        """
        chunk_scores: dict[str, float] = {}
        chunk_map: dict[str, QueryResult] = {}

        for result in semantic_results:
            key = result.chunk.chunk_id
            rrf_score = 1.0 / (self.RRF_K + result.rank)
            chunk_scores[key] = chunk_scores.get(key, 0.0) + rrf_score
            chunk_map[key] = result

        for result in bm25_results:
            key = result.chunk.chunk_id
            rrf_score = 1.0 / (self.RRF_K + result.rank)
            chunk_scores[key] = chunk_scores.get(key, 0.0) + rrf_score
            if key not in chunk_map:
                chunk_map[key] = result

        sorted_keys = sorted(chunk_scores, key=lambda k: chunk_scores[k], reverse=True)

        fused_results = []
        for key in sorted_keys:
            result = chunk_map[key]
            result.similarity_score = chunk_scores[key]
            fused_results.append(result)

        return fused_results
