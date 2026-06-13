# =============================================================================
# RAG Pipeline - Query Pipeline Orchestrator
# =============================================================================
# Orchestrates the full query flow: process -> retrieve -> rerank -> generate.

import logging  # Import logging for pipeline execution tracking

from rag_pipeline.generation.response_generator import ResponseGenerator  # Generation component
from rag_pipeline.models.query_result import QueryResponse  # Import response model
from rag_pipeline.query.query_processor import QueryProcessor  # Query preprocessing
from rag_pipeline.query.reranker import Reranker  # Result reranking component
from rag_pipeline.query.retriever import Retriever  # Retrieval component
from rag_pipeline.utils.metrics import PerformanceMetrics  # Timing utilities

logger = logging.getLogger(__name__)  # Create module-level logger instance


class QueryPipeline:
    """Orchestrates the full query flow from input to generated response.

    Coordinates the sequence: process query -> retrieve similar chunks ->
    rerank results -> generate response. Measures end-to-end latency.
    """

    def __init__(
        self,
        query_processor: QueryProcessor,
        retriever: Retriever,
        reranker: Reranker,
        response_generator: ResponseGenerator,
    ) -> None:
        """Initialize the query pipeline with all required components.

        Args:
            query_processor: Component for query preprocessing and validation.
            retriever: Component for similarity-based retrieval.
            reranker: Component for reranking retrieval results.
            response_generator: Component for generating the final response.
        """
        self._query_processor = query_processor  # Store query processor reference
        self._retriever = retriever  # Store retriever reference
        self._reranker = reranker  # Store reranker reference
        self._response_generator = response_generator  # Store response generator reference
        self._logger = logger  # Store logger reference for this instance

    def execute_query(self, raw_query: str) -> QueryResponse:
        """Execute the full query pipeline from raw input to generated response.

        Args:
            raw_query: The unprocessed user query string.

        Returns:
            QueryResponse with generated answer, sources, and metrics.

        Raises:
            RAGPipelineError: If any stage of the query pipeline fails.
        """
        self._logger.info("Executing query pipeline for: '%s'", raw_query[:50])  # Log start
        with PerformanceMetrics.measure_time() as timing:  # Measure total pipeline latency
            processed_query = self._query_processor.process_query(raw_query)  # Step 1: Process
            results = self._retriever.retrieve(processed_query)  # Step 2: Retrieve
            reranked_results = self._reranker.rerank(processed_query, results)  # Step 3: Rerank
            response = self._response_generator.generate_response(  # Step 4: Generate
                processed_query, reranked_results  # Pass query and reranked results
            )
        response.processed_query = processed_query  # Update with processed query
        response.original_query = raw_query  # Preserve the original query
        response.latency_ms = timing["elapsed_ms"]  # Record total pipeline latency
        self._logger.info(  # Log pipeline completion
            "Query pipeline complete: %d results, %.1fms latency",  # Format message
            len(reranked_results),  # Number of results returned
            timing["elapsed_ms"],  # Total latency
        )
        return response  # Return the complete query response
