# =============================================================================
# RAG Pipeline - Reranker
# =============================================================================
# Reranks retrieval results using keyword overlap and position scoring.

import logging  # Import logging for reranking operation tracking
import re  # Import regex for keyword extraction

from rag_pipeline.models.pipeline_config import RetrievalConfig  # Import config model
from rag_pipeline.models.query_result import QueryResult  # Import result model

logger = logging.getLogger(__name__)  # Create module-level logger instance


class Reranker:
    """Reranks retrieval results using keyword overlap and position scoring.

    For educational simplicity, uses a combination of:
    1. Original similarity score from the vector store
    2. Keyword overlap between the query and chunk text
    3. Position bonus for earlier chunks (closer to document start)

    In production, this would typically use a cross-encoder model.
    """

    # Weights for combining different scoring signals
    SIMILARITY_WEIGHT = 0.6  # Weight given to original vector similarity score
    KEYWORD_WEIGHT = 0.3  # Weight given to keyword overlap score
    POSITION_WEIGHT = 0.1  # Weight given to position-based score

    def __init__(self, config: RetrievalConfig) -> None:
        """Initialize the reranker with retrieval configuration.

        Args:
            config: RetrievalConfig with reranking settings.
        """
        self._config = config  # Store retrieval configuration
        self._logger = logger  # Store logger reference for this instance

    def rerank(self, query: str, results: list[QueryResult]) -> list[QueryResult]:
        """Rerank a list of retrieval results using multiple scoring signals.

        Args:
            query: The original user query for keyword matching.
            results: List of QueryResult objects to rerank.

        Returns:
            List of QueryResult objects sorted by combined score with updated ranks.
        """
        if not results:  # Handle empty results list
            return results  # Nothing to rerank
        if not self._config.rerank_enabled:  # Check if reranking is disabled
            self._logger.debug("Reranking disabled, returning original order")  # Log skip
            return results  # Return results in original order
        self._logger.debug("Reranking %d results", len(results))  # Log reranking start
        scored_results = self._compute_combined_scores(query, results)  # Score all results
        sorted_results = self._sort_by_final_score(scored_results)  # Sort by combined score
        ranked_results = self._assign_final_ranks(sorted_results)  # Update rank numbers
        self._logger.info(  # Log reranking completion
            "Reranked %d results", len(ranked_results)  # Show result count
        )
        return ranked_results  # Return reranked results

    def _compute_combined_scores(
        self, query: str, results: list[QueryResult]
    ) -> list[tuple[QueryResult, float]]:
        """Compute combined scores for all results using multiple signals.

        Args:
            query: The user query for keyword overlap calculation.
            results: List of results to score.

        Returns:
            List of tuples containing (QueryResult, combined_score).
        """
        total_results = len(results)  # Get total count for position scoring
        scored = []  # Initialize list for scored results
        for result in results:  # Process each result
            keyword_score = self._compute_keyword_overlap_score(  # Calculate keyword overlap
                query, result.chunk.content  # Compare query against chunk text
            )
            position_score = self._compute_position_score(  # Calculate position bonus
                result.rank, total_results  # Pass rank and total for normalization
            )
            combined = self._combine_scores(  # Combine all scoring signals
                result.similarity_score, keyword_score, position_score  # Three signals
            )
            scored.append((result, combined))  # Store result with its combined score
        return scored  # Return all scored results

    def _compute_keyword_overlap_score(self, query: str, chunk_text: str) -> float:
        """Compute keyword overlap between query and chunk text.

        Calculates the fraction of query keywords found in the chunk text.
        Uses case-insensitive matching on word boundaries.

        Args:
            query: The user query to extract keywords from.
            chunk_text: The chunk text to search for keywords in.

        Returns:
            Float between 0.0 and 1.0 representing keyword overlap ratio.
        """
        query_keywords = self._extract_keywords(query)  # Extract keywords from query
        if not query_keywords:  # If no keywords could be extracted
            return 0.0  # Return zero overlap score
        chunk_lower = chunk_text.lower()  # Lowercase chunk for case-insensitive matching
        matches = 0  # Initialize match counter
        for keyword in query_keywords:  # Check each query keyword
            if keyword in chunk_lower:  # If keyword found in chunk text
                matches += 1  # Increment match counter
        overlap_ratio = matches / len(query_keywords)  # Calculate ratio of matches to total
        return overlap_ratio  # Return the overlap score

    def _compute_position_score(self, original_rank: int, total: int) -> float:
        """Compute a position-based score favoring earlier results.

        Results appearing earlier in the original ranking get a higher
        position score, providing a tie-breaking signal.

        Args:
            original_rank: The 1-based rank of this result.
            total: Total number of results being ranked.

        Returns:
            Float between 0.0 and 1.0 representing position preference.
        """
        if total <= 1:  # If only one result, no position differentiation needed
            return 1.0  # Give maximum position score
        # Linear decay: rank 1 gets 1.0, last rank gets ~0.0
        score = 1.0 - ((original_rank - 1) / (total - 1))  # Linear interpolation
        return max(0.0, score)  # Ensure score doesn't go below zero

    def _combine_scores(
        self, similarity: float, keyword: float, position: float
    ) -> float:
        """Combine multiple scoring signals into a single score using weights.

        Args:
            similarity: Vector similarity score (0.0 to 1.0).
            keyword: Keyword overlap score (0.0 to 1.0).
            position: Position-based score (0.0 to 1.0).

        Returns:
            Weighted combined score.
        """
        combined = (  # Compute weighted sum of all signals
            (self.SIMILARITY_WEIGHT * similarity)  # Weighted similarity contribution
            + (self.KEYWORD_WEIGHT * keyword)  # Weighted keyword contribution
            + (self.POSITION_WEIGHT * position)  # Weighted position contribution
        )
        return round(combined, 4)  # Round to 4 decimal places for cleanliness

    def _sort_by_final_score(
        self, scored_results: list[tuple[QueryResult, float]]
    ) -> list[tuple[QueryResult, float]]:
        """Sort results by their combined scores in descending order.

        Args:
            scored_results: List of (QueryResult, score) tuples.

        Returns:
            Sorted list with highest scores first.
        """
        return sorted(  # Sort the scored results
            scored_results,  # Input list of tuples
            key=lambda x: x[1],  # Sort by the score (second element)
            reverse=True,  # Descending order (highest score first)
        )

    def _assign_final_ranks(
        self, sorted_results: list[tuple[QueryResult, float]]
    ) -> list[QueryResult]:
        """Assign new rank numbers based on the reranked order.

        Args:
            sorted_results: Sorted list of (QueryResult, score) tuples.

        Returns:
            List of QueryResult objects with updated rank values.
        """
        ranked = []  # Initialize list for ranked results
        for new_rank, (result, _score) in enumerate(sorted_results, start=1):  # Enumerate from 1
            result.rank = new_rank  # Update the result's rank to new position
            ranked.append(result)  # Add to ranked results list
        return ranked  # Return results with updated ranks

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text by removing stop words.

        Args:
            text: Input text to extract keywords from.

        Returns:
            List of lowercase keyword strings.
        """
        # Common English stop words to exclude from keyword matching
        stop_words = {  # Set of words with low semantic value
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "it", "this", "that", "these", "those", "i", "you", "he",
            "she", "we", "they", "what", "which", "who", "when", "where",
            "how", "not", "no", "and", "or", "but", "if", "then",
        }
        words = re.findall(r"\w+", text.lower())  # Extract all words as lowercase
        keywords = [w for w in words if w not in stop_words and len(w) > 2]  # Filter
        return keywords  # Return filtered keyword list
