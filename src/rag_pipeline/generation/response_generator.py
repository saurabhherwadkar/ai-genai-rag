# =============================================================================
# RAG Pipeline - Response Generator
# =============================================================================
# Generates final answers from retrieved context using template-based approach.

import logging  # Import logging for generation operation tracking

from rag_pipeline.generation.prompt_builder import PromptBuilder  # Prompt construction
from rag_pipeline.generation.template_engine import TemplateEngine  # Template rendering
from rag_pipeline.models.query_result import QueryResponse, QueryResult  # Result models

logger = logging.getLogger(__name__)  # Create module-level logger instance


class ResponseGenerator:
    """Generates a final answer from retrieved context using templates.

    This educational implementation uses string templates rather than an
    actual LLM, demonstrating the prompt construction and response assembly
    pattern that would feed into an LLM in production.
    """

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        template_engine: TemplateEngine,
        response_max_length: int = 2000,
    ) -> None:
        """Initialize the response generator with prompt and template components.

        Args:
            prompt_builder: Component for building LLM-ready prompts.
            template_engine: Component for rendering response templates.
            response_max_length: Maximum allowed length for generated responses.
        """
        self._prompt_builder = prompt_builder  # Store prompt builder reference
        self._template_engine = template_engine  # Store template engine reference
        self._response_max_length = response_max_length  # Store max response length
        self._logger = logger  # Store logger reference for this instance

    def generate_response(
        self, query: str, results: list[QueryResult]
    ) -> QueryResponse:
        """Generate a complete response from query and retrieval results.

        Args:
            query: The user's original query string.
            results: Ranked list of QueryResult objects from retrieval.

        Returns:
            QueryResponse object with generated answer and source attributions.
        """
        self._logger.debug("Generating response for query with %d results", len(results))
        context_string = self._build_context_string(results)  # Build context from results
        self._prompt_builder.build_prompt(query, context_string)  # Build prompt (for logging)
        generated_answer = self._generate_from_template(query, context_string)  # Generate answer
        sources = self._format_sources(results)  # Extract source attributions
        response = QueryResponse(  # Create the response object
            original_query=query,  # Store the original query
            processed_query=query,  # Processed query (same for now)
            results=results,  # Include the retrieval results
            generated_answer=generated_answer,  # The generated answer text
            sources=sources,  # Source attribution list
        )
        self._logger.info(  # Log generation completion
            "Response generated: %d chars, %d sources",  # Format message
            len(generated_answer),  # Answer length
            len(sources),  # Number of sources cited
        )
        return response  # Return the complete response

    def _build_context_string(self, results: list[QueryResult]) -> str:
        """Build a context string from retrieval results for prompt injection.

        Formats each result with its rank and source information
        to create a clear context block for the prompt.

        Args:
            results: List of QueryResult objects to include in context.

        Returns:
            Formatted context string with all relevant chunks.
        """
        if not results:  # Handle empty results
            return "No relevant context found."  # Return empty context indicator
        context_parts = []  # Initialize list for context sections
        for result in results:  # Process each retrieval result
            source_info = result.chunk.metadata.get("filename", "unknown")  # Get source name
            context_entry = (  # Format a single context entry
                f"[Source: {source_info}, Relevance: {result.similarity_score:.2f}]\n"
                f"{result.chunk.content}"  # Include the chunk text
            )
            context_parts.append(context_entry)  # Add entry to collection
        full_context = "\n\n---\n\n".join(context_parts)  # Join with separator
        return full_context  # Return the assembled context string

    def _generate_from_template(self, query: str, context: str) -> str:
        """Generate an answer using the template engine.

        In a production system, this would call an LLM API.
        Here we use templates to demonstrate the pattern.

        Args:
            query: The user's query.
            context: The formatted context string.

        Returns:
            Generated answer string.
        """
        try:  # Attempt to render response template
            variables = {  # Build template variables
                "query": query,  # The user's question
                "context": context,  # The retrieved context
            }
            answer = self._template_engine.render("response", variables)  # Render template
            return self._truncate_response(answer)  # Truncate if over length limit
        except KeyError:  # If response template is not found
            self._logger.warning("Response template not found, using fallback")  # Log warning
            return self._generate_fallback_response(query, context)  # Use fallback

    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generate a fallback response when template is unavailable.

        Args:
            query: The user's query.
            context: The formatted context string.

        Returns:
            A basic response constructed from the context.
        """
        fallback = (  # Build a simple fallback response
            f"Based on the available information regarding '{query}':\n\n"  # Header
            f"{context}\n\n"  # Include the context directly
            "Note: This is a template-based response. In production, "  # Disclaimer
            "an LLM would synthesize a natural language answer from this context."
        )
        return self._truncate_response(fallback)  # Truncate if needed

    def _format_sources(self, results: list[QueryResult]) -> list[str]:
        """Extract source attribution strings from retrieval results.

        Args:
            results: List of QueryResult objects to extract sources from.

        Returns:
            List of source reference strings for citation.
        """
        sources = []  # Initialize list for source references
        seen_sources = set()  # Track unique sources to avoid duplicates
        for result in results:  # Process each result
            source = result.chunk.metadata.get("filename", "unknown")  # Get source name
            if source not in seen_sources:  # If source not already recorded
                seen_sources.add(source)  # Mark as seen
                sources.append(source)  # Add to sources list
        return sources  # Return deduplicated source list

    def _truncate_response(self, response: str) -> str:
        """Truncate response to maximum configured length.

        Args:
            response: The generated response to potentially truncate.

        Returns:
            Response within the maximum length limit.
        """
        if len(response) <= self._response_max_length:  # Check if within limit
            return response  # Return unchanged
        truncated = response[: self._response_max_length - 3] + "..."  # Truncate with indicator
        self._logger.debug("Response truncated to %d chars", self._response_max_length)
        return truncated  # Return truncated response
