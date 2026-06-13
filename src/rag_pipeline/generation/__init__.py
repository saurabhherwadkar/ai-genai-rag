# =============================================================================
# RAG Pipeline - Generation Package
# =============================================================================
# Exports response generation components for building RAG answers.

from rag_pipeline.generation.prompt_builder import PromptBuilder  # Prompt construction
from rag_pipeline.generation.response_generator import ResponseGenerator  # Response generation
from rag_pipeline.generation.template_engine import TemplateEngine  # Template rendering

__all__ = [  # Public API of the generation package
    "TemplateEngine",
    "PromptBuilder",
    "ResponseGenerator",
]
