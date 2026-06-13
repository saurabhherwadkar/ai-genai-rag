# =============================================================================
# RAG Pipeline - Prompt Builder
# =============================================================================
# Constructs prompts that demonstrate the RAG prompt engineering pattern.

import logging  # Import logging for prompt construction tracking

logger = logging.getLogger(__name__)  # Create module-level logger instance


class PromptBuilder:
    """Constructs prompts that would be sent to an LLM in a production RAG system.

    Demonstrates prompt engineering patterns for RAG including:
    - System instruction with role and constraints
    - Context injection from retrieved documents
    - User query positioning
    - Guardrails for grounded responses
    """

    def __init__(self, system_template: str, user_template: str) -> None:
        """Initialize the prompt builder with system and user prompt templates.

        Args:
            system_template: The system instruction template (role, constraints).
            user_template: The user message template with {context} and {query} placeholders.
        """
        self._system_template = system_template  # Store the system prompt template
        self._user_template = user_template  # Store the user prompt template
        self._logger = logger  # Store logger reference for this instance

    def build_prompt(self, query: str, context: str) -> str:
        """Build the complete prompt by injecting context and query.

        Args:
            query: The user's question to answer.
            context: The retrieved context from the vector store.

        Returns:
            Complete formatted prompt string ready for LLM consumption.
        """
        user_prompt = self._inject_context(self._user_template, context)  # Add context
        user_prompt = self._inject_query(user_prompt, query)  # Add query
        full_prompt = self._assemble_full_prompt(user_prompt)  # Combine system + user
        guarded_prompt = self._apply_guardrails(full_prompt)  # Apply safety guardrails
        self._logger.debug(  # Log prompt construction
            "Prompt built: system=%d chars, user=%d chars",  # Format message
            len(self._system_template),  # System template length
            len(user_prompt),  # User prompt length
        )
        return guarded_prompt  # Return the complete prompt

    def _inject_context(self, template: str, context: str) -> str:
        """Replace the {context} placeholder with actual retrieved context.

        Args:
            template: Template string containing {context} placeholder.
            context: The context text to inject.

        Returns:
            Template with context injected.
        """
        return template.replace("{context}", context)  # Replace context placeholder

    def _inject_query(self, template: str, query: str) -> str:
        """Replace the {query} placeholder with the actual user query.

        Args:
            template: Template string containing {query} placeholder.
            query: The user query to inject.

        Returns:
            Template with query injected.
        """
        return template.replace("{query}", query)  # Replace query placeholder

    def _assemble_full_prompt(self, user_prompt: str) -> str:
        """Assemble the complete prompt from system and user components.

        Args:
            user_prompt: The formatted user message with context and query.

        Returns:
            Complete prompt combining system instruction and user message.
        """
        full_prompt = (  # Combine system and user prompts with separator
            f"[System]\n{self._system_template}\n\n"  # System instruction section
            f"[User]\n{user_prompt}"  # User message section
        )
        return full_prompt  # Return the assembled prompt

    def _apply_guardrails(self, prompt: str) -> str:
        """Apply response guardrails to encourage grounded answers.

        Appends instructions that encourage the model to stay grounded
        in the provided context and acknowledge uncertainty.

        Args:
            prompt: The assembled prompt to add guardrails to.

        Returns:
            Prompt with guardrail instructions appended.
        """
        guardrail = (  # Guardrail instruction text
            "\n\n[Guardrails]\n"  # Section header
            "- Only use information from the provided context.\n"  # Ground in context
            "- If the context does not contain enough information, say so clearly.\n"  # Honesty
            "- Do not make up or hallucinate information.\n"  # Prevent fabrication
            "- Cite which parts of the context support your answer."  # Attribution
        )
        return prompt + guardrail  # Append guardrails to the prompt
