# =============================================================================
# RAG Pipeline - Template Engine
# =============================================================================
# Simple template rendering for response generation without external deps.

import logging  # Import logging for template operation tracking
import re  # Import regex for template variable substitution

logger = logging.getLogger(__name__)  # Create module-level logger instance


class TemplateEngine:
    """Simple template rendering engine for response generation.

    Uses {variable_name} placeholders in templates and substitutes
    them with provided values. Designed to be lightweight and transparent
    for educational purposes.
    """

    # Regex pattern to find template variables like {variable_name}
    VARIABLE_PATTERN = re.compile(r"\{(\w+)\}")  # Match curly-brace enclosed words

    def __init__(self, templates: dict[str, str]) -> None:
        """Initialize the template engine with a dictionary of named templates.

        Args:
            templates: Dictionary mapping template names to template strings.
        """
        self._templates = templates  # Store the templates dictionary
        self._logger = logger  # Store logger reference for this instance

    def render(self, template_name: str, variables: dict[str, str]) -> str:
        """Render a named template by substituting variables.

        Args:
            template_name: The name of the template to render.
            variables: Dictionary of variable names to their values.

        Returns:
            The rendered template string with variables substituted.

        Raises:
            KeyError: If the template name is not found.
        """
        template = self._load_template(template_name)  # Retrieve the template string
        rendered = self._substitute_variables(template, variables)  # Replace placeholders
        self._logger.debug(  # Log template rendering
            "Rendered template '%s' (output length=%d)",  # Format message
            template_name,  # Template identifier
            len(rendered),  # Length of rendered output
        )
        return rendered  # Return the rendered string

    def _load_template(self, template_name: str) -> str:
        """Load a template string by name from the templates dictionary.

        Args:
            template_name: The name of the template to retrieve.

        Returns:
            The template string.

        Raises:
            KeyError: If the template name does not exist.
        """
        if template_name not in self._templates:  # Check if template exists
            self._logger.error("Template not found: '%s'", template_name)  # Log the error
            raise KeyError(f"Template '{template_name}' not found")  # Raise with name
        return self._templates[template_name]  # Return the template string

    def _substitute_variables(self, template: str, variables: dict[str, str]) -> str:
        """Replace {variable_name} placeholders with actual values.

        Args:
            template: Template string containing {variable} placeholders.
            variables: Dictionary mapping variable names to their values.

        Returns:
            String with all recognized variables substituted.
        """

        def replace_match(match: re.Match) -> str:
            """Replace a single regex match with its variable value."""
            var_name = match.group(1)  # Extract the variable name from the match
            if var_name in variables:  # If variable has a provided value
                return str(variables[var_name])  # Return the substitution value
            return match.group(0)  # Keep original placeholder if no value provided

        result = self.VARIABLE_PATTERN.sub(replace_match, template)  # Apply substitutions
        return result  # Return the fully substituted string

    def list_templates(self) -> list[str]:
        """List all available template names.

        Returns:
            List of template name strings.
        """
        return list(self._templates.keys())  # Return all template names as a list

    def get_required_variables(self, template_name: str) -> list[str]:
        """Get the list of variable names required by a template.

        Args:
            template_name: The template to inspect for variables.

        Returns:
            List of variable names found in the template.
        """
        template = self._load_template(template_name)  # Load the template
        matches = self.VARIABLE_PATTERN.findall(template)  # Find all variable names
        return list(set(matches))  # Return unique variable names
