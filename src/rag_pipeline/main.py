# =============================================================================
# RAG Pipeline - Main Application Entry Point
# =============================================================================
# Provides the CLI interface for document ingestion and interactive querying.

import argparse  # Import argparse for command-line argument parsing
import logging  # Import logging for application-level logging
import sys  # Import sys for exit codes and stdin handling
from pathlib import Path  # Import Path for file system operations

from rag_pipeline.config.logging_config import LoggingConfigurator  # Logging setup
from rag_pipeline.config.settings_loader import SettingsLoader  # Configuration loading
from rag_pipeline.generation.prompt_builder import PromptBuilder  # Prompt construction
from rag_pipeline.generation.response_generator import ResponseGenerator  # Response generation
from rag_pipeline.generation.template_engine import TemplateEngine  # Template rendering
from rag_pipeline.ingestion.document_loader import DocumentLoader  # Document loading
from rag_pipeline.ingestion.embedding_generator import EmbeddingGenerator  # Embedding generation
from rag_pipeline.ingestion.ingestion_pipeline import IngestionPipeline  # Ingestion orchestrator
from rag_pipeline.ingestion.text_chunker import TextChunker  # Text chunking
from rag_pipeline.models.pipeline_config import (  # Configuration models
    ChunkingConfig,
    EmbeddingConfig,
    RetrievalConfig,
)
from rag_pipeline.query.bm25_retriever import BM25Retriever  # BM25 keyword retrieval
from rag_pipeline.query.hybrid_retriever import HybridRetriever  # Hybrid RRF fusion
from rag_pipeline.query.query_pipeline import QueryPipeline  # Query orchestrator
from rag_pipeline.query.query_processor import QueryProcessor  # Query preprocessing
from rag_pipeline.query.reranker import Reranker  # Result reranking
from rag_pipeline.query.retriever import Retriever  # Similarity retrieval
from rag_pipeline.security.input_sanitizer import InputSanitizer  # Input security
from rag_pipeline.utils.exceptions import RAGPipelineError  # Base exception
from rag_pipeline.vectorstore.vector_store_manager import VectorStoreManager  # Vector store

logger = logging.getLogger(__name__)  # Create module-level logger instance


class RAGApplication:
    """Main application class that wires together all pipeline components.

    Provides methods for document ingestion, single queries, and an
    interactive query loop via the command line.
    """

    def __init__(self) -> None:
        """Initialize the RAG application (components are set up in setup())."""
        self._settings = None  # Configuration dictionary (populated during setup)
        self._ingestion_pipeline = None  # Ingestion pipeline (built during setup)
        self._query_pipeline = None  # Query pipeline (built during setup)
        self._vector_store = None  # Shared vector store instance
        self._bm25_retriever = None  # BM25 retriever (needs index rebuild after ingestion)

    def setup(self) -> None:
        """Initialize configuration, logging, and all pipeline components.

        Must be called before any ingestion or query operations.
        """
        config_dir = self._find_config_directory()  # Locate the config directory
        self._initialize_logging(config_dir)  # Set up logging from config
        self._settings = self._load_settings(config_dir)  # Load all settings
        self._vector_store = self._create_vector_store()  # Create shared vector store
        self._vector_store.initialize_store()  # Initialize the vector store
        self._ingestion_pipeline = self._build_ingestion_pipeline()  # Build ingestion pipeline
        self._query_pipeline = self._build_query_pipeline()  # Build query pipeline
        logger.info("RAG Application initialized successfully")  # Log successful setup

    def run_ingestion(self, source_path: str) -> None:
        """Run the ingestion pipeline on the specified source.

        Args:
            source_path: Path to a file or directory to ingest.
        """
        path = Path(source_path)  # Convert string to Path object
        if path.is_file():  # If source is a single file
            chunks_stored = self._ingestion_pipeline.ingest_file(path)  # Ingest single file
            print(f"Ingested file: {path.name} ({chunks_stored} chunks stored)")  # User output
        elif path.is_dir():  # If source is a directory
            extensions = self._settings.get("ingestion", {}).get(  # Get supported extensions
                "supported_extensions", [".txt", ".md"]  # Default extensions
            )
            chunks_stored = self._ingestion_pipeline.ingest_directory(  # Ingest directory
                path, extensions  # Pass path and extension filter
            )
            print(f"Ingested directory: {path} ({chunks_stored} total chunks stored)")
        else:  # Source path does not exist
            print(f"Error: Path not found: {source_path}")  # Report error to user
            sys.exit(1)  # Exit with error code
        if self._bm25_retriever is not None:
            self._bm25_retriever.build_index()

    def run_query(self, query: str) -> str:
        """Execute a single query through the RAG pipeline.

        Args:
            query: The user's question to answer.

        Returns:
            The generated answer string.
        """
        response = self._query_pipeline.execute_query(query)  # Execute the query pipeline
        return response.generated_answer  # Return just the answer text

    def run_interactive(self) -> None:
        """Run an interactive query loop accepting queries from stdin.

        Type 'quit' or 'exit' to stop the interactive session.
        """
        print("RAG Pipeline Interactive Mode")  # Print session header
        print("Type your question and press Enter. Type 'quit' to exit.")  # Instructions
        print("-" * 60)  # Separator line
        while True:  # Loop until user exits
            try:  # Handle keyboard interrupt gracefully
                query = input("\nQuery> ").strip()  # Prompt for user input
                if query.lower() in ("quit", "exit", "q"):  # Check for exit commands
                    print("Goodbye!")  # Farewell message
                    break  # Exit the loop
                if not query:  # Skip empty input
                    continue  # Prompt again
                response = self._query_pipeline.execute_query(query)  # Execute query
                print(f"\n{response.generated_answer}")  # Display the answer
                print(f"\nSources: {', '.join(response.sources)}")  # Display sources
                print(f"Latency: {response.latency_ms:.1f}ms")  # Display timing
            except RAGPipelineError as error:  # Catch pipeline errors
                print(f"\nError: {error}")  # Display user-friendly error message
            except KeyboardInterrupt:  # Handle Ctrl+C
                print("\nGoodbye!")  # Farewell message
                break  # Exit the loop

    def _find_config_directory(self) -> Path:
        """Locate the configuration directory relative to the project root.

        Returns:
            Path to the config directory.
        """
        # Check common config directory locations
        possible_paths = [  # List of possible config directory locations
            Path("config"),  # Relative to current directory
            Path(__file__).parent.parent.parent / "config",  # Relative to package
        ]
        for path in possible_paths:  # Try each possible location
            if path.exists():  # If directory exists
                return path  # Return the found path
        return Path("config")  # Default to relative path (may not exist yet)

    def _initialize_logging(self, config_dir: Path) -> None:
        """Initialize the logging system from configuration.

        Args:
            config_dir: Path to the configuration directory.
        """
        logging_config_path = config_dir / "logging.yaml"  # Build path to logging config
        configurator = LoggingConfigurator(logging_config_path)  # Create configurator
        configurator.configure()  # Apply logging configuration

    def _load_settings(self, config_dir: Path) -> dict:
        """Load application settings from configuration files.

        Args:
            config_dir: Path to the configuration directory.

        Returns:
            Complete merged settings dictionary.
        """
        loader = SettingsLoader(config_dir)  # Create settings loader
        return loader.load()  # Load and return merged settings

    def _create_vector_store(self) -> VectorStoreManager:
        """Create the shared vector store manager from settings.

        Returns:
            Configured VectorStoreManager instance.
        """
        vs_config = self._settings.get("vectorstore", {})  # Get vectorstore settings
        return VectorStoreManager(  # Create and return the manager
            collection_name=vs_config.get("collection_name", "rag_documents"),  # Collection name
            persist_directory=vs_config.get("persist_directory"),  # Optional persistence
            distance_metric=vs_config.get("distance_metric", "cosine"),  # Distance metric
        )

    def _build_ingestion_pipeline(self) -> IngestionPipeline:
        """Build the ingestion pipeline with all components wired together.

        Returns:
            Configured IngestionPipeline instance.
        """
        ingestion_config = self._settings.get("ingestion", {})  # Get ingestion settings
        chunking_settings = ingestion_config.get("chunking", {})  # Get chunking settings
        embedding_settings = ingestion_config.get("embedding", {})  # Get embedding settings
        chunking_config = ChunkingConfig(  # Create chunking configuration
            chunk_size=chunking_settings.get("chunk_size", 512),  # Chunk size
            chunk_overlap=chunking_settings.get("chunk_overlap", 50),  # Overlap
            separator=chunking_settings.get("separator", "\n\n"),  # Separator
            min_chunk_size=chunking_settings.get("min_chunk_size", 50),  # Minimum size
        )
        embedding_config = EmbeddingConfig(  # Create embedding configuration
            model_name=embedding_settings.get("model_name", "all-MiniLM-L6-v2"),  # Model
            batch_size=embedding_settings.get("batch_size", 32),  # Batch size
            normalize=embedding_settings.get("normalize", True),  # Normalization
        )
        document_loader = DocumentLoader(  # Create document loader
            supported_extensions=ingestion_config.get(  # Pass supported extensions
                "supported_extensions", [".txt", ".md", ".pdf", ".json"]
            )
        )
        text_chunker = TextChunker(chunking_config)  # Create text chunker
        embedding_generator = EmbeddingGenerator(embedding_config)  # Create embedding gen
        return IngestionPipeline(  # Assemble and return the pipeline
            document_loader=document_loader,  # Document loading component
            text_chunker=text_chunker,  # Text chunking component
            embedding_generator=embedding_generator,  # Embedding component
            vector_store=self._vector_store,  # Shared vector store
        )

    def _build_query_pipeline(self) -> QueryPipeline:
        """Build the query pipeline with all components wired together.

        Returns:
            Configured QueryPipeline instance.
        """
        retrieval_settings = self._settings.get("retrieval", {})  # Get retrieval settings
        generation_settings = self._settings.get("generation", {})  # Get generation settings
        security_settings = self._settings.get("security", {})  # Get security settings
        ingestion_config = self._settings.get("ingestion", {})  # Get ingestion settings
        embedding_settings = ingestion_config.get("embedding", {})  # Get embedding settings
        retrieval_config = RetrievalConfig(  # Create retrieval configuration
            top_k=retrieval_settings.get("top_k", 5),  # Top-K results
            similarity_threshold=retrieval_settings.get("similarity_threshold", 0.3),
            rerank_enabled=retrieval_settings.get("rerank_enabled", True),  # Reranking toggle
            max_query_length=retrieval_settings.get("max_query_length", 1000),  # Max length
            hybrid_search_enabled=retrieval_settings.get("hybrid_search_enabled", True),
            semantic_top_k=retrieval_settings.get("semantic_top_k", 50),
            bm25_top_k=retrieval_settings.get("bm25_top_k", 50),
        )
        embedding_config = EmbeddingConfig(  # Create embedding config for query embedding
            model_name=embedding_settings.get("model_name", "all-MiniLM-L6-v2"),
            batch_size=embedding_settings.get("batch_size", 32),
            normalize=embedding_settings.get("normalize", True),
        )
        input_sanitizer = InputSanitizer(  # Create input sanitizer
            max_input_length=security_settings.get("max_input_length", 5000)
        )
        query_processor = QueryProcessor(retrieval_config, input_sanitizer)  # Create processor
        embedding_generator = EmbeddingGenerator(embedding_config)  # Create embedding gen
        semantic_retriever = Retriever(  # Create semantic retriever
            self._vector_store, embedding_generator, retrieval_config  # Wire components
        )
        if retrieval_config.hybrid_search_enabled:
            self._bm25_retriever = BM25Retriever(
                self._vector_store, top_k=retrieval_config.bm25_top_k
            )
            self._bm25_retriever.build_index()
            retriever = HybridRetriever(
                semantic_retriever, self._bm25_retriever, retrieval_config
            )
        else:
            retriever = semantic_retriever
        reranker = Reranker(retrieval_config)  # Create reranker
        # Build response generation components
        system_template = generation_settings.get(  # Get system template
            "system_template", "Answer based only on the provided context."
        )
        user_template = generation_settings.get(  # Get user template
            "user_template", "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        prompt_builder = PromptBuilder(system_template, user_template)  # Create prompt builder
        template_engine = TemplateEngine({  # Create template engine with response template
            "response": (  # Define the response template
                "Based on the provided context, here is the answer to your question:\n\n"
                "Question: {query}\n\n"
                "Answer: The following information was found in the knowledge base:\n\n"
                "{context}\n\n"
                "Note: This response is generated from retrieved document chunks. "
                "In a production system, an LLM would synthesize a more natural response."
            )
        })
        response_generator = ResponseGenerator(  # Create response generator
            prompt_builder=prompt_builder,  # Prompt building component
            template_engine=template_engine,  # Template rendering component
            response_max_length=generation_settings.get("response_max_length", 2000),
        )
        return QueryPipeline(  # Assemble and return the pipeline
            query_processor=query_processor,  # Query preprocessing
            retriever=retriever,  # Similarity retrieval
            reranker=reranker,  # Result reranking
            response_generator=response_generator,  # Response generation
        )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for the RAG pipeline CLI.

    Returns:
        Configured ArgumentParser with all subcommands.
    """
    parser = argparse.ArgumentParser(  # Create the top-level parser
        prog="rag-pipeline",  # Program name for help text
        description="Educational RAG Pipeline - Document Ingestion and Query System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")  # Add subs
    # Ingest subcommand
    ingest_parser = subparsers.add_parser(  # Create ingest subcommand
        "ingest", help="Ingest documents into the vector store"  # Help text
    )
    ingest_parser.add_argument(  # Add source path argument
        "source", type=str, help="Path to file or directory to ingest"  # Description
    )
    # Query subcommand
    query_parser = subparsers.add_parser(  # Create query subcommand
        "query", help="Query the RAG pipeline with a question"  # Help text
    )
    query_parser.add_argument(  # Add query text argument
        "text", type=str, help="The question to ask the RAG pipeline"  # Description
    )
    # Interactive subcommand
    subparsers.add_parser(  # Create interactive subcommand
        "interactive", help="Start interactive query mode"  # Help text
    )
    # Stats subcommand
    subparsers.add_parser(  # Create stats subcommand
        "stats", help="Show vector store statistics"  # Help text
    )
    return parser  # Return the configured parser


def main() -> None:
    """Main entry point for the RAG pipeline CLI application."""
    parser = create_argument_parser()  # Create the argument parser
    args = parser.parse_args()  # Parse command-line arguments
    if not args.command:  # If no command specified
        parser.print_help()  # Show help message
        sys.exit(0)  # Exit cleanly
    app = RAGApplication()  # Create the application instance
    try:  # Wrap main execution in error handler
        app.setup()  # Initialize all components
        if args.command == "ingest":  # Handle ingest command
            app.run_ingestion(args.source)  # Run ingestion on the source
        elif args.command == "query":  # Handle query command
            answer = app.run_query(args.text)  # Execute the query
            print(answer)  # Print the answer to stdout
        elif args.command == "interactive":  # Handle interactive command
            app.run_interactive()  # Start interactive loop
        elif args.command == "stats":  # Handle stats command
            stats = app._vector_store.get_collection_stats()  # Get store statistics
            print("Vector Store Statistics:")  # Header
            for key, value in stats.items():  # Print each stat
                print(f"  {key}: {value}")  # Format key-value pair
    except RAGPipelineError as error:  # Catch all pipeline errors
        logger.error("Pipeline error: %s", error)  # Log the error
        print(f"Error: {error}")  # Show user-friendly message
        sys.exit(1)  # Exit with error code
    except KeyboardInterrupt:  # Handle Ctrl+C at top level
        print("\nOperation cancelled.")  # Inform user
        sys.exit(0)  # Exit cleanly


if __name__ == "__main__":  # Guard for direct script execution
    main()  # Call the main entry point
