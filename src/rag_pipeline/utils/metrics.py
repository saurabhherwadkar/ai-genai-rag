# =============================================================================
# RAG Pipeline - Performance Metrics Utilities
# =============================================================================
# Provides timing and measurement utilities for pipeline performance tracking.

import time  # Import time module for high-precision timing measurements
from collections.abc import Generator  # Import Generator type for context manager typing
from contextlib import contextmanager  # Import contextmanager for timing context


class PerformanceMetrics:
    """Utility class for measuring and reporting pipeline performance.

    Provides context managers and helper methods for timing operations
    and collecting performance statistics.
    """

    @staticmethod  # No instance state needed for this method
    @contextmanager  # Makes this method usable with the 'with' statement
    def measure_time() -> Generator[dict, None, None]:
        """Context manager that measures elapsed time in milliseconds.

        Usage:
            with PerformanceMetrics.measure_time() as timing:
                # ... operation to measure ...
            elapsed = timing['elapsed_ms']

        Yields:
            Dictionary that will contain 'elapsed_ms' after the block completes.
        """
        result: dict = {}  # Initialize result dictionary to store timing
        start_time = time.perf_counter()  # Record high-precision start time
        try:  # Execute the wrapped code block
            yield result  # Yield the result dict for the caller to use
        finally:  # Always calculate elapsed time, even if exception occurs
            end_time = time.perf_counter()  # Record high-precision end time
            elapsed_seconds = end_time - start_time  # Calculate duration in seconds
            result["elapsed_ms"] = round(elapsed_seconds * 1000, 2)  # Convert to milliseconds

    @staticmethod  # No instance state needed for this method
    def calculate_throughput(item_count: int, elapsed_ms: float) -> float:
        """Calculate processing throughput as items per second.

        Args:
            item_count: Number of items processed.
            elapsed_ms: Time taken in milliseconds.

        Returns:
            Throughput in items per second, or 0.0 if elapsed_ms is zero.
        """
        if elapsed_ms <= 0:  # Guard against division by zero
            return 0.0  # Return zero throughput for zero or negative time
        elapsed_seconds = elapsed_ms / 1000.0  # Convert milliseconds to seconds
        return round(item_count / elapsed_seconds, 2)  # Calculate and round throughput
