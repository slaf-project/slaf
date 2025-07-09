"""
Query optimization utilities for SLAF

This module contains optimized query strategies including:
- Adaptive batching for large scattered ID sets
- Range vs IN clause optimization
- CTE optimization for complex queries
- Selector-based query building for submatrix operations
"""

from typing import Any

import numpy as np


class QueryOptimizer:
    """Optimized query strategies for SLAF operations"""

    @staticmethod
    def is_consecutive(ids: list[int]) -> bool:
        """Check if integer IDs form a consecutive sequence"""
        if len(ids) <= 1:
            return True
        sorted_ids = sorted(ids)
        return all(
            sorted_ids[i + 1] - sorted_ids[i] == 1 for i in range(len(sorted_ids) - 1)
        )

    @staticmethod
    def adaptive_batch_ids(
        integer_ids: list[int], max_batch_size: int = 100, gap_threshold: int = 10
    ) -> list[list[int]]:
        """
        Create optimal batches based on ID distribution patterns

        Args:
            integer_ids: List of integer IDs to batch
            max_batch_size: Maximum size of each batch
            gap_threshold: Maximum gap between consecutive IDs before starting a new batch

        Returns:
            List of batched ID lists
        """
        if len(integer_ids) <= max_batch_size:
            return [integer_ids]

        sorted_ids = sorted(integer_ids)
        batches = []
        current_batch = [sorted_ids[0]]

        for i in range(1, len(sorted_ids)):
            current_id = sorted_ids[i]
            prev_id = sorted_ids[i - 1]

            # Check if we should start a new batch
            gap_too_large = (current_id - prev_id) > gap_threshold
            batch_too_large = len(current_batch) >= max_batch_size

            if gap_too_large or batch_too_large:
                batches.append(current_batch)
                current_batch = [current_id]
            else:
                current_batch.append(current_id)

        if current_batch:
            batches.append(current_batch)

        return batches

    @staticmethod
    def build_optimized_query(
        entity_ids: list[int],
        entity_type: str,
        use_adaptive_batching: bool = True,
        max_batch_size: int = 100,
    ) -> str:
        """
        Build an optimized SQL query for entity filtering

        Args:
            entity_ids: List of entity IDs to filter by
            entity_type: Either 'cell' or 'gene'
            use_adaptive_batching: Whether to use adaptive batching
            max_batch_size: Maximum batch size for adaptive batching

        Returns:
            Optimized SQL query string
        """
        if not entity_ids:
            return "SELECT * FROM expression WHERE FALSE"

        # Convert to integer IDs if needed (assuming they're already integers)
        integer_ids = entity_ids

        if use_adaptive_batching and len(integer_ids) > max_batch_size:
            # Use adaptive batching for large scattered sets
            batches = QueryOptimizer.adaptive_batch_ids(integer_ids, max_batch_size)

            union_queries = []
            for batch in batches:
                if QueryOptimizer.is_consecutive(batch):
                    min_id, max_id = min(batch), max(batch)
                    union_queries.append(
                        f"""
                    SELECT * FROM expression
                    WHERE {entity_type}_integer_id BETWEEN {min_id} AND {max_id}
                    """
                    )
                else:
                    ids_str = ",".join(map(str, batch))
                    union_queries.append(
                        f"""
                    SELECT * FROM expression
                    WHERE {entity_type}_integer_id IN ({ids_str})
                    """
                    )

            return " UNION ALL ".join(union_queries)

        else:
            # Use simple optimization for smaller sets
            if QueryOptimizer.is_consecutive(integer_ids):
                min_id, max_id = min(integer_ids), max(integer_ids)
                return f"""
                SELECT * FROM expression
                WHERE {entity_type}_integer_id BETWEEN {min_id} AND {max_id}
                """
            else:
                ids_str = ",".join(map(str, integer_ids))
                return f"""
                SELECT * FROM expression
                WHERE {entity_type}_integer_id IN ({ids_str})
                """

    @staticmethod
    def _normalize_slice_indices(
        selector: slice, max_size: int
    ) -> tuple[int, int, int]:
        """
        Normalize slice indices, handling negative indices and bounds

        Args:
            selector: Slice object
            max_size: Maximum size of the dimension

        Returns:
            Tuple of (start, stop, step) with normalized indices
        """
        start = selector.start or 0
        stop = selector.stop or max_size
        step = selector.step or 1

        # Handle negative indices
        if start < 0:
            start = max_size + start
        if stop < 0:
            stop = max_size + stop

        # Clamp bounds
        start = max(0, min(start, max_size))
        stop = max(0, min(stop, max_size))

        return start, stop, step

    @staticmethod
    def _build_slice_condition(
        start: int, stop: int, step: int, entity_type: str
    ) -> str:
        """
        Build SQL condition for slice-based selection

        Args:
            start: Start index
            stop: Stop index
            step: Step size
            entity_type: Either 'cell' or 'gene'

        Returns:
            SQL WHERE condition string
        """
        if step == 1:
            return f"{entity_type}_integer_id >= {start} AND {entity_type}_integer_id < {stop}"
        elif step <= 10:
            # Use modulo arithmetic for reasonable step sizes
            return (
                f"{entity_type}_integer_id >= {start} AND {entity_type}_integer_id < {stop} "
                f"AND ({entity_type}_integer_id - {start}) % {step} = 0"
            )
        else:
            # Handle large step sizes by generating explicit indices
            indices = list(range(start, stop, step))
            if len(indices) == 0:
                return "FALSE"
            if len(indices) <= 1000:  # Limit IN clause size
                return f"{entity_type}_integer_id IN ({','.join(map(str, indices))})"
            else:
                # For very large step sizes, fall back to modulo arithmetic
                return (
                    f"{entity_type}_integer_id >= {start} AND {entity_type}_integer_id < {stop} "
                    f"AND ({entity_type}_integer_id - {start}) % {step} = 0"
                )

    @staticmethod
    def _build_list_condition(
        selector: list | np.ndarray, entity_type: str, max_size: int
    ) -> str:
        """
        Build SQL condition for list/array-based selection

        Args:
            selector: List or numpy array of indices
            entity_type: Either 'cell' or 'gene'
            max_size: Maximum size of the dimension

        Returns:
            SQL WHERE condition string
        """
        if isinstance(selector, np.ndarray) and selector.dtype == bool:
            indices = np.where(selector)[0]
            if len(indices) == 0:
                return "FALSE"
            bool_indices_list = indices.tolist()
        else:
            # List of indices - handle negative indices
            list_indices: list[int] = []
            for idx in selector:
                if idx < 0:
                    idx = max_size + idx
                if 0 <= idx < max_size:
                    list_indices.append(idx)
            if len(list_indices) == 0:
                return "FALSE"

        # Build condition for the indices
        final_indices = (
            bool_indices_list if "bool_indices_list" in locals() else list_indices
        )
        if QueryOptimizer.is_consecutive(final_indices):
            min_id, max_id = min(final_indices), max(final_indices)
            return f"{entity_type}_integer_id BETWEEN {min_id} AND {max_id}"
        else:
            ids_str = ",".join(map(str, final_indices))
            return f"{entity_type}_integer_id IN ({ids_str})"

    @staticmethod
    def _build_single_index_condition(
        selector: int, entity_type: str, max_size: int
    ) -> str:
        """
        Build SQL condition for single index selection

        Args:
            selector: Single index
            entity_type: Either 'cell' or 'gene'
            max_size: Maximum size of the dimension

        Returns:
            SQL WHERE condition string
        """
        # Handle negative index
        if selector < 0:
            selector = max_size + selector
        if 0 <= selector < max_size:
            return f"{entity_type}_integer_id = {selector}"
        else:
            return "FALSE"

    @staticmethod
    def build_submatrix_query(
        cell_selector: Any | None = None,
        gene_selector: Any | None = None,
        cell_count: int | None = None,
        gene_count: int | None = None,
    ) -> str:
        """
        Build an optimized SQL query for submatrix selection

        Args:
            cell_selector: Cell selector (slice, list, boolean mask, int, or None)
            gene_selector: Gene selector (slice, list, boolean mask, int, or None)
            cell_count: Total number of cells (for bounds checking)
            gene_count: Total number of genes (for bounds checking)

        Returns:
            Optimized SQL query string
        """
        # Handle cell selector
        if cell_selector is None:
            cell_condition = "TRUE"
        elif isinstance(cell_selector, slice):
            if cell_count is None:
                raise ValueError("cell_count must be provided for slice selectors")
            start, stop, step = QueryOptimizer._normalize_slice_indices(
                cell_selector, cell_count
            )
            cell_condition = QueryOptimizer._build_slice_condition(
                start, stop, step, "cell"
            )
        elif isinstance(cell_selector, list | np.ndarray):
            if cell_count is None:
                raise ValueError("cell_count must be provided for list/array selectors")
            cell_condition = QueryOptimizer._build_list_condition(
                cell_selector, "cell", cell_count
            )
        else:
            # Single index
            if cell_count is None:
                raise ValueError("cell_count must be provided for index selectors")
            cell_condition = QueryOptimizer._build_single_index_condition(
                cell_selector, "cell", cell_count
            )

        # Handle gene selector
        if gene_selector is None:
            gene_condition = "TRUE"
        elif isinstance(gene_selector, slice):
            if gene_count is None:
                raise ValueError("gene_count must be provided for slice selectors")
            start, stop, step = QueryOptimizer._normalize_slice_indices(
                gene_selector, gene_count
            )
            gene_condition = QueryOptimizer._build_slice_condition(
                start, stop, step, "gene"
            )
        elif isinstance(gene_selector, list | np.ndarray):
            if gene_count is None:
                raise ValueError("gene_count must be provided for list/array selectors")
            gene_condition = QueryOptimizer._build_list_condition(
                gene_selector, "gene", gene_count
            )
        else:
            # Single index
            if gene_count is None:
                raise ValueError("gene_count must be provided for index selectors")
            gene_condition = QueryOptimizer._build_single_index_condition(
                gene_selector, "gene", gene_count
            )

        # Build final query
        where_clause = f"{cell_condition} AND {gene_condition}"
        return f"SELECT * FROM expression WHERE {where_clause}"

    @staticmethod
    def build_cte_query(entity_ids: list[int], entity_type: str) -> str:
        """
        Build a CTE-optimized query for large scattered ID sets

        Args:
            entity_ids: List of entity IDs to filter by
            entity_type: Either 'cell' or 'gene'

        Returns:
            CTE-optimized SQL query string
        """
        ids_str = ",".join(map(str, entity_ids))

        return f"""
        WITH filtered_expression AS (
            SELECT * FROM expression
            WHERE {entity_type}_integer_id IN ({ids_str})
        )
        SELECT * FROM filtered_expression
        """

    @staticmethod
    def estimate_query_strategy(
        entity_ids: list[int],
        consecutive_threshold: int = 50,
        batching_threshold: int = 100,
    ) -> str:
        """
        Estimate the best query strategy for a given set of entity IDs

        Args:
            entity_ids: List of entity IDs
            consecutive_threshold: Threshold for using BETWEEN vs IN
            batching_threshold: Threshold for using adaptive batching

        Returns:
            Strategy name: 'consecutive', 'scattered', 'batched', or 'cte'
        """
        if not entity_ids:
            return "empty"

        if len(entity_ids) <= 1:
            return "single"

        if QueryOptimizer.is_consecutive(entity_ids):
            return "consecutive"

        if len(entity_ids) > batching_threshold:
            return "batched"
        elif len(entity_ids) > consecutive_threshold:
            return "cte"
        else:
            return "scattered"


class PerformanceMetrics:
    """Track and analyze query performance"""

    def __init__(self):
        self.query_times = {}
        self.query_counts = {}

    def record_query(self, strategy: str, entity_count: int, query_time: float):
        """Record query performance metrics"""
        key = f"{strategy}_{entity_count}"
        if key not in self.query_times:
            self.query_times[key] = []
            self.query_counts[key] = 0

        self.query_times[key].append(query_time)
        self.query_counts[key] += 1

    def get_average_time(self, strategy: str, entity_count: int) -> float | None:
        """Get average query time for a strategy and entity count"""
        key = f"{strategy}_{entity_count}"
        if key in self.query_times and self.query_times[key]:
            return np.mean(self.query_times[key])
        return None

    def get_performance_summary(self) -> dict:
        """Get performance summary across all strategies"""
        summary: dict[str, dict[int, dict[str, float | int]]] = {}
        for key in self.query_times:
            if self.query_times[key]:
                strategy, count = key.split("_", 1)
                if strategy not in summary:
                    summary[strategy] = {}

                summary[strategy][int(count)] = {
                    "avg_time": float(np.mean(self.query_times[key])),
                    "min_time": float(np.min(self.query_times[key])),
                    "max_time": float(np.max(self.query_times[key])),
                    "count": self.query_counts[key],
                }

        return summary
