"""
Query optimization utilities for SLAF

This module contains optimized query strategies including:
- Adaptive batching for large scattered ID sets
- Range vs IN clause optimization
- CTE optimization for complex queries
- Selector-based query building for submatrix operations
- Histogram-based gene indexing for efficient gene queries
"""

from typing import Any

import numpy as np
import polars as pl


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
    def _normalize_index(index: int, max_size: int) -> int:
        """
        Normalize a single index, handling negative indices and bounds

        Args:
            index: Index to normalize
            max_size: Maximum size of the dimension

        Returns:
            Normalized index
        """
        if index < 0:
            index = max_size + index
        return max(0, min(index, max_size))

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
                normalized_idx = QueryOptimizer._normalize_index(idx, max_size)
                if normalized_idx < max_size:
                    list_indices.append(normalized_idx)
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
        normalized_idx = QueryOptimizer._normalize_index(selector, max_size)
        if normalized_idx < max_size:
            return f"{entity_type}_integer_id = {normalized_idx}"
        else:
            return "FALSE"

    @staticmethod
    def _process_selector(
        selector: Any, entity_type: str, max_size: int, size_name: str
    ) -> str:
        """
        Process any type of selector and return SQL condition

        Args:
            selector: Selector of any supported type
            entity_type: Either 'cell' or 'gene'
            max_size: Maximum size of the dimension
            size_name: Name of the size parameter for error messages

        Returns:
            SQL WHERE condition string
        """
        if selector is None:
            return "TRUE"
        elif isinstance(selector, slice):
            start, stop, step = QueryOptimizer._normalize_slice_indices(
                selector, max_size
            )
            return QueryOptimizer._build_slice_condition(start, stop, step, entity_type)
        elif isinstance(selector, list | np.ndarray):
            return QueryOptimizer._build_list_condition(selector, entity_type, max_size)
        else:
            # Single index
            return QueryOptimizer._build_single_index_condition(
                selector, entity_type, max_size
            )

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
        # Validate required parameters
        if cell_selector is not None and cell_count is None:
            raise ValueError("cell_count must be provided for cell selectors")
        if gene_selector is not None and gene_count is None:
            raise ValueError("gene_count must be provided for gene selectors")

        # Use unified selector processing
        cell_condition = QueryOptimizer._process_selector(
            cell_selector, "cell", cell_count or 0, "cell_count"
        )
        gene_condition = QueryOptimizer._process_selector(
            gene_selector, "gene", gene_count or 0, "gene_count"
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


class RowIndexMapper:
    """
    Simple row index mapping using cumulative sums for cell-based queries.

    This class provides efficient methods to get row ranges for cells
    using cumulative sums of gene counts. For gene-based queries,
    it falls back to table scans with WHERE clauses.

    Key Features:
        - Fast cell row range calculation using cumulative sums
        - Efficient gene filtering in Polars after cell loading
        - Simple and maintainable implementation

    Examples:
        >>> mapper = RowIndexMapper(slaf_array)
        >>> cell_ranges = mapper.get_cell_row_ranges([1, 2, 3])
        >>> expression_data = slaf_array.expression.take(cell_ranges)
    """

    def __init__(self, slaf_array):
        """
        Initialize RowIndexMapper with SLAF array.

        Args:
            slaf_array: SLAFArray instance containing the dataset
        """
        self.slaf_array = slaf_array
        self._cell_count = slaf_array.shape[0]
        self._gene_count = slaf_array.shape[1]

    def _normalize_selector_indices(self, selector: Any, max_size: int) -> list[int]:
        """
        Normalize selector indices to a list of valid indices

        Args:
            selector: Selector of any supported type
            max_size: Maximum size of the dimension

        Returns:
            List of normalized indices
        """
        if selector is None:
            return list(range(max_size))

        if isinstance(selector, int):
            # Single index
            normalized_idx = QueryOptimizer._normalize_index(selector, max_size)
            if normalized_idx < max_size:
                return [normalized_idx]
            else:
                raise ValueError(f"Cell index {selector} out of bounds")

        elif isinstance(selector, slice):
            # Slice selector
            start, stop, step = QueryOptimizer._normalize_slice_indices(
                selector, max_size
            )
            return list(range(start, stop, step))

        elif isinstance(selector, list):
            # List of indices
            if len(selector) == 0:
                return []

            indices_array = np.array(selector)
            # Handle negative indices vectorized
            negative_mask = indices_array < 0
            indices_array[negative_mask] += max_size

            # Check bounds
            if np.any((indices_array < 0) | (indices_array >= max_size)):
                invalid_indices = indices_array[
                    (indices_array < 0) | (indices_array >= max_size)
                ]
                raise ValueError(f"Indices out of bounds: {invalid_indices.tolist()}")

            return indices_array.tolist()

        elif isinstance(selector, np.ndarray) and selector.dtype == bool:
            # Boolean mask
            if len(selector) != max_size:
                raise ValueError(
                    f"Boolean mask length {len(selector)} doesn't match size {max_size}"
                )
            return np.where(selector)[0].tolist()

        else:
            raise ValueError(f"Unsupported selector type: {type(selector)}")

    def get_cell_row_ranges(self, cell_integer_ids: list[int]) -> list[int]:
        """
        Get row indices for specific cell integer IDs using cumulative sums.

        Args:
            cell_integer_ids: List of cell integer IDs

        Returns:
            List of row indices covering all expression records for the cells
        """
        start_indices = self.slaf_array._cell_start_index

        # Find original row positions for the requested cell_integer_ids (vectorized)
        # Create a mapping from cell_integer_id to original row position
        cell_id_to_position = (
            self.slaf_array.obs.with_row_index()
            .select(["cell_integer_id", "index"])
            .to_dict(as_series=False)
        )

        # Convert to numpy arrays for fast vectorized operations
        all_cell_ids = np.array(cell_id_to_position["cell_integer_id"])
        all_positions = np.array(cell_id_to_position["index"])

        # Find positions for requested cell IDs (vectorized)
        requested_cell_ids = np.array(cell_integer_ids)

        # Use numpy's searchsorted for O(log n) lookup
        # First, sort the original data for binary search
        sort_idx = np.argsort(all_cell_ids)
        sorted_cell_ids = all_cell_ids[sort_idx]
        sorted_positions = all_positions[sort_idx]

        # Find positions for requested cell IDs
        found_positions = []
        for cell_id in requested_cell_ids:
            # Binary search for the cell_id
            idx = np.searchsorted(sorted_cell_ids, cell_id)
            if idx < len(sorted_cell_ids) and sorted_cell_ids[idx] == cell_id:
                found_positions.append(sorted_positions[idx])
            else:
                raise ValueError(f"Cell integer ID {cell_id} not found in dataset")

        # Convert to numpy array for vectorized operations
        cell_positions = np.array(found_positions)

        # Get start and end indices for all cells at once (vectorized)
        start_idx_array = start_indices.gather(pl.Series(cell_positions)).to_numpy()
        end_idx_array = start_indices.gather(pl.Series(cell_positions + 1)).to_numpy()

        # Calculate row ranges for all cells at once (vectorized)
        # Use numpy's broadcast operations to create all ranges efficiently
        if len(start_idx_array) == 0:
            return []

        # Calculate the length of each range
        range_lengths = end_idx_array - start_idx_array

        # Create a single array with all the ranges using numpy's broadcast operations
        total_length = np.sum(range_lengths)
        if total_length == 0:
            return []

        # Use list comprehension for maximum efficiency
        # This is faster than explicit for loops and more Pythonic
        ranges = [
            np.arange(start, end)
            for start, end in zip(start_idx_array, end_idx_array, strict=False)
        ]
        result = np.concatenate(ranges) if ranges else np.array([], dtype=np.int64)

        return result.tolist()

    def get_cell_row_ranges_by_selector(self, cell_selector) -> list[int]:
        """
        Get row indices for cell selector.

        Args:
            cell_selector: Cell selector (slice, list, boolean mask, int, or None)

        Returns:
            List of row indices for the selected cells
        """
        # Use unified selector processing
        cell_indices = self._normalize_selector_indices(cell_selector, self._cell_count)

        if not cell_indices:
            return []

        # Get cell integer IDs vectorized
        cell_integer_ids = (
            self.slaf_array.obs["cell_integer_id"]
            .gather(pl.Series(cell_indices))
            .to_list()
        )
        return self.get_cell_row_ranges(cell_integer_ids)


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
