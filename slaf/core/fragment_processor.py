"""
Fragment-based processing for SLAF datasets.

This module provides fragment-based processing capabilities for large SLAF datasets,
enabling memory-efficient operations on datasets with millions of cells.
"""

import concurrent.futures
import threading
from typing import Any

import numpy as np
import polars as pl
from loguru import logger


class FragmentProcessor:
    """
    Fragment-based processing for large SLAF datasets.

    This class provides efficient fragment-based processing for large datasets
    by breaking them into manageable chunks and processing them in parallel.
    """

    def __init__(
        self,
        slaf_array,
        cell_selector: Any | None = None,
        gene_selector: Any | None = None,
        max_workers: int = 4,
        enable_caching: bool = True,
    ):
        """
        Initialize FragmentProcessor.

        Args:
            slaf_array: SLAFArray instance
            cell_selector: Cell selector for subsetting
            gene_selector: Gene selector for subsetting
            max_workers: Maximum number of parallel workers
            enable_caching: Whether to enable operation caching
        """
        self.slaf_array = slaf_array
        self.cell_selector = cell_selector
        self.gene_selector = gene_selector
        self.max_workers = max_workers
        self.enable_caching = enable_caching

        # Get fragments from the expression dataset
        self.fragments = slaf_array.expression.get_fragments()
        self.n_fragments = len(self.fragments)

        # Initialize cache
        self._cache: dict[str, pl.LazyFrame] = {}
        self._cache_lock = threading.Lock()

        logger.debug(f"Initialized FragmentProcessor with {self.n_fragments} fragments")

    def clear_cache(self) -> None:
        """Clear the operation cache."""
        with self._cache_lock:
            self._cache.clear()

    def build_lazy_pipeline(self, operation: str, **kwargs) -> pl.LazyFrame:
        """
        Build lazy computation graph for the specified operation.

        This method maintains backward compatibility by implementing the actual logic.

        Args:
            operation: Operation to perform ("normalize_total", "log1p", "mean", "sum")
            **kwargs: Operation-specific parameters

        Returns:
            LazyFrame representing the computation graph
        """
        # Check cache first (inlined cache logic)
        if self.enable_caching:
            # Create a deterministic cache key
            key_parts = [
                operation,
                str(self.cell_selector),
                str(self.gene_selector),
            ]
            # Add sorted kwargs for deterministic ordering
            if kwargs:
                sorted_kwargs = sorted(kwargs.items())
                key_parts.extend([f"{k}={v}" for k, v in sorted_kwargs])
            cache_key = "|".join(key_parts)

            # Get cached result
            with self._cache_lock:
                cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached result for operation: {operation}")
                return cached_result

        # Special handling for normalize_total - need global cell totals first
        if operation == "normalize_total":
            result = self._build_normalize_total_pipeline(**kwargs)
        else:
            # Create lazy frames for each fragment
            lazy_fragments = []

            for i, fragment in enumerate(self.fragments):
                logger.debug(
                    f"Building lazy pipeline for fragment {i + 1}/{self.n_fragments}"
                )

                # Create lazy dataframe from fragment
                lazy_df = pl.scan_pyarrow_dataset(fragment)

                # Apply cell and gene selectors to the fragment
                lazy_df = self._apply_selectors_to_fragment(lazy_df)

                # Apply operation to lazy dataframe
                processed_df = self._apply_operation(lazy_df, operation, **kwargs)
                lazy_fragments.append(processed_df)

            # Return lazy concatenation (Polars handles fusion automatically)
            if len(lazy_fragments) == 1:
                result = lazy_fragments[0]
            else:
                result = pl.concat(lazy_fragments, how="vertical")

        # Cache the result (inlined cache logic)
        if self.enable_caching:
            with self._cache_lock:
                self._cache[cache_key] = result
        return result

    def build_lazy_pipeline_parallel(self, operation: str, **kwargs) -> pl.LazyFrame:
        """
        Build lazy computation graph using parallel processing.

        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            LazyFrame representing the computation graph
        """
        # Special handling for normalize_total - need global cell totals first
        if operation == "normalize_total":
            return self._build_normalize_total_pipeline_parallel(**kwargs)

        # Process fragments in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submit fragment processing tasks
            future_to_fragment = {
                executor.submit(
                    self._process_single_fragment, fragment, operation, **kwargs
                ): i
                for i, fragment in enumerate(self.fragments)
            }

            # Collect results
            lazy_fragments = []
            for future in concurrent.futures.as_completed(future_to_fragment):
                fragment_idx = future_to_fragment[future]
                try:
                    result = future.result()
                    lazy_fragments.append(result)
                    logger.debug(
                        f"Completed fragment {fragment_idx + 1}/{self.n_fragments}"
                    )
                except Exception as e:
                    logger.error(f"Fragment {fragment_idx} failed: {e}")
                    raise

        # Return lazy concatenation
        if len(lazy_fragments) == 1:
            return lazy_fragments[0]
        else:
            return pl.concat(lazy_fragments, how="vertical")

    def build_lazy_pipeline_smart(self, operation: str, **kwargs) -> pl.LazyFrame:
        """
        Build lazy computation graph with smart strategy selection.

        Automatically chooses between sequential and parallel processing based on
        dataset size, operation type, and available resources.

        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            LazyFrame representing the computation graph
        """
        # Check cache first (inlined cache logic)
        if self.enable_caching:
            # Create a deterministic cache key
            key_parts = [
                operation,
                str(self.cell_selector),
                str(self.gene_selector),
            ]
            # Add sorted kwargs for deterministic ordering
            if kwargs:
                sorted_kwargs = sorted(kwargs.items())
                key_parts.extend([f"{k}={v}" for k, v in sorted_kwargs])
            cache_key = "|".join(key_parts)

            # Get cached result
            with self._cache_lock:
                cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached result for operation: {operation}")
                return cached_result

        # Determine optimal strategy
        use_parallel = self._should_use_parallel(operation, **kwargs)

        if use_parallel:
            logger.debug(f"Using parallel processing for operation: {operation}")
            result = self.build_lazy_pipeline_parallel(operation, **kwargs)
        else:
            logger.debug(f"Using sequential processing for operation: {operation}")
            result = self.build_lazy_pipeline(operation, **kwargs)

        # Cache the result (inlined cache logic)
        if self.enable_caching:
            with self._cache_lock:
                self._cache[cache_key] = result
        return result

    def _should_use_parallel(self, operation: str, **kwargs) -> bool:
        """
        Determine if parallel processing should be used.

        Args:
            operation: Operation type
            **kwargs: Operation parameters

        Returns:
            True if parallel processing should be used
        """
        # Always use parallel for normalize_total (most expensive operation)
        if operation == "normalize_total":
            return True

        # Use parallel if we have many fragments and multiple workers
        if self.n_fragments > 10 and self.max_workers > 1:
            return True

        # Use parallel for expensive operations with large datasets
        if operation in ["mean", "sum"] and self.n_fragments > 5:
            return True

        # Use parallel for operations that benefit from batching
        if operation in ["log1p", "compute_matrix"] and self.n_fragments > 8:
            return True

        # Use parallel for operations that can benefit from early termination
        if operation in ["highly_variable_genes", "variance"] and self.n_fragments > 3:
            return True

        # Use sequential for simple operations or small datasets
        return False

    def _process_single_fragment(
        self, fragment, operation: str, **kwargs
    ) -> pl.LazyFrame:
        """
        Process a single fragment.

        Args:
            fragment: Fragment to process
            operation: Operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            LazyFrame for the processed fragment
        """
        # Create lazy dataframe from fragment
        lazy_df = pl.scan_pyarrow_dataset(fragment)

        # Apply cell and gene selectors to the fragment
        lazy_df = self._apply_selectors_to_fragment(lazy_df)

        # Apply operation to lazy dataframe
        return self._apply_operation(lazy_df, operation, **kwargs)

    def _build_normalize_total_pipeline(self, **kwargs) -> pl.LazyFrame:
        """
        Build normalize_total pipeline with global cell totals.

        This method computes global cell totals first, then applies normalization
        to each fragment using the global totals.
        """
        target_sum = kwargs.get("target_sum", 1e4)

        # Step 1: Compute global cell totals across all fragments
        # We need to materialize the global totals first to ensure they're computed
        global_totals_fragments = []
        for i, fragment in enumerate(self.fragments):
            logger.debug(
                f"Computing cell totals for fragment {i + 1}/{self.n_fragments}"
            )
            lazy_df = pl.scan_pyarrow_dataset(fragment)
            cell_totals = (
                lazy_df.group_by("cell_integer_id")
                .agg(pl.col("value").sum().alias("fragment_total"))
                .collect()
            )  # Materialize this fragment's totals
            global_totals_fragments.append(cell_totals)

        # Combine all fragment totals to get global totals
        if len(global_totals_fragments) == 1:
            global_cell_totals = global_totals_fragments[0]
        else:
            global_cell_totals = pl.concat(global_totals_fragments, how="vertical")

        # Sum up totals for each cell across all fragments
        global_cell_totals = global_cell_totals.group_by("cell_integer_id").agg(
            pl.col("fragment_total").sum().alias("global_total")
        )

        # Step 2: Apply normalization to each fragment using global totals
        normalized_fragments = []
        for i, fragment in enumerate(self.fragments):
            logger.debug(f"Normalizing fragment {i + 1}/{self.n_fragments}")
            lazy_df = pl.scan_pyarrow_dataset(fragment)

            # Apply cell and gene selectors to the fragment
            lazy_df = self._apply_selectors_to_fragment(lazy_df)

            # Convert global cell totals to LazyFrame for proper join
            global_totals_lazy = pl.LazyFrame(global_cell_totals)

            # Join with global cell totals (materialized)
            normalized_df = (
                lazy_df.join(global_totals_lazy, on="cell_integer_id", how="left")
                .with_columns(
                    [
                        (pl.col("value") / pl.col("global_total") * target_sum).alias(
                            "normalized_value"
                        )
                    ]
                )
                .select(
                    [
                        "cell_integer_id",
                        "gene_integer_id",
                        pl.col("normalized_value").alias("value"),
                    ]
                )
            )
            normalized_fragments.append(normalized_df)

        # Return concatenated result
        if len(normalized_fragments) == 1:
            return normalized_fragments[0]
        else:
            return pl.concat(normalized_fragments, how="vertical")

    def _build_normalize_total_pipeline_parallel(self, **kwargs) -> pl.LazyFrame:
        """
        Build normalize_total pipeline with parallel global cell totals computation.

        This method computes global cell totals in parallel, then applies normalization
        to each fragment using the global totals.
        """
        target_sum = kwargs.get("target_sum", 1e4)

        # Step 1: Compute global cell totals across all fragments in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submit cell totals computation tasks
            future_to_fragment = {
                executor.submit(self._compute_fragment_cell_totals, fragment): i
                for i, fragment in enumerate(self.fragments)
            }

            # Collect results
            global_totals_fragments = []
            for future in concurrent.futures.as_completed(future_to_fragment):
                fragment_idx = future_to_fragment[future]
                try:
                    result = future.result()
                    global_totals_fragments.append(result)
                    logger.debug(
                        f"Completed cell totals for fragment {fragment_idx + 1}/{self.n_fragments}"
                    )
                except Exception as e:
                    logger.error(f"Fragment {fragment_idx} cell totals failed: {e}")
                    raise

        # Combine all fragment totals to get global totals
        if len(global_totals_fragments) == 1:
            global_cell_totals = global_totals_fragments[0]
        else:
            global_cell_totals = pl.concat(global_totals_fragments, how="vertical")

        # Sum up totals for each cell across all fragments
        global_cell_totals = global_cell_totals.group_by("cell_integer_id").agg(
            pl.col("fragment_total").sum().alias("global_total")
        )

        # Step 2: Apply normalization to each fragment using global totals in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submit normalization tasks
            future_to_fragment = {
                executor.submit(
                    self._normalize_fragment_with_global_totals,
                    fragment,
                    global_cell_totals,
                    target_sum,
                ): i
                for i, fragment in enumerate(self.fragments)
            }

            # Collect results
            normalized_fragments = []
            for future in concurrent.futures.as_completed(future_to_fragment):
                fragment_idx = future_to_fragment[future]
                try:
                    result = future.result()
                    normalized_fragments.append(result)
                    logger.debug(
                        f"Completed normalization for fragment {fragment_idx + 1}/{self.n_fragments}"
                    )
                except Exception as e:
                    logger.error(f"Fragment {fragment_idx} normalization failed: {e}")
                    raise

        # Return concatenated result as LazyFrame
        if len(normalized_fragments) == 1:
            return pl.LazyFrame(normalized_fragments[0])
        else:
            return pl.LazyFrame(pl.concat(normalized_fragments, how="vertical"))

    def _compute_fragment_cell_totals(self, fragment) -> pl.DataFrame:
        """
        Compute cell totals for a single fragment.

        Args:
            fragment: Fragment to process

        Returns:
            DataFrame with cell totals for this fragment
        """
        lazy_df = pl.scan_pyarrow_dataset(fragment)
        return (
            lazy_df.group_by("cell_integer_id")
            .agg(pl.col("value").sum().alias("fragment_total"))
            .collect()
        )

    def _normalize_fragment_with_global_totals(
        self, fragment, global_cell_totals: pl.DataFrame, target_sum: float
    ) -> pl.DataFrame:
        """
        Normalize a fragment using global cell totals.

        Args:
            fragment: Fragment to normalize
            global_cell_totals: Global cell totals DataFrame
            target_sum: Target sum for normalization

        Returns:
            DataFrame with normalized values
        """
        lazy_df = pl.scan_pyarrow_dataset(fragment)

        # Apply cell and gene selectors to the fragment
        lazy_df = self._apply_selectors_to_fragment(lazy_df)

        # Convert global cell totals to LazyFrame for proper join
        global_totals_lazy = pl.LazyFrame(global_cell_totals)

        # Join with global cell totals and normalize
        return (
            lazy_df.join(global_totals_lazy, on="cell_integer_id", how="left")
            .with_columns(
                [
                    (pl.col("value") / pl.col("global_total") * target_sum).alias(
                        "normalized_value"
                    )
                ]
            )
            .select(
                [
                    "cell_integer_id",
                    "gene_integer_id",
                    pl.col("normalized_value").alias("value"),
                ]
            )
            .collect()
        )

    def _apply_operation(
        self, lazy_df: pl.LazyFrame, operation: str, **kwargs
    ) -> pl.LazyFrame:
        """
        Apply operation to lazy dataframe.

        Args:
            lazy_df: Polars LazyFrame to process
            operation: Operation to apply
            **kwargs: Operation-specific parameters

        Returns:
            LazyFrame with applied operation
        """
        if operation == "normalize_total":
            target_sum = kwargs.get("target_sum", 1e4)
            return self._apply_normalize_total(lazy_df, target_sum)
        elif operation == "log1p":
            return self._apply_log1p(lazy_df)
        elif operation == "mean":
            axis = kwargs.get("axis", None)
            return self._apply_mean(lazy_df, axis)
        elif operation == "sum":
            axis = kwargs.get("axis", None)
            return self._apply_sum(lazy_df, axis)
        elif operation == "compute_matrix":
            return self._apply_compute_matrix(lazy_df)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _apply_compute_matrix(self, lazy_df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Apply matrix computation operation to lazy dataframe.

        This operation prepares the data for matrix conversion by ensuring
        all necessary columns are present and properly formatted.

        Args:
            lazy_df: Polars LazyFrame to process

        Returns:
            LazyFrame ready for matrix conversion
        """
        # For matrix computation, we just need to ensure the data is properly formatted
        # The actual matrix conversion happens in the calling code
        return lazy_df.select(["cell_integer_id", "gene_integer_id", "value"])

    def compute(self, lazy_pipeline: pl.LazyFrame) -> pl.DataFrame:
        """
        Execute the lazy pipeline and return results.

        Args:
            lazy_pipeline: Polars LazyFrame to execute

        Returns:
            Polars DataFrame with results

        Examples:
            >>> # Build and execute pipeline
            >>> pipeline = processor.build_lazy_pipeline("normalize_total", target_sum=1e4)
            >>> result = processor.compute(pipeline)
            >>> print(f"Computed {len(result)} records")
            Computed 5000000 records
        """
        logger.info("Executing lazy pipeline...")
        return lazy_pipeline.collect()

    def _apply_normalize_total(
        self, lazy_df: pl.LazyFrame, target_sum: float
    ) -> pl.LazyFrame:
        """
        Apply normalize_total operation to lazy dataframe.

        This implementation handles cells that are fully contained within a fragment
        by normalizing them using the fragment's data. For cells that span multiple
        fragments (boundary cells), we need special handling.

        Args:
            lazy_df: Polars LazyFrame to process
            target_sum: Target sum for normalization (default: 1e4)

        Returns:
            LazyFrame with normalized values
        """
        # For now, we'll use the simple approach and let the FragmentProcessor
        # handle the global coordination. This method will be called per fragment,
        # but we need to ensure the results can be combined correctly.

        # Compute cell totals within this fragment
        cell_totals = lazy_df.group_by("cell_integer_id").agg(
            pl.col("value").sum().alias("fragment_total")
        )

        # Join back and normalize
        return (
            lazy_df.join(cell_totals, on="cell_integer_id", how="left")
            .with_columns(
                [
                    (pl.col("value") / pl.col("fragment_total") * target_sum).alias(
                        "normalized_value"
                    )
                ]
            )
            .select(
                [
                    "cell_integer_id",
                    "gene_integer_id",
                    pl.col("normalized_value").alias("value"),
                ]
            )
        )

    def _apply_log1p(self, lazy_df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Apply log1p operation to lazy dataframe.

        Applies log1p transformation to expression values: log(1 + value).

        Args:
            lazy_df: Polars LazyFrame to process

        Returns:
            LazyFrame with log1p transformed values
        """
        return lazy_df.with_columns(
            [pl.col("value").log1p().alias("log1p_value")]
        ).select(
            ["cell_integer_id", "gene_integer_id", pl.col("log1p_value").alias("value")]
        )

    def _get_cell_fragment_mapping(self) -> dict:
        """
        Get mapping of which cells appear in which fragments.

        Returns:
            Dictionary mapping cell_integer_id to list of fragment indices
        """
        cell_fragment_map: dict[int, list[int]] = {}

        for fragment_idx, fragment in enumerate(self.fragments):
            # Get unique cell IDs in this fragment
            lazy_df = pl.scan_pyarrow_dataset(fragment)
            cell_ids = lazy_df.select("cell_integer_id").unique().collect()

            for cell_id in cell_ids["cell_integer_id"]:
                if cell_id not in cell_fragment_map:
                    cell_fragment_map[cell_id] = []
                cell_fragment_map[cell_id].append(fragment_idx)

        return cell_fragment_map

    def _get_boundary_cells(self) -> set:
        """
        Get cells that span multiple fragments (boundary cells).

        Returns:
            Set of cell_integer_ids that appear in multiple fragments
        """
        cell_fragment_map = self._get_cell_fragment_mapping()
        boundary_cells = {
            cell_id
            for cell_id, fragments in cell_fragment_map.items()
            if len(fragments) > 1
        }
        return boundary_cells

    def _apply_selectors_to_fragment(self, lazy_df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Apply cell and gene selectors to a fragment.

        Args:
            lazy_df: LazyFrame to filter

        Returns:
            Filtered LazyFrame
        """
        # Apply cell selector if specified
        if self.cell_selector is not None:
            if isinstance(self.cell_selector, slice):
                start = self.cell_selector.start or 0
                stop = self.cell_selector.stop or self.slaf_array.shape[0]
                step = self.cell_selector.step or 1

                # Handle negative indices
                if start < 0:
                    start = self.slaf_array.shape[0] + start
                if stop < 0:
                    stop = self.slaf_array.shape[0] + stop

                # Clamp bounds
                start = max(0, min(start, self.slaf_array.shape[0]))
                stop = max(0, min(stop, self.slaf_array.shape[0]))

                # Get cell integer IDs for the slice
                cell_ids = list(range(start, stop, step))
                lazy_df = lazy_df.filter(pl.col("cell_integer_id").is_in(cell_ids))
            elif isinstance(self.cell_selector, list):
                lazy_df = lazy_df.filter(
                    pl.col("cell_integer_id").is_in(self.cell_selector)
                )
            elif isinstance(self.cell_selector, int):
                lazy_df = lazy_df.filter(
                    pl.col("cell_integer_id") == self.cell_selector
                )

        # Apply gene selector if specified
        if self.gene_selector is not None:
            if isinstance(self.gene_selector, slice):
                start = self.gene_selector.start or 0
                stop = self.gene_selector.stop or self.slaf_array.shape[1]
                step = self.gene_selector.step or 1

                # Handle negative indices
                if start < 0:
                    start = self.slaf_array.shape[1] + start
                if stop < 0:
                    stop = self.slaf_array.shape[1] + stop

                # Clamp bounds
                start = max(0, min(start, self.slaf_array.shape[1]))
                stop = max(0, min(stop, self.slaf_array.shape[1]))

                # Get gene integer IDs for the slice
                gene_ids = list(range(start, stop, step))
                lazy_df = lazy_df.filter(pl.col("gene_integer_id").is_in(gene_ids))
            elif isinstance(self.gene_selector, list):
                lazy_df = lazy_df.filter(
                    pl.col("gene_integer_id").is_in(self.gene_selector)
                )
            elif isinstance(self.gene_selector, int):
                lazy_df = lazy_df.filter(
                    pl.col("gene_integer_id") == self.gene_selector
                )

        return lazy_df

    def _apply_mean(self, lazy_df: pl.LazyFrame, axis: int = None) -> pl.LazyFrame:
        """
        Apply mean aggregation to lazy dataframe with proper handling of boundary cells.
        """
        if axis == 0:  # Across cells (gene-wise)
            # Always compute partial sums and counts, then combine later
            # This handles boundary cells automatically
            return lazy_df.group_by("gene_integer_id").agg(
                [
                    pl.col("value").sum().alias("partial_sum"),
                    pl.col("value").count().alias("partial_count"),
                ]
            )
        elif axis == 1:  # Across genes (cell-wise)
            # Always compute partial sums and counts, then combine later
            return lazy_df.group_by("cell_integer_id").agg(
                [
                    pl.col("value").sum().alias("partial_sum"),
                    pl.col("value").count().alias("partial_count"),
                ]
            )
        else:  # Overall mean
            # Always compute partial sums and counts, then combine later
            return lazy_df.select(
                [
                    pl.col("value").sum().alias("partial_sum"),
                    pl.col("value").count().alias("partial_count"),
                ]
            )

    def _apply_sum(self, lazy_df: pl.LazyFrame, axis: int = None) -> pl.LazyFrame:
        """
        Apply sum aggregation to lazy dataframe with proper handling of boundary cells.
        """
        if axis == 0:  # Across cells (gene-wise)
            # For sum, we can just sum the values (partial sums will be combined later)
            return lazy_df.group_by("gene_integer_id").agg(
                pl.col("value").sum().alias("partial_sum")
            )
        elif axis == 1:  # Across genes (cell-wise)
            # For sum, we can just sum the values (partial sums will be combined later)
            return lazy_df.group_by("cell_integer_id").agg(
                pl.col("value").sum().alias("partial_sum")
            )
        else:  # Overall sum
            # For sum, we can just sum the values
            return lazy_df.select(pl.col("value").sum().alias("partial_sum"))

    def _convert_fragment_result_to_array(
        self, result_df: pl.DataFrame, operation: str, axis: int | None = None
    ) -> np.ndarray:
        """
        Convert fragment processing result to numpy array.

        Args:
            result_df: Polars DataFrame with fragment processing results
            operation: Operation type ("mean", "sum", etc.)
            axis: Aggregation axis

        Returns:
            Numpy array with results
        """
        if operation == "mean":
            if axis == 0:  # Gene-wise mean
                # Check if we have partial sums and counts (boundary cells)
                if (
                    "partial_sum" in result_df.columns
                    and "partial_count" in result_df.columns
                ):
                    # Combine partial sums for boundary genes
                    combined = result_df.group_by("gene_integer_id").agg(
                        pl.col("partial_sum").sum().alias("total_sum")
                    )
                    # Compute mean by dividing by total number of cells (including implicit zeros)
                    # This matches scanpy's canonical approach
                    total_cells = self.slaf_array.shape[0]
                    if self.cell_selector is not None:
                        if isinstance(self.cell_selector, slice):
                            start = self.cell_selector.start or 0
                            stop = self.cell_selector.stop or self.slaf_array.shape[0]
                            step = self.cell_selector.step or 1
                            # Handle negative indices
                            if start < 0:
                                start = self.slaf_array.shape[0] + start
                            if stop < 0:
                                stop = self.slaf_array.shape[0] + stop
                            # Clamp bounds
                            start = max(0, min(start, self.slaf_array.shape[0]))
                            stop = max(0, min(stop, self.slaf_array.shape[0]))
                            total_cells = len(range(start, stop, step))
                        elif isinstance(self.cell_selector, list):
                            total_cells = len(self.cell_selector)

                    combined = combined.with_columns(
                        (pl.col("total_sum") / total_cells).alias("mean_value")
                    )
                    # Sort by gene_integer_id and extract values
                    combined = combined.sort("gene_integer_id")

                    # Create array of exact slice size with zero-filling
                    if self.gene_selector is not None:
                        if isinstance(self.gene_selector, slice):
                            start = self.gene_selector.start or 0
                            stop = self.gene_selector.stop or self.slaf_array.shape[1]
                            step = self.gene_selector.step or 1
                            # Handle negative indices
                            if start < 0:
                                start = self.slaf_array.shape[1] + start
                            if stop < 0:
                                stop = self.slaf_array.shape[1] + stop
                            # Clamp bounds
                            start = max(0, min(start, self.slaf_array.shape[1]))
                            stop = max(0, min(stop, self.slaf_array.shape[1]))
                            result_size = len(range(start, stop, step))
                        elif isinstance(self.gene_selector, list):
                            result_size = len(self.gene_selector)
                        else:
                            result_size = self.slaf_array.shape[1]
                    else:
                        result_size = self.slaf_array.shape[1]

                    # Create result array with exact size and zero-filling
                    result_array = np.zeros(result_size)

                    # Map gene IDs to positions in result array
                    if self.gene_selector is not None:
                        if isinstance(self.gene_selector, slice):
                            start = self.gene_selector.start or 0
                            stop = self.gene_selector.stop or self.slaf_array.shape[1]
                            step = self.gene_selector.step or 1
                            # Handle negative indices
                            if start < 0:
                                start = self.slaf_array.shape[1] + start
                            if stop < 0:
                                stop = self.slaf_array.shape[1] + stop
                            # Clamp bounds
                            start = max(0, min(start, self.slaf_array.shape[1]))
                            stop = max(0, min(stop, self.slaf_array.shape[1]))
                            gene_to_pos = {
                                gene_id: i
                                for i, gene_id in enumerate(range(start, stop, step))
                            }
                        elif isinstance(self.gene_selector, list):
                            gene_to_pos = {
                                gene_id: i
                                for i, gene_id in enumerate(self.gene_selector)
                            }
                    else:
                        gene_to_pos = {
                            gene_id: gene_id
                            for gene_id in range(self.slaf_array.shape[1])
                        }

                    # Fill result array with computed values
                    gene_ids = combined["gene_integer_id"].to_numpy()
                    mean_values = combined["mean_value"].to_numpy()
                    for i, gene_id in enumerate(gene_ids):
                        if gene_id in gene_to_pos:
                            pos = gene_to_pos[gene_id]
                            if 0 <= pos < result_size:
                                result_array[pos] = mean_values[i]

                    return result_array.reshape(1, -1)  # Return (1, n_genes) for axis=0
                else:
                    # Direct mean computation
                    result_df = result_df.sort("gene_integer_id")
                    return (
                        result_df["mean_value"].to_numpy().reshape(1, -1)
                    )  # Return (1, n_genes) for axis=0
            elif axis == 1:  # Cell-wise mean
                # Check if we have partial sums and counts
                if (
                    "partial_sum" in result_df.columns
                    and "partial_count" in result_df.columns
                ):
                    # Combine partial sums for cells
                    combined = result_df.group_by("cell_integer_id").agg(
                        pl.col("partial_sum").sum().alias("total_sum")
                    )
                    # Compute mean by dividing by total number of genes (including implicit zeros)
                    # This matches scanpy's canonical approach
                    total_genes = self.slaf_array.shape[1]
                    if self.gene_selector is not None:
                        if isinstance(self.gene_selector, slice):
                            start = self.gene_selector.start or 0
                            stop = self.gene_selector.stop or self.slaf_array.shape[1]
                            step = self.gene_selector.step or 1
                            # Handle negative indices
                            if start < 0:
                                start = self.slaf_array.shape[1] + start
                            if stop < 0:
                                stop = self.slaf_array.shape[1] + stop
                            # Clamp bounds
                            start = max(0, min(start, self.slaf_array.shape[1]))
                            stop = max(0, min(stop, self.slaf_array.shape[1]))
                            total_genes = len(range(start, stop, step))
                        elif isinstance(self.gene_selector, list):
                            total_genes = len(self.gene_selector)

                    combined = combined.with_columns(
                        (pl.col("total_sum") / total_genes).alias("mean_value")
                    )
                    # Sort by cell_integer_id and extract values
                    combined = combined.sort("cell_integer_id")

                    # Create array of exact slice size with zero-filling
                    if self.cell_selector is not None:
                        if isinstance(self.cell_selector, slice):
                            start = self.cell_selector.start or 0
                            stop = self.cell_selector.stop or self.slaf_array.shape[0]
                            step = self.cell_selector.step or 1
                            # Handle negative indices
                            if start < 0:
                                start = self.slaf_array.shape[0] + start
                            if stop < 0:
                                stop = self.slaf_array.shape[0] + stop
                            # Clamp bounds
                            start = max(0, min(start, self.slaf_array.shape[0]))
                            stop = max(0, min(stop, self.slaf_array.shape[0]))
                            result_size = len(range(start, stop, step))
                        elif isinstance(self.cell_selector, list):
                            result_size = len(self.cell_selector)
                        else:
                            result_size = self.slaf_array.shape[0]
                    else:
                        result_size = self.slaf_array.shape[0]

                    # Create result array with exact size and zero-filling
                    result_array = np.zeros(result_size)

                    # Map cell IDs to positions in result array
                    if self.cell_selector is not None:
                        if isinstance(self.cell_selector, slice):
                            start = self.cell_selector.start or 0
                            stop = self.cell_selector.stop or self.slaf_array.shape[0]
                            step = self.cell_selector.step or 1
                            # Handle negative indices
                            if start < 0:
                                start = self.slaf_array.shape[0] + start
                            if stop < 0:
                                stop = self.slaf_array.shape[0] + stop
                            # Clamp bounds
                            start = max(0, min(start, self.slaf_array.shape[0]))
                            stop = max(0, min(stop, self.slaf_array.shape[0]))
                            cell_to_pos = {
                                cell_id: i
                                for i, cell_id in enumerate(range(start, stop, step))
                            }
                        elif isinstance(self.cell_selector, list):
                            cell_to_pos = {
                                cell_id: i
                                for i, cell_id in enumerate(self.cell_selector)
                            }
                    else:
                        cell_to_pos = {
                            cell_id: cell_id
                            for cell_id in range(self.slaf_array.shape[0])
                        }

                    # Fill result array with computed values
                    cell_ids = combined["cell_integer_id"].to_numpy()
                    mean_values = combined["mean_value"].to_numpy()
                    for i, cell_id in enumerate(cell_ids):
                        if cell_id in cell_to_pos:
                            pos = cell_to_pos[cell_id]
                            if 0 <= pos < result_size:
                                result_array[pos] = mean_values[i]

                    return result_array.reshape(-1, 1)  # Return (n_cells, 1) for axis=1
                else:
                    # Direct mean computation
                    result_df = result_df.sort("cell_integer_id")
                    return (
                        result_df["mean_value"].to_numpy().reshape(-1, 1)
                    )  # Return (n_cells, 1) for axis=1
            else:  # Overall mean
                # Check if we have total_sum
                if "total_sum" in result_df.columns:
                    total_sum = result_df["total_sum"].sum()
                    # Compute mean by dividing by total number of elements (including implicit zeros)
                    # This matches scanpy's canonical approach
                    total_cells = self.slaf_array.shape[0]
                    total_genes = self.slaf_array.shape[1]

                    if self.cell_selector is not None:
                        if isinstance(self.cell_selector, slice):
                            start = self.cell_selector.start or 0
                            stop = self.cell_selector.stop or self.slaf_array.shape[0]
                            step = self.cell_selector.step or 1
                            # Handle negative indices
                            if start < 0:
                                start = self.slaf_array.shape[0] + start
                            if stop < 0:
                                stop = self.slaf_array.shape[0] + stop
                            # Clamp bounds
                            start = max(0, min(start, self.slaf_array.shape[0]))
                            stop = max(0, min(stop, self.slaf_array.shape[0]))
                            total_cells = len(range(start, stop, step))
                        elif isinstance(self.cell_selector, list):
                            total_cells = len(self.cell_selector)

                    if self.gene_selector is not None:
                        if isinstance(self.gene_selector, slice):
                            start = self.gene_selector.start or 0
                            stop = self.gene_selector.stop or self.slaf_array.shape[1]
                            step = self.gene_selector.step or 1
                            # Handle negative indices
                            if start < 0:
                                start = self.slaf_array.shape[1] + start
                            if stop < 0:
                                stop = self.slaf_array.shape[1] + stop
                            # Clamp bounds
                            start = max(0, min(start, self.slaf_array.shape[1]))
                            stop = max(0, min(stop, self.slaf_array.shape[1]))
                            total_genes = len(range(start, stop, step))
                        elif isinstance(self.gene_selector, list):
                            total_genes = len(self.gene_selector)

                    total_elements = total_cells * total_genes
                    return np.array([total_sum / total_elements])
                else:
                    return result_df["mean_value"].to_numpy()

        elif operation == "sum":
            if axis == 0:  # Gene-wise sum
                # For sum, we can just sum the partial sums for boundary genes
                if "sum_value" in result_df.columns:
                    result_df = result_df.sort("gene_integer_id")
                else:
                    # Handle case where we might have partial sums
                    combined = result_df.group_by("gene_integer_id").agg(
                        pl.col("partial_sum").sum().alias("sum_value")
                    )
                    result_df = combined.sort("gene_integer_id")

                # Create array of exact slice size with zero-filling
                if self.gene_selector is not None:
                    if isinstance(self.gene_selector, slice):
                        start = self.gene_selector.start or 0
                        stop = self.gene_selector.stop or self.slaf_array.shape[1]
                        step = self.gene_selector.step or 1
                        # Handle negative indices
                        if start < 0:
                            start = self.slaf_array.shape[1] + start
                        if stop < 0:
                            stop = self.slaf_array.shape[1] + stop
                        # Clamp bounds
                        start = max(0, min(start, self.slaf_array.shape[1]))
                        stop = max(0, min(stop, self.slaf_array.shape[1]))
                        result_size = len(range(start, stop, step))
                    elif isinstance(self.gene_selector, list):
                        result_size = len(self.gene_selector)
                    else:
                        result_size = self.slaf_array.shape[1]
                else:
                    result_size = self.slaf_array.shape[1]

                # Create result array with exact size and zero-filling
                result_array = np.zeros(result_size)

                # Map gene IDs to positions in result array
                if self.gene_selector is not None:
                    if isinstance(self.gene_selector, slice):
                        start = self.gene_selector.start or 0
                        stop = self.gene_selector.stop or self.slaf_array.shape[1]
                        step = self.gene_selector.step or 1
                        # Handle negative indices
                        if start < 0:
                            start = self.slaf_array.shape[1] + start
                        if stop < 0:
                            stop = self.slaf_array.shape[1] + stop
                        # Clamp bounds
                        start = max(0, min(start, self.slaf_array.shape[1]))
                        stop = max(0, min(stop, self.slaf_array.shape[1]))
                        gene_to_pos = {
                            gene_id: i
                            for i, gene_id in enumerate(range(start, stop, step))
                        }
                    elif isinstance(self.gene_selector, list):
                        gene_to_pos = {
                            gene_id: i for i, gene_id in enumerate(self.gene_selector)
                        }
                else:
                    gene_to_pos = {
                        gene_id: gene_id for gene_id in range(self.slaf_array.shape[1])
                    }

                # Fill result array with computed values
                gene_ids = result_df["gene_integer_id"].to_numpy()
                sum_values = result_df["sum_value"].to_numpy()
                for i, gene_id in enumerate(gene_ids):
                    if gene_id in gene_to_pos:
                        pos = gene_to_pos[gene_id]
                        if 0 <= pos < result_size:
                            result_array[pos] = sum_values[i]

                return result_array.reshape(1, -1)  # Return (1, n_genes) for axis=0
            elif axis == 1:  # Cell-wise sum
                if "sum_value" in result_df.columns:
                    result_df = result_df.sort("cell_integer_id")
                else:
                    # Handle case where we might have partial sums
                    combined = result_df.group_by("cell_integer_id").agg(
                        pl.col("partial_sum").sum().alias("sum_value")
                    )
                    result_df = combined.sort("cell_integer_id")

                # Create array of exact slice size with zero-filling
                if self.cell_selector is not None:
                    if isinstance(self.cell_selector, slice):
                        start = self.cell_selector.start or 0
                        stop = self.cell_selector.stop or self.slaf_array.shape[0]
                        step = self.cell_selector.step or 1
                        # Handle negative indices
                        if start < 0:
                            start = self.slaf_array.shape[0] + start
                        if stop < 0:
                            stop = self.slaf_array.shape[0] + stop
                        # Clamp bounds
                        start = max(0, min(start, self.slaf_array.shape[0]))
                        stop = max(0, min(stop, self.slaf_array.shape[0]))
                        result_size = len(range(start, stop, step))
                    elif isinstance(self.cell_selector, list):
                        result_size = len(self.cell_selector)
                    else:
                        result_size = self.slaf_array.shape[0]
                else:
                    result_size = self.slaf_array.shape[0]

                # Create result array with exact size and zero-filling
                result_array = np.zeros(result_size)

                # Map cell IDs to positions in result array
                if self.cell_selector is not None:
                    if isinstance(self.cell_selector, slice):
                        start = self.cell_selector.start or 0
                        stop = self.cell_selector.stop or self.slaf_array.shape[0]
                        step = self.cell_selector.step or 1
                        # Handle negative indices
                        if start < 0:
                            start = self.slaf_array.shape[0] + start
                        if stop < 0:
                            stop = self.slaf_array.shape[0] + stop
                        # Clamp bounds
                        start = max(0, min(start, self.slaf_array.shape[0]))
                        stop = max(0, min(stop, self.slaf_array.shape[0]))
                        cell_to_pos = {
                            cell_id: i
                            for i, cell_id in enumerate(range(start, stop, step))
                        }
                    elif isinstance(self.cell_selector, list):
                        cell_to_pos = {
                            cell_id: i for i, cell_id in enumerate(self.cell_selector)
                        }
                else:
                    cell_to_pos = {
                        cell_id: cell_id for cell_id in range(self.slaf_array.shape[0])
                    }

                # Fill result array with computed values
                cell_ids = result_df["cell_integer_id"].to_numpy()
                sum_values = result_df["sum_value"].to_numpy()
                for i, cell_id in enumerate(cell_ids):
                    if cell_id in cell_to_pos:
                        pos = cell_to_pos[cell_id]
                        if 0 <= pos < result_size:
                            result_array[pos] = sum_values[i]

                return result_array.reshape(-1, 1)  # Return (n_cells, 1) for axis=1
            else:  # Overall sum
                if "total_sum" in result_df.columns:
                    return np.array([result_df["total_sum"].sum()])
                else:
                    return result_df["sum_value"].to_numpy()

        else:
            # For other operations, try to extract the main value column
            value_columns = [col for col in result_df.columns if "value" in col.lower()]
            if value_columns:
                return result_df[value_columns[0]].to_numpy()
            else:
                # Fallback: return first numeric column
                numeric_columns = [
                    col
                    for col in result_df.columns
                    if result_df[col].dtype
                    in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
                ]
                if numeric_columns:
                    return result_df[numeric_columns[0]].to_numpy()
                else:
                    raise ValueError(f"Cannot convert result for operation {operation}")
