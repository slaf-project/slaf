"""
Fragment-based processing for SLAF datasets.

This module provides fragment-based processing capabilities for large SLAF datasets,
enabling memory-efficient operations on datasets with millions of cells.
"""

import numpy as np
import polars as pl
from loguru import logger


class FragmentProcessor:
    """
    Fragment-based processing for SLAF datasets.

    This class enables building lazy computation graphs using Polars LazyFrames
    for fragment-based processing of large datasets. It provides a unified
    interface for various operations like normalization, aggregation, and
    matrix computation.

    Key Features:
        - Lazy evaluation with Polars LazyFrames
        - Fragment-based processing for memory efficiency
        - Support for cell and gene selectors (slicing)
        - Automatic query fusion for optimal performance
        - Progress feedback during processing

    Examples:
        >>> # Basic usage
        >>> processor = FragmentProcessor(slaf_array)
        >>> lazy_pipeline = processor.build_lazy_pipeline("normalize_total", target_sum=10000)
        >>> result = processor.compute(lazy_pipeline)

        >>> # With selectors (slicing)
        >>> processor = FragmentProcessor(slaf_array, cell_selector=slice(0, 1000), gene_selector=slice(0, 5000))
        >>> lazy_pipeline = processor.build_lazy_pipeline("mean", axis=0)
        >>> result = processor.compute(lazy_pipeline)
    """

    def __init__(self, slaf_array, cell_selector=None, gene_selector=None):
        """
        Initialize FragmentProcessor.

        Args:
            slaf_array: SLAFArray instance
            cell_selector: Cell selector (slice, list, etc.) for subsetting
            gene_selector: Gene selector (slice, list, etc.) for subsetting
        """
        self.slaf_array = slaf_array
        self.fragments = slaf_array.expression.get_fragments()
        self.n_fragments = len(self.fragments)
        self.cell_selector = cell_selector
        self.gene_selector = gene_selector

        logger.debug(f"Initialized FragmentProcessor with {self.n_fragments} fragments")
        if cell_selector is not None or gene_selector is not None:
            logger.debug(f"Using selectors: cell={cell_selector}, gene={gene_selector}")

    def build_lazy_pipeline(self, operation: str, **kwargs) -> pl.LazyFrame:
        """
        Build lazy computation graph for the specified operation.

        Creates a Polars LazyFrame that represents the computation graph for the
        specified operation across all fragments. The actual computation is deferred
        until compute() is called.

        Polars handles query fusion automatically for optimal performance.

        Args:
            operation: Operation to apply ("normalize_total", "log1p", "mean", "sum")
            **kwargs: Operation-specific parameters

        Returns:
            Polars LazyFrame representing the computation graph

        Raises:
            ValueError: If operation is not supported

        Examples:
            >>> # Build pipeline for normalize_total
            >>> pipeline = processor.build_lazy_pipeline("normalize_total", target_sum=1e4)
            >>>
            >>> # Build pipeline for log1p
            >>> pipeline = processor.build_lazy_pipeline("log1p")
            >>>
            >>> # Build pipeline for mean aggregation
            >>> pipeline = processor.build_lazy_pipeline("mean", axis=0)
        """
        # Special handling for normalize_total - need global cell totals first
        if operation == "normalize_total":
            return self._build_normalize_total_pipeline(**kwargs)

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
            return lazy_fragments[0]
        else:
            return pl.concat(lazy_fragments, how="vertical")

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
            Processed Polars LazyFrame

        Raises:
            ValueError: If operation is not supported
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

    def _apply_compute_matrix(self, lazy_df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Apply compute_matrix operation to lazy dataframe.

        Prepares data for sparse matrix computation by ensuring proper column structure.

        Args:
            lazy_df: Polars LazyFrame to process

        Returns:
            LazyFrame ready for matrix computation
        """
        # Ensure we have the required columns for matrix computation
        return lazy_df.select(["cell_integer_id", "gene_integer_id", "value"])

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
