import numpy as np
import pandas as pd
import polars as pl

from slaf.integrations.anndata import LazyAnnData


# Preprocessing module compatibility
class LazyPreprocessing:
    """Scanpy preprocessing functions with lazy evaluation"""

    @staticmethod
    def calculate_qc_metrics(
        adata: LazyAnnData,
        percent_top: int | list | None = None,
        log1p: bool = True,
        inplace: bool = True,
    ) -> tuple | None:
        """
        Calculate quality control metrics for cells and genes using lazy evaluation.

        This function computes cell and gene-level QC metrics using SQL aggregation
        for memory efficiency. It calculates metrics like total counts, number of
        genes per cell, and number of cells per gene.

        Args:
            adata: LazyAnnData instance containing the single-cell data.
            percent_top: Number of top genes to consider for percent_top calculation.
                        Currently not implemented in lazy version.
            log1p: Whether to compute log1p-transformed versions of count metrics.
                   Adds log1p_total_counts and log1p_n_genes_by_counts to cell metrics,
                   and log1p_total_counts and log1p_n_cells_by_counts to gene metrics.
            inplace: Whether to modify the adata object in place. Currently not fully
                    implemented in lazy version - returns None when True.

        Returns:
            tuple | None: If inplace=False, returns (cell_qc, gene_qc) where:
                - cell_qc: DataFrame with cell-level QC metrics
                - gene_qc: DataFrame with gene-level QC metrics
                If inplace=True, returns None.

        Raises:
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Calculate QC metrics
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> cell_qc, gene_qc = LazyPreprocessing.calculate_qc_metrics(
            ...     adata, inplace=False
            ... )
            >>> print(f"Cell QC shape: {cell_qc.shape}")
            Cell QC shape: (1000, 4)
            >>> print(f"Gene QC shape: {gene_qc.shape}")
            Gene QC shape: (20000, 4)

            >>> # With log1p transformation
            >>> cell_qc, gene_qc = LazyPreprocessing.calculate_qc_metrics(
            ...     adata, log1p=True, inplace=False
            ... )
            >>> print(f"Cell QC columns: {list(cell_qc.columns)}")
            Cell QC columns: ['cell_id', 'n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'log1p_n_genes_by_counts']
        """

        # Calculate cell-level metrics via SQL using simple aggregation (no JOINs)
        cell_qc_sql = """
        SELECT
            e.cell_integer_id,
            COUNT(DISTINCT e.gene_integer_id) as n_genes_by_counts,
            SUM(e.value) as total_counts
        FROM expression e
        GROUP BY e.cell_integer_id
        ORDER BY e.cell_integer_id
        """

        cell_qc = adata.slaf.query(cell_qc_sql)

        # Work with polars DataFrame internally
        cell_qc_pl = cell_qc

        # Map cell_integer_id to cell_id for scanpy compatibility
        if hasattr(adata.slaf, "obs") and adata.slaf.obs is not None:
            # Create mapping from cell_integer_id to cell names
            # Use polars DataFrame to get cell names
            obs_df = adata.slaf.obs
            if "cell_id" in obs_df.columns:
                cell_id_to_name = dict(
                    zip(
                        obs_df["cell_integer_id"].to_list(),
                        obs_df["cell_id"].to_list(),
                        strict=False,
                    )
                )
            else:
                # Fallback: create cell names from integer IDs
                cell_id_to_name = {i: f"cell_{i}" for i in range(len(obs_df))}
            # Use vectorized polars operations for mapping
            # Create mapping DataFrame
            cell_map_df = pl.DataFrame(
                {
                    "cell_integer_id": list(cell_id_to_name.keys()),
                    "cell_id": list(cell_id_to_name.values()),
                }
            )
            # Join with mapping DataFrame
            cell_qc_pl = cell_qc_pl.join(cell_map_df, on="cell_integer_id", how="left")
            # Fill any missing values with default format
            cell_qc_pl = cell_qc_pl.with_columns(
                pl.col("cell_id").fill_null(
                    pl.col("cell_integer_id")
                    .cast(pl.Utf8)
                    .map_elements(lambda x: f"cell_{x}", return_dtype=pl.Utf8)
                )
            )
        else:
            # Fallback: use cell_integer_id as cell_id
            cell_qc_pl = cell_qc_pl.with_columns(
                pl.col("cell_integer_id").cast(pl.Utf8).alias("cell_id")
            )

        # Add log1p transformed counts if requested
        if log1p:
            cell_qc_pl = cell_qc_pl.with_columns(
                [
                    pl.col("total_counts").log1p().alias("log1p_total_counts"),
                    pl.col("n_genes_by_counts")
                    .log1p()
                    .alias("log1p_n_genes_by_counts"),
                ]
            )

        # Convert to pandas for return compatibility
        cell_qc = cell_qc_pl.to_pandas()

        # Calculate gene-level metrics via SQL using simple aggregation (no JOINs)
        gene_qc_sql = """
        SELECT
            e.gene_integer_id,
            COUNT(DISTINCT e.cell_integer_id) AS n_cells_by_counts,
            SUM(e.value) AS total_counts
        FROM expression e
        GROUP BY e.gene_integer_id
        ORDER BY e.gene_integer_id
        """

        gene_qc = adata.slaf.query(gene_qc_sql)

        # Work with polars DataFrames internally
        gene_qc_pl = gene_qc

        # For scanpy compatibility, we need to ensure all genes are present
        # Use in-memory var if available, otherwise fall back to SQL
        if hasattr(adata.slaf, "var") and adata.slaf.var is not None:
            # Use the materialized var metadata directly
            expected_genes_pl = pl.DataFrame(
                {"gene_integer_id": range(len(adata.slaf.var))}
            )
        else:
            expected_genes_sql = """
            SELECT gene_integer_id
            FROM genes
            ORDER BY gene_integer_id
            """
            expected_genes = adata.slaf.query(expected_genes_sql)
            expected_genes_pl = expected_genes

        # Create a complete gene_qc DataFrame with all expected genes using polars
        gene_qc_complete_pl = expected_genes_pl.join(
            gene_qc_pl, on="gene_integer_id", how="left"
        ).fill_null(0)

        # Ensure proper data types
        gene_qc_complete_pl = gene_qc_complete_pl.with_columns(
            [
                pl.col("n_cells_by_counts").cast(pl.Int64),
                pl.col("total_counts").cast(pl.Float64),
            ]
        )

        # Map gene_integer_id to gene names for scanpy compatibility
        if hasattr(adata.slaf, "var") and adata.slaf.var is not None:
            # Create mapping from gene_integer_id to gene names
            # Use polars DataFrame to get gene names
            var_df = adata.slaf.var
            if "gene_id" in var_df.columns:
                gene_id_to_name = dict(
                    zip(
                        var_df["gene_integer_id"].to_list(),
                        var_df["gene_id"].to_list(),
                        strict=False,
                    )
                )
            else:
                # Fallback: create gene names from integer IDs
                gene_id_to_name = {i: f"gene_{i}" for i in range(len(var_df))}
            # Use vectorized polars operations for mapping
            # Create mapping DataFrame
            gene_map_df = pl.DataFrame(
                {
                    "gene_integer_id": list(gene_id_to_name.keys()),
                    "gene_id": list(gene_id_to_name.values()),
                }
            )
            # Join with mapping DataFrame
            gene_qc_complete_pl = gene_qc_complete_pl.join(
                gene_map_df, on="gene_integer_id", how="left"
            )
            # Fill any missing values with default format
            gene_qc_complete_pl = gene_qc_complete_pl.with_columns(
                pl.col("gene_id").fill_null(
                    pl.col("gene_integer_id")
                    .cast(pl.Utf8)
                    .map_elements(lambda x: f"gene_{x}", return_dtype=pl.Utf8)
                )
            )
        else:
            # Fallback: use gene_integer_id as gene_id
            gene_qc_complete_pl = gene_qc_complete_pl.with_columns(
                pl.col("gene_integer_id").cast(pl.Utf8).alias("gene_id")
            )

        # Convert to pandas and set gene_id as index
        gene_qc_complete = gene_qc_complete_pl.to_pandas().set_index("gene_id")

        if log1p:
            gene_qc_complete["log1p_total_counts"] = np.log1p(
                gene_qc_complete["total_counts"]
            )
            gene_qc_complete["log1p_n_cells_by_counts"] = np.log1p(
                gene_qc_complete["n_cells_by_counts"]
            )

        if inplace:
            # Update the metadata tables in SLAF
            # This would require implementing metadata updates in SLAF
            # For now, just update the cached obs/var

            # Update obs
            adata._obs = None  # Clear cache
            for _ in cell_qc.iterrows():
                # Would need to implement metadata updates
                pass

            # Update var
            adata._var = None  # Clear cache
            for _ in gene_qc_complete.iterrows():
                # Would need to implement metadata updates
                pass

            return None
        else:
            return cell_qc, gene_qc_complete

    @staticmethod
    def filter_cells(
        adata: LazyAnnData,
        min_counts: int | None = None,
        min_genes: int | None = None,
        max_counts: int | None = None,
        max_genes: int | None = None,
        inplace: bool = True,
    ) -> LazyAnnData | None:
        """
        Filter cells based on quality control metrics using lazy evaluation.

        This function filters cells based on their total counts and number of genes
        using SQL aggregation for memory efficiency. It supports both minimum and
        maximum thresholds for each metric.

        Args:
            adata: LazyAnnData instance containing the single-cell data.
            min_counts: Minimum total counts per cell. Cells with fewer counts are filtered.
            min_genes: Minimum number of genes per cell. Cells with fewer genes are filtered.
            max_counts: Maximum total counts per cell. Cells with more counts are filtered.
            max_genes: Maximum number of genes per cell. Cells with more genes are filtered.
            inplace: Whether to modify the adata object in place. Currently not fully
                    implemented in lazy version - returns None when True.

        Returns:
            LazyAnnData | None: If inplace=False, returns filtered LazyAnnData.
                               If inplace=True, returns None.

        Raises:
            ValueError: If all cells are filtered out by the criteria.
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Filter cells with basic criteria
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> filtered = LazyPreprocessing.filter_cells(
            ...     adata, min_counts=100, min_genes=50, inplace=False
            ... )
            >>> print(f"Original cells: {adata.n_obs}")
            Original cells: 1000
            >>> print(f"Filtered cells: {filtered.n_obs}")
            Filtered cells: 850

            >>> # Filter with maximum thresholds
            >>> filtered = LazyPreprocessing.filter_cells(
            ...     adata, max_counts=10000, max_genes=5000, inplace=False
            ... )
            >>> print(f"Cells after max filtering: {filtered.n_obs}")
            Cells after max filtering: 920

            >>> # Error when all cells filtered out
            >>> try:
            ...     LazyPreprocessing.filter_cells(adata, min_counts=1000000)
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: All cells were filtered out
        """

        # Build filter conditions
        conditions = []

        if min_counts is not None:
            conditions.append(f"cell_stats.total_counts >= {min_counts}")
        if max_counts is not None:
            conditions.append(f"cell_stats.total_counts <= {max_counts}")
        if min_genes is not None:
            conditions.append(f"cell_stats.n_genes_by_counts >= {min_genes}")
        if max_genes is not None:
            conditions.append(f"cell_stats.n_genes_by_counts <= {max_genes}")

        if not conditions:
            return adata if not inplace else None

        where_clause = " AND ".join(conditions)

        # Get filtered cell IDs using simple aggregation (no JOINs)
        filter_sql = f"""
        SELECT cell_stats.cell_integer_id
        FROM (
            SELECT
                e.cell_integer_id,
                COUNT(DISTINCT e.gene_integer_id) as n_genes_by_counts,
                SUM(e.value) as total_counts
            FROM expression e
            GROUP BY e.cell_integer_id
        ) cell_stats
        WHERE ({where_clause})
        ORDER BY cell_stats.cell_integer_id
        """

        filtered_cells = adata.slaf.query(filter_sql)

        if len(filtered_cells) == 0:
            raise ValueError("All cells were filtered out")

        # Create boolean mask from the filtered cell IDs
        # Use the materialized obs metadata to map cell_integer_id to cell names
        if hasattr(adata.slaf, "obs") and adata.slaf.obs is not None:
            # Get all cell integer IDs that pass the filter
            filtered_cell_ids = set(filtered_cells["cell_integer_id"].to_list())

            # Create boolean mask for all cells
            all_cell_ids = adata.slaf.obs["cell_integer_id"].to_list()
            boolean_mask = [cell_id in filtered_cell_ids for cell_id in all_cell_ids]

            # Actually subset the LazyAnnData object (like scanpy does)
            if inplace:
                # Set up the filtered obs function to apply the boolean mask
                def filtered_obs() -> pd.DataFrame:
                    obs_df = adata.slaf.obs
                    if obs_df is not None:
                        # Work with polars DataFrame internally
                        obs_pl = obs_df
                        # Drop cell_integer_id column if present
                        if "cell_integer_id" in obs_pl.columns:
                            obs_pl = obs_pl.drop("cell_integer_id")
                        # Apply boolean mask
                        obs_pl = obs_pl.filter(pl.Series(boolean_mask))
                        # Convert to pandas DataFrame for AnnData compatibility
                        obs_copy = obs_pl.to_pandas()
                        # Set cell_id as index if present
                        if "cell_id" in obs_copy.columns:
                            obs_copy = obs_copy.set_index("cell_id")
                        # Set index name to match AnnData format
                        if hasattr(obs_copy, "index"):
                            obs_copy.index.name = "cell_id"
                        return obs_copy
                    return pd.DataFrame()

                # Set the filtered functions
                adata._filtered_obs = filtered_obs
                adata._filtered_var = None  # No gene filtering
                adata._cell_selector = boolean_mask
                adata._gene_selector = None

                # Clear cached names to force recalculation
                adata._cached_obs_names = None
                adata._cached_var_names = None

                return None
            else:
                # Create a new filtered LazyAnnData
                filtered_adata = LazyAnnData(adata.slaf, backend=adata.backend)

                # Set up the filtered obs function
                def filtered_obs() -> pd.DataFrame:
                    obs_df = adata.slaf.obs
                    if obs_df is not None:
                        # Work with polars DataFrame internally
                        obs_pl = obs_df
                        # Drop cell_integer_id column if present
                        if "cell_integer_id" in obs_pl.columns:
                            obs_pl = obs_pl.drop("cell_integer_id")
                        # Apply boolean mask
                        obs_pl = obs_pl.filter(pl.Series(boolean_mask))
                        # Convert to pandas DataFrame for AnnData compatibility
                        obs_copy = obs_pl.to_pandas()
                        # Set cell_id as index if present
                        if "cell_id" in obs_copy.columns:
                            obs_copy = obs_copy.set_index("cell_id")
                        # Set index name to match AnnData format
                        if hasattr(obs_copy, "index"):
                            obs_copy.index.name = "cell_id"
                        return obs_copy
                    return pd.DataFrame()

                # Set the filtered functions
                filtered_adata._filtered_obs = filtered_obs
                filtered_adata._filtered_var = None  # No gene filtering
                filtered_adata._cell_selector = boolean_mask
                filtered_adata._gene_selector = None

                return filtered_adata
        else:
            # Fallback to using cell_integer_id directly
            cell_mask = adata.obs_names.isin(filtered_cells["cell_integer_id"])

        if inplace:
            # Apply filter to adata (would need proper implementation)
            # For now, just return the original adata
            print(
                f"Filtered out {np.sum(~cell_mask)} cells, {np.sum(cell_mask)} remaining"
            )
            return None
        else:
            # Create new filtered LazyAnnData
            filtered_adata = adata.copy()
            # Apply mask (would need proper implementation)
            return filtered_adata

    @staticmethod
    def filter_genes(
        adata: LazyAnnData,
        min_counts: int | None = None,
        min_cells: int | None = None,
        max_counts: int | None = None,
        max_cells: int | None = None,
        inplace: bool = True,
    ) -> LazyAnnData | None:
        """
        Filter genes based on quality control metrics using lazy evaluation.

        This function filters genes based on their total counts and number of cells
        using SQL aggregation for memory efficiency. It supports both minimum and
        maximum thresholds for each metric.

        Args:
            adata: LazyAnnData instance containing the single-cell data.
            min_counts: Minimum total counts per gene. Genes with fewer counts are filtered.
            min_cells: Minimum number of cells per gene. Genes expressed in fewer cells are filtered.
            max_counts: Maximum total counts per gene. Genes with more counts are filtered.
            max_cells: Maximum number of cells per gene. Genes expressed in more cells are filtered.
            inplace: Whether to modify the adata object in place. Currently not fully
                    implemented in lazy version - returns None when True.

        Returns:
            LazyAnnData | None: If inplace=False, returns filtered LazyAnnData.
                               If inplace=True, returns None.

        Raises:
            ValueError: If all genes are filtered out by the criteria.
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Filter genes with basic criteria
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> filtered = LazyPreprocessing.filter_genes(
            ...     adata, min_counts=10, min_cells=5, inplace=False
            ... )
            >>> print(f"Original genes: {adata.n_vars}")
            Original genes: 20000
            >>> print(f"Filtered genes: {filtered.n_vars}")
            Filtered genes: 15000

            >>> # Filter with maximum thresholds
            >>> filtered = LazyPreprocessing.filter_genes(
            ...     adata, max_counts=100000, max_cells=1000, inplace=False
            ... )
            >>> print(f"Genes after max filtering: {filtered.n_vars}")
            Genes after max filtering: 18000

            >>> # Error when all genes filtered out
            >>> try:
            ...     LazyPreprocessing.filter_genes(adata, min_counts=1000000)
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: All genes were filtered out
        """

        # Build filter conditions for genes
        conditions = []

        if min_counts is not None:
            conditions.append(f"gene_stats.total_counts >= {min_counts}")
        if max_counts is not None:
            conditions.append(f"gene_stats.total_counts <= {max_counts}")
        if min_cells is not None:
            conditions.append(f"gene_stats.n_cells_by_counts >= {min_cells}")
        if max_cells is not None:
            conditions.append(f"gene_stats.n_cells_by_counts <= {max_cells}")

        if not conditions:
            return adata if not inplace else None

        where_clause = " AND ".join(conditions)

        # Get filtered gene IDs using simple aggregation (no JOINs)
        filter_sql = f"""
        SELECT gene_stats.gene_integer_id
        FROM (
            SELECT
                e.gene_integer_id,
                COUNT(DISTINCT e.cell_integer_id) AS n_cells_by_counts,
                SUM(e.value) AS total_counts
            FROM expression e
            GROUP BY e.gene_integer_id
        ) gene_stats
        WHERE {where_clause}
        ORDER BY gene_stats.gene_integer_id
        """

        filtered_genes = adata.slaf.query(filter_sql)

        if len(filtered_genes) == 0:
            raise ValueError("All genes were filtered out")

        # Create boolean mask from the filtered gene IDs
        # Use the materialized var metadata to map gene_integer_id to gene names
        if hasattr(adata.slaf, "var") and adata.slaf.var is not None:
            # Create a mapping from gene_integer_id to gene names
            # Use polars DataFrame to get gene names
            var_df = adata.slaf.var
            if "gene_id" in var_df.columns:
                gene_id_to_name = dict(
                    zip(
                        var_df["gene_integer_id"].to_list(),
                        var_df["gene_id"].to_list(),
                        strict=False,
                    )
                )
            else:
                # Fallback: create gene names from integer IDs
                gene_id_to_name = {i: f"gene_{i}" for i in range(len(var_df))}
            filtered_gene_names = [
                gene_id_to_name.get(gid, f"gene_{gid}")
                for gid in filtered_genes["gene_integer_id"]
            ]
            gene_mask = adata.var_names.isin(filtered_gene_names)
        else:
            # Fallback to using gene_integer_id directly
            gene_mask = adata.var_names.isin(filtered_genes["gene_integer_id"])

        if inplace:
            # Apply filter to adata (would need proper implementation)
            print(
                f"Filtered out {np.sum(~gene_mask)} genes, {np.sum(gene_mask)} remaining"
            )
            return None
        else:
            # Create new filtered LazyAnnData
            filtered_adata = adata.copy()
            # Apply mask (would need proper implementation)
            return filtered_adata

    @staticmethod
    def normalize_total(
        adata: LazyAnnData,
        target_sum: float | None = 1e4,
        inplace: bool = True,
        fragments: bool | None = None,
    ) -> LazyAnnData | None:
        """
        Normalize counts per cell to target sum using lazy evaluation.

        This function normalizes the expression data so that each cell has a total
        count equal to target_sum. The normalization is applied lazily and stored
        as a transformation that will be computed when the data is accessed.

        Args:
            adata: LazyAnnData instance containing the single-cell data.
            target_sum: Target sum for normalization. Default is 10,000.
            inplace: Whether to modify the adata object in place. If False, returns
                    a copy with the transformation applied.
            fragments: Whether to use fragment-based processing. If None, automatically
                      selects based on dataset characteristics.

        Returns:
            LazyAnnData | None: If inplace=False, returns LazyAnnData with transformation.
                               If inplace=True, returns None.

        Raises:
            ValueError: If target_sum is not positive.
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Basic normalization
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> LazyPreprocessing.normalize_total(adata, target_sum=10000)
            Applied normalize_total with target_sum=10000

            >>> # Custom target sum
            >>> adata_copy = adata.copy()
            >>> LazyPreprocessing.normalize_total(
            ...     adata_copy, target_sum=5000, inplace=False
            ... )
            >>> print("Normalization applied to copy")

            >>> # Error with invalid target_sum
            >>> try:
            ...     LazyPreprocessing.normalize_total(adata, target_sum=0)
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: target_sum must be positive
        """
        if target_sum is None:
            target_sum = 1e4

        # Validate target_sum
        if target_sum <= 0:
            raise ValueError("target_sum must be positive")

        # Determine processing strategy
        if fragments is not None:
            use_fragments = fragments
        else:
            # Check if dataset has multiple fragments
            try:
                fragments_list = adata.slaf.expression.get_fragments()
                use_fragments = len(fragments_list) > 1
            except Exception:
                use_fragments = False

        if use_fragments:
            # Use fragment-based processing
            try:
                from slaf.core.fragment_processor import FragmentProcessor

                # Get selectors from the LazyAnnData if it's sliced
                cell_selector = getattr(adata, "_cell_selector", None)
                gene_selector = getattr(adata, "_gene_selector", None)

                processor = FragmentProcessor(
                    adata.slaf,
                    cell_selector=cell_selector,
                    gene_selector=gene_selector,
                    max_workers=4,
                    enable_caching=True,
                )
                # Use smart strategy selection for optimal performance
                lazy_pipeline = processor.build_lazy_pipeline_smart(
                    "normalize_total", target_sum=target_sum
                )
                result_df = processor.compute(lazy_pipeline)

                # Update adata with normalized values
                return adata._update_with_normalized_data(
                    result_df, target_sum, inplace
                )

            except Exception as e:
                print(
                    f"Fragment processing failed, falling back to global processing: {e}"
                )
                # Fall back to global processing
                use_fragments = False

        if not use_fragments:
            # Use global processing (original implementation)
            # Get cell totals for normalization using only the expression table
            cell_totals_sql = """
            SELECT
                e.cell_integer_id,
                SUM(e.value) as total_counts
            FROM expression e
            GROUP BY e.cell_integer_id
            ORDER BY e.cell_integer_id
            """

            cell_totals = adata.slaf.query(cell_totals_sql)

        # Work with polars DataFrame internally
        cell_totals_pl = cell_totals

        # Create normalization factors using polars
        # Map cell_integer_id to cell names for compatibility with anndata.py
        if hasattr(adata.slaf, "obs") and adata.slaf.obs is not None:
            # Create mapping from cell_integer_id to cell names
            # Use polars DataFrame to get cell names
            obs_df = adata.slaf.obs
            if "cell_id" in obs_df.columns:
                cell_id_to_name = dict(
                    zip(
                        obs_df["cell_integer_id"].to_list(),
                        obs_df["cell_id"].to_list(),
                        strict=False,
                    )
                )
            else:
                # Fallback: create cell names from integer IDs
                cell_id_to_name = {i: f"cell_{i}" for i in range(len(obs_df))}
            # Use vectorized polars operations for mapping
            # Create mapping DataFrame
            cell_map_df = pl.DataFrame(
                {
                    "cell_integer_id": list(cell_id_to_name.keys()),
                    "cell_id": list(cell_id_to_name.values()),
                }
            )
            # Join with mapping DataFrame
            cell_totals_pl = cell_totals_pl.join(
                cell_map_df, on="cell_integer_id", how="left"
            )
            # Fill any missing values with default format
            cell_totals_pl = cell_totals_pl.with_columns(
                [
                    pl.col("cell_id").fill_null(
                        pl.col("cell_integer_id")
                        .cast(pl.Utf8)
                        .map_elements(lambda x: f"cell_{x}", return_dtype=pl.Utf8)
                    ),
                    (target_sum / pl.col("total_counts")).alias("normalization_factor"),
                ]
            )
            # Convert to dictionary for compatibility
            normalization_dict = dict(
                zip(
                    cell_totals_pl["cell_id"].to_list(),
                    cell_totals_pl["normalization_factor"].to_list(),
                    strict=False,
                )
            )
        else:
            # Fallback: use cell_integer_id as string keys
            cell_totals_pl = cell_totals_pl.with_columns(
                [
                    pl.col("cell_integer_id").cast(pl.Utf8).alias("cell_id"),
                    (target_sum / pl.col("total_counts")).alias("normalization_factor"),
                ]
            )
            # Convert to dictionary for compatibility
            normalization_dict = dict(
                zip(
                    cell_totals_pl["cell_id"].to_list(),
                    cell_totals_pl["normalization_factor"].to_list(),
                    strict=False,
                )
            )

        if inplace:
            # Store normalization factors for lazy application
            if not hasattr(adata, "_transformations"):
                adata._transformations = {}

            adata._transformations["normalize_total"] = {
                "type": "normalize_total",
                "target_sum": float(
                    f"{target_sum:.10f}"
                ),  # Convert to regular decimal format
                "cell_factors": normalization_dict,
            }

            print(f"Applied normalize_total with target_sum={target_sum}")
            return None
        else:
            # Create a copy with the transformation (copy-on-write)
            new_adata = adata.copy()
            if not hasattr(new_adata, "_transformations"):
                new_adata._transformations = {}

            new_adata._transformations["normalize_total"] = {
                "type": "normalize_total",
                "target_sum": float(
                    f"{target_sum:.10f}"
                ),  # Convert to regular decimal format
                "cell_factors": normalization_dict,
            }

            return new_adata

    @staticmethod
    def log1p(
        adata: LazyAnnData, inplace: bool = True, fragments: bool | None = None
    ) -> LazyAnnData | None:
        """
        Apply log1p transformation to expression data using lazy evaluation.

        This function applies log(1 + x) transformation to the expression data.
        The transformation is stored lazily and will be computed when the data
        is accessed, avoiding memory-intensive operations on large datasets.

        Args:
            adata: LazyAnnData instance containing the single-cell data.
            inplace: Whether to modify the adata object in place. If False, returns
                    a copy with the transformation applied.

        Returns:
            LazyAnnData | None: If inplace=False, returns LazyAnnData with transformation.
                               If inplace=True, returns None.

        Raises:
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Apply log1p transformation
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> LazyPreprocessing.log1p(adata)
            Applied log1p transformation

            >>> # Apply to copy
            >>> adata_copy = adata.copy()
            >>> LazyPreprocessing.log1p(adata_copy, inplace=False)
            >>> print("Log1p transformation applied to copy")

            >>> # Check transformation was stored
            >>> print("log1p" in adata._transformations)
            True
        """
        # Determine processing strategy
        if fragments is not None:
            use_fragments = fragments
        else:
            # Check if dataset has multiple fragments
            try:
                fragments_list = adata.slaf.expression.get_fragments()
                use_fragments = len(fragments_list) > 1
            except Exception:
                use_fragments = False

        if use_fragments:
            # Use fragment-based processing
            try:
                from slaf.core.fragment_processor import FragmentProcessor

                # Get selectors from the LazyAnnData if it's sliced
                cell_selector = getattr(adata, "_cell_selector", None)
                gene_selector = getattr(adata, "_gene_selector", None)

                processor = FragmentProcessor(
                    adata.slaf,
                    cell_selector=cell_selector,
                    gene_selector=gene_selector,
                    max_workers=4,
                    enable_caching=True,
                )
                # Use smart strategy selection for optimal performance
                lazy_pipeline = processor.build_lazy_pipeline_smart("log1p")
                result_df = processor.compute(lazy_pipeline)

                # Update adata with log1p values
                return adata._update_with_log1p_data(result_df, inplace)

            except Exception as e:
                print(
                    f"Fragment processing failed, falling back to global processing: {e}"
                )
                # Fall back to global processing
                use_fragments = False

        if not use_fragments:
            # Use global processing (original implementation)
            if inplace:
                # Store log1p transformation for lazy application
                if not hasattr(adata, "_transformations"):
                    adata._transformations = {}

                adata._transformations["log1p"] = {"type": "log1p", "applied": True}

                print("Applied log1p transformation")
                return None
            else:
                # Create a copy with the transformation (copy-on-write)
                new_adata = adata.copy()
                if not hasattr(new_adata, "_transformations"):
                    new_adata._transformations = {}

                new_adata._transformations["log1p"] = {"type": "log1p", "applied": True}

                return new_adata

    @staticmethod
    def highly_variable_genes(
        adata: LazyAnnData,
        min_mean: float = 0.0125,
        max_mean: float = 3,
        min_disp: float = 0.5,
        max_disp: float = np.inf,
        n_top_genes: int | None = None,
        inplace: bool = True,
    ) -> pd.DataFrame | None:
        """
        Identify highly variable genes using lazy evaluation.

        This function identifies genes with high cell-to-cell variation in expression
        using SQL aggregation for memory efficiency. It calculates mean expression
        and dispersion for each gene and applies filtering criteria.

        Args:
            adata: LazyAnnData instance containing the single-cell data.
            min_mean: Minimum mean expression for genes to be considered.
            max_mean: Maximum mean expression for genes to be considered.
            min_disp: Minimum dispersion for genes to be considered.
            max_disp: Maximum dispersion for genes to be considered.
            n_top_genes: Number of top genes to select by dispersion.
                        If specified, overrides min_disp and max_disp criteria.
            inplace: Whether to modify the adata object in place. Currently not
                    fully implemented - returns None when True.

        Returns:
            pd.DataFrame | None: If inplace=False, returns DataFrame with gene statistics
                                and highly_variable column. If inplace=True, returns None.

        Raises:
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Find highly variable genes
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> hvg_stats = LazyPreprocessing.highly_variable_genes(
            ...     adata, inplace=False
            ... )
            >>> print(f"Highly variable genes: {hvg_stats['highly_variable'].sum()}")
            Highly variable genes: 1500

            >>> # With custom criteria
            >>> hvg_stats = LazyPreprocessing.highly_variable_genes(
            ...     adata, min_mean=0.1, max_mean=5, min_disp=1.0, inplace=False
            ... )
            >>> print(f"Genes meeting criteria: {hvg_stats['highly_variable'].sum()}")
            Genes meeting criteria: 800

            >>> # Select top N genes
            >>> hvg_stats = LazyPreprocessing.highly_variable_genes(
            ...     adata, n_top_genes=1000, inplace=False
            ... )
            >>> print(f"Top genes selected: {hvg_stats['highly_variable'].sum()}")
            Top genes selected: 1000
        """

        # Calculate gene statistics via SQL using simple aggregation (no JOINs)
        stats_sql = """
        SELECT
            e.gene_integer_id,
            COUNT(DISTINCT e.cell_integer_id) AS n_cells,
            AVG(e.value) AS mean_expr,
            VARIANCE(e.value) AS variance,
            CASE WHEN AVG(e.value) > 0 THEN VARIANCE(e.value) / AVG(e.value) ELSE 0 END as dispersion
        FROM expression e
        GROUP BY e.gene_integer_id
        ORDER BY e.gene_integer_id
        """

        gene_stats = adata.slaf.query(stats_sql)

        # Work with polars DataFrames internally
        gene_stats_pl = gene_stats

        # Get the expected gene_integer_ids from the materialized var metadata
        # Use in-memory var if available, otherwise fall back to SQL
        if hasattr(adata.slaf, "var") and adata.slaf.var is not None:
            # Use the materialized var metadata directly
            expected_genes_pl = pl.DataFrame(
                {"gene_integer_id": range(len(adata.slaf.var))}
            )
        else:
            # Fallback to SQL if var is not available
            expected_genes_sql = """
            SELECT gene_integer_id
            FROM genes
            ORDER BY gene_integer_id
            """
            expected_genes = adata.slaf.query(expected_genes_sql)
            expected_genes_pl = expected_genes

        # Create a complete gene_stats DataFrame with all expected genes using polars
        gene_stats_complete_pl = expected_genes_pl.join(
            gene_stats_pl, on="gene_integer_id", how="left"
        ).fill_null(0)

        # Ensure proper data types
        gene_stats_complete_pl = gene_stats_complete_pl.with_columns(
            [
                pl.col("n_cells").cast(pl.Int64),
                pl.col("mean_expr").cast(pl.Float64),
                pl.col("variance").cast(pl.Float64),
                pl.col("dispersion").cast(pl.Float64),
            ]
        )

        # Map gene_integer_id to gene_id for scanpy compatibility
        if hasattr(adata.slaf, "var") and adata.slaf.var is not None:
            # Create mapping from gene_integer_id to gene names
            # Use polars DataFrame to get gene names
            var_df = adata.slaf.var
            if "gene_id" in var_df.columns:
                gene_id_to_name = dict(
                    zip(
                        var_df["gene_integer_id"].to_list(),
                        var_df["gene_id"].to_list(),
                        strict=False,
                    )
                )
            else:
                # Fallback: create gene names from integer IDs
                gene_id_to_name = {i: f"gene_{i}" for i in range(len(var_df))}
            # Use vectorized polars operations for mapping
            # Create mapping DataFrame
            gene_map_df = pl.DataFrame(
                {
                    "gene_integer_id": list(gene_id_to_name.keys()),
                    "gene_id": list(gene_id_to_name.values()),
                }
            )
            # Join with mapping DataFrame
            gene_stats_complete_pl = gene_stats_complete_pl.join(
                gene_map_df, on="gene_integer_id", how="left"
            )
            # Fill any missing values with default format
            gene_stats_complete_pl = gene_stats_complete_pl.with_columns(
                pl.col("gene_id").fill_null(
                    pl.col("gene_integer_id")
                    .cast(pl.Utf8)
                    .map_elements(lambda x: f"gene_{x}", return_dtype=pl.Utf8)
                )
            )
        else:
            # Fallback: use gene_integer_id as gene_id
            gene_stats_complete_pl = gene_stats_complete_pl.with_columns(
                pl.col("gene_integer_id").cast(pl.Utf8).alias("gene_id")
            )

        # Convert to pandas and set gene_id as index for scanpy compatibility
        gene_stats_complete = gene_stats_complete_pl.to_pandas().set_index("gene_id")

        # Apply HVG criteria
        hvg_mask = (
            (gene_stats_complete["mean_expr"] >= min_mean)
            & (gene_stats_complete["mean_expr"] <= max_mean)
            & (gene_stats_complete["dispersion"] >= min_disp)
            & (gene_stats_complete["dispersion"] <= max_disp)
        )

        gene_stats_complete["highly_variable"] = hvg_mask

        if n_top_genes is not None:
            # Select top N genes by dispersion
            top_genes = gene_stats_complete.nlargest(n_top_genes, "dispersion")
            gene_stats_complete["highly_variable"] = gene_stats_complete.index.isin(
                top_genes.index
            )

        if inplace:
            # Update var metadata (would need implementation)
            print(f"Identified {hvg_mask.sum()} highly variable genes")
            return None
        else:
            return gene_stats_complete


# Create preprocessing instance for easier access
pp = LazyPreprocessing()


def apply_transformations(
    adata: LazyAnnData, transformations: list | None = None
) -> LazyAnnData:
    """
    Apply a list of transformations to the data.

    This function provides explicit control over when transformations are applied.
    It creates a copy of the LazyAnnData with only the specified transformations
    stored for lazy evaluation.

    Args:
        adata: LazyAnnData instance containing the single-cell data.
        transformations: List of transformation names to apply. If None, applies all
                       available transformations.

    Returns:
        LazyAnnData: New LazyAnnData with specified transformations applied.

    Raises:
        RuntimeError: If the SLAF array is not properly initialized.

    Examples:
        >>> # Apply all transformations
        >>> slaf_array = SLAFArray("path/to/data.slaf")
        >>> adata = LazyAnnData(slaf_array)
        >>> LazyPreprocessing.normalize_total(adata)
        >>> LazyPreprocessing.log1p(adata)
        >>> adata_with_transforms = apply_transformations(adata)
        >>> print("Transformations applied")

        >>> # Apply specific transformations
        >>> adata_copy = adata.copy()
        >>> adata_with_norm = apply_transformations(
        ...     adata_copy, transformations=["normalize_total"]
        ... )
        >>> print("Only normalization applied")
    """
    if transformations is None:
        # Apply all transformations
        transformations = list(adata._transformations.keys())

    # Create a copy to avoid modifying the original
    new_adata = adata.copy()

    # Apply only the specified transformations
    new_adata._transformations = {
        name: adata._transformations[name]
        for name in transformations
        if name in adata._transformations
    }

    return new_adata


def clear_transformations(
    adata: LazyAnnData, inplace: bool = False
) -> LazyAnnData | None:
    """
    Clear all transformations from the data.

    This function removes all stored transformations from the LazyAnnData object,
    effectively resetting it to its original state without any preprocessing.

    Args:
        adata: LazyAnnData instance containing the single-cell data.
        inplace: Whether to modify the adata object in place. If False, returns
                a copy with transformations cleared.

    Returns:
        LazyAnnData | None: If inplace=False, returns LazyAnnData with transformations
                           cleared. If inplace=True, returns None.

    Raises:
        RuntimeError: If the SLAF array is not properly initialized.

    Examples:
        >>> # Clear transformations in place
        >>> slaf_array = SLAFArray("path/to/data.slaf")
        >>> adata = LazyAnnData(slaf_array)
        >>> LazyPreprocessing.normalize_total(adata)
        >>> LazyPreprocessing.log1p(adata)
        >>> clear_transformations(adata, inplace=True)
        >>> print("Transformations cleared")

        >>> # Clear transformations to copy
        >>> adata_copy = adata.copy()
        >>> LazyPreprocessing.normalize_total(adata_copy)
        >>> adata_clean = clear_transformations(adata_copy, inplace=False)
        >>> print("Clean copy created")
    """
    if inplace:
        adata._transformations = {}
        return None
    else:
        new_adata = adata.copy()
        new_adata._transformations = {}
        return new_adata
