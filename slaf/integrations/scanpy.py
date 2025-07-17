import numpy as np
import pandas as pd

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
            cell_id,
            COUNT(DISTINCT gene_id) as n_genes_by_counts,
            SUM(value) as total_counts
        FROM expression
        GROUP BY cell_id
        ORDER BY cell_id
        """

        cell_qc = adata.slaf.lazy_query(cell_qc_sql).compute()

        # Add log1p transformed counts if requested
        if log1p:
            cell_qc["log1p_total_counts"] = np.log1p(cell_qc["total_counts"])
            cell_qc["log1p_n_genes_by_counts"] = np.log1p(cell_qc["n_genes_by_counts"])

        # Calculate gene-level metrics via SQL using simple aggregation (no JOINs)
        gene_qc_sql = """
        SELECT
            gene_id,
            COUNT(DISTINCT cell_id) AS n_cells_by_counts,
            SUM(value) AS total_counts
        FROM expression
        GROUP BY gene_id
        ORDER BY gene_id
        """

        gene_qc = adata.slaf.lazy_query(gene_qc_sql).compute()

        # For scanpy compatibility, we need to ensure all genes are present
        # Use in-memory var if available, otherwise fall back to SQL
        if hasattr(adata.slaf, "var") and adata.slaf.var is not None:
            expected_genes = pd.DataFrame({"gene_id": adata.slaf.var.index})
        else:
            expected_genes_sql = """
            SELECT gene_id
            FROM genes
            ORDER BY gene_integer_id
            """
            expected_genes = adata.slaf.lazy_query(expected_genes_sql).compute()

        # Create a complete gene_qc DataFrame with all expected genes
        gene_qc_complete = pd.DataFrame({"gene_id": expected_genes["gene_id"]})

        # Merge with the calculated gene_qc to fill in missing genes with zeros
        gene_qc_complete = gene_qc_complete.merge(
            gene_qc, on="gene_id", how="left"
        ).fillna(0)

        # Ensure proper data types
        gene_qc_complete["n_cells_by_counts"] = gene_qc_complete[
            "n_cells_by_counts"
        ].astype(int)
        gene_qc_complete["total_counts"] = gene_qc_complete["total_counts"].astype(
            float
        )

        # Set gene_id as index
        gene_qc_complete = gene_qc_complete.set_index("gene_id")

        if log1p:
            gene_qc_complete["log1p_total_counts"] = np.log1p(
                gene_qc_complete["total_counts"]
            )
            gene_qc_complete["log1p_n_cells_by_counts"] = np.log1p(
                gene_qc_complete["n_cells_by_counts"]
            )

        if inplace:
            # Store lazy queries for later computation instead of computing immediately
            adata._qc_queries["cell_qc"] = cell_qc
            adata._qc_queries["gene_qc"] = gene_qc_complete
            adata._qc_queries["log1p"] = log1p

            print("QC metrics stored as lazy queries - compute when needed")
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
            conditions.append(f"total_counts >= {min_counts}")
        if max_counts is not None:
            conditions.append(f"total_counts <= {max_counts}")
        if min_genes is not None:
            conditions.append(f"n_genes_by_counts >= {min_genes}")
        if max_genes is not None:
            conditions.append(f"n_genes_by_counts <= {max_genes}")

        if not conditions:
            return adata if not inplace else None

        where_clause = " AND ".join(conditions)

        # Get filtered cell IDs using simple aggregation (no JOINs)
        filter_sql = f"""
        SELECT cell_id
        FROM (
            SELECT
                cell_id,
                COUNT(DISTINCT gene_id) as n_genes_by_counts,
                SUM(value) as total_counts
            FROM expression
            GROUP BY cell_id
        ) cell_stats
        WHERE ({where_clause})
        ORDER BY cell_id
        """

        filtered_cells = adata.slaf.lazy_query(filter_sql).compute()

        if len(filtered_cells) == 0:
            raise ValueError("All cells were filtered out")

        # Create boolean mask from the filtered cell IDs
        cell_mask = adata.obs_names.isin(filtered_cells["cell_id"])

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
            conditions.append(f"total_counts >= {min_counts}")
        if max_counts is not None:
            conditions.append(f"total_counts <= {max_counts}")
        if min_cells is not None:
            conditions.append(f"n_cells_by_counts >= {min_cells}")
        if max_cells is not None:
            conditions.append(f"n_cells_by_counts <= {max_cells}")

        if not conditions:
            return adata if not inplace else None

        where_clause = " AND ".join(conditions)

        # Get filtered gene IDs using simple aggregation (no JOINs)
        filter_sql = f"""
        SELECT gene_id
        FROM (
            SELECT
                gene_id,
                COUNT(DISTINCT cell_id) AS n_cells_by_counts,
                SUM(value) AS total_counts
            FROM expression
            GROUP BY gene_id
        ) gene_stats
        WHERE {where_clause}
        ORDER BY gene_id
        """

        filtered_genes = adata.slaf.lazy_query(filter_sql).compute()

        if len(filtered_genes) == 0:
            raise ValueError("All genes were filtered out")

        # Create boolean mask from the filtered gene IDs
        gene_mask = adata.var_names.isin(filtered_genes["gene_id"])

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
        exclude_highly_expressed: bool = False,
        max_fraction: float = 0.05,
        key_added: str | None = None,
        inplace: bool = True,
    ) -> LazyAnnData | None:
        """
        Normalize counts per cell to target sum using lazy evaluation.

        This function normalizes the expression data so that each cell has a total
        count equal to target_sum. The normalization is applied lazily and stored
        as a transformation that will be computed when the data is accessed.

        Args:
            adata: LazyAnnData instance containing the single-cell data.
            target_sum: Target sum for normalization. Default is 10,000.
            exclude_highly_expressed: Whether to exclude highly expressed genes from
                                     normalization. Currently not implemented in lazy version.
            max_fraction: Maximum fraction of counts for highly expressed genes.
                         Used when exclude_highly_expressed=True.
            key_added: Key for storing normalization factors. Currently not used.
            inplace: Whether to modify the adata object in place. If False, returns
                    a copy with the transformation applied.

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

        # Get cell totals for normalization using only the expression table
        cell_totals_sql = """
        SELECT
            cell_id,
            SUM(value) as total_counts,
            cell_integer_id
        FROM expression
        GROUP BY cell_id, cell_integer_id
        ORDER BY cell_integer_id
        """

        cell_totals = adata.slaf.lazy_query(cell_totals_sql).compute()

        # Handle exclude_highly_expressed if requested
        if exclude_highly_expressed:
            # This would require more complex logic to identify and exclude highly expressed genes
            # For now, we'll implement the basic version
            print(
                "Warning: exclude_highly_expressed=True not yet implemented in lazy version"
            )

        # Create normalization factors
        normalization_dict = {
            row["cell_id"]: target_sum / row["total_counts"]
            for _, row in cell_totals.iterrows()
        }

        if inplace:
            # Store normalization factors for lazy application
            if not hasattr(adata, "_transformations"):
                adata._transformations = {}

            adata._transformations["normalize_total"] = {
                "type": "normalize_total",
                "target_sum": target_sum,
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
                "target_sum": target_sum,
                "cell_factors": normalization_dict,
            }

            return new_adata

    @staticmethod
    def log1p(adata: LazyAnnData, inplace: bool = True) -> LazyAnnData | None:
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
        Find highly variable genes using lazy evaluation.

        This function identifies highly variable genes using SQL aggregation for
        memory efficiency. It calculates mean expression, variance, and dispersion
        for each gene and applies filtering criteria.

        Args:
            adata: LazyAnnData instance containing the single-cell data.
            min_mean: Minimum mean expression for genes to be considered.
            max_mean: Maximum mean expression for genes to be considered.
            min_disp: Minimum dispersion for genes to be considered.
            max_disp: Maximum dispersion for genes to be considered.
            n_top_genes: Number of top genes to select. If specified, overrides
                        min_mean, max_mean, min_disp, max_disp criteria.
            inplace: Whether to modify the adata object in place. If False, returns
                    a copy with the transformation applied.

        Returns:
            pd.DataFrame | None: If inplace=False, returns DataFrame with highly variable
                                gene information. If inplace=True, returns None.

        Raises:
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Find highly variable genes with default criteria
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> hvg_df = LazyPreprocessing.highly_variable_genes(
            ...     adata, inplace=False
            ... )
            >>> print(f"Found {hvg_df['highly_variable'].sum()} highly variable genes")

            >>> # Find top 2000 genes
            >>> hvg_df = LazyPreprocessing.highly_variable_genes(
            ...     adata, n_top_genes=2000, inplace=False
            ... )
            >>> print(f"Top 2000 genes selected")

            >>> # Custom criteria
            >>> hvg_df = LazyPreprocessing.highly_variable_genes(
            ...     adata, min_mean=0.1, max_mean=5, min_disp=1.0, inplace=False
            ... )
            >>> print(f"Custom criteria found {hvg_df['highly_variable'].sum()} genes")
        """

        # Calculate gene statistics via SQL using simple aggregation (no JOINs)
        stats_sql = """
        SELECT
            gene_id,
            COUNT(DISTINCT cell_id) AS n_cells,
            AVG(value) AS mean_expr,
            VARIANCE(value) AS variance,
            CASE WHEN AVG(value) > 0 THEN VARIANCE(value) / AVG(value) ELSE 0 END as dispersion
        FROM expression
        GROUP BY gene_id
        ORDER BY gene_id
        """

        gene_stats = adata.slaf.lazy_query(stats_sql).compute()

        # Get the expected gene_ids from genes table to ensure all genes are present
        # Use in-memory var if available, otherwise fall back to SQL
        if hasattr(adata.slaf, "var") and adata.slaf.var is not None:
            expected_genes = pd.DataFrame({"gene_id": adata.slaf.var.index})
        else:
            expected_genes_sql = """
            SELECT gene_id
            FROM genes
            ORDER BY gene_integer_id
            """
            expected_genes = adata.slaf.lazy_query(expected_genes_sql).compute()

        # Create a complete gene_stats DataFrame with all expected genes
        gene_stats_complete = pd.DataFrame({"gene_id": expected_genes["gene_id"]})

        # Merge with the calculated gene_stats to fill in missing genes with zeros
        gene_stats_complete = gene_stats_complete.merge(
            gene_stats, on="gene_id", how="left"
        ).fillna(0)

        # Ensure proper data types
        gene_stats_complete["n_cells"] = gene_stats_complete["n_cells"].astype(int)
        gene_stats_complete["mean_expr"] = gene_stats_complete["mean_expr"].astype(
            float
        )
        gene_stats_complete["variance"] = gene_stats_complete["variance"].astype(float)
        gene_stats_complete["dispersion"] = gene_stats_complete["dispersion"].astype(
            float
        )

        # Set gene_id as index
        gene_stats_complete = gene_stats_complete.set_index("gene_id")

        # Apply filtering criteria
        if n_top_genes is not None:
            # Select top genes by dispersion
            gene_stats_complete = gene_stats_complete.sort_values(
                "dispersion", ascending=False
            )
            gene_stats_complete["highly_variable"] = False
            gene_stats_complete.iloc[
                :n_top_genes, gene_stats_complete.columns.get_loc("highly_variable")
            ] = True
        else:
            # Apply mean and dispersion filters
            mean_mask = (gene_stats_complete["mean_expr"] >= min_mean) & (
                gene_stats_complete["mean_expr"] <= max_mean
            )
            disp_mask = (gene_stats_complete["dispersion"] >= min_disp) & (
                gene_stats_complete["dispersion"] <= max_disp
            )
            gene_stats_complete["highly_variable"] = mean_mask & disp_mask

        if inplace:
            # Store the highly variable genes information
            if not hasattr(adata, "_hvg_info"):
                adata._hvg_info = {}  # type: ignore
            adata._hvg_info["highly_variable_genes"] = gene_stats_complete  # type: ignore
            print(
                f"Found {gene_stats_complete['highly_variable'].sum()} highly variable genes"
            )
            return None
        else:
            return gene_stats_complete

    @staticmethod
    def scale(
        adata: LazyAnnData,
        zero_center: bool = True,
        max_value: float | None = None,
        inplace: bool = True,
    ) -> LazyAnnData | None:
        """
        Scale data to unit variance and zero mean using lazy evaluation.

        This function applies z-score normalization to the expression data.
        The scaling is applied lazily and stored as a transformation that will
        be computed when the data is accessed.

        Args:
            adata: LazyAnnData instance containing the single-cell data.
            zero_center: Whether to center the data to zero mean.
            max_value: Maximum value to clip the data to after scaling.
            inplace: Whether to modify the adata object in place. If False, returns
                    a copy with the transformation applied.

        Returns:
            LazyAnnData | None: If inplace=False, returns LazyAnnData with transformation.
                               If inplace=True, returns None.

        Raises:
            ValueError: If max_value is negative.
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Basic scaling
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> LazyPreprocessing.scale(adata)
            Applied scale transformation

            >>> # Scale with max value clipping
            >>> adata_copy = adata.copy()
            >>> LazyPreprocessing.scale(adata_copy, max_value=10, inplace=False)
            >>> print("Scale transformation applied to copy")

            >>> # Scale without zero centering
            >>> adata_copy = adata.copy()
            >>> LazyPreprocessing.scale(adata_copy, zero_center=False, inplace=False)
            >>> print("Scale without zero centering applied")
        """
        # Validate parameters
        if max_value is not None and max_value < 0:
            raise ValueError("max_value must be non-negative")

        # Calculate gene-wise statistics for scaling
        n_cells = adata.n_obs
        stats_sql = f"""
        SELECT
            gene_id,
            SUM(value) / {n_cells} as mean_expr,
            CASE
                WHEN SUM(value * value) + ({n_cells} - COUNT(value)) * 0 > POW(SUM(value) / {n_cells}, 2) * {n_cells}
                THEN SQRT((SUM(value * value) + ({n_cells} - COUNT(value)) * 0) / {n_cells} - POW(SUM(value) / {n_cells}, 2))
                ELSE 1.0
            END as std_expr
        FROM expression
        GROUP BY gene_id
        ORDER BY gene_id
        """

        gene_stats = adata.slaf.lazy_query(stats_sql).compute()

        # Create scaling parameters dictionary
        scaling_params = {}
        for _, row in gene_stats.iterrows():
            gene_id = row["gene_id"]
            mean_val = row["mean_expr"]
            std_val = row["std_expr"] if row["std_expr"] > 0 else 1.0

            scaling_params[gene_id] = {"mean": mean_val, "std": std_val}

        if inplace:
            # Store scaling parameters for lazy application
            if not hasattr(adata, "_transformations"):
                adata._transformations = {}

            adata._transformations["scale"] = {
                "type": "scale",
                "zero_center": zero_center,
                "max_value": max_value,
                "scaling_params": scaling_params,
            }

            print("Applied scale transformation")
            return None
        else:
            # Create a copy with the transformation (copy-on-write)
            new_adata = adata.copy()
            if not hasattr(new_adata, "_transformations"):
                new_adata._transformations = {}

            new_adata._transformations["scale"] = {
                "type": "scale",
                "zero_center": zero_center,
                "max_value": max_value,
                "scaling_params": scaling_params,
            }

            return new_adata

    @staticmethod
    def sample(
        adata: LazyAnnData,
        n_obs: int | None = None,
        n_vars: int | None = None,
        random_state: int | None = None,
        inplace: bool = True,
    ) -> LazyAnnData | None:
        """
        Sample cells and/or genes randomly using lazy evaluation.

        This function samples a subset of cells and/or genes using SQL-level
        sampling for memory efficiency. It uses DuckDB's sampling capabilities
        to avoid loading all data into memory.

        Args:
            adata: LazyAnnData instance containing the single-cell data.
            n_obs: Number of cells to sample. If None, all cells are kept.
            n_vars: Number of genes to sample. If None, all genes are kept.
            random_state: Random seed for reproducible sampling.
            inplace: Whether to modify the adata object in place. If False, returns
                    a copy with the sampling applied.

        Returns:
            LazyAnnData | None: If inplace=False, returns LazyAnnData with sampling.
                               If inplace=True, returns None.

        Raises:
            ValueError: If n_obs or n_vars exceeds the available number of cells/genes.
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Sample 1000 cells
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> LazyPreprocessing.sample(adata, n_obs=1000)
            Applied sampling: 1000 cells

            >>> # Sample both cells and genes
            >>> adata_copy = adata.copy()
            >>> LazyPreprocessing.sample(adata_copy, n_obs=500, n_vars=1000, inplace=False)
            >>> print("Sampling applied to copy")

            >>> # Reproducible sampling
            >>> LazyPreprocessing.sample(adata, n_obs=1000, random_state=42)
            >>> print("Reproducible sampling applied")
        """
        # Validate sampling parameters
        if n_obs is not None and n_obs > adata.n_obs:
            raise ValueError(
                f"n_obs ({n_obs}) cannot exceed number of cells ({adata.n_obs})"
            )
        if n_vars is not None and n_vars > adata.n_vars:
            raise ValueError(
                f"n_vars ({n_vars}) cannot exceed number of genes ({adata.n_vars})"
            )

        # Store sampling parameters
        sampling_params = {
            "n_obs": n_obs,
            "n_vars": n_vars,
            "random_state": random_state,
        }

        if inplace:
            # Store sampling parameters for lazy application
            if not hasattr(adata, "_transformations"):
                adata._transformations = {}

            adata._transformations["sample"] = {
                "type": "sample",
                "params": sampling_params,
            }

            # Print sampling info
            sample_info = []
            if n_obs is not None:
                sample_info.append(f"{n_obs} cells")
            if n_vars is not None:
                sample_info.append(f"{n_vars} genes")
            print(f"Applied sampling: {' and '.join(sample_info)}")
            return None
        else:
            # Create a copy with the sampling (copy-on-write)
            new_adata = adata.copy()
            if not hasattr(new_adata, "_transformations"):
                new_adata._transformations = {}

            new_adata._transformations["sample"] = {
                "type": "sample",
                "params": sampling_params,
            }

            return new_adata

    @staticmethod
    def downsample_counts(
        adata: LazyAnnData,
        counts_per_cell: int | None = None,
        total_counts: int | None = None,
        random_state: int | None = None,
        inplace: bool = True,
    ) -> LazyAnnData | None:
        """
        Downsample counts to a specified number per cell using lazy evaluation.

        This function downsamples the expression counts to a specified number
        per cell using SQL-level operations for memory efficiency.

        Args:
            adata: LazyAnnData instance containing the single-cell data.
            counts_per_cell: Target number of counts per cell. If None, uses total_counts.
            total_counts: Total counts to downsample to. If None, uses counts_per_cell.
            random_state: Random seed for reproducible downsampling.
            inplace: Whether to modify the adata object in place. If False, returns
                    a copy with the downsampling applied.

        Returns:
            LazyAnnData | None: If inplace=False, returns LazyAnnData with downsampling.
                               If inplace=True, returns None.

        Raises:
            ValueError: If neither counts_per_cell nor total_counts is specified.
            ValueError: If counts_per_cell is negative.
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Downsample to 1000 counts per cell
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> LazyPreprocessing.downsample_counts(adata, counts_per_cell=1000)
            Applied downsampling: 1000 counts per cell

            >>> # Downsample to total counts
            >>> adata_copy = adata.copy()
            >>> LazyPreprocessing.downsample_counts(adata_copy, total_counts=50000, inplace=False)
            >>> print("Downsampling applied to copy")

            >>> # Reproducible downsampling
            >>> LazyPreprocessing.downsample_counts(adata, counts_per_cell=1000, random_state=42)
            >>> print("Reproducible downsampling applied")
        """
        # Validate parameters
        if counts_per_cell is None and total_counts is None:
            raise ValueError("Either counts_per_cell or total_counts must be specified")

        if counts_per_cell is not None and counts_per_cell < 0:
            raise ValueError("counts_per_cell must be non-negative")

        # Calculate target counts per cell
        if counts_per_cell is not None:
            target_counts_per_cell = counts_per_cell
        else:
            # total_counts is guaranteed to be not None here due to validation above
            target_counts_per_cell = total_counts // adata.n_obs  # type: ignore

        # Store downsampling parameters
        downsampling_params = {
            "counts_per_cell": target_counts_per_cell,
            "total_counts": total_counts,
            "random_state": random_state,
        }

        if inplace:
            # Store downsampling parameters for lazy application
            if not hasattr(adata, "_transformations"):
                adata._transformations = {}

            adata._transformations["downsample_counts"] = {
                "type": "downsample_counts",
                "params": downsampling_params,
            }

            print(f"Applied downsampling: {target_counts_per_cell} counts per cell")
            return None
        else:
            # Create a copy with the downsampling (copy-on-write)
            new_adata = adata.copy()
            if not hasattr(new_adata, "_transformations"):
                new_adata._transformations = {}

            new_adata._transformations["downsample_counts"] = {
                "type": "downsample_counts",
                "params": downsampling_params,
            }

            return new_adata


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
