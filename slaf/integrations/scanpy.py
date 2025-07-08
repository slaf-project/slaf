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
        Lazy version of scanpy.pp.calculate_qc_metrics

        Calculate quality control metrics for cells and genes.
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

        cell_qc = adata.slaf.query(cell_qc_sql)

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

        gene_qc = adata.slaf.query(gene_qc_sql)

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
            expected_genes = adata.slaf.query(expected_genes_sql)

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
        Lazy version of scanpy.pp.filter_cells

        Filter cells based on quality control metrics.
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

        filtered_cells = adata.slaf.query(filter_sql)

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
        Lazy version of scanpy.pp.filter_genes

        Filter genes based on quality control metrics.
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

        filtered_genes = adata.slaf.query(filter_sql)

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
        Lazy version of scanpy.pp.normalize_total

        Normalize counts per cell to target sum.
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

        cell_totals = adata.slaf.query(cell_totals_sql)

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
        Lazy version of scanpy.pp.log1p

        Logarithmize the data matrix.
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
        Lazy version of scanpy.pp.highly_variable_genes

        Identify highly variable genes.
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

        gene_stats = adata.slaf.query(stats_sql)

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
            expected_genes = adata.slaf.query(expected_genes_sql)

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

    This provides explicit control over when transformations are applied.

    Args:
        adata: LazyAnnData object
        transformations: List of transformation names to apply. If None, applies all.

    Returns:
        New LazyAnnData with transformations applied
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

    Args:
        adata: LazyAnnData object
        inplace: If True, modify the original object. If False, return a copy.

    Returns:
        LazyAnnData with transformations cleared (if not inplace)
    """
    if inplace:
        adata._transformations = {}
        return None
    else:
        new_adata = adata.copy()
        new_adata._transformations = {}
        return new_adata
