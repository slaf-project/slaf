import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests  # type: ignore
import scanpy as sc

DatasetType = Literal[
    "pbmc3k",
    "pbmc3k_10x_mtx",
    "pbmc3k_10x_h5",
    "pbmc_68k",
    "heart_10k",
    "synthetic",
    "tiny_sample",
]

DEFAULT_DATASET_DIR = str(
    (Path(__file__).parent.parent.parent.parent / "slaf-datasets").resolve()
)


def download_dataset(
    dataset_type: DatasetType, output_dir: str = DEFAULT_DATASET_DIR
) -> str | None:
    """Download dataset and return path to h5ad file"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if dataset_type == "pbmc3k":
        # Use scanpy's built-in dataset
        print("Downloading PBMC 3k dataset...")
        adata = sc.datasets.pbmc3k()
        adata.var_names_make_unique()
        output_file = output_path / "pbmc3k_raw.h5ad"
        adata.write(output_file)
        print(f"Saved: {output_file}")
        return str(output_file)

    elif dataset_type == "pbmc3k_10x_mtx":
        # Download PBMC3K in 10x MTX format
        print("Downloading PBMC3K 10x MTX dataset...")
        return _download_pbmc3k_10x_mtx(output_path)

    elif dataset_type == "pbmc3k_10x_h5":
        # Download PBMC3K in 10x H5 format
        print("Downloading PBMC3K 10x H5 dataset...")
        return _download_pbmc3k_10x_h5(output_path)

    elif dataset_type == "pbmc_68k":
        # Download PBMC 68K from 10X
        print("Downloading PBMC 68k dataset...")
        url = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_feature_bc_matrix.h5"
        h5_file = output_path / "pbmc_68k.h5"
        h5ad_file = output_path / "pbmc_68k_raw.h5ad"

        # Download h5 file
        response = requests.get(url)
        with open(h5_file, "wb") as f:
            f.write(response.content)

        # Convert to h5ad
        adata = sc.read_10x_h5(h5_file)
        adata.var_names_make_unique()
        adata.write(h5ad_file)
        print(f"Saved: {h5ad_file}")
        return str(h5ad_file)

    elif dataset_type == "heart_10k":
        # Download Heart 10K from 10X
        print("Downloading Heart 10k dataset...")
        url = "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/heart_10k_v3/heart_10k_v3_filtered_feature_bc_matrix.h5"
        h5_file = output_path / "heart_10k.h5"
        h5ad_file = output_path / "heart_10k_raw.h5ad"

        # Download h5 file
        response = requests.get(url)
        with open(h5_file, "wb") as f:
            f.write(response.content)

        # Convert to h5ad
        adata = sc.read_10x_h5(h5_file)
        adata.var_names_make_unique()
        adata.write(h5ad_file)
        print(f"Saved: {h5ad_file}")
        return str(h5ad_file)

    else:
        print(f"Unknown dataset type: {dataset_type}")
        return None


def _download_pbmc3k_10x_mtx(output_path: Path) -> str:
    """Create PBMC3K dataset in 10x MTX format from existing h5ad"""
    # Use the existing PBMC3K h5ad file as source
    pbmc3k_h5ad = output_path / "pbmc3k_raw.h5ad"

    if not pbmc3k_h5ad.exists():
        # Download PBMC3K first
        print("Downloading PBMC3K dataset first...")
        download_dataset("pbmc3k", str(output_path))

    # Create directory for MTX files
    mtx_dir = output_path / "pbmc3k_10x_mtx"
    mtx_dir.mkdir(exist_ok=True)

    # Load the h5ad file and convert to MTX format
    print("Converting PBMC3K to MTX format...")
    adata = sc.read_h5ad(pbmc3k_h5ad)

    # Write MTX files
    import pandas as pd
    from scipy.io import mmwrite

    # Write matrix.mtx
    matrix_path = mtx_dir / "matrix.mtx"
    # Convert to scipy sparse matrix and transpose for MTX format
    from scipy.sparse import csr_matrix

    # Convert to numpy array first, then to sparse matrix
    X_array = _get_array_from_adata(adata.X)
    X_sparse = csr_matrix(X_array)
    mmwrite(str(matrix_path), X_sparse.T)  # Transpose for MTX format

    # Write barcodes.tsv (cell names)
    barcodes_path = mtx_dir / "barcodes.tsv"
    pd.DataFrame(adata.obs_names).to_csv(
        barcodes_path, sep="\t", header=False, index=False
    )

    # Write genes.tsv (gene names)
    genes_path = mtx_dir / "genes.tsv"
    pd.DataFrame({"gene_id": adata.var_names, "gene_symbol": adata.var_names}).to_csv(
        genes_path, sep="\t", header=False, index=False
    )

    print(f"Saved MTX files to: {mtx_dir}")
    return str(mtx_dir)


def _download_pbmc3k_10x_h5(output_path: Path) -> str:
    """Create PBMC3K dataset in 10x H5 format from existing h5ad"""
    # Use the existing PBMC3K h5ad file as source
    pbmc3k_h5ad = output_path / "pbmc3k_raw.h5ad"

    if not pbmc3k_h5ad.exists():
        # Download PBMC3K first
        print("Downloading PBMC3K dataset first...")
        download_dataset("pbmc3k", str(output_path))

    # Load the h5ad file and convert to H5 format
    print("Converting PBMC3K to H5 format...")
    adata = sc.read_h5ad(pbmc3k_h5ad)

    # Save as H5 file using scanpy
    h5_file = output_path / "pbmc3k_10x_h5.h5"
    adata.write_h5ad(h5_file, compression="gzip")

    print(f"Saved: {h5_file}")
    return str(h5_file)


def create_dataset(
    dataset_type: DatasetType,
    n_cells: int = 500_000,
    n_genes: int = 20_000,
    sparsity: float = 0.95,
    output_dir: str = DEFAULT_DATASET_DIR,
) -> str:
    """Create synthetic dataset and return path to h5ad file"""

    if dataset_type != "synthetic":
        raise ValueError("create_dataset only supports 'synthetic' dataset_type")

    print(f"Creating synthetic dataset: {n_cells:,} cells × {n_genes:,} genes")

    from scipy.sparse import random

    # Generate sparse matrix
    density = 1 - sparsity
    X = random(n_cells, n_genes, density=density, format="csr")
    X.data = np.random.lognormal(0, 1, size=len(X.data))  # Realistic expression values

    # Create realistic metadata
    cell_types = ["T_cell", "B_cell", "NK_cell", "Monocyte", "DC", "Neutrophil"]
    batches = [f"batch_{i}" for i in range(10)]

    obs = pd.DataFrame(
        {
            "cell_type": np.random.choice(cell_types, n_cells),
            "batch": np.random.choice(batches, n_cells),
            "n_genes_by_counts": np.random.poisson(1000, n_cells),
            "total_counts": np.random.lognormal(8, 1, n_cells),
            "pct_counts_mt": np.random.beta(2, 8, n_cells) * 30,  # 0-30% range
            "leiden": np.random.choice([str(i) for i in range(20)], n_cells),
            "high_mito": np.random.choice([True, False], n_cells, p=[0.2, 0.8]),
            "high_genes": np.random.choice([True, False], n_cells, p=[0.3, 0.7]),
        }
    )
    obs.index = [f"cell_{i}" for i in range(n_cells)]

    # Create var
    var = pd.DataFrame(
        {
            "gene_symbol": [f"GENE_{i}" for i in range(n_genes)],
            "highly_variable": np.random.choice([True, False], n_genes, p=[0.2, 0.8]),
        }
    )
    var.index = [f"ENSG_{i:08d}" for i in range(n_genes)]

    # Create AnnData
    adata = sc.AnnData(X=X, obs=obs, var=var)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    filename = output_path / f"synthetic_{n_cells // 1000}k_cells_raw.h5ad"
    adata.write(filename)
    print(f"Saved: {filename}")

    return str(filename)


def prepare_dataset(
    dataset_type: DatasetType,
    raw_h5ad_path: str | None = None,
    output_dir: str = DEFAULT_DATASET_DIR,
) -> tuple[str, sc.AnnData]:
    """Prepare dataset with analysis and return (processed_path, adata)"""

    if raw_h5ad_path is None:
        # Auto-generate path based on dataset type
        raw_h5ad_path = f"{output_dir}/{dataset_type}_raw.h5ad"

    print(f"Preparing {dataset_type} dataset from {raw_h5ad_path}...")

    # Load raw data
    adata = sc.read_h5ad(raw_h5ad_path)
    print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # Basic preprocessing
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Calculate QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    # Debug: check what columns were created
    print(f"Available obs columns: {list(adata.obs.columns)}")

    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=200)
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]

    # Handle mitochondrial filtering
    mt_col = _find_mitochondrial_column(adata.obs.columns)
    if mt_col:
        adata = adata[adata.obs[mt_col] < 20, :].copy()
        print(f"Filtered by {mt_col} < 20")
    else:
        print("No mitochondrial percentage column found, skipping MT filtering")
        adata = adata.copy()

    # For synthetic data, skip the expensive analysis steps
    if dataset_type == "synthetic":
        # Just add batch info and use existing metadata
        adata.obs["batch"] = np.random.choice(
            ["batch_1", "batch_2"], size=len(adata), p=[0.6, 0.4]
        )

        # Ensure standardized mitochondrial column
        if mt_col:
            adata.obs["pct_counts_mt"] = adata.obs[mt_col]
            adata.obs["high_mito"] = adata.obs[mt_col] > 10
        else:
            adata.obs["pct_counts_mt"] = 0.0
            adata.obs["high_mito"] = False

        adata.obs["high_genes"] = adata.obs.n_genes_by_counts > 1500

    else:
        # Full analysis pipeline for real datasets
        adata = _run_full_analysis_pipeline(adata, mt_col)

    print(f"Processed dataset: {adata.n_obs} cells × {adata.n_vars} genes")
    if "leiden" in adata.obs.columns:
        print(f"Leiden clusters: {adata.obs.leiden.unique()}")
    print(f"Available metadata: {list(adata.obs.columns)}")

    # Save processed dataset
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    processed_filename = output_path / f"{dataset_type}_processed.h5ad"
    adata.write(processed_filename)
    print(f"Saved: {processed_filename}")

    return str(processed_filename), adata


def _find_mitochondrial_column(columns: pd.Index) -> str | None:
    """Find mitochondrial percentage column"""
    for col in columns:
        if "mt" in col.lower() and "pct" in col.lower():
            return col
    return None


def _run_full_analysis_pipeline(adata: sc.AnnData, mt_col: str | None) -> sc.AnnData:
    """Run full scanpy analysis pipeline"""

    # Normalization and log transform (for clustering)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]

    # Keep sparse until the end - only scale dense subset for PCA
    adata_scaled = adata.copy()
    sc.pp.scale(adata_scaled, max_value=10)  # This densifies but on smaller HVG set

    # PCA and neighbors on scaled data
    sc.tl.pca(adata_scaled, svd_solver="arpack")
    sc.pp.neighbors(adata_scaled, n_neighbors=10, n_pcs=40)

    # Clustering on scaled data
    sc.tl.leiden(adata_scaled, resolution=0.5)

    # Transfer clustering results back to original sparse data
    adata.obs["leiden"] = adata_scaled.obs["leiden"]

    # Add some realistic metadata for filtering
    adata.obs["batch"] = ["batch_1"] * (len(adata) // 2) + ["batch_2"] * (
        len(adata) - len(adata) // 2
    )

    # Use the actual mitochondrial column name we found earlier
    if mt_col:
        adata.obs["high_mito"] = adata.obs[mt_col] > 10
        adata.obs["pct_counts_mt"] = adata.obs[mt_col]  # Create standardized name
    else:
        # Fallback: create dummy column
        adata.obs["high_mito"] = False
        adata.obs["pct_counts_mt"] = 0.0

    adata.obs["high_genes"] = adata.obs.n_genes_by_counts > 1500

    return adata


def get_or_create_dataset(
    dataset_type: DatasetType,
    force_download: bool = False,
    force_prepare: bool = False,
    output_dir: str = DEFAULT_DATASET_DIR,
    **create_kwargs: Any,
) -> tuple[str, str | None]:
    """
    Get or create dataset, returning (raw_path, processed_path)

    Args:
        dataset_type: Type of dataset to get/create
        force_download: Force re-download even if file exists
        force_prepare: Force re-processing even if processed file exists
        output_dir: Output directory
        **create_kwargs: Additional arguments for create_dataset (for synthetic data)

    Returns:
        Tuple of (raw_path, processed_path)
        For 10x formats: returns (raw_format_path, None) - no processed version
        For other formats: returns (raw_h5ad_path, processed_h5ad_path)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Determine file paths
    if dataset_type == "synthetic":
        n_cells = create_kwargs.get("n_cells", 500_000)
        raw_filename = f"synthetic_{n_cells // 1000}k_cells_raw.h5ad"
        processed_filename = f"synthetic_{n_cells // 1000}k_cells_processed.h5ad"
    else:
        raw_filename = f"{dataset_type}_raw.h5ad"
        processed_filename = f"{dataset_type}_processed.h5ad"

    raw_path = output_path / raw_filename
    processed_path = output_path / processed_filename

    # Get/create raw dataset
    if not raw_path.exists() or force_download:
        if dataset_type == "synthetic":
            create_dataset(dataset_type, output_dir=output_dir, **create_kwargs)
        else:
            # Download the dataset
            downloaded_path = download_dataset(dataset_type, output_dir=output_dir)

            if downloaded_path is None:
                raise ValueError(f"Failed to download dataset: {dataset_type}")

            # For 10x formats, use the original format path directly
            if dataset_type in ["pbmc3k_10x_mtx", "pbmc3k_10x_h5"]:
                raw_path = Path(downloaded_path)
                # Return early - no processed version for 10x formats
                return str(raw_path), None

    # Get/create processed dataset (only for non-10x formats)
    if not processed_path.exists() or force_prepare:
        prepare_dataset(dataset_type, str(raw_path), output_dir=output_dir)

    return str(raw_path), str(processed_path)


def create_tiny_sample_from_pbmc3k(output_dir: str = DEFAULT_DATASET_DIR) -> bool:
    """Create a tiny sample dataset by sampling from PBMC3K"""

    print("Creating tiny sample dataset from PBMC3K...")

    # Load PBMC3K dataset
    pbmc3k_path = Path(output_dir) / "pbmc3k_processed.h5ad"
    if not pbmc3k_path.exists():
        print(f"PBMC3K dataset not found at {pbmc3k_path}")
        print("Please ensure the PBMC3K dataset is available.")
        return False

    print("Loading PBMC3K dataset...")
    adata = sc.read_h5ad(pbmc3k_path)
    print(f"Original PBMC3K shape: {adata.shape}")

    # Sample a small subset: 100 cells and 50 genes
    n_cells = 100
    n_genes = 50

    print(f"Sampling {n_cells} cells and {n_genes} genes using scanpy.pp.sample...")

    # First, sample cells that have non-zero expression
    # Handle different data types properly
    X_array = _get_array_from_adata(adata.X)

    cell_counts = X_array.sum(axis=1)
    cell_mask = cell_counts > 0
    print(
        f"Cells with non-zero expression: {np.sum(cell_mask)} out of {len(cell_mask)}"
    )

    # Sample cells with non-zero expression
    adata = sc.pp.sample(adata, n=n_cells, axis="obs", copy=True, rng=42, p=cell_mask)

    # Then, sample genes that have non-zero expression in the selected cells
    X_array = _get_array_from_adata(adata.X)

    gene_counts = X_array.sum(axis=0)
    gene_mask = gene_counts > 0
    print(
        f"Genes with non-zero expression: {np.sum(gene_mask)} out of {len(gene_mask)}"
    )

    # Sample genes with non-zero expression
    adata = sc.pp.sample(adata, n=n_genes, axis="var", copy=True, rng=42, p=gene_mask)

    tiny_adata = adata
    print(f"Tiny dataset shape: {tiny_adata.shape}")

    # Verify we have some non-zero data
    data_array = _get_array_from_adata(tiny_adata.X)
    print(f"Non-zero elements: {np.count_nonzero(data_array)}")
    print(f"Data range: {data_array.min():.3f} to {data_array.max():.3f}")

    # Save to temporary h5ad file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_h5ad = Path(temp_dir) / "tiny_sample.h5ad"
        tiny_adata.write(temp_h5ad)

        # Convert to SLAF
        output_path = Path(output_dir) / "tiny_sample_dataset.slaf"

        # Remove existing dataset if it exists
        if output_path.exists():
            print(f"Removing existing dataset at {output_path}")
            shutil.rmtree(output_path)

        print("Converting to SLAF format...")
        from slaf.data.converter import SLAFConverter

        converter = SLAFConverter()
        converter.convert(str(temp_h5ad), str(output_path))

        print(f"Tiny sample dataset created at: {output_path}")

        # Verify the SLAF dataset
        from slaf.core.slaf import SLAFArray

        slaf = SLAFArray(str(output_path))
        print(f"SLAF dataset shape: {slaf.shape}")

        # Test that the join works
        print("Testing join query...")
        try:
            join_result = slaf.query(
                """
                SELECT gene_metadata.gene_id as meta_id, genes.gene_id as genes_id
                FROM gene_metadata, genes
                WHERE gene_metadata.gene_id = genes.gene_id
                LIMIT 3
            """
            )
            print(f"Join test successful! Result:\n{join_result}")
            print("✅ Join query works on the new tiny dataset!")
        except Exception as e:
            print(f"❌ Join query failed: {e}")
            return False

        # Test the QC query
        print("Testing QC query...")
        try:
            qc_result = slaf.query(
                """
                SELECT
                    gm.gene_integer_id AS gene_id,
                    COUNT(*) AS n_cells_by_counts,
                    SUM(CAST(j.value AS FLOAT)) AS total_counts
                FROM
                    genes g
                JOIN gene_metadata gm ON g.gene_id = gm.gene_id,
                    json_each(g.sparse_data) AS j
                GROUP BY
                    gm.gene_integer_id
                ORDER BY
                    gm.gene_integer_id
                LIMIT 5
            """
            )
            print(f"QC query successful! Result:\n{qc_result}")
            print("✅ QC query works on the new tiny dataset!")
        except Exception as e:
            print(f"❌ QC query failed: {e}")
            return False

    print("Tiny sample dataset creation completed successfully!")
    return True


def _get_array_from_adata(X):
    """Helper function to safely convert AnnData.X to numpy array"""
    if hasattr(X, "toarray"):
        return X.toarray()
    elif hasattr(X, "A"):
        return X.A
    else:
        return np.array(X)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for SLAF benchmarking"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[
            "pbmc3k",
            "pbmc3k_10x_mtx",
            "pbmc3k_10x_h5",
            "pbmc_68k",
            "heart_10k",
            "synthetic",
            "tiny_sample",
            "all",
        ],
        default=["pbmc3k"],
        help="Datasets to prepare (default: pbmc3k)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Output directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Force re-processing even if processed files exist",
    )
    parser.add_argument(
        "--synthetic-sizes",
        nargs="+",
        type=int,
        default=[50_000, 200_000, 500_000],
        help="Cell counts for synthetic datasets (default: 50000 200000 500000)",
    )

    args = parser.parse_args()

    # Expand "all" to all real dataset types
    if "all" in args.datasets:
        args.datasets = [
            "pbmc3k",
            "pbmc3k_10x_mtx",
            "pbmc3k_10x_h5",
            "pbmc_68k",
            "heart_10k",
            "synthetic",
            "tiny_sample",
        ]

    print("🧬 SLAF Dataset Preparation")
    print("=" * 40)
    print(f"Output directory: {args.output_dir}")
    print(f"Datasets to prepare: {args.datasets}")
    print(f"Force download: {args.force_download}")
    print(f"Force prepare: {args.force_prepare}")

    if "synthetic" in args.datasets:
        print(f"Synthetic dataset sizes: {args.synthetic_sizes}")

    print("\n" + "=" * 40)

    # Prepare each dataset
    prepared_datasets: list[tuple[str, str | None]] = []

    for dataset_type in args.datasets:
        # Cast to proper type for type checking
        dataset_type_typed: DatasetType = dataset_type  # type: ignore

        if dataset_type_typed == "tiny_sample":
            # Create tiny sample dataset
            print("\n📊 Creating tiny sample dataset...")
            try:
                success = create_tiny_sample_from_pbmc3k(output_dir=args.output_dir)
                if success:
                    prepared_datasets.append(
                        ("tiny_sample", f"{args.output_dir}/tiny_sample_dataset.slaf")
                    )
                    print("✅ Successfully created tiny sample dataset")
                else:
                    print("❌ Failed to create tiny sample dataset")
            except Exception as e:
                print(f"❌ Failed to create tiny sample dataset: {e}")

        elif dataset_type_typed == "synthetic":
            # Create multiple synthetic datasets of different sizes
            for n_cells in args.synthetic_sizes:
                print(f"\n📊 Preparing synthetic dataset ({n_cells:,} cells)...")
                try:
                    raw_path, processed_path = get_or_create_dataset(
                        dataset_type_typed,
                        force_download=args.force_download,
                        force_prepare=args.force_prepare,
                        output_dir=args.output_dir,
                        n_cells=n_cells,
                        n_genes=min(
                            20_000 + (n_cells // 50_000) * 5_000, 30_000
                        ),  # Scale genes with cells
                        sparsity=0.95,
                    )
                    prepared_datasets.append(
                        (f"synthetic_{n_cells // 1000}k", processed_path)
                    )
                    print(
                        f"✅ Successfully prepared synthetic_{n_cells // 1000}k dataset"
                    )
                except Exception as e:
                    print(
                        f"❌ Failed to prepare synthetic_{n_cells // 1000}k dataset: {e}"
                    )

        else:
            # Prepare real dataset
            print(f"\n📊 Preparing {dataset_type_typed} dataset...")
            try:
                raw_path, processed_path = get_or_create_dataset(
                    dataset_type_typed,
                    force_download=args.force_download,
                    force_prepare=args.force_prepare,
                    output_dir=args.output_dir,
                )
                prepared_datasets.append((dataset_type_typed, processed_path))
                print(f"✅ Successfully prepared {dataset_type_typed} dataset")
            except Exception as e:
                print(f"❌ Failed to prepare {dataset_type_typed} dataset: {e}")

    # Summary
    print("\n" + "=" * 40)
    print("📋 DATASET PREPARATION SUMMARY")
    print("=" * 40)

    if prepared_datasets:
        print("✅ Successfully prepared datasets:")
        for name, path in prepared_datasets:
            # Get dataset info
            if path is None:
                print(f"  • {name}: (raw format only)")
            else:
                try:
                    adata = sc.read_h5ad(path, backed="r")  # Don't load into memory
                    print(f"  • {name}: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
                except Exception:
                    print(f"  • {name}: {path}")

        print("\n🎯 Ready for SLAF conversion and benchmarking!")
        print("   Next steps:")
        print("   1. Convert to SLAF: python converter.py")
        print("   2. Run benchmarks: python benchmarks.py")
    else:
        print("❌ No datasets were successfully prepared")

    print("\n" + "=" * 40)
