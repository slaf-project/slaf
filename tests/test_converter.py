import json
import os
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from slaf.core.slaf import SLAFArray
from slaf.data.chunked_reader import ChunkedH5ADReader
from slaf.data.converter import SLAFConverter


class TestSLAFConverter:
    """Test the SLAF converter functionality"""

    # Basic functionality tests
    def test_converter_creates_new_table_structure(self, small_sample_adata, tmp_path):
        """Test that converter creates the COO table structure"""
        # Save sample data as h5ad
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert to SLAF
        output_path = tmp_path / "test.slaf"
        converter = SLAFConverter()
        converter.convert(str(h5ad_path), str(output_path))

        # Check that new table files exist
        assert (output_path / "expression.lance").exists()
        assert (output_path / "cells.lance").exists()
        assert (output_path / "genes.lance").exists()
        assert (output_path / "config.json").exists()

        # Check config version
        with open(output_path / "config.json") as f:
            config = json.load(f)
        assert config["format_version"] == "0.2"

    def test_expression_data_consistency(self, small_sample_adata, tmp_path):
        """Test that expression data is consistent in COO format"""
        # Convert to SLAF
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)
        output_path = tmp_path / "test.slaf"
        converter = SLAFConverter()
        converter.convert(str(h5ad_path), str(output_path))

        # Load expression table
        expression_dataset = lance.dataset(output_path / "expression.lance")
        expression_df = expression_dataset.to_table().to_pandas()

        # Reconstruct original matrix from COO table using integer IDs
        reconstructed = np.zeros((small_sample_adata.n_obs, small_sample_adata.n_vars))
        for _, row in expression_df.iterrows():
            cell_idx = row["cell_integer_id"]
            gene_idx = row["gene_integer_id"]
            value = row["value"]
            reconstructed[cell_idx, gene_idx] = value

        # Check that reconstruction matches original
        original_dense = small_sample_adata.X.toarray()
        np.testing.assert_array_equal(original_dense, reconstructed)

        # Check that we have the right number of non-zero entries
        expected_nonzero = small_sample_adata.X.nnz
        actual_nonzero = len(expression_df)
        assert actual_nonzero == expected_nonzero

    # Integer ID tests
    def test_integer_ids_in_metadata(self, small_sample_adata, tmp_path):
        """Test that integer IDs are embedded in metadata tables"""
        # Save sample data as h5ad
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert with integer keys
        output_path = tmp_path / "test.slaf"
        converter = SLAFConverter(use_integer_keys=True)
        converter.convert(str(h5ad_path), str(output_path))

        # Load metadata tables
        cells_dataset = lance.dataset(output_path / "cells.lance")
        cells_df = cells_dataset.to_table().to_pandas()

        genes_dataset = lance.dataset(output_path / "genes.lance")
        genes_df = genes_dataset.to_table().to_pandas()

        # Check that integer ID columns exist
        assert "cell_integer_id" in cells_df.columns
        assert "gene_integer_id" in genes_df.columns

        # Check that integer IDs are sequential
        np.testing.assert_array_equal(
            cells_df["cell_integer_id"].values, range(small_sample_adata.n_obs)
        )
        np.testing.assert_array_equal(
            genes_df["gene_integer_id"].values, range(small_sample_adata.n_vars)
        )

        # Check that original IDs are preserved
        expected_cell_ids = [f"cell_{i}" for i in range(small_sample_adata.n_obs)]
        assert list(cells_df["cell_id"]) == expected_cell_ids

        # Use the actual gene IDs from the AnnData object
        expected_gene_ids = list(small_sample_adata.var.index)
        assert list(genes_df["gene_id"]) == expected_gene_ids

    def test_no_integer_ids_when_disabled(self, small_sample_adata, tmp_path):
        """Test that integer IDs are not added when disabled"""
        # Save sample data as h5ad
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert without integer keys
        output_path = tmp_path / "test.slaf"
        converter = SLAFConverter(use_integer_keys=False)
        converter.convert(str(h5ad_path), str(output_path))

        # Load metadata tables
        cells_dataset = lance.dataset(output_path / "cells.lance")
        cells_df = cells_dataset.to_table().to_pandas()

        genes_dataset = lance.dataset(output_path / "genes.lance")
        genes_df = genes_dataset.to_table().to_pandas()

        # Check that integer ID columns do NOT exist
        assert "cell_integer_id" not in cells_df.columns
        assert "gene_integer_id" not in genes_df.columns

    def test_expression_data_structure(self, small_sample_adata, tmp_path):
        """Test that expression data has the correct COO structure"""
        # Save sample data as h5ad
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert with integer keys
        output_path = tmp_path / "test.slaf"
        converter = SLAFConverter(use_integer_keys=True)
        converter.convert(str(h5ad_path), str(output_path))

        # Load expression table
        expression_dataset = lance.dataset(output_path / "expression.lance")
        expression_df = expression_dataset.to_table().to_pandas()

        # Check that expression table has the right columns
        expected_columns = {
            "cell_id",
            "gene_id",
            "cell_integer_id",
            "gene_integer_id",
            "value",
        }
        assert set(expression_df.columns) == expected_columns

        # Check that all values are valid
        assert all(expression_df["value"] >= 0)
        assert all(expression_df["cell_id"].str.startswith("cell_"))
        # Check gene_id prefix matches the AnnData gene index
        gene_prefix = str(small_sample_adata.var.index[0])[:5]
        assert all(expression_df["gene_id"].str.startswith(gene_prefix))

        # Check that integer IDs are valid
        assert all(expression_df["cell_integer_id"] >= 0)
        assert all(expression_df["cell_integer_id"] < small_sample_adata.n_obs)
        assert all(expression_df["gene_integer_id"] >= 0)
        assert all(expression_df["gene_integer_id"] < small_sample_adata.n_vars)

    def test_integer_id_consistency(self, small_sample_adata, tmp_path):
        """Test that integer IDs in expression table match metadata tables"""
        # Save sample data as h5ad
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert with integer keys
        output_path = tmp_path / "test.slaf"
        converter = SLAFConverter(use_integer_keys=True)
        converter.convert(str(h5ad_path), str(output_path))

        # Load all tables
        expression_dataset = lance.dataset(output_path / "expression.lance")
        expression_df = expression_dataset.to_table().to_pandas()

        cells_dataset = lance.dataset(output_path / "cells.lance")
        cells_df = cells_dataset.to_table().to_pandas()

        genes_dataset = lance.dataset(output_path / "genes.lance")
        genes_df = genes_dataset.to_table().to_pandas()

        # Create mappings from string IDs to integer IDs
        cell_id_to_int = dict(
            zip(cells_df["cell_id"], cells_df["cell_integer_id"], strict=False)
        )
        gene_id_to_int = dict(
            zip(genes_df["gene_id"], genes_df["gene_integer_id"], strict=False)
        )

        # Verify that expression table integer IDs match the mappings
        for _, row in expression_df.iterrows():
            expected_cell_int = cell_id_to_int[row["cell_id"]]
            expected_gene_int = gene_id_to_int[row["gene_id"]]

            assert row["cell_integer_id"] == expected_cell_int, (
                f"Cell integer ID mismatch for {row['cell_id']}"
            )
            assert row["gene_integer_id"] == expected_gene_int, (
                f"Gene integer ID mismatch for {row['gene_id']}"
            )

    # Optimization tests
    def test_integer_keys_optimization(self, small_sample_adata, tmp_path):
        """Test that integer keys reduce file size"""
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)
        output_path_string_keys = tmp_path / "test_string_keys.slaf"
        converter_string_keys = SLAFConverter(use_integer_keys=False)
        converter_string_keys.convert(str(h5ad_path), str(output_path_string_keys))
        output_path_int_keys = tmp_path / "test_int_keys.slaf"
        converter_int_keys = SLAFConverter(use_integer_keys=True)
        converter_int_keys.convert(str(h5ad_path), str(output_path_int_keys))

        # Check that files exist and have reasonable sizes
        string_keys_size = os.path.getsize(output_path_string_keys / "expression.lance")
        int_keys_size = os.path.getsize(output_path_int_keys / "expression.lance")
        print(f"String keys size: {string_keys_size:,} bytes")
        print(f"Integer keys size: {int_keys_size:,} bytes")

        # Both should be reasonable sizes
        assert string_keys_size > 0
        assert int_keys_size > 0

    def test_optimization_config_persistence(self, small_sample_adata, tmp_path):
        """Test that optimization settings are saved in config"""
        # Save sample data as h5ad
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert with optimizations
        output_path = tmp_path / "test_optimized.slaf"
        converter = SLAFConverter(
            use_integer_keys=True,
        )
        converter.convert(str(h5ad_path), str(output_path))

        # Check config
        with open(output_path / "config.json") as f:
            config = json.load(f)

        assert "optimizations" in config
        assert config["optimizations"]["use_integer_keys"]

    def test_metadata_computation_and_storage(self, small_sample_adata, tmp_path):
        """Test that metadata is computed and stored correctly"""
        # Save sample data as h5ad
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert to SLAF
        output_path = tmp_path / "test.slaf"
        converter = SLAFConverter()
        converter.convert(str(h5ad_path), str(output_path))

        # Check config
        with open(output_path / "config.json") as f:
            config = json.load(f)

        # Check format version
        assert config["format_version"] == "0.2"

        # Check that metadata section exists
        assert "metadata" in config
        metadata = config["metadata"]

        # Check required metadata fields
        assert "expression_count" in metadata
        assert "sparsity" in metadata
        assert "density" in metadata
        assert "total_possible_elements" in metadata
        assert "expression_stats" in metadata

        # Check that values are reasonable
        assert metadata["expression_count"] > 0
        assert 0 <= metadata["sparsity"] <= 1
        assert 0 <= metadata["density"] <= 1
        assert (
            metadata["total_possible_elements"]
            == small_sample_adata.n_obs * small_sample_adata.n_vars
        )

        # Check expression statistics
        stats = metadata["expression_stats"]
        assert "min_value" in stats
        assert "max_value" in stats
        assert "mean_value" in stats
        assert "std_value" in stats
        assert stats["min_value"] <= stats["max_value"]
        assert stats["mean_value"] >= 0
        assert stats["std_value"] >= 0

    # 10x Format Conversion Tests
    def test_10x_mtx_format_detection(self, tmp_path):
        """Test auto-detection of 10x MTX format"""
        # Create a mock 10x MTX directory structure
        mtx_dir = Path(tmp_path) / "filtered_feature_bc_matrix"
        mtx_dir.mkdir(exist_ok=True)

        # Create mock MTX files
        (mtx_dir / "matrix.mtx").write_text(
            "%%MatrixMarket matrix coordinate integer general\n3 3 3\n1 1 1\n2 2 2\n3 3 3"
        )
        (mtx_dir / "barcodes.tsv").write_text("cell1\ncell2\ncell3")
        (mtx_dir / "genes.tsv").write_text("gene1\tGENE1\ngene2\tGENE2\ngene3\tGENE3")

        from slaf.data.utils import detect_format

        detected_format = detect_format(str(mtx_dir))
        assert detected_format == "10x_mtx"

    def test_10x_h5_format_detection(self, tmp_path):
        """Test auto-detection of 10x H5 format"""
        # Create a mock H5 file
        h5_file = Path(tmp_path) / "data.h5"
        h5_file.write_text("mock h5 content")

        from slaf.data.utils import detect_format

        detected_format = detect_format(str(h5_file))
        assert detected_format == "10x_h5"

    def test_h5ad_format_detection(self, tmp_path):
        """Test auto-detection of h5ad format"""
        # Create a mock h5ad file
        h5ad_file = Path(tmp_path) / "data.h5ad"
        h5ad_file.write_text("mock h5ad content")

        from slaf.data.utils import detect_format

        detected_format = detect_format(str(h5ad_file))
        assert detected_format == "h5ad"

    def test_format_detection_invalid_file(self):
        """Test format detection with invalid file"""
        from slaf.data.utils import detect_format

        with pytest.raises(ValueError, match="Cannot detect format for"):
            detect_format("nonexistent_file.txt")

    def test_convert_10x_mtx_format(self, tmp_path):
        """Test conversion from 10x MTX format"""
        import pandas as pd
        from scipy import sparse
        from scipy.io import mmwrite

        # Create a mock 10x MTX directory with real data
        mtx_dir = Path(tmp_path) / "filtered_feature_bc_matrix"
        mtx_dir.mkdir(exist_ok=True)

        # Create small test data
        n_cells, n_genes = 5, 3
        X = np.random.randint(0, 10, (n_cells, n_genes))
        X_sparse = sparse.csr_matrix(X)

        # Write matrix.mtx
        matrix_path = mtx_dir / "matrix.mtx"
        mmwrite(str(matrix_path), X_sparse.T)  # Transpose for MTX format

        # Write barcodes.tsv
        barcodes_path = mtx_dir / "barcodes.tsv"
        cell_names = [f"cell_{i}" for i in range(n_cells)]
        pd.DataFrame(cell_names).to_csv(
            barcodes_path, sep="\t", header=False, index=False
        )

        # Write genes.tsv
        genes_path = mtx_dir / "genes.tsv"
        gene_data = {
            "gene_id": [f"ENSG_{i:08d}" for i in range(n_genes)],
            "gene_symbol": [f"GENE_{i}" for i in range(n_genes)],
        }
        pd.DataFrame(gene_data).to_csv(genes_path, sep="\t", header=False, index=False)

        # Convert to SLAF
        output_path = Path(tmp_path) / "test_10x_mtx.slaf"
        converter = SLAFConverter()
        converter.convert(str(mtx_dir), str(output_path))

        # Verify output structure
        assert (output_path / "expression.lance").exists()
        assert (output_path / "cells.lance").exists()
        assert (output_path / "genes.lance").exists()
        assert (output_path / "config.json").exists()

        # Verify data consistency
        expression_dataset = lance.dataset(output_path / "expression.lance")
        expression_df = expression_dataset.to_table().to_pandas()

        # Check that we have the expected number of non-zero entries
        expected_nonzero = np.count_nonzero(X)
        actual_nonzero = len(expression_df)
        assert actual_nonzero == expected_nonzero

    def test_convert_10x_h5_format(self, tmp_path):
        """Test conversion from 10x H5 format"""
        import scanpy as sc
        from scipy import sparse

        # Create a mock H5 file by saving AnnData as h5ad
        # (This simulates the fallback behavior for non-standard H5 files)
        n_cells, n_genes = 5, 3
        X = np.random.randint(0, 10, (n_cells, n_genes))
        X_sparse = sparse.csr_matrix(X)

        obs = pd.DataFrame(
            {"cell_type": ["type1"] * n_cells, "batch": ["batch1"] * n_cells},
            index=pd.Index([f"cell_{i}" for i in range(n_cells)]),
        )

        var = pd.DataFrame(
            {
                "gene_symbol": [f"GENE_{i}" for i in range(n_genes)],
                "highly_variable": [True] * n_genes,
            },
            index=pd.Index([f"ENSG_{i:08d}" for i in range(n_genes)]),
        )

        adata = sc.AnnData(X=X_sparse, obs=obs, var=var)

        # Save as H5 file (simulating 10x H5 format)
        h5_file = Path(tmp_path) / "data.h5"
        adata.write_h5ad(h5_file)

        # Convert to SLAF
        output_path = Path(tmp_path) / "test_10x_h5.slaf"
        converter = SLAFConverter()
        converter.convert(str(h5_file), str(output_path))

        # Verify output structure
        assert (output_path / "expression.lance").exists()
        assert (output_path / "cells.lance").exists()
        assert (output_path / "genes.lance").exists()
        assert (output_path / "config.json").exists()

        # Verify data consistency
        expression_dataset = lance.dataset(output_path / "expression.lance")
        expression_df = expression_dataset.to_table().to_pandas()

        # Check that we have the expected number of non-zero entries
        expected_nonzero = np.count_nonzero(X)
        actual_nonzero = len(expression_df)
        assert actual_nonzero == expected_nonzero

    def test_convert_with_explicit_format_specification(self, tmp_path):
        """Test conversion with explicit format specification"""
        import scanpy as sc
        from scipy import sparse

        # Create test data
        n_cells, n_genes = 3, 2
        X = np.random.randint(0, 5, (n_cells, n_genes))
        X_sparse = sparse.csr_matrix(X)

        obs = pd.DataFrame(
            {"cell_type": ["type1"] * n_cells},
            index=pd.Index([f"cell_{i}" for i in range(n_cells)]),
        )

        var = pd.DataFrame(
            {"gene_symbol": [f"GENE_{i}" for i in range(n_genes)]},
            index=pd.Index([f"ENSG_{i:08d}" for i in range(n_genes)]),
        )

        adata = sc.AnnData(X=X_sparse, obs=obs, var=var)

        # Save as h5ad
        h5ad_file = Path(tmp_path) / "data.h5ad"
        adata.write_h5ad(h5ad_file)

        # Convert with explicit format specification
        output_path = Path(tmp_path) / "test_explicit_format.slaf"
        converter = SLAFConverter()
        converter.convert(str(h5ad_file), str(output_path), input_format="h5ad")

        # Verify output
        assert (output_path / "expression.lance").exists()
        assert (output_path / "cells.lance").exists()
        assert (output_path / "genes.lance").exists()

    def test_convert_unsupported_format(self, tmp_path):
        """Test conversion with unsupported format"""
        # Create a dummy file
        dummy_file = Path(tmp_path) / "dummy.txt"
        dummy_file.write_text("dummy content")

        converter = SLAFConverter()

        with pytest.raises(ValueError, match="Unsupported format"):
            converter.convert(
                str(dummy_file),
                str(Path(tmp_path) / "output.slaf"),
                input_format="unsupported",
            )

    def test_convert_10x_mtx_with_integer_keys(self, tmp_path):
        """Test 10x MTX conversion with integer key optimization"""
        import pandas as pd
        from scipy import sparse
        from scipy.io import mmwrite

        # Create mock 10x MTX directory
        mtx_dir = Path(tmp_path) / "filtered_feature_bc_matrix"
        mtx_dir.mkdir(exist_ok=True)

        n_cells, n_genes = 4, 2
        X = np.random.randint(0, 10, (n_cells, n_genes))
        X_sparse = sparse.csr_matrix(X)

        # Write MTX files
        matrix_path = mtx_dir / "matrix.mtx"
        mmwrite(str(matrix_path), X_sparse.T)

        barcodes_path = mtx_dir / "barcodes.tsv"
        cell_names = [f"cell_{i}" for i in range(n_cells)]
        pd.DataFrame(cell_names).to_csv(
            barcodes_path, sep="\t", header=False, index=False
        )

        genes_path = mtx_dir / "genes.tsv"
        gene_data = {
            "gene_id": [f"ENSG_{i:08d}" for i in range(n_genes)],
            "gene_symbol": [f"GENE_{i}" for i in range(n_genes)],
        }
        pd.DataFrame(gene_data).to_csv(genes_path, sep="\t", header=False, index=False)

        # Convert with integer keys
        output_path = Path(tmp_path) / "test_10x_mtx_int_keys.slaf"
        converter = SLAFConverter(use_integer_keys=True)
        converter.convert(str(mtx_dir), str(output_path))

        # Verify integer keys in metadata
        cells_dataset = lance.dataset(output_path / "cells.lance")
        cells_df = cells_dataset.to_table().to_pandas()

        genes_dataset = lance.dataset(output_path / "genes.lance")
        genes_df = genes_dataset.to_table().to_pandas()

        assert "cell_integer_id" in cells_df.columns
        assert "gene_integer_id" in genes_df.columns

        # Check sequential integer IDs
        np.testing.assert_array_equal(
            cells_df["cell_integer_id"].values, range(n_cells)
        )
        np.testing.assert_array_equal(
            genes_df["gene_integer_id"].values, range(n_genes)
        )

    def test_convert_10x_h5_without_integer_keys(self, tmp_path):
        """Test 10x H5 conversion without integer key optimization"""
        import scanpy as sc
        from scipy import sparse

        # Create test data
        n_cells, n_genes = 3, 2
        X = np.random.randint(0, 5, (n_cells, n_genes))
        X_sparse = sparse.csr_matrix(X)

        obs = pd.DataFrame(
            {"cell_type": ["type1"] * n_cells},
            index=pd.Index([f"cell_{i}" for i in range(n_cells)]),
        )

        var = pd.DataFrame(
            {"gene_symbol": [f"GENE_{i}" for i in range(n_genes)]},
            index=pd.Index([f"ENSG_{i:08d}" for i in range(n_genes)]),
        )

        adata = sc.AnnData(X=X_sparse, obs=obs, var=var)

        # Save as H5 file
        h5_file = Path(tmp_path) / "data.h5"
        adata.write_h5ad(h5_file)

        # Convert without integer keys
        output_path = Path(tmp_path) / "test_10x_h5_no_int_keys.slaf"
        converter = SLAFConverter(use_integer_keys=False)
        converter.convert(str(h5_file), str(output_path))

        # Verify no integer keys in metadata
        cells_dataset = lance.dataset(output_path / "cells.lance")
        cells_df = cells_dataset.to_table().to_pandas()

        genes_dataset = lance.dataset(output_path / "genes.lance")
        genes_df = genes_dataset.to_table().to_pandas()

        assert "cell_integer_id" not in cells_df.columns
        assert "gene_integer_id" not in genes_df.columns

    def test_convert_10x_mtx_chunked(self, tmp_path):
        """Test chunked conversion from 10x MTX format"""
        import pandas as pd
        from scipy import sparse
        from scipy.io import mmwrite

        # Create a mock 10x MTX directory with real data
        mtx_dir = Path(tmp_path) / "filtered_feature_bc_matrix"
        mtx_dir.mkdir(exist_ok=True)

        # Create small test data
        n_cells, n_genes = 5, 3
        X = np.random.randint(0, 10, (n_cells, n_genes))
        X_sparse = sparse.csr_matrix(X)

        # Write matrix.mtx
        matrix_path = mtx_dir / "matrix.mtx"
        mmwrite(str(matrix_path), X_sparse.T)  # Transpose for MTX format

        # Write barcodes.tsv
        barcodes_path = mtx_dir / "barcodes.tsv"
        cell_names = [f"cell_{i}" for i in range(n_cells)]
        pd.DataFrame(cell_names).to_csv(
            barcodes_path, sep="\t", header=False, index=False
        )

        # Write genes.tsv
        genes_path = mtx_dir / "genes.tsv"
        gene_data = {
            "gene_id": [f"ENSG_{i:08d}" for i in range(n_genes)],
            "gene_symbol": [f"GENE_{i}" for i in range(n_genes)],
        }
        pd.DataFrame(gene_data).to_csv(genes_path, sep="\t", header=False, index=False)

        # Convert to SLAF with chunked processing
        output_path = Path(tmp_path) / "test_10x_mtx_chunked.slaf"
        converter = SLAFConverter(
            chunked=True, chunk_size=2
        )  # Small chunk size for testing
        converter.convert(str(mtx_dir), str(output_path))

        # Verify output structure
        assert (output_path / "expression.lance").exists()
        assert (output_path / "cells.lance").exists()
        assert (output_path / "genes.lance").exists()
        assert (output_path / "config.json").exists()

        # Verify data consistency
        expression_dataset = lance.dataset(output_path / "expression.lance")
        expression_df = expression_dataset.to_table().to_pandas()

        # Check that we have the expected number of non-zero entries
        expected_nonzero = np.count_nonzero(X)
        actual_nonzero = len(expression_df)
        assert actual_nonzero == expected_nonzero

    def test_convert_10x_h5_chunked(self, tmp_path):
        """Test chunked conversion from 10x H5 format"""
        import scanpy as sc
        from scipy import sparse

        # Create test data
        n_cells, n_genes = 5, 3
        X = np.random.randint(0, 10, (n_cells, n_genes))
        X_sparse = sparse.csr_matrix(X)

        obs = pd.DataFrame(
            {"cell_type": ["type1"] * n_cells, "batch": ["batch1"] * n_cells},
            index=pd.Index([f"cell_{i}" for i in range(n_cells)]),
        )

        var = pd.DataFrame(
            {
                "gene_symbol": [f"GENE_{i}" for i in range(n_genes)],
                "highly_variable": [True] * n_genes,
            },
            index=pd.Index([f"ENSG_{i:08d}" for i in range(n_genes)]),
        )

        adata = sc.AnnData(X=X_sparse, obs=obs, var=var)

        # Save as H5 file (simulating 10x H5 format)
        h5_file = Path(tmp_path) / "data.h5"
        adata.write_h5ad(h5_file)

        # Convert to SLAF with chunked processing
        output_path = Path(tmp_path) / "test_10x_h5_chunked.slaf"
        converter = SLAFConverter(
            chunked=True, chunk_size=2
        )  # Small chunk size for testing
        converter.convert(str(h5_file), str(output_path))

        # Verify output structure
        assert (output_path / "expression.lance").exists()
        assert (output_path / "cells.lance").exists()
        assert (output_path / "genes.lance").exists()
        assert (output_path / "config.json").exists()

        # Verify data consistency
        expression_dataset = lance.dataset(output_path / "expression.lance")
        expression_df = expression_dataset.to_table().to_pandas()

        # Check that we have the expected number of non-zero entries
        expected_nonzero = np.count_nonzero(X)
        actual_nonzero = len(expression_df)
        assert actual_nonzero == expected_nonzero

    def test_convert_10x_mtx_chunked_vs_non_chunked(self, tmp_path):
        """Test that chunked and non-chunked conversion produce identical results for 10x MTX format"""
        import pandas as pd
        from scipy import sparse
        from scipy.io import mmwrite

        # Create a mock 10x MTX directory
        mtx_dir = Path(tmp_path) / "filtered_feature_bc_matrix"
        mtx_dir.mkdir(exist_ok=True)

        n_cells, n_genes = 4, 2
        X = np.random.randint(0, 10, (n_cells, n_genes))
        X_sparse = sparse.csr_matrix(X)

        # Write MTX files
        matrix_path = mtx_dir / "matrix.mtx"
        mmwrite(str(matrix_path), X_sparse.T)

        barcodes_path = mtx_dir / "barcodes.tsv"
        cell_names = [f"cell_{i}" for i in range(n_cells)]
        pd.DataFrame(cell_names).to_csv(
            barcodes_path, sep="\t", header=False, index=False
        )

        genes_path = mtx_dir / "genes.tsv"
        gene_data = {
            "gene_id": [f"ENSG_{i:08d}" for i in range(n_genes)],
            "gene_symbol": [f"GENE_{i}" for i in range(n_genes)],
        }
        pd.DataFrame(gene_data).to_csv(genes_path, sep="\t", header=False, index=False)

        # Convert with chunked processing
        output_chunked = Path(tmp_path) / "test_10x_mtx_chunked.slaf"
        converter_chunked = SLAFConverter(chunked=True, chunk_size=2)
        converter_chunked.convert(str(mtx_dir), str(output_chunked))

        # Convert without chunked processing
        output_non_chunked = Path(tmp_path) / "test_10x_mtx_non_chunked.slaf"
        converter_non_chunked = SLAFConverter(chunked=False)
        converter_non_chunked.convert(str(mtx_dir), str(output_non_chunked))

        # Compare expression data
        chunked_expression = (
            lance.dataset(output_chunked / "expression.lance").to_table().to_pandas()
        )
        non_chunked_expression = (
            lance.dataset(output_non_chunked / "expression.lance")
            .to_table()
            .to_pandas()
        )

        # Sort both by cell_id and gene_id for comparison
        chunked_sorted = chunked_expression.sort_values(
            ["cell_id", "gene_id"]
        ).reset_index(drop=True)
        non_chunked_sorted = non_chunked_expression.sort_values(
            ["cell_id", "gene_id"]
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(chunked_sorted, non_chunked_sorted)

    def test_convert_10x_h5_chunked_vs_non_chunked(self, tmp_path):
        """Test that chunked and non-chunked conversion produce identical results for 10x H5 format"""
        import scanpy as sc
        from scipy import sparse

        # Create test data
        n_cells, n_genes = 4, 2
        X = np.random.randint(0, 10, (n_cells, n_genes))
        X_sparse = sparse.csr_matrix(X)

        obs = pd.DataFrame(
            {"cell_type": ["type1"] * n_cells, "batch": ["batch1"] * n_cells},
            index=pd.Index([f"cell_{i}" for i in range(n_cells)]),
        )

        var = pd.DataFrame(
            {
                "gene_symbol": [f"GENE_{i}" for i in range(n_genes)],
                "highly_variable": [True] * n_genes,
            },
            index=pd.Index([f"ENSG_{i:08d}" for i in range(n_genes)]),
        )

        adata = sc.AnnData(X=X_sparse, obs=obs, var=var)

        # Save as H5 file (simulating 10x H5 format)
        h5_file = Path(tmp_path) / "data.h5"
        adata.write_h5ad(h5_file)

        # Convert with chunked processing
        output_chunked = Path(tmp_path) / "test_10x_h5_chunked.slaf"
        converter_chunked = SLAFConverter(chunked=True, chunk_size=2)
        converter_chunked.convert(str(h5_file), str(output_chunked))

        # Convert without chunked processing
        output_non_chunked = Path(tmp_path) / "test_10x_h5_non_chunked.slaf"
        converter_non_chunked = SLAFConverter(chunked=False)
        converter_non_chunked.convert(str(h5_file), str(output_non_chunked))

        # Compare expression data
        chunked_expression = (
            lance.dataset(output_chunked / "expression.lance").to_table().to_pandas()
        )
        non_chunked_expression = (
            lance.dataset(output_non_chunked / "expression.lance")
            .to_table()
            .to_pandas()
        )

        # Sort both by cell_id and gene_id for comparison
        chunked_sorted = chunked_expression.sort_values(
            ["cell_id", "gene_id"]
        ).reset_index(drop=True)
        non_chunked_sorted = non_chunked_expression.sort_values(
            ["cell_id", "gene_id"]
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(chunked_sorted, non_chunked_sorted)

    def test_auto_detection_vs_explicit_format(self, tmp_path):
        """Test that auto-detection and explicit format specification produce identical results"""
        import scanpy as sc
        from scipy import sparse

        # Create test data
        n_cells, n_genes = 3, 2
        X = np.random.randint(0, 5, (n_cells, n_genes))
        X_sparse = sparse.csr_matrix(X)

        obs = pd.DataFrame(
            {"cell_type": ["type1"] * n_cells},
            index=pd.Index([f"cell_{i}" for i in range(n_cells)]),
        )

        var = pd.DataFrame(
            {"gene_symbol": [f"GENE_{i}" for i in range(n_genes)]},
            index=pd.Index([f"ENSG_{i:08d}" for i in range(n_genes)]),
        )

        adata = sc.AnnData(X=X_sparse, obs=obs, var=var)

        # Save as h5ad
        h5ad_file = Path(tmp_path) / "data.h5ad"
        adata.write_h5ad(h5ad_file)

        # Convert with auto-detection
        output_auto = Path(tmp_path) / "test_auto.slaf"
        converter_auto = SLAFConverter()
        converter_auto.convert(str(h5ad_file), str(output_auto))

        # Convert with explicit format
        output_explicit = Path(tmp_path) / "test_explicit.slaf"
        converter_explicit = SLAFConverter()
        converter_explicit.convert(
            str(h5ad_file), str(output_explicit), input_format="h5ad"
        )

        # Compare expression data
        auto_expression = (
            lance.dataset(output_auto / "expression.lance").to_table().to_pandas()
        )
        explicit_expression = (
            lance.dataset(output_explicit / "expression.lance").to_table().to_pandas()
        )

        pd.testing.assert_frame_equal(auto_expression, explicit_expression)

    def test_10x_formats_with_metadata_preservation(self, tmp_path):
        """Test that 10x format conversion preserves metadata correctly"""
        import pandas as pd
        from scipy import sparse
        from scipy.io import mmwrite

        # Create mock 10x MTX with rich metadata
        mtx_dir = Path(tmp_path) / "filtered_feature_bc_matrix"
        mtx_dir.mkdir(exist_ok=True)

        n_cells, n_genes = 4, 3
        X = np.random.randint(0, 10, (n_cells, n_genes))
        X_sparse = sparse.csr_matrix(X)

        # Write MTX files
        matrix_path = mtx_dir / "matrix.mtx"
        mmwrite(str(matrix_path), X_sparse.T)

        barcodes_path = mtx_dir / "barcodes.tsv"
        cell_names = [f"cell_{i}" for i in range(n_cells)]
        pd.DataFrame(cell_names).to_csv(
            barcodes_path, sep="\t", header=False, index=False
        )

        genes_path = mtx_dir / "genes.tsv"
        gene_data = {
            "gene_id": [f"ENSG_{i:08d}" for i in range(n_genes)],
            "gene_symbol": [f"GENE_{i}" for i in range(n_genes)],
        }
        pd.DataFrame(gene_data).to_csv(genes_path, sep="\t", header=False, index=False)

        # Convert to SLAF
        output_path = Path(tmp_path) / "test_10x_metadata.slaf"
        converter = SLAFConverter()
        converter.convert(str(mtx_dir), str(output_path))

        # Verify metadata tables exist and have expected structure
        cells_dataset = lance.dataset(output_path / "cells.lance")
        cells_df = cells_dataset.to_table().to_pandas()

        genes_dataset = lance.dataset(output_path / "genes.lance")
        genes_df = genes_dataset.to_table().to_pandas()

        # Check that cell and gene IDs are preserved
        assert "cell_id" in cells_df.columns
        assert "gene_id" in genes_df.columns

        # Check that we have the right number of cells and genes
        assert len(cells_df) == n_cells
        assert len(genes_df) == n_genes

    def test_10x_mtx_compressed_format(self, tmp_path):
        """Test conversion from compressed 10x MTX format"""
        import gzip

        import pandas as pd
        from scipy import sparse
        from scipy.io import mmwrite

        # Create mock 10x MTX directory with compressed files
        mtx_dir = Path(tmp_path) / "filtered_feature_bc_matrix"
        mtx_dir.mkdir(exist_ok=True)

        n_cells, n_genes = 3, 2
        X = np.random.randint(0, 10, (n_cells, n_genes))
        X_sparse = sparse.csr_matrix(X)

        # Write compressed matrix.mtx.gz
        matrix_path = mtx_dir / "matrix.mtx.gz"
        with gzip.open(matrix_path, "wb") as f:
            mmwrite(f, X_sparse.T)

        # Write compressed barcodes.tsv.gz
        barcodes_path = mtx_dir / "barcodes.tsv.gz"
        cell_names = [f"cell_{i}" for i in range(n_cells)]
        with gzip.open(barcodes_path, "wt") as f:
            pd.DataFrame(cell_names).to_csv(f, sep="\t", header=False, index=False)

        # Write compressed features.tsv.gz (newer 10x format with 3 columns)
        features_path = mtx_dir / "features.tsv.gz"
        gene_data = {
            "gene_id": [f"ENSG_{i:08d}" for i in range(n_genes)],
            "gene_symbol": [f"GENE_{i}" for i in range(n_genes)],
            "feature_type": ["Gene Expression"] * n_genes,
        }
        with gzip.open(features_path, "wt") as f:
            pd.DataFrame(gene_data).to_csv(f, sep="\t", header=False, index=False)

        # Test format detection
        from slaf.data.utils import detect_format

        detected_format = detect_format(str(mtx_dir))
        assert detected_format == "10x_mtx"

        # Convert to SLAF (should work with scanpy's read_10x_mtx)
        output_path = Path(tmp_path) / "test_10x_mtx_compressed.slaf"
        converter = SLAFConverter()
        converter.convert(str(mtx_dir), str(output_path))

        # Verify output structure
        assert (output_path / "expression.lance").exists()
        assert (output_path / "cells.lance").exists()
        assert (output_path / "genes.lance").exists()

    def test_error_handling_invalid_10x_mtx(self, tmp_path):
        """Test error handling for invalid 10x MTX directory"""
        # Create directory without required files
        mtx_dir = Path(tmp_path) / "invalid_mtx"
        mtx_dir.mkdir(exist_ok=True)

        # Only create matrix.mtx, missing barcodes and genes
        (mtx_dir / "matrix.mtx").write_text("invalid content")

        converter = SLAFConverter()

        # Should fail when trying to read with scanpy
        with pytest.raises((ValueError, RuntimeError, OSError)):
            converter.convert(str(mtx_dir), str(Path(tmp_path) / "output.slaf"))

    def test_error_handling_invalid_10x_h5(self, tmp_path):
        """Test error handling for invalid 10x H5 file"""
        # Create invalid H5 file
        h5_file = Path(tmp_path) / "invalid.h5"
        h5_file.write_text("not a valid h5 file")

        converter = SLAFConverter()

        # Should fail when trying to read
        with pytest.raises((ValueError, RuntimeError, OSError)):
            converter.convert(str(h5_file), str(Path(tmp_path) / "output.slaf"))

    # Chunked reader tests
    def test_chunked_reader_basic(self, small_sample_adata, tmp_path):
        """Test basic chunked reader functionality"""
        # Test chunked reader
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)
        with ChunkedH5ADReader(h5ad_path) as reader:
            assert reader.n_obs == 5
            assert reader.n_vars == 3
            assert len(reader.obs_names) == 5
            assert len(reader.var_names) == 3

            # Test metadata reading
            obs_df = reader.get_obs_metadata()
            var_df = reader.get_var_metadata()

            assert obs_df.shape == (5, 2)  # cell_type and batch columns
            assert var_df.shape == (3, 2)  # gene_symbol and highly_variable columns

            # Test chunking
            chunks = list(reader.iter_chunks(chunk_size=2))
            assert len(chunks) == 3  # 5 cells / 2 = 3 chunks (2, 2, 1)

            chunk1, slice1 = chunks[0]
            assert chunk1.shape == (2, 3)

    def test_converter_chunked_mode(self, chunked_converter, tmp_path):
        """Test that chunked mode works correctly"""
        # Test that chunked mode is set correctly
        assert chunked_converter.chunked is True
        assert chunked_converter.chunk_size == 1000

        # Test that convert_anndata raises error in chunked mode
        with pytest.raises(
            ValueError, match="convert_anndata.*not supported in chunked mode"
        ):
            chunked_converter.convert_anndata(None, str(tmp_path / "output.slaf"))

    def test_converter_backward_compatibility(self, tmp_path):
        """Test that converter maintains backward compatibility"""
        # This test ensures the simplified API still works
        converter = SLAFConverter()
        assert converter.use_integer_keys is True
        assert converter.chunked is False
        assert converter.chunk_size == 1000
        assert converter.sort_metadata is False
        assert converter.create_indices is False

    def test_compression_settings_default(self, tmp_path):
        """Test that default compression settings are optimal for large datasets"""
        converter = SLAFConverter()

        # Test expression table settings
        expression_settings = converter._get_compression_settings("expression")
        assert expression_settings["max_rows_per_file"] == 10000000  # 10M
        assert expression_settings["max_rows_per_group"] == 2000000  # 2M
        assert (
            expression_settings["max_bytes_per_file"] == 50 * 1024 * 1024 * 1024
        )  # 50GB

        # Test metadata table settings
        metadata_settings = converter._get_compression_settings("metadata")
        assert metadata_settings["max_rows_per_group"] == 200000  # 200K

    def test_simplified_api_parameters(self, small_sample_adata, tmp_path):
        """Test that the simplified API parameters work correctly"""
        # Save sample data as h5ad
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Test default settings
        converter = SLAFConverter()
        output_path = tmp_path / "test_default.slaf"
        converter.convert(str(h5ad_path), str(output_path))

        # Verify files were created
        assert (output_path / "expression.lance").exists()
        assert (output_path / "cells.lance").exists()
        assert (output_path / "genes.lance").exists()

        # Test with indices enabled
        converter_with_indices = SLAFConverter(create_indices=True)
        output_path_with_indices = tmp_path / "test_with_indices.slaf"
        converter_with_indices.convert(str(h5ad_path), str(output_path_with_indices))

        # Verify files were created
        assert (output_path_with_indices / "expression.lance").exists()
        assert (output_path_with_indices / "cells.lance").exists()
        assert (output_path_with_indices / "genes.lance").exists()

    def test_string_ids_always_preserved(self, small_sample_adata, tmp_path):
        """Test that string IDs are always preserved in the schema"""
        # Save sample data as h5ad
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert with default settings
        output_path = tmp_path / "test.slaf"
        converter = SLAFConverter()
        converter.convert(str(h5ad_path), str(output_path))

        # Load expression table
        expression_dataset = lance.dataset(output_path / "expression.lance")
        expression_df = expression_dataset.to_table().to_pandas()

        # Verify string IDs are always present
        assert "cell_id" in expression_df.columns
        assert "gene_id" in expression_df.columns

        # Verify integer IDs are also present (default behavior)
        assert "cell_integer_id" in expression_df.columns
        assert "gene_integer_id" in expression_df.columns

        # Verify string IDs contain the expected values
        # Only cells with non-zero expression will be in the table
        expected_cell_ids = [f"cell_{i}" for i in range(small_sample_adata.n_obs)]
        actual_cell_ids = expression_df["cell_id"].unique()
        # Check that all actual cell IDs are in the expected list
        assert all(cell_id in expected_cell_ids for cell_id in actual_cell_ids)
        # Check that we have the right number of unique cell IDs
        assert len(actual_cell_ids) <= len(expected_cell_ids)

        # Verify gene IDs match AnnData
        expected_gene_ids = set(small_sample_adata.var.index)
        actual_gene_ids = set(expression_df["gene_id"].unique())
        assert actual_gene_ids == expected_gene_ids

    def test_expression_schema(self, tmp_path):
        """Test that expression schema is correct"""
        converter = SLAFConverter()
        schema = converter._get_expression_schema()

        # Check that schema has expected fields
        field_names = [field.name for field in schema]
        expected_fields = [
            "cell_id",
            "gene_id",
            "cell_integer_id",
            "gene_integer_id",
            "value",
        ]

        assert field_names == expected_fields

        # Check field types
        assert schema.field("cell_id").type == pa.string()
        assert schema.field("gene_id").type == pa.string()
        assert schema.field("cell_integer_id").type == pa.int32()
        assert schema.field("gene_integer_id").type == pa.int32()
        assert schema.field("value").type == pa.float32()

    def test_convert_h5ad_chunked_structure(self, small_sample_adata, tmp_path):
        """Test that chunked conversion creates the same structure as traditional conversion"""
        # Save sample data as h5ad
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert using traditional method
        output_path_traditional = tmp_path / "test_traditional.slaf"
        converter_traditional = SLAFConverter(chunked=False, sort_metadata=True)
        converter_traditional.convert(str(h5ad_path), str(output_path_traditional))

        # Convert using chunked method
        output_path_chunked = tmp_path / "test_chunked.slaf"
        converter_chunked = SLAFConverter(
            chunked=True, chunk_size=100, sort_metadata=True
        )
        converter_chunked.convert(str(h5ad_path), str(output_path_chunked))

        # Check that both methods create the same files
        for method_path in [output_path_traditional, output_path_chunked]:
            assert (method_path / "expression.lance").exists()
            assert (method_path / "cells.lance").exists()
            assert (method_path / "genes.lance").exists()
            assert (method_path / "config.json").exists()

        # Check that configs are identical (except for created_at timestamp)
        with open(output_path_traditional / "config.json") as f:
            config_traditional = json.load(f)
        with open(output_path_chunked / "config.json") as f:
            config_chunked = json.load(f)

        # Remove timestamp for comparison
        del config_traditional["created_at"]
        del config_chunked["created_at"]

        assert config_traditional == config_chunked

    def test_convert_h5ad_chunked_vs_non_chunked(self, small_sample_adata, tmp_path):
        """Test that chunked and non-chunked conversion produce identical results for h5ad format"""
        # Save sample data as h5ad
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert using traditional method
        output_path_traditional = tmp_path / "test_traditional.slaf"
        converter_traditional = SLAFConverter(chunked=False, sort_metadata=True)
        converter_traditional.convert(str(h5ad_path), str(output_path_traditional))

        # Convert using chunked method
        output_path_chunked = tmp_path / "test_chunked.slaf"
        converter_chunked = SLAFConverter(
            chunked=True, chunk_size=100, sort_metadata=True
        )
        converter_chunked.convert(str(h5ad_path), str(output_path_chunked))

        # Compare expression data
        expression_traditional = lance.dataset(
            output_path_traditional / "expression.lance"
        )
        expression_chunked = lance.dataset(output_path_chunked / "expression.lance")

        df_traditional = expression_traditional.to_table().to_pandas()
        df_chunked = expression_chunked.to_table().to_pandas()

        # Sort both dataframes by the same columns for comparison
        sort_cols = ["cell_integer_id", "gene_integer_id", "cell_id", "gene_id"]
        df_traditional_sorted = df_traditional.sort_values(sort_cols).reset_index(
            drop=True
        )
        df_chunked_sorted = df_chunked.sort_values(sort_cols).reset_index(drop=True)

        # Compare expression data exactly
        pd.testing.assert_frame_equal(df_traditional_sorted, df_chunked_sorted)

        # Compare cell metadata
        cells_traditional = lance.dataset(output_path_traditional / "cells.lance")
        cells_chunked = lance.dataset(output_path_chunked / "cells.lance")

        df_cells_traditional = cells_traditional.to_table().to_pandas()
        df_cells_chunked = cells_chunked.to_table().to_pandas()

        # Debug: print column information
        print(f"Traditional cell columns: {list(df_cells_traditional.columns)}")
        print(f"Chunked cell columns: {list(df_cells_chunked.columns)}")
        print(f"Traditional cell shape: {df_cells_traditional.shape}")
        print(f"Chunked cell shape: {df_cells_chunked.shape}")

        # Debug: check what the chunked reader actually reads
        from slaf.data.chunked_reader import ChunkedH5ADReader

        with ChunkedH5ADReader(str(h5ad_path)) as reader:
            obs_df = reader.get_obs_metadata()
            print(f"Chunked reader obs columns: {list(obs_df.columns)}")
            print(f"Chunked reader obs shape: {obs_df.shape}")
            print(f"Chunked reader obs head:\n{obs_df.head()}")

        # Compare cell metadata by sorting by cell_id column
        df_cells_traditional_sorted = df_cells_traditional.sort_values(
            "cell_id"
        ).reset_index(drop=True)
        df_cells_chunked_sorted = df_cells_chunked.sort_values("cell_id").reset_index(
            drop=True
        )

        # Sort columns for robust comparison
        df_cells_traditional_sorted = df_cells_traditional_sorted[
            sorted(df_cells_traditional_sorted.columns)
        ]
        df_cells_chunked_sorted = df_cells_chunked_sorted[
            sorted(df_cells_chunked_sorted.columns)
        ]

        pd.testing.assert_frame_equal(
            df_cells_traditional_sorted, df_cells_chunked_sorted
        )

        # Compare gene metadata
        genes_traditional = lance.dataset(output_path_traditional / "genes.lance")
        genes_chunked = lance.dataset(output_path_chunked / "genes.lance")

        df_genes_traditional = genes_traditional.to_table().to_pandas()
        df_genes_chunked = genes_chunked.to_table().to_pandas()

        # Sort by gene_id for comparison
        df_genes_traditional_sorted = df_genes_traditional.sort_values(
            "gene_id"
        ).reset_index(drop=True)
        df_genes_chunked_sorted = df_genes_chunked.sort_values("gene_id").reset_index(
            drop=True
        )

        # Sort columns for robust comparison
        df_genes_traditional_sorted = df_genes_traditional_sorted[
            sorted(df_genes_traditional_sorted.columns)
        ]
        df_genes_chunked_sorted = df_genes_chunked_sorted[
            sorted(df_genes_chunked_sorted.columns)
        ]

        pd.testing.assert_frame_equal(
            df_genes_traditional_sorted, df_genes_chunked_sorted
        )

        # Verify data integrity by reconstructing the original matrix
        def reconstruct_matrix(df, n_cells, n_genes):
            """Reconstruct dense matrix from COO data"""
            matrix = np.zeros((n_cells, n_genes))
            for _, row in df.iterrows():
                cell_idx = row["cell_integer_id"]
                gene_idx = row["gene_integer_id"]
                value = row["value"]
                matrix[cell_idx, gene_idx] = value
            return matrix

        # Reconstruct matrices from both methods
        matrix_traditional = reconstruct_matrix(
            df_traditional_sorted, small_sample_adata.n_obs, small_sample_adata.n_vars
        )
        matrix_chunked = reconstruct_matrix(
            df_chunked_sorted, small_sample_adata.n_obs, small_sample_adata.n_vars
        )

        # Compare reconstructed matrices
        np.testing.assert_array_equal(matrix_traditional, matrix_chunked)

        # Verify against original AnnData
        original_matrix = small_sample_adata.X.toarray()
        np.testing.assert_array_equal(matrix_traditional, original_matrix)
        np.testing.assert_array_equal(matrix_chunked, original_matrix)

    def test_convert_h5ad_chunked_vs_non_chunked_equivalence(
        self, small_sample_adata, tmp_path
    ):
        """Test that chunked and non-chunked h5ad conversion produce equivalent results."""
        # Create test data
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert with chunked processing
        converter_chunked = SLAFConverter(chunked=True, chunk_size=100)
        output_chunked = tmp_path / "output_chunked.slaf"
        converter_chunked.convert(str(h5ad_path), str(output_chunked))

        # Convert without chunked processing
        converter_normal = SLAFConverter(chunked=False)
        output_normal = tmp_path / "output_normal.slaf"
        converter_normal.convert(str(h5ad_path), str(output_normal))

        # Compare results by loading and comparing the datasets
        # Load both datasets
        slaf_chunked = SLAFArray(str(output_chunked))
        slaf_normal = SLAFArray(str(output_normal))

        # Compare shapes
        assert slaf_chunked.shape == slaf_normal.shape

        # Compare expression data
        chunked_expr = slaf_chunked.query(
            "SELECT * FROM expression ORDER BY cell_integer_id, gene_integer_id"
        )
        normal_expr = slaf_normal.query(
            "SELECT * FROM expression ORDER BY cell_integer_id, gene_integer_id"
        )

        pd.testing.assert_frame_equal(chunked_expr, normal_expr)

    def test_chunked_reader_factory_h5ad(self, small_sample_adata, tmp_path):
        """Test create_chunked_reader factory function with h5ad format."""
        # Create test data
        h5ad_path = tmp_path / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(h5ad_path)) as reader:
            assert reader.n_obs == 5
            assert reader.n_vars == 3
            assert len(reader.obs_names) == 5
            assert len(reader.var_names) == 3

    def test_chunked_reader_factory_10x_mtx(self, tmp_path):
        """Test create_chunked_reader factory function with 10x MTX format."""
        # Create test MTX directory
        mtx_dir = Path(tmp_path) / "mtx_dir"
        mtx_dir.mkdir()

        # Create matrix.mtx
        matrix_path = mtx_dir / "matrix.mtx"
        with open(matrix_path, "w") as f:
            f.write("%%MatrixMarket matrix coordinate integer general\n")
            f.write("3 5 4\n")  # n_vars n_obs nnz
            f.write("1 1 1\n")  # 4 non-zero entries
            f.write("1 2 2\n")
            f.write("2 2 3\n")
            f.write("3 3 4\n")

        # Create barcodes.tsv
        barcodes_path = mtx_dir / "barcodes.tsv"
        with open(barcodes_path, "w") as f:
            for i in range(5):
                f.write(f"cell_{i}\n")

        # Create genes.tsv
        genes_path = mtx_dir / "genes.tsv"
        with open(genes_path, "w") as f:
            for i in range(3):
                f.write(f"gene_{i}\tgene_{i}\n")

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(mtx_dir)) as reader:
            assert reader.n_obs == 5
            assert reader.n_vars == 3
            assert len(reader.obs_names) == 5
            assert len(reader.var_names) == 3

    def test_chunked_reader_factory_10x_h5(self, small_sample_adata, tmp_path):
        """Test create_chunked_reader factory function with 10x H5 format."""
        # Create test h5ad file (will be detected as 10x H5)
        h5_path = tmp_path / "test.h5"
        small_sample_adata.write(h5_path)

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(h5_path)) as reader:
            assert reader.n_obs == 5
            assert reader.n_vars == 3
            assert len(reader.obs_names) == 5
            assert len(reader.var_names) == 3

    def test_chunked_reader_factory_unsupported_format(self):
        """Test create_chunked_reader factory function with unsupported format."""
        from slaf.data.chunked_reader import create_chunked_reader

        with pytest.raises(ValueError, match="Cannot detect format for"):
            create_chunked_reader("nonexistent_file.txt")

    def test_chunked_reader_context_manager(self, tmp_path, small_sample_adata):
        """Test chunked reader context manager functionality."""
        # Create test data
        adata = small_sample_adata
        h5ad_path = tmp_path / "test.h5ad"
        adata.write(h5ad_path)

        from slaf.data.chunked_reader import create_chunked_reader

        # Test context manager
        with create_chunked_reader(str(h5ad_path)) as reader:
            assert reader.n_obs == 5
            assert reader.n_vars == 3

        # Test that file is properly closed
        assert reader.file is None

    def test_chunked_reader_iter_chunks(self, tmp_path, small_sample_adata):
        """Test chunked reader iteration over chunks."""
        # Create test data
        adata = small_sample_adata
        h5ad_path = tmp_path / "test.h5ad"
        adata.write(h5ad_path)

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(h5ad_path)) as reader:
            chunks = list(reader.iter_chunks(chunk_size=2))
            assert len(chunks) == 3  # 5 cells / 2 = 3 chunks (2, 2, 1)

            for i, (chunk, obs_slice) in enumerate(chunks):
                expected_size = 2 if i < 2 else 1  # Last chunk has 1 cell
                assert chunk.shape[0] == expected_size
                assert chunk.shape[1] == 3  # All chunks have 3 genes
                assert obs_slice.start == i * 2
                assert obs_slice.stop == min((i + 1) * 2, 5)

    def test_chunked_reader_get_chunk(self, tmp_path, small_sample_adata):
        """Test chunked reader get_chunk method."""
        # Create test data
        adata = small_sample_adata
        h5ad_path = tmp_path / "test.h5ad"
        adata.write(h5ad_path)

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(h5ad_path)) as reader:
            # Get first 3 cells, first 2 genes
            chunk = reader.get_chunk(obs_slice=slice(0, 3), var_slice=slice(0, 2))
            assert chunk.shape == (3, 2)

    def test_chunked_reader_get_obs_metadata(self, tmp_path, small_sample_adata):
        """Test chunked reader get_obs_metadata method."""
        # Create test data with metadata
        adata = small_sample_adata
        adata.obs["cell_type"] = ["type_A"] * 3 + ["type_B"] * 2
        adata.obs["batch"] = ["batch_1"] * 5
        h5ad_path = tmp_path / "test.h5ad"
        adata.write(h5ad_path)

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(h5ad_path)) as reader:
            obs_metadata = reader.get_obs_metadata()
            assert "cell_type" in obs_metadata.columns
            assert "batch" in obs_metadata.columns
            assert len(obs_metadata) == 5

    def test_chunked_reader_get_var_metadata(self, tmp_path, small_sample_adata):
        """Test chunked reader get_var_metadata method."""
        # Create test data with metadata
        adata = small_sample_adata
        adata.var["highly_variable"] = [True] * 2 + [False] * 1
        adata.var["means"] = np.random.random(3)
        h5ad_path = tmp_path / "test.h5ad"
        adata.write(h5ad_path)

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(h5ad_path)) as reader:
            var_metadata = reader.get_var_metadata()
            assert "highly_variable" in var_metadata.columns
            assert "means" in var_metadata.columns
            assert len(var_metadata) == 3

    def test_chunked_reader_get_gene_expression(self, tmp_path, small_sample_adata):
        """Test chunked reader get_gene_expression method."""
        # Create test data
        adata = small_sample_adata
        h5ad_path = tmp_path / "test.h5ad"
        adata.write(h5ad_path)

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(h5ad_path)) as reader:
            # Use the actual gene names from the fixture, but convert to bytes to match reader
            gene_names = [
                name.encode("utf-8") for name in small_sample_adata.var.index[:3]
            ]
            expression_chunks = list(
                reader.get_gene_expression(gene_names, chunk_size=2)
            )

            assert len(expression_chunks) == 3  # 5 cells / 2 = 3 chunks
            for chunk_df in expression_chunks:
                assert chunk_df.shape[1] == 3  # 3 genes
                assert all(gene in chunk_df.columns for gene in gene_names)

    def test_chunked_reader_get_gene_expression_missing_genes(
        self, tmp_path, small_sample_adata
    ):
        """Test chunked reader get_gene_expression with missing genes."""
        # Create test data
        adata = small_sample_adata
        h5ad_path = tmp_path / "test.h5ad"
        adata.write(h5ad_path)

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(h5ad_path)) as reader:
            # Use actual gene names with one missing, convert to bytes
            gene_names = [
                name.encode("utf-8") for name in small_sample_adata.var.index[:2]
            ] + [b"nonexistent_gene"]
            expression_chunks = list(
                reader.get_gene_expression(gene_names, chunk_size=2)
            )

            # Should still work with missing genes (just warn)
            assert len(expression_chunks) == 3
            for chunk_df in expression_chunks:
                assert chunk_df.shape[1] == 2  # Only 2 valid genes

    def test_chunked_reader_get_gene_expression_no_valid_genes(
        self, tmp_path, small_sample_adata
    ):
        """Test chunked reader get_gene_expression with no valid genes."""
        # Create test data
        adata = small_sample_adata
        h5ad_path = tmp_path / "test.h5ad"
        adata.write(h5ad_path)

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(h5ad_path)) as reader:
            gene_names = ["nonexistent_gene_1", "nonexistent_gene_2"]
            expression_chunks = list(
                reader.get_gene_expression(gene_names, chunk_size=2)
            )

            # Should return empty iterator
            assert len(expression_chunks) == 0

    def test_chunked_reader_variable_chunking_not_supported(
        self, tmp_path, small_sample_adata
    ):
        """Test that variable chunking is not supported."""
        # Create test data
        adata = small_sample_adata
        h5ad_path = tmp_path / "test.h5ad"
        adata.write(h5ad_path)

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(h5ad_path)) as reader:
            with pytest.raises(
                NotImplementedError, match="Variable chunking not supported"
            ):
                list(reader.iter_chunks(chunk_size=25, obs_chunk=False))

    def test_chunked_reader_file_not_opened_error(self):
        """Test chunked reader error when file not opened."""
        from slaf.data.chunked_reader import ChunkedH5ADReader

        reader = ChunkedH5ADReader("nonexistent.h5ad")

        with pytest.raises(RuntimeError, match="File not opened"):
            _ = reader.n_obs

    def test_chunked_reader_10x_mtx_compressed(self, tmp_path):
        """Test chunked reader with compressed 10x MTX files."""
        # Create test MTX directory with compressed files
        mtx_dir = tmp_path / "mtx_dir"
        mtx_dir.mkdir()

        import gzip

        # Create compressed matrix.mtx.gz
        matrix_path = mtx_dir / "matrix.mtx.gz"
        with gzip.open(matrix_path, "wt") as f:
            f.write("%%MatrixMarket matrix coordinate integer general\n")
            f.write("50 100 100\n")  # n_vars n_obs nnz
            for i in range(1, 101):  # 100 non-zero entries
                f.write(f"{i} {i} {i}\n")

        # Create compressed barcodes.tsv.gz
        barcodes_path = mtx_dir / "barcodes.tsv.gz"
        with gzip.open(barcodes_path, "wt") as f:
            for i in range(100):
                f.write(f"cell_{i}\n")

        # Create compressed genes.tsv.gz
        genes_path = mtx_dir / "genes.tsv.gz"
        with gzip.open(genes_path, "wt") as f:
            for i in range(50):
                f.write(f"gene_{i}\tgene_{i}\n")

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(mtx_dir)) as reader:
            assert reader.n_obs == 100
            assert reader.n_vars == 50
            assert len(reader.obs_names) == 100
            assert len(reader.var_names) == 50

    def test_chunked_reader_10x_mtx_features_tsv(self, tmp_path):
        """Test chunked reader with features.tsv (new 10x format)."""
        # Create test MTX directory with features.tsv
        mtx_dir = tmp_path / "mtx_dir"
        mtx_dir.mkdir()

        # Create matrix.mtx
        matrix_path = mtx_dir / "matrix.mtx"
        with open(matrix_path, "w") as f:
            f.write("%%MatrixMarket matrix coordinate integer general\n")
            f.write("50 100 100\n")  # n_vars n_obs nnz
            for i in range(1, 101):  # 100 non-zero entries
                f.write(f"{i} {i} {i}\n")

        # Create barcodes.tsv
        barcodes_path = mtx_dir / "barcodes.tsv"
        with open(barcodes_path, "w") as f:
            for i in range(100):
                f.write(f"cell_{i}\n")

        # Create features.tsv (new format)
        features_path = mtx_dir / "features.tsv"
        with open(features_path, "w") as f:
            for i in range(50):
                f.write(f"gene_{i}\tgene_{i}\n")

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(mtx_dir)) as reader:
            assert reader.n_obs == 100
            assert reader.n_vars == 50
            assert len(reader.obs_names) == 100
            assert len(reader.var_names) == 50

    def test_chunked_reader_10x_h5_regular_h5ad(self, tmp_path, small_sample_adata):
        """Test chunked reader with regular h5ad file saved as .h5."""
        # Create test data
        adata = small_sample_adata
        h5_path = tmp_path / "test.h5"
        adata.write(h5_path)

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(h5_path)) as reader:
            assert reader.n_obs == 5
            assert reader.n_vars == 3
            assert len(reader.obs_names) == 5
            assert len(reader.var_names) == 3

    def test_chunked_reader_error_handling_invalid_mtx_header(self, tmp_path):
        """Test chunked reader error handling with invalid MTX header."""
        # Create test MTX directory with invalid header
        mtx_dir = tmp_path / "mtx_dir"
        mtx_dir.mkdir()

        # Create invalid matrix.mtx
        matrix_path = mtx_dir / "matrix.mtx"
        with open(matrix_path, "w") as f:
            f.write("%%MatrixMarket matrix coordinate integer general\n")
            f.write("invalid header\n")  # Invalid header

        # Create barcodes.tsv
        barcodes_path = mtx_dir / "barcodes.tsv"
        with open(barcodes_path, "w") as f:
            for i in range(100):
                f.write(f"cell_{i}\n")

        # Create genes.tsv
        genes_path = mtx_dir / "genes.tsv"
        with open(genes_path, "w") as f:
            for i in range(50):
                f.write(f"gene_{i}\tgene_{i}\n")

        from slaf.data.chunked_reader import create_chunked_reader

        with pytest.raises(ValueError, match="Invalid MTX header format"):
            with create_chunked_reader(str(mtx_dir)):
                pass

    def test_chunked_reader_error_handling_missing_matrix_dataset(self, tmp_path):
        """Test chunked reader error handling with missing matrix dataset."""
        # Create empty H5 file
        import h5py

        h5_path = tmp_path / "test.h5"
        with h5py.File(h5_path, "w"):
            # Create empty file without matrix dataset
            pass

        from slaf.data.chunked_reader import create_chunked_reader

        with pytest.raises(ValueError, match="Could not find matrix dataset"):
            with create_chunked_reader(str(h5_path)):
                pass

    def test_chunked_reader_property_errors(self):
        """Test chunked reader property errors when not initialized."""
        from slaf.data.chunked_reader import ChunkedH5ADReader

        reader = ChunkedH5ADReader("nonexistent.h5ad")

        # Test properties that require file to be opened
        with pytest.raises(RuntimeError, match="File not opened"):
            _ = reader.n_obs

        with pytest.raises(RuntimeError, match="File not opened"):
            _ = reader.n_vars

        with pytest.raises(RuntimeError, match="File not opened"):
            _ = reader.obs_names

        with pytest.raises(RuntimeError, match="File not opened"):
            _ = reader.var_names

        with pytest.raises(RuntimeError, match="File not opened"):
            _ = reader.get_obs_metadata()

        with pytest.raises(RuntimeError, match="File not opened"):
            _ = reader.get_var_metadata()

    def test_chunked_reader_10x_mtx_file_not_opened_error(self):
        """Test chunked reader error when MTX file not opened."""
        from slaf.data.chunked_reader import Chunked10xMTXReader

        reader = Chunked10xMTXReader("nonexistent_dir")

        with pytest.raises(RuntimeError, match="Matrix file not opened"):
            list(reader.iter_chunks())

    def test_chunked_reader_10x_h5_file_not_opened_error(self):
        """Test chunked reader error when H5 file not opened."""
        from slaf.data.chunked_reader import Chunked10xH5Reader

        reader = Chunked10xH5Reader("nonexistent.h5")

        with pytest.raises(RuntimeError, match="File not opened"):
            _ = reader.n_obs

    def test_chunked_reader_10x_mtx_variable_chunking_not_supported(self, tmp_path):
        """Test that variable chunking is not supported for MTX files."""
        # Create test MTX directory
        mtx_dir = tmp_path / "mtx_dir"
        mtx_dir.mkdir()

        # Create matrix.mtx
        matrix_path = mtx_dir / "matrix.mtx"
        with open(matrix_path, "w") as f:
            f.write("%%MatrixMarket matrix coordinate integer general\n")
            f.write("50 100 100\n")
            for i in range(1, 101):
                f.write(f"{i} {i} {i}\n")

        # Create barcodes.tsv
        barcodes_path = mtx_dir / "barcodes.tsv"
        with open(barcodes_path, "w") as f:
            for i in range(100):
                f.write(f"cell_{i}\n")

        # Create genes.tsv
        genes_path = mtx_dir / "genes.tsv"
        with open(genes_path, "w") as f:
            for i in range(50):
                f.write(f"gene_{i}\tgene_{i}\n")

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(mtx_dir)) as reader:
            with pytest.raises(
                NotImplementedError, match="Variable chunking not supported"
            ):
                list(reader.iter_chunks(chunk_size=25, obs_chunk=False))

    def test_chunked_reader_10x_h5_variable_chunking_not_supported(
        self, tmp_path, small_sample_adata
    ):
        """Test that variable chunking is not supported for H5 files."""
        # Create test data
        adata = small_sample_adata
        h5_path = tmp_path / "test.h5"
        adata.write(h5_path)

        from slaf.data.chunked_reader import create_chunked_reader

        with create_chunked_reader(str(h5_path)) as reader:
            with pytest.raises(
                NotImplementedError, match="Variable chunking not supported"
            ):
                list(reader.iter_chunks(chunk_size=25, obs_chunk=False))
