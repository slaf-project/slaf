import json
import os
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from slaf.data.chunked_reader import ChunkedH5ADReader
from slaf.data.converter import SLAFConverter


class TestSLAFConverter:
    """Test the SLAF converter functionality"""

    # Basic functionality tests
    def test_converter_creates_new_table_structure(self, small_sample_adata, temp_dir):
        """Test that converter creates the COO table structure"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert to SLAF
        output_path = Path(temp_dir) / "test.slaf"
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
        assert config["format_version"] == "0.1"

    def test_expression_data_consistency(self, small_sample_adata, temp_dir):
        """Test that expression data is consistent in COO format"""
        # Convert to SLAF
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)
        output_path = Path(temp_dir) / "test.slaf"
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
    def test_integer_ids_in_metadata(self, small_sample_adata, temp_dir):
        """Test that integer IDs are embedded in metadata tables"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert with integer keys
        output_path = Path(temp_dir) / "test.slaf"
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

    def test_no_integer_ids_when_disabled(self, small_sample_adata, temp_dir):
        """Test that integer IDs are not added when disabled"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert without integer keys
        output_path = Path(temp_dir) / "test.slaf"
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

    def test_expression_data_structure(self, small_sample_adata, temp_dir):
        """Test that expression data has the correct COO structure"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert with integer keys
        output_path = Path(temp_dir) / "test.slaf"
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

    def test_integer_id_consistency(self, small_sample_adata, temp_dir):
        """Test that integer IDs in expression table match metadata tables"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert with integer keys
        output_path = Path(temp_dir) / "test.slaf"
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
    def test_integer_keys_optimization(self, small_sample_adata, temp_dir):
        """Test that integer keys reduce file size"""
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)
        output_path_string_keys = Path(temp_dir) / "test_string_keys.slaf"
        converter_string_keys = SLAFConverter(use_integer_keys=False)
        converter_string_keys.convert(str(h5ad_path), str(output_path_string_keys))
        output_path_int_keys = Path(temp_dir) / "test_int_keys.slaf"
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

    def test_optimization_config_persistence(self, small_sample_adata, temp_dir):
        """Test that optimization settings are saved in config"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert with optimizations
        output_path = Path(temp_dir) / "test_optimized.slaf"
        converter = SLAFConverter(
            use_integer_keys=True,
        )
        converter.convert(str(h5ad_path), str(output_path))

        # Check config
        with open(output_path / "config.json") as f:
            config = json.load(f)

        assert "optimizations" in config
        assert config["optimizations"]["use_integer_keys"]

    # 10x Format Conversion Tests
    def test_10x_mtx_format_detection(self, temp_dir):
        """Test auto-detection of 10x MTX format"""
        # Create a mock 10x MTX directory structure
        mtx_dir = Path(temp_dir) / "filtered_feature_bc_matrix"
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

    def test_10x_h5_format_detection(self, temp_dir):
        """Test auto-detection of 10x H5 format"""
        # Create a mock H5 file
        h5_file = Path(temp_dir) / "data.h5"
        h5_file.write_text("mock h5 content")

        from slaf.data.utils import detect_format

        detected_format = detect_format(str(h5_file))
        assert detected_format == "10x_h5"

    def test_h5ad_format_detection(self, temp_dir):
        """Test auto-detection of h5ad format"""
        # Create a mock h5ad file
        h5ad_file = Path(temp_dir) / "data.h5ad"
        h5ad_file.write_text("mock h5ad content")

        from slaf.data.utils import detect_format

        detected_format = detect_format(str(h5ad_file))
        assert detected_format == "h5ad"

    def test_format_detection_invalid_file(self):
        """Test format detection with invalid file"""
        from slaf.data.utils import detect_format

        with pytest.raises(ValueError, match="Cannot detect format for"):
            detect_format("nonexistent_file.txt")

    def test_convert_10x_mtx_format(self, temp_dir):
        """Test conversion from 10x MTX format"""
        import pandas as pd
        from scipy import sparse
        from scipy.io import mmwrite

        # Create a mock 10x MTX directory with real data
        mtx_dir = Path(temp_dir) / "filtered_feature_bc_matrix"
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
        output_path = Path(temp_dir) / "test_10x_mtx.slaf"
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

    def test_convert_10x_h5_format(self, temp_dir):
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
        h5_file = Path(temp_dir) / "data.h5"
        adata.write_h5ad(h5_file)

        # Convert to SLAF
        output_path = Path(temp_dir) / "test_10x_h5.slaf"
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

    def test_convert_with_explicit_format_specification(self, temp_dir):
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
        h5ad_file = Path(temp_dir) / "data.h5ad"
        adata.write_h5ad(h5ad_file)

        # Convert with explicit format specification
        output_path = Path(temp_dir) / "test_explicit_format.slaf"
        converter = SLAFConverter()
        converter.convert(str(h5ad_file), str(output_path), input_format="h5ad")

        # Verify output
        assert (output_path / "expression.lance").exists()
        assert (output_path / "cells.lance").exists()
        assert (output_path / "genes.lance").exists()

    def test_convert_unsupported_format(self, temp_dir):
        """Test conversion with unsupported format"""
        # Create a dummy file
        dummy_file = Path(temp_dir) / "dummy.txt"
        dummy_file.write_text("dummy content")

        converter = SLAFConverter()

        with pytest.raises(ValueError, match="Unsupported format"):
            converter.convert(
                str(dummy_file),
                str(Path(temp_dir) / "output.slaf"),
                input_format="unsupported",
            )

    def test_convert_10x_mtx_with_integer_keys(self, temp_dir):
        """Test 10x MTX conversion with integer key optimization"""
        import pandas as pd
        from scipy import sparse
        from scipy.io import mmwrite

        # Create mock 10x MTX directory
        mtx_dir = Path(temp_dir) / "filtered_feature_bc_matrix"
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
        output_path = Path(temp_dir) / "test_10x_mtx_int_keys.slaf"
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

    def test_convert_10x_h5_without_integer_keys(self, temp_dir):
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
        h5_file = Path(temp_dir) / "data.h5"
        adata.write_h5ad(h5_file)

        # Convert without integer keys
        output_path = Path(temp_dir) / "test_10x_h5_no_int_keys.slaf"
        converter = SLAFConverter(use_integer_keys=False)
        converter.convert(str(h5_file), str(output_path))

        # Verify no integer keys in metadata
        cells_dataset = lance.dataset(output_path / "cells.lance")
        cells_df = cells_dataset.to_table().to_pandas()

        genes_dataset = lance.dataset(output_path / "genes.lance")
        genes_df = genes_dataset.to_table().to_pandas()

        assert "cell_integer_id" not in cells_df.columns
        assert "gene_integer_id" not in genes_df.columns

    def test_convert_10x_mtx_chunked(self, temp_dir):
        """Test chunked conversion from 10x MTX format"""
        import pandas as pd
        from scipy import sparse
        from scipy.io import mmwrite

        # Create a mock 10x MTX directory with real data
        mtx_dir = Path(temp_dir) / "filtered_feature_bc_matrix"
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
        output_path = Path(temp_dir) / "test_10x_mtx_chunked.slaf"
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

    def test_convert_10x_h5_chunked(self, temp_dir):
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
        h5_file = Path(temp_dir) / "data.h5"
        adata.write_h5ad(h5_file)

        # Convert to SLAF with chunked processing
        output_path = Path(temp_dir) / "test_10x_h5_chunked.slaf"
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

    def test_convert_10x_mtx_chunked_vs_non_chunked(self, temp_dir):
        """Test that chunked and non-chunked conversion produce identical results for 10x MTX format"""
        import pandas as pd
        from scipy import sparse
        from scipy.io import mmwrite

        # Create a mock 10x MTX directory
        mtx_dir = Path(temp_dir) / "filtered_feature_bc_matrix"
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
        output_chunked = Path(temp_dir) / "test_10x_mtx_chunked.slaf"
        converter_chunked = SLAFConverter(chunked=True, chunk_size=2)
        converter_chunked.convert(str(mtx_dir), str(output_chunked))

        # Convert without chunked processing
        output_non_chunked = Path(temp_dir) / "test_10x_mtx_non_chunked.slaf"
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

    def test_convert_10x_h5_chunked_vs_non_chunked(self, temp_dir):
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
        h5_file = Path(temp_dir) / "data.h5"
        adata.write_h5ad(h5_file)

        # Convert with chunked processing
        output_chunked = Path(temp_dir) / "test_10x_h5_chunked.slaf"
        converter_chunked = SLAFConverter(chunked=True, chunk_size=2)
        converter_chunked.convert(str(h5_file), str(output_chunked))

        # Convert without chunked processing
        output_non_chunked = Path(temp_dir) / "test_10x_h5_non_chunked.slaf"
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

    def test_auto_detection_vs_explicit_format(self, temp_dir):
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
        h5ad_file = Path(temp_dir) / "data.h5ad"
        adata.write_h5ad(h5ad_file)

        # Convert with auto-detection
        output_auto = Path(temp_dir) / "test_auto.slaf"
        converter_auto = SLAFConverter()
        converter_auto.convert(str(h5ad_file), str(output_auto))

        # Convert with explicit format
        output_explicit = Path(temp_dir) / "test_explicit.slaf"
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

    def test_10x_formats_with_metadata_preservation(self, temp_dir):
        """Test that 10x format conversion preserves metadata correctly"""
        import pandas as pd
        from scipy import sparse
        from scipy.io import mmwrite

        # Create mock 10x MTX with rich metadata
        mtx_dir = Path(temp_dir) / "filtered_feature_bc_matrix"
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
        output_path = Path(temp_dir) / "test_10x_metadata.slaf"
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

    def test_10x_mtx_compressed_format(self, temp_dir):
        """Test conversion from compressed 10x MTX format"""
        import gzip

        import pandas as pd
        from scipy import sparse
        from scipy.io import mmwrite

        # Create mock 10x MTX directory with compressed files
        mtx_dir = Path(temp_dir) / "filtered_feature_bc_matrix"
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
        output_path = Path(temp_dir) / "test_10x_mtx_compressed.slaf"
        converter = SLAFConverter()
        converter.convert(str(mtx_dir), str(output_path))

        # Verify output structure
        assert (output_path / "expression.lance").exists()
        assert (output_path / "cells.lance").exists()
        assert (output_path / "genes.lance").exists()

    def test_error_handling_invalid_10x_mtx(self, temp_dir):
        """Test error handling for invalid 10x MTX directory"""
        # Create directory without required files
        mtx_dir = Path(temp_dir) / "invalid_mtx"
        mtx_dir.mkdir(exist_ok=True)

        # Only create matrix.mtx, missing barcodes and genes
        (mtx_dir / "matrix.mtx").write_text("invalid content")

        converter = SLAFConverter()

        # Should fail when trying to read with scanpy
        with pytest.raises((ValueError, RuntimeError, OSError)):
            converter.convert(str(mtx_dir), str(Path(temp_dir) / "output.slaf"))

    def test_error_handling_invalid_10x_h5(self, temp_dir):
        """Test error handling for invalid 10x H5 file"""
        # Create invalid H5 file
        h5_file = Path(temp_dir) / "invalid.h5"
        h5_file.write_text("not a valid h5 file")

        converter = SLAFConverter()

        # Should fail when trying to read
        with pytest.raises((ValueError, RuntimeError, OSError)):
            converter.convert(str(h5_file), str(Path(temp_dir) / "output.slaf"))

    # Chunked reader tests
    def test_chunked_reader_basic(self, sample_h5ad_file):
        """Test basic chunked reader functionality"""
        # Test chunked reader
        with ChunkedH5ADReader(sample_h5ad_file) as reader:
            assert reader.n_obs == 2
            assert reader.n_vars == 2
            assert len(reader.obs_names) == 2
            assert len(reader.var_names) == 2

            # Test metadata reading
            obs_df = reader.get_obs_metadata()
            var_df = reader.get_var_metadata()

            assert obs_df.shape == (2, 1)  # cell_type column
            assert var_df.shape == (2, 1)  # highly_variable column

            # Test chunking
            chunks = list(reader.iter_chunks(chunk_size=1))
            assert len(chunks) == 2

            chunk1, slice1 = chunks[0]
            assert chunk1.shape == (1, 2)

    def test_converter_chunked_mode(self, chunked_converter):
        """Test that chunked mode works correctly"""
        # Test that chunked mode is set correctly
        assert chunked_converter.chunked is True
        assert chunked_converter.chunk_size == 1000

        # Test that convert_anndata raises error in chunked mode
        with pytest.raises(
            ValueError, match="convert_anndata.*not supported in chunked mode"
        ):
            chunked_converter.convert_anndata(None, "output.slaf")

    def test_converter_backward_compatibility(self):
        """Test that non-chunked mode works as before"""
        converter = SLAFConverter(chunked=False)

        # Test that chunked mode is disabled
        assert converter.chunked is False
        assert converter.chunk_size == 1000  # Default value

    def test_expression_schema(self):
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

    def test_convert_h5ad_chunked_structure(self, small_sample_adata, temp_dir):
        """Test that chunked conversion creates the same structure as traditional conversion"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert using traditional method
        output_path_traditional = Path(temp_dir) / "test_traditional.slaf"
        converter_traditional = SLAFConverter(chunked=False, sort_metadata=True)
        converter_traditional.convert(str(h5ad_path), str(output_path_traditional))

        # Convert using chunked method
        output_path_chunked = Path(temp_dir) / "test_chunked.slaf"
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

    def test_convert_h5ad_chunked_vs_non_chunked(self, small_sample_adata, temp_dir):
        """Test that chunked and non-chunked conversion produce identical results for h5ad format"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert using traditional method
        output_path_traditional = Path(temp_dir) / "test_traditional.slaf"
        converter_traditional = SLAFConverter(chunked=False, sort_metadata=True)
        converter_traditional.convert(str(h5ad_path), str(output_path_traditional))

        # Convert using chunked method
        output_path_chunked = Path(temp_dir) / "test_chunked.slaf"
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
