# Migrating to SLAF

## Converting from AnnData

### Using the CLI

The easiest way to convert your data:

```bash
# Convert a single h5ad file
slaf convert data.h5ad output.slaf

# Convert with verbose output
slaf convert data.h5ad output.slaf --verbose
```

### Using the Python API

```python
import slaf
from slaf.data import SLAFConverter

# Basic conversion
converter = SLAFConverter()
converter.convert("data.h5ad", "output.slaf")

# Convert existing AnnData object
import scanpy as sc
adata = sc.read_h5ad("data.h5ad")
converter.convert_anndata(adata, "output.slaf")
```

## Current Support

**Currently Supported:**

- **AnnData** (.h5ad files) - Full support

**Coming Soon:**

- **10x Genomics** (Cell Ranger output)
- **Parquet** files (Tahoe-100M format)

We're actively working on expanding format support. If you need to convert from other formats, please [open an issue](https://github.com/your-repo/slaf/issues) to let us know what formats are most important to you.

## Next Steps

After converting your data:

1. **Explore** your data: `slaf info output.slaf`
2. **Query** your data: `slaf query output.slaf "SELECT * FROM expression LIMIT 10"`
3. **Use in Python**: `import slaf; data = slaf.load("output.slaf")`

See the [Getting Started](../getting-started/quickstart.md) guide for more examples.
