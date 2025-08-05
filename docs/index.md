# **SLAF (Sparse Lazy Array Format)**

<div class="hero-section">
  <div class="logo-container">
    <img src="assets/slaf-logo-light.svg" alt="SLAF Logo" class="light-logo"/>
    <img src="assets/slaf-logo-dark.svg" alt="SLAF Logo" class="dark-logo"/>
  </div>
  <div class="hero-text">
    <strong>SLAF</strong> is a high-performance format for single-cell data that combines the power of SQL with lazy evaluation, built on top of the <a href="https://lancedb.github.io/lance/">Lance</a> table format, <a href="https://duckdb.org/">DuckDB</a> and <a href="pola.rs">Polars</a>.
  </div>
</div>

<div class="grid cards" markdown>

- :fontawesome-solid-database: **OLAP-Powered SQL**
  Embedded, in-process OLAP engines for blazing-fast, pushdown SQL queries and advanced analytics.

- :fontawesome-solid-memory: **Memory Efficient**
  Lazy evaluation means you only load what you need, perfect for large-scale single-cell analysis.

- :fontawesome-solid-dna: **Scanpy Compatible**
  Drop-in replacement for AnnData workflows with familiar numpy-like slicing.

- :fontawesome-solid-cloud: **Concurrent, Cloud-Scale Access**
  Built for distributed teams and interactive exploration with high QPS, zero-copy storage.

- :fontawesome-solid-brain: **Foundation Model Ready**
  Designed for distributed ML training with SQL-level tokenization and pre-built dataloaders.

- :fontawesome-solid-chart-line: **Visualization Ready**
  Designed to keep your scGPT (or favorite) embeddings alongside your UMAPs and build interactive dashboards like cellxgene using vector search

</div>

See our detailed benchmarks for [bioinformaticians](benchmarks/bioinformatics_benchmarks.md) and [ML engineers](benchmarks/ml_benchmarks.md).

---

## **Why SLAF?**

!!! quote "_Single-cell datasets have scaled **2,000-fold** in less than a decade._"

    A typical study used to have 50k cells that could easily be copied from object store to a SSD across the network. It could then be read entirely into memory and processed with in-memory operations. At the 100M-cell scale: network, storage, and memory become bottlenecks.

!!! quote "_The analytic workload is stuck in in-memory single-node operations._"

    Traditional bioinformatics workflows comprised cell and gene filtering, count normalization, visualization via PCA or UMAP, interactive cell typing, and statistical analysis of differential expression. Today, we need to do all those things at **2000x the scale**.

!!! quote "_New fundamentally different **AI-native workflows** have arrived._"

    Unlike before, we want to:

    - Scale cell typing using nearest neighbor search on cell embeddings
    - Rank gene-gene relationships using nearest neighbor search on gene embeddings
    - Train transformer-like foundation models with efficient tokenization
    - Distribute workloads across nodes or GPUs by streaming random batches concurrently

_We need **cloud-native, zero-copy, query-in-place storage systems**, rather than maintaining multiple copies of massive datasets per embedding model, node or experiment while continuing to experience the endorphins of numpy-like sparse matrix slicing, and the scanpy pipelines we've built over the years._

---

## **Who is SLAF for?**

SLAF is designed for the modern single-cell ecosystem facing scale challenges:

<div class="grid" markdown>

<div markdown>

!!! tip "**Bioinformaticians**"

    Struggling with OOM errors and data transfer issues on 10M+ cell datasets. Can't do self-service analysis without infrastructure engineers. **SLAF eliminates the human bottleneck** with lazy evaluation.

!!! success "**Foundation Model Builders**"

    Need to maximize experiments per unit time and resource. Currently copying data per node on attached storage. **SLAF enables cloud-native streaming** to eliminate data duplication.

!!! warning "**Tech Leaders & Architects**"

    Managing storage/compute infrastructure for teams. 5 bioinformaticians × 500GB dataset = 2.5TB of duplicated data. **SLAF provides zero-copy, query-in-place storage**.

</div>

<div markdown>

!!! note "**Tool Builders**"

    Want to deliver better interactive experiences on massive datasets using commodity web services. **SLAF enables concurrent, cloud-scale access** with high QPS.

!!! info "**Atlas Builders**"

    Need to serve massive datasets to the research community. **SLAF provides cloud-native, zero-copy storage** for global distribution.

!!! example "**Data Integrators**"

    Harmonizing PB-scale datasets across atlases. **SLAF's SQL-native design** enables complex data integration with pushdown optimization.

</div>

</div>

---

## **Win with SLAF**

### Leverage pushdown filtering and SQL query optimization from cloud-native OLAP databases

<div class="grid" markdown>

<div markdown>

**Do this:**

```python
from slaf import SLAFArray

# Access your data directly from the cloud
slaf_array = SLAFArray("s3://bucket/large_dataset.slaf")

# Query just what you need with SQL
results = slaf_array.query("""
    SELECT
      cell_type,
      AVG(total_counts) as avg_counts
    FROM cells
    WHERE batch = "batch1"
      AND cell_type IN ("T cells", "B cells")
    GROUP BY cell_type
    ORDER BY avg_counts DESC
""")
```

</div>

<div markdown>

**Instead of:**

```python
import scanpy as sc

# Download large dataset from the cloud
!aws s3 cp s3://bucket/large_dataset.h5ad .

# Load entire dataset into memory
adata = sc.read_h5ad("large_dataset.h5ad")

# Filter in memory
subset = adata[adata.obs.batch == "batch1"]
subset = subset[subset.obs.cell_type.isin(["T cells", "B cells"])]

# Aggregate in memory
results = subset.obs.groupby("cell_type")["total_counts"].mean()
```

</div>

</div>

### Evaluate lazily with numpy-like slicing using familiar scanpy idioms

<div class="grid" markdown>

<div markdown>

**Do this:**

```python
from slaf.integrations import read_slaf

# Load as lazy AnnData
adata = read_slaf("s3://bucket/large_dataset.slaf")

# Operations are lazy until you call .compute()
subset = adata[adata.obs.cell_type == "T cells", :]
first_ten_cells = subset[:10, :]
expression = first_ten_cells.X.compute()  # Only now is data loaded
```

</div>

<div markdown>

**Instead of:**

```python
# Download large dataset from the cloud
!aws s3 cp s3://bucket/large_dataset.h5ad .

# Load entire dataset into memory
adata = sc.read_h5ad("large_dataset.h5ad")

# Filter in memory (expensive)
subset = adata[adata.obs.cell_type == "T cells", :]
first_ten_cells = subset[:10, :]
expression = subset.X  # Always loads full data
```

</div>

</div>

### Stream tokenized batches using pre-built dataloaders directly from the cloud to GPU

<div class="grid" markdown>

<div markdown>

**Do this:**

```python
from slaf import SLAFArray
from slaf.ml.dataloaders import SLAFDataLoader

# Access your data directly from the cloud
slaf_array = slaf.SLAFArray("s3://bucket/large_dataset.slaf")

# Create production-ready DataLoader
dataloader = SLAFDataLoader(
    slaf_array=slaf_array,
    tokenizer_type="geneformer",
    batch_size=32,
    max_genes=2048,
    vocab_size=50000
)

# Stream batches for training
for batch in dataloader:
    input_ids = batch["input_ids"]      # Already tokenized
    attention_mask = batch["attention_mask"]
    cell_ids = batch["cell_ids"]
    # Your training code here
```

</div>

<div markdown>

**Instead of:**

```python
import scanpy as sc
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Download large dataset from the cloud
!aws s3 cp s3://bucket/large_dataset.h5ad .

# Load entire dataset into memory
adata = sc.read_h5ad("large_dataset.h5ad")

# Build gene vocabulary (manual implementation)
gene_counts = adata.var.index.tolist()
vocab_size = 50000
gene_vocab = {gene: i + 4 for i, gene in enumerate(gene_counts[:vocab_size])}
special_tokens = {"PAD": 0, "CLS": 1, "SEP": 2, "UNK": 3}

# Custom tokenization function
def tokenize_geneformer(expression_row, max_genes=2048):
    # Rank genes by expression
    nonzero_indices = np.nonzero(expression_row)[0]
    if len(nonzero_indices) == 0:
        return [special_tokens["PAD"]] * max_genes

    # Sort by expression level
    sorted_indices = sorted(nonzero_indices,
                          key=lambda i: expression_row[i], reverse=True)

    # Convert to tokens
    tokens = []
    for gene_idx in sorted_indices[:max_genes]:
        gene_id = adata.var.index[gene_idx]
        token = gene_vocab.get(gene_id, special_tokens["UNK"])
        tokens.append(token)

    # Pad to max_genes
    while len(tokens) < max_genes:
        tokens.append(special_tokens["PAD"])

    return tokens[:max_genes]

# Create custom dataset class
class SingleCellDataset(Dataset):
    def __init__(self, adata, max_genes=2048):
        self.adata = adata
        self.max_genes = max_genes
        self.expression_matrix = adata.X.toarray()  # Convert to dense

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        # Tokenize on-the-fly (expensive)
        expression_row = self.expression_matrix[idx]
        tokens = tokenize_geneformer(expression_row, self.max_genes)

        # Create attention mask
        attention_mask = [1 if token != special_tokens["PAD"] else 0
                        for token in tokens]

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "cell_ids": torch.tensor([idx], dtype=torch.long)
        }

# Create dataset and dataloader
dataset = SingleCellDataset(adata)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for batch in dataloader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    cell_ids = batch["cell_ids"]
    # Your training code here
```

</div>

---
