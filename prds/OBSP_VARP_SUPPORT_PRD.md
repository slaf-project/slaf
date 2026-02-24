# obsp and varp Support - Product Requirements Document

## Overview

This PRD defines the implementation plan for adding **obsp** (pairwise observations) and **varp** (pairwise variables) support to SLAF, completing AnnData compatibility after PR #27 (obsm, varm, obs, var, layers, uns). In AnnData:

- **obsp**: Mutable mapping of string keys to array-like values. Each value is a **square matrix** of shape `(n_obs, n_obs)` (e.g. `connectivities`, `distances` from graph/neighbors).
- **varp**: Same for variables: square matrices of shape `(n_vars, n_vars)`.

Both axes are aligned to observations (cells) or variables (genes) respectively.

## Goals

1. **Storage**: Add two new Lance tables: `cellsxcells.lance` (obsp) and `genesxgenes.lance` (varp).
2. **Conversion**: Convert obsp/varp from h5ad during `SLAFConverter.convert()` and persist keys/dimensions in config.
3. **Lazy views**: Expose `LazyAnnData.obsp` and `LazyAnnData.varp` with dict-like access; each key returns a 2D numpy array (square matrix).
4. **I/O**: Read/write obsp/varp keys (create, mutate, delete) with config and table updates; support selectors so that sliced AnnData returns correctly sliced square matrices.
5. **Tests**: Unit and integration tests mirroring obsm/varm (conversion e2e, mutations, selectors, immutability).

## Out of Scope (for this PR)

- Backward compatibility with format versions before 0.4 (obsp/varp only exist when tables are present).

---

## 1. Table Schemas (COO / coordinate format)

Pairwise matrices are stored in **COO (coordinate)** form: one row per `(i, j)` entry, with one value column per key. This avoids storing huge dense rows and fits sparse graphs (e.g. k-NN connectivities) naturally.

### 1.1 `cellsxcells.lance` (obsp)

- **Purpose**: Store obsp matrices in COO form; one row per `(i, j)` pair that has at least one non-zero value across any key.
- **Schema**:
  - `cell_integer_id_i`: `uint32` (row index; same semantics as cell_integer_id in cells.lance).
  - `cell_integer_id_j`: `uint32` (column index).
  - One column per key (e.g. `connectivities`, `distances`): type `float32`. Value 0.0 means no edge / zero at that (i, j) for that key.
- **Semantics**: Only rows where at least one key column is non-zero need to be stored (sparse COO). To reconstruct matrix for key `k`: create dense `(n_cells, n_cells)` zeros, then fill from table rows using `(cell_integer_id_i, cell_integer_id_j, k)`. Dense matrices from h5ad can be converted by writing only non-zero (i, j) entries (sparse COO) or all (i, j) for dense COO.

### 1.2 `genesxgenes.lance` (varp)

- **Purpose**: Store varp matrices in COO form; one row per `(i, j)` pair with at least one non-zero across any key.
- **Schema**:
  - `gene_integer_id_i`: `uint16` (row index).
  - `gene_integer_id_j`: `uint16` (column index).
  - One column per key: type `float32`.

### 1.3 Design notes

- COO avoids large fixed-size list columns (n_cells or n_genes per row) and fits sparse pairwise data (e.g. neighbor graphs) naturally.
- Multiple keys share the same (i, j) rows: one table with `_i`, `_j`, and one value column per key. When adding a new key, new (i, j) rows may be added for entries that only appear in that key; existing rows get the new column (0.0 where not present).
- Tables are **optional**: only created when at least one obsp or varp key is converted or added.

---

## 2. Config.json

### 2.1 Tables

- If obsp is present: `config["tables"]["cellsxcells"] = "cellsxcells.lance"`.
- If varp is present: `config["tables"]["genesxgenes"] = "genesxgenes.lance"`.

### 2.2 obsp / varp sections

Mirror obsm/varm:

```json
"obsp": {
  "available": ["connectivities", "distances"],
  "immutable": ["connectivities", "distances"],
  "mutable": [],
  "dimensions": {
    "connectivities": 100,
    "distances": 100
  }
}
```

- **dimensions**: For square matrices, store the side length (e.g. `n_cells` for obsp, `n_genes` for varp). Used for validation and schema when adding columns.

---

## 3. Conversion (from h5ad)

### 3.1 When to create tables

- **cellsxcells.lance**: Created when `hasattr(adata, "obsp") and adata.obsp and len(adata.obsp) > 0`.
- **genesxgenes.lance**: Created when `hasattr(adata, "varp") and adata.varp and len(adata.varp) > 0`.

### 3.2 Conversion steps (per axis)

1. **obsp**
   - For each key in `adata.obsp`:
     - Validate: value is 2D, shape `(n_cells, n_cells)` (dense numpy or scipy sparse).
     - Convert to COO: if dense, iterate (i, j) and emit rows where value[i,j] != 0 (or emit all (i, j) for dense COO); if sparse, use `scipy.sparse` (row, col, data) and map to `cell_integer_id` via the same index→integer_id mapping as cells.lance.
     - Build/merge into one table: columns `cell_integer_id_i`, `cell_integer_id_j`, and one `float32` column per key. Union of (i, j) across keys; 0.0 for a key at (i, j) when that key has no entry there.
   - If no obsp keys: do not create `cellsxcells.lance`.
   - If any obsp keys: create `cellsxcells.lance` with the COO table; update config with `obsp.available`, `obsp.immutable`, `obsp.dimensions` (side length = n_cells).

2. **varp**
   - Same logic for `adata.varp`: shape `(n_genes, n_genes)`, COO with `gene_integer_id_i`, `gene_integer_id_j`, one float32 column per key; write to `genesxgenes.lance`, update config `varp.*` (dimensions = n_genes).

### 3.3 Converter API

- Add `_convert_obsp(obsp, output_path, cell_ids, cell_id_mapping) -> list[str]`.
- Add `_convert_varp(varp, output_path, gene_ids, gene_id_mapping) -> list[str]`.
- In `_convert_anndata()`: after varm, call `_convert_obsp` and `_convert_varp` if present; pass `obsp_keys` and `varp_keys` into `_save_config()`.
- In `_save_config()`: add `obsp_keys` and `varp_keys` arguments; when non-empty, set `config["tables"]["cellsxcells"]` / `config["tables"]["genesxgenes"]` and `config["obsp"]` / `config["varp"]` (available, immutable, dimensions).

### 3.4 Creating the table on first key

- **obsp**: If `cellsxcells.lance` does not exist, create it with schema `cell_integer_id_i`, `cell_integer_id_j`, and one float32 column per key (first key, then add_columns for additional keys). Rows = COO entries (union of (i, j) across keys).
- **varp**: Same for `genesxgenes.lance` with `gene_integer_id_i`, `gene_integer_id_j`, and one float32 column per key.

---

## 4. Lazy Views (anndata.py)

### 4.1 New view classes

- **LazyObspView**: dict-like view over obsp; table_name = `"cellsxcells"`, table_type = `"obsp"`. Uses COO columns `cell_integer_id_i`, `cell_integer_id_j` and one value column per key.
- **LazyVarpView**: table_name = `"genesxgenes"`, table_type = `"varp"`; COO columns `gene_integer_id_i`, `gene_integer_id_j` and one value column per key.

### 4.2 COO-backed access (not vector mixin)

- obsp/varp are **COO-backed**: scalar columns (i, j, value per key), not FixedSizeListArray. So they do **not** use the obsm/varm vector mixin; they need dedicated get/set logic.
- **keys()**: From config `obsp.available` / `varp.available` (optionally synced with schema: columns that are float32 and not _i / _j).
- **__getitem__(key)**: Read table, select column `key` plus `_i`, `_j`. Filter to (i, j) where key column is present (or non-zero). If selector active: filter to (i, j) where both i and j are in the selected entity set; then build matrix of shape `(len(selector), len(selector))` with indices mapped to 0..len(selector)-1. Fill from COO rows; default 0.0. Return dense numpy `(n, n)` or (for very large/sparse) scipy sparse in a later iteration.
- **__setitem__(key, value)**: Validate value shape `(n_entities, n_entities)`. Convert to COO: (i, j) where value[i,j] != 0 (or all (i, j) for dense). If table exists and has other keys: merge (i, j) with existing rows, add/overwrite column `key`. If new table: create with _i, _j, key. Update config (available, mutable, dimensions).
- **__delitem__(key)**: Drop column `key` from table; remove key from config. If table has no key columns left, table can be removed or kept with only _i, _j (implementation choice).
- **Selector mapping**: Reuse same pattern as obsm/varm for axis: `obsp` → cell selector, `varp` → gene selector. Use for filtering (i, j) to selected IDs and for building the output square matrix shape.

### 4.3 Square-matrix semantics

- **Validation on set**: For obsp, require `value.shape == (n_cells, n_cells)`; for varp, `(n_genes, n_genes)`. When a selector is active, require shape `(len(selector), len(selector))`; write only (i, j) in the selected set, with indices in selector order.
- **Get with selector**: Filter COO rows to `_i` and `_j` in selected entity set; build matrix of shape `(len(selector), len(selector))` with IDs mapped to 0..len(selector)-1.
- **Dimensions in config**: For obsp, dimension = n_cells; for varp, n_genes. Used for validation and matrix shape on read.

### 4.4 LazyAnnData properties

- `LazyAnnData.obsp` -> `LazyObspView(self)` (lazy-instantiated, cached in `_obsp`).
- `LazyAnnData.varp` -> `LazyVarpView(self)` (cached in `_varp`).
- In `_invalidate_metadata_cache`, obsp/varp do **not** back obs/var, so we do not invalidate `_obs`/`_var` when only obsp/varp change; only clear view-level caches if any.

### 4.5 SLAFArray dataset setup

- In `_setup_datasets()` (slaf.py): if `config["tables"].get("cellsxcells")` exists, load `self.cellsxcells = lance.dataset(...)`; else `self.cellsxcells = None`. Same for `genesxgenes`. LazyObspView / LazyVarpView should handle `getattr(self._slaf_array, self.table_name, None)` being None (return empty keys, raise on __getitem__ if key requested). No reuse of vector mixin: obsp/varp use COO table columns (_i, _j, key1, key2, ...).

---

## 5. I/O and Mutations

- **Read**: Scan config for `obsp.available` / `varp.available`; read COO table (columns _i, _j, key); build dense matrix (n_entities, n_entities), fill from COO; apply selector by filtering (i, j) to selected IDs and mapping to (len(selector), len(selector)).
- **Add key**: Convert input matrix to COO; if table exists, add new float32 column and merge (i, j) rows (new rows for (i, j) only in new key, 0.0 for other keys; existing rows get new column). If table does not exist, create with _i, _j, key. Update config (available, mutable, dimensions).
- **Delete key**: Only if key is in mutable; drop that column from the COO table; update config.
- **Overwrite**: Only if key is mutable; drop column then add new column (from new matrix COO); update config.
- **Immutability**: Keys coming from h5ad conversion are in `immutable`; user-created keys go to `mutable`.

---

## 6. Tests

### 6.1 Fixtures and conftest

- Add `anndata_with_obsp_varp` (or extend `anndata_with_metadata`): AnnData with `obsp["connectivities"]`, `obsp["distances"]` (e.g. small dense matrices), and optionally `varp["some_key"]` (n_genes × n_genes). Reuse existing cell/gene IDs and sizes.

### 6.2 Conversion (e2e)

- `test_convert_anndata_with_obsp`: Convert h5ad with obsp (dense or sparse); assert `cellsxcells.lance` exists with COO columns (_i, _j, key), config has `obsp.available` / `obsp.immutable` / `obsp.dimensions`, and `LazyAnnData.obsp[key]` matches original after load (dense matrix built from COO).
- `test_convert_anndata_with_varp`: Same for varp and `genesxgenes.lance`.
- `test_obsp_accessible_after_conversion`, `test_varp_accessible_after_conversion`: Load converted SLAF; compare adata.obsp[key] and adata.varp[key] to original arrays.
- `test_obsp_varp_immutable_after_conversion`: Deleting or overwriting converted obsp/varp key raises.

### 6.3 Mutations

- `test_create_new_obsp_key`, `test_create_new_varp_key`: Add new obsp/varp key; assert in config mutable/available, correct shape, and round-trip after reload.
- `test_delete_mutable_obsp_key` / `test_delete_mutable_varp_key`: Add key, delete it, assert removed from config and table.
- `test_update_mutable_obsp_key` / `test_update_mutable_varp_key`: Overwrite mutable key; assert updated after reload.

### 6.4 Selectors

- `test_obsp_selector_support`: Set cell selector; `adata_subset.obsp["connectivities"]` has shape `(len(subset), len(subset))` and matches the same rows/columns of the full matrix.
- `test_varp_selector_support`: Same for varp with gene selector.

### 6.5 Edge cases

- Convert h5ad without obsp/varp: no cellsxcells/genesxgenes tables, no obsp/varp in config (or empty); `adata.obsp.keys()` / `adata.varp.keys()` are empty.
- Empty obsp/varp after conversion: `len(adata.obsp) == 0`, `adata.obsp["nonexistent"]` raises KeyError.

---

## 7. Implementation Order (checklist)

- [ ] **Step 1: Schema and config (PRD only)**
  - Define COO table schemas and config layout (this PRD).
  - **Commit:** `docs(prd): add obsp/varp implementation checklist and commit messages`

- [ ] **Step 2: Converter**
  - Implement `_convert_obsp` and `_convert_varp` (dense/sparse → COO); create `cellsxcells.lance` / `genesxgenes.lance` when present; extend `_save_config` for obsp_keys/varp_keys.
  - **Commit:** `feat(converter): add obsp and varp COO conversion from h5ad`

- [ ] **Step 3: SLAFArray**
  - In `_setup_datasets()`, optionally load `cellsxcells` and `genesxgenes` from config.
  - **Commit:** `feat(slaf): load cellsxcells and genesxgenes tables when present`

- [ ] **Step 4: COO views**
  - Implement `LazyObspView` and `LazyVarpView` (COO get/set/del, keys, selectors); add `obsp` and `varp` on `LazyAnnData`.
  - **Commit:** `feat(anndata): add LazyObspView and LazyVarpView with COO backend`

- [ ] **Step 5: Tests**
  - Conversion e2e, mutations, selectors, immutability, empty/absent obsp/varp.
  - **Commit:** `test(obsp,varp): add conversion, mutations, and selector tests`

---

## 8. References

- PR #27: Comprehensive support for anndata objects (obsm, varm, layers, obs_view, var_view, uns).
- PRDs: `OBSM_VARM_REFACTORING_ANALYSIS.md`, `OBS_VAR_VIEW_DATAFRAME_ANALYSIS.md`, `OBS_VAR_DEPRECATION_PLAN.md`, `LAYERS_SUPPORT_PRD.md`.
- AnnData: `obsp` / `varp` are square pairwise matrices aligned to obs/var.
- Tests: `tests/test_metadata_conversion_e2e.py`, `tests/test_metadata_mutations.py` (obsm/varm patterns).
