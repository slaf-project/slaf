import json
from dataclasses import dataclass
from typing import Any

import lance
import numpy as np
import pyarrow as pa
import polars as pl
import scipy.sparse


@dataclass(frozen=True)
class SparseTableDescriptor:
    table_attr: str
    table_config_key: str
    default_filename: str
    row_id_col: str
    col_id_col: str
    value_col: str = "value"
    key_col: str | None = None
    row_dtype: pl.DataType = pl.UInt32
    col_dtype: pl.DataType = pl.UInt32


EXPRESSION_SPARSE_TABLE = SparseTableDescriptor(
    table_attr="expression",
    table_config_key="expression",
    default_filename="expression.lance",
    row_id_col="cell_integer_id",
    col_id_col="gene_integer_id",
    key_col=None,
    row_dtype=pl.UInt32,
    col_dtype=pl.UInt16,
)

OBSM_SPARSE_TABLE = SparseTableDescriptor(
    table_attr="cells_sparse",
    table_config_key="cells_sparse",
    default_filename="cells_sparse.lance",
    row_id_col="cell_integer_id",
    col_id_col="gene_integer_id",
    key_col="obsm_key",
    row_dtype=pl.UInt32,
    col_dtype=pl.UInt32,
)

OBSM_SPARSE_COUNTS_TABLE = SparseTableDescriptor(
    table_attr="cells_sparse_counts",
    table_config_key="cells_sparse_counts",
    default_filename="cells_sparse_counts.lance",
    row_id_col="cell_integer_id",
    col_id_col="row_count",
    key_col="obsm_key",
    row_dtype=pl.UInt32,
    col_dtype=pl.UInt32,
)


def get_sparse_table_path(slaf_array: Any, descriptor: SparseTableDescriptor) -> str:
    return slaf_array._join_path(
        slaf_array.slaf_path,
        slaf_array.config.get("tables", {}).get(
            descriptor.table_config_key,
            descriptor.default_filename,
        ),
    )


def get_sparse_table(slaf_array: Any, descriptor: SparseTableDescriptor):
    return getattr(slaf_array, descriptor.table_attr, None)


def ensure_sparse_table_registered(
    slaf_array: Any,
    descriptor: SparseTableDescriptor,
) -> str:
    table_path = get_sparse_table_path(slaf_array, descriptor)
    if descriptor.table_config_key not in slaf_array.config.get("tables", {}):
        config_path = slaf_array._join_path(slaf_array.slaf_path, "config.json")
        with slaf_array._open_file(config_path) as f:
            config = json.load(f)
        config.setdefault("tables", {})[descriptor.table_config_key] = (
            descriptor.default_filename
        )
        with slaf_array._open_file(config_path, "w") as f:
            json.dump(config, f, indent=2)
        slaf_array.config = config
    return table_path


def _empty_sparse_frame(descriptor: SparseTableDescriptor) -> pl.DataFrame:
    data: dict[str, pl.Series] = {
        descriptor.row_id_col: pl.Series([], dtype=descriptor.row_dtype),
        descriptor.col_id_col: pl.Series([], dtype=descriptor.col_dtype),
        descriptor.value_col: pl.Series([], dtype=pl.Float32),
    }
    if descriptor.key_col is not None:
        data[descriptor.key_col] = pl.Series([], dtype=pl.Utf8)
    return pl.DataFrame(data)


def _write_sparse_row_counts(
    slaf_array: Any,
    *,
    logical_key: str,
    counts: np.ndarray,
) -> None:
    descriptor = OBSM_SPARSE_COUNTS_TABLE
    table_path = ensure_sparse_table_registered(slaf_array, descriptor)
    table = get_sparse_table(slaf_array, descriptor)
    cell_ids = np.flatnonzero(counts).astype(np.uint32, copy=False)
    row_counts = counts[cell_ids].astype(np.uint32, copy=False)
    if len(cell_ids) == 0:
        new_df = pl.DataFrame(
            {
                "obsm_key": pl.Series([], dtype=pl.Utf8),
                "cell_integer_id": pl.Series([], dtype=pl.UInt32),
                "row_count": pl.Series([], dtype=pl.UInt32),
            }
        )
    else:
        new_df = pl.DataFrame(
            {
                "obsm_key": np.full(len(cell_ids), logical_key, dtype=object),
                "cell_integer_id": cell_ids,
                "row_count": row_counts,
            }
        ).with_columns(
            pl.col("obsm_key").cast(pl.Utf8),
            pl.col("cell_integer_id").cast(pl.UInt32),
            pl.col("row_count").cast(pl.UInt32),
        )
    if table is None:
        lance.write_dataset(
            new_df.to_arrow(),
            table_path,
            mode="overwrite",
            data_storage_version="2.2",
        )
        setattr(slaf_array, descriptor.table_attr, lance.dataset(table_path))
        return
    try:
        existing_df = pl.from_arrow(
            table.to_table(columns=["obsm_key", "cell_integer_id", "row_count"])
        )
    except Exception:
        existing_df = pl.DataFrame(
            {
                "obsm_key": pl.Series([], dtype=pl.Utf8),
                "cell_integer_id": pl.Series([], dtype=pl.UInt32),
                "row_count": pl.Series([], dtype=pl.UInt32),
            }
        )
    kept_df = existing_df.filter(pl.col("obsm_key") != logical_key)
    merged_df = (
        pl.concat([kept_df, new_df], how="vertical_relaxed")
        if len(new_df) > 0
        else kept_df
    ).sort(["obsm_key", "cell_integer_id"])
    lance.write_dataset(
        merged_df.to_arrow(),
        table_path,
        mode="overwrite",
        data_storage_version="2.2",
    )
    setattr(slaf_array, descriptor.table_attr, lance.dataset(table_path))


def _load_sparse_frame(
    slaf_array: Any,
    descriptor: SparseTableDescriptor,
    *,
    logical_key: str | None = None,
) -> pl.DataFrame:
    table = get_sparse_table(slaf_array, descriptor)
    if table is None:
        return _empty_sparse_frame(descriptor)

    columns = [
        descriptor.row_id_col,
        descriptor.col_id_col,
        descriptor.value_col,
    ]
    if descriptor.key_col is not None:
        if logical_key is None:
            raise ValueError("logical_key is required for keyed sparse tables.")
        columns.append(descriptor.key_col)
        try:
            df = pl.from_arrow(
                table.to_table(
                    columns=columns,
                    filter=f"{descriptor.key_col} = '{logical_key}'",
                )
            )
            return df
        except TypeError:
            pass

    df = pl.from_arrow(table.to_table(columns=columns))
    if descriptor.key_col is not None:
        df = df.filter(pl.col(descriptor.key_col) == logical_key)
    return df


def _is_full_row_write(selected_row_ids: np.ndarray, n_rows: int) -> bool:
    return len(selected_row_ids) == n_rows and np.array_equal(
        selected_row_ids,
        np.arange(n_rows, dtype=selected_row_ids.dtype),
    )


def _sparse_matrix_to_frame(
    descriptor: SparseTableDescriptor,
    sparse_value: scipy.sparse.csr_matrix,
    selected_row_ids: np.ndarray,
    *,
    logical_key: str | None = None,
) -> pl.DataFrame:
    coo = sparse_value.tocoo()
    if coo.nnz == 0:
        return _empty_sparse_frame(descriptor)

    new_data: dict[str, Any] = {
        descriptor.row_id_col: selected_row_ids[coo.row],
        descriptor.col_id_col: coo.col.astype(np.uint32, copy=False),
        descriptor.value_col: coo.data.astype(np.float32, copy=False),
    }
    new_df = pl.DataFrame(new_data).with_columns(
        pl.col(descriptor.row_id_col).cast(descriptor.row_dtype),
        pl.col(descriptor.col_id_col).cast(descriptor.col_dtype),
        pl.col(descriptor.value_col).cast(pl.Float32),
    )
    if descriptor.key_col is not None:
        if logical_key is None:
            raise ValueError("logical_key is required for keyed sparse tables.")
        new_df = new_df.with_columns(pl.lit(logical_key).alias(descriptor.key_col))
    return new_df


def read_sparse_matrix(
    slaf_array: Any,
    descriptor: SparseTableDescriptor,
    *,
    selected_row_ids: np.ndarray,
    n_cols: int,
    selected_col_ids: np.ndarray | None = None,
    logical_key: str | None = None,
) -> scipy.sparse.csr_matrix:
    n_rows = len(selected_row_ids)
    if n_rows == 0 or n_cols == 0:
        return scipy.sparse.csr_matrix((n_rows, n_cols), dtype=np.float32)

    df = _load_sparse_frame(slaf_array, descriptor, logical_key=logical_key).filter(
        pl.col(descriptor.row_id_col).is_in(selected_row_ids)
    )
    if len(df) == 0:
        return scipy.sparse.csr_matrix((n_rows, n_cols), dtype=np.float32)

    row_ids = df[descriptor.row_id_col].to_numpy().astype(np.int64, copy=False)
    col_ids = df[descriptor.col_id_col].to_numpy().astype(np.int64, copy=False)
    values = df[descriptor.value_col].to_numpy().astype(np.float32, copy=False)

    sort_order = np.argsort(selected_row_ids)
    sorted_selected_row_ids = selected_row_ids[sort_order]
    positions = np.searchsorted(sorted_selected_row_ids, row_ids)
    valid = (positions < n_rows) & (sorted_selected_row_ids[positions] == row_ids)
    if not np.any(valid):
        return scipy.sparse.csr_matrix((n_rows, n_cols), dtype=np.float32)

    local_rows = sort_order[positions[valid]].astype(np.int64, copy=False)
    local_cols = col_ids[valid]
    local_values = values[valid]

    if selected_col_ids is not None:
        col_sort_order = np.argsort(selected_col_ids)
        sorted_selected_col_ids = selected_col_ids[col_sort_order]
        col_positions = np.searchsorted(sorted_selected_col_ids, local_cols)
        valid_cols = (col_positions < n_cols) & (
            sorted_selected_col_ids[col_positions] == local_cols
        )
        if not np.any(valid_cols):
            return scipy.sparse.csr_matrix((n_rows, n_cols), dtype=np.float32)
        local_rows = local_rows[valid_cols]
        local_cols = col_sort_order[col_positions[valid_cols]].astype(
            np.int64, copy=False
        )
        local_values = local_values[valid_cols]

    return scipy.sparse.csr_matrix(
        (
            local_values,
            (local_rows, local_cols),
        ),
        shape=(n_rows, n_cols),
        dtype=np.float32,
    )


def write_sparse_matrix(
    slaf_array: Any,
    descriptor: SparseTableDescriptor,
    *,
    matrix: scipy.sparse.spmatrix,
    selected_row_ids: np.ndarray,
    logical_key: str | None = None,
) -> None:
    sparse_value = scipy.sparse.csr_matrix(matrix, dtype=np.float32)
    new_df = _sparse_matrix_to_frame(
        descriptor,
        sparse_value,
        selected_row_ids,
        logical_key=logical_key,
    )

    full_row_write = _is_full_row_write(selected_row_ids, sparse_value.shape[0])
    table = get_sparse_table(slaf_array, descriptor)

    if table is None and full_row_write:
        table_path = ensure_sparse_table_registered(slaf_array, descriptor)
        lance.write_dataset(
            new_df.to_arrow(),
            table_path,
            mode="overwrite",
            data_storage_version="2.2",
        )
        setattr(slaf_array, descriptor.table_attr, lance.dataset(table_path))
        if descriptor.key_col is not None and logical_key is not None:
            row_counts = np.zeros(sparse_value.shape[0], dtype=np.int64)
            coo = sparse_value.tocoo(copy=False)
            if coo.nnz > 0:
                row_counts[selected_row_ids] = np.bincount(
                    coo.row, minlength=sparse_value.shape[0]
                ).astype(np.int64, copy=False)
            _write_sparse_row_counts(
                slaf_array,
                logical_key=logical_key,
                counts=row_counts,
            )
        return

    existing_df = _load_sparse_frame(slaf_array, descriptor)

    remove_expr = pl.col(descriptor.row_id_col).is_in(selected_row_ids)
    if descriptor.key_col is not None:
        if logical_key is None:
            raise ValueError("logical_key is required for keyed sparse tables.")
        remove_expr = remove_expr & (pl.col(descriptor.key_col) == logical_key)
    kept_df = existing_df.filter(~remove_expr)

    if len(new_df) > 0:
        merged_df = pl.concat([kept_df, new_df], how="vertical_relaxed")
    else:
        merged_df = kept_df

    sort_cols = [descriptor.row_id_col, descriptor.col_id_col]
    if descriptor.key_col is not None:
        sort_cols = [descriptor.key_col, *sort_cols]
    merged_df = merged_df.sort(sort_cols)

    table_path = ensure_sparse_table_registered(slaf_array, descriptor)
    lance.write_dataset(
        merged_df.to_arrow(),
        table_path,
        mode="overwrite",
        data_storage_version="2.2",
    )
    setattr(slaf_array, descriptor.table_attr, lance.dataset(table_path))
    if descriptor.key_col is not None and logical_key is not None:
        logical_df = merged_df.filter(pl.col(descriptor.key_col) == logical_key)
        row_counts = np.zeros(sparse_value.shape[0], dtype=np.int64)
        if len(logical_df) > 0:
            grouped = logical_df.group_by(descriptor.row_id_col).agg(
                pl.len().alias("n_rows")
            )
            row_ids = grouped[descriptor.row_id_col].to_numpy().astype(
                np.int64, copy=False
            )
            counts = grouped["n_rows"].to_numpy().astype(np.int64, copy=False)
            valid = row_ids < row_counts.shape[0]
            row_counts[row_ids[valid]] = counts[valid]
        _write_sparse_row_counts(
            slaf_array,
            logical_key=logical_key,
            counts=row_counts,
        )


def delete_sparse_matrix(
    slaf_array: Any,
    descriptor: SparseTableDescriptor,
    *,
    logical_key: str | None = None,
    selected_row_ids: np.ndarray | None = None,
) -> None:
    existing_df = _load_sparse_frame(slaf_array, descriptor)
    if len(existing_df) == 0:
        return

    remove_expr = pl.lit(True)
    if selected_row_ids is not None:
        remove_expr = remove_expr & pl.col(descriptor.row_id_col).is_in(selected_row_ids)
    if descriptor.key_col is not None:
        if logical_key is None:
            raise ValueError("logical_key is required for keyed sparse tables.")
        remove_expr = remove_expr & (pl.col(descriptor.key_col) == logical_key)

    filtered_df = existing_df.filter(~remove_expr)
    sort_cols = [descriptor.row_id_col, descriptor.col_id_col]
    if descriptor.key_col is not None:
        sort_cols = [descriptor.key_col, *sort_cols]
    filtered_df = filtered_df.sort(sort_cols)

    table_path = ensure_sparse_table_registered(slaf_array, descriptor)
    lance.write_dataset(
        filtered_df.to_arrow(),
        table_path,
        mode="overwrite",
        data_storage_version="2.2",
    )
    setattr(slaf_array, descriptor.table_attr, lance.dataset(table_path))
    if descriptor.key_col is not None and logical_key is not None:
        logical_df = filtered_df.filter(pl.col(descriptor.key_col) == logical_key)
        max_row = int(logical_df[descriptor.row_id_col].max()) if len(logical_df) > 0 else -1
        row_counts = np.zeros(max_row + 1 if max_row >= 0 else 0, dtype=np.int64)
        if len(logical_df) > 0:
            grouped = logical_df.group_by(descriptor.row_id_col).agg(
                pl.len().alias("n_rows")
            )
            row_ids = grouped[descriptor.row_id_col].to_numpy().astype(
                np.int64, copy=False
            )
            counts = grouped["n_rows"].to_numpy().astype(np.int64, copy=False)
            row_counts[row_ids] = counts
        _write_sparse_row_counts(
            slaf_array,
            logical_key=logical_key,
            counts=row_counts,
        )


def compute_row_counts(
    slaf_array: Any,
    descriptor: SparseTableDescriptor,
    *,
    n_rows: int,
    logical_key: str | None = None,
) -> np.ndarray:
    if descriptor == EXPRESSION_SPARSE_TABLE:
        obs_df = getattr(slaf_array, "obs", None)
        if obs_df is not None:
            if "cell_start_index" in obs_df.columns:
                cell_start_index = (
                    obs_df["cell_start_index"].to_numpy().astype(np.int64, copy=False)
                )
                if len(cell_start_index) == n_rows:
                    total_expression_count = int(slaf_array.expression.count_rows())
                    boundaries = np.concatenate(
                        [cell_start_index, np.asarray([total_expression_count])],
                    )
                    return np.diff(boundaries)
            if "n_genes" in obs_df.columns:
                return obs_df["n_genes"].to_numpy().astype(np.int64, copy=False)
            if "gene_count" in obs_df.columns:
                return obs_df["gene_count"].to_numpy().astype(np.int64, copy=False)

    if descriptor.key_col is not None and logical_key is not None:
        table = get_sparse_table(slaf_array, OBSM_SPARSE_COUNTS_TABLE)
        if table is not None:
            try:
                df = pl.from_arrow(
                    table.to_table(
                        columns=["cell_integer_id", "row_count", "obsm_key"],
                        filter=f"obsm_key = '{logical_key}'",
                    )
                )
            except TypeError:
                df = pl.from_arrow(
                    table.to_table(columns=["cell_integer_id", "row_count", "obsm_key"])
                ).filter(pl.col("obsm_key") == logical_key)
            counts = np.zeros(n_rows, dtype=np.int64)
            if len(df) == 0:
                return counts
            row_ids = df["cell_integer_id"].to_numpy().astype(np.int64, copy=False)
            row_counts = df["row_count"].to_numpy().astype(np.int64, copy=False)
            valid = row_ids < n_rows
            counts[row_ids[valid]] = row_counts[valid]
            return counts

    counts = np.zeros(n_rows, dtype=np.int64)
    df = _load_sparse_frame(slaf_array, descriptor, logical_key=logical_key)
    if len(df) == 0:
        return counts

    grouped = df.group_by(descriptor.row_id_col).agg(pl.len().alias("n_rows"))
    row_ids = grouped[descriptor.row_id_col].to_numpy().astype(np.int64, copy=False)
    row_counts = grouped["n_rows"].to_numpy().astype(np.int64, copy=False)
    counts[row_ids] = row_counts
    return counts
