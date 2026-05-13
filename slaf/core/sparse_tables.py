import json
from dataclasses import dataclass
from typing import Any, cast

import lance
import numpy as np
import polars as pl
import scipy.sparse

from slaf.core.sparse_ops import LazySparseMixin


@dataclass(frozen=True)
class SparseTableDescriptor:
    table_attr: str
    table_config_key: str
    default_filename: str
    row_id_col: str
    col_id_col: str
    value_col: str = "value"
    key_col: str | None = None
    row_dtype: pl.DataType = pl.UInt32()
    col_dtype: pl.DataType = pl.UInt32()


EXPRESSION_SPARSE_TABLE = SparseTableDescriptor(
    table_attr="expression",
    table_config_key="expression",
    default_filename="expression.lance",
    row_id_col="cell_integer_id",
    col_id_col="gene_integer_id",
    key_col=None,
    row_dtype=pl.UInt32(),
    col_dtype=pl.UInt16(),
)

OBSM_SPARSE_TABLE = SparseTableDescriptor(
    table_attr="cells_sparse",
    table_config_key="cells_sparse",
    default_filename="cells_sparse.lance",
    row_id_col="cell_integer_id",
    col_id_col="gene_integer_id",
    key_col="obsm_key",
    row_dtype=pl.UInt32(),
    col_dtype=pl.UInt32(),
)

OBSM_SPARSE_COUNTS_TABLE = SparseTableDescriptor(
    table_attr="cells_sparse_counts",
    table_config_key="cells_sparse_counts",
    default_filename="cells_sparse_counts.lance",
    row_id_col="cell_integer_id",
    col_id_col="row_count",
    key_col="obsm_key",
    row_dtype=pl.UInt32(),
    col_dtype=pl.UInt32(),
)


class LazySparseObsmMatrix(LazySparseMixin):
    """Lazy sparse matrix view for one sparse obsm key."""

    def __init__(
        self,
        slaf_array: Any,
        *,
        key: str,
        n_obs: int,
        n_features: int,
    ):
        super().__init__()
        self.slaf_array = slaf_array
        self.key = key
        self._base_shape = (n_obs, n_features)
        self._shape = self._base_shape
        self._cell_selector: Any = None
        self._feature_selector: Any = None

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def __getitem__(self, key) -> "LazySparseObsmMatrix":
        cell_selector, feature_selector = self._parse_key(key)
        new_matrix = LazySparseObsmMatrix(
            self.slaf_array,
            key=self.key,
            n_obs=self._base_shape[0],
            n_features=self._base_shape[1],
        )
        new_matrix._cell_selector = self._compose_selectors(
            self._cell_selector, cell_selector, axis=0
        )
        new_matrix._feature_selector = self._compose_selectors(
            self._feature_selector, feature_selector, axis=1
        )
        new_matrix._update_shape()
        return new_matrix

    def _compose_selectors(self, old, new, axis):
        axis_size = self._base_shape[axis]
        if old is None:
            return new
        if new is None or (isinstance(new, slice) and new == slice(None)):
            return old
        if isinstance(old, slice):
            old_start = old.start or 0
            old_stop = old.stop or axis_size
            old_step = old.step or 1
            if old_start < 0:
                old_start = axis_size + old_start
            if old_stop < 0:
                old_stop = axis_size + old_stop
            old_start = max(0, min(old_start, axis_size))
            old_stop = max(0, min(old_stop, axis_size))
            old_range = list(range(old_start, old_stop, old_step))
            if isinstance(new, slice):
                new_start = new.start or 0
                new_stop = new.stop or len(old_range)
                new_step = new.step or 1
                if new_start < 0:
                    new_start = len(old_range) + new_start
                if new_stop < 0:
                    new_stop = len(old_range) + new_stop
                new_start = max(0, min(new_start, len(old_range)))
                new_stop = max(0, min(new_stop, len(old_range)))
                return old_range[new_start:new_stop:new_step]
            if isinstance(new, int | np.integer):
                return [old_range[new]] if 0 <= new < len(old_range) else []
            if isinstance(new, list | np.ndarray):
                result = []
                for idx in new:
                    if 0 <= idx < len(old_range):
                        result.append(old_range[idx])
                return result
            return new
        if isinstance(old, list | np.ndarray):
            if isinstance(new, slice):
                new_start = new.start or 0
                new_stop = new.stop or len(old)
                new_step = new.step or 1
                if new_start < 0:
                    new_start = len(old) + new_start
                if new_stop < 0:
                    new_stop = len(old) + new_stop
                new_start = max(0, min(new_start, len(old)))
                new_stop = max(0, min(new_stop, len(old)))
                return old[new_start:new_stop:new_step]
            if isinstance(new, int | np.integer):
                return [old[new]] if 0 <= new < len(old) else []
            if isinstance(new, list | np.ndarray):
                result = []
                for idx in new:
                    if 0 <= idx < len(old):
                        result.append(old[idx])
                return result
            return new
        return new

    def _calculate_selected_count(self, selector, axis: int) -> int:
        axis_size = self._base_shape[axis]
        if selector is None or (
            isinstance(selector, slice) and selector == slice(None)
        ):
            return axis_size
        if isinstance(selector, slice):
            start = selector.start or 0
            stop = selector.stop or axis_size
            step = selector.step or 1
            start = max(0, min(start, axis_size))
            stop = max(0, min(stop, axis_size))
            return len(range(start, stop, step))
        if isinstance(selector, list | np.ndarray):
            if isinstance(selector, np.ndarray) and selector.dtype == bool:
                return int(np.sum(selector))
            return len(selector)
        if isinstance(selector, int | np.integer):
            return 1
        return axis_size

    def _update_shape(self):
        self._shape = (
            self._calculate_selected_count(self._cell_selector, axis=0),
            self._calculate_selected_count(self._feature_selector, axis=1),
        )

    def _selector_to_ids(self, selector, axis: int) -> np.ndarray:
        axis_size = self._base_shape[axis]
        dtype = np.uint32
        if selector is None or (
            isinstance(selector, slice) and selector == slice(None)
        ):
            return np.arange(axis_size, dtype=dtype)
        if isinstance(selector, slice):
            start = selector.start or 0
            stop = selector.stop or axis_size
            step = selector.step or 1
            return np.arange(start, stop, step, dtype=dtype)
        if isinstance(selector, list):
            return np.asarray(selector, dtype=dtype)
        if isinstance(selector, np.ndarray):
            if selector.dtype == bool:
                return np.flatnonzero(selector).astype(dtype, copy=False)
            return selector.astype(dtype, copy=False)
        if isinstance(selector, int | np.integer):
            return np.asarray([int(selector)], dtype=dtype)
        raise TypeError(f"Unsupported selector type: {type(selector)}")

    def compute(self) -> scipy.sparse.csr_matrix:
        selected_row_ids = self._selector_to_ids(self._cell_selector, axis=0)
        selected_col_ids = (
            None
            if self._feature_selector is None
            or (
                isinstance(self._feature_selector, slice)
                and self._feature_selector == slice(None)
            )
            else self._selector_to_ids(self._feature_selector, axis=1)
        )
        return read_sparse_matrix(
            self.slaf_array,
            OBSM_SPARSE_TABLE,
            selected_row_ids=selected_row_ids,
            selected_col_ids=selected_col_ids,
            n_cols=self.shape[1],
            logical_key=self.key,
        )

    def toarray(self) -> np.ndarray:
        return self.compute().toarray()


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
    cell_ids: np.ndarray = np.flatnonzero(counts).astype(np.uint32, copy=False)
    row_counts: np.ndarray = counts[cell_ids].astype(np.uint32, copy=False)
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
        existing_df = cast(
            pl.DataFrame,
            pl.from_arrow(
                table.to_table(columns=["obsm_key", "cell_integer_id", "row_count"])
            ),
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
        cast(pl.DataFrame, pl.concat([kept_df, new_df], how="vertical_relaxed"))
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
            df = cast(
                pl.DataFrame,
                pl.from_arrow(
                    table.to_table(
                        columns=columns,
                        filter=f"{descriptor.key_col} = '{logical_key}'",
                    )
                ),
            )
            return df
        except TypeError:
            pass

    df = cast(pl.DataFrame, pl.from_arrow(table.to_table(columns=columns)))
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
    positions = np.asarray(np.searchsorted(sorted_selected_row_ids, row_ids))
    valid = (positions < n_rows) & (sorted_selected_row_ids[positions] == row_ids)
    if not np.any(valid):
        return scipy.sparse.csr_matrix((n_rows, n_cols), dtype=np.float32)

    local_rows = sort_order[positions[valid]].astype(np.int64, copy=False)
    local_cols = col_ids[valid]
    local_values = values[valid]

    if selected_col_ids is not None:
        col_sort_order = np.argsort(selected_col_ids)
        sorted_selected_col_ids = selected_col_ids[col_sort_order]
        col_positions = np.asarray(np.searchsorted(sorted_selected_col_ids, local_cols))
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
        merged_df = cast(
            pl.DataFrame,
            pl.concat([kept_df, new_df], how="vertical_relaxed"),
        )
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
            row_ids = (
                grouped[descriptor.row_id_col].to_numpy().astype(np.int64, copy=False)
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
        remove_expr = remove_expr & pl.col(descriptor.row_id_col).is_in(
            selected_row_ids
        )
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
        row_id_values = logical_df.get_column(descriptor.row_id_col).to_numpy()
        max_row = int(row_id_values.max()) if len(row_id_values) > 0 else -1
        row_counts: np.ndarray = np.zeros(
            max_row + 1 if max_row >= 0 else 0,
            dtype=np.int64,
        )
        if len(logical_df) > 0:
            grouped = logical_df.group_by(descriptor.row_id_col).agg(
                pl.len().alias("n_rows")
            )
            row_ids = (
                grouped[descriptor.row_id_col].to_numpy().astype(np.int64, copy=False)
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
                df = cast(
                    pl.DataFrame,
                    pl.from_arrow(
                        table.to_table(
                            columns=["cell_integer_id", "row_count", "obsm_key"],
                            filter=f"obsm_key = '{logical_key}'",
                        )
                    ),
                )
            except TypeError:
                df = cast(
                    pl.DataFrame,
                    pl.from_arrow(
                        table.to_table(
                            columns=["cell_integer_id", "row_count", "obsm_key"]
                        )
                    ),
                ).filter(pl.col("obsm_key") == logical_key)
            counts: np.ndarray = np.zeros(n_rows, dtype=np.int64)
            if len(df) == 0:
                return counts
            row_ids = (
                df.get_column("cell_integer_id").to_numpy().astype(np.int64, copy=False)
            )
            row_counts = (
                df.get_column("row_count").to_numpy().astype(np.int64, copy=False)
            )
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
