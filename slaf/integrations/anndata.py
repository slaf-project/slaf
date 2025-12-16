from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import scipy.sparse

from slaf.core.slaf import SLAFArray
from slaf.core.sparse_ops import LazySparseMixin

if TYPE_CHECKING:
    import scanpy as sc

# Import FragmentProcessor for runtime use
try:
    from slaf.core.fragment_processor import FragmentProcessor
except ImportError:
    FragmentProcessor = None  # type: ignore


class LazyExpressionMatrix(LazySparseMixin):
    """
    Lazy expression matrix backed by SLAF with scipy.sparse interface.

    LazyExpressionMatrix provides a scipy.sparse-compatible interface for accessing
    single-cell expression data stored in SLAF format. It implements lazy evaluation
    to avoid loading all data into memory, making it suitable for large datasets.

    Key Features:
        - scipy.sparse-compatible interface
        - Lazy evaluation for memory efficiency
        - Caching for repeated queries
        - Support for cell and gene subsetting
        - Integration with AnnData objects

    Examples:
        >>> # Basic usage
        >>> slaf_array = SLAFArray("path/to/data.slaf")
        >>> lazy_matrix = LazyExpressionMatrix(slaf_array)
        >>> print(f"Matrix shape: {lazy_matrix.shape}")
        Matrix shape: (1000, 20000)

        >>> # With AnnData integration
        >>> adata = LazyAnnData(slaf_array)
        >>> matrix = adata.X
        >>> print(f"Expression matrix shape: {matrix.shape}")
        Expression matrix shape: (1000, 20000)

        >>> # Subsetting operations
        >>> subset_matrix = matrix[:100, :5000]  # First 100 cells, first 5000 genes
        >>> print(f"Subset shape: {subset_matrix.shape}")
        Subset shape: (100, 5000)
    """

    def __init__(
        self,
        slaf_array: SLAFArray,
        table_name: str = "expression",
        layer_name: str | None = None,
    ):
        """
        Initialize lazy expression matrix with SLAF array.

        Args:
            slaf_array: SLAFArray instance containing the single-cell data.
                       Used for database queries and metadata access.
            table_name: Table name to query ("expression" or "layers"). Default: "expression"
            layer_name: Layer name for layers table (required when table_name="layers").
                       Default: None

        Examples:
            >>> # Basic initialization
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> matrix = LazyExpressionMatrix(slaf_array)
            >>> print(f"Initialized with shape: {matrix.shape}")
            Initialized with shape: (1000, 20000)

            >>> # Check parent reference
            >>> print(f"Parent adata: {matrix.parent_adata}")
            Parent adata: None

            >>> # Initialize for layers table
            >>> layer_matrix = LazyExpressionMatrix(slaf_array, table_name="layers", layer_name="spliced")
            >>> print(f"Layer matrix shape: {layer_matrix.shape}")
            Layer matrix shape: (1000, 20000)
        """
        super().__init__()
        self.slaf_array = slaf_array
        self.table_name = table_name
        self.layer_name = layer_name
        self.parent_adata: LazyAnnData | None = None
        # Store slicing selectors
        self._cell_selector: Any = None
        self._gene_selector: Any = None
        # Initialize shape attribute (required by LazySparseMixin)
        self._shape = self.slaf_array.shape
        self._cache: dict[str, Any] = {}  # Simple caching for repeated queries
        # Validate parameters
        if table_name == "layers" and layer_name is None:
            raise ValueError("layer_name must be provided when table_name='layers'")

    @property
    def shape(self) -> tuple[int, int]:
        """
        Shape of the expression matrix.

        Returns:
            Tuple of (n_cells, n_genes) representing the matrix dimensions.

        Examples:
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> matrix = LazyExpressionMatrix(slaf_array)
            >>> print(f"Matrix shape: {matrix.shape}")
            Matrix shape: (1000, 20000)
        """
        return self._shape

    @property
    def obs_names(self) -> pd.Index | None:
        """
        Cell names (observations).

        Returns:
            pandas.Index of cell names if parent AnnData is available, None otherwise.

        Examples:
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> matrix = adata.X
            >>> print(f"Cell names: {len(matrix.obs_names)}")
            Cell names: 1000
        """
        if hasattr(self, "parent_adata") and self.parent_adata is not None:
            return self.parent_adata.obs_names
        return None

    @property
    def var_names(self) -> pd.Index | None:
        """
        Gene names (variables).

        Returns:
            pandas.Index of gene names if parent AnnData is available, None otherwise.

        Examples:
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> matrix = adata.X
            >>> print(f"Gene names: {len(matrix.var_names)}")
            Gene names: 20000
        """
        if hasattr(self, "parent_adata") and self.parent_adata is not None:
            return self.parent_adata.var_names
        return None

    def _update_shape(self):
        """Update the shape attribute based on current selectors"""
        if self._cell_selector is not None or self._gene_selector is not None:
            # Calculate the shape based on selectors
            cell_selector = (
                self._cell_selector if self._cell_selector is not None else slice(None)
            )
            gene_selector = (
                self._gene_selector if self._gene_selector is not None else slice(None)
            )

            # Use the same logic as _get_result_shape in LazySparseMixin
            cell_selector = self._compose_selectors(cell_selector, None, axis=0)
            gene_selector = self._compose_selectors(gene_selector, None, axis=1)

            n_cells = self._calculate_selected_count(cell_selector, axis=0)
            n_genes = self._calculate_selected_count(gene_selector, axis=1)

            self._shape = (n_cells, n_genes)
        else:
            # No selectors applied, return original shape
            self._shape = self.slaf_array.shape

    def _calculate_selected_count(self, selector, axis: int) -> int:
        """Calculate the number of selected entities for a given selector"""
        if selector is None or (
            isinstance(selector, slice) and selector == slice(None)
        ):
            return self.slaf_array.shape[axis]

        if isinstance(selector, slice):
            start = selector.start or 0
            stop = selector.stop or self.slaf_array.shape[axis]
            step = selector.step or 1

            # Clamp bounds to actual data size
            start = max(0, min(start, self.slaf_array.shape[axis]))
            stop = max(0, min(stop, self.slaf_array.shape[axis]))

            return len(range(start, stop, step))
        elif isinstance(selector, list | np.ndarray):
            if isinstance(selector, np.ndarray) and selector.dtype == bool:
                return np.sum(selector)
            return len(selector)
        elif isinstance(selector, int | np.integer):
            return 1
        else:
            return self.slaf_array.shape[axis]

    def __getitem__(self, key) -> "LazyExpressionMatrix":
        """
        Lazy slicing - returns a new LazyExpressionMatrix with composed selectors
        No computation happens until .compute() is called
        """
        cell_selector, gene_selector = self._parse_key(key)

        # Handle selectors directly as integer indices
        # For slices, lists, ints - these are already relative to the current view
        # We need to compose them with existing selectors

        # Create a new LazyExpressionMatrix with composed selectors
        new_matrix = LazyExpressionMatrix(
            self.slaf_array, table_name=self.table_name, layer_name=self.layer_name
        )
        new_matrix.parent_adata = self.parent_adata

        # Compose selectors (these are relative to current view)
        new_matrix._cell_selector = self._compose_selectors(
            self._cell_selector, cell_selector, axis=0
        )
        new_matrix._gene_selector = self._compose_selectors(
            self._gene_selector, gene_selector, axis=1
        )

        # Update shape based on new selectors
        new_matrix._update_shape()

        return new_matrix

    def _compose_selectors(self, old, new, axis):
        """Compose two selectors (helper method)"""
        if old is None:
            return new
        if new is None or (isinstance(new, slice) and new == slice(None)):
            return old

        # If old is a slice, we need to apply new to the range defined by old
        if isinstance(old, slice):
            # Get the range from the old slice
            old_start = old.start or 0
            old_stop = old.stop or self.slaf_array.shape[axis]
            old_step = old.step or 1

            # Handle negative indices in old slice
            if old_start < 0:
                old_start = self.slaf_array.shape[axis] + old_start
            if old_stop < 0:
                old_stop = self.slaf_array.shape[axis] + old_stop

            # Clamp old slice bounds
            old_start = max(0, min(old_start, self.slaf_array.shape[axis]))
            old_stop = max(0, min(old_stop, self.slaf_array.shape[axis]))

            # Create the range from the old slice
            old_range = list(range(old_start, old_stop, old_step))

            # Now apply the new selector to this range
            if isinstance(new, slice):
                # Apply new slice to the old range
                new_start = new.start or 0
                new_stop = new.stop or len(old_range)
                new_step = new.step or 1

                # Handle negative indices in new slice
                if new_start < 0:
                    new_start = len(old_range) + new_start
                if new_stop < 0:
                    new_stop = len(old_range) + new_stop

                # Clamp new slice bounds
                new_start = max(0, min(new_start, len(old_range)))
                new_stop = max(0, min(new_stop, len(old_range)))

                # Apply the new slice to the old range
                result_indices = old_range[new_start:new_stop:new_step]
                return result_indices
            elif isinstance(new, int | np.integer):
                # Single index into the old range
                if 0 <= new < len(old_range):
                    return [old_range[new]]
                else:
                    return []
            elif isinstance(new, list | np.ndarray):
                # List of indices into the old range
                result = []
                for idx in new:
                    if 0 <= idx < len(old_range):
                        result.append(old_range[idx])
                return result
            else:
                return new

        # If old is a list of indices, apply new to those indices
        elif isinstance(old, list | np.ndarray):
            if isinstance(new, slice):
                # Apply slice to the old list
                new_start = new.start or 0
                new_stop = new.stop or len(old)
                new_step = new.step or 1

                # Handle negative indices
                if new_start < 0:
                    new_start = len(old) + new_start
                if new_stop < 0:
                    new_stop = len(old) + new_stop

                # Clamp bounds
                new_start = max(0, min(new_start, len(old)))
                new_stop = max(0, min(new_stop, len(old)))

                return old[new_start:new_stop:new_step]
            elif isinstance(new, int | np.integer):
                # Single index into the old list
                if 0 <= new < len(old):
                    return [old[new]]
                else:
                    return []
            elif isinstance(new, list | np.ndarray):
                # List of indices into the old list
                result = []
                for idx in new:
                    if 0 <= idx < len(old):
                        result.append(old[idx])
                return result
            else:
                return new

        # For other cases, return the new selector as fallback
        return new

    def _estimate_slice_size(self, cell_selector, gene_selector) -> int:
        """Estimate the size of the slice for strategy selection"""
        cell_count = self._estimate_selected_count(cell_selector, axis=0)
        gene_count = self._estimate_selected_count(gene_selector, axis=1)
        return cell_count * gene_count

    def _apply_transformations(
        self,
        matrix: scipy.sparse.csr_matrix,
        cell_selector,
        gene_selector,
    ) -> scipy.sparse.csr_matrix:
        """Apply any stored transformations to the matrix"""
        # Get transformations from the parent LazyAnnData if available
        if self.parent_adata is not None and hasattr(
            self.parent_adata, "_transformations"
        ):
            transformations = self.parent_adata._transformations
        else:
            return matrix

        # Avoid copying if no transformations
        if not transformations:
            return matrix

        # Try to apply transformations at SQL level first
        sql_transformed = self._apply_sql_transformations(
            cell_selector, gene_selector, transformations
        )

        if sql_transformed is not None:
            # SQL transformations were applied, reconstruct matrix
            return self._reconstruct_sparse_matrix(
                sql_transformed, cell_selector, gene_selector
            )

        # Fall back to numpy transformations
        return self._apply_numpy_transformations(
            matrix, cell_selector, gene_selector, transformations
        )

    def _apply_sql_transformations(
        self, cell_selector, gene_selector, transformations
    ) -> pd.DataFrame | None:
        """Apply transformations at SQL level when possible"""
        # Check if we can apply all transformations in SQL
        sql_applicable = []
        numpy_needed = []

        for transform_name, transform_data in transformations.items():
            if transform_name == "normalize_total":
                # normalize_total can be applied in SQL
                sql_applicable.append((transform_name, transform_data))
            elif transform_name == "log1p":
                # log1p can be applied in SQL
                sql_applicable.append((transform_name, transform_data))
            else:
                # Unknown transformation, needs numpy
                numpy_needed.append((transform_name, transform_data))

        # If we have numpy-only transformations, don't use SQL
        if numpy_needed:
            return None

        # Build SQL query with transformations
        # Convert list of tuples to dictionary for _build_transformed_query
        sql_transformations = dict(sql_applicable)
        query = self._build_transformed_query(
            cell_selector, gene_selector, sql_transformations
        )

        try:
            # Execute the transformed query
            return self.slaf_array.query(query)
        except Exception:
            # If SQL transformation fails, fall back to numpy
            return None

    def _build_transformed_query(
        self, cell_selector, gene_selector, transformations
    ) -> str:
        """Build SQL query with transformations applied"""
        # Build base SQL query string
        base_query = self._build_submatrix_sql(cell_selector, gene_selector)

        # Apply transformations in order
        transformed_query = base_query
        for transform_name, transform_data in transformations.items():
            if transform_name == "normalize_total":
                transformed_query = self._apply_sql_normalize_total(
                    transformed_query, transform_data
                )
            elif transform_name == "log1p":
                transformed_query = self._apply_sql_log1p(transformed_query)

        return transformed_query

    def _build_submatrix_sql(self, cell_selector, gene_selector) -> str:
        """Build SQL query string for submatrix selection"""
        # Use the QueryOptimizer to build the SQL string
        from slaf.core.query_optimizer import QueryOptimizer

        return QueryOptimizer.build_submatrix_query(
            cell_selector=cell_selector,
            gene_selector=gene_selector,
            cell_count=self.slaf_array.shape[0],  # Use original dataset dimensions
            gene_count=self.slaf_array.shape[1],  # Use original dataset dimensions
            table_name=self.table_name,
            layer_name=self.layer_name,
        )

    def _apply_sql_normalize_total(self, query: str, transform_data: dict) -> str:
        """Apply normalize_total transformation in SQL"""
        cell_factors = transform_data["cell_factors"]
        # target_sum = transform_data["target_sum"]  # Removed unused variable

        # Create a CASE statement for cell factors
        case_statements = []
        for cell_id, factor in cell_factors.items():
            # Convert scientific notation to regular decimal format for SQL compatibility
            factor_str = (
                f"{factor:.10f}".rstrip("0").rstrip(".") if factor != 0 else "0"
            )

            # Handle both integer and string cell IDs
            if isinstance(cell_id, str) and cell_id.startswith("cell_"):
                # Extract integer from string like "cell_0" -> 0
                try:
                    cell_integer_id_int = int(cell_id.split("_")[1])
                except (ValueError, IndexError):
                    # Fallback: skip this cell if we can't parse it
                    continue
            elif isinstance(cell_id, int | np.integer):
                cell_integer_id_int = int(cell_id)
            else:
                # Skip if we can't handle this cell ID type
                continue

            case_statements.append(
                f"WHEN e.cell_integer_id = {cell_integer_id_int} THEN {factor_str}"
            )

        factor_case = "CASE " + " ".join(case_statements) + " ELSE 1.0 END"

        # Wrap the query to apply normalization
        return f"""
        SELECT
            e.cell_integer_id,
            e.gene_integer_id,
            e.value * {factor_case} as value
        FROM ({query}) as base_data e
        """

    def _apply_sql_log1p(self, query: str) -> str:
        """Apply log1p transformation in SQL, only to nonzero values (sparse semantics)"""
        return f"""
        SELECT
            e.cell_integer_id,
            e.gene_integer_id,
            CASE WHEN e.value != 0 THEN LN(1 + e.value) ELSE 0 END as value
        FROM ({query}) as base_data e
        """

    def _apply_numpy_transformations(
        self,
        matrix: scipy.sparse.csr_matrix,
        cell_selector,
        gene_selector,
        transformations,
    ) -> scipy.sparse.csr_matrix:
        """Apply transformations using numpy operations"""
        # Apply transformations in the order they were added (preserves order)
        result = matrix
        for transform_name, transform_data in transformations.items():
            if transform_name == "normalize_total":
                result = self._apply_normalize_total(
                    result, cell_selector, transform_data
                )
            elif transform_name == "log1p":
                result = self._apply_log1p(result)

        return result

    def _apply_normalize_total(
        self,
        matrix: scipy.sparse.csr_matrix,
        cell_selector,
        transform_data,
    ) -> scipy.sparse.csr_matrix:
        """Apply normalize_total transformation using vectorized operations"""
        cell_factors = transform_data.get("cell_factors", {})
        obs_names_local = []  # Always a list
        obs_names = None
        if hasattr(self, "parent_adata") and self.parent_adata is not None:
            try:
                obs_names = self.parent_adata.obs_names
            except (AttributeError, TypeError):
                obs_names = None
        if obs_names is None or not isinstance(obs_names, list | np.ndarray | pd.Index):
            if (
                matrix is not None
                and hasattr(matrix, "shape")
                and matrix.shape is not None
            ):
                obs_names_local = [f"cell_{i}" for i in range(matrix.shape[0])]
            else:
                obs_names_local = []
        else:
            obs_names_local = list(obs_names)
        # Now obs_names_local is always a list
        # Determine selected_cell_names based on cell_selector
        if cell_selector is None or (
            isinstance(cell_selector, slice) and cell_selector == slice(None)
        ):
            selected_cell_names = obs_names_local
        elif isinstance(cell_selector, slice):
            start = cell_selector.start or 0
            stop = cell_selector.stop or len(obs_names_local)
            step = cell_selector.step or 1
            # Clamp bounds
            start = max(0, min(start, len(obs_names_local)))
            stop = max(0, min(stop, len(obs_names_local)))
            selected_cell_names = obs_names_local[start:stop:step]
        elif isinstance(cell_selector, list | np.ndarray):
            if len(obs_names_local) > 0:
                if (
                    isinstance(cell_selector, np.ndarray)
                    and cell_selector.dtype == bool
                ):
                    selected_cell_names = [
                        obs_names_local[i]
                        for i, keep in enumerate(cell_selector)
                        if keep and 0 <= i < len(obs_names_local)
                    ]
                else:
                    selected_cell_names = [
                        obs_names_local[i]
                        for i in cell_selector
                        if isinstance(i, int | np.integer)
                        and 0 <= i < len(obs_names_local)
                    ]
            else:
                selected_cell_names = []
        elif isinstance(cell_selector, int | np.integer):
            if 0 <= cell_selector < len(obs_names_local):
                selected_cell_names = [obs_names_local[cell_selector]]
            else:
                selected_cell_names = []
        else:
            selected_cell_names = obs_names_local
        # Create a vector of factors for all cells at once
        cell_factors_vector = np.array(
            [cell_factors.get(name, 1.0) for name in selected_cell_names]
        )
        # Apply vectorized scaling using CSR matrix properties
        # Create a copy only if we need to modify the data
        if not np.allclose(cell_factors_vector, 1.0):
            result = matrix.copy()
            # Apply scaling to each row using vectorized operations
            for i in range(result.shape[0]):
                if i < len(cell_factors_vector):
                    factor = cell_factors_vector[i]
                    if factor != 1.0:
                        # Scale the non-zero elements in this row
                        start_idx = result.indptr[i]
                        end_idx = result.indptr[i + 1]
                        result.data[start_idx:end_idx] *= factor
            return result
        else:
            # No scaling needed, return original matrix
            return matrix

    def _apply_log1p(self, matrix: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
        """Apply log1p transformation using vectorized operations"""
        # Create a copy only if we need to modify the data
        result = matrix.copy()
        # Apply log1p to all non-zero values using vectorized operation
        result.data = np.log1p(result.data)
        return result

    def __array_function__(self, func, types, args, kwargs):
        """Intercept numpy functions for lazy evaluation"""
        if func == np.mean:
            axis = kwargs.get("axis", None)
            return self.mean(axis=axis)
        elif func == np.sum:
            axis = kwargs.get("axis", None)
            return self.sum(axis=axis)
        elif func == np.var:
            axis = kwargs.get("axis", None)
            return self.var(axis=axis)
        elif func == np.std:
            axis = kwargs.get("axis", None)
            return self.std(axis=axis)
        else:
            # Fall back to materializing the matrix
            matrix = self.compute()
            return func(matrix, *args[1:], **kwargs)

    def mean(
        self, axis: int | None = None, fragments: bool | None = None
    ) -> float | np.ndarray:
        """Compute mean along axis via SQL aggregation"""
        return self._aggregation_with_fragments("mean", fragments, axis=axis)

    def sum(
        self, axis: int | None = None, fragments: bool | None = None
    ) -> float | np.ndarray:
        """Compute sum along axis via SQL aggregation"""
        return self._aggregation_with_fragments("sum", fragments, axis=axis)

    def var(self, axis: int | None = None) -> float | np.ndarray:
        """Compute variance along axis via SQL aggregation"""
        return self._sql_aggregation("variance", axis)

    def std(self, axis: int | None = None) -> float | np.ndarray:
        """Compute standard deviation along axis"""
        return self._sql_aggregation("stddev", axis)

    def toarray(self) -> np.ndarray:
        """Convert to dense numpy array"""
        matrix = self.compute()
        return matrix.toarray()

    def compute(self, fragments: bool | None = None) -> scipy.sparse.csr_matrix:
        """
        Explicitly compute the matrix with all transformations applied.
        This is where the actual SQL query and materialization happens.

        Args:
            fragments: Whether to use fragment processing (None for automatic)

        Returns:
            Sparse matrix with transformations applied
        """
        # Determine processing strategy
        if fragments is not None:
            use_fragments = fragments
        else:
            # Check if dataset has multiple fragments
            try:
                fragments_list = self.slaf_array.expression.get_fragments()
                use_fragments = len(fragments_list) > 1
            except Exception:
                use_fragments = False

        # Fragment processing now supports both expression and layers tables
        if use_fragments:
            processor = FragmentProcessor(
                self.slaf_array,
                cell_selector=self._cell_selector,
                gene_selector=self._gene_selector,
                max_workers=4,
                enable_caching=True,
                table_name=self.table_name,
                layer_name=self.layer_name,
            )
            # Use smart strategy selection for optimal performance
            lazy_pipeline = processor.build_lazy_pipeline_smart("compute_matrix")
            result = processor.compute(lazy_pipeline)
            return self._convert_to_sparse_matrix(result)
        else:
            return self._compute_global()

    def _compute_global(self) -> scipy.sparse.csr_matrix:
        """Compute the matrix globally (original implementation)"""
        # Build the SQL query with transformations
        if self.parent_adata is not None and hasattr(
            self.parent_adata, "_transformations"
        ):
            transformations = self.parent_adata._transformations
        else:
            transformations = {}

        # Try SQL-level transformations first
        if transformations:
            sql_result = self._apply_sql_transformations(
                self._cell_selector, self._gene_selector, transformations
            )
            if sql_result is not None:
                return self._reconstruct_sparse_matrix(
                    sql_result, self._cell_selector, self._gene_selector
                )

        # Fall back to base query + numpy transformations
        base_query = self._build_submatrix_sql(self._cell_selector, self._gene_selector)
        base_result = self.slaf_array.query(base_query)

        # Reconstruct base matrix
        base_matrix = self._reconstruct_sparse_matrix(
            base_result, self._cell_selector, self._gene_selector
        )

        # Apply transformations in numpy if needed
        if transformations:
            return self._apply_numpy_transformations(
                base_matrix,
                self._cell_selector,
                self._gene_selector,
                transformations,
            )

        return base_matrix

    def _convert_to_sparse_matrix(
        self, result_df: pl.DataFrame
    ) -> scipy.sparse.csr_matrix:
        """
        Convert fragment processing result to sparse matrix.

        Args:
            result_df: Polars DataFrame from fragment processing

        Returns:
            Sparse matrix representation
        """
        if len(result_df) == 0:
            # Return empty matrix with appropriate shape
            return scipy.sparse.csr_matrix(self.shape)

        # Convert to COO format for efficient sparse matrix construction
        rows = result_df["cell_integer_id"].to_numpy()
        cols = result_df["gene_integer_id"].to_numpy()
        data = result_df["value"].to_numpy()

        # Remap integer IDs to local coordinates if selectors are applied
        if self._cell_selector is not None:
            if isinstance(self._cell_selector, slice):
                start = self._cell_selector.start or 0
                if start > 0:
                    rows = rows - start
            elif isinstance(self._cell_selector, list | np.ndarray):
                # For list/array selectors, we need to create a mapping
                # But this is complex, so for now we'll assume the query already filtered correctly
                # and we just need to remap if it's a slice
                pass

        if self._gene_selector is not None:
            if isinstance(self._gene_selector, slice):
                start = self._gene_selector.start or 0
                if start > 0:
                    cols = cols - start
            elif isinstance(self._gene_selector, list | np.ndarray):
                # For list/array selectors, we need to create a mapping
                # But this is complex, so for now we'll assume the query already filtered correctly
                # and we just need to remap if it's a slice
                pass

        # Create sparse matrix
        return scipy.sparse.coo_matrix((data, (rows, cols)), shape=self.shape).tocsr()


class LazyDictionaryViewMixin:
    """
    Base mixin for dictionary-like views (layers, obs, var).

    Provides common dictionary interface methods that are identical across
    all view types: layers, obs columns, and var columns.
    """

    def keys(self) -> list[str]:
        """
        Return list of keys.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement keys()")

    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return key in self.keys()

    def __len__(self) -> int:
        """Number of keys"""
        return len(self.keys())

    def __iter__(self):
        """Iterate over keys"""
        return iter(self.keys())

    def _validate_name(self, key: str):
        """
        Validate name (alphanumeric + underscore, non-empty).

        Args:
            key: Name to validate

        Raises:
            ValueError: If name is empty or contains invalid characters
        """
        if not key:
            raise ValueError("Name cannot be empty")
        if not key.replace("_", "").isalnum():
            raise ValueError(
                f"Name '{key}' contains invalid characters. "
                "Only alphanumeric characters and underscores are allowed."
            )


class LazyLayersView(LazyDictionaryViewMixin):
    """
    Dictionary-like view of layers with lazy evaluation.

    LazyLayersView provides a dictionary-like interface for accessing AnnData layers
    stored in the layers.lance table. It supports reading layers as LazyExpressionMatrix
    objects and provides methods to list, check, and iterate over available layers.

    Key Features:
        - Dictionary-like interface: layers["name"], "name" in layers, len(layers)
        - Lazy evaluation: layers are accessed on-demand
        - Config.json consistency: reads from config for fast layer discovery
        - Backward compatibility: works with datasets without layers

    Examples:
        >>> # Access a layer
        >>> slaf_array = SLAFArray("data.slaf")
        >>> adata = LazyAnnData(slaf_array)
        >>> spliced = adata.layers["spliced"]
        >>> print(f"Layer shape: {spliced.shape}")
        Layer shape: (1000, 20000)

        >>> # List available layers
        >>> print(list(adata.layers.keys()))
        ['spliced', 'unspliced']

        >>> # Check if layer exists
        >>> assert "spliced" in adata.layers
        >>> assert "nonexistent" not in adata.layers
    """

    def __init__(self, lazy_adata: "LazyAnnData"):
        """
        Initialize LazyLayersView with LazyAnnData.

        Args:
            lazy_adata: LazyAnnData instance containing the single-cell data.
        """
        self.lazy_adata = lazy_adata
        self._slaf_array = lazy_adata.slaf

    def _get_layers_from_config(self) -> list[str]:
        """Get layer names from config.json (fast path)"""
        if self._slaf_array.layers is None:
            return []

        layers_config = self._slaf_array.config.get("layers", {})
        return layers_config.get("available", [])

    def _get_layers_from_table(self) -> list[str]:
        """Get layer names by querying layers table (fallback)"""
        if self._slaf_array.layers is None:
            return []

        try:
            # Query to get distinct layer column names
            # In wide format, we need to check which columns exist
            schema = self._slaf_array.layers.schema
            column_names = [field.name for field in schema]

            # Filter out cell_integer_id and gene_integer_id
            layer_names = [
                col
                for col in column_names
                if col not in ("cell_integer_id", "gene_integer_id")
            ]
            return layer_names
        except Exception:
            return []

    def keys(self) -> list[str]:
        """List all available layer names"""
        # Fast path: read from config.json
        config_layers = self._get_layers_from_config()

        if config_layers:
            # Verify consistency with table if possible
            table_layers = self._get_layers_from_table()
            if table_layers and set(config_layers) != set(table_layers):
                # Log warning but prefer config
                import warnings

                warnings.warn(
                    f"Layer names in config.json ({config_layers}) don't match "
                    f"layers table ({table_layers}). Using config.json values.",
                    UserWarning,
                    stacklevel=2,
                )
            return config_layers

        # Fallback: query table
        return self._get_layers_from_table()

    def __getitem__(self, key: str) -> LazyExpressionMatrix:
        """Get a layer as LazyExpressionMatrix"""
        # Validate layer exists
        if key not in self.keys():
            raise KeyError(f"Layer '{key}' not found")

        # Create LazyExpressionMatrix pointing to layers table
        layer_matrix = LazyExpressionMatrix(
            self._slaf_array, table_name="layers", layer_name=key
        )
        layer_matrix.parent_adata = self.lazy_adata

        # Propagate cell/gene selectors from parent LazyAnnData
        if hasattr(self.lazy_adata, "_cell_selector"):
            layer_matrix._cell_selector = self.lazy_adata._cell_selector
        if hasattr(self.lazy_adata, "_gene_selector"):
            layer_matrix._gene_selector = self.lazy_adata._gene_selector
        layer_matrix._update_shape()

        return layer_matrix

    def _is_immutable(self, key: str) -> bool:
        """Check if a layer is immutable (converted from h5ad)"""
        layers_config = self._slaf_array.config.get("layers", {})
        immutable_layers = layers_config.get("immutable", [])
        return key in immutable_layers

    def __setitem__(
        self, key: str, value: "LazyExpressionMatrix | scipy.sparse.spmatrix"
    ):
        """
        Create or update a layer (lazy write - requires commit()).

        Stores the assignment in a pending writes queue. The actual write to
        layers.lance happens when commit() is called. This allows batching
        multiple layer operations and ensures config.json consistency.

        Args:
            key: Layer name (must be alphanumeric + underscore, non-empty)
            value: LazyExpressionMatrix or scipy sparse matrix to assign.
                   Must have the same shape as adata.X.

        Raises:
            ValueError: If layer name is invalid, shape doesn't match X,
                       or trying to overwrite an immutable layer.
        """
        # Validate layer name
        self._validate_name(key)

        # Validate shape matches X
        if value.shape != self.lazy_adata.shape:
            raise ValueError(
                f"Layer shape {value.shape} doesn't match X shape {self.lazy_adata.shape}"
            )

        # Check if layer already exists and is immutable
        if key in self.keys():
            if self._is_immutable(key):
                raise ValueError(
                    f"Layer '{key}' is immutable (converted from h5ad) and cannot be overwritten"
                )

        # Convert to materialized sparse matrix if needed
        from_lazy = isinstance(value, LazyExpressionMatrix)
        if from_lazy:
            value = value.compute()  # Materialize the lazy matrix

        # Ensure it's a sparse matrix
        import scipy.sparse

        if not scipy.sparse.issparse(value):
            # Convert dense to sparse
            value = scipy.sparse.csr_matrix(value)

        # Write immediately (eager write)
        self._write_layer_immediate(key, value, from_lazy=from_lazy)

    def _write_layer_immediate(
        self, layer_name: str, layer_matrix, from_lazy: bool = False
    ):
        """
        Write layer immediately to layers.lance (eager write).

        This method handles the immediate write of a layer to the layers.lance table.
        It's designed for small datasets (~10k cells) where the entire layer fits in memory.

        Args:
            layer_name: Name of the layer
            layer_matrix: Sparse matrix to write
            from_lazy: If True, the matrix came from LazyExpressionMatrix.compute(),
                      so coo.row and coo.col are already integer IDs
        """
        # Ensure layers table exists (create if needed)
        layers_path = self._slaf_array._join_path(
            self._slaf_array.slaf_path,
            self._slaf_array.config.get("tables", {}).get("layers", "layers.lance"),
        )

        # Convert sparse matrix to COO
        coo = layer_matrix.tocoo()

        # Get cell and gene integer IDs
        if from_lazy:
            # Matrix came from LazyExpressionMatrix.compute(), so row/col are already integer IDs
            cell_integer_ids = coo.row
            gene_integer_ids = coo.col
        else:
            # Matrix came from external source, need to map row/col indices to integer IDs
            obs_df = self.lazy_adata.slaf.obs
            var_df = self.lazy_adata.slaf.var
            cell_integer_ids = obs_df["cell_integer_id"].to_numpy()[coo.row]
            gene_integer_ids = var_df["gene_integer_id"].to_numpy()[coo.col]

        # Optimize dtype
        values, value_pa_type = self._optimize_dtype_for_layer(coo.data)

        # Create PyArrow table with the new layer column
        layer_table = pa.table(
            {
                "cell_integer_id": pa.array(cell_integer_ids, type=pa.uint32()),
                "gene_integer_id": pa.array(gene_integer_ids, type=pa.uint16()),
                layer_name: pa.array(values, type=value_pa_type),
            }
        )

        # Check if layers table exists
        if self._slaf_array.layers is None:
            # Create new layers table from expression table structure
            self._create_layers_table_with_layer(layer_table, layer_name, layers_path)
        else:
            # Update existing layers table
            self._update_layers_table_with_layer(layer_table, layer_name, layers_path)

        # Update config.json atomically
        self._update_config_layers_list([layer_name], add=True)

    def _optimize_dtype_for_layer(self, data: np.ndarray) -> tuple[np.ndarray, str]:
        """
        Optimize dtype for layer values.

        If float32 data contains only integers within uint16 range, convert to uint16.
        Otherwise, use float32.

        Args:
            data: Array of layer values

        Returns:
            Tuple of (optimized values array, value type string)
        """
        if len(data) == 0:
            return np.array([], dtype=np.float32), "float32"

        # Sample data to determine dtype
        sample_size = min(10000, len(data))
        sample_data = data[:sample_size]

        # Check if data is integer or float
        is_integer = np.issubdtype(sample_data.dtype, np.integer)
        max_value = np.max(data)
        min_value = np.min(data)

        if is_integer and max_value <= 65535 and min_value >= 0:
            return data.astype(np.uint16), "uint16"
        elif not is_integer:
            # Check if float data contains only integer values
            rounded_data = np.round(data)
            is_integer_values = np.allclose(data, rounded_data, rtol=1e-10)

            if is_integer_values and max_value <= 65535 and min_value >= 0:
                return rounded_data.astype(np.uint16), "uint16"
            else:
                return data.astype(np.float32), "float32"
        else:
            return data.astype(np.float32), "float32"

    def _get_pyarrow_type(self, value_type: str) -> pa.DataType:
        """
        Get PyArrow data type for a given value type string.

        Args:
            value_type: "uint16" or "float32"

        Returns:
            PyArrow data type
        """
        if value_type == "uint16":
            return pa.uint16()
        elif value_type == "float32":
            return pa.float32()
        else:
            raise ValueError(f"Unsupported value type: {value_type}")

    def _create_layers_table_with_layer(
        self, layer_table: pa.Table, layer_name: str, layers_path: str
    ):
        """Create new layers table with a single layer."""
        import lance

        # Get base structure from expression table (all cell-gene pairs)
        expression_df = (
            pl.scan_pyarrow_dataset(self._slaf_array.expression)
            .select(["cell_integer_id", "gene_integer_id"])
            .unique()
            .collect()
        )

        # Convert to PyArrow
        base_table = expression_df.to_arrow()

        # Join with new layer data
        layer_df = pl.from_arrow(layer_table)
        base_df = pl.from_arrow(base_table)

        # Left join to add layer column (nullable for sparse data)
        combined_df = base_df.join(
            layer_df, on=["cell_integer_id", "gene_integer_id"], how="left"
        )

        # Write new layers table
        lance.write_dataset(
            combined_df.to_arrow(),
            layers_path,
            mode="create",
            max_rows_per_file=10000000,
        )

        # Reload layers dataset
        self._slaf_array.layers = lance.dataset(layers_path)

    def _update_layers_table_with_layer(
        self, layer_table: pa.Table, layer_name: str, layers_path: str
    ):
        """Update existing layers table with a new/updated layer using add_columns() with UDF."""
        import lance

        layers_dataset = self._slaf_array.layers

        # Check if layer column already exists
        schema = layers_dataset.schema
        column_names = [field.name for field in schema]

        # Drop the old column if it exists (using Lance native method - metadata-only, very fast)
        if layer_name in column_names:
            layers_dataset = layers_dataset.drop_columns([layer_name])
            # Reload from path after drop_columns to get the updated dataset
            layers_dataset = lance.dataset(layers_path)

        # Convert layer data to polars for efficient lookup in UDF
        layer_df = pl.from_arrow(layer_table)

        # Create UDF that returns just the new column data
        # Note: Pass function directly to add_columns(), not decorated
        def add_layer_column_udf(batch):
            """
            UDF to add layer column by joining batch with layer data.

            Receives batch with existing columns (cell_integer_id, gene_integer_id, etc.),
            returns RecordBatch with just the new layer column.

            This processes the dataset in batches, joining each batch with the
            new layer data to add the column efficiently without rewriting
            existing data.
            """
            # Convert batch to polars to access cell_integer_id and gene_integer_id
            batch_df = pl.from_arrow(batch)

            # Join with layer data (left join preserves all rows, nullable for sparse data)
            result_df = batch_df.join(
                layer_df, on=["cell_integer_id", "gene_integer_id"], how="left"
            )

            # Extract just the new layer column and convert to single array
            new_layer_chunked = result_df.select([layer_name]).to_arrow().column(0)
            new_layer_array = new_layer_chunked.combine_chunks()

            # Return RecordBatch with just the new column (names must match column name)
            return pa.RecordBatch.from_arrays(
                [new_layer_array],
                names=[layer_name],
            )

        # Add the column using add_columns() with UDF
        # This processes in batches and doesn't rewrite existing data
        layers_dataset.add_columns(add_layer_column_udf)

        # Reload layers dataset from path to ensure consistency
        self._slaf_array.layers = lance.dataset(layers_path)

    def _update_config_layers_list(self, layer_names: list[str], add: bool):
        """
        Update config.json to add or remove layers from available/mutable lists.

        Args:
            layer_names: List of layer names to add or remove
            add: If True, add layers; if False, remove layers
        """
        config_path = self._slaf_array._join_path(
            self._slaf_array.slaf_path, "config.json"
        )

        # Load existing config
        with self._slaf_array._open_file(config_path) as f:
            import json

            config = json.load(f)

        # Ensure layers config exists
        if "layers" not in config:
            config["layers"] = {"available": [], "immutable": [], "mutable": []}

        # Ensure tables config includes layers
        if "tables" not in config:
            config["tables"] = {}
        if "layers" not in config["tables"]:
            config["tables"]["layers"] = "layers.lance"

        layers_config = config["layers"]
        available = set(layers_config.get("available", []))
        immutable = set(layers_config.get("immutable", []))
        mutable = set(layers_config.get("mutable", []))

        if add:
            # Add layers to available and mutable (new layers are mutable)
            for layer_name in layer_names:
                available.add(layer_name)
                mutable.add(layer_name)
                # Remove from immutable if it was there (shouldn't happen, but be safe)
                immutable.discard(layer_name)
        else:
            # Remove layers from all lists
            for layer_name in layer_names:
                available.discard(layer_name)
                mutable.discard(layer_name)
                immutable.discard(layer_name)

        # Update config
        layers_config["available"] = sorted(available)
        layers_config["immutable"] = sorted(immutable)
        layers_config["mutable"] = sorted(mutable)

        # Save updated config
        # Note: this will not work for huggingface remote
        with self._slaf_array._open_file(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def __delitem__(self, key: str):
        """
        Delete a layer (only if mutable).

        Args:
            key: Layer name to delete

        Raises:
            KeyError: If layer doesn't exist
            ValueError: If layer is immutable and cannot be deleted
        """
        import lance

        # Check if layer exists
        if key not in self.keys():
            raise KeyError(f"Layer '{key}' not found")

        # Check if layer is immutable
        if self._is_immutable(key):
            raise ValueError(
                f"Layer '{key}' is immutable (converted from h5ad) and cannot be deleted"
            )

        # Get layers path
        layers_path = self._slaf_array._join_path(
            self._slaf_array.slaf_path,
            self._slaf_array.config.get("tables", {}).get("layers", "layers.lance"),
        )

        # Drop the column using Lance's drop_columns() method (metadata-only, very fast)
        layers_dataset = self._slaf_array.layers
        layers_dataset = layers_dataset.drop_columns([key])

        # Reload layers dataset from path to ensure consistency
        self._slaf_array.layers = lance.dataset(layers_path)

        # Update config.json atomically
        self._update_config_layers_list([key], add=False)


class LazyMetadataViewMixin(LazySparseMixin, LazyDictionaryViewMixin):
    """
    Mixin class for metadata view operations (obs/var columns).

    This mixin provides shared functionality for LazyObsView and LazyVarView,
    eliminating code duplication. It handles:
    - Dictionary-like interface (keys, __getitem__, __setitem__, __delitem__)
    - Column management (create, update, delete)
    - Config.json synchronization
    - Selector support
    - Immutability tracking

    Required attributes (set by subclasses):
    - table_type: "obs" or "var"
    - table_name: "cells" or "genes"
    - id_column: "cell_integer_id" or "gene_integer_id"
    - lazy_adata: LazyAnnData instance
    - _slaf_array: SLAFArray instance
    - _shape: tuple[int, int] shape
    """

    # Type annotations for required attributes (set by subclasses)
    table_type: str
    table_name: str
    id_column: str
    lazy_adata: "LazyAnnData"
    _slaf_array: SLAFArray
    _shape: tuple[int, int]

    def _get_axis(self) -> int:
        """Get axis for this view (0 for obs/cells/obsm, 1 for var/genes/varm)"""
        return 0 if self.table_type in ("obs", "obsm") else 1

    def _get_current_selector(self) -> Any:
        """Get current selector from parent LazyAnnData"""
        # Map table_type to actual selector attribute names in LazyAnnData
        # obsm uses cell selector, varm uses gene selector
        selector_map = {
            "obs": "_cell_selector",
            "var": "_gene_selector",
            "obsm": "_cell_selector",
            "varm": "_gene_selector",
        }
        selector_attr = selector_map.get(self.table_type)
        if selector_attr:
            return getattr(self.lazy_adata, selector_attr, None)
        return None

    def _get_entity_count(self) -> int:
        """Get count of entities considering selectors"""
        selector = self._get_current_selector()
        axis = self._get_axis()
        if selector is None:
            return self._shape[axis]
        return self._get_selector_size(selector, axis)

    def _invalidate_metadata_cache(self):
        """
        Invalidate cached obs/var DataFrames when table structure changes.

        When we modify cells.lance or genes.lance (add/remove columns),
        the cached DataFrames in both LazyAnnData and SLAFArray become stale
        and need to be cleared so they reload from the updated tables.
        """
        # Invalidate LazyAnnData cache
        if hasattr(self.lazy_adata, "_obs"):
            self.lazy_adata._obs = None
        if hasattr(self.lazy_adata, "_var"):
            self.lazy_adata._var = None
        if hasattr(self.lazy_adata, "_cached_obs_names"):
            self.lazy_adata._cached_obs_names = None
        if hasattr(self.lazy_adata, "_cached_var_names"):
            self.lazy_adata._cached_var_names = None

        # Invalidate SLAFArray cache (which LazyAnnData.obs/var load from)
        # This ensures that when LazyAnnData.obs/var are accessed again,
        # they reload from the updated Lance tables
        if self.table_type in ("obs", "obsm"):
            # Modified cells.lance - invalidate SLAFArray._obs
            if hasattr(self._slaf_array, "_obs"):
                self._slaf_array._obs = None
            if hasattr(self._slaf_array, "_obs_columns"):
                self._slaf_array._obs_columns = None
            # Mark metadata as not loaded so it gets reloaded
            if hasattr(self._slaf_array, "_metadata_loaded"):
                self._slaf_array._metadata_loaded = False
        elif self.table_type in ("var", "varm"):
            # Modified genes.lance - invalidate SLAFArray._var
            if hasattr(self._slaf_array, "_var"):
                self._slaf_array._var = None
            if hasattr(self._slaf_array, "_var_columns"):
                self._slaf_array._var_columns = None
            # Mark metadata as not loaded so it gets reloaded
            if hasattr(self._slaf_array, "_metadata_loaded"):
                self._slaf_array._metadata_loaded = False

        # Invalidate view's cached DataFrame
        if hasattr(self, "_dataframe"):
            self._dataframe = None

    def _sql_condition_to_polars(self, sql_condition: str, id_column: str) -> pl.Expr:
        """Convert SQL WHERE condition to Polars expression"""
        if sql_condition == "TRUE":
            return pl.lit(True)
        if sql_condition == "FALSE":
            return pl.lit(False)

        # Handle range: "cell_integer_id >= 0 AND cell_integer_id < 100"
        if ">=" in sql_condition and "<" in sql_condition:
            parts = sql_condition.split(" AND ")
            ge_part = [p for p in parts if ">=" in p][0]
            lt_part = [p for p in parts if "<" in p][0]
            try:
                ge_value = int(ge_part.split(">=")[1].strip())
                lt_value = int(lt_part.split("<")[1].strip())
                return (pl.col(id_column) >= ge_value) & (pl.col(id_column) < lt_value)
            except ValueError:
                # If values are not integers, return False condition
                return pl.lit(False)

        # Handle IN clause: "cell_integer_id IN (0,1,2,3)"
        if " IN " in sql_condition:
            values_str = sql_condition.split(" IN ")[1].strip("()")
            # Filter out non-numeric values (like "False", "True", etc.)
            values = []
            for v in values_str.split(","):
                v = v.strip()
                try:
                    values.append(int(v))
                except ValueError:
                    # Skip non-integer values
                    continue
            if values:
                return pl.col(id_column).is_in(values)
            else:
                # If no valid values, return False condition
                return pl.lit(False)

        # Handle equality: "cell_integer_id = 5"
        if " = " in sql_condition:
            value = int(sql_condition.split(" = ")[1].strip())
            return pl.col(id_column) == value

        return pl.lit(True)  # Fallback: no filtering

    def _build_filtered_query(self, columns: list[str]) -> pl.LazyFrame:
        """Build filtered query for table with selectors"""
        table = getattr(self._slaf_array, self.table_name)
        query = pl.scan_pyarrow_dataset(table).select(columns)

        # Apply selector filtering using mixin utilities
        selector = self._get_current_selector()
        if selector is not None:
            # Convert selector to SQL condition, then to Polars filter
            axis = self._get_axis()
            entity_type = "cell" if self.table_type == "obs" else "gene"
            sql_condition = self._selector_to_sql_condition(
                selector, axis=axis, entity_type=entity_type
            )
            filter_expr = self._sql_condition_to_polars(sql_condition, self.id_column)
            query = query.filter(filter_expr)

        return query

    def _get_columns_from_config(self) -> list[str]:
        """Get column names from config.json (fast path)"""
        config = self._slaf_array.config
        if self.table_type in config and "available" in config[self.table_type]:
            return config[self.table_type]["available"]
        return []

    def _get_columns_from_table(self) -> list[str]:
        """Get column names by querying table (fallback)"""
        table = getattr(self._slaf_array, self.table_name, None)
        if table is None:
            return []

        try:
            schema = table.schema
            column_names = [field.name for field in schema]
            # Filter out system/internal columns:
            # - {id_column}: internal integer ID (not user-facing)
            # - {table_name}_start_index: Lance internal column (if present)
            system_cols = {self.id_column}
            # Handle both "cells" -> "cell_start_index" and "genes" -> "gene_start_index"
            if self.table_name == "cells":
                system_cols.add("cell_start_index")
            elif self.table_name == "genes":
                system_cols.add("gene_start_index")
            return [col for col in column_names if col not in system_cols]
        except Exception:
            return []

    def keys(self) -> list[str]:
        """List all available column/vector names"""
        # Route to vector or scalar method based on table_type
        if self.table_type in ("obsm", "varm"):
            return self._keys_vector()
        else:
            # Scalar column keys
            # Fast path: read from config.json
            config_columns = self._get_columns_from_config()

            if config_columns:
                # Verify consistency with table if possible
                table_columns = self._get_columns_from_table()
                if table_columns:
                    config_set = set(config_columns)
                    table_set = set(table_columns)

                    # Check for columns in config but not in table (real problem)
                    missing_in_table = config_set - table_set
                    if missing_in_table:
                        import warnings

                        warnings.warn(
                            f"Column names in config.json ({sorted(missing_in_table)}) "
                            f"not found in {self.table_name} table. These will be ignored.",
                            UserWarning,
                            stacklevel=2,
                        )

                    # Auto-sync: add columns from table that are missing in config
                    missing_in_config = table_set - config_set
                    if missing_in_config:
                        # Add missing columns to config (treat as immutable if they existed before)
                        self._sync_missing_columns_to_config(list(missing_in_config))
                        # Return updated config columns
                        config_columns = self._get_columns_from_config()

                return config_columns

            # Fallback: query table
            return self._get_columns_from_table()

    def __getitem__(self, key):
        """
        Get column/vector or DataFrame slice (AnnData-compatible DataFrame interface).

        - String key: Returns pandas Series (DataFrame-like behavior, matches AnnData)
        - Other keys (slice, list, etc.): Delegates to DataFrame indexing (DataFrame-like behavior)

        Args:
            key: Column name (str) or DataFrame indexer (slice, list, etc.)

        Returns:
            pandas Series if key is string, otherwise DataFrame slice
        """
        # Route to vector or scalar method based on table_type
        if self.table_type in ("obsm", "varm"):
            # For obsm/varm, only support string keys (dict-like)
            if not isinstance(key, str):
                raise TypeError(
                    f"{self.table_type} only supports string keys, got {type(key)}"
                )
            return self._get_vector_item(key)
        else:
            # For obs/var, use DataFrame-like access (AnnData-compatible)
            # Check if key exists first for better error messages
            if isinstance(key, str) and key not in self.keys():
                raise KeyError(f"Column '{key}' not found")
            # Always delegate to underlying DataFrame to get Series for string keys
            return self._get_dataframe().__getitem__(key)

    def _is_immutable(self, key: str) -> bool:
        """Check if column/vector key is immutable (converted from h5ad)"""
        # Route to vector or scalar method based on table_type
        if self.table_type in ("obsm", "varm"):
            return self._is_immutable_vector(key)
        else:
            # Scalar column immutability check
            config = self._slaf_array.config
            if self.table_type in config and "immutable" in config[self.table_type]:
                return key in config[self.table_type]["immutable"]
            return False

    def _optimize_dtype_for_column(
        self, data: np.ndarray
    ) -> tuple[np.ndarray, pa.DataType]:
        """
        Optimize dtype for column values.

        If float32 data contains only integers within uint16 range, convert to uint16.
        Otherwise, use float32.

        Args:
            data: Array of column values

        Returns:
            Tuple of (optimized values array, PyArrow data type)
        """
        if len(data) == 0:
            return np.array([], dtype=np.float32), pa.float32()

        # Handle string arrays
        if data.dtype.kind in [
            "U",
            "S",
            "O",
        ]:  # Unicode, byte string, or object (often strings)
            # Convert to string array and use PyArrow string type
            if data.dtype.kind == "O":
                # Object array - try to convert to string
                try:
                    data = np.array([str(x) for x in data], dtype="U")
                except (TypeError, ValueError):
                    # If conversion fails, keep as object
                    return data, pa.string()
            # Convert to UTF-8 string array
            return data.astype("U"), pa.string()

        # Sample data to determine dtype
        sample_size = min(10000, len(data))
        sample_data = data[:sample_size]

        # Check if data is integer or float
        is_integer = np.issubdtype(sample_data.dtype, np.integer)
        max_value = np.max(data)
        min_value = np.min(data)

        if is_integer and max_value <= 65535 and min_value >= 0:
            return data.astype(np.uint16), pa.uint16()
        elif not is_integer:
            # Check if float data contains only integer values
            rounded_data = np.round(data)
            is_integer_values = np.allclose(data, rounded_data, rtol=1e-10)

            if is_integer_values and max_value <= 65535 and min_value >= 0:
                return rounded_data.astype(np.uint16), pa.uint16()
            else:
                return data.astype(np.float32), pa.float32()
        else:
            return data.astype(np.float32), pa.float32()

    def __setitem__(self, key, value):
        """
        Create or update a column/vector or DataFrame slice (dual interface).

        - String key: Column assignment (dict-like behavior)
        - Other keys: DataFrame assignment (DataFrame-like behavior)

        Args:
            key: Column name (str) or DataFrame indexer
            value: numpy array, pandas Series, or DataFrame slice value
        """
        # Route to vector or scalar method based on table_type
        if self.table_type in ("obsm", "varm"):
            # For obsm/varm, only support string keys (dict-like)
            if not isinstance(key, str):
                raise TypeError(
                    f"{self.table_type} only supports string keys, got {type(key)}"
                )
            # Convert to numpy array if needed
            if isinstance(value, pd.Series):
                value = value.values
            value = np.asarray(value)
            # For vectors, value should be 2D (n_entities, n_dims)
            if len(value.shape) == 1:
                # If 1D, treat as single dimension vector
                value = value.reshape(-1, 1)
            self._set_vector_item(key, value)
        else:
            # For obs/var, support both dict-like and DataFrame-like assignment
            if isinstance(key, str):
                # Dict-like: column assignment
                self._set_column_item(key, value)
            else:
                # DataFrame-like: delegate to underlying DataFrame
                # Note: This will modify the DataFrame but won't persist to Lance
                # For now, we'll raise an error to guide users to use string keys for mutations
                raise NotImplementedError(
                    f"DataFrame-like assignment (e.g., obs[{key}] = ...) is not supported. "
                    f"Use column assignment (e.g., obs['{key}'] = ...) for mutations."
                )

    def _set_column_item(self, key: str, value: np.ndarray | pd.Series):
        """
        Create or update a column (eager write - immediate).

        Args:
            key: Column name (must be alphanumeric + underscore, non-empty)
            value: numpy array or pandas Series to assign.
                   Must have length matching entity count (considering selectors).

        Raises:
            ValueError: If column name is invalid, length doesn't match,
                       or trying to overwrite an immutable column.
        """
        # Validate column name
        self._validate_name(key)

        # Validate shape matches entity count (considering selectors)
        expected_count = self._get_entity_count()
        entity_name = self.table_type  # "obs" or "var"
        if len(value) != expected_count:
            raise ValueError(
                f"Column length {len(value)} doesn't match {entity_name} count {expected_count}"
            )

        # Check if column already exists and is immutable
        if key in self.keys() and self._is_immutable(key):
            raise ValueError(
                f"Column '{key}' is immutable (converted from h5ad) and cannot be overwritten"
            )

        # Convert to numpy array
        if isinstance(value, pd.Series):
            value = value.values
        value = np.asarray(value)

        # Optimize dtype
        values, value_pa_type = self._optimize_dtype_for_column(value)

        # Get integer IDs in order (respecting selectors)
        selector = self._get_current_selector()
        table = getattr(self._slaf_array, self.table_name)
        if selector is None:
            # No selector - use all entities
            df = table.to_table().to_pandas()
            integer_ids = df[self.id_column].values
        else:
            # Apply selector to get integer IDs
            query = self._build_filtered_query([self.id_column])
            df = query.collect().sort(self.id_column)
            integer_ids = df[self.id_column].to_numpy()

        # Determine ID column type (uint32 for cells, uint16 for genes)
        id_pa_type = pa.uint32() if self.table_type == "obs" else pa.uint16()

        # Create PyArrow table with the new column
        column_table = pa.table(
            {
                self.id_column: pa.array(integer_ids, type=id_pa_type),
                key: pa.array(values, type=value_pa_type),
            }
        )

        # Write to table
        table_path = self._slaf_array._join_path(
            self._slaf_array.slaf_path,
            self._slaf_array.config.get("tables", {}).get(
                self.table_name, f"{self.table_name}.lance"
            ),
        )

        if table is None:
            raise ValueError(f"{self.table_name}.lance table not found")

        # Update table (similar to layers update logic)
        self._update_table_with_column(column_table, key, table_path)

        # Update config.json atomically
        self._update_config_columns_list([key], add=True)

        # Reload config to ensure it's up-to-date (config is cached in SLAFArray)
        config_path = self._slaf_array._join_path(
            self._slaf_array.slaf_path, "config.json"
        )
        with self._slaf_array._open_file(config_path) as f:
            import json

            self._slaf_array.config = json.load(f)

        # Invalidate cached obs/var DataFrames since table structure changed
        self._invalidate_metadata_cache()

    def _update_table_with_column(
        self, column_table: pa.Table, column_name: str, table_path: str
    ):
        """Update existing table with a new/updated column using add_columns() with UDF."""
        import lance

        table_dataset = getattr(self._slaf_array, self.table_name)

        # Check if column already exists
        schema = table_dataset.schema
        column_names = [field.name for field in schema]

        # Drop the old column if it exists (using Lance native method - metadata-only, very fast)
        if column_name in column_names:
            table_dataset = table_dataset.drop_columns([column_name])
            # Reload from path after drop_columns to get the updated dataset
            table_dataset = lance.dataset(table_path)

        # Convert column data to polars for efficient lookup in UDF
        column_df = pl.from_arrow(column_table)

        # Create UDF that returns just the new column data
        def add_column_udf(batch):
            """
            UDF to add column by joining batch with column data.

            Receives batch with existing columns (cell_integer_id, etc.),
            returns RecordBatch with just the new column.
            """
            # Convert batch to polars
            batch_df = pl.from_arrow(batch)

            # Join with column data (left join preserves all rows)
            result_df = batch_df.join(column_df, on=[self.id_column], how="left")

            # Extract just the new column and convert to single array
            new_column_chunked = result_df.select([column_name]).to_arrow().column(0)
            new_column_array = new_column_chunked.combine_chunks()

            # Return RecordBatch with just the new column
            return pa.RecordBatch.from_arrays(
                [new_column_array],
                names=[column_name],
            )

        # Add the column using add_columns() with UDF
        table_dataset.add_columns(add_column_udf)

        # Reload table dataset from path to ensure consistency
        setattr(self._slaf_array, self.table_name, lance.dataset(table_path))

    def _update_config_columns_list(self, column_names: list[str], add: bool):
        """
        Update config.json to add or remove columns from available/mutable lists.

        Args:
            column_names: List of column names to add or remove
            add: If True, add columns; if False, remove columns
        """
        config_path = self._slaf_array._join_path(
            self._slaf_array.slaf_path, "config.json"
        )

        # Load existing config
        with self._slaf_array._open_file(config_path) as f:
            import json

            config = json.load(f)

        # Ensure table config exists
        if self.table_type not in config:
            config[self.table_type] = {"available": [], "immutable": [], "mutable": []}

        table_config = config[self.table_type]
        available = set(table_config.get("available", []))
        immutable = set(table_config.get("immutable", []))
        mutable = set(table_config.get("mutable", []))

        if add:
            # Add columns to available and mutable (new columns are mutable)
            for column_name in column_names:
                available.add(column_name)
                mutable.add(column_name)
                # Remove from immutable if it was there (shouldn't happen, but be safe)
                immutable.discard(column_name)
        else:
            # Remove columns from all lists
            for column_name in column_names:
                available.discard(column_name)
                mutable.discard(column_name)
                immutable.discard(column_name)

        # Update config
        table_config["available"] = sorted(available)
        table_config["immutable"] = sorted(immutable)
        table_config["mutable"] = sorted(mutable)

        # Save updated config
        with self._slaf_array._open_file(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _sync_missing_columns_to_config(self, column_names: list[str]):
        """
        Sync missing columns from table to config.json.
        These columns are treated as immutable (they existed before Phase 6.5).

        Args:
            column_names: List of column names to add to config
        """
        if not column_names:
            return

        config_path = self._slaf_array._join_path(
            self._slaf_array.slaf_path, "config.json"
        )

        # Load existing config
        with self._slaf_array._open_file(config_path) as f:
            import json

            config = json.load(f)

        # Ensure table config exists
        if self.table_type not in config:
            config[self.table_type] = {"available": [], "immutable": [], "mutable": []}

        table_config = config[self.table_type]
        available = set(table_config.get("available", []))
        immutable = set(table_config.get("immutable", []))
        mutable = set(table_config.get("mutable", []))

        # Add missing columns to available and immutable (they existed before)
        for column_name in column_names:
            available.add(column_name)
            immutable.add(column_name)
            # Remove from mutable if it was there (shouldn't happen, but be safe)
            mutable.discard(column_name)

        # Update config
        table_config["available"] = sorted(available)
        table_config["immutable"] = sorted(immutable)
        table_config["mutable"] = sorted(mutable)

        # Save updated config
        with self._slaf_array._open_file(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def __delitem__(self, key: str):
        """
        Delete a column/vector (only if mutable).

        Args:
            key: Column/vector name to delete

        Raises:
            KeyError: If column/vector doesn't exist
            ValueError: If column/vector is immutable and cannot be deleted
        """
        # Route to vector or scalar method based on table_type
        if self.table_type in ("obsm", "varm"):
            self._del_vector_item(key)
        else:
            # Scalar column deletion
            import lance

            # Check if column exists
            if key not in self.keys():
                raise KeyError(f"Column '{key}' not found")

            # Check if column is immutable
            if self._is_immutable(key):
                raise ValueError(
                    f"Column '{key}' is immutable (converted from h5ad) and cannot be deleted"
                )

            # Get table path
            table_path = self._slaf_array._join_path(
                self._slaf_array.slaf_path,
                self._slaf_array.config.get("tables", {}).get(
                    self.table_name, f"{self.table_name}.lance"
                ),
            )

            # Drop the column using Lance's drop_columns() method (metadata-only, very fast)
            table_dataset = getattr(self._slaf_array, self.table_name)
            table_dataset = table_dataset.drop_columns([key])

            # Reload table dataset from path to ensure consistency
            setattr(self._slaf_array, self.table_name, lance.dataset(table_path))

            # Update config.json atomically
            self._update_config_columns_list([key], add=False)

            # Reload config to ensure it's up-to-date (config is cached in SLAFArray)
            config_path = self._slaf_array._join_path(
                self._slaf_array.slaf_path, "config.json"
            )
            with self._slaf_array._open_file(config_path) as f:
                import json

                self._slaf_array.config = json.load(f)

            # Invalidate cached obs/var DataFrames since table structure changed
            self._invalidate_metadata_cache()

    # ==================== Vector-specific methods (for obsm/varm) ====================

    def _detect_vector_columns(self) -> dict[str, int]:
        """
        Detect FixedSizeListArray columns from schema and return key -> dimension mapping.

        Returns:
            Dictionary mapping vector key names to their dimensions
        """
        table = getattr(self._slaf_array, self.table_name, None)
        if table is None:
            return {}

        vector_columns = {}
        schema = table.schema

        for field in schema:
            # Check if field is a FixedSizeListArray (vector type)
            if isinstance(field.type, pa.FixedSizeListType):
                # Use column name directly as the key
                key = field.name
                # Get dimension from FixedSizeListType
                n_dims = field.type.list_size
                vector_columns[key] = n_dims

        return vector_columns

    def _get_vector_item(self, key: str) -> np.ndarray:
        """Retrieve multi-dimensional array (respects selectors from parent)"""
        if key not in self.keys():
            raise KeyError(f"{self.table_type} key '{key}' not found")

        # Build query for the vector column (using mixin for filtering)
        query = self._build_filtered_query([self.id_column, key])
        df = query.collect()

        # Sort by integer ID
        df = df.sort(self.id_column)

        # Extract vector column and convert to numpy array
        # FixedSizeListArray columns are stored as lists/arrays
        vector_data = df[key].to_numpy()

        # Convert list of arrays to 2D numpy array
        if len(vector_data) > 0 and isinstance(vector_data[0], list | np.ndarray):
            return np.array([np.array(v) for v in vector_data])
        else:
            # Already in correct format
            return np.asarray(vector_data)

    def _set_vector_item(self, key: str, value: np.ndarray):
        """Store multi-dimensional array as FixedSizeListArray column"""
        import lance
        import pyarrow as pa

        # Validate shape using mixin utilities (respects selectors)
        expected_count = self._get_entity_count()
        if value.shape[0] != expected_count:
            raise ValueError(
                f"Array first dimension {value.shape[0]} doesn't match {self.table_type} count {expected_count}"
            )

        # Validate key name
        self._validate_name(key)

        # Check immutability
        if key in self.keys() and self._is_immutable(key):
            raise ValueError(
                f"{self.table_type} key '{key}' is immutable and cannot be overwritten"
            )

        # Convert to numpy array
        value = np.asarray(value)
        n_dims = value.shape[1] if len(value.shape) > 1 else 1

        # Get integer IDs in order (respecting selectors)
        selector = self._get_current_selector()
        table = getattr(self._slaf_array, self.table_name)
        if selector is None:
            # No selector - use all entities
            df = table.to_table().to_pandas()
            integer_ids = df[self.id_column].values
        else:
            # Apply selector to get integer IDs
            query = self._build_filtered_query([self.id_column])
            df = query.collect().sort(self.id_column)
            integer_ids = df[self.id_column].to_numpy()

        # Create FixedSizeListArray directly from numpy array
        # Determine value dtype (use float32 for embeddings)
        value_dtype = pa.float32() if value.dtype.kind == "f" else pa.float32()

        # Flatten the 2D array and create FixedSizeListArray
        # FixedSizeListArray.from_arrays takes a flat array and list_size
        flat_values = pa.array(value.flatten(), type=value_dtype)
        vector_array = pa.FixedSizeListArray.from_arrays(flat_values, n_dims)

        # If overwriting, drop old column first
        if key in self.keys():
            table_dataset = getattr(self._slaf_array, self.table_name)
            table_dataset = table_dataset.drop_columns([key])
            table_path = self._slaf_array._join_path(
                self._slaf_array.slaf_path,
                self._slaf_array.config.get("tables", {}).get(
                    self.table_name, f"{self.table_name}.lance"
                ),
            )
            setattr(self._slaf_array, self.table_name, lance.dataset(table_path))

        # Create table with id_column and vector column
        table_path = self._slaf_array._join_path(
            self._slaf_array.slaf_path,
            self._slaf_array.config.get("tables", {}).get(
                self.table_name, f"{self.table_name}.lance"
            ),
        )

        # Determine ID column type (uint32 for cells, uint16 for genes)
        id_pa_type = pa.uint32() if self.table_type == "obsm" else pa.uint16()

        # Create table with integer IDs and vector column
        column_table = pa.table(
            {
                self.id_column: pa.array(integer_ids, type=id_pa_type),
                key: vector_array,
            }
        )

        # Update table
        self._update_table_with_column(column_table, key, table_path)

        # Update config.json
        self._update_config_vector_list([key], add=True, n_dims=n_dims)

        # Reload config to ensure it's up-to-date (config is cached in SLAFArray)
        config_path = self._slaf_array._join_path(
            self._slaf_array.slaf_path, "config.json"
        )
        with self._slaf_array._open_file(config_path) as f:
            import json

            self._slaf_array.config = json.load(f)

        # Invalidate cached obs/var DataFrames since table structure changed
        self._invalidate_metadata_cache()

    def _del_vector_item(self, key: str):
        """Delete vector key (drops the vector column)"""
        import lance

        if key not in self.keys():
            raise KeyError(f"{self.table_type} key '{key}' not found")

        if self._is_immutable(key):
            raise ValueError(
                f"{self.table_type} key '{key}' is immutable and cannot be deleted"
            )

        # Drop the vector column
        table_dataset = getattr(self._slaf_array, self.table_name)
        table_dataset = table_dataset.drop_columns([key])

        # Reload dataset
        table_path = self._slaf_array._join_path(
            self._slaf_array.slaf_path,
            self._slaf_array.config.get("tables", {}).get(
                self.table_name, f"{self.table_name}.lance"
            ),
        )
        setattr(self._slaf_array, self.table_name, lance.dataset(table_path))

        # Update config.json
        self._update_config_vector_list([key], add=False)

        # Reload config to ensure it's up-to-date (config is cached in SLAFArray)
        config_path = self._slaf_array._join_path(
            self._slaf_array.slaf_path, "config.json"
        )
        with self._slaf_array._open_file(config_path) as f:
            import json

            self._slaf_array.config = json.load(f)

        # Invalidate cached obs/var DataFrames since table structure changed
        self._invalidate_metadata_cache()

    def _keys_vector(self) -> list[str]:
        """List all available vector keys by detecting FixedSizeListArray columns"""
        # Fast path: read from config.json
        config = self._slaf_array.config
        if self.table_type in config and "available" in config[self.table_type]:
            config_keys = config[self.table_type]["available"]
            # Verify against schema (auto-sync if needed)
            schema_keys = list(self._detect_vector_columns().keys())
            if set(config_keys) != set(schema_keys):
                # Auto-sync: update config with schema keys
                missing_in_config = set(schema_keys) - set(config_keys)
                if missing_in_config:
                    # Add missing keys to config (treat as immutable if they existed before)
                    self._update_config_vector_list(list(missing_in_config), add=True)
                    # Re-read config
                    config = self._slaf_array.config
                    return config.get(self.table_type, {}).get("available", [])
            return config_keys

        # Fallback: detect from schema
        return list(self._detect_vector_columns().keys())

    def _is_immutable_vector(self, key: str) -> bool:
        """Check if vector key is immutable"""
        config = self._slaf_array.config
        if self.table_type in config and "immutable" in config[self.table_type]:
            return key in config[self.table_type]["immutable"]
        return False

    def _update_config_vector_list(
        self, keys: list[str], add: bool, n_dims: int | None = None
    ):
        """Update config.json to add or remove vector keys (unified for obsm/varm)"""
        config_path = self._slaf_array._join_path(
            self._slaf_array.slaf_path, "config.json"
        )

        # Load existing config
        with self._slaf_array._open_file(config_path) as f:
            import json

            config = json.load(f)

        # Ensure vector config exists
        if self.table_type not in config:
            config[self.table_type] = {
                "available": [],
                "immutable": [],
                "mutable": [],
                "dimensions": {},
            }

        vector_config = config[self.table_type]
        available = set(vector_config.get("available", []))
        immutable = set(vector_config.get("immutable", []))
        mutable = set(vector_config.get("mutable", []))
        dimensions = vector_config.get("dimensions", {})

        if add:
            # Add keys to available and mutable (new keys are mutable)
            for key in keys:
                available.add(key)
                mutable.add(key)
                immutable.discard(key)
                if n_dims is not None:
                    dimensions[key] = n_dims
        else:
            # Remove keys from all lists
            for key in keys:
                available.discard(key)
                mutable.discard(key)
                immutable.discard(key)
                dimensions.pop(key, None)

        # Update config
        vector_config["available"] = sorted(available)
        vector_config["immutable"] = sorted(immutable)
        vector_config["mutable"] = sorted(mutable)
        vector_config["dimensions"] = dimensions

        # Save updated config
        with self._slaf_array._open_file(config_path, "w") as f:
            json.dump(config, f, indent=2)


class LazyObsView(LazyMetadataViewMixin):
    """
    Dual-interface view of obs columns: DataFrame-like and dictionary-like.

    LazyObsView provides both DataFrame-like and dictionary-like interfaces for accessing
    and mutating cell metadata columns stored in the cells.lance table.

    Key Features:
        - DataFrame-like interface: obs.columns, obs.head(), obs[slice] (AnnData-compatible)
        - AnnData-compatible access: obs["col"] returns pd.Series (not np.ndarray)
        - Dictionary-like interface: "col" in obs, len(obs), obs.keys()
        - Lazy evaluation: columns are accessed on-demand
        - Selector support: respects cell selectors from parent LazyAnnData
        - Immutability: prevents deletion/modification of converted columns
        - Config.json consistency: reads from config for fast column discovery

    Examples:
        >>> # DataFrame-like access (AnnData-compatible)
        >>> slaf_array = SLAFArray("data.slaf")
        >>> adata = LazyAnnData(slaf_array)
        >>> df = adata.obs  # Returns DataFrame-like view
        >>> print(df.columns)  # DataFrame columns
        >>> print(df.head())  # DataFrame methods work

        >>> # AnnData-compatible column access (returns Series)
        >>> cluster = adata.obs["cluster"]  # Returns pd.Series (AnnData-compatible)
        >>> print(type(cluster))
        <class 'pandas.core.series.Series'>

        >>> # Create a new column
        >>> adata.obs["new_cluster"] = new_cluster_labels
        >>> assert "new_cluster" in adata.obs

        >>> # List available columns
        >>> print(list(adata.obs.keys()))
        ['cell_id', 'total_counts', 'cluster', 'new_cluster']
    """

    def __init__(self, lazy_adata: "LazyAnnData"):
        """
        Initialize LazyObsView with LazyAnnData.

        Args:
            lazy_adata: LazyAnnData instance containing the single-cell data.
        """
        super().__init__()
        self.lazy_adata = lazy_adata
        self._slaf_array = lazy_adata.slaf
        # Required by LazySparseMixin
        self.slaf_array = lazy_adata.slaf
        self._shape = lazy_adata.shape  # (n_cells, n_genes)
        self.table_name = "cells"
        self.id_column = "cell_integer_id"
        self.table_type = "obs"
        self._dataframe: pd.DataFrame | None = None  # Cached DataFrame

    @property
    def shape(self) -> tuple[int, int]:
        """Required by LazySparseMixin - uses parent's shape"""
        return self._shape

    def __len__(self) -> int:
        """
        Return number of rows (DataFrame-like behavior).

        For DataFrame compatibility, len(obs) should return the number of rows,
        not the number of columns (which is what the dictionary interface would return).
        """
        return len(self._get_dataframe())

    def _get_dataframe(self) -> pd.DataFrame:
        """
        Get underlying DataFrame (lazy-loaded, respects selectors).

        Returns:
            pandas DataFrame with all columns from cells.lance (excluding vector columns)
        """
        # If parent has filtered_obs function, use it instead
        if (
            hasattr(self.lazy_adata, "_filtered_obs")
            and self.lazy_adata._filtered_obs is not None
        ):
            return self.lazy_adata._filtered_obs()

        if self._dataframe is None:
            # Get all column names from table schema
            table = getattr(self._slaf_array, self.table_name)
            schema = table.schema
            all_columns = [field.name for field in schema]

            # Build DataFrame from cells.lance with selectors
            query = self._build_filtered_query(all_columns)
            df = query.collect()

            # Sort by integer ID to match order
            df = df.sort(self.id_column)

            # Convert to pandas DataFrame
            obs_df = df.to_pandas()

            # Drop system columns
            if self.id_column in obs_df.columns:
                obs_df = obs_df.drop(columns=[self.id_column])

            # Filter out vector columns (obsm) - these are FixedSizeListArray columns
            cells_table = getattr(self._slaf_array, self.table_name, None)
            if cells_table is not None:
                schema = cells_table.schema
                vector_column_names = {
                    field.name
                    for field in schema
                    if isinstance(field.type, pa.FixedSizeListType)
                }
                columns_to_drop = [
                    col for col in obs_df.columns if col in vector_column_names
                ]
                if columns_to_drop:
                    obs_df = obs_df.drop(columns=columns_to_drop)

            # Set cell_id as index if present
            if "cell_id" in obs_df.columns:
                obs_df = obs_df.set_index("cell_id")
                obs_df.index.name = "cell_id"

            self._dataframe = obs_df

        return self._dataframe

    def __getattr__(self, name: str):
        """
        Delegate DataFrame attributes to underlying DataFrame.

        This allows obs to behave like a DataFrame when accessed directly,
        e.g., obs.columns, obs.head(), obs.shape, etc.
        """
        # Don't delegate special methods or our own methods
        if name.startswith("_") or name in dir(self):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Delegate to underlying DataFrame
        try:
            return getattr(self._get_dataframe(), name)
        except AttributeError as err:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from err


class LazyVarView(LazyMetadataViewMixin):
    """
    Dual-interface view of var columns: DataFrame-like and dictionary-like.

    LazyVarView provides both DataFrame-like and dictionary-like interfaces for accessing
    and mutating gene metadata columns stored in the genes.lance table.

    Key Features:
        - DataFrame-like interface: var.columns, var.head(), var[slice] (AnnData-compatible)
        - AnnData-compatible access: var["col"] returns pd.Series (not np.ndarray)
        - Dictionary-like interface: "col" in var, len(var), var.keys()
        - Lazy evaluation: columns are accessed on-demand
        - Selector support: respects gene selectors from parent LazyAnnData
        - Immutability: prevents deletion/modification of converted columns
        - Config.json consistency: reads from config for fast column discovery

    Examples:
        >>> # DataFrame-like access (AnnData-compatible)
        >>> slaf_array = SLAFArray("data.slaf")
        >>> adata = LazyAnnData(slaf_array)
        >>> df = adata.var  # Returns DataFrame-like view
        >>> print(df.columns)  # DataFrame columns
        >>> print(df.head())  # DataFrame methods work

        >>> # AnnData-compatible column access (returns Series)
        >>> hvg = adata.var["highly_variable"]  # Returns pd.Series (AnnData-compatible)
        >>> print(type(hvg))
        <class 'pandas.core.series.Series'>

        >>> # Create a new column
        >>> adata.var["new_annotation"] = new_annotations
        >>> assert "new_annotation" in adata.var
    """

    def __init__(self, lazy_adata: "LazyAnnData"):
        """
        Initialize LazyVarView with LazyAnnData.

        Args:
            lazy_adata: LazyAnnData instance containing the single-cell data.
        """
        super().__init__()
        self.lazy_adata = lazy_adata
        self._slaf_array = lazy_adata.slaf
        # Required by LazySparseMixin
        self.slaf_array = lazy_adata.slaf
        self._shape = lazy_adata.shape  # (n_cells, n_genes)
        self.table_name = "genes"
        self.id_column = "gene_integer_id"
        self.table_type = "var"
        self._dataframe: pd.DataFrame | None = None  # Cached DataFrame

    @property
    def shape(self) -> tuple[int, int]:
        """Required by LazySparseMixin - uses parent's shape"""
        return self._shape

    def __len__(self) -> int:
        """
        Return number of rows (DataFrame-like behavior).

        For DataFrame compatibility, len(var) should return the number of rows,
        not the number of columns (which is what the dictionary interface would return).
        """
        return len(self._get_dataframe())

    def _get_dataframe(self) -> pd.DataFrame:
        """
        Get underlying DataFrame (lazy-loaded, respects selectors).

        Returns:
            pandas DataFrame with all columns from genes.lance (excluding vector columns)
        """
        # If parent has filtered_var function, use it instead
        if (
            hasattr(self.lazy_adata, "_filtered_var")
            and self.lazy_adata._filtered_var is not None
        ):
            return self.lazy_adata._filtered_var()

        if self._dataframe is None:
            # Get all column names from table schema
            table = getattr(self._slaf_array, self.table_name)
            schema = table.schema
            all_columns = [field.name for field in schema]

            # Build DataFrame from genes.lance with selectors
            query = self._build_filtered_query(all_columns)
            df = query.collect()

            # Sort by integer ID to match order
            df = df.sort(self.id_column)

            # Convert to pandas DataFrame
            var_df = df.to_pandas()

            # Drop system columns
            if self.id_column in var_df.columns:
                var_df = var_df.drop(columns=[self.id_column])

            # Filter out vector columns (varm) - these are FixedSizeListArray columns
            genes_table = getattr(self._slaf_array, self.table_name, None)
            if genes_table is not None:
                schema = genes_table.schema
                vector_column_names = {
                    field.name
                    for field in schema
                    if isinstance(field.type, pa.FixedSizeListType)
                }
                columns_to_drop = [
                    col for col in var_df.columns if col in vector_column_names
                ]
                if columns_to_drop:
                    var_df = var_df.drop(columns=columns_to_drop)

            # Set gene_id as index if present
            if "gene_id" in var_df.columns:
                var_df = var_df.set_index("gene_id")
                var_df.index.name = "gene_id"

            self._dataframe = var_df

        return self._dataframe

    def __getattr__(self, name: str):
        """
        Delegate DataFrame attributes to underlying DataFrame.

        This allows var to behave like a DataFrame when accessed directly,
        e.g., var.columns, var.head(), var.shape, etc.
        """
        # Don't delegate special methods or our own methods
        if name.startswith("_") or name in dir(self):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Delegate to underlying DataFrame
        try:
            return getattr(self._get_dataframe(), name)
        except AttributeError as err:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from err


class LazyObsmView(LazyMetadataViewMixin):
    """
    Dictionary-like view of obsm (multi-dimensional obs annotations).

    LazyObsmView provides a dictionary-like interface for accessing and mutating
    multi-dimensional cell annotations (e.g., UMAP, PCA embeddings) stored as
    separate columns in the cells.lance table. Each key maps to a 2D numpy array.

    Key Features:
        - Dictionary-like interface: obsm["X_umap"], "X_umap" in obsm, len(obsm)
        - Multi-dimensional arrays: stored as FixedSizeListArray columns (native Lance vector type)
        - Schema-based detection: automatically detects vector columns from Lance schema
        - Selector support: respects cell selectors from parent LazyAnnData
        - Immutability: prevents deletion/modification of converted embeddings
        - Config.json consistency: reads from config for fast key discovery

    Examples:
        >>> # Access an embedding
        >>> slaf_array = SLAFArray("data.slaf")
        >>> adata = LazyAnnData(slaf_array)
        >>> umap = adata.obsm["X_umap"]
        >>> print(f"UMAP shape: {umap.shape}")
        UMAP shape: (1000, 2)

        >>> # Create a new embedding
        >>> adata.obsm["X_pca"] = pca_coords  # shape: (1000, 50)
        >>> assert "X_pca" in adata.obsm
    """

    def __init__(self, lazy_adata: "LazyAnnData"):
        """
        Initialize LazyObsmView with LazyAnnData.

        Args:
            lazy_adata: LazyAnnData instance containing the single-cell data.
        """
        super().__init__()
        self.lazy_adata = lazy_adata
        self._slaf_array = lazy_adata.slaf
        # Required by LazySparseMixin
        self.slaf_array = lazy_adata.slaf
        self._shape = lazy_adata.shape  # (n_cells, n_genes)
        self.table_name = "cells"
        self.id_column = "cell_integer_id"
        self.table_type = "obsm"

    @property
    def shape(self) -> tuple[int, int]:
        """Required by LazySparseMixin - uses parent's shape"""
        return self._shape


class LazyVarmView(LazyMetadataViewMixin):
    """
    Dictionary-like view of varm (multi-dimensional var annotations).

    LazyVarmView provides a dictionary-like interface for accessing and mutating
    multi-dimensional gene annotations (e.g., PCA loadings) stored as separate
    columns in the genes.lance table. Each key maps to a 2D numpy array.

    Key Features:
        - Dictionary-like interface: varm["PCs"], "PCs" in varm, len(varm)
        - Multi-dimensional arrays: stored as FixedSizeListArray columns (native Lance vector type)
        - Schema-based detection: automatically detects vector columns from Lance schema
        - Selector support: respects gene selectors from parent LazyAnnData
        - Immutability: prevents deletion/modification of converted embeddings
        - Config.json consistency: reads from config for fast key discovery

    Examples:
        >>> # Access gene loadings
        >>> slaf_array = SLAFArray("data.slaf")
        >>> adata = LazyAnnData(slaf_array)
        >>> pcs = adata.varm["PCs"]
        >>> print(f"PCs shape: {pcs.shape}")
        PCs shape: (20000, 50)
    """

    def __init__(self, lazy_adata: "LazyAnnData"):
        """
        Initialize LazyVarmView with LazyAnnData.

        Args:
            lazy_adata: LazyAnnData instance containing the single-cell data.
        """
        super().__init__()
        self.lazy_adata = lazy_adata
        self._slaf_array = lazy_adata.slaf
        # Required by LazySparseMixin
        self.slaf_array = lazy_adata.slaf
        self._shape = lazy_adata.shape  # (n_cells, n_genes)
        self.table_name = "genes"
        self.id_column = "gene_integer_id"
        self.table_type = "varm"

    @property
    def shape(self) -> tuple[int, int]:
        """Required by LazySparseMixin - uses parent's shape"""
        return self._shape


class LazyUnsView(LazyDictionaryViewMixin):
    """
    Dictionary-like view of uns (unstructured metadata).

    LazyUnsView provides a dictionary-like interface for accessing and mutating
    unstructured metadata stored in uns.json. Unlike obs/var/obsm/varm, uns
    metadata is always mutable and stored as JSON.

    Key Features:
        - Dictionary-like interface: uns["key"], "key" in uns, len(uns)
        - JSON storage: stored in uns.json file
        - Always mutable: no immutability tracking
        - JSON serialization: automatically converts numpy/pandas objects

    Examples:
        >>> # Store metadata
        >>> slaf_array = SLAFArray("data.slaf")
        >>> adata = LazyAnnData(slaf_array)
        >>> adata.uns["neighbors"] = {"params": {"n_neighbors": 15}}
        >>> print(adata.uns["neighbors"])
        {'params': {'n_neighbors': 15}}
    """

    def __init__(self, lazy_adata: "LazyAnnData"):
        """
        Initialize LazyUnsView with LazyAnnData.

        Args:
            lazy_adata: LazyAnnData instance containing the single-cell data.
        """
        self.lazy_adata = lazy_adata
        self._slaf_array = lazy_adata.slaf
        self._uns_path = self._slaf_array._join_path(
            self._slaf_array.slaf_path, "uns.json"
        )
        self._uns_data: dict | None = None

    def _load_uns(self) -> dict:
        """Load uns.json if it exists, otherwise return empty dict"""
        if self._uns_data is not None:
            return self._uns_data

        # Check if file exists (not directory)
        import os

        if self._slaf_array._is_cloud_path(self._uns_path):
            # For cloud paths, try to open the file
            try:
                with self._slaf_array._open_file(self._uns_path) as f:
                    import json

                    self._uns_data = json.load(f)
            except Exception:
                self._uns_data = {}
        else:
            # For local paths, check if file exists
            if os.path.exists(self._uns_path) and os.path.isfile(self._uns_path):
                with self._slaf_array._open_file(self._uns_path) as f:
                    import json

                    self._uns_data = json.load(f)
            else:
                self._uns_data = {}

        return self._uns_data

    def keys(self) -> list[str]:
        """List all available keys"""
        return list(self._load_uns().keys())

    def __getitem__(self, key: str) -> Any:
        """Retrieve metadata value"""
        uns_data = self._load_uns()
        if key not in uns_data:
            raise KeyError(f"uns key '{key}' not found")
        return uns_data[key]

    def __setitem__(self, key: str, value: Any):
        """Store metadata value"""
        uns_data = self._load_uns()

        # Validate key name
        self._validate_name(key)

        # Convert numpy arrays and pandas objects to JSON-serializable
        value = self._json_serialize(value)

        uns_data[key] = value
        self._save_uns(uns_data)

    def __delitem__(self, key: str):
        """Delete metadata key"""
        uns_data = self._load_uns()
        if key not in uns_data:
            raise KeyError(f"uns key '{key}' not found")
        del uns_data[key]
        self._save_uns(uns_data)

    def _json_serialize(self, value: Any) -> Any:
        """Convert value to JSON-serializable format"""
        import pandas as pd

        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.integer | np.floating):
            return value.item()
        elif isinstance(value, pd.Series):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: self._json_serialize(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._json_serialize(item) for item in value]
        return value

    def _save_uns(self, uns_data: dict):
        """Save uns.json atomically"""
        import json

        # Use same cloud-compatible file writing as config.json
        with self._slaf_array._open_file(self._uns_path, "w") as f:
            json.dump(uns_data, f, indent=2)

        # Invalidate cache
        self._uns_data = uns_data


class LazyAnnData(LazySparseMixin):
    """
    AnnData-compatible interface for SLAF data with lazy evaluation.

    LazyAnnData provides a drop-in replacement for AnnData objects that works with
    SLAF datasets. It implements lazy evaluation to avoid loading all data into memory,
    making it suitable for large single-cell datasets.

    Key Features:
        - AnnData-compatible interface
        - Lazy evaluation for memory efficiency
        - Support for cell and gene subsetting
        - Integration with scanpy workflows
        - Automatic metadata loading
        - Transformation caching

    Examples:
        >>> # Basic usage
        >>> slaf_array = SLAFArray("path/to/data.slaf")
        >>> adata = LazyAnnData(slaf_array)
        >>> print(f"AnnData shape: {adata.shape}")
        AnnData shape: (1000, 20000)

        >>> # Access expression data
        >>> print(f"Expression matrix shape: {adata.X.shape}")
        Expression matrix shape: (1000, 20000)

        >>> # Access metadata
        >>> print(f"Cell metadata columns: {list(adata.obs.columns)}")
        Cell metadata columns: ['cell_type', 'total_counts', 'batch']
        >>> print(f"Gene metadata columns: {list(adata.var.columns)}")
        Gene metadata columns: ['gene_type', 'chromosome']

        >>> # Subsetting operations
        >>> subset = adata[:100, :5000]  # First 100 cells, first 5000 genes
        >>> print(f"Subset shape: {subset.shape}")
        Subset shape: (100, 5000)
    """

    def __init__(
        self,
        slaf_array: SLAFArray,
        backend: str = "auto",
    ):
        """
        Initialize LazyAnnData with SLAF array.

        Args:
            slaf_array: SLAFArray instance containing the single-cell data.
                       Used for database queries and metadata access.
            backend: Backend for expression matrix. Currently supports "scipy" and "auto".
                    "auto" defaults to "scipy" for sparse matrix operations.

        Raises:
            ValueError: If the backend is not supported.
            RuntimeError: If the SLAF array is not properly initialized.

        Examples:
            >>> # Basic initialization
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> print(f"Backend: {adata.backend}")
            Backend: auto

            >>> # With explicit backend
            >>> adata = LazyAnnData(slaf_array, backend="scipy")
            >>> print(f"Backend: {adata.backend}")
            Backend: scipy

            >>> # Error handling for unsupported backend
            >>> try:
            ...     adata = LazyAnnData(slaf_array, backend="unsupported")
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Unknown backend: unsupported
        """
        super().__init__()
        self.slaf = slaf_array
        self.backend = backend

        # Initialize expression matrix
        if backend == "scipy" or backend == "auto":
            self._X = LazyExpressionMatrix(slaf_array)
            self._X.parent_adata = self  # Set parent reference for transformations
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Lazy-loaded metadata
        self._obs = None
        self._var = None
        self._cached_obs_names: pd.Index | None = None
        self._cached_var_names: pd.Index | None = None

        # Filter selectors for subsetting
        self._cell_selector: Any = None
        self._gene_selector: Any = None
        self._filtered_obs: Callable[[], pd.DataFrame] | None = None
        self._filtered_var: Callable[[], pd.DataFrame] | None = None

        # Reference to parent LazyAnnData (for sliced objects)
        self._parent_adata: LazyAnnData | None = None

        # Transformations for lazy evaluation
        self._transformations: dict[str, Any] = {}

        # Layers view (lazy initialization)
        self._layers: LazyLayersView | None = None

    @property
    def layers(self) -> LazyLayersView:
        """
        Access to layers (dictionary-like interface).

        Returns a dictionary-like view of layers that provides access to alternative
        representations of the expression matrix (e.g., spliced, unspliced, counts).
        Layers have the same dimensions as X.

        Returns:
            LazyLayersView providing dictionary-like access to layers.

        Examples:
            >>> # Access a layer
            >>> slaf_array = SLAFArray("data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> spliced = adata.layers["spliced"]
            >>> print(f"Layer shape: {spliced.shape}")
            Layer shape: (1000, 20000)

            >>> # List available layers
            >>> print(list(adata.layers.keys()))
            ['spliced', 'unspliced']

            >>> # Check if layer exists
            >>> assert "spliced" in adata.layers
        """
        if self._layers is None:
            self._layers = LazyLayersView(self)
        return self._layers

    @property
    def obs(self) -> LazyObsView:
        """
        Cell metadata (observations) - mutable view with DataFrame-like and dict-like interfaces.

        Returns a view that provides both DataFrame-like and dictionary-like access to cell
        metadata columns with support for creating, updating, and deleting columns. This view
        respects cell selectors from parent LazyAnnData.

        When accessed with a string key (e.g., ``obs["cluster"]``), returns a pandas Series
        (AnnData-compatible). When accessed directly, behaves like a DataFrame.

        Returns:
            LazyObsView providing DataFrame-like and dictionary-like access to obs columns.

        Examples:
            >>> # DataFrame-like access (AnnData-compatible)
            >>> slaf_array = SLAFArray("data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> df = adata.obs  # Returns DataFrame-like view
            >>> cluster = adata.obs["cluster"]  # Returns pd.Series (AnnData-compatible)
            >>> print(cluster.head())

            >>> # Create a new column
            >>> adata.obs["new_cluster"] = new_cluster_labels
            >>> assert "new_cluster" in adata.obs

            >>> # List available columns
            >>> print(list(adata.obs.keys()))
            ['cell_id', 'total_counts', 'cluster', 'new_cluster']
        """
        if not hasattr(self, "_obs_view"):
            self._obs_view = LazyObsView(self)
        return self._obs_view

    @property
    def var(self) -> LazyVarView:
        """
        Gene metadata (variables) - mutable view with DataFrame-like and dict-like interfaces.

        Returns a view that provides both DataFrame-like and dictionary-like access to gene
        metadata columns with support for creating, updating, and deleting columns. This view
        respects gene selectors from parent LazyAnnData.

        When accessed with a string key (e.g., ``var["highly_variable"]``), returns a pandas
        Series (AnnData-compatible). When accessed directly, behaves like a DataFrame.

        Returns:
            LazyVarView providing DataFrame-like and dictionary-like access to var columns.

        Examples:
            >>> # DataFrame-like access (AnnData-compatible)
            >>> slaf_array = SLAFArray("data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> df = adata.var  # Returns DataFrame-like view
            >>> hvg = adata.var["highly_variable"]  # Returns pd.Series (AnnData-compatible)
            >>> print(hvg.head())

            >>> # Create a new column
            >>> adata.var["new_annotation"] = new_annotations
            >>> assert "new_annotation" in adata.var
        """
        if not hasattr(self, "_var_view"):
            self._var_view = LazyVarView(self)
        return self._var_view

    @property
    def obsm(self) -> LazyObsmView:
        """
        Multi-dimensional obs annotations (embeddings, PCA, etc.).

        Returns a dictionary-like view that provides access to multi-dimensional
        cell annotations stored as separate columns in cells.lance. Each key maps
        to a 2D numpy array.

        Returns:
            LazyObsmView providing dictionary-like access to obsm keys.

        Examples:
            >>> # Access an embedding
            >>> slaf_array = SLAFArray("data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> umap = adata.obsm["X_umap"]
            >>> print(f"UMAP shape: {umap.shape}")
            UMAP shape: (1000, 2)

            >>> # Create a new embedding
            >>> adata.obsm["X_pca"] = pca_coords  # shape: (1000, 50)
            >>> assert "X_pca" in adata.obsm
        """
        if not hasattr(self, "_obsm"):
            self._obsm = LazyObsmView(self)
        return self._obsm

    @property
    def varm(self) -> LazyVarmView:
        """
        Multi-dimensional var annotations (gene loadings, etc.).

        Returns a dictionary-like view that provides access to multi-dimensional
        gene annotations stored as separate columns in genes.lance. Each key maps
        to a 2D numpy array.

        Returns:
            LazyVarmView providing dictionary-like access to varm keys.

        Examples:
            >>> # Access gene loadings
            >>> slaf_array = SLAFArray("data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> pcs = adata.varm["PCs"]
            >>> print(f"PCs shape: {pcs.shape}")
            PCs shape: (20000, 50)
        """
        if not hasattr(self, "_varm"):
            self._varm = LazyVarmView(self)
        return self._varm

    @property
    def uns(self) -> LazyUnsView:
        """
        Unstructured metadata (analysis parameters, etc.).

        Returns a dictionary-like view that provides access to unstructured
        metadata stored in uns.json. Unlike obs/var/obsm/varm, uns metadata
        is always mutable and stored as JSON.

        Returns:
            LazyUnsView providing dictionary-like access to uns keys.

        Examples:
            >>> # Store metadata
            >>> slaf_array = SLAFArray("data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> adata.uns["neighbors"] = {"params": {"n_neighbors": 15}}
            >>> print(adata.uns["neighbors"])
            {'params': {'n_neighbors': 15}}
        """
        if not hasattr(self, "_uns"):
            self._uns = LazyUnsView(self)
        return self._uns

    @property
    def X(self) -> LazyExpressionMatrix:
        """
        Access to expression data.

        Returns the lazy expression matrix that provides scipy.sparse-compatible
        access to the single-cell expression data. The matrix is lazily evaluated
        to avoid loading all data into memory.

        Returns:
            LazyExpressionMatrix providing scipy.sparse-compatible interface.

        Examples:
            >>> # Access expression data
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> adata = LazyAnnData(slaf_array)
            >>> print(f"Expression matrix shape: {adata.X.shape}")
            Expression matrix shape: (1000, 20000)

            >>> # Subsetting expression data
            >>> subset_X = adata.X[:100, :5000]
            >>> print(f"Subset expression shape: {subset_X.shape}")
            Subset expression shape: (100, 5000)

            >>> # Check matrix type
            >>> print(f"Matrix type: {type(adata.X)}")
            Matrix type: <class 'slaf.integrations.anndata.LazyExpressionMatrix'>
        """
        return self._X

    @property
    def obs_deprecated(self) -> pd.DataFrame:
        """
        Cell metadata (observations) - DEPRECATED.

        .. deprecated:: 0.X
            This property is deprecated. Use :attr:`obs` (formerly :attr:`obs`) instead.
            This will be removed in a future version.

        This property triggers computation of metadata when accessed.
        For lazy access to metadata structure only, use obs.columns, obs.index, etc.
        """
        import warnings

        warnings.warn(
            "obs_deprecated is deprecated and will be removed in a future version. "
            "Use obs instead, which provides both DataFrame-like and dict-like access "
            "with mutation support.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._filtered_obs is not None:
            result = self._filtered_obs()
            if isinstance(result, pd.DataFrame):
                return result
            return pd.DataFrame()
        if self._obs is None:
            # Use the obs from SLAFArray (now polars DataFrame)
            obs_df = getattr(self.slaf, "obs", None)
            if obs_df is not None:
                # Work with polars DataFrame internally
                obs_pl = obs_df
                # Drop cell_integer_id column if present to match AnnData expectations
                if "cell_integer_id" in obs_pl.columns:
                    obs_pl = obs_pl.drop("cell_integer_id")
                # Filter out vector columns (obsm) - these are FixedSizeListArray columns
                # Vector columns should only be accessed via adata.obsm, not adata.obs
                cells_table = getattr(self.slaf, "cells", None)
                if cells_table is not None:
                    schema = cells_table.schema
                    vector_column_names = {
                        field.name
                        for field in schema
                        if isinstance(field.type, pa.FixedSizeListType)
                    }
                    # Drop vector columns from obs
                    columns_to_drop = [
                        col for col in obs_pl.columns if col in vector_column_names
                    ]
                    if columns_to_drop:
                        obs_pl = obs_pl.drop(columns_to_drop)
                # Convert to pandas DataFrame for AnnData compatibility at API boundary
                obs_copy = obs_pl.to_pandas()
                # Set cell_id as index if present, otherwise use default index
                if "cell_id" in obs_copy.columns:
                    obs_copy = obs_copy.set_index("cell_id")
                # Set index name to match AnnData format
                if hasattr(obs_copy, "index"):
                    obs_copy.index.name = "cell_id"
                self._obs = obs_copy
            else:
                self._obs = pd.DataFrame()
        return self._obs

    @property
    def var_deprecated(self) -> pd.DataFrame:
        """
        Gene metadata (variables) - DEPRECATED.

        .. deprecated:: 0.X
            This property is deprecated. Use :attr:`var` (formerly :attr:`var`) instead.
            This will be removed in a future version.

        This property triggers computation of metadata when accessed.
        For lazy access to metadata structure only, use var.columns, var.index, etc.
        """
        import warnings

        warnings.warn(
            "var_deprecated is deprecated and will be removed in a future version. "
            "Use var instead, which provides both DataFrame-like and dict-like access "
            "with mutation support.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._filtered_var is not None:
            result = self._filtered_var()
            if isinstance(result, pd.DataFrame):
                return result
            return pd.DataFrame()
        if self._var is None:
            var_df = getattr(self.slaf, "var", None)
            if var_df is not None:
                # Work with polars DataFrame internally
                var_pl = var_df
                # Drop gene_integer_id column if present to match AnnData expectations
                if "gene_integer_id" in var_pl.columns:
                    var_pl = var_pl.drop("gene_integer_id")
                # Filter out vector columns (varm) - these are FixedSizeListArray columns
                # Vector columns should only be accessed via adata.varm, not adata.var
                genes_table = getattr(self.slaf, "genes", None)
                if genes_table is not None:
                    schema = genes_table.schema
                    vector_column_names = {
                        field.name
                        for field in schema
                        if isinstance(field.type, pa.FixedSizeListType)
                    }
                    # Drop vector columns from var
                    columns_to_drop = [
                        col for col in var_pl.columns if col in vector_column_names
                    ]
                    if columns_to_drop:
                        var_pl = var_pl.drop(columns_to_drop)
                # Convert to pandas DataFrame for AnnData compatibility at API boundary
                var_copy = var_pl.to_pandas()
                # Set gene_id as index if present, otherwise use default index
                if "gene_id" in var_copy.columns:
                    var_copy = var_copy.set_index("gene_id")
                # Set index name to match AnnData format
                if hasattr(var_copy, "index"):
                    var_copy.index.name = "gene_id"
                self._var = var_copy
            else:
                self._var = pd.DataFrame()
        return self._var

    @property
    def obs_names(self) -> pd.Index:
        """Cell names"""
        if self._cached_obs_names is None:
            # Use the DataFrame's index from the view
            self._cached_obs_names = self.obs._get_dataframe().index
        return self._cached_obs_names

    @property
    def var_names(self) -> pd.Index:
        """Gene names"""
        if self._cached_var_names is None:
            # Use the DataFrame's index from the view
            self._cached_var_names = self.var._get_dataframe().index
        return self._cached_var_names

    @property
    def n_obs(self) -> int:
        """Number of observations (cells)"""
        return self.shape[0]

    @property
    def n_vars(self) -> int:
        """Number of variables (genes)"""
        return self.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the data, accounting for any applied filters"""
        if (
            getattr(self, "_cell_selector", None) is not None
            or getattr(self, "_gene_selector", None) is not None
        ):
            # Calculate the shape based on selectors
            cell_selector = (
                self._cell_selector
                if getattr(self, "_cell_selector", None) is not None
                else slice(None)
            )
            gene_selector = (
                self._gene_selector
                if getattr(self, "_gene_selector", None) is not None
                else slice(None)
            )

            # Use the same logic as _get_result_shape in LazySparseMixin
            n_cells = self._calculate_selected_count(cell_selector, axis=0)
            n_genes = self._calculate_selected_count(gene_selector, axis=1)

            return (n_cells, n_genes)
        else:
            # No filters applied, return original shape
            return self.slaf.shape

    def _calculate_selected_count(self, selector, axis: int) -> int:
        """Calculate the number of selected entities for a given selector"""
        if selector is None or (
            isinstance(selector, slice) and selector == slice(None)
        ):
            return self.slaf.shape[axis]

        if isinstance(selector, slice):
            start = selector.start or 0
            stop = selector.stop or self.slaf.shape[axis]
            step = selector.step or 1

            # Handle negative indices
            if start < 0:
                start = self.slaf.shape[axis] + start
            if stop < 0:
                stop = self.slaf.shape[axis] + stop

            # Clamp bounds to actual data size
            start = max(0, min(start, self.slaf.shape[axis]))
            stop = max(0, min(stop, self.slaf.shape[axis]))

            return len(range(start, stop, step))
        elif isinstance(selector, list | np.ndarray):
            if isinstance(selector, np.ndarray) and selector.dtype == bool:
                return np.sum(selector)
            return len(selector)
        elif isinstance(selector, int | np.integer):
            return 1
        else:
            return self.slaf.shape[axis]

    def __getitem__(self, key) -> "LazyAnnData":
        """Subset the data, composing selectors if already sliced"""
        # Disallow chained slicing on LazyAnnData objects
        if self._cell_selector is not None or self._gene_selector is not None:
            raise NotImplementedError(
                "Chained slicing on LazyAnnData objects is not supported. "
                "Please use adata[rows, cols] for single-step slicing instead of "
                "adata[rows][:, cols]. For chained slicing on the expression matrix, "
                "use adata.X[rows][:, cols]."
            )

        # Parse indexing key using the same logic as LazyExpressionMatrix
        cell_selector, gene_selector = self._parse_key(key)

        # Create a new LazyAnnData with the same backend
        new_adata = LazyAnnData(self.slaf, backend=self.backend)

        # Store reference to parent for accessing full DataFrames
        new_adata._parent_adata = self

        # Store the selectors for lazy filtering
        new_adata._cell_selector = cell_selector
        new_adata._gene_selector = gene_selector

        # Update the LazyExpressionMatrix to know about the slicing
        if isinstance(new_adata._X, LazyExpressionMatrix):
            new_adata._X.parent_adata = new_adata
            # Store the selectors in the LazyExpressionMatrix so it can use them
            new_adata._X._cell_selector = cell_selector
            new_adata._X._gene_selector = gene_selector
            # Update the shape to reflect the slicing
            new_adata._X._update_shape()

        # Override obs and var properties to apply filtering
        def filtered_obs() -> pd.DataFrame:
            # Always apply the composed selector to the original obs DataFrame
            # Get the FULL DataFrame from parent (without selectors) to apply our selector
            # Access parent's view without selectors
            parent = getattr(self, "_parent_adata", None)
            if parent is None:
                parent = self
            parent_obs_view = parent.obs
            # Temporarily clear selectors to get full DataFrame
            original_cell_selector = getattr(parent, "_cell_selector", None)
            original_gene_selector = getattr(parent, "_gene_selector", None)
            # Temporarily clear selectors
            parent._cell_selector = None
            parent._gene_selector = None
            # Clear cache to force reload
            if hasattr(parent_obs_view, "_dataframe"):
                parent_obs_view._dataframe = None
            # Get full DataFrame
            obs_df = parent_obs_view._get_dataframe()
            # Restore selectors
            parent._cell_selector = original_cell_selector
            parent._gene_selector = original_gene_selector
            if cell_selector is None or (
                isinstance(cell_selector, slice) and cell_selector == slice(None)
            ):
                pass
            else:
                obs_df = obs_df.copy()
                if isinstance(cell_selector, slice):
                    start = cell_selector.start
                    stop = cell_selector.stop
                    step = cell_selector.step if cell_selector.step is not None else 1

                    # Handle None values
                    if start is None:
                        start = 0
                    if stop is None:
                        stop = len(obs_df)

                    # Handle negative indices
                    if start < 0:
                        start = len(obs_df) + start
                    if stop < 0:
                        stop = len(obs_df) + stop

                    # Clamp bounds to valid range
                    start = max(0, min(start, len(obs_df)))
                    stop = max(0, min(stop, len(obs_df)))
                    obs_df = obs_df.iloc[start:stop:step]
                elif (
                    isinstance(cell_selector, np.ndarray)
                    and cell_selector.dtype == bool
                ):
                    # Boolean mask - ensure it matches the length
                    if len(cell_selector) == len(obs_df):
                        obs_df = obs_df[cell_selector]
                    else:
                        # Pad or truncate the mask to match
                        if len(cell_selector) < len(obs_df):
                            mask: np.ndarray = np.zeros(len(obs_df), dtype=bool)
                            mask[: len(cell_selector)] = cell_selector
                        else:
                            mask = cell_selector[: len(obs_df)]
                        obs_df = obs_df[mask]
                elif isinstance(cell_selector, list | np.ndarray):
                    # Integer indices
                    obs_df = obs_df.iloc[cell_selector]
                elif isinstance(cell_selector, int | np.integer):
                    obs_df = obs_df.iloc[[int(cell_selector)]]
                # else: leave as is
            # Remove unused categories for all categorical columns if not empty
            if obs_df is not None and not obs_df.empty:
                for col in obs_df.select_dtypes(include="category").columns:
                    col_data = obs_df[col]
                    # Type-safe check for pandas categorical data
                    if isinstance(col_data, pd.Series) and hasattr(col_data, "cat"):
                        obs_df[col] = col_data.cat.remove_unused_categories()
            if isinstance(obs_df, pd.DataFrame):
                return obs_df
            return pd.DataFrame()

        def filtered_var() -> pd.DataFrame:
            # Always apply the composed selector to the original var DataFrame
            # Get the FULL DataFrame from parent (without selectors) to apply our selector
            # Access parent's view without selectors
            parent = getattr(self, "_parent_adata", None)
            if parent is None:
                parent = self
            parent_var_view = parent.var
            # Temporarily clear selectors to get full DataFrame
            original_cell_selector = getattr(parent, "_cell_selector", None)
            original_gene_selector = getattr(parent, "_gene_selector", None)
            # Temporarily clear selectors
            parent._cell_selector = None
            parent._gene_selector = None
            # Clear cache to force reload
            if hasattr(parent_var_view, "_dataframe"):
                parent_var_view._dataframe = None
            # Get full DataFrame
            var_df = parent_var_view._get_dataframe()
            # Restore selectors
            parent._cell_selector = original_cell_selector
            parent._gene_selector = original_gene_selector

            if gene_selector is None or (
                isinstance(gene_selector, slice) and gene_selector == slice(None)
            ):
                pass
            else:
                var_df = var_df.copy()
                if isinstance(gene_selector, slice):
                    start = gene_selector.start
                    stop = gene_selector.stop
                    step = gene_selector.step if gene_selector.step is not None else 1

                    # Handle None values
                    if start is None:
                        start = 0
                    if stop is None:
                        stop = len(var_df)

                    # Handle negative indices
                    if start < 0:
                        start = len(var_df) + start
                    if stop < 0:
                        stop = len(var_df) + stop

                    # Clamp bounds to valid range
                    start = max(0, min(start, len(var_df)))
                    stop = max(0, min(stop, len(var_df)))
                    var_df = var_df.iloc[start:stop:step]
                elif (
                    isinstance(gene_selector, np.ndarray)
                    and gene_selector.dtype == bool
                ):
                    # Boolean mask - ensure it matches the length
                    if len(gene_selector) == len(var_df):
                        var_df = var_df[gene_selector]
                    else:
                        # Pad or truncate the mask to match
                        if len(gene_selector) < len(var_df):
                            mask: np.ndarray = np.zeros(len(var_df), dtype=bool)
                            mask[: len(gene_selector)] = gene_selector
                        else:
                            mask = gene_selector[: len(var_df)]
                        var_df = var_df[mask]
                elif isinstance(gene_selector, list | np.ndarray):
                    # Integer indices
                    var_df = var_df.iloc[gene_selector]
                elif isinstance(gene_selector, int | np.integer):
                    var_df = var_df.iloc[[int(gene_selector)]]
                # else: leave as is
            # Remove unused categories for all categorical columns if not empty
            if var_df is not None and not var_df.empty:
                for col in var_df.select_dtypes(include="category").columns:
                    col_data = var_df[col]
                    # Type-safe check for pandas categorical data
                    if isinstance(col_data, pd.Series) and hasattr(col_data, "cat"):
                        var_df[col] = col_data.cat.remove_unused_categories()
            if isinstance(var_df, pd.DataFrame):
                return var_df
            return pd.DataFrame()

        # Store the filter functions
        new_adata._filtered_obs = filtered_obs
        new_adata._filtered_var = filtered_var

        # Ensure obs_names and var_names match the filtered DataFrames
        new_adata._cached_obs_names = new_adata._filtered_obs().index
        new_adata._cached_var_names = new_adata._filtered_var().index

        # Copy transformations
        new_adata._transformations = self._transformations.copy()

        # Copy layers view (will be recreated with new selectors when accessed)
        new_adata._layers = None

        return new_adata

    def copy(self) -> "LazyAnnData":
        """Create a copy"""
        new_adata = LazyAnnData(self.slaf, backend=self.backend)

        # Copy selectors
        new_adata._cell_selector = self._cell_selector
        new_adata._gene_selector = self._gene_selector

        # Copy filtered metadata functions
        new_adata._filtered_obs = self._filtered_obs
        new_adata._filtered_var = self._filtered_var

        # Copy transformations
        new_adata._transformations = self._transformations.copy()

        # Set up parent reference for the expression matrix
        if isinstance(new_adata._X, LazyExpressionMatrix):
            new_adata._X.parent_adata = new_adata

        return new_adata

    def to_memory(self) -> scipy.sparse.spmatrix:
        """Load entire matrix into memory"""
        return self.get_expression_data()

    def write(self, filename: str):
        """Write to h5ad format (would need implementation)"""
        raise NotImplementedError("Writing LazyAnnData not yet implemented")

    def get_expression_data(self) -> scipy.sparse.csr_matrix:
        """Get expression data with any applied filters"""
        if isinstance(self._X, LazyExpressionMatrix):
            # If the LazyExpressionMatrix already has selectors stored, just get the data
            # without passing additional selectors to avoid double-composition
            if self._X._cell_selector is not None or self._X._gene_selector is not None:
                return self._X[:, :].compute()
            elif self._cell_selector is not None or self._gene_selector is not None:
                cell_sel = (
                    self._cell_selector
                    if self._cell_selector is not None
                    else slice(None)
                )
                gene_sel = (
                    self._gene_selector
                    if self._gene_selector is not None
                    else slice(None)
                )
                return self._X[cell_sel, gene_sel].compute()
            else:
                return self._X[:, :].compute()
        else:
            raise NotImplementedError(
                "Dask backend not yet implemented for get_expression_data"
            )

    def compute(self) -> "sc.AnnData":
        """Explicitly compute and return a native AnnData object"""
        import scanpy as sc

        # Create native AnnData object (get DataFrames from views)
        adata = sc.AnnData(
            X=self._X.compute(),
            obs=self.obs._get_dataframe(),
            var=self.var._get_dataframe(),
        )

        return adata

    def _update_with_normalized_data(
        self, result_df: pl.DataFrame, target_sum: float, inplace: bool
    ) -> "LazyAnnData | None":
        """
        Update AnnData with normalized data from fragment processing.

        Args:
            result_df: Polars DataFrame with normalized values
            target_sum: Target sum used for normalization
            inplace: Whether to modify in place

        Returns:
            Updated LazyAnnData or None if inplace=True
        """
        if inplace:
            # Store normalization transformation
            if not hasattr(self, "_transformations"):
                self._transformations = {}

            self._transformations["normalize_total"] = {
                "type": "normalize_total",
                "target_sum": target_sum,
                "fragment_processed": True,
                "result_df": result_df,
            }

            print(
                f"Applied normalize_total with target_sum={target_sum} (fragment processing)"
            )
            return None
        else:
            # Create a copy with the transformation
            new_adata = self.copy()
            if not hasattr(new_adata, "_transformations"):
                new_adata._transformations = {}

            new_adata._transformations["normalize_total"] = {
                "type": "normalize_total",
                "target_sum": target_sum,
                "fragment_processed": True,
                "result_df": result_df,
            }

            return new_adata

    def _update_with_log1p_data(
        self, result_df: pl.DataFrame, inplace: bool
    ) -> "LazyAnnData | None":
        """
        Update AnnData with log1p data from fragment processing.

        Args:
            result_df: Polars DataFrame with log1p values
            inplace: Whether to modify in place

        Returns:
            Updated LazyAnnData or None if inplace=True
        """
        if inplace:
            # Store log1p transformation
            if not hasattr(self, "_transformations"):
                self._transformations = {}

            self._transformations["log1p"] = {
                "type": "log1p",
                "fragment_processed": True,
                "result_df": result_df,
            }

            print("Applied log1p transformation (fragment processing)")
            return None
        else:
            # Create a copy with the transformation
            new_adata = self.copy()
            if not hasattr(new_adata, "_transformations"):
                new_adata._transformations = {}

            new_adata._transformations["log1p"] = {
                "type": "log1p",
                "fragment_processed": True,
                "result_df": result_df,
            }

            return new_adata

    def _get_processing_strategy(self, fragments: bool | None = None) -> bool:
        """Determine the processing strategy based on fragments and dataset size"""
        if fragments is not None:
            return fragments
        try:
            fragments_list = self.slaf.expression.get_fragments()
            return len(fragments_list) > 1
        except Exception:
            return False


def read_slaf(filename: str, backend: str = "auto") -> LazyAnnData:
    """Read SLAF file as LazyAnnData object"""
    slaf_array = SLAFArray(filename)
    return LazyAnnData(slaf_array, backend=backend)
