from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import scipy


class LazySparseMixin:
    """
    Mixin class for lazy sparse matrix operations with SLAF integration.

    LazySparseMixin provides a foundation for implementing lazy sparse matrix
    operations that work with SLAF data. It defines the interface for classes
    that need to perform sparse matrix operations without loading all data into memory.

    Key Features:
        - Lazy evaluation of sparse matrix operations
        - Integration with SLAFArray for database queries
        - Support for cell and gene name mapping
        - Memory-efficient sparse operations

    Implementing classes must provide:
        - shape: Tuple[int, int] - the shape of the matrix (n_cells, n_genes)
        - slaf_array: SLAFArray object for database queries
        - obs_names: Optional[pd.Index] - cell names/IDs (for cell-wise operations)
        - var_names: Optional[pd.Index] - gene names/IDs (for gene-wise operations)

    Examples:
        >>> # Basic usage in a subclass
        >>> class MyLazyMatrix(LazySparseMixin):
        ...     def __init__(self, slaf_array):
        ...         super().__init__()
        ...         self.slaf_array = slaf_array
        ...         self._shape = slaf_array.shape
        ...
        ...     @property
        ...     def shape(self):
        ...         return self._shape
        >>>
        >>> slaf_array = SLAFArray("path/to/data.slaf")
        >>> lazy_matrix = MyLazyMatrix(slaf_array)
        >>> print(f"Matrix shape: {lazy_matrix.shape}")
        Matrix shape: (1000, 20000)

        >>> # With cell and gene name mapping
        >>> class NamedLazyMatrix(LazySparseMixin):
        ...     def __init__(self, slaf_array, obs_names, var_names):
        ...         super().__init__()
        ...         self.slaf_array = slaf_array
        ...         self._shape = slaf_array.shape
        ...         self._obs_names = obs_names
        ...         self._var_names = var_names
        ...
        ...     @property
        ...     def shape(self):
        ...         return self._shape
        ...
        ...     @property
        ...     def obs_names(self):
        ...         return self._obs_names
        ...
        ...     @property
        ...     def var_names(self):
        ...         return self._var_names
        >>>
        >>> obs_names = pd.Index([f"cell_{i}" for i in range(1000)])
        >>> var_names = pd.Index([f"gene_{i}" for i in range(20000)])
        >>> named_matrix = NamedLazyMatrix(slaf_array, obs_names, var_names)
        >>> print(f"Cell names: {len(named_matrix.obs_names)}")
        Cell names: 1000
    """

    # Required attributes that implementing classes must provide
    # These are defined as properties in implementing classes
    slaf_array: Any

    @property
    def shape(self) -> tuple[int, int]:
        """
        Shape of the matrix - must be implemented by subclasses.

        Returns:
            Tuple of (n_cells, n_genes) representing the matrix dimensions.

        Raises:
            NotImplementedError: If the subclass doesn't implement this property.

        Examples:
            >>> # In a subclass implementation
            >>> class MyMatrix(LazySparseMixin):
            ...     def __init__(self, shape):
            ...         super().__init__()
            ...         self._shape = shape
            ...
            ...     @property
            ...     def shape(self):
            ...         return self._shape
            >>>
            >>> matrix = MyMatrix((100, 200))
            >>> print(f"Shape: {matrix.shape}")
            Shape: (100, 200)
        """
        raise NotImplementedError("Subclasses must implement shape property")

    def __init__(self):
        """
        Initialize the mixin.

        This method sets up the basic structure for lazy sparse matrix operations.
        Implementing classes should call this method and then set the required
        attributes (slaf_array, shape, etc.).

        Examples:
            >>> # Proper initialization in a subclass
            >>> class MyLazyMatrix(LazySparseMixin):
            ...     def __init__(self, slaf_array):
            ...         super().__init__()  # Call parent init
            ...         self.slaf_array = slaf_array  # Set required attribute
            ...         self._shape = slaf_array.shape
            ...
            ...     @property
            ...     def shape(self):
            ...         return self._shape
            >>>
            >>> slaf_array = SLAFArray("path/to/data.slaf")
            >>> matrix = MyLazyMatrix(slaf_array)
            >>> print(f"Initialized with shape: {matrix.shape}")
            Initialized with shape: (1000, 20000)
        """
        # Implementing classes will set the required attributes
        pass

    def _parse_key(self, key) -> tuple[Any, Any]:
        """Parse numpy-style indexing key into cell and gene selectors"""
        if not isinstance(key, tuple):
            return key, slice(None)

        if len(key) == 1:
            return key[0], slice(None)
        elif len(key) == 2:
            # Handle np.ix_ case: key is (row_selector, col_selector)
            # where each selector is a 2D array
            row_selector, col_selector = key

            # If these are 2D arrays from np.ix_, extract the 1D indices
            if isinstance(row_selector, np.ndarray) and row_selector.ndim == 2:
                # Extract the row indices from the 2D array
                row_selector = row_selector.flatten()
            if isinstance(col_selector, np.ndarray) and col_selector.ndim == 2:
                # Extract the column indices from the 2D array
                col_selector = col_selector.flatten()

            return row_selector, col_selector
        else:
            raise IndexError("Too many indices for 2D array")

    def _selector_to_sql_condition(self, selector, axis: int, entity_type: str) -> str:
        """Convert various selector types to SQL WHERE conditions"""
        # Type check to ensure we have the required attributes
        if not hasattr(self, "shape"):
            raise AttributeError("Implementing class must provide 'shape' attribute")

        if selector is None or (
            isinstance(selector, slice) and selector == slice(None)
        ):
            return "TRUE"  # No filtering

        # Use integer ID columns for filtering
        if entity_type == "cell":
            entity_id_col = "cell_integer_id"
        else:
            entity_id_col = "gene_integer_id"

        if isinstance(selector, slice):
            start = selector.start or 0
            stop = selector.stop or (self.shape[axis])
            step = selector.step or 1

            if step == 1:
                return f"{entity_id_col} >= {start} AND {entity_id_col} < {stop}"
            else:
                # Handle step != 1 by generating explicit list
                indices = list(range(start, stop, step))
                return f"{entity_id_col} IN ({','.join(map(str, indices))})"

        elif isinstance(selector, list | np.ndarray):
            if isinstance(selector, np.ndarray) and selector.dtype == bool:
                # Boolean mask - get the range of True indices
                true_indices = np.where(selector)[0]
                if len(true_indices) == 0:
                    return "FALSE"
                elif len(true_indices) == len(selector):
                    return "TRUE"
                else:
                    # Use IN clause for efficiency
                    return f"{entity_id_col} IN ({','.join(map(str, true_indices))})"

            # Handle integer indexing
            return f"{entity_id_col} IN ({','.join(map(str, selector))})"

        elif isinstance(selector, int | np.integer):
            return f"{entity_id_col} = {selector}"

        else:
            raise TypeError(f"Unsupported selector type: {type(selector)}")

    def _build_submatrix_query(
        self, cell_selector, gene_selector, transformations=None
    ) -> pd.DataFrame:
        """Build submatrix query using the efficient get_submatrix() function"""

        # Pass selectors directly to get_submatrix for optimal integer-based queries
        # The get_submatrix method already handles all selector types efficiently
        sub_matrix = self.slaf_array.get_submatrix(
            cell_selector=cell_selector,
            gene_selector=gene_selector,
        )

        return sub_matrix

    def _selector_to_range(self, selector, axis: int) -> tuple[int, int]:
        """Convert selector to integer range for LanceDB queries"""
        if selector is None or (
            isinstance(selector, slice) and selector == slice(None)
        ):
            return 0, self.shape[axis]

        if isinstance(selector, slice):
            start = selector.start or 0
            stop = selector.stop or self.shape[axis]
            step = selector.step or 1

            # For step != 1, we need to handle this differently
            # For now, just use the full range and filter later
            if step != 1:
                return 0, self.shape[axis]

            return start, stop

        elif isinstance(selector, list | np.ndarray):
            if isinstance(selector, np.ndarray) and selector.dtype == bool:
                # Boolean mask - get the range of True indices
                true_indices = np.where(selector)[0]
                if len(true_indices) == 0:
                    return 0, 0
                return int(true_indices.min()), int(true_indices.max() + 1)
            else:
                # List of indices - get the range
                if len(selector) == 0:
                    return 0, 0
                return int(min(selector)), int(max(selector) + 1)

        elif isinstance(selector, int | np.integer):
            return int(selector), int(selector) + 1

        else:
            return 0, self.shape[axis]

    def _estimate_selected_count(self, selector, axis: int) -> int:
        """Estimate number of selected entities for strategy selection"""
        if selector is None or (
            isinstance(selector, slice) and selector == slice(None)
        ):
            return self.shape[axis]

        if isinstance(selector, slice):
            start = selector.start or 0
            stop = selector.stop or self.shape[axis]
            step = selector.step or 1
            return len(range(start, stop, step))

        elif isinstance(selector, list | np.ndarray):
            if isinstance(selector, np.ndarray) and selector.dtype == bool:
                return np.sum(selector)
            return len(selector)

        elif isinstance(selector, int | np.integer):
            return 1

        return self.shape[axis]

    def _boolean_mask_to_sql(self, selector, entity_col: str) -> str:
        """Convert boolean mask to efficient SQL condition"""

        if isinstance(selector, np.ndarray) and selector.dtype == bool:
            # Convert boolean mask to index list
            bool_indices = np.where(selector)[0]
            if len(bool_indices) == 0:
                return "FALSE"
            elif len(bool_indices) == len(selector):
                return "TRUE"
            else:
                # Use IN clause for efficiency
                return f"{entity_col} IN ({','.join(map(str, bool_indices))})"

        elif isinstance(selector, slice):
            start = selector.start or 0
            stop = selector.stop or (
                self.shape[0] if entity_col == "cell_id" else self.shape[1]
            )
            step = selector.step or 1
            if step == 1:
                return f"{entity_col} >= {start} AND {entity_col} < {stop}"
            else:
                indices: list[int] = list(range(start, stop, step))
                return f"{entity_col} IN ({','.join(map(str, indices))})"

        elif isinstance(selector, list | np.ndarray):
            if len(selector) == 0:
                return "FALSE"
            return f"{entity_col} IN ({','.join(map(str, selector))})"

        elif isinstance(selector, int | np.integer):
            return f"{entity_col} = {selector}"

        else:
            return "TRUE"  # Default to no filtering

    def _reconstruct_sparse_matrix(
        self, records: pl.DataFrame, cell_selector, gene_selector
    ) -> scipy.sparse.csr_matrix:
        """Reconstruct scipy sparse matrix from SLAF query results"""
        import scipy.sparse

        if len(records) == 0:
            # Return empty matrix with appropriate shape
            result_shape = self._get_result_shape(cell_selector, gene_selector)
            return scipy.sparse.csr_matrix(result_shape)

        # Get integer IDs directly from the expression table
        cell_integer_ids = records["cell_integer_id"].to_numpy().astype(np.int64)
        gene_integer_ids = records["gene_integer_id"].to_numpy().astype(np.int64)
        values = records["value"].to_numpy()

        # Build output row indices
        if cell_selector is None or (
            isinstance(cell_selector, slice) and cell_selector == slice(None)
        ):
            output_cell_ids = np.arange(self.slaf_array.shape[0])
        elif isinstance(cell_selector, slice):
            start = cell_selector.start or 0
            stop = cell_selector.stop or self.slaf_array.shape[0]
            step = cell_selector.step or 1

            # Handle negative indices
            if start < 0:
                start = self.slaf_array.shape[0] + start
            if stop < 0:
                stop = self.slaf_array.shape[0] + stop

            # Clamp bounds
            start = max(0, min(start, self.slaf_array.shape[0]))
            stop = max(0, min(stop, self.slaf_array.shape[0]))

            output_cell_ids = np.arange(start, stop, step)
        elif isinstance(cell_selector, np.ndarray) and cell_selector.dtype == bool:
            output_cell_ids = np.where(cell_selector)[0]
        elif isinstance(cell_selector, list | np.ndarray):
            output_cell_ids = np.array(cell_selector)
        elif isinstance(cell_selector, int | np.integer):
            output_cell_ids = np.array([cell_selector])
        else:
            output_cell_ids = np.arange(self.slaf_array.shape[0])

        # Build output col indices
        if gene_selector is None or (
            isinstance(gene_selector, slice) and gene_selector == slice(None)
        ):
            output_gene_ids = np.arange(self.slaf_array.shape[1])
        elif isinstance(gene_selector, slice):
            start = gene_selector.start or 0
            stop = gene_selector.stop or self.slaf_array.shape[1]
            step = gene_selector.step or 1

            # Handle negative indices
            if start < 0:
                start = self.slaf_array.shape[1] + start
            if stop < 0:
                stop = self.slaf_array.shape[1] + stop

            # Clamp bounds
            start = max(0, min(start, self.slaf_array.shape[1]))
            stop = max(0, min(stop, self.slaf_array.shape[1]))

            output_gene_ids = np.arange(start, stop, step)
        elif isinstance(gene_selector, np.ndarray) and gene_selector.dtype == bool:
            output_gene_ids = np.where(gene_selector)[0]
        elif isinstance(gene_selector, list | np.ndarray):
            output_gene_ids = np.array(gene_selector)
        elif isinstance(gene_selector, int | np.integer):
            output_gene_ids = np.array([gene_selector])
        else:
            output_gene_ids = np.arange(self.slaf_array.shape[1])

        # Create mappings from original integer IDs to output positions
        # Only include IDs that are actually in the output range
        cell_id_to_row = {}
        for i, cid in enumerate(output_cell_ids):
            cell_id_to_row[cid] = i

        gene_id_to_col = {}
        for i, gid in enumerate(output_gene_ids):
            gene_id_to_col[gid] = i

        # Filter records to only include those that map to valid output positions
        valid_mask = np.logical_and(
            np.isin(cell_integer_ids, list(cell_id_to_row.keys())),
            np.isin(gene_integer_ids, list(gene_id_to_col.keys())),
        )

        if not np.any(valid_mask):
            # No valid records - return empty matrix
            result_shape = (len(output_cell_ids), len(output_gene_ids))
            return scipy.sparse.csr_matrix(result_shape)

        # Apply mask to get only valid records
        valid_cell_ids = cell_integer_ids[valid_mask]
        valid_gene_ids = gene_integer_ids[valid_mask]
        valid_values = values[valid_mask]

        # Map to output positions
        rows = np.array(
            [cell_id_to_row[cid] for cid in valid_cell_ids], dtype=int
        ).reshape(-1)
        cols = np.array(
            [gene_id_to_col[gid] for gid in valid_gene_ids], dtype=int
        ).reshape(-1)

        result_shape = (len(output_cell_ids), len(output_gene_ids))
        return scipy.sparse.csr_matrix((valid_values, (rows, cols)), shape=result_shape)

    def _create_id_mapping(self, selector, axis: int) -> dict:
        """Create a mapping from database IDs to matrix indices for any selector type"""
        if selector is None or (
            isinstance(selector, slice) and selector == slice(None)
        ):
            # Full slice - map each ID to itself
            return {i: i for i in range(self.shape[axis])}

        if isinstance(selector, slice):
            # Range slice - map IDs to slice-relative positions
            start = selector.start or 0
            stop = selector.stop or self.shape[axis]
            step = selector.step or 1

            mapping = {}
            for i, db_id in enumerate(range(start, stop, step)):
                mapping[db_id] = i
            return mapping

        elif isinstance(selector, np.ndarray) and selector.dtype == bool:
            # Boolean mask - map True positions to sequential indices
            true_indices = [int(x) for x in np.where(selector)[0]]
            bool_mapping: dict[int, int] = {}
            for i, db_id in enumerate(true_indices):
                bool_mapping[db_id] = i
            return bool_mapping

        elif isinstance(selector, list | np.ndarray):
            # Arbitrary list - map each ID to its position in the list
            list_mapping: dict[int, int] = {}
            for i, db_id in enumerate(selector):
                list_mapping[int(db_id)] = int(i)
            return list_mapping

        elif isinstance(selector, int | np.integer):
            # Single index - map the ID to position 0
            return {int(selector): 0}

        else:
            # Fallback - empty mapping
            return {}

    def _get_result_shape(self, cell_selector, gene_selector) -> tuple[int, int]:
        """Calculate the shape of the result matrix"""
        if not hasattr(self, "shape"):
            raise AttributeError("Implementing class must provide 'shape' attribute")

        # Calculate number of cells
        if cell_selector is None or (
            isinstance(cell_selector, slice) and cell_selector == slice(None)
        ):
            n_cells = self.shape[0]
        elif isinstance(cell_selector, slice):
            start = cell_selector.start or 0
            stop = cell_selector.stop or self.shape[0]
            step = cell_selector.step or 1

            # Clamp bounds to actual data size
            start = max(0, min(start, self.shape[0]))
            stop = max(0, min(stop, self.shape[0]))

            n_cells = len(range(start, stop, step))
        elif isinstance(cell_selector, list | np.ndarray):
            if isinstance(cell_selector, np.ndarray) and cell_selector.dtype == bool:
                n_cells = np.sum(cell_selector)
            else:
                n_cells = len(cell_selector)
        elif isinstance(cell_selector, int | np.integer):
            n_cells = 1
        else:
            n_cells = self.shape[0]

        # Calculate number of genes
        if gene_selector is None or (
            isinstance(gene_selector, slice) and gene_selector == slice(None)
        ):
            n_genes = self.shape[1]
        elif isinstance(gene_selector, slice):
            start = gene_selector.start or 0
            stop = gene_selector.stop or self.shape[1]
            step = gene_selector.step or 1

            # Clamp bounds to actual data size
            start = max(0, min(start, self.shape[1]))
            stop = max(0, min(stop, self.shape[1]))

            n_genes = len(range(start, stop, step))
        elif isinstance(gene_selector, list | np.ndarray):
            if isinstance(gene_selector, np.ndarray) and gene_selector.dtype == bool:
                n_genes = np.sum(gene_selector)
            else:
                n_genes = len(gene_selector)
        elif isinstance(gene_selector, int | np.integer):
            n_genes = 1
        else:
            n_genes = self.shape[1]

        return (n_cells, n_genes)

    def _sql_aggregation(self, operation: str, axis: int | None = None) -> np.ndarray:
        """Perform aggregation operations via SQL with optimized performance"""
        if not hasattr(self, "shape"):
            raise AttributeError("Implementing class must provide 'shape' attribute")
        if not hasattr(self, "slaf_array"):
            raise AttributeError(
                "Implementing class must provide 'slaf_array' attribute"
            )

        if operation.upper() == "AVG" or operation.upper() == "MEAN":
            # For mean, we need to compute sum and divide by total elements
            return self._sql_mean_aggregation(axis)
        elif operation.upper() == "VARIANCE" or operation.upper() == "VAR":
            # For variance, we need custom calculation accounting for implicit zeros
            return self._sql_variance_aggregation(axis)
        elif operation.upper() == "STDDEV" or operation.upper() == "STD":
            # For standard deviation, compute variance and take sqrt
            variance = self._sql_variance_aggregation(axis)
            return np.sqrt(variance)
        else:
            # For other operations, use the optimized approach
            return self._sql_other_aggregation(operation, axis)

    def _sql_mean_aggregation(self, axis: int | None = None) -> np.ndarray:
        """Optimized mean aggregation with vectorized operations"""
        if axis == 0:  # Gene-wise aggregation
            # Single optimized query with proper ordering
            sql = """
            SELECT
                gene_integer_id,
                SUM(value) as total_sum
            FROM expression
            GROUP BY gene_integer_id
            ORDER BY gene_integer_id
            """
            result_df = self.slaf_array.query(sql)

            # Vectorized result construction
            full_result = np.zeros(self.shape[1])
            total_cells = self.shape[0]

            if len(result_df) > 0:
                # Use vectorized operations instead of loops
                gene_indices = result_df["gene_integer_id"].to_numpy()
                sums = result_df["total_sum"].to_numpy()

                # Ensure indices are within bounds
                valid_mask = (gene_indices >= 0) & (gene_indices < self.shape[1])
                if np.any(valid_mask):
                    full_result[gene_indices[valid_mask]] = (
                        sums[valid_mask] / total_cells
                    )

            return full_result.reshape(1, -1)

        elif axis == 1:  # Cell-wise aggregation
            # Single optimized query with proper ordering
            sql = """
            SELECT
                cell_integer_id,
                SUM(value) as total_sum
            FROM expression
            GROUP BY cell_integer_id
            ORDER BY cell_integer_id
            """
            result_df = self.slaf_array.query(sql)

            # Vectorized result construction
            full_result = np.zeros(self.shape[0])
            total_genes = self.shape[1]

            if len(result_df) > 0:
                # Use vectorized operations instead of loops
                cell_indices = result_df["cell_integer_id"].to_numpy()
                sums = result_df["total_sum"].to_numpy()

                # Ensure indices are within bounds
                valid_mask = (cell_indices >= 0) & (cell_indices < self.shape[0])
                if np.any(valid_mask):
                    full_result[cell_indices[valid_mask]] = (
                        sums[valid_mask] / total_genes
                    )

            return full_result.reshape(-1, 1)

        else:  # Global aggregation
            # Optimized global query
            sql = "SELECT SUM(value) as total_sum FROM expression"
            result = self.slaf_array.query(sql)

            if len(result) > 0:
                total_sum = result.item(0, "total_sum")
                total_elements = self.shape[0] * self.shape[1]
                global_mean = total_sum / total_elements
                return np.array([global_mean])
            else:
                return np.array([0.0])

    def _sql_variance_aggregation(self, axis: int | None = None) -> np.ndarray:
        """Optimized variance aggregation with vectorized operations"""
        if axis == 0:  # Gene-wise aggregation
            # Single optimized query with all needed statistics
            sql = """
            SELECT
                gene_integer_id,
                SUM(value) as total_sum,
                SUM(value * value) as sum_squares
            FROM expression
            GROUP BY gene_integer_id
            ORDER BY gene_integer_id
            """
            result_df = self.slaf_array.query(sql)

            # Vectorized result construction
            full_result = np.zeros(self.shape[1])
            total_cells = self.shape[0]

            if len(result_df) > 0:
                # Use vectorized operations instead of loops
                gene_indices = result_df["gene_integer_id"].to_numpy()
                sums = result_df["total_sum"].to_numpy()
                sum_squares = result_df["sum_squares"].to_numpy()

                # Ensure indices are within bounds
                valid_mask = (gene_indices >= 0) & (gene_indices < self.shape[1])
                if np.any(valid_mask):
                    valid_indices = gene_indices[valid_mask]
                    valid_sums = sums[valid_mask]
                    valid_sum_squares = sum_squares[valid_mask]

                    # Vectorized variance calculation
                    means = valid_sums / total_cells
                    variances = (valid_sum_squares / total_cells) - (means * means)
                    full_result[valid_indices] = variances

            return full_result.reshape(1, -1)

        elif axis == 1:  # Cell-wise aggregation
            # Single optimized query with all needed statistics
            sql = """
            SELECT
                cell_integer_id,
                SUM(value) as total_sum,
                SUM(value * value) as sum_squares
            FROM expression
            GROUP BY cell_integer_id
            ORDER BY cell_integer_id
            """
            result_df = self.slaf_array.query(sql)

            # Vectorized result construction
            full_result = np.zeros(self.shape[0])
            total_genes = self.shape[1]

            if len(result_df) > 0:
                # Use vectorized operations instead of loops
                cell_indices = result_df["cell_integer_id"].to_numpy()
                sums = result_df["total_sum"].to_numpy()
                sum_squares = result_df["sum_squares"].to_numpy()

                # Ensure indices are within bounds
                valid_mask = (cell_indices >= 0) & (cell_indices < self.shape[0])
                if np.any(valid_mask):
                    valid_indices = cell_indices[valid_mask]
                    valid_sums = sums[valid_mask]
                    valid_sum_squares = sum_squares[valid_mask]

                    # Vectorized variance calculation
                    means = valid_sums / total_genes
                    variances = (valid_sum_squares / total_genes) - (means * means)
                    full_result[valid_indices] = variances

            return full_result.reshape(-1, 1)

        else:  # Global aggregation
            # Optimized global variance query
            sql = """
            SELECT
                SUM(value) as total_sum,
                SUM(value * value) as sum_squares
            FROM expression
            """
            result = self.slaf_array.query(sql)

            if len(result) > 0:
                total_sum = result.item(0, "total_sum")
                sum_squares = result.item(0, "sum_squares")
                total_elements = self.shape[0] * self.shape[1]

                # Global variance calculation
                global_mean = total_sum / total_elements
                global_variance = (sum_squares / total_elements) - (
                    global_mean * global_mean
                )
                return np.array([global_variance])
            else:
                return np.array([0.0])

    def _sql_other_aggregation(
        self, operation: str, axis: int | None = None
    ) -> np.ndarray:
        """Optimized non-mean aggregation operations with vectorized operations"""
        if axis == 0:  # Gene-wise aggregation
            sql = f"""
            SELECT
                gene_integer_id,
                {operation.upper()}(value) as result
            FROM expression
            GROUP BY gene_integer_id
            ORDER BY gene_integer_id
            """
            result_df = self.slaf_array.query(sql)

            # Vectorized result construction
            full_result = np.zeros(self.shape[1])

            if len(result_df) > 0:
                # Use vectorized operations instead of loops
                gene_indices = result_df["gene_integer_id"].to_numpy()
                results = result_df["result"].to_numpy()

                # Ensure indices are within bounds
                valid_mask = (gene_indices >= 0) & (gene_indices < self.shape[1])
                if np.any(valid_mask):
                    full_result[gene_indices[valid_mask]] = results[valid_mask]

            return full_result.reshape(1, -1)

        elif axis == 1:  # Cell-wise aggregation
            sql = f"""
            SELECT
                cell_integer_id,
                {operation.upper()}(value) as result
            FROM expression
            GROUP BY cell_integer_id
            ORDER BY cell_integer_id
            """
            result_df = self.slaf_array.query(sql)

            # Vectorized result construction
            full_result = np.zeros(self.shape[0])

            if len(result_df) > 0:
                # Use vectorized operations instead of loops
                cell_indices = result_df["cell_integer_id"].to_numpy()
                results = result_df["result"].to_numpy()

                # Ensure indices are within bounds
                valid_mask = (cell_indices >= 0) & (cell_indices < self.shape[0])
                if np.any(valid_mask):
                    full_result[cell_indices[valid_mask]] = results[valid_mask]

            return full_result.reshape(-1, 1)

        else:  # Global aggregation
            sql = f"SELECT {operation.upper()}(value) as result FROM expression"
            result = self.slaf_array.query(sql)
            return (
                np.array([float(result.item(0, "result"))])
                if len(result) > 0
                else np.array([0.0])
            )

    def _sql_multi_aggregation(self, operations: list, axis: int | None = None) -> dict:
        """Optimized multi-aggregation in a single query for better performance"""
        if not hasattr(self, "shape"):
            raise AttributeError("Implementing class must provide 'shape' attribute")
        if not hasattr(self, "slaf_array"):
            raise AttributeError(
                "Implementing class must provide 'slaf_array' attribute"
            )

        if axis == 0:  # Gene-wise aggregation
            # Build dynamic SQL with multiple aggregations
            agg_clauses = []
            for op in operations:
                if op.upper() in ["MEAN", "AVG"]:
                    agg_clauses.append(f"SUM(value) as {op.lower()}_sum")
                elif op.upper() in ["VARIANCE", "VAR"]:
                    agg_clauses.append(f"SUM(value) as {op.lower()}_sum")
                    agg_clauses.append(
                        f"SUM(value * value) as {op.lower()}_sum_squares"
                    )
                else:
                    agg_clauses.append(f"{op.upper()}(value) as {op.lower()}_result")

            sql = f"""
            SELECT
                gene_integer_id,
                {", ".join(agg_clauses)}
            FROM expression
            GROUP BY gene_integer_id
            ORDER BY gene_integer_id
            """
            result_df = self.slaf_array.query(sql)

            # Process results for each operation
            gene_results: dict[str, np.ndarray] = {}
            total_cells = self.shape[0]

            for op in operations:
                if op.upper() in ["MEAN", "AVG"]:
                    full_result = np.zeros(self.shape[1])
                    if len(result_df) > 0:
                        gene_indices = result_df["gene_integer_id"].values
                        sums = result_df[f"{op.lower()}_sum"].values
                        valid_mask = (gene_indices >= 0) & (
                            gene_indices < self.shape[1]
                        )
                        if np.any(valid_mask):
                            full_result[gene_indices[valid_mask]] = (
                                sums[valid_mask] / total_cells
                            )
                    gene_results[op.lower()] = full_result.reshape(1, -1).astype(
                        np.float64
                    )

                elif op.upper() in ["VARIANCE", "VAR"]:
                    full_result = np.zeros(self.shape[1])
                    if len(result_df) > 0:
                        gene_indices = result_df["gene_integer_id"].values
                        sums = result_df[f"{op.lower()}_sum"].values
                        sum_squares = result_df[f"{op.lower()}_sum_squares"].values
                        valid_mask = (gene_indices >= 0) & (
                            gene_indices < self.shape[1]
                        )
                        if np.any(valid_mask):
                            valid_indices = gene_indices[valid_mask]
                            valid_sums = sums[valid_mask]
                            valid_sum_squares = sum_squares[valid_mask]
                            means = valid_sums / total_cells
                            variances = (valid_sum_squares / total_cells) - (
                                means * means
                            )
                            full_result[valid_indices] = variances
                    gene_results[op.lower()] = full_result.reshape(1, -1).astype(
                        np.float64
                    )

                else:
                    full_result = np.zeros(self.shape[1])
                    if len(result_df) > 0:
                        gene_indices = result_df["gene_integer_id"].values
                        op_results = result_df[f"{op.lower()}_result"].values
                        valid_mask = (gene_indices >= 0) & (
                            gene_indices < self.shape[1]
                        )
                        if np.any(valid_mask):
                            full_result[gene_indices[valid_mask]] = op_results[
                                valid_mask
                            ]
                    gene_results[op.lower()] = full_result.reshape(1, -1).astype(
                        np.float64
                    )

            return gene_results

        elif axis == 1:  # Cell-wise aggregation
            # Similar optimization for cell-wise operations
            agg_clauses = []
            for op in operations:
                if op.upper() in ["MEAN", "AVG"]:
                    agg_clauses.append(f"SUM(value) as {op.lower()}_sum")
                elif op.upper() in ["VARIANCE", "VAR"]:
                    agg_clauses.append(f"SUM(value) as {op.lower()}_sum")
                    agg_clauses.append(
                        f"SUM(value * value) as {op.lower()}_sum_squares"
                    )
                else:
                    agg_clauses.append(f"{op.upper()}(value) as {op.lower()}_result")

            sql = f"""
            SELECT
                cell_integer_id,
                {", ".join(agg_clauses)}
            FROM expression
            GROUP BY cell_integer_id
            ORDER BY cell_integer_id
            """
            result_df = self.slaf_array.query(sql)

            # Process results for each operation
            cell_results: dict[str, np.ndarray] = {}
            total_genes = self.shape[1]

            for op in operations:
                if op.upper() in ["MEAN", "AVG"]:
                    full_result = np.zeros(self.shape[0])
                    if len(result_df) > 0:
                        cell_indices = result_df["cell_integer_id"].values
                        sums = result_df[f"{op.lower()}_sum"].values
                        valid_mask = (cell_indices >= 0) & (
                            cell_indices < self.shape[0]
                        )
                        if np.any(valid_mask):
                            full_result[cell_indices[valid_mask]] = (
                                sums[valid_mask] / total_genes
                            )
                    cell_results[op.lower()] = full_result.reshape(-1, 1).astype(
                        np.float64
                    )

                elif op.upper() in ["VARIANCE", "VAR"]:
                    full_result = np.zeros(self.shape[0])
                    if len(result_df) > 0:
                        cell_indices = result_df["cell_integer_id"].values
                        sums = result_df[f"{op.lower()}_sum"].values
                        sum_squares = result_df[f"{op.lower()}_sum_squares"].values
                        valid_mask = (cell_indices >= 0) & (
                            cell_indices < self.shape[0]
                        )
                        if np.any(valid_mask):
                            valid_indices = cell_indices[valid_mask]
                            valid_sums = sums[valid_mask]
                            valid_sum_squares = sum_squares[valid_mask]
                            means = valid_sums / total_genes
                            variances = (valid_sum_squares / total_genes) - (
                                means * means
                            )
                            full_result[valid_indices] = variances
                    cell_results[op.lower()] = full_result.reshape(-1, 1).astype(
                        np.float64
                    )

                else:
                    full_result = np.zeros(self.shape[0])
                    if len(result_df) > 0:
                        cell_indices = result_df["cell_integer_id"].values
                        op_results = result_df[f"{op.lower()}_result"].values
                        valid_mask = (cell_indices >= 0) & (
                            cell_indices < self.shape[0]
                        )
                        if np.any(valid_mask):
                            full_result[cell_indices[valid_mask]] = op_results[
                                valid_mask
                            ]
                    cell_results[op.lower()] = full_result.reshape(-1, 1).astype(
                        np.float64
                    )

            return cell_results

        else:  # Global aggregation
            # Global multi-aggregation
            agg_clauses = []
            for op in operations:
                if op.upper() in ["MEAN", "AVG"]:
                    agg_clauses.append(f"SUM(value) as {op.lower()}_sum")
                elif op.upper() in ["VARIANCE", "VAR"]:
                    agg_clauses.append(f"SUM(value) as {op.lower()}_sum")
                    agg_clauses.append(
                        f"SUM(value * value) as {op.lower()}_sum_squares"
                    )
                else:
                    agg_clauses.append(f"{op.upper()}(value) as {op.lower()}_result")

            sql = f"SELECT {', '.join(agg_clauses)} FROM expression"
            result = self.slaf_array.query(sql)

            global_results: dict[str, np.ndarray] = {}
            if len(result) > 0:
                total_elements = self.shape[0] * self.shape[1]

                for op in operations:
                    if op.upper() in ["MEAN", "AVG"]:
                        total_sum = result.item(0, f"{op.lower()}_sum")
                        global_results[op.lower()] = np.array(
                            [total_sum / total_elements]
                        )
                    elif op.upper() in ["VARIANCE", "VAR"]:
                        total_sum = result.item(0, f"{op.lower()}_sum")
                        sum_squares = result.item(0, f"{op.lower()}_sum_squares")
                        global_mean = total_sum / total_elements
                        global_variance = (sum_squares / total_elements) - (
                            global_mean * global_mean
                        )
                        global_results[op.lower()] = np.array([global_variance])
                    else:
                        op_result = result.item(0, f"{op.lower()}_result")
                        global_results[op.lower()] = np.array([float(op_result)])
            else:
                for op in operations:
                    global_results[op.lower()] = np.array([0.0])

            return global_results
