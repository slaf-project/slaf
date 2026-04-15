"""
Shared tabular schema for COO-style pipelines (single-node ML and distributed).

Keeps ``slaf.ml`` independent of ``slaf.distributed`` while giving both the same
column contract for window / group-by steps.
"""


class DataSchema:
    """
    Generic schema configuration for tabular data processing.

    Describes the structure of input data and output aggregations.
    The input data must be tabular with at least three columns:
    - group_key: Column to group by
    - item_key: Column for items within groups
    - value_key: Column for values

    Output columns after aggregation:
    - group_key_out: Output group key (defaults to group_key if None)
    - item_list_key: Aggregated list of items per group
    - value_list_key: Optional aggregated list of values per group
    """

    # Input columns (required)
    group_key: str
    item_key: str
    value_key: str

    # Output columns (after window/aggregation)
    group_key_out: str | None
    item_list_key: str
    value_list_key: str | None

    def __init__(
        self,
        group_key: str,
        item_key: str,
        value_key: str,
        group_key_out: str | None = None,
        item_list_key: str = "item_list",
        value_list_key: str | None = None,
    ):
        """
        Initialize data schema.

        Args:
            group_key: Column to group by
            item_key: Column for items within groups
            value_key: Column for values
            group_key_out: Output group key (defaults to group_key if None)
            item_list_key: Aggregated list of items per group
            value_list_key: Optional aggregated list of values per group
        """
        self.group_key = group_key
        self.item_key = item_key
        self.value_key = value_key
        self.group_key_out = group_key_out
        self.item_list_key = item_list_key
        self.value_list_key = value_list_key


# Default tabular layout for SLAF ``expression.lance`` COO rows and matching window outputs.
SLAF_LANCE_COO_SCHEMA = DataSchema(
    group_key="cell_integer_id",
    item_key="gene_integer_id",
    value_key="value",
    group_key_out="cell_integer_id",
    item_list_key="gene_sequence",
    value_list_key="expr_sequence",
)
