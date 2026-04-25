"""Optional COO value transforms before window rank / bin (training pipeline)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from slaf.core.tabular_schema import DataSchema


@dataclass(frozen=True)
class ExpressionPreprocessor:
    """Per-cell and element-wise transforms on the COO ``value`` column."""

    normalize_total_target: float | None = None
    """If set, scale each cell so its values sum to this target (Scanpy-style)."""

    log1p: bool = False
    """If True, apply ``log1p`` element-wise after optional normalization."""


def apply_expression_preprocessor(
    fragment_df: pl.DataFrame,
    schema: DataSchema,
    preprocessor: ExpressionPreprocessor | None,
) -> pl.DataFrame:
    """Update ``schema.value_key`` when ``preprocessor`` applies ops; no-op if ``None``."""
    if preprocessor is None:
        return fragment_df
    if not isinstance(preprocessor, ExpressionPreprocessor):
        raise TypeError(
            "expression_preprocessor must be an ExpressionPreprocessor instance or None, "
            f"got {type(preprocessor)!r}"
        )
    if preprocessor.normalize_total_target is None and not preprocessor.log1p:
        return fragment_df

    vk = schema.value_key
    out = fragment_df
    if preprocessor.normalize_total_target is not None:
        t = float(preprocessor.normalize_total_target)
        cell_sum = pl.col(vk).sum().over(schema.group_key)
        out = out.with_columns(
            pl.when(cell_sum > 0)
            .then(pl.col(vk) * (t / cell_sum))
            .otherwise(0.0)
            .alias(vk)
        )
    if preprocessor.log1p:
        out = out.with_columns(pl.col(vk).log1p().alias(vk))
    return out
