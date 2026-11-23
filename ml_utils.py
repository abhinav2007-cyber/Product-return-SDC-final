"""Shared feature definitions so training and the web app can never drift apart."""
import pandas as pd

NUMERIC = [
    "line_count", "total_abs_quantity", "average_unit_price", "unique_products",
    "avg_qty_per_line", "month", "weekday", "hour",
]
CATEGORICAL = ["country"]
FEATURES = NUMERIC + CATEGORICAL


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add features computed from other invoice-level columns."""
    df = df.copy()
    df["avg_qty_per_line"] = df["total_abs_quantity"] / df["line_count"].clip(lower=1)
    return df
