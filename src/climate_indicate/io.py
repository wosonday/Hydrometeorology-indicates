from __future__ import annotations
from pathlib import Path
import pandas as pd


__all__ = ["load_daily_precip", "to_monthly_totals"]


def load_daily_precip(path: Path, date_column: str = "date", value_column: int = -1) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[date_column]).set_index(date_column)
    return df.iloc[:, value_column]


def to_monthly_totals(daily: pd.Series) -> pd.Series:
    """min_count=1: tháng toàn NaN → NaN, không âm thầm thành 0."""
    return daily.resample("ME").sum(min_count=1)
