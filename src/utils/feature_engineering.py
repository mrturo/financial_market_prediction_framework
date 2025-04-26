"""
Feature engineering utilities for financial market models.

Provides centralized logic to enrich OHLCV dataframes with technical indicators,
session-based flags, calendar placeholders, and target label generation
for multi-class classification tasks (Down, Neutral, Up).
"""

import pandas as pd

from utils.indicators import IndicatorCalculator
from utils.parameters import ParameterLoader

_PARAMS = ParameterLoader()
_DEFAULT_CLASSIFICATION_THRESHOLD = _PARAMS.get("classification_threshold")
_DEFAULT_PERIODS_AHEAD = _PARAMS.get("classification_periods_ahead")


def enrich_with_common_features(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """
    Applies common feature engineering steps including:

    * Technical indicators
    * Session flags
    * Holiday/fed placeholders
    * Target generation logic.

    Args:
        df (pd.DataFrame): Input OHLCV data indexed by datetime.
        symbol (str, optional): Symbol name to assign if needed.

    Returns:
        pd.DataFrame: Feature-enriched DataFrame.
    """
    df = df.copy()
    df.sort_index(inplace=True)

    if symbol:
        df["symbol"] = symbol

    df = IndicatorCalculator.compute_technical_indicators(df)

    if "datetime" in df.columns:
        df.set_index("datetime", inplace=True)

    df["hour"] = df.index.hour + df.index.minute / 60
    df["is_pre_market"] = ((df["hour"] >= 4.0) & (df["hour"] < 9.5)).astype(int)
    df["is_post_market"] = ((df["hour"] >= 16.0) & (df["hour"] <= 20.0)).astype(int)
    df["day_of_week"] = df.index.dayofweek
    df["is_fed_day"] = 0
    df["is_holiday"] = 0
    df["is_before_holiday"] = 0

    df["date"] = df.index.date
    daily_close = df.groupby("date")["close"].last()
    prev_map = {
        k2: k1 for k1, k2 in zip(daily_close.index, list(daily_close.index)[1:])
    }
    df["prev_day"] = df["date"].map(prev_map)
    df["close_yesterday"] = df["prev_day"].map(daily_close.to_dict())
    df["today_close"] = df["date"].map(daily_close.to_dict())

    df["target"] = classify_target(df)

    return df


def prepare_raw_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Standardizes timestamp parsing, indexing and calls enrich_with_common_features.

    Args:
        df (pd.DataFrame): Raw input dataframe.
        symbol (str): Symbol to assign.

    Returns:
        pd.DataFrame: Preprocessed and enriched dataframe.
    """
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df["symbol"] = symbol

    return enrich_with_common_features(df, symbol)


def classify_target(
    df: pd.DataFrame,
    forward_col: str = "close",
    periods_ahead: int = _DEFAULT_PERIODS_AHEAD,
    threshold: float = _DEFAULT_CLASSIFICATION_THRESHOLD,
) -> pd.Series:
    """
    Classifies future returns into Down / Neutral / Up based on thresholds.

    Args:
        df (pd.DataFrame): Input dataframe with at least one price column.
        forward_col (str): Column to use for future return calculation.
        periods_ahead (int): How many periods ahead to look for future value.
        threshold (float): Minimum percentage change to classify as Up or Down.

    Returns:
        pd.Series: Series of integer labels (0=Down, 1=Neutral, 2=Up).
    """
    future_returns = df[forward_col].shift(-periods_ahead) / df[forward_col] - 1

    def label(r):
        if pd.isna(r):
            return pd.NA
        if r > threshold:
            return 2  # UP
        if r < -threshold:
            return 0  # DOWN
        return 1  # NEUTRAL

    return future_returns.apply(label).astype("Int8")
