"""
Utility functions for timezone normalization in financial market data processing.

Includes tools for localizing naive or timezone-aware DataFrame timestamps to UTC
based on a given market timezone. Designed to support consistent datetime handling
throughout the market data ingestion and training pipeline.
"""

import pandas as pd
import pytz

from utils.parameters import ParameterLoader

_PARAMS = ParameterLoader()
_MARKET_TZ = _PARAMS.get("market_tz", "America/New_York")


def localize_to_market_time(
    df: pd.DataFrame, market_tz: str = _MARKET_TZ
) -> pd.DataFrame:
    """Converts naive or localized timestamps to UTC based on market timezone."""
    if df.empty:
        return df

    market_tz = pytz.timezone(market_tz)

    if df.index.tzinfo is None or df.index.tz is None:
        df.index = df.index.tz_localize(market_tz)

    return df.tz_convert("UTC")
