"""Provides technical indicator computation utilities used in prediction and backtesting."""

import numpy as np
import pandas as pd

from utils.parameters import ParameterLoader


class IndicatorCalculator:
    """Computes technical indicators for backtesting."""

    # Load parameters
    _PARAMS = ParameterLoader()
    _RSI_WINDOW = _PARAMS.get("rsi_window_backtest")
    _MACD_FAST = _PARAMS.get("macd_fast")
    _MACD_SLOW = _PARAMS.get("macd_slow")
    _MACD_SIGNAL = _PARAMS.get("macd_signal")
    _BOLLINGER_WINDOW = _PARAMS.get("bollinger_window")
    _BOLLINGER_BAND_METHOD = _PARAMS.get("bollinger_band_method")
    _STOCH_RSI_WINDOW = _PARAMS.get("stoch_rsi_window")
    _STOCH_RSI_MIN_PERIODS = _PARAMS.get("stoch_rsi_min_periods")
    _OBV_FILL_METHOD = _PARAMS.get("obv_fill_method")
    _ATR_WINDOW = _PARAMS.get("atr_window")
    _WILLIAMS_R_WINDOW = _PARAMS.get("williams_r_window")

    @staticmethod
    def compute_rsi(series: pd.Series, window: int) -> pd.Series:
        """Calculate the Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).astype("float32")

    @staticmethod
    def compute_macd_diff(
        series: pd.Series, fast: int, slow: int, signal: int
    ) -> pd.Series:
        """Calculate MACD line minus signal line."""
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return (macd - signal_line).astype("float32")

    @staticmethod
    def compute_stoch_rsi(series: pd.Series, window: int) -> pd.Series:
        """Calculate stochastic RSI."""
        min_val = series.rolling(
            window=window, min_periods=IndicatorCalculator._STOCH_RSI_MIN_PERIODS
        ).min()
        max_val = series.rolling(
            window=window, min_periods=IndicatorCalculator._STOCH_RSI_MIN_PERIODS
        ).max()
        return ((series - min_val) / (max_val - min_val) * 100).astype("float32")

    @staticmethod
    def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Apply technical indicator calculations to OHLCV dataframe."""
        close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

        df["return_1h"] = close.pct_change().astype("float32")
        df["volatility_3h"] = df["return_1h"].rolling(window=3).std().astype("float32")
        df["momentum_3h"] = (close - close.shift(3)).astype("float32")
        df["rsi"] = IndicatorCalculator.compute_rsi(
            close, IndicatorCalculator._RSI_WINDOW
        )
        df["macd"] = IndicatorCalculator.compute_macd_diff(
            close,
            IndicatorCalculator._MACD_FAST,
            IndicatorCalculator._MACD_SLOW,
            IndicatorCalculator._MACD_SIGNAL,
        )
        df["volume"] = volume.astype("float32")

        if IndicatorCalculator._BOLLINGER_BAND_METHOD == "max-min":
            df["bb_width"] = (
                close.rolling(IndicatorCalculator._BOLLINGER_WINDOW)
                .apply(lambda x: x.max() - x.min(), raw=True)
                .astype("float32")
            )

        df["stoch_rsi"] = IndicatorCalculator.compute_stoch_rsi(
            close, IndicatorCalculator._STOCH_RSI_WINDOW
        )
        df["obv"] = (
            (np.sign(close.diff()) * volume)
            .fillna(IndicatorCalculator._OBV_FILL_METHOD)
            .cumsum()
            .astype("float32")
        )

        tr_components = pd.concat(
            [
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        )

        df["atr"] = (
            tr_components.max(axis=1)
            .rolling(window=IndicatorCalculator._ATR_WINDOW)
            .mean()
            .astype("float32")
        )

        high_max = high.rolling(IndicatorCalculator._WILLIAMS_R_WINDOW).max()
        low_min = low.rolling(IndicatorCalculator._WILLIAMS_R_WINDOW).min()
        df["williams_r"] = (-100 * ((high_max - close) / (high_max - low_min))).astype(
            "float32"
        )

        return df

    @staticmethod
    def get_indicator_parameters() -> dict:
        """Returns a dictionary with all technical indicator parameters."""
        return {
            "rsi_window": IndicatorCalculator._RSI_WINDOW,
            "macd_fast": IndicatorCalculator._MACD_FAST,
            "macd_slow": IndicatorCalculator._MACD_SLOW,
            "macd_signal": IndicatorCalculator._MACD_SIGNAL,
            "bollinger_window": IndicatorCalculator._BOLLINGER_WINDOW,
            "bollinger_band_method": IndicatorCalculator._BOLLINGER_BAND_METHOD,
            "stoch_rsi_window": IndicatorCalculator._STOCH_RSI_WINDOW,
            "stoch_rsi_min_periods": IndicatorCalculator._STOCH_RSI_MIN_PERIODS,
            "obv_fill_method": IndicatorCalculator._OBV_FILL_METHOD,
            "atr_window": IndicatorCalculator._ATR_WINDOW,
            "williams_r_window": IndicatorCalculator._WILLIAMS_R_WINDOW,
        }
