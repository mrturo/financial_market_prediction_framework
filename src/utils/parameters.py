"""
ParameterLoader — Central configuration manager.

This module handles the loading and initialization of both static and dynamic parameters,
including symbol lists, cutoff dates, model configuration values, and paths to key artifacts.
It integrates with SymbolRepository and supports centralized access to runtime configuration
through dictionary-style and method-based access.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

from utils.symbols import SymbolPaths, SymbolRepository


class ParameterLoader:
    """Centralized configuration manager for all pipeline parameters."""

    _CONF_MATRIX_PLOT_FILENAME = "confusion_matrix_percentage.png"
    _EXPORT_SYMBOL_SUMMARY_TO_CSV = "data/market_data_summary.csv"
    _F1_SCORE_PLOT_FILENAME = "f1_score_by_class.png"
    _MARKETDATA_FILEPATH = "data/market_data.json"
    _MODEL_FILEPATH = "data/model/model.pkl"
    _OPTUNA_FILEPATH = "data/model/optuna_study.pkl"
    _SCALER_FILEPATH = "data/model/scaler.pkl"
    _SYMBOLS_BASE_PATH = "config/symbols"
    _SYMBOLS_CORRELATIVE_PATH = "/symbols_correlative.json"
    _SYMBOLS_INVALID_PATH = "/symbols_invalid.json"
    _SYMBOLS_MERCADO_PAGO_PATH = "/symbols_mercado_pago.json"
    _SYMBOLS_TRAINING_PATH = "/symbols_training.json"
    _SYMBOLS_XTB_PATH = "/symbols_xtb.json"

    _ENV_PATH = ".env"

    @staticmethod
    def _build_symbol_path(filename: str) -> str:
        return f"{ParameterLoader._SYMBOLS_BASE_PATH}{filename}"

    def __init__(self, last_updated: pd.Timestamp = None):
        self.env_path = Path(ParameterLoader._ENV_PATH)
        load_dotenv(dotenv_path=self.env_path)
        self.symbol_repo = SymbolRepository(
            SymbolPaths(
                mercado_pago=os.path.join(
                    ParameterLoader._build_symbol_path(self._SYMBOLS_MERCADO_PAGO_PATH)
                ),
                xtb=os.path.join(
                    ParameterLoader._build_symbol_path(self._SYMBOLS_XTB_PATH)
                ),
                correlative=os.path.join(
                    ParameterLoader._build_symbol_path(self._SYMBOLS_CORRELATIVE_PATH)
                ),
                invalid=os.path.join(
                    ParameterLoader._build_symbol_path(self._SYMBOLS_INVALID_PATH)
                ),
                training=os.path.join(
                    ParameterLoader._build_symbol_path(self._SYMBOLS_TRAINING_PATH)
                ),
            )
        )
        self.last_update = (
            datetime.now(timezone.utc) if last_updated is None else last_updated
        )
        self._parameters: Dict[str, Any] = self._initialize_parameters()

    def _initialize_parameters(self) -> Dict[str, Any]:
        """Initializes the parameters dictionary by merging static JSON and dynamic values."""
        round_last_update = self.last_update.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        one_year_ago_last_update = (
            round_last_update - relativedelta(years=1) - timedelta(days=1)
        )

        dynamic_params = {
            "cutoff_date": one_year_ago_last_update.strftime("%Y-%m-%d"),
            "training_symbols": self.symbol_repo.get_training_symbols(),
            "all_symbols": self.symbol_repo.get_all_symbols(),
            "mercado_pago_symbols": self.symbol_repo.get_mercado_pago_symbols(),
            "xtb_symbols": self.symbol_repo.get_xtb_symbols(),
            "correlative_symbols": self.symbol_repo.get_correlative_symbols(),
            "invalid_symbols_path": f"{self._SYMBOLS_BASE_PATH}{self._SYMBOLS_INVALID_PATH}",
        }

        constant_params = {
            "weekdays": [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            "stale_days_threshold": 7,
            "confusion_matrix_figsize": [6, 5],
            "f1_score_figsize": [6, 4],
            "features": [
                "return_1h",
                "volatility_3h",
                "momentum_3h",
                "rsi",
                "macd",
                "volume",
                "bb_width",
                "stoch_rsi",
                "obv",
                "atr",
                "williams_r",
                "is_pre_market",
                "is_post_market",
                "day_of_week",
                "is_fed_day",
                "is_holiday",
                "is_before_holiday",
            ],
            "fed_event_days": [
                "2024-06-12",
                "2024-07-31",
                "2024-09-18",
                "2024-11-06",
                "2024-12-18",
            ],
            "market_open_days": [0, 1, 2, 3, 4],
            "required_market_columns": [
                "datetime",
                "close",
                "open",
                "high",
                "low",
                "volume",
            ],
            "target_labels": ["↓ Down", "→ Neutral", "↑ Up"],
        }

        vulnerable_params = {
            "atr_window": int(os.getenv("ATR_WINDOW")),
            "backtesting_output_folder": os.getenv("BACKTESTING_OUTPUT_FOLDER"),
            "block_days": int(os.getenv("BLOCK_DAYS")),
            "bollinger_band_method": os.getenv("BOLLINGER_BAND_METHOD"),
            "bollinger_window": int(os.getenv("BOLLINGER_WINDOW")),
            "cutoff_minutes": int(os.getenv("CUTOFF_MINUTES")),
            "dashboard_footer_caption": os.getenv("DASHBOARD_FOOTER_CAPTION"),
            "dashboard_layout": os.getenv("DASHBOARD_LAYOUT"),
            "dashboard_page_title": os.getenv("DASHBOARD_PAGE_TITLE"),
            "download_retries": int(os.getenv("DOWNLOAD_RETRIES")),
            "updater_retries": int(os.getenv("UPDATER_RETRIES")),
            "down_threshold": float(os.getenv("DOWN_THRESHOLD")),
            "evaluation_report_folder": os.getenv("EVALUATION_REPORT_FOLDER"),
            "historical_days_fallback": int(os.getenv("HISTORICAL_DAYS_FALLBACK")),
            "historical_window_days": int(os.getenv("HISTORICAL_WINDOW_DAYS")),
            "interval": os.getenv("INTERVAL"),
            "macd_fast": int(os.getenv("MACD_FAST")),
            "macd_signal": int(os.getenv("MACD_SIGNAL")),
            "macd_slow": int(os.getenv("MACD_SLOW")),
            "market_tz": os.getenv("MARKET_TZ"),
            "n_splits": int(os.getenv("N_SPLITS")),
            "n_trials": int(os.getenv("N_TRIALS")),
            "obv_fill_method": int(os.getenv("OBV_FILL_METHOD")),
            "prediction_workers": int(os.getenv("PREDICTION_WORKERS")),
            "retry_sleep_seconds": int(os.getenv("RETRY_SLEEP_SECONDS")),
            "rsi_window": int(os.getenv("RSI_WINDOW")),
            "rsi_window_backtest": int(os.getenv("RSI_WINDOW_BACKTEST")),
            "start_hour_today_check": int(os.getenv("START_HOUR_TODAY_CHECK")),
            "stoch_rsi_min_periods": int(os.getenv("STOCH_RSI_MIN_PERIODS")),
            "stoch_rsi_window": int(os.getenv("STOCH_RSI_WINDOW")),
            "threshold": float(os.getenv("THRESHOLD")),
            "up_threshold": float(os.getenv("UP_THRESHOLD")),
            "volume_window": int(os.getenv("VOLUME_WINDOW")),
            "williams_r_window": int(os.getenv("WILLIAMS_R_WINDOW")),
            "market_calendar_name": os.getenv("MARKET_CALENDAR_NAME"),
            "holiday_country": os.getenv("HOLIDAY_COUNTRY"),
            "date_format": os.getenv("DATE_FORMAT"),
            "classification_periods_ahead": int(
                os.getenv("CLASSIFICATION_PERIODS_AHEAD")
            ),
            "classification_threshold": float(os.getenv("CLASSIFICATION_THRESHOLD")),
            "f1_score_plot_title": os.getenv("F1_SCORE_PLOT_TITLE"),
        }

        path_params = {
            "conf_matrix_plot_filename": self._CONF_MATRIX_PLOT_FILENAME,
            "correlative_path": self._build_symbol_path(self._SYMBOLS_CORRELATIVE_PATH),
            "export_symbol_summary_to_csv": self._EXPORT_SYMBOL_SUMMARY_TO_CSV,
            "f1_score_plot_filename": self._F1_SCORE_PLOT_FILENAME,
            "invalid_path": self._build_symbol_path(self._SYMBOLS_INVALID_PATH),
            "marketdata_filepath": self._MARKETDATA_FILEPATH,
            "mercado_pago_path": self._build_symbol_path(
                self._SYMBOLS_MERCADO_PAGO_PATH
            ),
            "model_filepath": self._MODEL_FILEPATH,
            "optuna_filepath": self._OPTUNA_FILEPATH,
            "scaler_filepath": self._SCALER_FILEPATH,
            "training_path": self._build_symbol_path(self._SYMBOLS_TRAINING_PATH),
            "xtb_path": self._build_symbol_path(self._SYMBOLS_XTB_PATH),
        }

        return {**vulnerable_params, **dynamic_params, **constant_params, **path_params}

    def get_all(self) -> Any:
        """Return all parameter."""
        return self._parameters

    def get(self, key: str, default: Any = None) -> Any:
        """Return parameter value if exists, else None."""
        try:
            return self._parameters[key]
        except KeyError:
            return default

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access to parameters."""
        return self._parameters[key]
