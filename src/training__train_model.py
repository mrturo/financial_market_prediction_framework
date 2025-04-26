"""
Training pipeline for price movement prediction.

Implements data preparation, cross-feature engineering, filtering, and training with grouped
cross-validation using the CoreTrainer base class.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from market_data.gateway import Gateway
from training__train_core import CoreTrainer, apply_cutoff_filters
from utils.feature_engineering import prepare_raw_dataframe
from utils.logger import Logger
from utils.parameters import ParameterLoader

# Constantes
SYMBOL_COLUMN = "symbol"
DATETIME_COLUMN = "datetime"
TARGET_COLUMN = "target"


class Trainer(CoreTrainer):
    """Trainer for symbol classifiers with feature engineering and grouped validation."""

    _PARAMS = ParameterLoader()
    _CUTOFF_DATE = _PARAMS.get("cutoff_date")
    _CUTOFF_MINUTES = _PARAMS.get("cutoff_minutes")
    _MARKET_TZ = _PARAMS.get("market_tz")

    def __init__(self):
        super().__init__()
        self.cross_features = []

    def load_combined_dataset(self) -> pd.DataFrame:
        """Load symbol data with feature engineering and cutoff filtering."""
        market_data = Gateway.load()
        combined = []

        cutoff_date = pd.to_datetime(Trainer._CUTOFF_DATE)

        if pd.isna(cutoff_date):
            cutoff_date = None
        elif pd.notna(cutoff_date) and cutoff_date.tzinfo is None:
            cutoff_date = cutoff_date.tz_localize(Trainer._MARKET_TZ)

        for entry in market_data:
            symbol = entry["symbol"]
            df = pd.DataFrame(entry["historical_prices"])

            if df.empty or not {"close", "high", "low", "volume"}.issubset(df.columns):
                Logger.warning(f"  Skipping {symbol}: insufficient data.")
                continue

            try:
                df = apply_cutoff_filters(
                    df,
                    cutoff_from=None,
                    cutoff_to=cutoff_date,
                    timezone=Trainer._MARKET_TZ,
                    cutoff_minutes=Trainer._CUTOFF_MINUTES,
                )
            except (KeyError, TypeError, ValueError) as e:
                Logger.warning(
                    f"  Skipping {symbol}: error during cutoff filtering: {e}"
                )
                continue

            required_columns = {"close", "high", "low", "volume"}
            if df.empty or not required_columns.issubset(df.columns):
                Logger.warning(
                    f"  Skipping {symbol}: data missing after cutoff filtering."
                )
                continue

            try:
                df = prepare_raw_dataframe(df, symbol)
            except (KeyError, TypeError, ValueError) as e:
                Logger.warning(
                    f"  Skipping {symbol}: error during feature engineering: {e}"
                )
                continue

            combined.append(df)

        if not combined:
            raise ValueError("No symbol had valid data after applying cutoff filters.")

        return pd.concat(combined).reset_index()

    def engineer_cross_features(
        self, df: pd.DataFrame, base_symbols: list
    ) -> pd.DataFrame:
        """Agrega spreads y correlaciones entre activos."""
        df = df.copy()
        df.set_index(DATETIME_COLUMN, inplace=True)

        for base in base_symbols:
            base_df = df[df[SYMBOL_COLUMN] == base]
            if base_df.empty or "return_1h" not in base_df.columns:
                continue

            spread_col = f"spread_vs_{base}".lower()
            corr_col = f"corr_5d_{base}".lower()
            return_col = f"{base}_return".lower()

            base_returns = base_df["return_1h"].rename(return_col)
            df = df.join(base_returns, on=DATETIME_COLUMN)

            if df[return_col].isna().all():
                Logger.warning(
                    f"⚠️ All values are NaN for {base} — skipping {spread_col} and {corr_col}."
                )
                df.drop(columns=[return_col], inplace=True)
                continue

            df[spread_col] = df["return_1h"] - df[return_col]
            df[corr_col] = (
                df["return_1h"]
                .rolling(window=5)
                .corr(df[return_col])
                .astype(np.float32)
            )

            df.drop(columns=[return_col], inplace=True)

        df.reset_index(inplace=True)
        return df

    def get_valid_cross_features(self, df: pd.DataFrame) -> list:
        """Filtra los features de cruce que no son todos NaN."""
        features = []
        for base in Trainer._PARAMS.get("correlative_symbols"):
            spread = f"spread_vs_{base}".lower()
            corr = f"corr_5d_{base}".lower()
            if spread in df.columns and not df[spread].isna().all():
                features.append(spread)
            else:
                Logger.warning(f"Skipping {spread} — missing or all NaN")
            if corr in df.columns and not df[corr].isna().all():
                features.append(corr)
            else:
                Logger.warning(f"Skipping {corr} — missing or all NaN")
        return features

    def filter_valid_cross_features(
        self, df: pd.DataFrame, features: list, min_rows_required: int = 500
    ) -> list:
        """Filtra features con pocos valores válidos."""
        valid = []
        for feat in features:
            count = df[feat].notna().sum()
            if count >= min_rows_required:
                valid.append(feat)
            else:
                Logger.warning(
                    f"⚠️ Dropping {feat} — only {count} valid rows"
                    f" (min required: {min_rows_required})"
                )
        return valid

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, Optional[np.ndarray]]:
        Logger.info("🔄 Loading and processing multiactive dataset...")
        df = self.load_combined_dataset()
        df = self.engineer_cross_features(
            df, Trainer._PARAMS.get("correlative_symbols")
        )

        # Selección de features de cruce válidos
        self.cross_features = self.get_valid_cross_features(df)
        Logger.info(f"🧩 Initial cross features: {self.cross_features}")
        self.cross_features = self.filter_valid_cross_features(df, self.cross_features)
        Logger.info(f"✅ Final cross features: {self.cross_features}")

        feature_cols = (
            Trainer._PARAMS.get("features") + self.cross_features + [TARGET_COLUMN]
        )
        df = df[[col for col in feature_cols + [SYMBOL_COLUMN] if col in df.columns]]

        # Validación por cantidad de NaNs por fila
        min_valid_cols = int(len(feature_cols) * 0.8)
        df = df[df[feature_cols].notna().sum(axis=1) >= min_valid_cols]
        if df.empty:
            raise ValueError("No data left after filtering NaNs by row threshold.")

        # Eliminar símbolos con pocas muestras
        symbol_counts = df[SYMBOL_COLUMN].value_counts()
        valid_symbols = symbol_counts[
            symbol_counts >= Trainer._PARAMS.get("n_splits")
        ].index
        df = df[df[SYMBOL_COLUMN].isin(valid_symbols)]
        if df.empty:
            raise ValueError("No symbols with enough data to support GroupKFold.")

        x = df[Trainer._PARAMS.get("features") + self.cross_features + [SYMBOL_COLUMN]]
        y = df[TARGET_COLUMN].astype(int)
        groups = df[SYMBOL_COLUMN]

        return x, y, groups


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
