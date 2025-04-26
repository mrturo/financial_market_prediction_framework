"""Prediction pipeline for symbol models using shared core prediction utilities."""

import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import pandas as pd

from market_data.gateway import Gateway
from prediction__predict_core import (
    ArtifactLoader,
    Handler,
    PredictionResult,
    calculate_expected_return,
    custom_predict_class,
    filter_market_data_by_symbol,
    get_next_business_day,
    print_prediction_result,
)
from training__train_model import Trainer
from utils.feature_engineering import prepare_raw_dataframe
from utils.parameters import ParameterLoader

# Load parameters and metadata
_PARAMS = ParameterLoader(Gateway.get_last_update())
_XTB_SYMBOLS = _PARAMS.get("xtb_symbols")
_MERCADO_PAGO_SYMBOLS = _PARAMS.get("mercado_pago_symbols")
_FEATURES = _PARAMS.get("features")
_CORRELATIVE_SYMBOLS = _PARAMS.get("correlative_symbols")
_UP_THRESHOLD = _PARAMS.get("up_threshold")
_DOWN_THRESHOLD = _PARAMS.get("down_threshold")
_PREDICTION_WORKERS = _PARAMS.get("prediction_workers")
_MODEL_FILEPATH = _PARAMS.get("model_filepath")
_SYMBOL_COLUMN = "symbol"


# pylint: disable=too-few-public-methods
class SymbolPredictorConfig:
    """Encapsulates configuration parameters for the predictor."""

    def __init__(self):
        self.model = ArtifactLoader.load(_MODEL_FILEPATH)


class SymbolPredictor:
    """Handles predictions across financial symbols using cross-feature modeling."""

    def __init__(self, market_data: Optional[dict] = None):
        self.config = SymbolPredictorConfig()
        self.market_data_dict = market_data or Handler.load_market_data()

    def prepare_features(
        self, symbol: str, current_date: Optional[datetime.date]
    ) -> Optional[pd.DataFrame]:
        """Prepares and enriches feature dataframe for a single symbol."""
        records = filter_market_data_by_symbol(
            self.market_data_dict, symbol, current_date
        )
        if not records:
            return None

        df = pd.DataFrame(records)
        if df.empty:
            return None

        df = prepare_raw_dataframe(df, symbol)

        return df.reset_index()

    def engineer_cross_features(self, full_df: pd.DataFrame) -> pd.DataFrame:
        """Adds cross-symbol features using Trainer."""
        trainer = Trainer()
        return trainer.engineer_cross_features(full_df, _CORRELATIVE_SYMBOLS)

    def _prepare_batch_data(
        self,
        symbols: List[str],
        current_date: Optional[datetime.date],
    ) -> pd.DataFrame:
        """Loads and concatenates prepared DataFrames for each symbol in parallel."""
        processed = []

        def process(symbol):
            df = self.prepare_features(symbol, current_date)
            return df if df is not None and not df.empty else None

        with ThreadPoolExecutor(max_workers=_PREDICTION_WORKERS) as executor:
            futures = {executor.submit(process, s): s for s in symbols}
            for future in as_completed(futures):
                df = future.result()
                if df is not None:
                    processed.append(df)

        return pd.concat(processed) if processed else pd.DataFrame()

    def predict_next_day(
        self,
        symbols: Optional[List[str]] = None,
        current_date: Optional[datetime.date] = None,
    ) -> Optional[PredictionResult]:
        """Predicts the best symbol with a strong UP signal for the next business day."""
        symbols = symbols or list(self.market_data_dict.keys())
        combined_df = self._prepare_batch_data(symbols, current_date)
        if combined_df.empty:
            return None

        combined_df = self.engineer_cross_features(combined_df)
        latest_rows, feature_cols = self._build_feature_matrix(combined_df)
        if latest_rows.empty:
            return None

        input_df = latest_rows[feature_cols]
        predictions = self.config.model.predict_proba(input_df)

        results = self._evaluate_predictions(latest_rows, predictions)
        return max(results, key=lambda r: r.expected_return_score) if results else None

    def print_prediction(self, label: str, symbols: Optional[List[str]] = None):
        """Prints the best UP signal with probability and score."""
        print_prediction_result(label, self.predict_next_day, symbols)

    def _build_feature_matrix(self, combined_df: pd.DataFrame) -> tuple:
        """Filters rows and builds the input feature matrix."""
        cross_features = [
            col
            for col in combined_df.columns
            if col.startswith("spread_vs_") or col.startswith("corr_5d_")
        ]
        all_features = _FEATURES + cross_features
        feature_cols = [col for col in all_features if col in combined_df.columns]

        filtered = combined_df[
            combined_df[feature_cols].notna().sum(axis=1)
            >= int(len(feature_cols) * 0.8)
        ]
        latest_rows = (
            filtered.groupby(_SYMBOL_COLUMN).tail(1).dropna(subset=feature_cols)
        )
        return latest_rows, feature_cols

    def _evaluate_predictions(
        self, latest_rows: pd.DataFrame, predictions: np.ndarray
    ) -> List[PredictionResult]:
        """Generates PredictionResult list for symbols with UP signal."""
        results = []
        for (_, row), probs in zip(latest_rows.iterrows(), predictions):
            if probs[2] <= _UP_THRESHOLD and probs[0] <= _DOWN_THRESHOLD:
                continue
            prediction = custom_predict_class(probs, _UP_THRESHOLD, _DOWN_THRESHOLD)
            if prediction["value"] != 1:
                continue
            expected_return = calculate_expected_return(row.to_frame().T)
            score = expected_return * (probs[2] - max(probs[0], probs[1]))
            results.append(
                PredictionResult(
                    symbol=row[_SYMBOL_COLUMN],
                    next_date=get_next_business_day(row["date"]),
                    direction="UP",
                    expected_return_score=score,
                )
            )
        return results


if __name__ == "__main__":
    predictor = SymbolPredictor()
    predictor.print_prediction("XTB", _XTB_SYMBOLS)
    predictor.print_prediction("MercadoPago", _MERCADO_PAGO_SYMBOLS)
