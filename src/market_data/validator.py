from typing import List

import pandas as pd

from market_data.gateway import Gateway
from utils.logger import Logger
from utils.parameters import ParameterLoader

_PARAMS = ParameterLoader()
_REQUIRED_MARKET_COLUMNS = _PARAMS.get("required_market_columns")


def validate_symbols(data: List[dict], required_columns: List[str]):
    """Validates all symbols for structure, columns, and chronological order."""
    success_count = 0
    fail_count = 0
    issues = {}

    for entry in data:
        symbol = entry.get("symbol", "<UNKNOWN>")
        historical_prices = entry.get("historical_prices", [])

        try:
            df = pd.DataFrame(historical_prices)
            if df.empty:
                raise ValueError("Historical prices are empty.")

            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing column: {col}")

            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.sort_values("datetime")

            if not df["datetime"].is_monotonic_increasing:
                raise ValueError("datetime column is not sorted.")

            success_count += 1

        except (ValueError, KeyError, TypeError) as error:
            fail_count += 1
            issues[symbol] = str(error)

    return success_count, fail_count, issues


def validate_market_data():
    """Validates integrity of the market data JSON file."""
    Logger.debug("Validation Summary")

    raw_data = list(Gateway.load()["symbols"].values())
    if not isinstance(raw_data, list):
        Logger.error(
            "Invalid data format for validation. Expected list of symbol entries."
        )
        return

    total_symbols = len(raw_data)
    all_symbols_in_file = {
        entry.get("symbol") for entry in raw_data if "symbol" in entry
    }

    invalid_symbols_in_file = all_symbols_in_file.intersection(
        _PARAMS.symbol_repo.get_invalid_symbols()
    )
    valid_expected_symbols = total_symbols - len(invalid_symbols_in_file)

    Logger.debug(f"  * Total symbols in file: {total_symbols}")
    if valid_expected_symbols != total_symbols:
        Logger.warning(f"  * Expected valid symbols: {valid_expected_symbols}")

    success_symbols, failed_symbols, issues = validate_symbols(
        raw_data, _REQUIRED_MARKET_COLUMNS
    )

    Logger.debug(f"  * Symbols passed: {success_symbols}")
    Logger.debug(f"  * Symbols failed: {failed_symbols}")

    if issues:
        Logger.warning("Issues encountered:")
        for symbol, error in issues.items():
            Logger.error(f"   - {symbol}: {error}")
