import time
from typing import List

import pandas as pd

from market_data.gateway import Gateway
from utils.logger import Logger
from utils.parameters import ParameterLoader

_PARAMS = ParameterLoader()
_WEEKDAYS: List[str] = _PARAMS.get("weekdays")
_EXPORT_SYMBOL_SUMMARY_TO_CSV = _PARAMS.get("export_symbol_summary_to_csv")


def print_sumary(symbols, procesed_symbols, start_time) -> None:
    """
    Print a summary report of the symbol update process.

    Args:
        symbols (List[str]): List of all symbols processed.
        procesed_symbols (dict): Dictionary with update counts including keys:
                                 'updated', 'skipped', 'no_new', 'failed'.
        start_time (float): Timestamp indicating when the process started.
    """
    updated = procesed_symbols["updated"]
    skipped = procesed_symbols["skipped"]
    no_new = procesed_symbols["no_new"]
    failed = procesed_symbols["failed"]

    Logger.debug("Summary")
    Logger.debug(f"  * Symbols processed: {len(symbols)}")
    Logger.debug(f"  * Symbols updated: {updated}")
    Logger.debug(f"  * Symbols skipped: {skipped}")
    Logger.debug(f"  * Symbols with no new data: {no_new}")
    if failed > 0:
        Logger.error(f"  * Symbols failed: {failed}")
    total_time = time.time() - start_time
    Logger.debug(f"  Total time: {int(total_time // 60)}m {int(total_time % 60)}s")


def export_symbol_summary_to_csv() -> None:
    """Exports symbol metadata and historical price stats into a CSV file."""
    Gateway.load()
    index = Gateway.get_symbols()
    rows = []

    for symbol, data in index.items():
        row = {
            "symbol": symbol,
            "name": data.get("name"),
            "type": data.get("type"),
            "sector": data.get("sector"),
            "industry": data.get("industry"),
            "currency": data.get("currency"),
            "exchange": data.get("exchange"),
        }

        historical = data.get("historical_prices", [])
        datetimes = pd.to_datetime(
            [p.get("datetime") for p in historical if p.get("datetime")],
            utc=True,
            errors="coerce",
        ).dropna()

        row["num_prices"] = len(datetimes)
        row["min_datetime"] = datetimes.min() if not datetimes.empty else None
        row["max_datetime"] = datetimes.max() if not datetimes.empty else None

        schedule = data.get("schedule", {})
        for day in _WEEKDAYS:
            day_lower = day.lower()
            day_data = schedule[day_lower]
            day_short = day_lower[:3]
            row[f"all_day_{day_short}"] = day_data["all_day"]
            row[f"open_{day_short}"] = day_data["min_open"]
            row[f"close_{day_short}"] = day_data["max_close"]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(_EXPORT_SYMBOL_SUMMARY_TO_CSV, index=False)
