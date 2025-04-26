"""Market data gateway for managing symbol metadata and historical prices."""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from market_data.schedule_builder import ScheduleBuilder
from utils.json_manager import JsonManager
from utils.parameters import ParameterLoader


class Gateway:
    """Centralized access point for managing symbol data and market schedules."""

    _PARAMS = ParameterLoader()
    _STALE_DAYS_THRESHOLD = _PARAMS.get("stale_days_threshold")
    _SYMBOL_REPO = _PARAMS.symbol_repo
    _WEEKDAYS = _PARAMS.get("weekdays")
    _MARKETDATA_FILEPATH = _PARAMS.get("marketdata_filepath")

    _symbols: Dict[str, dict] = {}
    _last_updated: Optional[pd.Timestamp] = None
    _latest_price_date: Optional[pd.Timestamp] = None
    _stale_symbols: List[str] = []
    _global_schedule: Dict[str, dict] = {}

    @staticmethod
    def _get_filepath(filepath: str, default=None) -> str:
        if filepath is None or len(filepath.strip()) == 0:
            return default
        return filepath

    @staticmethod
    def get_symbols() -> Dict[str, dict]:
        """Return all loaded symbols."""
        return Gateway._symbols

    @staticmethod
    def set_symbols(symbols: Dict[str, dict]) -> None:
        """Set the dictionary of symbols."""
        Gateway._symbols = symbols

    @staticmethod
    def get_symbol(symbol: str):
        """Retrieve metadata for a specific symbol."""
        return Gateway._symbols.get(symbol)

    @staticmethod
    def set_symbol(symbol: str, historical_prices: List[dict], metadata: dict):
        """Set symbol metadata and historical prices."""
        Gateway._symbols[symbol] = {
            "symbol": symbol,
            "name": metadata.get("name"),
            "type": metadata.get("type"),
            "sector": metadata.get("sector"),
            "industry": metadata.get("industry"),
            "currency": metadata.get("currency"),
            "exchange": metadata.get("exchange"),
            "schedule": metadata.get("schedule"),
            "historical_prices": historical_prices,
        }

    @staticmethod
    def get_last_update() -> pd.Timestamp:
        """Return the last update timestamp."""
        return Gateway._last_updated

    @staticmethod
    def get_latest_price_date() -> Optional[pd.Timestamp]:
        """Return the latest price timestamp among all symbols."""
        return Gateway._latest_price_date

    @staticmethod
    def set_latest_price_date(date: Optional[pd.Timestamp]) -> None:
        """Manually set the latest price date."""
        Gateway._latest_price_date = date

    @staticmethod
    def get_stale_symbols() -> List[str]:
        """Return the list of stale symbols."""
        return Gateway._stale_symbols

    @staticmethod
    def set_stale_symbols(stale_symbols: List[str]) -> None:
        """Manually set stale symbols."""
        Gateway._stale_symbols = stale_symbols

    @staticmethod
    def get_global_schedule() -> Dict[str, dict]:
        """Return the globalschedule of symbols."""
        return Gateway._global_schedule

    @staticmethod
    def invalid_symbols() -> set:
        """Return the set of currently invalid symbols."""
        return Gateway._SYMBOL_REPO.get_invalid_symbols()

    @staticmethod
    def latest_price_date() -> pd.Timestamp:
        """Alias to get the latest price date."""
        return Gateway._latest_price_date

    @staticmethod
    def load(filepath: str = None) -> Dict:
        """Load market data from disk and initialize internal structures."""
        filepath = Gateway._get_filepath(filepath, Gateway._MARKETDATA_FILEPATH)
        raw_data = JsonManager.load(filepath)
        raw_last_updated: pd.Timestamp = None
        symbols: list = []

        if raw_data is not None:
            raw_last_updated = raw_data["last_updated"]
            symbols = raw_data["symbols"]

        Gateway._last_updated = (
            pd.to_datetime(raw_last_updated)
            if raw_last_updated is not None and isinstance(raw_last_updated, str)
            else pd.Timestamp.now(tz="UTC")
        )

        enriched = {}
        for entry in symbols:
            symbol = entry.get("symbol")
            df = pd.DataFrame(entry["historical_prices"])
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            entry["historical_prices"] = df.to_dict(orient="records")
            enriched[symbol] = entry

        Gateway._symbols = {
            sym: data
            for sym, data in enriched.items()
            if sym not in Gateway._SYMBOL_REPO.get_invalid_symbols()
        }
        Gateway._update_latest_price_date()
        Gateway._update_stale_symbols()
        Gateway._update_global_schedule()
        return {
            "last_updated": Gateway._last_updated,
            "schedule": Gateway._global_schedule,
            "symbols": Gateway._symbols,
        }

    @staticmethod
    def save(filepath: str = None) -> None:
        """Persist current symbol data to disk."""
        result = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "schedule": Gateway._global_schedule,
            "symbols": list(Gateway._symbols.values()),
        }
        filepath = Gateway._get_filepath(filepath, Gateway._MARKETDATA_FILEPATH)
        JsonManager.save(result, filepath)
        return result

    @staticmethod
    def _update_latest_price_date() -> None:
        """Update the latest price date across all symbols."""
        max_date = None
        for entry in Gateway._symbols.values():
            datetimes = [
                pd.to_datetime(p["datetime"], utc=True, errors="coerce")
                for p in entry.get("historical_prices", [])
            ]
            entry_max = max(filter(pd.notnull, datetimes), default=None)
            if entry_max:
                max_date = max(max_date or entry_max, entry_max)
        Gateway._latest_price_date = max_date

    @staticmethod
    def _update_stale_symbols() -> None:
        """Identify and mark symbols with outdated price data."""
        latest_date = Gateway._latest_price_date or pd.Timestamp.now(tz="UTC")
        current_invalids = set(Gateway._SYMBOL_REPO.get_invalid_symbols())

        stale, updated_invalids = Gateway._detect_stale_symbols(
            latest_date, current_invalids
        )

        Gateway._stale_symbols = stale
        Gateway._SYMBOL_REPO.set_invalid_symbols(updated_invalids)

    @staticmethod
    def _detect_stale_symbols(
        latest_date: pd.Timestamp, invalid_symbols: set
    ) -> tuple[list, set]:
        """Detect stale symbols and return updated lists (for internal testing)."""
        stale = []
        for symbol, entry in Gateway._symbols.items():
            datetimes = [
                pd.to_datetime(p["datetime"], utc=True, errors="coerce")
                for p in entry.get("historical_prices", [])
            ]
            symbol_latest = max(filter(pd.notnull, datetimes), default=None)
            if (
                not symbol_latest
                or (latest_date - symbol_latest).days > Gateway._STALE_DAYS_THRESHOLD
            ):
                stale.append(symbol)
                invalid_symbols.add(symbol)
        return stale, invalid_symbols

    @staticmethod
    def _update_global_schedule() -> None:
        """Construct a combined trading schedule from all symbols."""

        def collect_variants(symbols: List[dict], weekdays: List[str]) -> dict:
            variants = {day.lower(): {} for day in weekdays}
            for entry in symbols:
                symbol = entry.get("symbol")
                schedule = entry.get("schedule", {})
                for day in weekdays:
                    sch = schedule.get(day.lower(), {})
                    key = (sch.get("min_open"), sch.get("max_close"))
                    if key[0] and key[1]:
                        variants[day.lower()].setdefault(key, []).append(symbol)
            return variants

        def build_day_schedule(variants: dict) -> dict:
            opens, closes, schedules = [], [], []
            for (open_str, close_str), symbols in variants.items():
                open_dt = datetime.strptime(open_str, "%H:%M:%S")
                close_dt = datetime.strptime(close_str, "%H:%M:%S")
                if close_dt <= open_dt:
                    close_dt += timedelta(days=1)
                opens.append(open_dt)
                closes.append(close_dt)
                schedules.append(
                    {
                        "all_day": (
                            open_str == close_str and open_str == "00:00:00"
                            if open_str and close_str
                            else None
                        ),
                        "open": open_str,
                        "close": close_str,
                        "symbols": sorted(symbols),
                        "open_hours": ScheduleBuilder.calc_open_hours(
                            open_str, close_str
                        ),
                    }
                )

            min_open = min(opens).strftime("%H:%M:%S") if opens else None
            max_close = max(closes).time().strftime("%H:%M:%S") if closes else None

            return ScheduleBuilder.build_schedule_sumary(
                min_open, max_close, schedules or [{"all_day": None}]
            )

        variants_by_day = collect_variants(
            list(Gateway._symbols.values()), Gateway._WEEKDAYS
        )
        Gateway._global_schedule = {
            day.lower(): build_day_schedule(variants_by_day[day.lower()])
            for day in Gateway._WEEKDAYS
        }
