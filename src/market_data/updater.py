"""Main orchestrator for updating and validating historical market data."""

import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import ta

from market_data.downloader import Downloader
from market_data.gateway import Gateway
from market_data.service import Service
from market_data.summarizer import export_symbol_summary_to_csv, print_sumary
from market_data.validator import validate_market_data
from utils.logger import Logger
from utils.parameters import ParameterLoader

_PARAMS = ParameterLoader()
_BLOCK_DAYS = _PARAMS.get("block_days")
_CALENDAR_NAME = _PARAMS.get("market_calendar_name")
_CUTOFF_MINUTES = _PARAMS.get("cutoff_minutes")
_DOWNLOAD_RETRIES = _PARAMS.get("download_retries")
_INTERVAL = _PARAMS.get("interval")
_RETRY_SLEEP_SECONDS = _PARAMS.get("retry_sleep_seconds")
_SYMBOLS = _PARAMS.get("all_symbols")
_UPDATER_RETRIES = _PARAMS.get("updater_retries")
_VOLUME_WINDOW = _PARAMS.get("volume_window")
_WEEKDAYS: List[str] = _PARAMS.get("weekdays")
_DOWNLOADER = Downloader(_DOWNLOAD_RETRIES, _RETRY_SLEEP_SECONDS)
_SERVICE = Service(_INTERVAL, _BLOCK_DAYS, _DOWNLOADER)


def _is_valid_symbol(symbol: str) -> bool:
    """Checks if the symbol is valid and has market data."""
    try:
        info = _DOWNLOADER.get_metadata(symbol)
        return info["name"] is not None
    except (KeyError, ValueError):
        return False


def _enrich_historical_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches historical prices with computed features."""
    if df.empty or any(col not in df.columns for col in ["close", "open", "volume"]):
        Logger.warning("  Cannot enrich historical prices: essential columns missing.")
        return df

    df = df.sort_values("datetime").copy()
    df.reset_index(drop=True, inplace=True)

    # Basic features
    df["return"] = df["close"].pct_change()
    df["volatility"] = (df["high"] - df["low"]) / df["open"].replace(0, pd.NA)
    df["price_change"] = df["close"] - df["open"]
    df["volume_change"] = df["volume"].pct_change()
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["average_price"] = (df["high"] + df["low"]) / 2
    epsilon = 1e-6  # Ajusta según la escala de tus datos
    conditions = [
        (df["close"] - df["open"]) > epsilon,
        (df["open"] - df["close"]) > epsilon,
    ]
    choices = [1, -1]
    df["candle"] = np.select(conditions, choices, default=0)
    df["range"] = df["high"] - df["low"]
    df["relative_volume"] = (
        df["volume"] / df["volume"].rolling(window=_VOLUME_WINDOW, min_periods=1).mean()
    )

    # New features
    df["atr_14"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()

    df["overnight_return"] = df["open"].pct_change().fillna(0)
    df["intraday_return"] = (df["close"] / df["open"]) - 1

    df["volume_rvol_20d"] = (
        df["volume"] / df["volume"].rolling(window=20, min_periods=1).mean()
    )
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["close"], volume=df["volume"]
    ).on_balance_volume()

    adx = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )
    df["adx_14"] = adx.adx()

    boll = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bollinger_pct_b"] = boll.bollinger_pband()

    df["pattern_bullish_engulfing"] = (
        (df["close"] > df["open"])
        & (df["close"].shift(1) < df["open"].shift(1))
        & (df["close"] > df["open"].shift(1))
        & (df["open"] < df["close"].shift(1))
    ).astype(int)
    df["pattern_bearish_engulfing"] = (
        (df["close"] < df["open"])
        & (df["close"].shift(1) > df["open"].shift(1))
        & (df["open"] > df["close"].shift(1))
        & (df["close"] < df["open"].shift(1))
    ).astype(int)

    max_52w = df["close"].rolling(window=252, min_periods=1).max()
    df["pct_from_52w_high"] = (max_52w - df["close"]) / max_52w

    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)

    Logger.debug("  Enriched historical prices with additional metadata.")
    return df


def _should_skip_symbol(symbol: str, entry: dict, invalid_symbols: set) -> bool:
    """Determines whether a symbol should be skipped."""
    if not (entry and entry.get("historical_prices")):
        if symbol in invalid_symbols:
            Logger.warning(f"  Skipping {symbol}: previously marked as invalid.")
            return True
        if not _is_valid_symbol(symbol):
            Logger.warning(f"  Skipping {symbol}: appears invalid or delisted.")
            invalid_symbols.add(symbol)
            return True
    return False


def _generate_daily_ranges(df: pd.DataFrame) -> dict:
    """Groups historical data by date and computes open/close times."""
    daily_ranges = defaultdict(lambda: {"open": None, "close": None})
    for date, group in df.groupby("date"):
        open_dt = group["datetime"].min()
        close_dt = group["datetime"].max() + timedelta(hours=1)
        daily_ranges[date]["open"] = open_dt.time()
        daily_ranges[date]["close"] = close_dt.time()
    return daily_ranges


def _compute_open_close_extremes(daily_ranges: dict) -> tuple:
    """Computes the earliest open and latest close time."""
    base_date = datetime(2000, 1, 1)
    opens = []
    closes = []
    for v in daily_ranges.values():
        open_dt = datetime.combine(base_date, v["open"])
        close_dt = datetime.combine(base_date, v["close"])
        if close_dt <= open_dt:
            close_dt += timedelta(days=1)
        opens.append(open_dt)
        closes.append(close_dt)
    min_open = min(opens).time().strftime("%H:%M:%S") if opens else None
    max_close = max(closes).time().strftime("%H:%M:%S") if closes else None
    return min_open, max_close


def _group_schedule_variants(daily_ranges: dict) -> list:
    """Groups date ranges by open-close combinations and classifies habitual ranges."""
    range_groups = defaultdict(list)
    for date, times in daily_ranges.items():
        key = (
            times["open"].strftime("%H:%M:%S"),
            times["close"].strftime("%H:%M:%S"),
        )
        range_groups[key].append(str(date))

    total_days = sum(len(dates) for dates in range_groups.values())
    threshold = 0.2 * total_days

    return [
        {
            "open": open_str,
            "close": close_str,
            "dates": dates,
            "habitual": len(dates) >= threshold,
        }
        for (open_str, close_str), dates in range_groups.items()
    ]


def build_schedule(entry: dict) -> dict:
    """Constructs a trading schedule per weekday based on the symbol's historical timestamps."""
    df = pd.DataFrame(entry["historical_prices"])
    result = {}

    if "datetime" in df.columns and not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df.dropna(subset=["datetime"], inplace=True)
        now = datetime.now(timezone.utc)
        latest_dt = df["datetime"].max()

        if latest_dt.date() == now.date():
            elapsed_minutes = (now - latest_dt).total_seconds() / 60.0
            if elapsed_minutes < _CUTOFF_MINUTES:
                df = df[df["datetime"].dt.date < now.date()]

        df["weekday"] = df["datetime"].dt.day_name()
        df["date"] = df["datetime"].dt.date

        for day in _WEEKDAYS:
            lower_day = day.lower()
            day_data = df[df["weekday"] == day]

            if day_data.empty:
                result[lower_day] = {
                    "all_day": None,
                    "min_open": None,
                    "max_close": None,
                    "schedules": [],
                }
                continue

            daily_ranges = _generate_daily_ranges(day_data)
            min_open, max_close = _compute_open_close_extremes(daily_ranges)
            schedule_variants = _group_schedule_variants(daily_ranges)

            result[lower_day] = {
                "all_day": (
                    min_open == max_close and min_open == "00:00:00"
                    if min_open and max_close
                    else None
                ),
                "min_open": min_open,
                "max_close": max_close,
                "schedules": schedule_variants,
            }

    return result


def estimate_total_business_days_month(
    year: int, month: int, calendar_type: int
) -> int:
    """Estimates number of business days in a given month according to a calendar."""
    start = pd.Timestamp(f"{year}-{month:02d}-01")
    end = (start + pd.offsets.MonthEnd(1)).normalize()
    result = np.nan

    if calendar_type == 5:
        nyse = mcal.get_calendar(_CALENDAR_NAME)
        schedule = nyse.schedule(start_date=start, end_date=end)
        result = len(schedule)
    elif calendar_type == 6:
        result = sum(
            (start + pd.Timedelta(days=i)).weekday() < 6
            for i in range((end - start).days + 1)
        )
    elif calendar_type == 7:
        result = (end - start).days + 1

    return result


def estimate_total_business_days_year(year: int, calendar_type: int) -> int:
    """Estimate the number of business days in a given year based on calendar type.

    Args:
        year (int): The year for which to estimate business days.
        calendar_type (int): Type of calendar (5=NYSE, 6=Mon-Sat, 7=all days).

    Returns:
        int: Estimated number of business days in the year.
    """
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    result = np.nan

    if calendar_type == 5:
        cal = mcal.get_calendar(_CALENDAR_NAME)
        schedule = cal.schedule(start_date=start, end_date=end)
        result = len(schedule)
    elif calendar_type == 6:
        result = sum(
            (start + pd.Timedelta(days=i)).weekday() < 6
            for i in range((end - start).days + 1)
        )
    elif calendar_type == 7:
        result = (end - start).days + 1

    return result


def _get_business_days(
    start: pd.Timestamp, end: pd.Timestamp, calendar_type: int
) -> list:
    if calendar_type == 5:
        cal = mcal.get_calendar(_CALENDAR_NAME)
        schedule = cal.schedule(start_date=start, end_date=end)
        return pd.to_datetime(schedule.index).date
    if calendar_type == 6:
        return [
            (start + pd.Timedelta(days=i)).date()
            for i in range((end - start).days + 1)
            if (start + pd.Timedelta(days=i)).weekday() < 6
        ]
    if calendar_type == 7:
        return [
            (start + pd.Timedelta(days=i)).date() for i in range((end - start).days + 1)
        ]
    return []


def _build_day_index_maps(
    unique_dates: pd.DataFrame, calendar_type: int
) -> tuple[dict, dict]:
    month_day_map = {}
    year_day_map = {}

    for (year, month), _ in unique_dates.groupby(["local_year", "local_month"]):
        start = pd.Timestamp(f"{year}-{month:02d}-01")
        end = (start + pd.offsets.MonthEnd(1)).normalize()
        business_days = _get_business_days(start, end, calendar_type)
        month_day_map[(year, month)] = {
            d: i + 1 for i, d in enumerate(sorted(business_days))
        }

    for year, _ in unique_dates.groupby("local_year"):
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        business_days = _get_business_days(start, end, calendar_type)
        year_day_map[year] = {d: i + 1 for i, d in enumerate(sorted(business_days))}

    return month_day_map, year_day_map


def _merge_and_estimate_totals(
    unique_dates: pd.DataFrame, calendar_type: int
) -> pd.DataFrame:
    first_period = unique_dates[["local_year", "local_month"]].iloc[0]
    last_period = unique_dates[["local_year", "local_month"]].iloc[-1]

    total_counts = (
        unique_dates.groupby(["local_year", "local_month"])["local_month_days_current"]
        .max()
        .reset_index()
        .rename(columns={"local_month_days_current": "local_month_days_total"})
    )
    for period in [first_period, last_period]:
        y, m = period["local_year"], period["local_month"]
        total_counts.loc[
            (total_counts["local_year"] == y) & (total_counts["local_month"] == m),
            "local_month_days_total",
        ] = estimate_total_business_days_month(y, m, calendar_type)

    year_counts = (
        unique_dates.groupby("local_year")["local_year_days_current"]
        .max()
        .reset_index()
        .rename(columns={"local_year_days_current": "local_year_days_total"})
    )
    for y in unique_dates["local_year"].unique():
        year_counts.loc[year_counts["local_year"] == y, "local_year_days_total"] = (
            estimate_total_business_days_year(y, calendar_type)
        )

    merged = pd.merge(
        unique_dates, total_counts, on=["local_year", "local_month"], how="left"
    )
    return pd.merge(merged, year_counts, on="local_year", how="left")


def add_month_and_year_days(df: pd.DataFrame, calendar_type: int = 7) -> pd.DataFrame:
    """Add month/day progress tracking columns using calendar type logic."""
    df = df.copy()
    df["local_datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["local_date"] = df["local_datetime"].dt.date
    df["local_year"] = df["local_datetime"].dt.year
    df["local_month"] = df["local_datetime"].dt.month

    unique_dates = (
        df[["local_year", "local_month", "local_date"]]
        .drop_duplicates()
        .sort_values("local_date")
    )

    month_day_map, year_day_map = _build_day_index_maps(unique_dates, calendar_type)

    unique_dates["local_month_days_current"] = unique_dates.apply(
        lambda row: month_day_map.get((row["local_year"], row["local_month"]), {}).get(
            row["local_date"], np.nan
        ),
        axis=1,
    )
    unique_dates["local_year_days_current"] = unique_dates.apply(
        lambda row: year_day_map.get(row["local_year"], {}).get(
            row["local_date"], np.nan
        ),
        axis=1,
    )

    merged = _merge_and_estimate_totals(unique_dates, calendar_type)

    df = df.merge(merged, on=["local_year", "local_month", "local_date"], how="left")
    df["month_days_current"] = df["local_month_days_current"]
    df["month_days_total"] = df["local_month_days_total"]
    df["year_days_current"] = df["local_year_days_current"]
    df["year_days_total"] = df["local_year_days_total"]
    df.drop(
        columns=[
            "local_year",
            "local_month",
            "local_date",
            "local_datetime",
            "local_month_days_current",
            "local_month_days_total",
            "local_year_days_current",
            "local_year_days_total",
        ],
        inplace=True,
    )
    return df


def process_single_symbol(symbol: str, entry: dict, service: Service) -> tuple:
    """Processes a single symbol and returns result tuple."""
    existing_df = pd.DataFrame(entry["historical_prices"]) if entry else pd.DataFrame()
    existing_metadata = (
        {
            k: entry.get(k)
            for k in ["name", "type", "sector", "industry", "currency", "exchange"]
        }
        if entry
        else {}
    )

    if existing_df.empty or "datetime" not in existing_df.columns:
        last_dt = None
    else:
        existing_df["datetime"] = pd.to_datetime(
            existing_df["datetime"], utc=True, errors="coerce"
        )
        last_dt = existing_df["datetime"].max()

    incremental = service.get_incremental_data(symbol, last_dt)

    if not incremental.empty and not existing_df.empty:
        incremental = incremental[
            ~incremental["datetime"].isin(existing_df["datetime"])
        ].reset_index(drop=True)
        existing_df = existing_df.reset_index(drop=True)

    if incremental.empty:
        updated_entry = {"historical_prices": existing_df.to_dict(orient="records")}
        updated_entry["schedule"] = build_schedule(updated_entry)
        existing_metadata["schedule"] = updated_entry["schedule"]
        Gateway.set_symbol(
            symbol, updated_entry["historical_prices"], existing_metadata
        )
        return "no_new", existing_metadata["name"]

    incremental = (
        incremental.loc[:, ~incremental.columns.duplicated()]
        .copy()
        .reset_index(drop=True)
    )
    existing_df = (
        existing_df.loc[:, ~existing_df.columns.duplicated()]
        .copy()
        .reset_index(drop=True)
    )

    combined_df = pd.concat([incremental, existing_df], axis=0, copy=False)
    combined_df = (
        combined_df.drop_duplicates(subset=["datetime"], keep="first")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    combined_df = _enrich_historical_prices(combined_df)

    # Add weekday, record number in day, and total records in day
    combined_df["datetime"] = pd.to_datetime(
        combined_df["datetime"], utc=True, errors="coerce"
    )
    combined_df["date"] = combined_df["datetime"].dt.date

    # Compute record number and total records per day
    combined_df["records_day_current"] = combined_df.groupby("date").cumcount() + 1
    combined_df["records_day_total"] = combined_df.groupby("date")[
        "datetime"
    ].transform("count")

    combined_df["weekdays_current"] = combined_df["datetime"].dt.isocalendar().day
    combined_df["weekdays_total"] = combined_df["datetime"].dt.dayofweek.nunique()

    # Deduce calendar type from unique weekdays used
    active_weekdays = combined_df["datetime"].dt.dayofweek.unique()
    if set(active_weekdays) == set(range(7)):
        calendar_type = 7
    elif set(active_weekdays) == set(range(6)):
        calendar_type = 6
    else:
        calendar_type = 5

    combined_df = add_month_and_year_days(combined_df, calendar_type)

    combined_df["pct_day"] = (
        combined_df["records_day_current"] / combined_df["records_day_total"]
    )
    combined_df["pct_week"] = (
        (combined_df["records_day_total"] * (combined_df["weekdays_current"] - 1))
        + combined_df["records_day_current"]
    ) / (combined_df["records_day_total"] * combined_df["weekdays_total"])
    combined_df["pct_month"] = (
        (combined_df["records_day_total"] * (combined_df["month_days_current"] - 1))
        + combined_df["records_day_current"]
    ) / (combined_df["records_day_total"] * combined_df["month_days_total"])
    combined_df["pct_year"] = (
        (combined_df["records_day_total"] * (combined_df["year_days_current"] - 1))
        + combined_df["records_day_current"]
    ) / (combined_df["records_day_total"] * combined_df["year_days_total"])

    # Format datetime as string for JSON export
    combined_df["datetime"] = combined_df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    combined_df = combined_df.drop(columns=["date"])

    metadata = (
        _DOWNLOADER.get_metadata(symbol)
        if not existing_metadata.get("name")
        else existing_metadata
    )
    updated_entry = {"historical_prices": combined_df.to_dict(orient="records")}
    updated_entry["schedule"] = build_schedule(updated_entry)
    metadata["schedule"] = updated_entry["schedule"]

    Gateway.set_symbol(symbol, updated_entry["historical_prices"], metadata)

    Logger.debug(f"  Updated data for {symbol} with {len(incremental)} new rows.")
    return "updated", metadata["name"]


def process_symbols(symbols: List[str], invalid_symbols: List[str], service: Service):
    """
    Download and update historical market data for a list of symbols.

    Args:
        symbols (List[str]): List of ticker symbols to process.
        invalid_symbols (set): Set of previously invalid or skipped symbols.
        service (Service): Service instance to manage downloading and enrichment.

    Returns:
        dict: Summary statistics including updated, skipped, no_new, failed counts,
              and the final set of invalid symbols.
    """
    counts = {"updated": 0, "skipped": 0, "no_new": 0, "failed": 0}
    Gateway.load()
    invalid_symbols_set = set(invalid_symbols)

    for idx, symbol in enumerate(symbols, start=1):
        Logger.info(
            f"Processing symbol ({idx}/{len(symbols)} → "
            f"{(((idx-1)*100)/len(symbols)):.2f}%): {symbol}"
        )
        entry = Gateway.get_symbol(symbol)
        symbol_start_time = time.time()

        if _should_skip_symbol(symbol, entry, invalid_symbols_set):
            counts["skipped"] += 1
            continue

        try:
            result, name = process_single_symbol(symbol, entry, service)
            counts[result] += 1
            name = name.strip() if name is not None else ""
            name_symbol = f"{symbol} ({name})" if len(name) > 0 else symbol
            Logger.success(
                f"  Completed {name_symbol} in "
                f"{(time.time() - symbol_start_time):.2f} seconds."
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            Logger.error(f"  Failed to update {symbol}: {error}")
            counts["failed"] += 1

    Gateway.save()
    counts["invalid_symbols"] = invalid_symbols_set
    return counts


def _perform_single_update(symbols: List[str], local_start_time: float) -> dict:
    """Executes a single update attempt and logs summary."""
    symbol_repository = _PARAMS.symbol_repo
    invalid_symbols = set(symbol_repository.get_invalid_symbols())
    processed_symbols = process_symbols(
        symbols,
        invalid_symbols,
        _SERVICE,
    )
    symbol_repository.set_invalid_symbols(processed_symbols["invalid_symbols"])
    print_sumary(symbols, processed_symbols, local_start_time)
    Logger.separator()
    return processed_symbols


def update_data(in_max_retries: int = 0):
    """Updates all symbols' historical market data and enriches them."""
    start_time = time.time()

    max_retries = (
        _UPDATER_RETRIES
        if in_max_retries <= 0 or in_max_retries is None
        else in_max_retries
    )
    max_retries = max(1, max_retries)

    current_try = 0
    while current_try < max_retries:
        current_try += 1
        local_start_time = time.time()

        if max_retries > 1:
            Logger.info(f"Update try No. {current_try}/{max_retries}:")

        processed_symbols = _perform_single_update(_SYMBOLS, local_start_time)

        if current_try >= max_retries or processed_symbols.get("updated", 0) == 0:
            total_time = time.time() - start_time
            Logger.success(
                f"Update process is completed. Number of tries: {current_try}. "
                f"Total time: {int(total_time // 60)}m {int(total_time % 60)}s"
            )
            latest_price_date = Gateway.get_latest_price_date()
            stale_symbols = Gateway.get_stale_symbols()
            Logger.debug(f"Latest price date: {latest_price_date}")
            Logger.debug(f"Stale symbols: {stale_symbols}")
            Logger.separator()
            break

    validate_market_data()


if __name__ == "__main__":
    update_data()
    export_symbol_summary_to_csv()
