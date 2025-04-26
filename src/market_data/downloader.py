"""Downloader module for historical market data."""

import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from market_data.provider import PriceDataConfig, Provider, TickerMetadata
from utils.logger import Logger
from utils.output_suppressor import OutputSuppressor
from utils.parameters import ParameterLoader
from utils.timezone import localize_to_market_time


class Downloader:
    """Handles downloading of market data."""

    _PARAMS = ParameterLoader()
    _MARKET_TZ = _PARAMS.get("market_tz")

    # Market data provider instance
    _PROVIDER = Provider()

    def _build_ticker_name(self, ticker: TickerMetadata) -> str:
        """
        Build the standardized name of a ticker from its metadata.

        Parameters:
            metadata (dict): Dictionary containing ticker metadata.

        Returns:
            str: Ticker's display name or short name.
        """
        symbol = (
            ticker.symbol.strip().upper()
            if ticker.symbol is not None and len(ticker.symbol.strip()) > 0
            else None
        )
        display_name = (
            ticker.display_name.strip()
            if ticker.display_name is not None and len(ticker.display_name.strip()) > 0
            else None
        )
        short_name = (
            ticker.short_name.strip()
            if ticker.short_name is not None and len(ticker.short_name.strip()) > 0
            else None
        )
        long_name = (
            ticker.long_name.strip()
            if ticker.long_name is not None and len(ticker.long_name.strip()) > 0
            else None
        )
        return (
            display_name
            if display_name is not None
            else (
                short_name
                if short_name is not None
                else (long_name if long_name is not None else symbol)
            )
        )

    def _build_ticker_type(self, ticker: TickerMetadata) -> str:
        """
        Extract the ticker type (e.g., 'Equity', 'ETF') from metadata.

        Parameters:
            metadata (dict): Dictionary containing ticker metadata.

        Returns:
            str: Ticker type description.
        """
        type_disp = (
            ticker.type_disp.strip().lower()
            if ticker.type_disp is not None and len(ticker.type_disp.strip()) > 0
            else None
        )
        quote_type = (
            ticker.quote_type.strip()
            if ticker.quote_type is not None and len(ticker.quote_type.strip()) > 0
            else None
        )
        return type_disp if type_disp is not None else quote_type

    def _build_ticker_sector(self, ticker: TickerMetadata) -> str:
        """
        Retrieve the sector classification of the ticker.

        Parameters:
            metadata (dict): Dictionary containing ticker metadata.

        Returns:
            str: Sector name (e.g., 'Technology', 'Healthcare').
        """
        sector = (
            ticker.sector.strip()
            if ticker.sector is not None and len(ticker.sector.strip()) > 0
            else None
        )
        sector_key = (
            ticker.sector_key.strip()
            if ticker.sector_key is not None and len(ticker.sector_key.strip()) > 0
            else None
        )
        sector_disp = (
            ticker.sector_disp.strip()
            if ticker.sector_disp is not None and len(ticker.sector_disp.strip()) > 0
            else None
        )
        return (
            sector_key
            if sector_key is not None
            else (sector_disp if sector_disp is not None else sector)
        )

    def _build_ticker_industry(self, ticker: TickerMetadata) -> str:
        """
        Retrieve the industry classification of the ticker.

        Parameters:
            metadata (dict): Dictionary containing ticker metadata.

        Returns:
            str: Industry name (e.g., 'Consumer Electronics').
        """
        industry = (
            ticker.industry.strip()
            if ticker.industry is not None and len(ticker.industry.strip()) > 0
            else None
        )
        industry_key = (
            ticker.industry_key.strip()
            if ticker.industry_key is not None and len(ticker.industry_key.strip()) > 0
            else None
        )
        industry_disp = (
            ticker.industry_disp.strip()
            if ticker.industry_disp is not None
            and len(ticker.industry_disp.strip()) > 0
            else None
        )
        return (
            industry_key
            if industry_key is not None
            else (industry_disp if industry_disp is not None else industry)
        )

    def _build_ticker_currency(self, ticker: TickerMetadata) -> str:
        """
        Get the trading currency of the ticker.

        Parameters:
            metadata (dict): Dictionary containing ticker metadata.

        Returns:
            str: ISO currency code (e.g., 'USD').
        """
        currency = (
            ticker.currency.strip()
            if ticker.currency is not None and len(ticker.currency.strip()) > 0
            else None
        )
        financial_currency = (
            ticker.financial_currency.strip()
            if ticker.financial_currency is not None
            and len(ticker.financial_currency.strip()) > 0
            else None
        )
        return currency if currency is not None else financial_currency

    def _build_ticker_exchange(self, ticker: TickerMetadata) -> str:
        """
        Get the name of the exchange where the ticker is listed.

        Parameters:
            metadata (dict): Dictionary containing ticker metadata.

        Returns:
            str: Exchange name (e.g., 'NasdaqGS').
        """
        exchange = (
            ticker.exchange.strip()
            if ticker.exchange is not None and len(ticker.exchange.strip()) > 0
            else None
        )
        return exchange

    def __init__(self, retries: int, sleep_seconds: int):
        self.retries = retries
        self.sleep_seconds = sleep_seconds

    def get_price_data(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> pd.DataFrame:
        """Attempts to download market data with retries."""
        attempt = 0
        config = PriceDataConfig(
            symbols=symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        while attempt < self.retries:
            try:
                with OutputSuppressor.suppress():
                    df = Downloader._PROVIDER.get_price_data(config)
                if not df.empty:
                    df = localize_to_market_time(df, Downloader._MARKET_TZ)
                    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
                    df = df[(df.index >= start) & (df.index <= end)]
                    return df
            except (ValueError, IOError) as error:
                Logger.error(
                    f"  Error downloading {symbol} on attempt {attempt + 1}: {error}"
                )
            attempt += 1
            time.sleep(self.sleep_seconds)
        return pd.DataFrame()

    def get_metadata(self, symbol: str) -> dict:
        """Fetches symbol metadata using provider."""
        local_symbol = symbol.strip().upper()
        attempt = 0
        metadata = {
            "name": None,
            "type": None,
            "sector": None,
            "industry": None,
            "currency": "USD",
            "exchange": None,
        }
        while attempt < self.retries:
            try:
                ticker = Downloader._PROVIDER.get_metadata(symbol)
                metadata = {
                    "name": self._build_ticker_name(ticker),
                    "type": self._build_ticker_type(ticker),
                    "sector": self._build_ticker_sector(ticker),
                    "industry": self._build_ticker_industry(ticker),
                    "currency": self._build_ticker_currency(ticker),
                    "exchange": self._build_ticker_exchange(ticker),
                }
                Logger.debug(f"  Metadata fetched for {symbol}")
                attempt = self.retries
            except (KeyError, ValueError) as error:
                Logger.error(f"  Failed to fetch metadata for {local_symbol}: {error}")
                time.sleep(self.sleep_seconds)
            attempt += 1
        return metadata

    def is_symbol_available(self, symbol: str) -> bool:
        """Check if a given symbol has valid downloadable data."""
        try:
            df = self.get_price_data(
                symbol,
                datetime.now(timezone.utc) - timedelta(days=7),
                datetime.now(timezone.utc),
                "1d",
            )
            return not df.empty
        except (ValueError, IOError, KeyError) as error:
            Logger.error(f"  Failed to update {symbol}: {error}")
            return False
