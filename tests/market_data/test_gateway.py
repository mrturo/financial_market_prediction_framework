"""Unit tests for the market_data.gateway module."""

# pylint: disable=protected-access

from unittest.mock import patch

import pandas as pd
import pytest

from market_data.gateway import Gateway
from utils.parameters import ParameterLoader

_PARAMS = ParameterLoader()
_TEST_MARKETDATA_FILEPATH = _PARAMS.get("test_marketdata_filepath")


@pytest.fixture
def sample_symbol_data(symbol_metadata):
    """Fixture providing sample symbol data for testing."""
    symbol = symbol_metadata.copy()
    symbol["schedule"] = {
        "monday": {"min_open": "09:30:00", "max_close": "16:00:00"},
        "tuesday": {"min_open": "09:30:00", "max_close": "16:00:00"},
        "wednesday": {"min_open": "09:30:00", "max_close": "16:00:00"},
        "thursday": {"min_open": "09:30:00", "max_close": "16:00:00"},
        "friday": {"min_open": "09:30:00", "max_close": "16:00:00"},
    }
    symbol["historical_prices"] = [
        {"datetime": "2025-05-16T20:00:00Z", "close": 150.0},
        {"datetime": "2025-05-17T20:00:00Z", "close": 152.0},
    ]
    return symbol


# pylint: disable=redefined-outer-name
def test_set_and_get_symbol(sample_symbol_data):
    """Test setting and retrieving a single symbol."""
    Gateway.set_symbol(
        symbol=sample_symbol_data["symbol"],
        historical_prices=sample_symbol_data["historical_prices"],
        metadata=sample_symbol_data,
    )
    retrieved = Gateway.get_symbol("AAPL")
    if retrieved["name"] != "Apple Inc.":
        raise AssertionError("Symbol name mismatch.")
    if len(retrieved["historical_prices"]) != 2:
        raise AssertionError("Incorrect number of historical prices.")


# pylint: disable=redefined-outer-name
def test_set_and_get_symbols(sample_symbol_data):
    """Test setting and retrieving all symbols."""
    Gateway.set_symbols({"AAPL": sample_symbol_data})
    symbols = Gateway.get_symbols()
    if "AAPL" not in symbols:
        raise AssertionError("Symbol not found.")
    if symbols["AAPL"]["sector"] != "Technology":
        raise AssertionError("Incorrect sector.")


def test_set_and_get_latest_price_date():
    """Test manually setting and retrieving the latest price date."""
    test_date = pd.Timestamp("2025-05-17T20:00:00Z")
    Gateway.set_latest_price_date(test_date)
    if Gateway.get_latest_price_date() != test_date:
        raise AssertionError("get_latest_price_date mismatch.")
    if Gateway.latest_price_date() != test_date:
        raise AssertionError("latest_price_date mismatch.")


def test_set_and_get_stale_symbols():
    """Test setting and retrieving the list of stale symbols."""
    stale_list = ["AAPL", "GOOG"]
    Gateway.set_stale_symbols(stale_list)
    if Gateway.get_stale_symbols() != stale_list:
        raise AssertionError("Stale symbols mismatch.")


def test_invalid_symbols():
    """Test retrieval of invalid symbols from repository."""
    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = {"FAKE"}
        if Gateway.invalid_symbols() != {"FAKE"}:
            raise AssertionError("Invalid symbols mismatch.")


def test_load_market_data(monkeypatch, symbol_metadata):
    """Test loading market data from disk and initialization of gateway state."""
    symbol = symbol_metadata.copy()
    symbol["schedule"] = {
        "monday": {"min_open": "09:30:00", "max_close": "16:00:00"},
    }
    symbol["historical_prices"] = [
        {"datetime": "2025-05-16T20:00:00Z", "close": 150.0},
        {"datetime": "2025-05-17T20:00:00Z", "close": 152.0},
    ]

    raw_data = {
        "last_updated": "2025-05-18T10:00:00Z",
        "symbols": [symbol],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("market_data.gateway.JsonManager.load", mock_load)

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        result = Gateway.load(_TEST_MARKETDATA_FILEPATH)

    if "AAPL" not in result["symbols"]:
        raise AssertionError("Symbol 'AAPL' not loaded correctly.")
    if not isinstance(result["last_updated"], pd.Timestamp):
        raise AssertionError("last_updated is not a pd.Timestamp.")
    if result["symbols"]["AAPL"]["name"] != "Apple Inc.":
        raise AssertionError("Incorrect symbol metadata after load.")


def test_get_last_update_from_load(monkeypatch):
    """Test retrieving last update timestamp via load."""
    raw_data = {
        "last_updated": "2025-05-18T10:00:00Z",
        "symbols": [],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("market_data.gateway.JsonManager.load", mock_load)

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        Gateway.load(_TEST_MARKETDATA_FILEPATH)
        last_update = Gateway.get_last_update()
        if last_update != pd.Timestamp("2025-05-18T10:00:00Z"):
            raise AssertionError("Incorrect last update timestamp.")


# pylint: disable=redefined-outer-name
def test_get_global_schedule_from_update_via_load(monkeypatch, sample_symbol_data):
    """Test global schedule built indirectly through load process."""

    raw_data = {
        "last_updated": "2025-05-18T10:00:00Z",
        "symbols": [sample_symbol_data],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("market_data.gateway.JsonManager.load", mock_load)

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        Gateway.load(_TEST_MARKETDATA_FILEPATH)
        schedule = Gateway.get_global_schedule()
        if "monday" not in schedule:
            raise AssertionError("Missing 'monday' in schedule.")
        if not schedule["monday"].get("min_open"):
            raise AssertionError("Missing 'min_open' in schedule.")


# pylint: disable=redefined-outer-name
def test_save_market_data_via_interface_indirect(monkeypatch, sample_symbol_data):
    """Test saving market data using only public interfaces (indirect schedule update)."""

    raw_data = {
        "last_updated": "2025-05-19T13:00:00Z",
        "symbols": [sample_symbol_data],
    }

    def mock_load(_filepath):
        return raw_data

    saved_output = {}

    def mock_save(data, _filepath):
        nonlocal saved_output
        saved_output = data

    monkeypatch.setattr("market_data.gateway.JsonManager.load", mock_load)
    monkeypatch.setattr("market_data.gateway.JsonManager.save", mock_save)

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        Gateway.load(_TEST_MARKETDATA_FILEPATH)
        result = Gateway.save(_TEST_MARKETDATA_FILEPATH)

    if "last_updated" not in result:
        raise AssertionError("Missing 'last_updated' in result.")
    if not saved_output.get("symbols"):
        raise AssertionError("Symbols not found in saved output.")
    if "schedule" not in saved_output:
        raise AssertionError("Schedule not saved.")


# pylint: disable=redefined-outer-name
def test_schedule_all_day_flag(monkeypatch, sample_symbol_data):
    """Test global schedule detects all_day market correctly."""

    # Forzar min_open == max_close == "00:00:00"
    sample_symbol_data["schedule"] = {
        "monday": {"min_open": "00:00:00", "max_close": "00:00:00"},
    }

    raw_data = {
        "last_updated": "2025-05-18T10:00:00Z",
        "symbols": [sample_symbol_data],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("market_data.gateway.JsonManager.load", mock_load)

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        Gateway.load(_TEST_MARKETDATA_FILEPATH)
        schedule = Gateway.get_global_schedule()
        if not schedule["monday"].get("all_day"):
            raise AssertionError(
                "Expected 'all_day' flag to be True for 00:00:00 open/close."
            )


# pylint: disable=redefined-outer-name
def test_save_returns_result_structure(monkeypatch, sample_symbol_data):
    """Test that save() returns the expected result structure."""

    raw_data = {
        "last_updated": "2025-05-19T13:00:00Z",
        "symbols": [sample_symbol_data],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("market_data.gateway.JsonManager.load", mock_load)

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        Gateway.load(_TEST_MARKETDATA_FILEPATH)
        result = Gateway.save(_TEST_MARKETDATA_FILEPATH)

    if not isinstance(result, dict):
        raise AssertionError("save() did not return a dictionary.")
    if "symbols" not in result or "schedule" not in result:
        raise AssertionError("Returned result missing expected keys.")


# pylint: disable=redefined-outer-name
def test_schedule_all_day_flag_missing_values(monkeypatch, sample_symbol_data):
    """Test all_day logic path when min_open or max_close is missing."""

    # Faltando 'max_close'
    sample_symbol_data["schedule"] = {"monday": {"min_open": "00:00:00"}}

    raw_data = {
        "last_updated": "2025-05-18T10:00:00Z",
        "symbols": [sample_symbol_data],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("market_data.gateway.JsonManager.load", mock_load)

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        Gateway.load(_TEST_MARKETDATA_FILEPATH)
        schedule = Gateway.get_global_schedule()
        if schedule["monday"].get("all_day") is not None:
            raise AssertionError(
                "Expected 'all_day' to be None when values are missing."
            )


# pylint: disable=redefined-outer-name
def test_schedule_all_day_flag_false(monkeypatch, sample_symbol_data):
    """Test that all_day is False when min_open and max_close are not both 00:00:00."""

    sample_symbol_data["schedule"] = {
        "monday": {"min_open": "09:00:00", "max_close": "17:00:00"},
    }

    raw_data = {
        "last_updated": "2025-05-18T10:00:00Z",
        "symbols": [sample_symbol_data],
    }

    def mock_load(_filepath):
        return raw_data

    monkeypatch.setattr("market_data.gateway.JsonManager.load", mock_load)

    with patch.object(Gateway, "_SYMBOL_REPO", autospec=True) as mock_repo:
        mock_repo.get_invalid_symbols.return_value = set()
        Gateway.load(_TEST_MARKETDATA_FILEPATH)
        schedule = Gateway.get_global_schedule()
        if schedule["monday"].get("all_day") is True:
            raise AssertionError("Expected 'all_day' to be False.")


def test_set_and_get_stale_symbols_forced():
    """Ensure set_stale_symbols updates internal state using public interface only."""
    # Reset using public interface
    Gateway.set_stale_symbols([])

    test_symbols = ["MSFT", "TSLA"]
    Gateway.set_stale_symbols(test_symbols)

    result = Gateway.get_stale_symbols()
    if result != test_symbols:
        raise AssertionError("Stale symbols not set correctly.")


def test_detect_stale_symbols_logic():
    """Test internal logic for stale symbol detection."""
    symbol = "XXXX"
    outdated = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=500)).isoformat()

    Gateway.set_symbols(
        {
            symbol: {
                "symbol": symbol,
                "historical_prices": [{"datetime": outdated, "close": 1.0}],
            }
        }
    )

    latest = pd.Timestamp.now(tz="UTC")
    current_invalids = set()

    stale, updated_invalids = Gateway._detect_stale_symbols(latest, current_invalids)

    if symbol not in stale:
        raise AssertionError(f"{symbol} should be marked as stale.")
    if symbol not in updated_invalids:
        raise AssertionError(f"{symbol} should be added to invalid symbols.")


def test_get_filepath_returns_valid_path():
    """Test that _get_filepath returns the original filepath when it is valid."""
    result = Gateway._get_filepath("path/to/file.json", "default.json")
    if result != "path/to/file.json":
        raise AssertionError("Expected filepath to be returned when valid")


def test_get_filepath_returns_default_when_none():
    """Test that _get_filepath returns the default path when filepath is None."""
    result = Gateway._get_filepath(None, "default.json")
    if result != "default.json":
        raise AssertionError("Expected default to be returned when filepath is None")


def test_get_filepath_returns_default_when_empty():
    """Test that _get_filepath returns the default path when filepath is an empty string."""
    result = Gateway._get_filepath("   ", "default.json")
    if result != "default.json":
        raise AssertionError(
            "Expected default to be returned when filepath is empty string"
        )
