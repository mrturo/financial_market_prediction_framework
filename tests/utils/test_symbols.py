"""Unit tests for the SymbolRepository class and its JSON loading logic."""

import json

import pytest

from utils.symbols import SymbolPaths, SymbolRepository


@pytest.fixture
def temp_symbol_paths(tmp_path):
    """Fixture that creates temporary JSON for different symbol categories and returns paths."""

    def create_file(name, contents):
        path = tmp_path / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(contents, f)
        return str(path)

    return SymbolPaths(
        mercado_pago=create_file("mercado_pago.json", ["AAPL", "TSLA"]),
        training=create_file("training.json", ["AAPL", "GOOG"]),
        xtb=create_file("xtb.json", ["TSLA", "AMZN"]),
        correlative=create_file("correlative.json", ["GOOG", "MSFT"]),
        invalid=create_file("invalid.json", ["AMZN"]),
    )


# pylint: disable=redefined-outer-name
def test_load_invalid_symbols(temp_symbol_paths):
    """Test that the method correctly loads symbols from the 'invalid' file."""
    repo = SymbolRepository(temp_symbol_paths)
    assert repo.get_invalid_symbols() == ["AMZN"]  # nosec B101


# pylint: disable=redefined-outer-name
def test_load_filtered_symbols_excludes_invalids(temp_symbol_paths):
    """Test that 'training' and 'xtb' symbol loaders correctly exclude invalid symbols."""
    repo = SymbolRepository(temp_symbol_paths)
    assert repo.get_training_symbols() == ["AAPL", "GOOG"]  # nosec B101
    assert repo.get_xtb_symbols() == ["TSLA"]  # nosec B101


# pylint: disable=redefined-outer-name
def test_load_all_symbols(temp_symbol_paths):
    """Test that all unique symbols except invalid ones are loaded and returned sorted."""
    repo = SymbolRepository(temp_symbol_paths)
    expected = sorted(set(["AAPL", "GOOG", "TSLA", "MSFT"]))
    assert repo.get_all_symbols() == expected  # nosec B101


def test_save_invalid_symbols(tmp_path):
    """Test that a given set of invalid symbols is saved to the corresponding file."""
    invalid_path = tmp_path / "invalid.json"
    paths = SymbolPaths(
        mercado_pago="dummy",
        training="dummy",
        xtb="dummy",
        correlative="dummy",
        invalid=str(invalid_path),
    )
    repo = SymbolRepository(paths)
    repo.set_invalid_symbols({"NFLX", "AMZN"})

    with open(invalid_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == ["AMZN", "NFLX"]  # nosec B101
