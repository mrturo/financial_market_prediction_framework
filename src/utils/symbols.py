"""
Repository and manager for symbol lists used in training and analysis pipelines.

Includes utilities for loading symbol lists from JSON files and managing invalid symbols.
"""

from dataclasses import dataclass
from typing import List, Set

from utils.json_manager import JsonManager


@dataclass
class Symbols:
    """
    Load and store a unique sorted list of financial symbols from a JSON file.

    Attributes:
        file_path (str): Path to the JSON file containing the symbol list.
        list (List[str]): Sorted list of unique symbols loaded from the file.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        if not JsonManager.exists(file_path):
            JsonManager.save([], file_path)
            self.list = []
        else:
            self.list = sorted(set(JsonManager.load(file_path)))


@dataclass
class SymbolPaths:
    """
    Data container for storing file paths to different symbol categories.

    Attributes:
        mercado_pago (str): Path to Mercado Pago symbols file.
        training (str): Path to training symbols file.
        xtb (str): Path to XTB symbols file.
        correlative (str): Path to correlative symbols file.
        invalid (str): Path to invalid symbols file.
    """

    mercado_pago: str
    training: str
    xtb: str
    correlative: str
    invalid: str


class SymbolRepository:
    """
    Repository and interface for accessing categorized symbol lists.

    Manages access to multiple categories of financial symbols while
    filtering out any symbols marked as invalid.

    Attributes:
        correlative (Symbols): Correlative symbols list.
        invalid (Symbols): Invalid symbols list.
        mercado_pago (Symbols): Mercado Pago symbols list.
        training (Symbols): Training symbols list.
        xtb (Symbols): XTB symbols list.
    """

    def __init__(self, paths: SymbolPaths):
        self.correlative = Symbols(paths.correlative)
        self.invalid = Symbols(paths.invalid)
        self.mercado_pago = Symbols(paths.mercado_pago)
        self.training = Symbols(paths.training)
        self.xtb = Symbols(paths.xtb)

    def _remove_invalid(self, symbols: list) -> List[str]:
        """Remove invalid symbols of symbol list."""
        return sorted(list(set(symbols) - set(self.invalid.list)))

    def get_all_symbols(self) -> List[str]:
        """Get all symbols."""
        all_symbols = list(
            set(
                self.training.list
                + self.mercado_pago.list
                + self.xtb.list
                + self.correlative.list
            )
        )
        return self._remove_invalid(all_symbols)

    def get_correlative_symbols(self) -> List[str]:
        """Get correlative symbols."""
        return self._remove_invalid(self.correlative.list)

    def get_invalid_symbols(self) -> List[str]:
        """Get list of invalid symbols."""
        return self.invalid.list

    def set_invalid_symbols(self, invalid_symbols: Set[str]) -> None:
        """Set invalid symbols."""
        self.invalid.list = sorted(invalid_symbols)
        JsonManager.save(self.invalid.list, self.invalid.file_path)
        return self.invalid.list

    def get_mercado_pago_symbols(self) -> List[str]:
        """Get Mercado Pago symbols."""
        return self._remove_invalid(self.mercado_pago.list)

    def get_training_symbols(self) -> List[str]:
        """Get training symbols."""
        return self._remove_invalid(self.training.list)

    def get_xtb_symbols(self) -> List[str]:
        """Get XTB symbols."""
        return self._remove_invalid(self.xtb.list)
