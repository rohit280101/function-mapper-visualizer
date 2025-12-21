"""Matching module for matching algorithms."""
# src/matching.py
"""
Core matching logic:

1. Select four best ideal functions for each training function using least squares.
2. For test data, decide whether each point can be assigned to one of the four
   ideal functions using the maximum deviation * sqrt(2) rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .database import DatabaseManager
from math import sqrt


@dataclass
class SelectedIdealFunctions:
    """
    Data container for the indices and deviation limits of the four selected ideal functions.
    """
    mapping: Dict[str, int]           # training_y_column -> ideal_function_index
    max_deviation: Dict[str, float]   # training_y_column -> max |y_train - y_ideal|


class IdealFunctionSelector:
    """
    Selects the best ideal functions for each training function based on least squares.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def select_best_ideals(self) -> SelectedIdealFunctions:
        """
        Determine which ideal function best matches each training function.

        Returns:
            SelectedIdealFunctions object holding mapping and max devs.
        """
        # Load tables as DataFrames
        training_df = self.db_manager.read_table("training_data")
        ideal_df = self.db_manager.read_table("ideal_functions")

        # Align by x_value to be safe, even though datasets should already match
        merged = pd.merge(training_df, ideal_df, on="x_value", suffixes=("_train", "_ideal"))

        # Extract all ideal columns (assumed to start with "y" and contain "_ideal")
        ideal_columns = [col for col in merged.columns if col.startswith("y") and "_ideal" in col]

        # Extract training y columns
        training_columns = [col for col in training_df.columns if col.startswith("y")]

        mapping: Dict[str, int] = {}
        max_deviation: Dict[str, float] = {}

        # For each training column, compute least-squares fit to each ideal column
        for train_col in training_columns:
            best_ideal_col = None
            best_sse = np.inf
            best_max_dev = np.inf

            train_values = merged[train_col].values

            for ideal_col in ideal_columns:
                ideal_values = merged[ideal_col].values

                # Sum of squared errors
                diff = train_values - ideal_values
                sse = np.sum(diff ** 2)

                if sse < best_sse:
                    best_sse = sse
                    best_ideal_col = ideal_col
                    best_max_dev = float(np.max(np.abs(diff)))

            if best_ideal_col is None:
                raise RuntimeError(f"No ideal function found for training column {train_col}.")

            # Ideal function number from column name, e.g., y_ideal_17 -> 17
            ideal_index = self._extract_ideal_index(best_ideal_col)

            mapping[train_col] = ideal_index
            max_deviation[train_col] = best_max_dev

        return SelectedIdealFunctions(mapping=mapping, max_deviation=max_deviation)

    @staticmethod
    def _extract_ideal_index(column_name: str) -> int:
        """
        Extract numeric suffix from an ideal column name, e.g. 'y_ideal_17' -> 17.
        Adjust according to your column naming scheme.
        """
        # Example: 'y_ideal_17'
        parts = column_name.split("_")
        return int(parts[-1])


class TestDataMapper:
    """
    Uses the selected ideal functions to map test points into the database.
    """

    def __init__(self, db_manager: DatabaseManager, selected_ideals: SelectedIdealFunctions) -> None:
        self.db_manager = db_manager
        self.selected_ideals = selected_ideals

    def map_test_points(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each test point, decide if it can be associated to any of the
        four selected ideal functions according to the assignment rule.

        Rule:
            |y_test - y_ideal(x)| <= max_training_deviation * sqrt(2)

        If multiple functions match, we usually assign the one with the smallest deviation.

        Returns:
            DataFrame with columns:
                x_value, y_test, delta_y, ideal_function_number
        """
        ideal_df = self.db_manager.read_table("ideal_functions")

        # Build a dictionary: ideal_index -> ideal y-values (Series)
        ideal_series_by_index: Dict[int, pd.Series] = {}
        for train_col, ideal_index in self.selected_ideals.mapping.items():
            ideal_column_name = f"y_ideal_{ideal_index}"
            ideal_series_by_index[ideal_index] = ideal_df.set_index("x_value")[ideal_column_name]

        mapped_rows = []

        # Iterate over test points
        for _, row in test_df.iterrows():
            x_value = float(row["x_value"])
            y_test = float(row["y_test"])

            best_ideal_index = None
            best_delta = np.inf

            # Try each of the four chosen ideal functions
            for train_col, ideal_index in self.selected_ideals.mapping.items():
                # Allowed maximum deviation for this training-ideal pair
                max_dev_train = self.selected_ideals.max_deviation[train_col]
                allowed_deviation = max_dev_train * sqrt(2.0)

                ideal_y_series = ideal_series_by_index[ideal_index]

                # Some x-values from test may not exist in ideal table; skip if missing
                try:
                    y_ideal = float(ideal_y_series.loc[x_value])
                except KeyError:
                    continue

                this_delta = abs(y_test - y_ideal)

                # Check criterion
                if this_delta <= allowed_deviation and this_delta < best_delta:
                    best_delta = this_delta
                    best_ideal_index = ideal_index

            if best_ideal_index is not None:
                mapped_rows.append(
                    {
                        "x_value": x_value,
                        "y_test": y_test,
                        "delta_y": best_delta,
                        "ideal_function_number": int(best_ideal_index),
                    }
                )

        mapped_df = pd.DataFrame(mapped_rows)

        # Persist to DB (append to existing table, if any)
        self.db_manager.write_dataframe(mapped_df, table_name="test_mapping", if_exists="replace")

        return mapped_df