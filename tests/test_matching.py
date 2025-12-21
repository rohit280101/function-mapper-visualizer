"""Tests for matching module."""
# tests/test_matching.py
"""
Tests for IdealFunctionSelector logic on a small synthetic dataset.
"""

import pandas as pd
from src.database import DatabaseManager
from src.matching import IdealFunctionSelector


def test_select_best_ideals(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    db_manager = DatabaseManager(db_url)

    # Simple synthetic data: training y1_train matches ideal y_ideal_1 exactly
    training_df = pd.DataFrame(
        {
            "x_value": [0.0, 1.0, 2.0],
            "y1_train": [1.0, 2.0, 3.0],
            "y2_train": [0.0, 0.0, 0.0],
            "y3_train": [0.0, 0.0, 0.0],
            "y4_train": [0.0, 0.0, 0.0],
        }
    )
    ideal_df = pd.DataFrame(
        {
            "x_value": [0.0, 1.0, 2.0],
            "y_ideal_1": [1.0, 2.0, 3.0],
            "y_ideal_2": [10.0, 10.0, 10.0],
        }
    )

    db_manager.write_dataframe(training_df, "training_data")
    db_manager.write_dataframe(ideal_df, "ideal_functions")

    selector = IdealFunctionSelector(db_manager)
    selected = selector.select_best_ideals()

    # y1_train should map to ideal 1
    assert selected.mapping["y1_train"] == 1