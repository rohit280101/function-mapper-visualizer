"""Tests for database module."""
# tests/test_database.py
"""
Simple tests for DatabaseManager.
"""

from pathlib import Path
import pandas as pd

from src.database import DatabaseManager


def test_write_and_read_dataframe(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"
    db_manager = DatabaseManager(db_url)

    df_in = pd.DataFrame({"x_value": [0.0, 1.0], "y1_train": [2.0, 3.0]})
    db_manager.write_dataframe(df_in, "training_data")
    df_out = db_manager.read_table("training_data")

    assert len(df_out) == 2
    assert list(df_out["x_value"]) == [0.0, 1.0]