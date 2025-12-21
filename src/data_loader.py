"""Data loader module for loading and preprocessing data."""
# src/data_loader.py
"""
Load CSV files into Pandas DataFrames and then into the database.

Responsibilities:
- Read training CSVs and combine into a single table.
- Read ideal function CSVs.
- Read test data line by line for mapping.
"""

from pathlib import Path
from typing import List

import pandas as pd

from .database import DatabaseManager, DatabaseError


class DataLoaderError(Exception):
    """Custom exception for data loading errors."""
    pass


class DataLoader:
    """
    Handles all CSV reading and transformation to the structures
    required for the database.
    """

    def __init__(self, db_manager: DatabaseManager, data_dir: Path) -> None:
        self.db_manager = db_manager
        self.data_dir = data_dir

    def load_training_data(self, train_filenames: List[str]) -> None:
        """
        Load four training CSV files and combine them into a single
        DataFrame with columns: x, y1, y2, y3, y4.
        """
        try:
            # Read each training CSV into a DataFrame
            training_frames = []
            for filename in train_filenames:
                csv_path = self.data_dir / filename
                df = pd.read_csv(csv_path)
                training_frames.append(df)

            # Validate that all x columns are identical
            x_reference = training_frames[0]["x"]
            for frame in training_frames[1:]:
                if not x_reference.equals(frame["x"]):
                    raise DataLoaderError("Training CSV files do not share identical x values.")

            # Build combined frame
            combined_frame = pd.DataFrame()
            combined_frame["x_value"] = x_reference
            for index, frame in enumerate(training_frames, start=1):
                combined_frame[f"y{index}_train"] = frame["y"].values

            # Persist to database
            self.db_manager.write_dataframe(combined_frame, table_name="training_data")

        except FileNotFoundError as exc:
            raise DataLoaderError(f"Training file not found: {exc}") from exc

    def load_ideal_functions(self, ideal_filename: str) -> None:
        """
        Load ideal functions CSV and write to database.
        Expected CSV columns: x, y1, ..., y50.
        """
        try:
            csv_path = self.data_dir / ideal_filename
            df = pd.read_csv(csv_path)

            # Rename x column to x_value to match ORM convention
            df = df.rename(columns={"x": "x_value"})

            self.db_manager.write_dataframe(df, table_name="ideal_functions")

        except FileNotFoundError as exc:
            raise DataLoaderError(f"Ideal function file not found: {exc}") from exc

    def load_test_data(self, test_filename: str) -> pd.DataFrame:
        """
        Load test data CSV as DataFrame. We keep it in memory because
        mapping (deviation check) is easier that way.
        """
        try:
            csv_path = self.data_dir / test_filename
            df = pd.read_csv(csv_path)
            df = df.rename(columns={"x": "x_value", "y": "y_test"})
            return df
        except FileNotFoundError as exc:
            raise DataLoaderError(f"Test data file not found: {exc}") from exc