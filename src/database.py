"""Database module for handling data persistence."""
# src/database.py
"""
Database access layer using SQLAlchemy.

Responsible for:
- Creating the SQLite file.
- Creating all tables.
- Providing helper methods to write/read Pandas DataFrames.
"""

from __future__ import annotations
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd

from .models import Base


class DatabaseError(Exception):
    """Custom exception type for database-related errors."""
    pass


class DatabaseManager:
    """
    Encapsulates all low-level database operations.

    Attributes:
        db_url: SQLAlchemy URL for the SQLite database file.
        engine: SQLAlchemy engine instance.
        SessionLocal: Session factory.
    """

    def __init__(self, db_url: str = "sqlite:///assignment.db") -> None:
        """
        Initialize the manager with the given database URL.
        """
        self.db_url = db_url
        self.engine = create_engine(self.db_url, echo=False, future=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    def create_schema(self) -> None:
        """
        Create all database tables as defined in the ORM models.
        """
        try:
            Base.metadata.create_all(self.engine)
        except Exception as error:  # noqa: BLE001
            raise DatabaseError(f"Failed to create database schema: {error}") from error

    def get_session(self) -> Session:
        """
        Create and return a new SQLAlchemy Session instance.
        """
        return self.SessionLocal()

    def write_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> None:
        """
        Persist a Pandas DataFrame into the SQLite database using DataFrame.to_sql.

        Args:
            df: DataFrame to persist.
            table_name: Database table name.
            if_exists: Behavior if table already exists (replace/append/fail).
        """
        try:
            df.to_sql(table_name, self.engine, index=False, if_exists=if_exists)
        except Exception as error:  # noqa: BLE001
            raise DatabaseError(f"Failed to write DataFrame to table '{table_name}': {error}") from error

    def read_table(self, table_name: str) -> pd.DataFrame:
        """
        Read the full content of the given table into a DataFrame.
        """
        try:
            return pd.read_sql_table(table_name, self.engine)
        except Exception as error:  # noqa: BLE001
            raise DatabaseError(f"Failed to read table '{table_name}': {error}") from error