"""Models module for ML models."""
# src/models.py
"""
SQLAlchemy ORM models for the assignment.

- BaseDataTable: common base class for tables that store x/y-style data.
- TrainingData, IdealFunction, TestMapping: specific table classes.
"""

from typing import Optional
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import Integer, Float, String

Base = declarative_base()


class BaseDataTable(Base):
    """
    Abstract base class for database tables that store x values.
    This class demonstrates inheritance: specific tables inherit from it.
    """
    __abstract__ = True

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    x_value: Mapped[float] = mapped_column(Float, nullable=False)


class TrainingData(BaseDataTable):
    """
    Table for the four training functions.
    Columns:
        x_value, y1_train, y2_train, y3_train, y4_train
    """
    __tablename__ = "training_data"

    y1_train: Mapped[float] = mapped_column(Float, nullable=False)
    y2_train: Mapped[float] = mapped_column(Float, nullable=False)
    y3_train: Mapped[float] = mapped_column(Float, nullable=False)
    y4_train: Mapped[float] = mapped_column(Float, nullable=False)


class IdealFunction(BaseDataTable):
    """
    Table for the 50 ideal functions.
    Columns:
        x_value, y_ideal_1, ..., y_ideal_50
    """
    __tablename__ = "ideal_functions"

    # In practice you’d add 50 columns. Here is a pattern; adapt for all 50.
    # Using a few as illustration – extend in your own code.
    y_ideal_1: Mapped[float] = mapped_column(Float, nullable=False)
    y_ideal_2: Mapped[float] = mapped_column(Float, nullable=False)
    # ...
    # y_ideal_50: Mapped[float] = mapped_column(Float, nullable=False)


class TestMapping(BaseDataTable):
    """
    Table for mapped test points.
    Columns:
        x_value, y_test, delta_y, ideal_function_number
    """
    __tablename__ = "test_mapping"

    y_test: Mapped[float] = mapped_column(Float, nullable=False)
    delta_y: Mapped[float] = mapped_column(Float, nullable=False)
    ideal_function_number: Mapped[int] = mapped_column(Integer, nullable=False)