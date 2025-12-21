"""Main module - entry point for the application."""
# src/main.py
"""
Entry point of the program.

Workflow:
1. Create DB schema.
2. Load training, ideal, and test data.
3. Select best ideal functions for each training function.
4. Map test points to the four selected ideals.
5. Visualize everything using Bokeh.
"""

from pathlib import Path

from .database import DatabaseManager
from .data_loader import DataLoader
from .matching import IdealFunctionSelector, TestDataMapper
from .visualization import plot_training_and_ideals, plot_test_mapping


def main() -> None:
    """
    Main control function for the assignment workflow.
    """
    data_dir = Path("data")

    # 1. Setup database
    db_manager = DatabaseManager("sqlite:///assignment.db")
    db_manager.create_schema()

    loader = DataLoader(db_manager=db_manager, data_dir=data_dir)

    # 2. Load training and ideal data
    loader.load_training_data(["train_1.csv", "train_2.csv", "train_3.csv", "train_4.csv"])
    loader.load_ideal_functions("ideal.csv")

    # 3. Load test data (kept in memory)
    test_df = loader.load_test_data("test.csv")

    # 4. Compute best ideal matches for each training function
    selector = IdealFunctionSelector(db_manager)
    selected_ideals = selector.select_best_ideals()

    # 5. Map test points
    mapper = TestDataMapper(db_manager, selected_ideals)
    mapped_df = mapper.map_test_points(test_df)

    # 6. Visualize results
    training_df = db_manager.read_table("training_data")
    ideal_df = db_manager.read_table("ideal_functions")

    plot_training_and_ideals(training_df, ideal_df, selected_ideals.mapping)
    plot_test_mapping(test_df, mapped_df)


if __name__ == "__main__":
    main()