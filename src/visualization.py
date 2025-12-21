"""Visualization module for plotting and visual analysis."""
# src/visualization.py
"""
Visualization of training data, ideal functions, and mapped test data using Bokeh.
"""

from typing import Dict

import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import Legend


def plot_training_and_ideals(
    training_df: pd.DataFrame,
    ideal_df: pd.DataFrame,
    selected_ideals: Dict[str, int],
    output_html: str = "training_ideal_plot.html",
) -> None:
    """
    Plot the four training functions together with their matched ideal functions.

    Args:
        training_df: DataFrame with columns x_value, y1_train, ...
        ideal_df: DataFrame with columns x_value, y_ideal_1, ...
        selected_ideals: mapping of training column -> ideal function index.
        output_html: name of the output HTML created by Bokeh.
    """
    output_file(output_html, title="Training vs Ideal Functions")

    plot = figure(title="Training vs Ideal Functions", x_axis_label="x", y_axis_label="y")

    legend_items = []

    # Plot training functions
    x_values = training_df["x_value"]

    for train_col, color in zip(sorted(selected_ideals.keys()), ["red", "green", "blue", "orange"]):
        line_renderer = plot.circle(
            x=x_values,
            y=training_df[train_col],
            size=4,
            color=color,
            alpha=0.6,
        )
        legend_items.append((f"{train_col}", [line_renderer]))

        # Plot corresponding ideal function
        ideal_index = selected_ideals[train_col]
        ideal_col = f"y_ideal_{ideal_index}"
        ideal_line = plot.line(
            x=ideal_df["x_value"],
            y=ideal_df[ideal_col],
            line_width=2,
            color=color,
            alpha=0.9,
        )
        legend_items.append((f"Ideal {ideal_index}", [ideal_line]))

    legend = Legend(items=legend_items)
    plot.add_layout(legend, "right")

    save(plot)


def plot_test_mapping(
    test_df: pd.DataFrame,
    mapped_df: pd.DataFrame,
    output_html: str = "test_mapping_plot.html",
) -> None:
    """
    Plot all test points and highlight mapped ones colored by ideal function.
    """
    output_file(output_html, title="Test Data Mapping")

    plot = figure(title="Test Data Mapping", x_axis_label="x", y_axis_label="y")

    # Plot all test points in grey
    plot.circle(
        x=test_df["x_value"],
        y=test_df["y_test"],
        size=4,
        color="grey",
        alpha=0.4,
        legend_label="Unmapped test points",
    )

    # For mapped points, color by ideal function number
    colors = ["red", "green", "blue", "orange", "purple", "brown"]
    for ideal_index in sorted(mapped_df["ideal_function_number"].unique()):
        sub = mapped_df[mapped_df["ideal_function_number"] == ideal_index]
        color = colors[(ideal_index - 1) % len(colors)]

        plot.circle(
            x=sub["x_value"],
            y=sub["y_test"],
            size=6,
            color=color,
            alpha=0.8,
            legend_label=f"Ideal {ideal_index}",
        )

    save(plot)