
import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processor import IdealFunctionSelector, TestDataMapper

def generate_mock_data():
    x = np.arange(0, 10, 1)
    training = pd.DataFrame({
        "x": x,
        "y1": x * 2,
        "y2": x * 3,
        "y3": x * 4,
        "y4": x * 5
    })

    ideal = pd.DataFrame({"x": x})
    for i in range(1, 51):
        ideal[f"y{i}"] = x * i

    test = pd.DataFrame({"x": [1, 2, 3], "y": [2, 6, 8]})
    return training, ideal, test

def test_function_selection():
    training, ideal, _ = generate_mock_data()
    selector = IdealFunctionSelector(training, ideal)
    result = selector.select_best_ideal_functions()
    assert len(result) == 4
    assert all(1 <= idx <= 50 for idx in result.values())

def test_test_mapping():
    training, ideal, test = generate_mock_data()
    selector = IdealFunctionSelector(training, ideal)
    selected = selector.select_best_ideal_functions()
    max_dev = selector.calculate_max_deviation()
    mapper = TestDataMapper(test, ideal, selected, max_dev)
    mapped = mapper.map_test_data()
    assert not mapped.empty
    assert all(col in mapped.columns for col in ["x", "y", "delta_y", "ideal_func_no"])
