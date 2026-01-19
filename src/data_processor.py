
import pandas as pd
import numpy as np

class IdealFunctionSelector:
    def __init__(self, training_df, ideal_df):
        self.training_df = training_df
        self.ideal_df = ideal_df
        self.selected_functions = {}

    def calculate_least_square(self, y_train, y_ideal):
        return ((y_train - y_ideal) ** 2).sum()

    def select_best_ideal_functions(self):
        selected = []
        for i in range(1, 5):  # y1 to y4 in training
            y_train = self.training_df.iloc[:, i]
            min_error = float("inf")
            best_func_index = -1
            for j in range(1, 51):  # y1 to y50 in ideal
                y_ideal = self.ideal_df.iloc[:, j]
                error = self.calculate_least_square(y_train, y_ideal)
                if error < min_error:
                    min_error = error
                    best_func_index = j
            self.selected_functions[f"y{i}"] = best_func_index
        return self.selected_functions

    def calculate_max_deviation(self):
        max_deviation = {}
        for key, idx in self.selected_functions.items():
            y_train = self.training_df[key]
            y_ideal = self.ideal_df.iloc[:, idx]
            max_deviation[key] = np.max(np.abs(y_train - y_ideal))
        return max_deviation

class TestDataMapper:
    def __init__(self, test_df, ideal_df, selected_functions, max_deviation):
        self.test_df = test_df
        self.ideal_df = ideal_df
        self.selected_functions = selected_functions
        self.max_deviation = max_deviation

    def map_test_data(self):
        mapped_results = []
        for index, row in self.test_df.iterrows():
            x, y = row["x"], row["y"]
            for key, idx in self.selected_functions.items():
                ideal_func_values = self.ideal_df.iloc[:, idx]
                ideal_x_values = self.ideal_df.iloc[:, 0]
                y_ideal = ideal_func_values[ideal_x_values == x].values
                if len(y_ideal) > 0:
                    deviation = abs(y - y_ideal[0])
                    threshold = self.max_deviation[key] * np.sqrt(2)
                    if deviation <= threshold:
                        mapped_results.append({
                            "x": x,
                            "y": y,
                            "delta_y": deviation,
                            "ideal_func_no": idx
                        })
                        break
        return pd.DataFrame(mapped_results)
