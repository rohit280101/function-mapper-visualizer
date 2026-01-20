
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column

class Visualizer:
    def __init__(self, training_df, ideal_df, selected_functions, test_mapping_df):
        self.training_df = training_df
        self.ideal_df = ideal_df
        self.selected_functions = selected_functions
        self.test_mapping_df = test_mapping_df

    def plot_all(self, output_html="outputs/visualization.html"):
        plots = []
        for i in range(1, 5):
            y_label = f"y{i}"
            ideal_idx = self.selected_functions[y_label]
            p = figure(title=f"Training {y_label} vs Ideal y{ideal_idx}", x_axis_label='x', y_axis_label='y')
            p.circle(self.training_df["x"], self.training_df[y_label], size=5, color="blue", alpha=0.5, legend_label="Training")
            p.line(self.ideal_df["x"], self.ideal_df.iloc[:, ideal_idx], line_width=2, color="green", legend_label=f"Ideal y{ideal_idx}")
            relevant_test = self.test_mapping_df[self.test_mapping_df["ideal_func_no"] == ideal_idx]
            p.triangle(relevant_test["x"], relevant_test["y"], size=8, color="red", alpha=0.7, legend_label="Mapped Test")
            p.legend.location = "top_left"
            plots.append(p)
        output_file(output_html)
        save(column(*plots))
