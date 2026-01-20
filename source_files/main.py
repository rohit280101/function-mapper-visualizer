import pandas as pd
import os
from database import Database_Handler
from data_processor import IdealFunctionSelector, TestDataMapper
from visualization import Visualizer

def main():
    # Get the project root directory (parent of source_files)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    training_df = pd.read_csv(os.path.join(project_root, "datasets/train.csv"))
    ideal_df = pd.read_csv(os.path.join(project_root, "datasets/ideal.csv"))
    test_df = pd.read_csv(os.path.join(project_root, "datasets/test.csv"))

    training_df.rename(columns={training_df.columns[0]: "x"}, inplace=True)
    ideal_df.rename(columns={ideal_df.columns[0]: "x"}, inplace=True)
    test_df.rename(columns={test_df.columns[0]: "x", test_df.columns[1]: "y"}, inplace=True)

    selector = IdealFunctionSelector(training_df, ideal_df)
    selected = selector.select_best_ideal_functions()
    max_dev = selector.calculate_max_deviation()

    mapper = TestDataMapper(test_df, ideal_df, selected, max_dev)
    mapped_df = mapper.map_test_data()

    db = Database_Handler()
    db.insert_training_data(training_df)
    db.insert_test_mapping(mapped_df)

    visualizer = Visualizer(training_df, ideal_df, selected, mapped_df)
    output_path = os.path.join(project_root, "outputs/visualization.html")
    visualizer.plot_all(output_path)

if __name__ == "__main__":
    main()
