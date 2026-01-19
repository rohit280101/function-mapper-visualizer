import pandas as pd
from database import DatabaseHandler
from data_processor import IdealFunctionSelector, TestDataMapper
from visualization import Visualizer

def main():
    training_df = pd.read_csv("data/train.csv")
    ideal_df = pd.read_csv("data/ideal.csv")
    test_df = pd.read_csv("data/test.csv")

    training_df.rename(columns={training_df.columns[0]: "x"}, inplace=True)
    ideal_df.rename(columns={ideal_df.columns[0]: "x"}, inplace=True)
    test_df.rename(columns={test_df.columns[0]: "x", test_df.columns[1]: "y"}, inplace=True)

    selector = IdealFunctionSelector(training_df, ideal_df)
    selected = selector.select_best_ideal_functions()
    max_dev = selector.calculate_max_deviation()

    mapper = TestDataMapper(test_df, ideal_df, selected, max_dev)
    mapped_df = mapper.map_test_data()

    db = DatabaseHandler()
    db.insert_training_data(training_df)
    db.insert_test_mapping(mapped_df)

    visualizer = Visualizer(training_df, ideal_df, selected, mapped_df)
    visualizer.plot_all("outputs/visualization.html")

if __name__ == "__main__":
    main()
