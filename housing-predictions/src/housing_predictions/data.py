import pandas as pd
import os
from config import DATA_DIR


class HousingData:
    def __init__(self):
        train_path = os.path.join(DATA_DIR, "input", "train.csv")
        if not os.path.exists(train_path):
            raise Exception(f"Train data not found at: {train_path}")
        self.train_data = pd.read_csv(train_path)

        test_path = os.path.join(DATA_DIR, "input", "test.csv")
        if not os.path.exists(test_path):
            raise Exception(f"Test data not found at: {test_path}")
        self.test_data = pd.read_csv(test_path)


class TrainingData:
    def __init__(self, training_data, label_column_name, feature_pipeline=None):
        self.training_data = training_data
        self.label_column_name = label_column_name
        self.feature_pipeline = feature_pipeline

    @property
    def target(self):
        return self.training_data[self.label_column_name]

    @property
    def features(self):
        return self.training_data.drop(self.label_column_name, axis=1)

if __name__ == '__main__':
    data = HousingData()
    print(data.train_data.head())
    print(data.train_data.dtypes)