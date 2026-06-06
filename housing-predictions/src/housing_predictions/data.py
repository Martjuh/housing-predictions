import pandas as pd
import os
from config import DATA_DIR
import math


class HousingData:
    def __init__(self, split_train_validation = 0.9):
        self.split_train_validation = split_train_validation
        self._load_train_validation()
        self._load_test()

    def _load_train_validation(self):
        train_path = os.path.join(DATA_DIR, "input", "train.csv")
        if not os.path.exists(train_path):
            raise Exception(f"Train data not found at: {train_path}")

        csv_data = pd.read_csv(train_path)
        split_at_row = self._get_split_row(csv_data)
        self.train_data = csv_data.iloc[:split_at_row]
        self.validation_data = csv_data.iloc[split_at_row:]

    def _load_test(self):
        test_path = os.path.join(DATA_DIR, "input", "test.csv")
        if not os.path.exists(test_path):
            raise Exception(f"Test data not found at: {test_path}")
        self.test_data = pd.read_csv(test_path)

    def _get_split_row(self, csv_data):
        num_rows = csv_data.shape[0]
        split_at_row = int(num_rows * self.split_train_validation)
        return split_at_row


class TrainingData:
    def __init__(self,
                 training_data,
                 validation_data,
                 test_data,
                 label_column_name,
                 feature_pipeline=None):
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.label_column_name = label_column_name
        self.feature_pipeline = feature_pipeline

    @property
    def training_target(self):
        return self.training_data[self.label_column_name]

    @property
    def training_features(self):
        return self.training_data.drop(self.label_column_name, axis=1)

    @property
    def validation_target(self):
        return self.validation_data[self.label_column_name]

    @property
    def validation_features(self):
        return self.validation_data.drop(self.label_column_name, axis=1)

    @property
    def test_features(self):
        if self.label_column_name in self.test_data.columnns:
            return self.test_data.drop(self.label_column_name, axis=1)
        else:
            return self.test_data

if __name__ == '__main__':
    data = HousingData()
    print(data.train_data.head())
    print(data.train_data.dtypes)