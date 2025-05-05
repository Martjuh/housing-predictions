import pandas as pd
from sklearn import preprocessing

class Feature:
    def __init__(self, name : str, feature_type = ''):
        self.name = name
        self.feature_type = feature_type

    def _calculate(self, input_data):
        return input_data


class FeaturePipeline:
    def __init__(self, label_column_name = '', keep_columns = None):
        self.pipeline = []
        self.label_column_name = label_column_name
        self.keep_columns = keep_columns.copy()
        if label_column_name not in self.keep_columns:
            self.keep_columns.append(label_column_name)

    def add(self, feature : Feature):
        self.pipeline.append(feature)

    def show(self):
        print('Used calculations:')
        for feature in self.pipeline:
            print(f'- Feature {feature.name}')

    def set_label_column(self, label_column_name):
        self.label_column_name = label_column_name

    def generate_output(self, data : pd.DataFrame):
        if not isinstance(self.keep_columns, list):
            print(f'Warning! Empty or incorrect value for keep_columns: {self.keep_columns}.\n Starting with empty list!') # TODO Proper logging
            keep_columns = []
        else:
            keep_columns = self.keep_columns

        for feature in self.pipeline:
           data = feature.calculate(data)

        data = data[keep_columns]
        return data.dropna()


class MeanNormalizer(Feature):
    def __init__(self,
                 column_names,
                 name = 'MeanNormalizer',
                 feature_type = 'scaler'):
        super().__init__(name, feature_type)
        self.column_names = column_names
        self.scaler = preprocessing.Normalizer()

    def calculate(self, input_data):
        input_data[self.column_names] = self.scaler.fit_transform(input_data[self.column_names])
        return input_data

class MinMaxNormalizer(Feature):
    def __init__(self,
                 column_names,
                 name = 'MinMaxNormalizer',
                 feature_type = 'scaler'):
        super().__init__(name, feature_type)
        self.column_names = column_names
        self.scaler = preprocessing.MinMaxScaler()

    def calculate(self, input_data):
        input_data[self.column_names] =  self.scaler.fit_transform(input_data[self.column_names])
        return input_data