import torch
import pandas as pd
import numpy as np

class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_data):
        super(LinearRegressionModel, self).__init__()
        num_features = len(input_data.columns) - 1
        self.model = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        y_prediction = self.model(x)
        return y_prediction

    def evaluate(self, prediction, target):
        prediction_log = np.log(prediction.numpy())
        target_log = np.log(target.numpy())

        df = pd.DataFrame(data=(prediction_log, target_log), columns=['Predicted', 'Target'])
        print(df)

