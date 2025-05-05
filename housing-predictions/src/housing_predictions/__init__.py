import torch.nn

from data import HousingData, TrainingData
from features import *
from models import LinearRegressionModel

housing_data = HousingData()

basic_features = [
    'LotArea',
    'OverallQual',
    'OverallCond',
    'MasVnrArea',
    'PoolArea',
    'EnclosedPorch'
]

label_column_name = 'SalePrice'
feature_generation = FeaturePipeline(label_column_name=label_column_name,
                                     keep_columns=basic_features)
feature_generation.add(MinMaxNormalizer(basic_features))


training_data = feature_generation.generate_output(housing_data.train_data)
data = TrainingData(training_data,
                    label_column_name)

model = LinearRegressionModel(data.training_data)

EPOCHS = 1000
EVAL_INTERVAL = 50
LR = 1
BATCH_SIZE = 4

loss_function_MSE = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

feature_data = torch.tensor(data.features.values).to(torch.float32)
target_data = torch.tensor(data.target.values).to(torch.float32)

dataset = torch.utils.data.TensorDataset(feature_data, target_data)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
losses = []

print(f'Dataset size: {data.features.shape[0]}')
for epoch in range(EPOCHS):
    for id_batch, (features, target) in enumerate(data_loader):
        prediction = model(features).squeeze(-1)
        loss_RMSE = torch.sqrt(loss_function_MSE(prediction, target))

        optimizer.zero_grad()
        loss_RMSE.backward()
        optimizer.step()


    loss = torch.sqrt(loss_function_MSE(model(feature_data).squeeze(-1), target_data))
    losses.append(loss)
    if epoch % EVAL_INTERVAL == 0:
        print(f'Epoch {epoch}:')
        print(f'Training loss: {loss}, Test loss: {loss}')

