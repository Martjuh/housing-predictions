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
validation_data = feature_generation.generate_output(housing_data.validation_data)
test_data = feature_generation.generate_output(housing_data.test_data, includes_label=False)

data = TrainingData(training_data,
                    validation_data,
                    test_data,
                    label_column_name)

model = LinearRegressionModel(data.training_data)

EPOCHS = 1000
VALIDATION_INTERVAL = 50
LR = 1
BATCH_SIZE = 4

loss_function_MSE = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

feature_train_data = torch.tensor(data.training_features.values).to(torch.float32)
feature_validation_data = torch.tensor(data.validation_features.values).to(torch.float32)
target_train_data = torch.tensor(data.training_target.values).to(torch.float32)
target_validation_data = torch.tensor(data.validation_target.values).to(torch.float32)

dataset = torch.utils.data.TensorDataset(feature_train_data, target_train_data)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
losses = []

print(f'Dataset size: {data.training_features.shape[0]}')
for epoch in range(EPOCHS):
    epoch_losses = []
    for id_batch, (features, target) in enumerate(data_loader):
        prediction = model(features).squeeze(-1)
        loss_RMSE = torch.sqrt(loss_function_MSE(prediction, target))
        epoch_losses.append(loss_RMSE.item())

        optimizer.zero_grad()
        loss_RMSE.backward()
        optimizer.step()

    epoch_loss = sum(epoch_losses) / len(epoch_losses)

    loss = torch.sqrt(loss_function_MSE(model(feature_validation_data).squeeze(-1), target_validation_data))
    losses.append(loss)
    if epoch % VALIDATION_INTERVAL == 0:
        print(f'---------------------------------')
        print(f'Epoch {epoch + 1}:')
        print(f'Training loss: {epoch_loss}, Test loss: {loss}')

