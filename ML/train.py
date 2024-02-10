import mlflow
import numpy as np
from data import trainX, trainY, testY, testX
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import ParameterGrid
from params import lst_params_grid
from utils import eval_metrics
from model import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
from data import df_exito


sc = MinMaxScaler()
training_data = sc.fit_transform(df_exito)
# Loop through the hyperparameter combinations and log results in separate runs

with mlflow.start_run():

        warnings.filterwarnings("ignore")
        np.random.seed(40)
        num_epochs = 500
        learning_rate = 0.01

        input_size = 1
        hidden_size = 256
        num_layers = 1
        weight_decay = 0.01

        num_classes = 1

        lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

        criterion = torch.nn.MSELoss()  # mean-squared error for regression
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate,weight_decay=weight_decay)

        # Train the model
        for epoch in range(num_epochs):
            outputs = lstm(trainX)
            optimizer.zero_grad()

            # obtain the loss function
            loss = criterion(outputs, trainY)

            loss.backward()

            optimizer.step()
            data_predict = outputs.data.numpy()
            dataY_plot = trainY.data.numpy()

            data_predict = sc.inverse_transform(data_predict)
            dataY_plot = sc.inverse_transform(dataY_plot)
            rmse = np.sqrt(mean_squared_error(dataY_plot, data_predict))
            mse = mean_squared_error(dataY_plot, data_predict)
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f, rmse: %0.f" % (epoch, mse, rmse))

            lstm.eval()
            train_predict = lstm(testX)

            data_predict = train_predict.data.numpy()
            dataY_plot = testY.data.numpy()

            data_predict = sc.inverse_transform(data_predict)
            dataY_plot = sc.inverse_transform(dataY_plot)

            y = sc.inverse_transform(data_predict)
            x = sc.inverse_transform(testY)

        lstm.eval()
        train_predict = lstm(testX)

        data_predict = train_predict.data.numpy()
        dataY_plot = testY.data.numpy()

        data_predict = sc.inverse_transform(data_predict)
        dataY_plot = sc.inverse_transform(dataY_plot)
        metrics = eval_metrics(dataY_plot,data_predict)
        # Logging the inputs such as dataset
        mlflow.log_input(
            mlflow.data.from_numpy(trainX.numpy()),
            context='Training dataset'
        )

        mlflow.log_input(
            mlflow.data.from_numpy(testX.numpy()),
            context='Validation dataset'
        )

        # Logging hyperparameters
        #mlflow.log_params(params)

        # Logging metrics
        mlflow.log_metrics(metrics)
        signature = {
            "inputs": [
                {"name": "input_tensor", "type": "torch.Tensor"}
            ],
            "outputs": [
                {"name": "output_tensor", "type": "torch.Tensor"}
            ]
        }
        # Log the trained model
        mlflow.pytorch.log_model(
            pytorch_model = lstm,
            #signature = signature,
            input_example= np.array(trainX),
            code_paths=['train.py','data.py','params.py','utils.py'],
            artifact_path = "C:/Users/mario/OneDrive]/Escritorio/Machine_Learning/mlflow/mlflow-artifacts"
        )

