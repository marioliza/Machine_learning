import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder




def mes(row):
    if row['mes'] < 10:
        return '0' + str(row['mes'])
    else:
        return str(row['mes'])

###---------------------------------------------------------
df = pd.read_pickle('./Data/lstm_mensual_2.pkl')
df['mes'] = df.apply(lambda row: mes(row), axis=1)
df['Fecha_Inicio'] = df['year'].astype(str) + '-' + df['mes'].astype(str) + '-01'

df.Fecha_Inicio = pd.to_datetime(df.Fecha_Inicio)
df_exito = df[(df.Razon_Social_Cliente == 'Almacenes Exito Sa')& (df.Fecha_Inicio < '2023-11-01')].groupby(['Fecha_Inicio'],as_index=False).COP.sum()
df_exito = df_exito.set_index('Fecha_Inicio')
df_exito = df_exito.sort_index(ascending=False)
df_exito.COP / 100000
# Train test split ------------------------ Window view ------------------
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

sc = MinMaxScaler()
training_data = sc.fit_transform(df_exito)

seq_length = 3
x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * 0.80)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))



