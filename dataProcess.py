#Importing Libraries
import random
from keras import initializers, regularizers, constraints
from pandas import read_csv
import numpy as np
from keras import Model
from keras.layers import Layer, Lambda, MaxPool2D
import keras.backend as K
from keras.layers import Input, Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.metrics import mean_squared_error
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
from pandas import concat
from math import sqrt
from numpy import concatenate
import tensorflow as tf
import seaborn as sns
from keras.losses import mean_squared_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import math
import datetime as dt
from keras.layers import Dense, Activation, BatchNormalization, LSTM, Bidirectional, TimeDistributed, Conv1D, \
    MaxPooling1D, Flatten, ConvLSTM2D, Conv2D, MaxPooling2D, Dropout, AveragePooling3D, RepeatVector, GRU, SimpleRNN
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import optimizers, Model, Input
from keras import backend as K
import xlsxwriter as xlsxwriter
import time


def dataProcess(featureNumber,patientFlag, plotFlag):

    ####################################################################################################################
    # Initialization
    patientTrainList=['all_data/540training.csv', 'all_data/544training.csv', 'all_data/552training.csv',
                      'all_data/559training.csv', 'all_data/563training.csv', 'all_data/567training.csv',
                      'all_data/570training.csv', 'all_data/575training.csv', 'all_data/584training.csv',
                      'all_data/588training.csv', 'all_data/591training.csv', 'all_data/596training.csv']

    patientTestList = ['all_data/540testing.csv', 'all_data/544testing.csv', 'all_data/552testing.csv',
                        'all_data/559testing.csv', 'all_data/563testing.csv', 'all_data/567testing.csv',
                        'all_data/570testing.csv', 'all_data/575testing.csv', 'all_data/584testing.csv',
                        'all_data/588testing.csv', 'all_data/591testing.csv', 'all_data/596testing.csv' ]

    ####################################################################################################################


    dataset = pd.read_csv(patientTrainList[patientFlag], header=0, index_col=0, usecols=[i for i in range(featureNumber + 1)])
    #dataset.drop(dataset.columns[[2]], axis=1, inplace=True)
    test_dataset = pd.read_csv(patientTestList[patientFlag], header=0, index_col=0, usecols=[i for i in range(featureNumber + 1)])
    #test_dataset.drop(test_dataset.columns[[2]], axis=1, inplace=True)
    

    if plotFlag == 1:
        corr = dataset.corr()
        # print(corr)
        # Increase the size of the heatmap.
        plt.figure(figsize=(16, 6))
        # Store heatmap object in a variable to easily access it when you want to include more features (such as title).
        # Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
        heatmap = sns.heatmap(dataset.corr(), vmin=-1, vmax=1, annot=True)
        # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
        plt.show()

        plt.figure(figsize=(8, 12))
        heatmap = sns.heatmap(dataset.corr()[['CGM']].sort_values(by='CGM', ascending=False), vmin=-1, vmax=1, annot=True,
                              cmap='BrBG')
        heatmap.set_title('Features Correlating with Glucose Level', fontdict={'fontsize': 18}, pad=16)
        plt.show()

    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in - 1, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out + 1):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    df5 = series_to_supervised(dataset, 6, 6)
    # print(np.shape(df5))
    df5_test = series_to_supervised(test_dataset, 6, 6)

    if featureNumber == 7:
        df5.drop(df5.columns[
                     [83, 82, 81, 80, 79, 78, 76, 75, 74, 73, 72, 71, 69, 68, 67, 66, 65, 64, 62, 61, 60, 59, 58, 57,
                      55, 54, 53, 52, 51, 50, 48, 47, 46, 45, 44, 43]], axis=1, inplace=True)

    elif featureNumber == 6:
        df5.drop(df5.columns[
                     [71, 70, 69, 68, 67, 65, 64, 63, 62, 61, 59, 58, 57, 56, 55, 53, 52, 51, 50, 49, 47, 46, 45, 44,
                      43, 41, 40, 39, 38, 37]], axis=1, inplace=True)
    elif featureNumber == 5:
        df5.drop(df5.columns[
                     [59, 58, 57, 56, 54, 53, 52, 51, 49, 48, 47, 46, 44, 43, 42, 41, 39, 38, 37, 36, 34, 33, 32, 31]],
                 axis=1, inplace=True)
    elif featureNumber == 4:
        df5.drop(df5.columns[[47, 46, 45, 43, 42, 41, 39, 38, 37, 35, 34, 33, 31, 30, 29, 27, 26, 25]], axis=1,
                 inplace=True)
    elif featureNumber == 3:
        df5.drop(df5.columns[[35, 34, 32, 31, 29, 28, 26, 25, 23, 22, 20, 19]], axis=1, inplace=True)
    elif featureNumber == 2:
        df5.drop(df5.columns[[23, 21, 19, 17, 15, 13]], axis=1, inplace=True)

    if featureNumber == 7:
        df5_test.drop(df5_test.columns[
                          [83, 82, 81, 80, 79, 78, 76, 75, 74, 73, 72, 71, 69, 68, 67, 66, 65, 64, 62, 61, 60, 59, 58,
                           57, 55, 54, 53, 52, 51, 50, 48, 47, 46, 45, 44, 43]], axis=1, inplace=True)

    elif featureNumber == 6:
        df5_test.drop(df5_test.columns[
                          [71, 70, 69, 68, 67, 65, 64, 63, 62, 61, 59, 58, 57, 56, 55, 53, 52, 51, 50, 49, 47, 46, 45,
                           44, 43, 41, 40, 39, 38, 37]], axis=1, inplace=True)
    elif featureNumber == 5:
        df5_test.drop(
            df5_test.columns[
                [59, 58, 57, 56, 54, 53, 52, 51, 49, 48, 47, 46, 44, 43, 42, 41, 39, 38, 37, 36, 34, 33, 32, 31]],
            axis=1, inplace=True)
    elif featureNumber == 4:
        df5_test.drop(df5_test.columns[[47, 46, 45, 43, 42, 41, 39, 38, 37, 35, 34, 33, 31, 30, 29, 27, 26, 25]],
                      axis=1, inplace=True)
    elif featureNumber == 3:
        df5_test.drop(df5_test.columns[[35, 34, 32, 31, 29, 28, 26, 25, 23, 22, 20, 19]], axis=1, inplace=True)
    elif featureNumber == 2:
        df5_test.drop(df5_test.columns[[23, 21, 19, 17, 15, 13]], axis=1, inplace=True)

 #ensure all data is float
    values5 = df5.values
    values5 = values5.astype('float32')
    #test_values = test_values.astype('float32')
    test_values5 = df5_test.values
    test_values5 = test_values5.astype('float32')
    # normalize features
    scaler = StandardScaler()
    #scaler_df = MinMaxScaler(feature_range=(0, 1))

    scaled5 = scaler.fit_transform(values5)
    test_scaled = scaler.fit_transform(test_values5)

    train5=scaled5
    test5=test_scaled

    split_v= round(len(train5)*0.80)


    # split into input and outputs
    train_X0, train_y0 = train5[:split_v, :-6], train5[:split_v, -6:]
    val_X0, val_y0 = train5[split_v:,:-6], train5[split_v:,-6:]
    test_X0, test_y0 = test5[:, :-6], test5[:, -6:]

    #train_y=np.asarray(train_y0).reshape(( -1 , 1 ))
    #val_y=np.asarray(val_y0).reshape(( -1 , 1 ))
    #test_y=np.asarray(test_y0).reshape(( -1 , 1 ))

    # reshape input to be 3D [samples, timesteps, features]
    train_X0 = train_X0.reshape((train_X0.shape[0], 1, train_X0.shape[1]))
    val_X0 = val_X0.reshape((val_X0.shape[0], 1, val_X0.shape[1]))
    test_X0= test_X0.reshape((test_X0.shape[0], 1, test_X0.shape[1]))
    print("Train X shape: ", train_X0.shape)
    print("Train y shape: ", train_y0.shape)
    print("Validation X shape: ", val_X0.shape)
    print("Validation y shape: ", val_y0.shape)
    print("Test X shape: ", test_X0.shape)
    print("Test y shape: ", test_y0.shape)

    return  train_X0, train_y0, val_X0, val_y0, test_X0, test_y0, scaler