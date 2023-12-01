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
from tensorflow.python.keras import backend as K
from dataProcess import dataProcess

# from TrainablePositionalEmbeddings import TransformerPositionalEmbedding
# from TransformerEncoder import TransformerEncoder

from keras.utils import plot_model

#from attention import Attention
from transformer import Transformer

def train_model(seed_num, epoch, modelType, testFlag, patientFlag, layerNumber, featureNumber, plotFlag,horizon):
    os.environ['PYTHONHASHSEED'] = str(seed_num)

    random.seed(seed_num)
    np.random.seed(seed_num)
    tf.random.set_seed(seed_num)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                            inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    #K.set_session(sess)

    train_X0, train_y0, val_X0, val_y0, test_X0, test_y0, scaler = dataProcess(featureNumber,patientFlag, plotFlag)
    # positional_ff_dim   = 128
    # batch_size = 128 # How many time series to train on before updating model's weight parameters
    # output_seq_len = 6 # How many months to predict into the future
    # input_seq_len = 24 # How many months to train on in the past

    # # Internal neural network parameters
    # input_dim =1
    # output_dim = 1 # Univariate time series (predicting future values based on stream of historical values)
    # hidden_dim = 100  # Number of neurons in each recurrent unit
    # num_layers = 2  # Number of stacked recurrent cells (number of recurrent layers)

    # # Optimizer parameters
    # learning_rate = 0.005  # Small lr helps not to diverge during training.
    # epochs = 300  # How many times we perform a training step (how many times we show a batch)
    # lr_decay = 0.9  # default: 0.9 . Simulated annealing.
    # momentum = 0.2  # default: 0.0 . Momentum technique in weights update
    # lambda_l2_reg = 0.003  # L2 regularization of weights - reduces overfitting
    val_X1 = val_X0.reshape((val_X0.shape[0], val_X0.shape[2]))
    inv_y = concatenate((val_X1,val_y0), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y1 = inv_y[:,-6:]
    tr = Transformer(epoch,horizon)
    history=tr.train(train_X0, train_y0, val_X0,val_y0, epoch)
    yhat = tr.evaluate(val_X0,val_y0)
    min_val = min((history.history['val_loss']))
    epoch_val = history.history['val_loss'].index(min_val)
    epoch_val = epoch_val + 1

    return min_val, epoch_val