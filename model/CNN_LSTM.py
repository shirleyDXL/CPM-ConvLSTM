import pandas as pd
import numpy as np


# cnn lstm model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation


"""
train_X.shape = (21581, 19, 39, 23, 1)
train_y.shape = (21581, 897)
"""


def CNN_LSTM_model(trainX, trainy, test_X, test_y,epochs):
    # define model
    verbose, batch_size = 0,16
#     n_timesteps, n_row,n_col, n_outputs = trainX.shape[1], trainX.shape[2],trainX.shape[3],trainy.shape[-1]
    n_timesteps, n_row,n_col,n_dims, n_outputs = trainX.shape[1], trainX.shape[2],trainX.shape[3],trainX.shape[4], trainy.shape[-1]

    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv2D(filters=64, kernel_size=3, activation='relu', padding = "same"), input_shape=(n_timesteps, n_row, n_col, n_dims)))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=2, padding = "same")))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50,return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(100,return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(n_outputs))
#     model.add(Activation("linear"))

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mae'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    predicted = model.predict(test_X)
    # evaluate model
    scores = model.evaluate(test_X, test_y, batch_size=16)
    print("\nevaluate result: \nmse={:.6f}\nmae={:.6f}".format(scores[0], scores[1]))


    return model, test_y, predicted







