# convlstm model
import pandas as pd
import numpy as np


from util.params import Params as param
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import ConvLSTM2D
from keras.layers.convolutional import Conv3D
from keras.layers import Activation


"""
train_X.shape = (21581, 19, 39, 23, 1)
train_y.shape = (21581, 39, 23, 1)
"""

# fit and evaluate a model
def convLSTM_model(trainX, trainy, testX, testy,filters_num,epochs):
    # define model

    verbose, batch_size = 0, param.batch_size
    n_steps, n_row, n_col, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainX.shape[3], trainX.shape[4], trainy.shape[-1]
    # define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=filters_num, kernel_size=(3,3), activation=param.activation, padding="same", input_shape=(n_steps, n_row, n_col, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dense(n_outputs))#分类使用这个很差
    model.add(Activation("linear"))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(n_outputs, activation='softmax'))


    model.compile(loss="mse", optimizer="rmsprop", metrics=['mae'])

    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    predicted = model.predict(testX)
    # evaluate model
    scores = model.evaluate(testX, testy, batch_size=batch_size)
    print("\nevaluate result: \nmse={:.6f}\nmae={:.6f}".format(scores[0], scores[1]))
    
    return model, testy, predicted