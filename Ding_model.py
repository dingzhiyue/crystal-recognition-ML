import keras as K
from pandas import read_csv
import numpy as np
import json
import os

path = os.path.abspath(os.path.dirname(os.getcwd()))

def data_load():
    X_train = read_csv(path + '\\X_train_shuffled.csv', dtype=float)
    Y_train = read_csv(path + '\\Y_train_shuffled.csv', dtype=float)
    Y_train_onehot = K.utils.to_categorical(Y_train.values)
    return X_train.values, Y_train_onehot


X_train, Y_train = data_load()


def model_Ding(input_size):
    X_input = K.layers.Input(shape=input_size, name='input')
    X = K.layers.Dense(units=250, activation='relu', kernel_initializer=K.initializers.he_normal(seed=1), use_bias=True,
                       bias_initializer=K.initializers.Zeros(), name='Dense1')(X_input)
    X = K.layers.Dense(units=100, activation='relu', kernel_initializer=K.initializers.he_normal(seed=1), use_bias=True,
                       bias_initializer=K.initializers.Zeros(), name='Dense2')(X)
    X = K.layers.Dropout(rate=0.2)(X)
    X = K.layers.Dense(units=50, activation='relu', kernel_initializer=K.initializers.he_normal(seed=1), use_bias=True,
                       bias_initializer=K.initializers.Zeros(), name='Dense3')(X)
    X = K.layers.Dropout(rate=0.2)(X)
    X = K.layers.Dense(units=3, activation='softmax', kernel_initializer=K.initializers.he_normal(seed=1), use_bias=True,
                       bias_initializer=K.initializers.Zeros(), name='Dense4')(X)
    model = K.Model(inputs=X_input, outputs=X)
    return model

Ding_model = model_Ding((31,))
Ding_model.compile(optimizer=K.optimizers.Adam(lr=0.0001), loss=K.losses.categorical_crossentropy, metrics=['accuracy'])
his = Ding_model.fit(X_train, Y_train, validation_split=0.1, batch_size=64, epochs=1, shuffle=True, verbose=2)

with open('Ding_history.json', 'w') as f:
    json.dump(his.history, f)

Ding_model.save('Ding_model.h5')

#onehot_decoding = np.argmax(onehot_encoded[0])# one hot decoding