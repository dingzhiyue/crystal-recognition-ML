import keras as K
from pandas import read_csv
import numpy as np
import json
import os

path = os.path.abspath(os.path.dirname(os.getcwd()))

def concate_training_his(his_old, his_new):
    val_loss = his_old['val_loss'] + his_new['val_loss']
    val_acc = his_old['val_acc'] + his_new['val_acc']
    loss = his_old['loss'] + his_new['loss']
    acc = his_old['acc'] + his_new['acc']
    history = {}
    history['val_loss'] = val_loss
    history['val_acc'] = val_acc
    history['loss'] = loss
    history['acc'] = acc
    return history
#test
#a = {}
#a['val_loss'] = [1,2,3]
#a['val_acc'] = [4,5,6]
#a['loss'] = [7,8,9]
#a['acc'] = [0,0,0]
#b = {}
#b['val_loss'] = [1,2,3]
#b['val_acc'] = [4,5,6]
#b['loss'] = [7,8,9]
#b['acc'] = [0,0,0]
#z=concate_training_his(a,b)


def data_load():
    X_train = read_csv(path + '\\X_train_shuffled.csv', dtype=float)
    Y_train = read_csv(path + '\\Y_train_shuffled.csv', dtype=float)
    Y_train_onehot = K.utils.to_categorical(Y_train.values)
    return X_train.values, Y_train_onehot
X_train, Y_train = data_load()

model_continue = K.models.load_model('Ding_model.h5')
model_continue.compile(optimizer=K.optimizers.Adam(lr=0.0001), loss=K.losses.categorical_crossentropy, metrics=['accuracy'])
his = model_continue.fit(X_train, Y_train, validation_split=0.1, batch_size=64, epochs=700, shuffle=True, verbose=2)

with open('Ding_history.json', 'r') as f:
    history_old = json.load(f)
    history_new = his.history
    history = concate_training_his(history_old, history_new)
with open('Ding_history.json', 'w') as f2:
    json.dump(history, f2)

model_continue.save('Ding_model.h5')