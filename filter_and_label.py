from Data_save_load import *
import numpy as np


def filter_and_label(data, s, type='bcc'):#s filter threshold
    if type == 'bcc':
        label = 1
    elif type == 'fcc':
        label = 2
    elif type == 'hcp':
        label = 3
    labels=[]
    for item in data:
        if item[-1] < s:
            labels.append(0)
        else:
            labels.append(label)
    return np.array(labels)

def concat(disorder_threshold, lattice_type='bcc'):
    load_path = 'D:\\pk4ML\\kodiak\\' + lattice_type + '\\' + lattice_type
    data = data_load_feature(load_path + str(3) + '\\')
    for i in range(4, 16):
        feature = data_load_feature(load_path + str(i) + '\\')
        data = np.concatenate((data, feature), axis=0)

    labels = filter_and_label(data, disorder_threshold, lattice_type)
    return data, np.reshape(labels, (-1, 1))
data, lab = concat(0.3, lattice_type='fcc')

def training_data(disorder_threshold):
    bcc_feature, bcc_label = concat(disorder_threshold, lattice_type='bcc')
    fcc_feature, fcc_label = concat(disorder_threshold, lattice_type='fcc')
    hcp_feature, hcp_label = concat(disorder_threshold, lattice_type='hcp')
    X_train = np.concatenate((bcc_feature, fcc_feature, hcp_feature), axis=0)
    Y_train = np.concatenate((bcc_label, fcc_label, hcp_label), axis=0)

    save_path = 'D:\\pk4ML\\training_data\\'
    np.savetxt(save_path + 'X_train.csv', X_train[:, 0:31], delimiter=',')
    np.savetxt(save_path + 'Y_train.csv', Y_train, delimiter=',')
    print('training_data saved')


training_data(0.3)
