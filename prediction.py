import numpy as np
import keras.models as K
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from pre_processing import *
from features_extraction import *
from mpl_toolkits.mplot3d import Axes3D

def data_pre_process(path):
    df = read_csv(path)
    data = df.values
    filtered_particle, filtered_features, disorders, unfilterable_particle, particle_eliminated = pre_processing(
        data[:,0:3])
    return filtered_particle, filtered_features, disorders, unfilterable_particle, particle_eliminated

def prediction(x, model_path):
    model = K.load_model(model_path)
    y_pred = model.predict(x)
    predictions = np.argmax(y_pred, axis=1)#bcc=0, fcc=1, hcp=2
    return predictions


def classify(filtered_particle, predictions, unfilterable_particle, particle_eliminated):
    particle = [item[1] for item in filtered_particle]
    particle = np.array(particle)
    p_bcc = particle[predictions == 0, :]
    p_fcc = particle[predictions == 1, :]
    p_hcp = particle[predictions == 2, :]
    boundary_eliminate_temp = [item[1] for item in particle_eliminated]
    boundary_elimination = np.array(boundary_eliminate_temp)
    boundary2_temp = [item[0][1] for item in unfilterable_particle]
    boundary2 = np.array(boundary2_temp)

    return p_bcc, p_fcc, p_hcp, boundary2, boundary_elimination

def plot(p_bcc, p_fcc, p_hcp, boundary2, boundary_elimination):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if len(p_bcc) != 0:
        x_bcc,y_bcc,z_bcc = zip(*p_bcc)
        ax.scatter(x_bcc, y_bcc, z_bcc,c='blue', marker='o')
    if len(p_fcc) != 0:
        x_fcc, y_fcc, z_fcc = zip(*p_fcc)
        ax.scatter(x_fcc, y_fcc, z_fcc, c='black', marker='o')
    if len(p_hcp) != 0:
        x_hcp, y_hcp, z_hcp = zip(*p_hcp)
        ax.scatter(x_hcp, y_hcp, z_hcp, c='green', marker='o')
    #if len(boundary2) != 0:
    #    x_b2, y_b2, z_b2 = zip(*boundary2)
     #   ax.scatter(x_b2, y_b2, z_b2, c='red', marker='o')
    #if len(boundary_elimination) != 0:
      #  x_b, y_b, z_b = zip(*boundary_elimination)
       # ax.scatter(x_b, y_b, z_b, c='red', marker='o')
    ax.legend(['bcc', 'fcc', 'hcp'])
    plt.show()

def stat(p_bcc, p_fcc, p_hcp, boundary2, boundary_elimination):
    n_bcc = p_bcc.shape[0]
    n_fcc = p_fcc.shape[0]
    n_hcp = p_hcp.shape[0]
    n_b2 = boundary2.shape[0]
    n_b = boundary_elimination.shape[0]
    n_total_useful = n_bcc+n_fcc+n_hcp
    n_total = n_total_useful+n_b+n_b2

    bcc_portion = n_bcc / n_total_useful
    fcc_portion = n_fcc / n_total_useful
    hcp_portion = n_hcp / n_total_useful
    boundary_portion = (n_b+n_b2) / n_total
    return bcc_portion, fcc_portion, hcp_portion, boundary_portion


#verify pk4
def pick_interested_region(bcc,fcc, hcp, x1,x2,y1,y2,z1,z2 ):
    bcc = bcc.tolist()
    fcc = fcc.tolist()
    hcp = hcp.tolist()
    bcc_remain = [p for p in bcc if (p[0]>=x1) and (p[0]<x2) and (p[1]>=y1) and (p[1]<=y2) and (p[2]>=z1) and (p[2]<=z2)]
    fcc_remain = [p for p in fcc if
                  (p[0] >= x1) and (p[0] < x2) and (p[1] >= y1) and (p[1] <= y2) and (p[2] >= z1) and (p[2] <= z2)]
    hcp_remain = [p for p in hcp if
                  (p[0] >= x1) and (p[0] < x2) and (p[1] >= y1) and (p[1] <= y2) and (p[2] >= z1) and (p[2] <= z2)]

    bcc_pred = []
    for item in bcc_remain:
        item_temp = item.copy()
        item_temp.append(0)
        bcc_pred.append(item_temp)
    fcc_pred = []
    for item in fcc_remain:
        item_temp = item.copy()
        item_temp.append(1)
        fcc_pred.append(item_temp)
    hcp_pred = []
    for item in hcp_remain:
        item_temp = item.copy()
        item_temp.append(2)
        hcp_pred.append(item_temp)


    data_predict = bcc_pred + fcc_pred + hcp_pred
    data_predict = np.array(data_predict)
    pd.DataFrame(data_predict).to_csv('data_predict.csv')
    return bcc_remain, fcc_remain, hcp_remain

    #x_fcc, y_fcc, z_fcc = zip(*fcc_remain)
    #x_bcc, y_bcc, z_bcc = zip(*bcc_remain)
    #x_hcp, y_hcp, z_hcp = zip(*hcp_remain)
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.scatter(x_fcc, y_fcc, z_fcc, color='red', marker='o')
    #ax.scatter(x_bcc, y_bcc, z_bcc, color='blue', marker='o')
    #ax.scatter(x_hcp, y_hcp, z_hcp, color='green', marker='o')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #plt.show()

# main prediction
def prediction_main(data_path, model_path, if_plot='on'):#data in csv, model in .h5
    filtered_particle, filtered_features, disorders, unfilterable_particle, particle_eliminated = data_pre_process(data_path)




    x_pred = np.array(filtered_features)
    y_pred = prediction(x_pred, model_path)
    bcc, fcc, hcp, boundary2, boundary_elimination= classify(filtered_particle, y_pred, unfilterable_particle, particle_eliminated)
    #bcc_remain, fcc_remain, hcp_remain = pick_interested_region(bcc, fcc, hcp, 3, 7, 4, 6, 0, 11)
    #bcc_portion, fcc_portion, hcp_portion, boundary_portion = stat(bcc, fcc, hcp, boundary2, boundary_elimination)

    if if_plot=='on':
        plot(bcc, fcc, hcp, boundary2, boundary_elimination)
    return bcc, fcc, hcp, boundary2, boundary_elimination




def compare():
    data_interested = pd.read_csv('data_interested.csv')
    data_predict = pd.read_csv('data_predict.csv')
    data_interested = data_interested.values
    data_predict = data_predict.values

    data_interested2 = data_interested.tolist()
    data_predict2 = data_predict.tolist()

    data1 = [tuple(i) for i in data_interested2]
    data2 = [tuple(i) for i in data_predict2]
    s1 = set(data1)
    s2 = set(data2)

    intersection = list(s1 & s2)
    data3 = list(s1.difference(s1 & s2))
    intersection = np.array(intersection)
    data3 = np.array(data3)
    x1, y1, z1 = zip(*intersection[:,0:3])
    x2, y2, z2 = zip(*data3[:,0:3])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x1, y1, z1, color='blue', marker='o')
    ax.scatter(x2, y2, z2, color='black', marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(['correct', 'incorrect'])
    print(intersection.shape[0]/len(data1))
    plt.show()




    bcc_in = data_interested[data_interested[:, 3] == 0, 0:3]
    fcc_in = data_interested[data_interested[:, 3] == 1, 0:3]
    hcp_in = data_interested[data_interested[:, 3] == 2, 0:3]

    bcc_pred = data_predict[data_predict[:, 3] == 0, 0:3]
    fcc_pred = data_predict[data_predict[:, 3] == 1, 0:3]
    hcp_pred = data_predict[data_predict[:, 3] == 2, 0:3]

    bcc_in_x, bcc_in_y, bcc_in_z = zip(*bcc_in)
    fcc_in_x, fcc_in_y, fcc_in_z = zip(*fcc_in)
    hcp_in_x, hcp_in_y, hcp_in_z = zip(*hcp_in)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(fcc_in_x, fcc_in_y, fcc_in_z, color='black', marker='o')
    ax.scatter(bcc_in_x, bcc_in_y, bcc_in_z, color='blue', marker='o')
    ax.scatter(hcp_in_x, hcp_in_y, hcp_in_z, color='green', marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(['fcc', 'bcc', 'hcp'])
    plt.show()

    bcc_p_x, bcc_p_y, bcc_p_z = zip(*bcc_pred)
    fcc_p_x, fcc_p_y, fcc_p_z = zip(*fcc_pred)
    hcp_p_x, hcp_p_y, hcp_p_z = zip(*hcp_pred)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(fcc_p_x, fcc_p_y, fcc_p_z, color='black', marker='o')
    ax.scatter(bcc_p_x, bcc_p_y, bcc_p_z, color='blue', marker='o')
    ax.scatter(hcp_p_x, hcp_p_y, hcp_p_z, color='green', marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(['fcc','bcc','hcp'])
    plt.show()


if __name__=='__main__':
    data_path1 = 'bu_pk4\Latt3D_bgsubbed.csv'
    data_path2 = 'bu_pk4\Latt3D_cropped.csv'
    data_path3 = 'bu_pk4\Latt3D_highthresh.csv'
    data_path4 = 'bu_pk4\Latt3D_lowthresh.csv'
    model_path = 'Ding_model3.h5'
    #filtered_particle, filtered_features, disorders, unfilterable_particle, particle_eliminated = data_pre_process(data_path)
    bcc, fcc, hcp, boundary2, boundary_elimination = prediction_main(data_path1, model_path, if_plot = 'on')
    #bcc_portion, fcc_portion, hcp_portion, boundary_portion = prediction_main(data_path2, model_path, if_plot='on')
    #bcc_portion, fcc_portion, hcp_portion, boundary_portion = prediction_main(data_path3, model_path, if_plot='on')
    #bcc_portion, fcc_portion, hcp_portion, boundary_portion = prediction_main(data_path4, model_path, if_plot='on')
    #compare()


    a = pd.read_csv('data.csv')
    d = a.values
    x,y,z,p = zip(*d)
    x=np.array(x)
    y = np.array(y)
    z = np.array(z)
    p = np.array(p)
    f = plt.figure()
    ax = f.gca(projection='3d')
    ax.scatter(x[p == 3.0],y[p == 3.0],z[p==3.0],color='red')
    ax.scatter(x[p == 0.0], y[p == 0.0], z[p == 0.0], color='blue')
    ax.scatter(x[p == 1.0], y[p == 1.0], z[p == 1.0], color='black')
    ax.scatter(x[p == 2.0], y[p == 2.0], z[p == 2.0], color='green')
    ax.legend(['outside particle','fcc', 'bcc', 'hcp'])
    plt.show()

    filtered_particle, filtered_features, disorders, unfilterable_particle, particle_eliminated=  data_pre_process(data_path1)