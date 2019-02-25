from pandas import read_csv
import numpy as np

save_path='D:\\pk4ML\\data\\bcc0\\'


def data_save(crystal_lattice, filtered_particle, filtered_features, disorders, unfilterable_particle, particle_eliminated):

    #crystal_lattice.csv = [[x,y,z],...]
    crystal_L = np.array(crystal_lattice)
    np.savetxt(save_path + 'crystal_lattice.csv', crystal_L, delimiter=',')

    #filtered_particle.csv = [index,x,y,z] dimension =4
    filtered_P = []
    for item in filtered_particle:
        a = item[1].copy()
        a.insert(0, item[0])
        filtered_P.append(a)

    np.savetxt(save_path + 'filtered_particle.csv', np.array(filtered_P), delimiter=',')

    #filtered_features.csv = [features, disorders] 31 + 1
    filtered_F = []
    for i, item in enumerate(filtered_features):
        if disorders[i].imag != 0:
            print('disorders is complex')
        else:
            a = item.copy()
            a.append(disorders[i].real)
            filtered_F.append(a)
    filtered_F = np.array(filtered_F)
    if filtered_F.dtype.type is np.complex128 or filtered_F.dtype.type is np.complex:
        print('flitered feature contains complex, need double check')
        filtered_F = filtered_F.real
    np.savetxt(save_path + 'filtered_features.csv', filtered_F, delimiter=',')

    #unfilterable_particles.csv = [[index, x,y,z,features],....] dimension=35
    unfilterable_P = []
    for item in unfilterable_particle:
        a = item[0][1].copy() + item[1].copy()
        a.insert(0, item[0][0])
        unfilterable_P.append(a)
    unfilterable_P = np.array(unfilterable_P)
    if unfilterable_P.dtype.type is np.complex128 or unfilterable_P.dtype.type is np.complex:
        print('unfilterable feature contains complex, need double check')
        unfilterable_P = unfilterable_P.real
    np.savetxt(save_path + 'unfilterable_particles.csv', unfilterable_P, delimiter=',')

    #eliminated_particle.csv = [[index,x,y,z],...] 4
    eliminated_P = []
    for item in particle_eliminated:
        a = item[1].copy()
        a.insert(0, item[0])
        eliminated_P.append(a)
        np.savetxt(save_path + 'eliminated_particle.csv', np.array(eliminated_P), delimiter=',')

    print('data saved')


def data_load(load_path):
    filtered_features = read_csv(load_path + 'filtered_features.csv', dtype=float)
    filtered_particles = read_csv(load_path + 'filtered_particle.csv', dtype=float)
    eliminated_particles = read_csv(load_path + 'eliminated_particle.csv', dtype=float)
    unfilterable_particles = read_csv(load_path + 'unfilterable_particles.csv', dtype=float)
    crystal_lattice = read_csv(load_path + 'crystal_lattice.csv', dtype=float)
    return filtered_features.values, filtered_particles.values, eliminated_particles.values, unfilterable_particles.values, crystal_lattice.values


def data_load_feature(load_path):
    filtered_features = read_csv(load_path + 'filtered_features.csv', dtype=float)
    return filtered_features.values

