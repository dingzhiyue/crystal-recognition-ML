from features_extraction import *
from build_crystal_lattice import *
from scipy.spatial import Voronoi
#import matplotlib.pyplot as plt
from Data_save_load import *

def q_l6m_tilt_fun(q_l6_m):#input list for 1 paritcle
    q_l6_m = np.array(q_l6_m)
    s = np.sum(pow(abs(q_l6_m), 2))
    q_l6_m_tile = q_l6_m / math.sqrt(s)
    return q_l6_m_tile.tolist()
#test
#a=[1,2,3]
#b = q_l6m_tilt_fun(a)
#print(np.array(a) / math.sqrt(14))


def disorder_fun(ref_P_ql6m_tilt, NN_index, particle_index, ql6m_tilt):#list
    NN_ql6m = []
    ref_P_ql6m_tilt = np.array(ref_P_ql6m_tilt)
    ql6m_tilt = np.array(ql6m_tilt)

    for item in NN_index:
        NN_ql6m.append(ql6m_tilt[particle_index.index(item)])

    s = 0
    for item in NN_ql6m:
        s = s + np.dot(ref_P_ql6m_tilt, np.conj(item))
    s = s / float(len(NN_index))
    return np.around(s, 2)

def disorder_filter(particle, cache_NN, cache_ql6m, features):  #input complete info list particle [(index, [coordiante]),...], cahce_NN [[indices],...], chace_ql6m[[],...], features [[],...]
    ql6m_tilt = [q_l6m_tilt_fun(item) for item in cache_ql6m]
    unfilterable_particle = []
    filtered_particle = []
    filtered_features = []
    disorders = []

    index, particle_position = zip(*particle)
    for i in range(len(cache_NN)):
        if set(cache_NN[i]) < set(index):
            s = disorder_fun(cache_ql6m[i], cache_NN[i], index, ql6m_tilt)
            filtered_particle.append(particle[i])
            filtered_features.append(features[i])
            disorders.append(np.around(s, 3))
        else:
            unfilterable_particle.append([particle[i], features[i]])

    return filtered_particle, filtered_features, disorders, unfilterable_particle  # filtered_particle list [(index,[coordinate]),...], filtered_features list[[],...], disorders list [...], unfilterable_particle list [[(index, [coordinate]),[feature]],...]


def pre_processing(crystal_lattice):
    particle = []
    particle_eliminated = []
    features = []
    cache_NN_index = []
    cache_q_l6_m = []

    lattice_voronoi = Voronoi(crystal_lattice)

    points = lattice_voronoi.points
    point_region = lattice_voronoi.point_region
    regions = lattice_voronoi.regions
    vertices = lattice_voronoi.vertices
    ridge_points = lattice_voronoi.ridge_points
    ridge_vertices = lattice_voronoi.ridge_vertices

    for i in range(len(points)):
        region_index = point_region[i]
        cell = regions[region_index]
        if -1 in cell:
            particle_eliminated.append((i, points[i].tolist()))
        else:
            particle.append((i, points[i].tolist()))
            h_dis, h_angle, q_w, minkowski_eig, NN_count, cache_NN_index_temp, cache_q_l6_m_temp = features_extract(i, ridge_points, ridge_vertices, points, vertices)
            feature_temp = output_features(h_dis, h_angle, q_w, minkowski_eig, NN_count)
            features.append(feature_temp)
            cache_NN_index.append(cache_NN_index_temp)
            cache_q_l6_m.append(cache_q_l6_m_temp)

    filtered_particle, filtered_features, disorders, unfilterable_particle = disorder_filter(particle, cache_NN_index, cache_q_l6_m, features)
    return filtered_particle, filtered_features, disorders, unfilterable_particle, particle_eliminated # filtered_particle/eliminated list [(particle_index, particle coordinate)...], filter_features list [[]...], disorders list [],unfilterable_particle list[[(index, coordinate),[feature]],...]


#test
crystal_lattice = crystal_lattice_bcc(5)
filtered_particle, filtered_features, disorders, unfilterable_particle, particle_eliminated = pre_processing(crystal_lattice)
print(len(filtered_particle))
print(len(unfilterable_particle))
print(len(particle_eliminated))
#inner = [item[1] for item in particle]
#x,y,z = zip(*inner)
#xx,yy,zz = zip(*crystal_lattice)
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(x,y,z,color='black', marker='o')
#ax.scatter(xx,yy,zz,color='green', marker='*')
#plt.show()

data_save(crystal_lattice, filtered_particle, filtered_features, disorders, unfilterable_particle, particle_eliminated)

