from build_crystal_lattice import *
import numpy as np
from scipy.spatial import Voronoi
import itertools
import scipy
from scipy import special
from sympy.physics.wigner import wigner_3j

def index2position(list_of_index, list_of_position):#given index list find the corresponding item in list_of_position
    position = []
    for item in list_of_index:
        position.append(list_of_position[item])
    return position
#test
#a=[[1,1],[2,5],[2,1],[0,4]]
#ind=[0,2]
#p=index2position(ind, a)
#print(p)

def clockwise_sort(facet, normal_vector):#input list coplanar points
    #check normal
    check_vect1 = np.array(facet[0])-np.array(facet[1])
    check_vect2 = np.array(facet[1])-np.array(facet[2])
    if abs(np.dot(check_vect1, normal_vector)) > 1e-6 or abs(np.dot(check_vect2, normal_vector)) > 1e-6:
        print('facet_area error: points are not coplanar')

    #orthognal vectors
    epsilon = 1e-15
    a1 = np.random.rand()
    b1 = np.random.rand()
    c1 = (a1 * (epsilon + normal_vector[0]) + b1 * (epsilon + normal_vector[1])) / (-normal_vector[2]-epsilon)
    v1 = [a1, b1, c1]
    v1 = v1 / np.linalg.norm(v1)

    c2 = np.random.rand()
    A = np.array([[(epsilon + normal_vector[0]), (epsilon + normal_vector[1])], [a1, b1]])
    B = np.array([-(epsilon + normal_vector[2]) * c2, -c2 * c1]).T
    r= np.linalg.solve(A, B)
    v2 = [r[0], r[1], c2]
    v2 = v2 / np.linalg.norm(v2)
    if abs(np.dot(v1, v2)) > 1e-10 or abs(np.dot(v1, normal_vector)) > 1e-10:
        print('built vectors are not orthognal')

    #projection to the normvector plane
    coplanar = [np.dot(facet, v1).tolist(), np.dot(facet, v2).tolist()]
    temp_coplanar_facet = list(zip(coplanar[0], coplanar[1]))
    #get polar angle
    center = np.sum(temp_coplanar_facet, 0) / len(temp_coplanar_facet)
    facet_2d = []
    for i in range(len(temp_coplanar_facet)):
        item = temp_coplanar_facet[i]
        x = item[0] - center[0]
        y = item[1] - center[1]
        phi = np.arctan2(y, x)
        temp = [item[0], item[1], phi]
        facet_2d.append(temp)

    #clockwise sort
    #print('before sort',facet)
    facet_2d.sort(key=lambda x:x[2])
    facet_sorted = []
    for item in facet_2d:
        facet_sorted.append(item[0:2])
    #print('sort',facet_sorted)
    return facet_sorted
#test
#f= clockwise_sort([[4,4,2],[2,1,4],[1,2,3]], [1,1,1])
def polyarea(facet, normal_vector):#input single facet[[],[],[]...], normal_vector []
    facet = clockwise_sort(facet, normal_vector)
    n = len(facet)
    area = 0.0
    j =n-1
    for i in range(n):
        area += (facet[j][0] + facet[i][0])*(facet[j][1] - facet[i][1])
        j = i
    return abs(area/2.0)
#test
#b3=polyarea([[1,0,0],[0,1,0],[1,0,2],[0,1,4]],[1,1,0])
#b3=polyarea([[0,0,0],[0,1,0],[0,0,1]],[1,0,0])
#b3=polyarea([[0.5,1,1],[0,0.5,1],[0,1,0.5]],[-1,1,1])
#print(b3)
#print(2*math.sqrt(2)*1.5)
def arrange_particle(P_index, pair):#P_index reference particle indice, pair = [P_index,other index]
    refer_P_index = pair.tolist().index(P_index)
    return pair[refer_P_index], pair[1-refer_P_index]#return value in pair
def NN_and_weightedRidge(P_index, Points_ridge, Vertices_ridge, Points, Vertices):#input P_index:single particle index, Points_ridge, Vertices_ridge, points, Vertices: array whole data
    norm_vector = []
    area = []
    NN = []
    NN_index = []
    for index, pair in enumerate(Points_ridge):
        if P_index in pair:
            ref_index, other_index = arrange_particle(P_index, pair)
            norm_vector_temp = Points[other_index] - Points[ref_index]
            temp_norm_vector = norm_vector_temp / np.linalg.norm(norm_vector_temp)
            facet = index2position(Vertices_ridge[index], Vertices)#facet is vertice coordinates of one facet of Voronoi cell [[x,y,z],[]...]
            temp_area = polyarea(facet, norm_vector_temp)
            norm_vector.append(temp_norm_vector.tolist())
            area.append(temp_area)
            NN.append(Points[other_index].tolist())#NN in particle position coordinate
            NN_index.append(other_index)#NN index [index1,index2...]
    s = sum(area)
    weighted_area = [item / s for item in area]
    return NN, norm_vector, weighted_area, NN_index #NN: list [[x,y,z],[],[]..], norm_vector:list [[x,y,z],[],[]...], weighted_area:list [a,b,c...]




def bond_angles_fun(ref_particle, particle_j, particle_k):
    #input as lists
    v1 = np.array(particle_j) - np.array(ref_particle)
    v2 = np.array(particle_k) - np.array(ref_particle)
    bond_angle = np.dot(np.squeeze(v1), np.squeeze(v2))/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return bond_angle
def feature_Distance_BondAngle(ref_particle, NN):#ref_position coordinate list [x,y,z], NN from NN_and_weightedRidge
    combination = list(itertools.combinations(NN, 2))
    Distance = []
    Bond_angle = []
    for item in combination:
        v = np.array(item[0])-np.array(item[1])
        Distance.append(np.linalg.norm(v))
        temp_bond_angle = bond_angles_fun(ref_particle, item[0], item[1])
        Bond_angle.append(temp_bond_angle)
    h_dis, bin_dis = np.histogram(Distance, 12)
    h_angle, bin_angle = np.histogram(Bond_angle, 8)
    return h_dis.tolist(), h_angle.tolist() #return list
#test
#nn=[[1,1,1],[2,2,2],[3,3,3]]
#hd ,ha = feature_Distance_BondAngle([5,4,3], nn)




def geo_angle_fun(norm_vector):#norm_vector:list [[x,y,z],[],[]...]
    geo_angle = []
    for item in norm_vector:
        phi = math.atan2(item[1], item[0])#azimuth
        if phi < 0:
            phi = phi + 2 * math.pi
        r = np.linalg.norm(item[0:2])
        theta = math.atan2(r, item[2])#polar
        geo_angle.append([phi, theta])
    return geo_angle# geo_angle list [[phi,theta],[],[]..]
#test
#a=[[1,2,3],[1,1,1],[0,1,0]]
#g = geo_angle_fun(a)
def spherical_harmonic_fun(geo_angle, weighted_area):# for one ref_particle list
    if len(geo_angle)!=len(weighted_area):
        print('geo_angle.length != area.length')
    spherical_harmonics_l4 = []
    spherical_harmonics_l6 = []
    for facet_index in range(len(geo_angle)):
        Y_4_neg4 = weighted_area[facet_index] * scipy.special.sph_harm(-4, 4, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_4_neg3 = weighted_area[facet_index] * scipy.special.sph_harm(-3, 4, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_4_neg2 = weighted_area[facet_index] * scipy.special.sph_harm(-2, 4, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_4_neg1 = weighted_area[facet_index] * scipy.special.sph_harm(-1, 4, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_4_0 = weighted_area[facet_index] * scipy.special.sph_harm(0, 4, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_4_1 = weighted_area[facet_index] * scipy.special.sph_harm(1, 4, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_4_2 = weighted_area[facet_index] * scipy.special.sph_harm(2, 4, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_4_3 = weighted_area[facet_index] * scipy.special.sph_harm(3, 4, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_4_4 = weighted_area[facet_index] * scipy.special.sph_harm(4, 4, geo_angle[facet_index][0], geo_angle[facet_index][1])
        temp_l4 = [Y_4_neg4, Y_4_neg3, Y_4_neg2, Y_4_neg1, Y_4_0, Y_4_1, Y_4_2, Y_4_3, Y_4_4]
        spherical_harmonics_l4.append(temp_l4)

        Y_6_neg6 = weighted_area[facet_index] * scipy.special.sph_harm(-6, 6, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_6_neg5 = weighted_area[facet_index] * scipy.special.sph_harm(-5, 6, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_6_neg4 = weighted_area[facet_index] * scipy.special.sph_harm(-4, 6, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_6_neg3 = weighted_area[facet_index] * scipy.special.sph_harm(-3, 6, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_6_neg2 = weighted_area[facet_index] * scipy.special.sph_harm(-2, 6, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_6_neg1 = weighted_area[facet_index] * scipy.special.sph_harm(-1, 6, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_6_0 = weighted_area[facet_index] * scipy.special.sph_harm(0, 6, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_6_1 = weighted_area[facet_index] * scipy.special.sph_harm(1, 6, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_6_2 = weighted_area[facet_index] * scipy.special.sph_harm(2, 6, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_6_3 = weighted_area[facet_index] * scipy.special.sph_harm(3, 6, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_6_4 = weighted_area[facet_index] * scipy.special.sph_harm(4, 6, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_6_5 = weighted_area[facet_index] * scipy.special.sph_harm(5, 6, geo_angle[facet_index][0], geo_angle[facet_index][1])
        Y_6_6 = weighted_area[facet_index] * scipy.special.sph_harm(6, 6, geo_angle[facet_index][0], geo_angle[facet_index][1])
        temp_l6 = [Y_6_neg6, Y_6_neg5, Y_6_neg4, Y_6_neg3, Y_6_neg2, Y_6_neg1, Y_6_0, Y_6_1, Y_6_2, Y_6_3, Y_6_4, Y_6_5, Y_6_6]
        spherical_harmonics_l6.append(temp_l6)
    q_l4_m = np.zeros((1, 9))
    for item in spherical_harmonics_l4:
        q_l4_m = q_l4_m + item
    q_l6_m = np.zeros((1, 13))
    for item in spherical_harmonics_l6:
        q_l6_m = q_l6_m + item
    return np.squeeze(q_l4_m), np.squeeze(q_l6_m)#ndarray
#test
#spherical_harmonic_fun([[0.1,0.1],[0.4,0.2],[1,1.2]], [1,1,1])
def feature_ql(q_l4_m, q_l6_m):#input ndarray
    s_l4 = pow(abs(q_l4_m), 2)
    q_l4 = math.sqrt(np.sum(s_l4) * 4 * math.pi / 9)
    s_l6 = pow(abs(q_l6_m), 2)
    q_l6 = math.sqrt(np.sum(s_l6) * 4 * math.pi / 13)
    decimal_place = 3
    q_l4 = np.around(q_l4, decimal_place)
    q_l6 = np.around(q_l6, decimal_place)
    return q_l4, q_l6#float
#test
#norm=[[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1],[-1,1,-1],[-1,-1,1],[1,-1,-1],[-1,-1,-1]]
#g=geo_angle_fun(norm)
#print(np.array(g)/math.pi)
#w=[0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]
#q4,q6 = spherical_harmonic_fun(g,w)
#ql4,ql6 = feature_ql(q4,q6)
#print('ql4+ql6',ql4,ql6)
def sum3(array):#input list
    combination = list(itertools.combinations(array, 3))
    output = []
    for item in combination:
        if item[0] + item[1] + item[2] == 0:
            output.append(item)
    return output #[(),()...]
#test
#sum3([-4,-3,-2,-1,0,1,2,3,4])
#sum3([-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
def read_wigner(wigner_3j):
    a = eval(str(wigner_3j.args[0]))
    b = eval('math.'+str(wigner_3j.args[1]))
    c = a * b
    return c
#test
#a = wigner_3j(4,4,4,0,-1,1)
#c=read_wigner(a)
#print(-3/2002)
#print(math.sqrt(2002))
def feature_wl(q_l4_m, q_l6_m, q_l4, q_l6):
    c4 = sum3([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    c6 = sum3([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
    w_l4 = 0
    w_l6 = 0
    for item in c4:
        wigner_l4 = read_wigner(wigner_3j(4, 4, 4, item[0], item[1], item[2]))
        w_l4 = w_l4 + wigner_l4 * q_l4_m[item[0]+4] * q_l4_m[item[1]+4] * q_l4_m[item[2]+4] / pow(q_l4, 3)
    w_l4 = w_l4 * 6
    if np.around(w_l4.imag, 5) != 0:
        print('w_l4 is complex')
    else:
        w_l4 = w_l4.real

    for item in c6:
        wigner_l6 = read_wigner(wigner_3j(6, 6, 6, item[0], item[1], item[2]))
        w_l6 = w_l6 + wigner_l6 * q_l6_m[item[0]+6] * q_l6_m[item[1]+6] * q_l6_m[item[2]+6] / pow(q_l6, 3)
    w_l6 = w_l6 * 6
    if np.around(w_l6.imag, 5) != 0:
        print('w_l6 is complex')
    else:
        w_l6 = w_l6.real

    decimal = 4
    return np.around(w_l4, decimal), np.around(w_l6, decimal) #float




def built_Minkowski_tensor(norm_vector, weighted_area):
    n = len(norm_vector)
    norm_vector = np.array(norm_vector)
    minkowski_tensor = np.zeros((3, 3, 3, 3))
    for i in range(n):
        tensor = norm_vector[i]
        for j in range(3):
            tensor = np.tensordot(tensor, norm_vector[i], axes=0)
        minkowski_tensor = minkowski_tensor + weighted_area[i] * tensor
    return minkowski_tensor #ndarray
#test
#a=[[1,2,3],[1,2,3],[1,2,3]]
#m,n=built_Minkowski_tensor(a, [1,1,1])
#print(m.shape)
def voigt_notation(minkowski_tensor):#input ndarray
    voigt_matrix = np.zeros((6, 6))
    voigt_index = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    for i in range(6):
        for j in range(6):
            voigt_matrix[i][j] = minkowski_tensor[voigt_index[i][0]][voigt_index[i][1]][voigt_index[j][0]][voigt_index[j][1]]
    eigenvalue, eigenvector = np.linalg.eig(voigt_matrix)
    decimal_place = 5
    eigenvalue = np.around(eigenvalue, decimal_place)
    eigenvalue.sort()
    return eigenvalue.tolist()#output list
#test
#a = [1,2,3]
#b=np.array(a)
#c=np.tensordot(a,a, axes = 0)
#d=np.tensordot(c,a,axes = 0)
#e=np.tensordot(d,a,axes = 0)
#print(e[0][1][2][1])
#print(e[0][2][1][1])
#g,v = voigt_notation(e)
#z1,z2 = np.linalg.eig(v)
#z,zz=scipy.linalg.eig(v)
#print(z1)
#print(z)


def features_extract(P_index, Points_ridge, Vertices_ridge, Points, Vertices):# for 1 particle
    NN, norm_vector, weighted_area, cache_NN_index = NN_and_weightedRidge(P_index, Points_ridge, Vertices_ridge, Points, Vertices)
    ref_particle_coordinate = Points[P_index]
    #feature 1&2
    h_dis, h_angle = feature_Distance_BondAngle(ref_particle_coordinate, NN)
    #feature 3
    geo_angle = geo_angle_fun(norm_vector)
    q_l4_m, q_l6_m = spherical_harmonic_fun(geo_angle, weighted_area)
    q_l4, q_l6 = feature_ql(q_l4_m, q_l6_m)
    cache_q_l6_m = q_l6_m.tolist()

    w_l4, w_l6 = feature_wl(q_l4_m, q_l6_m, q_l4, q_l6)
    #feature 4
    minkowski_tensor = built_Minkowski_tensor(norm_vector, weighted_area)
    minkowski_eig = voigt_notation(minkowski_tensor)
    #feature 5
    NN_count = len(NN)
    return h_dis, h_angle, [q_l4, q_l6, w_l4, w_l6], minkowski_eig, [NN_count], cache_NN_index, cache_q_l6_m#output list
def output_features(h_dis, h_angle, q_w, minkowski_eig, NN_count):
    h_dis = [float(item) for item in h_dis]
    h_angle = [float(item) for item in h_angle]
    NN_count = [float(NN_count[0])]
    features = h_dis + h_angle + q_w + minkowski_eig + NN_count
    return features







