import numpy as np
import math
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

def crystal_lattice_bcc(size, a=1, noise_level=None):
    # L=crystal size normalized to d_0, a=lattice const

    np.random.seed(1)
    d_0 = math.sqrt(3) / 2 * a
    L = math.ceil(size * d_0 / a)
    if noise_level is None:
        deviation = 0
        turn_on_noise = 0
    else:
        deviation = noise_level * d_0 / 100
        turn_on_noise = 1

    crystal_lattice = []
    # edge particle (order in x,y,z)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                particle = [k * a + (turn_on_noise * np.random.normal(0, deviation)),
                            j * a + (turn_on_noise * np.random.normal(0, deviation)),
                            i * a + (turn_on_noise * np.random.normal(0, deviation))]
                crystal_lattice.append(particle)
                particle2 = [(k + 0.5) * a + (turn_on_noise * np.random.normal(0, deviation)),
                             (j + 0.5) * a + (turn_on_noise * np.random.normal(0, deviation)),
                             (i + 0.5) * a + (turn_on_noise * np.random.normal(0, deviation))]
                crystal_lattice.append(particle2)

    print('total_particle_number=', len(crystal_lattice))

    #x, y, z = zip(*crystal_lattice)
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.scatter(x, y, z, color='r', marker='o')
    #plt.show()

    return crystal_lattice

def crystal_lattice_fcc(size, a=1, noise_level=None):
    np.random.seed(1)
    d_0 = math.sqrt(2) / 2 * a
    L = math.ceil(size * d_0 / a)

    if noise_level is None:
        deviation = 0
        turn_on_noise = 0
    else:
        deviation = noise_level * d_0 / 100
        turn_on_noise = 1

    crystal_lattice = []
    for i in range(L):
        for j in range(L):
            for k in range(L):
                particle = [k * a + (turn_on_noise * np.random.normal(0, deviation)),
                            j * a + (turn_on_noise * np.random.normal(0, deviation)),
                            i * a + (turn_on_noise * np.random.normal(0, deviation))]
                crystal_lattice.append(particle)
                particle2 = [(k + 0.5) * a + (turn_on_noise * np.random.normal(0, deviation)),
                             (j + 0.5) * a + (turn_on_noise * np.random.normal(0, deviation)),
                             i * a + (turn_on_noise * np.random.normal(0, deviation))]
                crystal_lattice.append(particle2)
                particle3 = [(k + 0.5) * a + (turn_on_noise * np.random.normal(0, deviation)),
                             j * a + (turn_on_noise * np.random.normal(0, deviation)),
                             (i + 0.5) * a + (turn_on_noise * np.random.normal(0, deviation))]
                crystal_lattice.append(particle3)
                particle4 = [k * a + (turn_on_noise * np.random.normal(0, deviation)),
                             (j + 0.5) * a + (turn_on_noise * np.random.normal(0, deviation)),
                             (i + 0.5) * a + (turn_on_noise * np.random.normal(0, deviation))]
                crystal_lattice.append(particle4)

    print('total_particle_number=', len(crystal_lattice))

    #x, y, z = zip(*crystal_lattice)
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.scatter(x, y, z, color='r', marker='o')
    #plt.show()

    return crystal_lattice

def crystal_lattice_hcp(size, a=1, noise_level=None):
    np.random.seed(1)
    d_0 = a
    h = math.sqrt(6) * 2 / 3 * a

    L_x = math.ceil(size * d_0 / a)
    L_y = math.ceil(size * d_0 / (math.sqrt(3) * a))
    L_z = math.ceil(size * d_0 / (math.sqrt(6) * 2 / 3 * a))

    x_unit = a
    y_unit = math.sqrt(3) * a
    z_unit = math.sqrt(6) * 2 / 3 * a

    if noise_level is None:
        deviation = 0
        turn_on_noise = 0
    else:
        deviation = noise_level * d_0 / 100
        turn_on_noise = 1

    crystal_lattice = []
    for i in range(L_z):
        for j in range(L_y):
            for k in range(L_x):
                particle = [k * x_unit + (turn_on_noise * np.random.normal(0, deviation)),
                            j * y_unit + (turn_on_noise * np.random.normal(0, deviation)),
                            i * z_unit + (turn_on_noise * np.random.normal(0, deviation))]
                crystal_lattice.append(particle)
                particle2 = [(k + 0.5) * x_unit + (turn_on_noise * np.random.normal(0, deviation)),
                             (j + (1 / 6)) * y_unit + (turn_on_noise * np.random.normal(0, deviation)),
                             (i + 0.5) * z_unit + (turn_on_noise * np.random.normal(0, deviation))]
                crystal_lattice.append(particle2)

    for i in range(L_z):
        for j in range(L_y):
            for k in range(L_x):
                particle = [(k + 0.5) * x_unit + (turn_on_noise * np.random.normal(0, deviation)),
                            (j + 0.5) * y_unit + (turn_on_noise * np.random.normal(0, deviation)),
                            i * z_unit + (turn_on_noise * np.random.normal(0, deviation))]
                crystal_lattice.append(particle)
                particle2 = [(k + 0.5 + 0.5) * x_unit + (turn_on_noise * np.random.normal(0, deviation)),
                             (j + 0.5 + (1 / 6)) * y_unit + (turn_on_noise * np.random.normal(0, deviation)),
                             (i + 0.5) * z_unit + (turn_on_noise * np.random.normal(0, deviation))]
                crystal_lattice.append(particle2)

    print('total_particle_number=', len(crystal_lattice))

    #x, y, z = zip(*crystal_lattice)
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.scatter(x, y, z, color='r', marker='o')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #plt.show()

    return crystal_lattice


#c = crystal_lattice_bcc(3)
