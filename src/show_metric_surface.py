from __future__ import division

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import operator

#x - Theta, y - Distance, z - cost
COST_SCALING = 10
DISTANCE_SCALING = 1

THETA_END = 360
D_MAX = 100 * DISTANCE_SCALING
LINE_THETA_ZERO = ((0, 0, 2 * COST_SCALING), (0, D_MAX, 1.5 * COST_SCALING))
LINE_THETA_END = ((THETA_END, 0, 1 * COST_SCALING), (THETA_END, D_MAX, 0 * COST_SCALING))

def cost(theta, d):
    k = 200
    pt_zero = find_section_point(d/D_MAX, LINE_THETA_ZERO)
    pt_end = find_section_point(d/D_MAX, LINE_THETA_END)
    a, b = find_coefficients(pt_zero, pt_end, k)
    return k/(theta-a) + b

def find_section_point(ratio, end_points):
    m = ratio
    n = 1 - ratio
    section_point = tuple([m * c2 + n * c1 for c1, c2 in zip(*end_points)])
    return section_point

def find_coefficients(pt1, pt2, k):
    x2, _, y2 = pt2
    x1, _, y1 = pt1
    b = -1*(y1 + y2)
    a = 1
    c = y1*y2 + (k*(y1 - y2))/(x1 -x2)
    y_offset = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    x_offset = (y_offset *(x2 - x1) + x1*y1 - x2*y2)/(y1 - y2)
    return x_offset, y_offset

def main():
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    theta = np.linspace(0, 20, 100)
    dist = np.linspace(0, D_MAX, 100)
    Theta, Dist = np.meshgrid(theta, dist)
    Cost = np.array([cost(t, d) for t, d in zip(np.ravel(Theta), np.ravel(Dist))])
    Cost = Cost.reshape(Theta.shape)

    ax.plot_surface(Theta, Dist, Cost)
    points = [(3.80921574603, 25.2512375934), (3.79398838838, 0.999999999999)]
    for p in points:
        print cost(p[0], p[1] * DISTANCE_SCALING)
    theta_list, dist_list = zip(*points)
    ax.scatter(theta_list, dist_list, [cost(t, d) for t, d in points], c = ['r' for p in points])

    ax.set_xlabel('Theta')
    ax.set_ylabel('Distance')
    ax.set_zlabel('Cost')

    plt.show()

    # plt.figure(2)
    # plt.plot(Y, Z)
    # plt.xlabel("Theta")
    # plt.ylabel("Cost")
    # plt.show()

    # plt.figure(3)
    # plt.plot(Y, Z)
    # plt.xlabel("Distance")
    # plt.ylabel("Cost")
    # plt.show()


if __name__ == '__main__':
    main()
