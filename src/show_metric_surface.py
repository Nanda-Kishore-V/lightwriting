import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import operator

#x - Theta, y - Distance, z - cost
D_MAX = 100
THETA_END = 360
COST_SCALING = 10
LINE_THETA_ZERO = ((0, 0, 2 * COST_SCALING), (0, D_MAX, 1.5 * COST_SCALING))
LINE_THETA_END = ((THETA_END, 0, 1 * COST_SCALING), (THETA_END, D_MAX, 0 * COST_SCALING))

def cost(theta, d):
    k = 20 * 1 / (D_MAX - 0.99 * d)
    pt_zero = find_section_point(d/D_MAX, LINE_THETA_ZERO)
    pt_end = find_section_point(d/D_MAX, LINE_THETA_END)
    print 'pt_zero_end', pt_zero, pt_end
    a, b = find_coefficients(pt_zero, pt_end, k)
    return k/(theta-a) + b

def find_section_point(ratio, end_points):
    m = ratio
    n = 1 - ratio
    print 'ratio', ratio
    # exit()
    section_point = tuple([m * c2 + n * c1 for c1, c2 in zip(*end_points)])
    # difference = tuple(map(operator.sub, end_points[0], end_points[1]))
    # print 'end pts', end_points
    # print 'diff', difference
    # return tuple(map(operator.add, end_points[0], tuple([ratio * d for d in difference])))
    # print 'section point', section_point
    return section_point

def find_coefficients(pt1, pt2, k):
    # print pt1, pt2
    x2, _, y2 = pt2
    x1, _, y1 = pt1
    b = -1*(y1 + y2)
    a = 1
    c = y1*y2 + (k*(y1 - y2))/(x1 -x2)
    y_offset = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    # print y_offset
    # print y1 - y2
    # exit()
    x_offset = (y_offset *(x2 - x1) + x1*y1 - x2*y2)/(y1 - y2)
    # print x_offset, y_offset
    return x_offset, y_offset

def main():

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    # theta = np.arange(0, 360, 0.05)
    # dist = np.arange(0, 3, 0.05)
    theta = np.arange(0, 90, 1)
    dist = np.arange(0, D_MAX, 0.1)
    Theta, Dist = np.meshgrid(theta, dist)
    Cost = np.array([cost(t, d) for t, d in zip(np.ravel(Theta), np.ravel(Dist))])
    Cost = Cost.reshape(Theta.shape)

    ax.plot_surface(Theta, Dist, Cost)

    ax.set_xlabel('Theta')
    ax.set_ylabel('Distance')
    ax.set_zlabel('Cost')

    plt.show()

    # plt.figure(2)
    # plt.plot(X, Z)
    # plt.xlabel("Theta")
    # plt.ylabel("Cost")
    # plt.show()
    #
    # plt.figure(3)
    # plt.plot(Y, Z)
    # plt.xlabel("Distance")
    # plt.ylabel("Cost")
    # plt.show()

if __name__ == '__main__':
    main()
