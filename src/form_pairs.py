from __future__ import division

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import operator
import math

from constants import HOME

D_MAX = 100
THETA_END = 360
COST_SCALING = 10
LINE_THETA_ZERO = ((0, 0, 2 * COST_SCALING), (0, D_MAX, 1.5 * COST_SCALING))
LINE_THETA_END = ((THETA_END, 0, 1 * COST_SCALING), (THETA_END, D_MAX, 0 * COST_SCALING))

def metric(start, end):
    point_start = start[1]
    vector_start = start[2]
    point_end = end[1]
    vector_end = end[2]
    d = distance(point_start, point_end)
    theta = angle_between_vectors(vector_start, vector_end)
    return cost(theta, d)

def cost(theta, d):
    print 'theta, d', theta, d
    k = 200
    point_zero = find_section_point(d/D_MAX, LINE_THETA_ZERO)
    point_end = find_section_point(d/D_MAX, LINE_THETA_END)
    a, b = find_coefficients(point_zero, point_end, k)
    return k/(theta-a) + b

def find_section_point(ratio, end_points):
    m = ratio
    n = 1 - ratio
    section_point = tuple([m * c2 + n * c1 for c1, c2 in zip(*end_points)])
    return section_point

def find_coefficients(point1, point2, k):
    x2, _, y2 = point2
    x1, _, y1 = point1
    b = -1 * (y1 + y2)
    a = 1
    c = y1 * y2 + (k * (y1 - y2)) / (x1 - x2)
    y_offset = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    x_offset = (y_offset * (x2 - x1) + x1 * y1 - x2 * y2) / (y1 - y2)
    return x_offset, y_offset

def distance(point1, point2):
    return np.linalg.norm(tuple(map(operator.sub, point2, point1)))

def unit_vector(v):
    return v / np.linalg.norm(v)

def angle_between_vectors(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def vector(point_start, point_end):
    if len(point_start) != len(point_end):
        return None
    return tuple(map(operator.sub, point_end, point_start))

def main():
    f = open(HOME + "data/tangents.csv", "rb")
    data = np.genfromtxt(f, dtype= float, delimiter=',')
    data = np.delete(data, (0), axis = 0)

    #index, point, vector
    points = []
    for index, d in enumerate(data):
        points.append([index, tuple(d[1:4]), tuple(d[4:7])])
        points.append([index, tuple(d[7:10]), tuple(d[10:13])])

    print 'point # : [segment id, point, tangent at point]'
    for i, p in enumerate(points):
        print i, ':', p

    # form pairs now
    n_required_segments = 2
    point_initial = points[1]
    point_initial[2] = tuple([-1 * x for x in point_initial[2]])
    m = -1
    for i, p in enumerate(points):
        if i == 1:
            print 'same point'
            print 'd, theta'
            continue
        temp_m = metric(point_initial, p)
        print temp_m
        m = max(m, temp_m)

if __name__ == "__main__":
    main()
