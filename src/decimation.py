import math
import numpy as np
import operator
import cv2
import math
import cmath

from constants import *
from debug import *

def distance_perpendicular(line_points, point):
    x0, y0 = point
    x1, y1 = line_points[0]
    x2, y2 = line_points[1]
    distance = abs((y2 - y1) * x0  - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return distance

def decimate(points, reduced_points_min_dist=25, epsilon=1, epsilon_increment=0.5):
    if VERBOSE_TEXT: print('Current epsilon is: ', epsilon)
    start = 0
    end = len(points) - 1

    end_point_index_pairs = [(start, end)]
    points_reduced = [None] * len(points)
    points_reduced[start] = points[start]
    points_reduced[end] = points[end]
    
    point_removed = False
    if VERBOSE_TEXT: print 'epsilon', epsilon
    while end_point_index_pairs:
        start, end = end_point_index_pairs.pop(0)
        if start + 1 == end:
            continue

        line = (points[start], points[end])
        errors = [distance_perpendicular(line, p) for i, p in enumerate(points) if start < i < end]

        error_max = max(errors)
        if error_max < epsilon:
            continue

        index = start + errors.index(error_max) + 1
        points_reduced[index] = points[index]
        end_point_index_pairs.append((start, index))
        end_point_index_pairs.append((index, end))
        point_removed = True

    points_reduced = [p for p in points_reduced if p is not None]
    consecutive_distances = []
    for i, p in enumerate(points_reduced[1:]):
        if VERBOSE_TEXT: print p, points_reduced[i], np.linalg.norm((p[0] - points[i][0], p[1] - points[i][1]))
        consecutive_distances.append(np.linalg.norm(tuple(map(operator.sub, p, points_reduced[i]))))
    if VERBOSE_TEXT: print 'min conseq dist', min(consecutive_distances)
    if min(consecutive_distances) < reduced_points_min_dist:
        epsilon += epsilon_increment 
        if VERBOSE_TEXT: print('We need to keep decimating. New epsilon is {0}'.format(epsilon))
        if point_removed:
            if VERBOSE_IMAGE: show_points_as_image(points_reduced)
        return decimate(points, epsilon=epsilon)

    if len(points_reduced) == 2:
        x0, y0 = points_reduced[0]
        x1, y1 = points_reduced[1]
        print points_reduced
        midpoint = ((x0 + x1) / 2, (y0 + y1) / 2)
        points_reduced.insert(1, midpoint) 
        print points_reduced
        if VERBOSE_TEXT: print('len', len(points_reduced))

    return points_reduced

if __name__ == '__main__':
    points = []
    with open(HOME + 'data/segments/segment' + '0', 'r') as f:
        lines = f.readlines()
        for line in lines:
            point = tuple(int(x) for x in line.strip('()\n').split(','))
            points.append(point)
    if VERBOSE_TEXT: print points

    if len(points) <= 2:
        if VERBOSE_TEXT: print('Segment has less than 2 points. No decimation needed.')
        exit()

    points_reduced = decimate(points)
    if VERBOSE_TEXT: print('Length of points_reduced: ', len(points_reduced))
    if VERBOSE_TEXT: print('Reduced points: ', points_reduced)

    image = np.zeros((500, 500), np.uint8)
    for p in points_reduced:
        image[p] = WHITE
    if VERBOSE_IMAGE: show_and_destory(image)
