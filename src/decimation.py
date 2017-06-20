import math
import numpy as np
import operator
import cv2
import math
import cmath

from constants import *
from debug import *

from geometry import Point

def decimate(points, reduced_points_min_dist=25, epsilon=1, epsilon_increment=0.5):
    '''Returns a list of Point by removing intermediate Points such that shape of curve is maintained
        and distance between 2 consecutive Points is more than reduced_points_min_dist'''
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

        line_end_points = (points[start], points[end])
        errors = [Point.distance_perpendicular_from_line(p, line_end_points) for i, p in enumerate(points) if start < i < end]

        error_max = max(errors)
        if error_max < epsilon:
            continue

        index = start + errors.index(error_max) + 1
        points_reduced[index] = points[index]
        end_point_index_pairs.append((start, index))
        end_point_index_pairs.append((index, end))
        point_removed = True

    points_reduced = [p for p in points_reduced if p is not None]
    consecutive_distances = [Point.distance(p, points_reduced[i]) for i, p in enumerate(points_reduced[1:])]
    if min(consecutive_distances) < reduced_points_min_dist:
        epsilon += epsilon_increment
        if VERBOSE_TEXT: print('We need to keep decimating. New epsilon is {}'.format(epsilon))
        if point_removed:
            if VERBOSE_IMAGE: show_points_as_image(points_reduced)
        return decimate(points, epsilon=epsilon)

    return points_reduced

if __name__ == '__main__':
    points = []
    raise IOError('file is no longer generated')
    with open(HOME + 'data/segments/segment' + '0', 'r') as f:
        lines = f.readlines()
        for line in lines:
            point = tuple(int(x) for x in line.strip('()\n').split(','))
            print point
            points.append(point)
    if VERBOSE_TEXT: print points

    if len(points) <= 2:
        if VERBOSE_TEXT: print('Segment has less than 2 points. No decimation needed.')

    points_reduced = decimate(points)
    if VERBOSE_TEXT: print('Length of points_reduced: ', len(points_reduced))
    if VERBOSE_TEXT: print('Reduced points: ', points_reduced)

    image = np.zeros((500, 500), np.uint8)
    for p in points_reduced:
        image[p] = WHITE
    if VERBOSE_IMAGE: show_and_destory(image)
