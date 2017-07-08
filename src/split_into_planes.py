from __future__ import division, print_function

import numpy as np
import csv
from pypoly import Polynomial
import operator
import json
import matplotlib.pyplot as plt

from sets import Set
from constants_env import HOME
from constants_debug import VERBOSE_TEXT
from constants_crazyswarm import (
    HEIGHT_OFFSET,
    X_OFFSET,
    POST_SCALING_FACTOR,
    COLLISION_DIST,
    CAMERA_DISTANCE,
)
from geometry import (
    Segment,
    Point,
)

ROS_WS = HOME + '../crazyswarm/ros_ws/src/crazyswarm/'

def second_largest(numbers):
    first, second = None, None
    for n in numbers:
        if n > first:
            first, second = n, first
        elif first > n > second:
            second = n
    return second

def distance(point1, point2):
    return np.linalg.norm(map(operator.sub, point2, point1))

def return_location(segment, time):
    shift_times = [segment[i][1] for i in range(len(segment))]
    total_time = sum(shift_times)
    if time > total_time:
        Px = Polynomial(*segment[-1][2:10])
        Py = Polynomial(*segment[-1][10:18])
        Pz = Polynomial(*segment[-1][18:26])
        return Px(total_time), Py(total_time), Pz(total_time)
    s = 0
    for index, t in enumerate(shift_times):
        s += t
        if s >= time:
            s -= t
            break
    time -= s
    Px = Polynomial(*segment[index][2:10])
    Py = Polynomial(*segment[index][10:18])
    Pz = Polynomial(*segment[index][18:26])
    return Px(time), Py(time), Pz(time)

def promising(node, color, intersecting_pairs, color_of_segments):
    for neighbor in range(len(color_of_segments)):
        color_neigh = -1
        if (node < neighbor and (node, neighbor) in intersecting_pairs) or (node > neighbor and (neighbor, node) in intersecting_pairs):
            color_neigh = color_of_segments[neighbor]
        if color_neigh == color:
            return False
    return True

def get_color(node, intersecting_pairs, color_of_segments):
    for color in range(len(color_of_segments)):
        if promising(node, color, intersecting_pairs, color_of_segments):
            return color
    if VERBOSE_TEXT: print("Something is wrong with the number of segments.")
    return None

def main():
    f = open(HOME + "data/temp.csv", "r")
    csv_reader = csv.reader(f)
    first_line = next(csv_reader)
    matrix = np.loadtxt(f, delimiter=",", skiprows=0, ndmin=2)
    matrix = np.split(matrix, np.where(np.diff(matrix[:,0]))[0]+1)

    times = [sum(segment[i][1] for i in range(len(segment))) for segment in matrix]
    dt = min(times)/500.0
    t = 0
    limit = max(times)

    intersecting_pairs = Set([])
    locations = []
    for segment in matrix:
        locations.append([0, segment[0][10], 0])

    combinations = [(i,j) for i in range(len(locations)) for j in range(len(locations)) if i < j]
    distances = [distance(locations[i], locations[j]) for i, j in combinations]
    is_intersecting = map(lambda x : x < COLLISION_DIST, distances)
    for index, elem in enumerate(is_intersecting):
        if elem:
            intersecting_pairs.add(combinations[index])

    while t <= limit:
        locations = []
        for segment in matrix:
            locations.append(return_location(segment, t))

        combinations = [(i,j) for i in range(len(locations)) for j in range(len(locations)) if i < j]
        distances = [distance(locations[i], locations[j]) for i, j in combinations]
        is_intersecting = map(lambda x : x < COLLISION_DIST, distances)
        for index, elem in enumerate(is_intersecting):
            if elem:
                intersecting_pairs.add(combinations[index])
        t += dt
    if VERBOSE_TEXT: print(intersecting_pairs)
    color_of_segments = {}
    for i in range(len(matrix)):
        color_of_segments[i] = -1
    for node in range(len(matrix)):
        color_of_segments[node] = get_color(node, intersecting_pairs, color_of_segments)
    if VERBOSE_TEXT: print(color_of_segments)
    print("Number of planes: {}".format(max(color_of_segments.values())+1))

    initialPositions = []
    heights = []
    for segment_num, segment in enumerate(matrix):
        # with open(ROS_WS + 'scripts/traj/plane{0}_quad{1}.csv'.format(color_of_segments[segment_num], segment_num), "w") as filename:
        with open(ROS_WS + 'scripts/traj/trajectory{0}.csv'.format(segment_num + 1), 'w') as filename:
            writer = csv.writer(filename)
            writer.writerow(np.concatenate([['duration'],[axis + '^' + str(i) for axis in ['x', 'y', 'z', 'yaw'] for i in range(8)]]))
            initialPositions.append((color_of_segments[segment_num]*-1*X_OFFSET, segment[0][10], 0))
            heights.append(segment[0][2])
            for piece in segment:
                # temp = piece[10:18].copy()
                piece[18:26] = piece[2:10].copy()
                # piece[10:18] = temp
                piece[2:10] = [0 for _ in range(8)]
                writer.writerow(np.concatenate([[piece[1]], [(i) for i in piece[2:]]]))

    if VERBOSE_TEXT: print("InitialPositions: " + str(initialPositions))
    if VERBOSE_TEXT: print("Heights: " + str(heights))

    channel = [100, 110, 120]
    with open(ROS_WS + "launch/crazyflies.yaml", "w") as filename:
        filename.write("crazyflies:\n")
        for segment_num in range(len(matrix)):
            filename.write(' - id: ' + str(segment_num + 1) + '\n')
            filename.write('   channel: {0}\n'.format(channel[segment_num%3]))
            filename.write('   initialPosition: [{0}, {1}, {2}]\n'.format(*initialPositions[segment_num]))

    with open(ROS_WS + "launch/heights.yaml", "w") as filename:
        for segment_num in range(len(matrix)):
            filename.write('{0}: {1}\n'.format(segment_num + 1, heights[segment_num]))

if __name__ == "__main__":
    main()
