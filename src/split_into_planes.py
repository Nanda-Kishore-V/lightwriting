from __future__ import division, print_function

import numpy as np
import csv
from pypoly import Polynomial
import operator
import json
import yaml
import matplotlib.pyplot as plt
import os
import math

from sets import Set
from constants_env import (
    HOME,
    ROS_WS,
)
from constants_debug import VERBOSE_TEXT
from constants_crazyswarm import (
    HEIGHT_OFFSET,
    X_OFFSET,
    HOVER_OFFSET,
    POST_SCALING_FACTOR,
    COLLISION_DIST,
    CAMERA_DISTANCE,
    HOVER_PAUSE_TIME,
    TAKE_OFF_TIME,
)
from geometry import (
    Segment,
    Point,
)

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

    with open(HOME + 'data/all49.yaml') as infile:
        crazyflies = yaml.load(infile)['crazyflies']
    available_ids = np.loadtxt(HOME + 'data/quads_list.txt', delimiter=' ')
    ids = available_ids[:len(matrix)]
    print(ids)
    selected_crazyflies = map(lambda x: crazyflies[int(x) - 1], ids)
    
    with open(ROS_WS + 'launch/crazyflies.yaml', 'w') as outfile:
        yaml.dump({'crazyflies': selected_crazyflies}, outfile)

    ground_positions = []
    for cf in crazyflies:
        if cf['id'] in ids:
            ground_positions.append(np.concatenate([cf['initialPosition'], [cf['id']]]))
    # print("Start Positions before sorting:")
    # print(ground_positions)

    ground_positions.sort(key=tuple)
    # print("Start Positions after sorting:")
    # print(ground_positions)

    start_positions = []
    for segment_num, segment in enumerate(matrix):
        start_positions.append([color_of_segments[segment_num]*-1*X_OFFSET, segment[0][10], segment[0][2], segment_num])
    # print("Start Positions before sorting:")
    # print(start_positions)

    start_positions.sort(key=tuple)
    # print("Start Positions after sorting:")
    # print(start_positions)

    start_trajectories = [(ground, start) for ground, start in zip(ground_positions, start_positions)]
    # print('start_trajectories')
    # print(start_trajectories, sep='\n')

    start_trajectories.sort(key=lambda x: x[0][3])
    # print('start_trajectories')
    # print(start_trajectories, sep='\n')

    # delete all .csv files in traj
    dir_path = ROS_WS + 'scripts/traj/'
    for file in os.listdir(dir_path):
        if file.endswith('.csv'):
            os.remove(dir_path + file)


    end_positions = {}
    YAW = 0 #math.pi
    for quadcopter_index in range(len(matrix)):
        with open(ROS_WS + 'scripts/traj/trajectory{0}.csv'.format(int(start_trajectories[quadcopter_index][0][3])), 'w') as filename:
            writer = csv.writer(filename)
            writer.writerow(np.concatenate([['duration'],[axis + '^' + str(i) for axis in ['x', 'y', 'z', 'yaw'] for i in range(8)]]))
            duration_of_segment = 0
            x0, y0, z0 = start_trajectories[quadcopter_index][0][:3]
            z0 += HOVER_OFFSET
            dx, dy, dz = (start_trajectories[quadcopter_index][1][:3] - np.array([x0, y0, z0])) / TAKE_OFF_TIME
            writer.writerow(np.concatenate([[TAKE_OFF_TIME], [x0, dx], [0] * 6, [y0, dy], [0] * 6, [z0, dz], [0] * 6, [YAW], [0]*7]))
            curr_segment = matrix[start_trajectories[quadcopter_index][1][3]]
            first_piece = curr_segment[0]
            hover_x = start_trajectories[quadcopter_index][1][0]
            hover_y = first_piece[10]
            hover_z = first_piece[2]
            writer.writerow(np.concatenate([[HOVER_PAUSE_TIME], [hover_x], [0] * 7, [hover_y], [0] * 7, [hover_z], [0] * 7, [YAW], [0] * 7]))
            for piece in curr_segment:
                piece[18:26] = piece[2:10].copy()
                piece[2:10] = [0 for _ in range(8)]
                piece[26] = YAW
                writer.writerow(np.concatenate([[piece[1]], [hover_x], [(i) for i in piece[3:-1]]]))
                duration_of_segment += piece[1]
            end_x = start_trajectories[quadcopter_index][1][0]
            end_y = np.poly1d(piece[10:18][::-1])(piece[1])
            end_z = np.poly1d(piece[18:26][::-1])(piece[1])
            end_positions[int(start_trajectories[quadcopter_index][0][3])] = [end_x, end_y, end_z]
            writer.writerow(np.concatenate([[limit - duration_of_segment], [hover_x], [0] * 7, \
             [end_y], [0] * 7, [end_z], [0] * 15]))

    print(end_positions)
    with open(ROS_WS + 'scripts/data/lights.csv', 'w') as filename:
        writer = csv.writer(filename)
        writer.writerow(['index', 'time', 'state_change'])
        for quadcopter_index in range(len(matrix)):
            writer.writerow([start_trajectories[quadcopter_index][0][3], 0, 0])
            curr_time = TAKE_OFF_TIME + HOVER_PAUSE_TIME
            curr_state = 0 
            segment = matrix[start_trajectories[quadcopter_index][1][3]]
            for piece in segment:
                # print(quadcopter_index, curr_time, curr_state, piece[0], piece[1], piece[-1], sep='\t')
                if curr_state != piece[-1]:
                    curr_state = piece[-1]
                    writer.writerow([start_trajectories[quadcopter_index][0][3], curr_time, curr_state])
                curr_time += piece[1]

    # channel = [100, 110, 120]
    # with open(ROS_WS + "launch/crazyflies.yaml", "w") as filename:
    #     filename.write("crazyflies:\n")
    #     for segment_num in range(len(matrix)):
    #         filename.write(' - id: ' + str(segment_num + 1) + '\n')
    #         filename.write('   channel: {0}\n'.format(channel[segment_num%3]))
    #         filename.write('   initialPosition: [{0}, {1}, {2}]\n'.format(*initialPositions[segment_num]))

if __name__ == "__main__":
    main()
