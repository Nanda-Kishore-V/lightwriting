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
import itertools

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    MAX_QUADROTOR_VELOCITY,
    MAX_TAKEOFF_VELOCITY,
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
            ground_positions.append(tuple(np.concatenate([cf['initialPosition'], [cf['id']]])))
    # print("Start Positions before sorting:")
    # print(ground_positions)

    ground_positions.sort(key=tuple, reverse=False)
    # print("Start Positions after sorting:")

    start_positions = []
    for segment_num, segment in enumerate(matrix):
        start_positions.append(tuple([color_of_segments[segment_num]*-1*X_OFFSET, segment[0][10], segment[0][2], segment_num]))
    # print("Start Positions before sorting:")
    # print(start_positions)

    start_positions.sort(key=tuple, reverse=False)
    start_positions_groups = [list(g) for k, g in itertools.groupby(start_positions, lambda x: x[0])]
    print('start positions groups')
    print(*start_positions_groups, sep='\n')
    # print("Start Positions after sorting:")
    # print(start_positions)

    max_start_trajectory_times = []
    start_trajectories = []
    taken_ground_positions = []
    taken_start_positions = []
    for wave_number, start_positions_group in enumerate(start_positions_groups):
        n = len(start_positions_group)
        ground_positions_group, ground_positions = ground_positions[:n], ground_positions[n:]

        start_trajectory_distances = [(distance(g[:3], s[:3]), g, s) for g in ground_positions_group for s in start_positions_group]
        start_trajectory_distances.sort()
        while(start_trajectory_distances):
            potential_trajectory = start_trajectory_distances.pop(0)
            if potential_trajectory[1] not in taken_ground_positions and potential_trajectory[2] not in taken_start_positions:
                start_trajectories.append(tuple([wave_number] + list(potential_trajectory[1:])))
                taken_ground_positions.append(potential_trajectory[1])
                taken_start_positions.append(potential_trajectory[2])
        start_trajectory_times = [distance(g[:3], s[:3]) / MAX_TAKEOFF_VELOCITY for (_, g, s) in start_trajectories]
        max_start_trajectory_times.append(max(start_trajectory_times))

    print('start trajectories')
    print(*start_trajectories, sep='\n')
    start_trajectory_groups = [list(g) for k, g in itertools.groupby(start_trajectories, lambda x: x[0])]
    print('start trajectory groups')
    print(*start_trajectory_groups, sep='\n')
    for group in start_trajectory_groups:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for trajectory in group:
            # print(zip(trajectory[1][0][:3], trajectory[1][1][:3]))
            ax.plot(*zip(trajectory[1][:3], trajectory[2][:3]))
        plt.show()

    # take 3


    # take 2 - we found sometihing better than this
    # start_trajectory_distances = [(distance(g[:3], s[:3]), g, s) for g in ground_positions for s in start_positions]
    # start_trajectory_distances.sort()
    # start_trajectories = []
    # taken_ground_positions = []
    # taken_start_positions = []
    # while(start_trajectory_distances):
    #     potential_trajectory = start_trajectory_distances.pop(0)
    #     if potential_trajectory[1] not in taken_ground_positions and potential_trajectory[2] not in taken_start_positions:
    #         start_trajectories.append(potential_trajectory[1:])
    #         taken_ground_positions.append(potential_trajectory[1])
    #         taken_start_positions.append(potential_trajectory[2])
    # start_trajectory_times = [distance(g[:3], s[:3]) / MAX_QUADROTOR_VELOCITY for (g, s) in start_trajectories]
    # max_start_trajectory_times = max(start_trajectory_times)

    # take 1 - failed
    # start_positions_copy = start_positions[:]
    # start_trajectories = []
    # for ground_position in ground_positions:
    #     start_position_best = None
    #     distance_shortest = float('inf')
    #     for start_position in start_positions_copy:
    #         distance_curr = distance(ground_position[:3], start_position[:3])
    #         if distance_curr < distance_shortest:
    #             distance_shortest = distance_curr
    #             start_position_best = start_position
    #     start_trajectories.append((ground_position, start_position))
    #     start_positions_copy.remove(start_position)

    # original
    # start_trajectories = [(ground, start) for ground, start in zip(ground_positions, start_positions)]

    # print('start_trajectories')
    # print(start_trajectories, sep='\n')

    # print(*start_trajectories, sep='\n')
    # start_trajectories, start_trajectory_times = zip(*sorted(zip(start_trajectories, start_trajectory_times), key=lambda x: x[0][0][3]))
    # print('times and trajectories')
    # print(*zip(start_trajectory_times, [(distance(g[:3], s[:3])) for (g,s) in start_trajectories]), sep='\n')
    # print('start trajectory times')
    # print(*start_trajectory_times, sep='\n')
    # print('start_trajectories')
    # print(start_trajectories, sep='\n')

    # delete all .csv files in traj
    dir_path = ROS_WS + 'scripts/traj/'
    for file in os.listdir(dir_path):
        if file.endswith('.csv'):
            os.remove(dir_path + file)

    wait_times_before = [sum(max_start_trajectory_times[:i]) for i in range(len(max_start_trajectory_times))]
    wait_times_after = [sum(max_start_trajectory_times[i + 1:]) for i in range(len(max_start_trajectory_times))]
    end_positions = {}
    YAW = 0
    YAW_TIME = 0.0
    for segment_index in range(len(matrix)):
        quadcopter_index = int(start_trajectories[segment_index][1][3])
        curr_trajectory = start_trajectories[segment_index]
        wave_number, curr_ground_position, curr_start_position = curr_trajectory
        curr_start_position = (curr_start_position[0],
                               curr_start_position[1]*(CAMERA_DISTANCE + curr_start_position[0]) / CAMERA_DISTANCE,
                               curr_start_position[2]*(CAMERA_DISTANCE + curr_start_position[0]) / CAMERA_DISTANCE,
                               curr_start_position[3])
        with open(ROS_WS + 'scripts/traj/trajectory{0}.csv'.format(quadcopter_index), 'w') as filename:
            writer = csv.writer(filename)
            writer.writerow(np.concatenate([['duration'],[axis + '^' + str(i) for axis in ['x', 'y', 'z', 'yaw'] for i in range(8)]]))
            duration_of_segment = 0
            x0, y0, z0 = curr_ground_position[:3]
            z0 += HOVER_OFFSET
            # writer.writerow(np.concatenate([[YAW_TIME, x0], [0] * 7, [y0], [0] * 7, [z0], [0] * 8, [YAW/YAW_TIME], [0] * 6]))
            duration_of_wait = wait_times_before[wave_number]
            writer.writerow(np.concatenate([[duration_of_wait], [x0], [0] * 7, [y0], [0] * 7, [z0], [0] * 7, [YAW], [0] * 7]))
            dx, dy, dz = (np.array(curr_start_position[:3]) - np.array([x0, y0, z0])) / start_trajectory_times[segment_index]
            writer.writerow(np.concatenate([[start_trajectory_times[segment_index]], [x0, dx], [0] * 6, \
             [y0, dy], [0] * 6, [z0, dz], [0] * 6, [YAW], [0]*7]))
            curr_segment = matrix[curr_start_position[3]]
            first_piece = curr_segment[0]
            hover_x = curr_start_position[0]
            hover_y = first_piece[10] * (CAMERA_DISTANCE + curr_start_position[0]) / CAMERA_DISTANCE
            hover_z = first_piece[2] * (CAMERA_DISTANCE + curr_start_position[0]) / CAMERA_DISTANCE
            duration_of_wait = wait_times_after[wave_number] + max_start_trajectory_times[wave_number] - start_trajectory_times[segment_index]
            writer.writerow(np.concatenate([[duration_of_wait], [hover_x], [0] * 7, [hover_y], [0] * 7, [hover_z], [0] * 7, [YAW], [0] * 7]))
            for piece in curr_segment:
                piece[18:26] = piece[2:10].copy()
                piece[2:10] = [0 for _ in range(8)]
                piece[26] = YAW
                writer.writerow(np.concatenate([[piece[1]], [hover_x], [(i*(CAMERA_DISTANCE + curr_start_position[0]) / CAMERA_DISTANCE) for i in piece[3:-1]]]))
                duration_of_segment += piece[1]
            end_x = curr_start_position[0]
            end_y = np.poly1d(piece[10:18][::-1])(piece[1])
            end_z = np.poly1d(piece[18:26][::-1])(piece[1])
            end_positions[quadcopter_index] = [end_x, end_y, end_z]
            writer.writerow(np.concatenate([[limit - duration_of_segment], [hover_x], [0] * 7, \
             [end_y], [0] * 7, [end_z], [0] * 7, [YAW], [0] * 7]))

    print(end_positions)
    with open(ROS_WS + 'scripts/data/lights.csv', 'w') as filename:
        writer = csv.writer(filename)
        writer.writerow(['index', 'time', 'state_change'])
        time_origin = wait_times_before[wave_number] + wait_times_after[wave_number] + max_start_trajectory_times[wave_number] + YAW_TIME
        print('origin', time_origin)
        for segment_index in range(len(matrix)):
            quadcopter_index = int(start_trajectories[segment_index][1][3])
            curr_trajectory = start_trajectories[segment_index]
            wave_number, curr_ground_position, curr_start_position = curr_trajectory
            writer.writerow([curr_ground_position[3], 0, 0])
            curr_time = time_origin
            curr_state = 0
            segment = matrix[curr_start_position[3]]
            for piece in segment:
                # print(segment_index, curr_time, curr_state, piece[0], piece[1], piece[-1], sep='\t')
                if curr_state != piece[-1]:
                    curr_state = piece[-1]
                    writer.writerow([curr_ground_position[3], curr_time, curr_state])
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
