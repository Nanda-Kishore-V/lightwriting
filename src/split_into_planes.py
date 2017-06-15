import numpy as np
import csv
from pypoly import Polynomial
import operator
import matplotlib.pyplot as plt

from sets import Set
from constants import HOME

epsilon = 5
ROS_WS = "/home/nanda/Documents/Intern/crazyflie/crazyswarm/ros_ws/src/crazyswarm/scripts/"

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
    print "Something is wrong with the number of segments."
    return None

def main():
    f = open(HOME + "data/output.csv", "r")
    csv_reader = csv.reader(f)
    first_line = next(csv_reader)
    matrix = np.loadtxt(f, delimiter=",", skiprows=0)
    matrix = np.split(matrix, np.where(np.diff(matrix[:,0]))[0]+1)

    times = [sum(segment[i][1] for i in range(len(segment))) for segment in matrix]
    dt = min(times)/100.0
    t = 0
    limit = second_largest(times)

    intersecting_pairs = Set([])
    while t <= limit:
        locations = []
        for segment in matrix:
            locations.append(return_location(segment, t))

        combinations = [(i,j) for i in range(len(locations)) for j in range(len(locations)) if i < j]
        distances = [distance(locations[i], locations[j]) for i, j in combinations]
        is_intersecting = map(lambda x : x < epsilon, distances)
        for index, elem in enumerate(is_intersecting):
            if elem:
                intersecting_pairs.add(combinations[index])
        t += dt
    print intersecting_pairs
    color_of_segments = {}
    for i in range(len(matrix)):
        color_of_segments[i] = -1
    for node in range(len(matrix)):
        color_of_segments[node] = get_color(node, intersecting_pairs, color_of_segments)
    print color_of_segments

    for segment_num, segment in enumerate(matrix):
        with open(ROS_WS + "/traj/trajectory" + str(segment_num) + ".csv", "w") as filename:
            writer = csv.writer(filename)
            writer.writerow(np.concatenate([['duration'],[axis + '^' + str(i) for axis in ['x', 'y', 'z', 'yaw'] for i in range(8)]]))
            for piece in segment:
                writer.writerow(np.concatenate([[i] for i in piece[1:]]))

if __name__ == "__main__":
    main()
