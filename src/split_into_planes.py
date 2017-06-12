import numpy as np
import csv
from pypoly import Polynomial
import operator

from sets import Set
from constants import HOME

epsilon = 5

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
    for i in range(len(shift_times)):
        s += shift_times[i]
        if s >= time:
            break
    Px = Polynomial(*segment[i][2:10])
    Py = Polynomial(*segment[i][10:18])
    Pz = Polynomial(*segment[i][18:26])
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
    dt = min(times)/100
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
    # for segment_1, segment_2 in intersecting_pairs:
    #     if segment_1 in list_segments:
    #         # TODO: Add segment_1 to some set
    #     if segment_2 in list_segments:
    #         # TODO: Add segment_2 to some set

if __name__ == "__main__":
    main()
