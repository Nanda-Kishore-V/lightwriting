from __future__ import division, print_function

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import json

from constants import (
    HOME,
    VERBOSE_TEXT,
)
from geometry import (
    Vector,
    Point,
    Segment,
    Path,
    MetricSurface,
)

def form_pairs(
    file_ip,
    file_op=None,
    max_time_per_segment=20,
    max_pieces_per_segment=30,
    max_number_of_paths = 2
):

    if file_op is None:
        file_op = file_ip

    with open(file_ip, "rb") as f:
        data = np.genfromtxt(f, dtype= float, delimiter=',')
        data = np.delete(data, (0), axis = 0)

    paths = []
    for index, d in enumerate(data):
        temp_data = {
            'point1': tuple(d[1:4]),
            'tangent1': Vector(tuple(d[4:7])),
            'point2': tuple(d[7:10]),
            'tangent2': Vector(tuple(d[10:13])),
        }

        start = Point(temp_data['point1'], temp_data['tangent1'])
        end = Point(temp_data['point2'], temp_data['tangent2'])
        s = Segment([start, end])
        if VERBOSE_TEXT: print('s', s)
        p = Path([s])
        paths.append(p)

    paths.sort(key = lambda p: p.length)

    m = MetricSurface()
    # form pairs now
    while len(paths) > max_number_of_paths:
        path_smallest = paths.pop(0)
        metric_highest = -1
        index_best = None
        points_end_best = None
        for i, p in enumerate(paths):
            metric_curr, points_end_candidate = Path.select_pair(path_smallest, p, m)
            if metric_curr > metric_highest:
                index_best = i
                metric_highest = metric_curr
                points_end_best = points_end_candidate

        path_new = Path.join(path_smallest, paths[index_best], *points_end_best)
        del paths[index_best]
        # optimize later to insert path_new into correct position
        paths.append(path_new)
        paths.sort(key = lambda p: p.length)

    for i, p in enumerate(paths):
        x = []
        y = []
        for s in p.segments:
            x.append(s.points[0].coords[0])
            x.append(s.points[1].coords[0])
            y.append(s.points[0].coords[1])
            y.append(s.points[1].coords[1])
        plt.figure(i)
        plt.scatter(y, x, c='r')
        plt.xlim([0, 100])
        plt.ylim([0, 100])
        plt.show()

    path_dicts = [Path.to_dict(p) for p in paths]
    if VERBOSE_TEXT: print(*path_dicts, sep='\n')
    with open(file_op, 'w') as f:
        json.dump(path_dicts, f)

def main():
    file_ip = HOME + 'data/tangents.csv'
    file_op = HOME + 'data/long_paths.json'
    form_pairs(file_ip, file_op)

if __name__ == "__main__":
    main()
