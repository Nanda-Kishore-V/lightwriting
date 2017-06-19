from __future__ import (
    division,
    print_function,
    )

from mpl_toolkits.mplot3d import (
        Axes3D,
        )

import numpy as np
import matplotlib.pyplot as plt
import json

from constants import (
    HOME
    )
from geometry import (
    Vector,
    Point,
    Segment,
    Path,
    MetricSurface,
    )

def form_pairs(
        filename_input,
        filename_output=None,
        max_time_per_segment=20,
        max_pieces_per_segment=30,
        ):

    if filename_output is None:
        filename_output = filename_input

    with open(filename_input, "rb") as f:
        data = np.genfromtxt(f, dtype= float, delimiter=',')
        data = np.delete(data, (0), axis = 0)

    #index, distance between end points, end point 1, tangent 1, end point 2, tangent 2
    paths = []
    for index, d in enumerate(data):
        temp_data = {}
        temp_data['point1'] = tuple(d[1:4])
        temp_data['tangent1'] = Vector(tuple(d[4:7]))
        temp_data['point2'] = tuple(d[7:10])
        temp_data['tangent2'] = Vector(tuple(d[10:13]))

        start = Point(temp_data['point1'], temp_data['tangent1'])
        end = Point(temp_data['point2'], temp_data['tangent2'])
        s = Segment([start, end])
        print('s', s)
        p = Path([s])
        paths.append(p)

    '''
    print('paths before sorting:')
    for p in paths:
        print(p)
    '''
    paths.sort(key = lambda p: p.length)
    '''
    print('paths after sorting:')
    for p in paths:
        print(p)
    
    print('-' * 80)
    '''

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

    '''
    print('paths after combining:')
    for p in paths:
        print(p)
    '''

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
    print(*path_dicts, sep='\n')
    with open(filename_output, 'w') as f:
        json.dump(path_dicts, f)

def main():
    filename_input = HOME + 'data/tangents.csv'
    filename_output = HOME + 'data/long_paths.json'
    form_pairs(filename_input, 2, filename_output)

if __name__ == "__main__":
    main()
