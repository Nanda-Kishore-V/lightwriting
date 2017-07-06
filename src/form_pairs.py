from __future__ import division, print_function

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import json
import csv

from constants import (
    HOME,
    VERBOSE_TEXT,
    MAX_PIECEWISE_POLYNOMIALS,
    CAMERA_EXPOSURE_TIME_LIMIT,
    MAX_QUADROTOR_VELOCITY,
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
    file_op,
    max_time=CAMERA_EXPOSURE_TIME_LIMIT,
    max_pieces=MAX_PIECEWISE_POLYNOMIALS,
):

    with open(file_ip, "rb") as f:
        data = np.loadtxt(f, delimiter=',', skiprows=1, ndmin=2)

    paths = []
    for index, d in enumerate(data):
        temp_data = {
            'time': d[0],
            'point1': tuple(d[1:4]),
            'tangent1': Vector(tuple(d[4:7])),
            'point2': tuple(d[7:10]),
            'tangent2': Vector(tuple(d[10:13])),
        }

        start = Point(temp_data['point1'], temp_data['tangent1'])
        end = Point(temp_data['point2'], temp_data['tangent2'])
        s = Segment([start, end], time=temp_data['time'], index=index)
        if VERBOSE_TEXT: print('s', s)
        p = Path([s])
        paths.append(p)

    paths.sort(key = lambda p: p.time)

    for p in paths:
        if p.time() > max_time or p.pieces() > max_pieces:
            print('Error! Path is initially to long in terms of time or pieces')
            print('Path: ', p)
            exit()

    if VERBOSE_TEXT:
        print('Paths before forming pairs: ', len(paths))
        print('Paths:')
        print(*paths, sep='\n')
        raw_input('-' * 80)

    paths_done = []
    m = MetricSurface()
    # form pairs now
    while paths:
        if VERBOSE_TEXT:
            raw_input('-' * 80)
            print('len of paths: ', len(paths))
        path_smallest = paths.pop(0)
        metric_max = float('-inf')
        index_best = None
        path_best = None
        for i, p in enumerate(paths):
            metric, points_end = Path.select_pair(path_smallest, p, m)
            path = Path.join(path_smallest, p, *points_end)
            if path.time() > max_time or path.pieces() > max_pieces:
                continue
            if metric > metric_max:
                index_best = i
                metric_max = metric
                path_best = path

        if index_best is None:
            if VERBOSE_TEXT:
                print('No partner could be found for path_smallest: ', path_smallest)
            paths_done.append(path_smallest)
            if VERBOSE_TEXT:
                print('path_smallest was moved to paths_done')
            continue
        if VERBOSE_TEXT:
            print('Combining path_smallest: ', path_smallest)
            print('With path with index_best: ', paths[index_best])
        del paths[index_best]
        # optimize later to insert path_new into correct position
        paths.append(path_best)
        if VERBOSE_TEXT:
            print('Newly added path is: ', path_best)
        paths.sort(key = lambda p: p.time)

    if VERBOSE_TEXT:
        raw_input('-' * 80)
    paths = paths_done
    paths.sort(key = lambda p: p.time)
    if VERBOSE_TEXT:
        print('Paths after forming pairs: ', len(paths))
        print('Paths:')
        print(*paths, sep='\n')
        raw_input('-' * 80)

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
        plt.axis('equal')
        plt.show()

    path_dicts = [Path.to_dict(p) for p in paths]
    if VERBOSE_TEXT: print(*path_dicts, sep='\n')
    with open(file_op, 'w') as f:
        json.dump(path_dicts, f)

    with open(HOME + 'data/output.csv') as f:
        matrix = np.loadtxt(f, delimiter=',', skiprows=1, ndmin=2)
        matrix = np.split(matrix, np.where(np.diff(matrix[:,0]))[0]+1)

    file_temp = HOME + 'data/temp.csv'
    with open(file_temp, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(np.concatenate([['Index', 'Duration'], [x + '^' + str(i) for x in ['x', 'y', 'z', 'yaw'] for i in range(8)]]))
        for index_path, p in enumerate(paths):
            for s in p.segments:
                if s.index is None:
                    x0, y0, z0 = s.points[0].coords
                    x1, y1, z1 = s.points[-1].coords
                    x_vel = (x1 - x0) / s.time
                    y_vel = (y1 - y0) / s.time
                    z_vel = (z1 - z0) / s.time
                    single_row = [index_path, s.time] + [x0, x_vel] + [0] * 6 + [y0, y_vel] + [0] * 6 + [z0, z_vel] + [0] * 14
                    writer.writerow(single_row)
                    continue
                if s.is_reversed:
                    reversed_segment = []
                    for piece in matrix[s.index]:
                        duration = float(piece[1])
                        change_of_variable = np.poly1d([1, -1.0*duration])

                        poly1d_x = piece[2:10][::-1]
                        poly1d_x = np.poly1d([p*(-1)**(index+1) for index, p in enumerate(poly1d_x)])
                        poly1d_x = np.array(np.polyval(poly1d_x, change_of_variable))[::-1]
                        poly1d_y = piece[10:18][::-1]
                        poly1d_y = np.poly1d([p*(-1)**(index+1) for index, p in enumerate(poly1d_y)])
                        poly1d_y = np.array(np.polyval(poly1d_y, change_of_variable))[::-1]
                        poly1d_z = piece[18:26][::-1]
                        poly1d_z = np.poly1d([p*(-1)**(index+1) for index, p in enumerate(poly1d_z)])
                        poly1d_z = np.array(np.polyval(poly1d_z, change_of_variable))[::-1]

                        if len(poly1d_x) < 8:
                            poly1d_x = np.concatenate((poly1d_x, [0. for _ in range(8 - len(poly1d_x))]))
                        if len(poly1d_y) < 8:
                            poly1d_y = np.concatenate((poly1d_y, [0. for _ in range(8 - len(poly1d_y))]))
                        if len(poly1d_z) < 8:
                            poly1d_z = np.concatenate((poly1d_z, [0. for _ in range(8 - len(poly1d_z))]))

                        new_piece = np.concatenate(([0, duration], poly1d_x, poly1d_y, poly1d_z, [0 for _ in range(8)]))
                        reversed_segment.append(new_piece)
                    matrix[s.index] = reversed_segment[::-1]
                for line in matrix[s.index]:
                        writer.writerow(np.concatenate([[index_path], line[1:]]))

def main():
    file_ip = HOME + 'data/tangents.csv'
    file_op = HOME + 'data/long_paths.json'
    form_pairs(file_ip, file_op)

if __name__ == "__main__":
    main()
