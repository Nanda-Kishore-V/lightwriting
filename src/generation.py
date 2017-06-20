from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from pypoly import Polynomial
import csv
import operator
import json

import snap
from constants import HOME
from geometry import Segment

DER = 3
MAX_VELOCITY = 3

def find_value(rows,time):
    if len(rows) == 1:
        return Polynomial(*rows[0])(time)
    P = []
    for index, row in enumerate(rows):
        P.append(Polynomial(*row))
    return tuple([poly(time) for poly in P])

def find_derivative(row,time):
    del_t = 0.001
    return (find_value(row, time+del_t) - find_value(row, time))/del_t

def find_derivative_multiple(rows,time):
    der = []
    for row in rows:
        der.append(find_derivative([row], time))
    return tuple(der)

# def


if __name__=="__main__":
    xy_figure = plt.figure()
    xy_axis = xy_figure.add_subplot(111)
    xy_figure.canvas.set_window_title('XY Top-Down View')
    xy_background = xy_figure.canvas.copy_from_bbox(xy_axis.bbox)
    xy_axis.set_xlabel('x [m]')
    xy_axis.set_ylabel('y [m]')

    # load data
    with open(HOME + 'data/waypoints.json') as f:
        segment_dicts = json.load(f)
    segments = [Segment.from_dict(s_dict) for s_dict in segment_dicts]
    n_segments = len(segments)
    print(*segments, sep='\n')
    matrix = [[index, p.coords[0], p.coords[1], 0, 0]for index, s in enumerate(segments) for p in s.points]
    print(*matrix, sep='\n')

    num_segments = int(n_segments)

    input_data_multiple = [[[], [], [], []] for x in range(num_segments)]
    curr_segment = 0
    for row in matrix:
        if row[0] != curr_segment:
            curr_segment += 1
        for x in range(len(row)-1):
            #print('current segment:{0}\tx: {1}'.format(curr_segment, x))
            input_data_multiple[curr_segment][x].append([row[x+1]])

    open(HOME + "data/output.csv", "w").close()
    open(HOME + "data/tangents.csv", "w").close()
    data = [[None for i in range(13)] for i in range(num_segments)]

    for segment_num,input_data in enumerate(input_data_multiple):
        print(segment_num, " is being optimized.")
        wap_num = len(input_data[0])
        print("wap_num {0}".format(wap_num))

        xy_axis.plot(input_data[1], input_data[0], 'ro')
        xy_lines = []

        if wap_num == 2:
            T = (input_data[0][1][0] - input_data[0][0][0])**2
            T += (input_data[1][1][0] - input_data[1][0][0])**2
            T += (input_data[2][1][0] - input_data[2][0][0])**2
            T = [(T**0.5)/MAX_VELOCITY]
            x = [input_data[0][0][0], (input_data[0][1][0] - input_data[0][0][0])/T[0], 0, 0, 0, 0, 0, 0]
            y = [input_data[1][0][0], (input_data[1][1][0] - input_data[1][0][0])/T[0], 0, 0, 0, 0, 0, 0]
            z = [input_data[2][0][0], (input_data[2][1][0] - input_data[2][0][0])/T[0], 0, 0, 0, 0, 0, 0]
            psi = [0] * 8

        else:
            # begin and end with no velocity and acceleration
            for i in range(3):
                input_data[i][0].extend([0.0, 0.0])
                input_data[i][-1].extend([0.0, 0.0])

            x_wp = np.array(input_data[0])
            y_wp = np.array(input_data[1])
            z_wp = np.array(input_data[2])
            psi_wp = np.array(input_data[3])
            #x = snap.Trajectory1D(x_wp)
            #y = snap.Trajectory1D(y_wp)
            x = snap.Trajectory1D(x_wp, der=DER)
            y = snap.Trajectory1D(y_wp, der=DER)
            z = snap.Trajectory1D(z_wp, der=DER)
            psi = snap.Trajectory1D(psi_wp, der=DER)


            print('Running optimization routine...')
            trajectory = snap.QrPath(x, y, z, psi, power=10.00, tilt=0.25, guess=5.00)
            T = trajectory.optimize()  # an array of segment time length

        with open(HOME + 'data/output.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            if segment_num == 0:
                temp_x = ["x^"+str(degree) for degree in range(2*(DER+1))]
                temp_y = ["y^"+str(degree) for degree in range(2*(DER+1))]
                temp_z = ["z^"+str(degree) for degree in range(2*(DER+1))]
                temp_yaw = ["yaw^"+str(degree) for degree in range(2*(DER+1))]
                spamwriter.writerow(np.concatenate([[int(num_segments)],['duration'], temp_x, temp_y, temp_z, temp_yaw]))
            if wap_num == 2:
                spamwriter.writerow(np.concatenate([[int(segment_num)],T, x, y, z, psi]))
            else:
                for i in range(wap_num-1):
                    spamwriter.writerow(np.concatenate([[int(segment_num)],[T[i]], x.p[i], y.p[i], z.p[i], psi.p[i]]))

        tdata = np.arange(0, sum(T), 0.1)
        if wap_num == 2:
            xdata = [find_value([x], t) for t in tdata]
            ydata = [find_value([y], t) for t in tdata]
            zdata = [find_value([z], t) for t in tdata]
        else:
            xdata = [x(t) for t in tdata]
            ydata = [y(t) for t in tdata]
            zdata = [z(t) for t in tdata]

        xy_lines.append(Line2D(ydata, xdata, linestyle='-', marker='', color='b'))

        xy_axis.add_line(xy_lines[-1])
        xy_figure.canvas.restore_region(xy_background)
        xy_figure.canvas.blit(xy_axis.bbox)

        wp_t = [0]
        for t in T:
            wp_t.append(wp_t[-1] + t)

        if wap_num == 2:
            data[segment_num][0] = T[0]
            data[segment_num][1:4] = find_value([x, y, z], 0)
            start_vector = np.array(find_derivative_multiple([x, y, z], 0))
            data[segment_num][4:7] = start_vector / np.linalg.norm(start_vector)
            data[segment_num][7:10] = find_value([x, y, z], T[0])
            end_vector = -1 * np.array(find_derivative_multiple([x, y, z], T[0]))
            data[segment_num][10:13] = end_vector / np.linalg.norm(end_vector)
        else:
            data[segment_num][0] = wp_t[-1]
            data[segment_num][1:4] = find_value([x.p[0], y.p[0], z.p[0]], 0)
            start_vector = np.array(find_derivative_multiple([x.p[0], y.p[0], z.p[0]], 0))
            data[segment_num][4:7] = start_vector / np.linalg.norm(start_vector)
            data[segment_num][7:10] = find_value([x.p[-1], y.p[-1], z.p[-1]], T[-1])
            end_vector = -1 * np.array(find_derivative_multiple([x.p[-1], y.p[-1], z.p[-1]], T[-1]))
            data[segment_num][10:13] = end_vector / np.linalg.norm(end_vector)

        with open(HOME + 'data/tangents.csv', 'a', newline='') as tangent_file:
            writer = csv.writer(tangent_file)
            if segment_num == 0:
                axes = ['x', 'y', 'z']
                start = ['start ' + i for i in axes]
                s_vector = ['start vector ' + i for i in axes]
                end = ['end ' + i for i in axes]
                e_vector = ['end vector ' + i for i in axes]
                writer.writerow(np.concatenate([['time per segment'], start, s_vector, end, e_vector]))
            # segment id, start point coords, start tangent coords, end point coords, end tangent coords
            writer.writerow(data[segment_num])


    origin_x = [];  origin_y = []
    vector_x = [];  vector_y = []
    for row in data:
        origin_x.append(row[1])
        origin_y.append(row[2])
        vector_x.append(row[4])
        vector_y.append(row[5])
        origin_x.append(row[7])
        origin_y.append(row[8])
        vector_x.append(row[10])
        vector_y.append(row[11])

    colour = ['r' if i%2 == 0 else 'b' for i in range(len(origin_x))]
    for i in range(len(origin_x)):
        if i%2 == 0:
            xy_axis.annotate(str(int(i/2)), xy=(origin_y[i], origin_x[i]), xytext=(origin_y[i] + 1, origin_x[i] + 1),
                arrowprops=dict(facecolor='white', shrink=0.05))
    xy_axis.quiver(origin_y, origin_x, vector_y, vector_x, color=colour)
    # plt.xlim(0,500)
    # plt.ylim(0,500)
    plt.show()
