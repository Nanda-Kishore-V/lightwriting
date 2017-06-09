from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from pypoly import Polynomial
import csv
import operator

import snap
from constants import HOME

DER = 3

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


if __name__=="__main__":
    xy_figure = plt.figure()
    xy_axis = xy_figure.add_subplot(111)
    xy_figure.canvas.set_window_title('XY Top-Down View')
    xy_background = xy_figure.canvas.copy_from_bbox(xy_axis.bbox)
    xy_axis.set_xlabel('x [m]')
    xy_axis.set_ylabel('y [m]')

    # load data
    f = open(HOME + "data/waypoints.csv", "r")
    csv_reader = csv.reader(f)
    first_line = next(csv_reader)
    matrix = np.loadtxt(f, delimiter=",", skiprows=0)

    num_segments = int(first_line[0])

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

        xy_axis.plot(input_data[1], input_data[0], 'ro')

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
        psi = snap.Trajectory1D(psi_wp, der=2)

        xy_lines = []

        print('Running optimization routine...')
        trajectory = snap.QrPath(x, y, z, psi, power=10.00, tilt=0.25, guess=5.00)
        T = trajectory.optimize()  # an array of segment time length

        with open(HOME + '/data/output.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            if segment_num == 0:
                temp_x = ["x^"+str(degree) for degree in range(2*(DER+1))]
                temp_y = ["y^"+str(degree) for degree in range(2*(DER+1))]
                temp_z = ["z^"+str(degree) for degree in range(2*(DER+1))]
                temp_yaw = ["yaw^"+str(degree) for degree in range(6)]
                spamwriter.writerow(np.concatenate([[int(num_segments)],['duration'], temp_x, temp_y, temp_z, temp_yaw]))
            for i in range(wap_num-1):
                spamwriter.writerow(np.concatenate([[int(segment_num)],[T[i]], x.p[i], y.p[i], z.p[i], psi.p[i]]))

        tdata = np.arange(0, sum(T), 0.1)
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

        data[segment_num][0] = wp_t[-1]
        data[segment_num][1:4] = find_value([x.p[0], y.p[0], z.p[0]], 0)
        start_vector = np.array(find_derivative_multiple([x.p[0], y.p[0], z.p[0]], 0))
        data[segment_num][4:7] = start_vector / np.linalg.norm(start_vector)
        data[segment_num][7:10] = find_value([x.p[-1], y.p[-1], z.p[-1]], T[-1])
        start_vector = -1 * np.array(find_derivative_multiple([x.p[-1], y.p[-1], z.p[-1]], wp_t[-1]))
        data[segment_num][10:13] = start_vector / np.linalg.norm(start_vector)

        with open(HOME + '/data/tangents.csv', 'a', newline='') as tangent_file:
            writer = csv.writer(tangent_file)
            if segment_num == 0:
                axes = ['x', 'y', 'z']
                start = ['start ' + i for i in axes]
                s_vector = ['start vector ' + i for i in axes]
                end = ['end ' + i for i in axes]
                e_vector = ['end vector ' + i for i in axes]
                writer.writerow(np.concatenate([['time per segment'], start, s_vector, end, e_vector]))
            # segment id, start point coordinates, start tangent coordinates, end point coordinates, end tangent coordinates
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
    xy_axis.annotate(str(0), xy=(origin_y[0], origin_x[0]), xytext=(origin_y[0] + 1, origin_x[0] + 1),
            arrowprops=dict(facecolor='white', shrink=0.05))
    xy_axis.annotate(str(1), xy=(origin_y[2], origin_x[2]), xytext=(origin_y[2] + 1, origin_x[2] + 1),
            arrowprops=dict(facecolor='white', shrink=0.05))
    xy_axis.annotate(str(2), xy=(origin_y[4], origin_x[4]), xytext=(origin_y[4] + 1, origin_x[4] + 1),
            arrowprops=dict(facecolor='white', shrink=0.05))
    xy_axis.quiver(origin_y, origin_x, vector_y, vector_x, color=colour)
    # plt.xlim(0,500)
    # plt.ylim(0,500)
    plt.show()
