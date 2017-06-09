from mpl_toolkits.mplot3d.art3d import Line3D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import snap
import csv

from constants import HOME

xy_figure = plt.figure()
xy_axis = xy_figure.add_subplot(111)
xy_figure.canvas.set_window_title('XY Top-Down View')
xy_background = xy_figure.canvas.copy_from_bbox(xy_axis.bbox)
#xy_axis.set_xlim(-10, 10)
#xy_axis.set_ylim(-10, 10)
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
    x = snap.Trajectory1D(x_wp, der=2)
    y = snap.Trajectory1D(y_wp, der=2)
    z = snap.Trajectory1D(z_wp, der=3)
    psi = snap.Trajectory1D(psi_wp, der=3)

    xy_lines = []

    print('Running optimization routine...')
    trajectory = snap.QrPath(x, y, z, psi, power=10.00, tilt=0.25, guess=5.00)
    T = trajectory.optimize()  # an array of segment time length

    with open(HOME + '/data/output.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        if segment_num == 0:
            temp_x = ["x^"+str(degree) for degree in range(8)]
            temp_y = ["y^"+str(degree) for degree in range(8)]
            temp_z = ["z^"+str(degree) for degree in range(8)]
            temp_yaw = ["yaw^"+str(degree) for degree in range(8)]
            spamwriter.writerow(np.concatenate([['segment'],['duration'], temp_x, temp_y, temp_z, temp_yaw]))
        for i in range(wap_num-1):
            spamwriter.writerow(np.concatenate([[int(segment_num)],[T[i]], x.p[i], y.p[i], z.p[i], psi.p[i]]))

    tdata = np.arange(0, sum(T), 0.1)
    xdata = [x(t) for t in tdata]
    ydata = [y(t) for t in tdata]
    zdata = [z(t) for t in tdata]

    xy_lines.append(Line2D(ydata, xdata, linestyle='-', marker='', color='b'))

    xy_axis.add_line(xy_lines[-1])
    xy_figure.canvas.restore_region(xy_background)
    #xy_axis.draw_artist(xy_lines[-1])
    xy_figure.canvas.blit(xy_axis.bbox)

    wp_t = [0]
    for t in T:
        wp_t.append(wp_t[-1] + t)

plt.show()
