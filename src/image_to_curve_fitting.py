from __future__ import division, print_function

import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar
from skeletonization import get_skeleton
from segmentation import segmentation
from constants_env import HOME
from constants_debug import VERBOSE_TEXT, VERBOSE_IMAGE
from constants_crazyswarm import (
    WHITE,
    HEIGHT_OFFSET,
    POST_SCALING_FACTOR,
    SCALING_FACTOR,
    MAX_QUADROTOR_VELOCITY,
    CAMERA_EXPOSURE_TIME_LIMIT,
)
from debug_cv2 import show_and_destroy
from geometry import Point, Segment

TIME_PER_SEGMENT = CAMERA_EXPOSURE_TIME_LIMIT

def image_to_segments(filename):
    image = cv2.imread(filename, 0)
    width, height = image.shape

    if VERBOSE_IMAGE:
        show_and_destroy('Original Image', image)

    image = get_skeleton(image)
    print('skeletonization done')

    segments = segmentation(image)
    print('junction segmentation done')

    limit = 0.1 * max([len(s.points) for s in segments])
    segments = [s for s in segments if len(s.points) > limit]

    return segments, width, height

def main():
    filename = HOME + 'data/images/black_text.bmp'
    segments, width, height = image_to_segments(filename)
    print("Number of segments: {}".format(len(segments)))

    out_file = open(HOME + "data/output.csv", "w")
    output_writer = csv.writer(out_file)
    temp_x = ["x^"+str(degree) for degree in range(8)]
    temp_y = ["y^"+str(degree) for degree in range(8)]
    temp_z = ["z^"+str(degree) for degree in range(8)]
    temp_yaw = ["yaw^"+str(degree) for degree in range(8)]
    output_writer.writerow(np.concatenate([[len(segments)],['duration'], temp_x, temp_y, temp_z, temp_yaw]))

    tangent_file = open(HOME + "data/tangents.csv", "w")
    tangent_writer = csv.writer(tangent_file)
    axes = ['x', 'y', 'z']
    start = ['start ' + i for i in axes]
    s_vector = ['start vector ' + i for i in axes]
    end = ['end ' + i for i in axes]
    e_vector = ['end vector ' + i for i in axes]
    tangent_writer.writerow(np.concatenate([['time per segment'], start, s_vector, end, e_vector]))

    plt.figure(1)
    plt.axis('equal')
    for segment_num, s in enumerate(segments):
        print("Segment {} is being optimized.".format(segment_num))
        points = s.points
        x = np.array([(width - p.coords[0]) / (SCALING_FACTOR * POST_SCALING_FACTOR) + (HEIGHT_OFFSET) for p in points])
        y = np.array([-1*(p.coords[1] - (height/2)) / (SCALING_FACTOR * POST_SCALING_FACTOR) for p in points])

        t = np.linspace(0, TIME_PER_SEGMENT, len(points))
        p = np.polyfit(t, zip(x, y), 7)
        poly1d_x = np.poly1d(p[:,0])
        poly1d_y = np.poly1d(p[:,1])

        dx = np.polyder(poly1d_x)
        dy = np.polyder(poly1d_y)
        velocity_sq = dx * dx + dy * dy

        duration = TIME_PER_SEGMENT
        res = minimize_scalar(-1 * velocity_sq, bounds=(0, TIME_PER_SEGMENT), method='bounded')
        max_velocity = velocity_sq(res.x)**0.5

        duration = TIME_PER_SEGMENT * max_velocity/MAX_QUADROTOR_VELOCITY
        scaling_polynomial = np.poly1d([MAX_QUADROTOR_VELOCITY/max_velocity, 0])
        poly1d_x = np.polyval(poly1d_x, scaling_polynomial)
        poly1d_y = np.polyval(poly1d_y, scaling_polynomial)
        t = np.linspace(0, duration, len(points))

        output_writer.writerow(np.concatenate([[int(segment_num)], [duration], np.array(poly1d_x)[::-1], np.array(poly1d_y)[::-1], [0.0] * 8, [0.0] * 8]))

        # destination_x = 0.5
        # end_x = poly1d_x(duration)
        # end_y = poly1d_y(duration)
        # time_end_segment = abs(destination_x - end_x) / MAX_QUADROTOR_VELOCITY
        # piece_x = [end_x, np.sign(destination_x - end_x) * MAX_QUADROTOR_VELOCITY] + [0.0] * 6
        # piece_y = [end_y] + [0.0] * 7
        #
        # output_writer.writerow(np.concatenate([[int(segment_num)], [time_end_segment], piece_x, piece_y, [0.0] * 16]))

        start_pt = [poly1d_x(0), poly1d_y(0), 0]
        end_pt = [poly1d_x(duration), poly1d_y(duration), 0]
        start_vector = [np.polyder(poly1d_x)(0), np.polyder(poly1d_y)(0), 0]
        end_vector = [-1*np.polyder(poly1d_x)(duration), -1*np.polyder(poly1d_y)(duration), 0]
        tangent_writer.writerow(np.concatenate([[duration], start_pt, start_vector, end_pt, end_vector]))

        plt.plot(y, x, '.y', lw=3)
        plt.plot([poly1d_y(time) for time in t], [poly1d_x(time) for time in t], '-g', lw=3)
        plt.annotate(str(segment_num), xy=(start_pt[1], start_pt[0]), xytext=(start_pt[1] + 0.05, start_pt[0] + 0.05),
            arrowprops=dict(facecolor='white', shrink=0.05))
        plt.quiver(start_pt[1], start_pt[0], start_vector[1], start_vector[0], color='r')
        plt.quiver(end_pt[1], end_pt[0], end_vector[1], end_vector[0], color='b')

    out_file.close()
    tangent_file.close()
    plt.show()

if __name__=="__main__":
    main()
