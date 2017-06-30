from __future__ import division, print_function

import cv2
import numpy as np
import json
import csv
import matplotlib.pyplot as plt

from skeletonization import get_skeleton
from decimation import decimate
from junction_segmentation import junction_segmentation
from constants import (
    HOME,
    WHITE,
    SCALING_FACTOR,
    VERBOSE_TEXT,
    VERBOSE_IMAGE,
    HEIGHT_OFFSET,
    SCALING_FACTOR,
    POST_SCALING_FACTOR,
)
from geometry import Point, Segment

TIME_PER_SEGMENT = 20.0

def image_to_segments(filename):
    image = cv2.imread(filename, 0)
    width, height = image.shape

    if VERBOSE_IMAGE:
        show_and_destroy('Original Image', image)

    image = get_skeleton(image)
    print('skeletonization done')

    segments = junction_segmentation(image)
    print('junction segmentation done')

    limit = 0.1 * max([len(s.points) for s in segments])
    segments = [s for s in segments if len(s.points) > limit]

    return segments, width, height

def main():
    # segments, width, height = image_to_segments(HOME + 'data/images/black_text.bmp')
    filename = HOME + 'data/images/black_text.bmp'
    image = cv2.imread(filename, 0)
    width, height = image.shape

    if VERBOSE_IMAGE:
        show_and_destroy('Original Image', image)

    image = get_skeleton(image)
    print('skeletonization done')

    segments = junction_segmentation(image)
    print('junction segmentation done')

    limit = 0.1 * max([len(s.points) for s in segments])
    segments = [s for s in segments if len(s.points) > limit]
    num_segments = len(segments)
    print("Number of segments: {}".format(len(segments)))

    out_file = open(HOME + "data/output.csv", "w")
    output_writer = csv.writer(out_file)
    temp_x = ["x^"+str(degree) for degree in range(8)]
    temp_y = ["y^"+str(degree) for degree in range(8)]
    temp_z = ["z^"+str(degree) for degree in range(8)]
    temp_yaw = ["yaw^"+str(degree) for degree in range(8)]
    output_writer.writerow(np.concatenate([[int(num_segments)],['duration'], temp_x, temp_y, temp_z, temp_yaw]))

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
        x = np.array([(width - p.coords[0]) / SCALING_FACTOR + (HEIGHT_OFFSET * POST_SCALING_FACTOR) for p in points])
        y = np.array([p.coords[1] / SCALING_FACTOR for p in points])

        t = np.linspace(0, TIME_PER_SEGMENT, len(points))
        p = np.polyfit(t, zip(x, y), 7)

        poly1d_x = np.poly1d(p[:,0])
        poly1d_y = np.poly1d(p[:,1])
        plt.plot(y, x, '.y', lw=3)
        plt.plot([poly1d_y(time) for time in t], [poly1d_x(time) for time in t], '-g', lw=3)

        output_writer.writerow(np.concatenate([[int(segment_num)], [TIME_PER_SEGMENT], p[:,0][::-1], p[:,1][::-1], [0.0] * 8, [0.0] * 8]))

        start_pt = [poly1d_x(0), poly1d_y(0), 0]
        end_pt = [poly1d_x(20), poly1d_y(20), 0]
        start_vector = [np.polyder(poly1d_x)(0), np.polyder(poly1d_y)(0), 0]
        end_vector = [np.polyder(poly1d_x)(20), np.polyder(poly1d_y)(20), 0]
        tangent_writer.writerow(np.concatenate([[TIME_PER_SEGMENT], start_pt, start_vector, end_pt, end_vector]))

        plt.annotate(str(segment_num), xy=(start_pt[1], start_pt[0]), xytext=(start_pt[1] + 1, start_pt[0] + 1),
            arrowprops=dict(facecolor='white', shrink=0.05))
        plt.quiver(start_pt[1], start_pt[0], start_vector[1], start_vector[0], color='r')
        plt.quiver(end_pt[1], end_pt[0], end_vector[1], end_vector[0], color='b')

    out_file.close()
    tangent_file.close()
    plt.show()

if __name__=="__main__":
    main()
