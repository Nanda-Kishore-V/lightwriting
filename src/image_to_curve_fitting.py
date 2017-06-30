from __future__ import division, print_function

from scipy.interpolate import UnivariateSpline
import cv2
import numpy as np
import json
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

def image_to_segments(filename):
    image = cv2.imread(filename, 0)
    width, height = image.shape

    if VERBOSE_IMAGE:
        show_and_destroy('Original Image', image)

    image = get_skeleton(image)
    print('skeletonization done')

    segments = junction_segmentation(image)
    print('junction segmentation done')

    return segments

def main():
    segments = image_to_segments(HOME + 'data/images/black_text.bmp')

    print('len of segments', len(segments))
    for s in segments:
        points = s.points

        x = np.array([p.coords[0] for p in points])
        y = np.array([p.coords[1] for p in points])
        t = np.linspace(0, 20, len(points))
        print('t', t.shape)
        print('x', x.shape)
        print('y', y.shape)
        p = np.polyfit(t, zip(x, y), 7)
        print('p', p)

        plt.axis('equal')
        px = [np.poly1d(p[:,0])(time) for time in t]
        py = [np.poly1d(p[:,1])(time) for time in t]
        plt.plot(y, x, '.r', lw=3)
        plt.plot(py, px, '-g', lw=3)
        plt.show()

    '''
    limit = 0.1 * max([len(s.points) for s in segments])
    segments = [Segment(decimate(s.points)) for s in segments if len(s.points) > limit]
    if VERBOSE_TEXT:
        print('after decimate')
        print('len of segments ' + str(len(segments)))

    limit = 0.1 * max([len(s.points) for s in segments])
    segments = [s for s in segments if len(s.points) > limit]
    if VERBOSE_TEXT: print('len of segments after removing small segments ' + str(len(segments)))

    if VERBOSE_TEXT:
        print('after decimate')
        print('segments lengths and # of points')
        for i, s in enumerate(segments):
            print('{}: {}'.format(i, len(s.points)))

    if VERBOSE_IMAGE:
        for index, segment in enumerate(segments):
            image_segment = np.zeros(image.shape)
            for segment in segments:
                for point in segment.points:
                    image_segment[point.coords] = WHITE
                show_and_destroy('Image' + str(index), image_segment)
        cv2.destroyAllWindows()

    # segments = [s for s in segments if len(s.points) > 2]

    scaled_segments = []
    for s in segments:
        points = []
        for p in s.points:
            new_point = (width - p.coords[0] + (HEIGHT_OFFSET * POST_SCALING_FACTOR * SCALING_FACTOR), p.coords[1], 0)
            new_point = tuple([x / SCALING_FACTOR for x in new_point] + [0])
            points.append(Point(new_point))
        scaled_segments.append(Segment(points))

    segment_dicts = [Segment.to_dict(s) for s in scaled_segments]
    with open(HOME + 'data/waypoints.json', 'w') as f:
        json.dump(segment_dicts, f)
    '''

if __name__=="__main__":
    main()
