from __future__ import division, print_function

import cv2
import numpy as np
import json

from skeletonization import get_skeleton
from decimation import decimate
from junction_segmentation import junction_segmentation
from constants import (
    HOME,
    WHITE,
    SCALING_FACTOR,
    VERBOSE_TEXT,
    VERBOSE_IMAGE,
)
from geometry import Point, Segment

def main():
    image = cv2.imread(HOME + 'data/images/black_text.bmp',0)
    width, height = image.shape

    if VERBOSE_IMAGE:
        show_and_destroy('Original Image', image)

    image = get_skeleton(image)
    print('skeletonization done')

    segments = junction_segmentation(image)
    print('junction segmentation done')

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
            new_point = (width - p.coords[0], p.coords[1], 0)
            new_point = tuple([SCALING_FACTOR * x for x in new_point] + [0])
            points.append(Point(new_point))
        scaled_segments.append(Segment(points))

    segment_dicts = [Segment.to_dict(s) for s in scaled_segments]
    with open(HOME + 'data/waypoints.json', 'w') as f:
        json.dump(segment_dicts, f)

if __name__=="__main__":
    main()
