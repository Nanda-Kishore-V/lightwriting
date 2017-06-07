import cv2
import numpy as np
import operator
import csv

from skeletonization import get_skeleton
from decimation import decimate
from junction_segmentation import junction_segmentation
from constants import *
from debug import *

def main():
    image = cv2.imread(HOME + 'data/images/black_text.bmp',0)
    width, height = image.shape

    if VERBOSE_IMAGE:
        show_and_destroy('Original Image', image)

    image = get_skeleton(image)
    print 'skeletonization done'

    segments = junction_segmentation(image)
    print 'junction segmentation done'

    segment_lengths = []
    for s in segments:
        consecutive_distances = []
        for i, p in enumerate(s[1:]):
            consecutive_distances.append(np.linalg.norm(tuple(map(operator.sub, p, s[i]))))
        segment_lengths.append(sum(consecutive_distances))
    print 'before decimate seg lengths'
    for s in sorted(segment_lengths):
        print int(s)

    print 'after decimate'
    for s in sorted(segment_lengths):
        print int(s)

    limit = 0.1 * max(segment_lengths)
    segments = [decimate(s) for i, s in enumerate(segments) if segment_lengths[i] >= limit]
    print 'after decimate'
    print 'len of segments', len(segments)

    segment_lengths = []
    for s in segments:
        consecutive_distances = []
        for i, p in enumerate(s[1:]):
            consecutive_distances.append(np.linalg.norm(tuple(map(operator.sub, p, s[i]))))
        segment_lengths.append(sum(consecutive_distances))
    print segment_lengths

    limit = 0.1 * max(segment_lengths)
    segments = [s for i, s in enumerate(segments) if segment_lengths[i] >= limit]
    print 'len of segments after removing small segments', len(segments)

    print 'after decimate seg lengths', segment_lengths
    print 'segment lengths and # of points'
    for i, s in enumerate(segments):
        print i, segment_lengths[i], s

    for index,segment in enumerate(segments):
        image_segment = np.zeros((image.shape[0],image.shape[1]))
        for p in segment:
            image_segment[p] = WHITE
        if VERBOSE_IMAGE:
            show_and_destroy('Image' + str(index), image_segment)

    if VERBOSE_IMAGE:
        cv2.destroyAllWindows()

    segments = [s for s in segments if len(s) > 2]
    with open(HOME + "data/waypoints.csv", 'wb') as myfile:
        wr = csv.writer(myfile)
        wr.writerow([str(len(segments)), "x", "y", "z", "yaw"])
        for segment_index, segment in enumerate(segments):
            for point_index, point in enumerate(segment):
                wr.writerow([int(segment_index), SCALING_FACTOR * int(width - point[0]), SCALING_FACTOR * int(point[1]), SCALING_FACTOR * int(0), int(0)])


if __name__=="__main__":
    main()
