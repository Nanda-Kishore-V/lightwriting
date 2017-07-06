from __future__ import division, print_function
import cv2
import numpy as np

from debug_cv2 import show_and_destroy, show_and_wait

from constants import (
    HOME,
    DIRECTIONS,
    WHITE,
    BLACK,
    VERBOSE_TEXT,
    VERBOSE_IMAGE,
)
from geometry import (
    Point,
    Segment,
)

def find_white_neighbors(image, point):
    x, y = point
    width, height = image.shape
    return [
        (x + dx, y + dy)
        for dx, dy in DIRECTIONS
        if 0 <= x + dx < width
        and 0 <= y + dy < height
        and image[x + dx, y + dy] == WHITE
    ]

def is_corner(image, point):
    return len(find_white_neighbors(image, point)) == 1

def return_corner(image, points):
    for p in points:
        if is_corner(image, p):
            return p
    return None

def reinitialize(image, points_white):
    point_next = return_corner(image, points_white)
    segment_curr = []
    if point_next is None and points_white:
        point_next = points_white[0]
    return point_next, segment_curr

def segmentation(image):
    img = image.copy()
    width, height = img.shape

    points_white = [
        (i, j)
        for i in range(width)
        for j in range(height)
        if img[i, j] == WHITE
    ]
    points_white_to_be_removed = [
        p
        for p in points_white
        if not 1 <= len(find_white_neighbors(img, p)) < 3
    ]
    for p in points_white_to_be_removed:
        img[p] = BLACK
        points_white.remove(p)

    # why doesn't this show the image?
    if VERBOSE_IMAGE:
        show_and_destroy('junctions removed image', img)

    segments = []
    point_next, segment_curr = reinitialize(img, points_white)
    if point_next is None:
        print('Something really wrong!')
        exit()
    while points_white:
        point_curr = point_next
        segment_curr.append(point_curr)
        neighbors = find_white_neighbors(img, point_curr)
        points_white.remove(point_curr)
        img[point_curr] = BLACK
        if len(neighbors) == 0:
            segments.append(segment_curr)
            point_next, segment_curr = reinitialize(img, points_white)
        else:
            point_next = neighbors[0]

    if VERBOSE_IMAGE:
        for index, segment in enumerate(segments):
            image_segment = np.zeros(image.shape)
            for p in segment:
                image_segment[p] = WHITE
            show_and_destroy("Image"+str(index),image_segment)

    # above this line, segments are just lists of tuples
    # now, segments become a list of Segment objects, each of which contains Point objects
    segments = [Segment([Point(p) for p in s]) for s in segments]
    if VERBOSE_TEXT:
        print('segments')
        print(*segments, sep='\n')

    # now we need to detect which of the segments contain straight lines
    # and split those lines into segments of their own
    '''
    refined_segments = []
    for s in segments:
        refined_segments += segmentation_straight_lines(s)
    '''
    return segments

def segmentation_straight_lines(points):
    '''
    points: list of Points

    Splits segment into a list of Segments, each of which is either a
    straight line, or a curve.

    Returns list of Segments
    '''
    pass

def main():
    image = cv2.imread(HOME + 'data/images/skeleton_text.png', 0)
    width, height = image.shape
    if VERBOSE_IMAGE:
        show_and_destroy('original image', image)

    points_white = [(i, j) for i in range(width) for j in range(height) if image[i, j] == WHITE]
    segments = segmentation(image)
    for index, s in enumerate(segments):
        for wp in points_white:
            if wp in s:
                image[wp] = WHITE
            else:
                image[wp] = BLACK
        if VERBOSE_IMAGE: show_and_wait('segment #' + str(index), image)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
