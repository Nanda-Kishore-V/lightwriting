from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import operator

from constants import *
from geometry import Point, Segment

def find_white_neighbors(image, point):
    x, y = point
    width, height = image.shape
    return [(x + dx, y + dy) for dx, dy in directions if 0 <= x + dx < width and 0 <= y + dy < height and image[x + dx, y + dy] == WHITE]

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

def junction_segmentation(image):
    img = image.copy()
    width, height = img.shape

    points_white = [(i, j) for i in range(width) for j in range(height) if img[i, j] == WHITE]
    points_white_to_be_removed = [p for p in points_white if not 1 <= len(find_white_neighbors(img, p)) < 3]
    for p in points_white_to_be_removed:
        img[p] = BLACK
        points_white.remove(p)

    if VERBOSE_IMAGE: show_and_wait('junctions removed image', img)

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

    for index, segment in enumerate(segments):
        image_segment = np.zeros(image.shape)
        for p in segment:
            image_segment[p] = WHITE
        if VERBOSE_IMAGE: show_and_destroy("Image"+str(index),image_segment)
    cv2.destroyAllWindows()

    # above this line, segments are just lists of tuples
    # now, segments become a list of Segment objects, each of which contains Point objects
    segments = [Segment([Point(p) for p in s]) for s in segments]
    print('segments')
    print(*segments, sep='\n')
    return segments

def main():
    image = cv2.imread(HOME + 'data/images/skeleton_text.png', 0)
    width, height = image.shape
    if VERBOSE_IMAGE: show_and_wait('original image', image)

    points_white = [(i, j) for i in range(width) for j in range(height) if image[i, j] == WHITE]
    segments = junction_segmentation(image)
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
