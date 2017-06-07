import cv2
import numpy as np
import operator

from constants import *

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

def reinitialize(image, white_points):
    next_point = return_corner(image, white_points)
    curr_segment = []
    if next_point is None and white_points:
        next_point = white_points[0]
    return next_point, curr_segment

def junction_segmentation(image):
    img = image.copy()
    width, height = img.shape

    white_points = [(i, j) for i in range(width) for j in range(height) if img[i, j] == WHITE]
    white_points_to_be_removed = [p for p in white_points if not 0 < len(find_white_neighbors(img, p)) <= 2]
    for p in white_points_to_be_removed:
        img[p] = BLACK
        white_points.remove(p)

    if VERBOSE_IMAGE: show_and_wait('junctions removed image', img)

    segments = []
    next_point, curr_segment = reinitialize(img, white_points)
    if next_point is None:
        print 'Something really wrong!'
        exit()
    while white_points:
        curr_point = next_point
        curr_segment.append(curr_point)
        neighbors = find_white_neighbors(img, curr_point)
        white_points.pop(white_points.index(curr_point))
        img[curr_point] = BLACK
        if len(neighbors) == 0:
            segments.append(curr_segment)
            next_point, curr_segment = reinitialize(img, white_points)

        else:
            next_point = neighbors[0]

    for index, segment in enumerate(segments):
        image_segment = np.zeros((image.shape[0],image.shape[1]))
        for p in segment:
            image_segment[p[0]][p[1]] = WHITE
        with open(HOME + 'data/segments/segment' + str(index), 'w') as f:
            for p in segment:
                f.write(str(p))
                f.write('\n')
        if VERBOSE_IMAGE: show_and_destroy("Image"+str(index),image_segment)
    cv2.destroyAllWindows()

    return segments

def main():
    image = cv2.imread(HOME + 'data/images/skeleton_text.png', 0)
    width, height = image.shape
    if VERBOSE_IMAGE: show_and_wait('original image', image)

    white_points = [(i, j) for i in range(width) for j in range(height) if image[i, j] == WHITE]
    segments = junction_segmentation(image)
    for index, s in enumerate(segments):
        for wp in white_points:
            if wp in s:
                image[wp] = WHITE
            else:
                image[wp] = BLACK
        if VERBOSE_IMAGE: show_and_wait('segment #' + str(index), image)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
