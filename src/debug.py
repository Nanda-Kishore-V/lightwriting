import cv2
import operator
import numpy as np

from constants import *

def show_and_wait(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)

def show_and_destroy(title, image):
    show_and_wait(title, image)
    cv2.destroyWindow(title)

def show_points_as_image(points):
    width = max([p[0] for p in points]) + 1
    height = max([p[1] for p in points]) + 1
    image = np.zeros((width, height), dtype=np.uint8)
    for p in points:
        image[p] = WHITE

    show_and_destroy('points as image', image)
