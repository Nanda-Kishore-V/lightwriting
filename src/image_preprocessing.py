from __future__ import division, print_function

import cv2
from PIL import Image
import numpy as np

from constants import (
    HOME,
    FINAL_WIDTH,
    FINAL_HEIGHT,
    TOTAL_SCALING_FACTOR,
)

def main():
    image = cv2.imread(HOME + 'data/images/temporary.bmp', 0)
    (thresh, image) = cv2.threshold(image, 128, 255, cv2.THRESH_OTSU)

    # scale image to fit warehouse flying area
    max_size = (TOTAL_SCALING_FACTOR * FINAL_WIDTH, \
            TOTAL_SCALING_FACTOR * FINAL_HEIGHT)
    image = Image.fromarray(image).thumbnail(max_size, Image.ANTIALIAS)
    image = np.asarray(image, dtype=np.uint8)

    cv2.imwrite(HOME + 'data/images/black_text.bmp', image)

if __name__ == '__main__':
    main()
