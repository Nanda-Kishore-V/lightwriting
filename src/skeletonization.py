from skimage.morphology import skeletonize
from skimage.filters import threshold_mean
from skimage.util import invert
from skimage import img_as_ubyte
import cv2

from constants_env import HOME
from constants_debug import VERBOSE_TEXT, VERBOSE_IMAGE
from debug_cv2 import show_and_destroy

def get_skeleton(image):
    threshold = threshold_mean(image)
    if VERBOSE_TEXT: print('threshold: {}'.format(threshold))
    image = image < threshold
    show_and_destroy('image', img_as_ubyte(image))
    skeleton = img_as_ubyte(skeletonize(image))

    show_and_destroy('skeleton', skeleton)
    cv2.imwrite(HOME + 'data/images/skeleton_text.png', skeleton)

    return skeleton

if __name__ == '__main__':
    image = cv2.imread(HOME + 'data/images/black_text.bmp', 0)
    get_skeleton(image)
