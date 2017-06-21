import os

# env
HOME = os.getcwd()
if '/src' == HOME[-4:]:
    HOME = HOME[:-4]
HOME += '/'

# debug
VERBOSE_IMAGE = False
VERBOSE_TEXT = False

# standard constants
BLACK = 0
WHITE = 255

offset_list = [-1,0,1]
DIRECTIONS = [(i,j) for i in offset_list for j in offset_list]
DIRECTIONS.remove((0,0))
del offset_list

MAX_QUADROTOR_VELOCITY = 3 # meter / sec
MAX_PIECEWISE_POLYNOMIALS = 30

CAMERA_EXPOSURE_TIME_LIMIT = 20 # sec

SCALING_FACTOR = 0.1
