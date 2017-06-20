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

SIZE_OF_QUAD = 0.1 # in m
MAX_QUADROTOR_VELOCITY = 1.5 # in m/s

FINAL_WIDTH = 6.0 # in m
FINAL_HEIGHT = 1.5 # in m

HEIGHT_OFFSET = 1.0 # in m
X_OFFSET = (3 * 0.02 + 1.414 * SIZE_OF_QUAD) # in m
COLLISION_DIST = (3 * 0.02 + 1.414 * SIZE_OF_QUAD) # in m

#SCALING_FACTOR = 0.05
SCALING_FACTOR = 10.0
POST_SCALING_FACTOR = 50.0
