# env
HOME = '/home/aditya/repos/light_writing/'

# debug
VERBOSE_IMAGE = False
VERBOSE_TEXT = False

# standard constants
BLACK = 0
WHITE = 255

offset_list = [-1,0,1]
directions = [(i,j) for i in offset_list for j in offset_list]
directions.remove((0,0))

SCALING_FACTOR = 0.05
