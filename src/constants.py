import os

# env
HOME = os.getcwd() 
if '/src' == HOME[-4:]:
    HOME = HOME[:-4]
HOME = HOME + '/'
print('home', HOME)

# debug
VERBOSE_IMAGE = False
VERBOSE_TEXT = False

# standard constants
BLACK = 0
WHITE = 255

offset_list = [-1,0,1]
directions = [(i,j) for i in offset_list for j in offset_list]
directions.remove((0,0))

#SCALING_FACTOR = 0.05
SCALING_FACTOR = 0.25
