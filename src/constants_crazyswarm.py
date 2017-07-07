BLACK = 0
WHITE = 255

offset_list = [-1,0,1]
DIRECTIONS = [(i,j) for i in offset_list for j in offset_list]
DIRECTIONS.remove((0,0))
del offset_list

SIZE_OF_QUAD = 0.1 # in m
MAX_QUADROTOR_VELOCITY = 0.25 # in m/s
MAX_PIECEWISE_POLYNOMIALS = 30
CAMERA_EXPOSURE_TIME_LIMIT = 20 # in s
#Distance of camera from the first plane
CAMERA_DISTANCE = 1 # in m
CAMERA_Y = 2.5 # in m
CAMERA_Z = 3 # in m

ARENA_WIDTH = 5.0 # in m
ARENA_HEIGHT = 1 # in m

HEIGHT_OFFSET = 1.0 # in m
# X_OFFSET = (3 * 0.02 + 1.414 * SIZE_OF_QUAD) # in m
X_OFFSET = 0.3 # in m
# COLLISION_DIST = (3 * 0.02 + 1.414 * SIZE_OF_QUAD) # in m
COLLISION_DIST = 0.3 # in m

SCALING_FACTOR = 10.0
POST_SCALING_FACTOR = 50.0
TOTAL_SCALING_FACTOR = SCALING_FACTOR * POST_SCALING_FACTOR # in pixel/m

FINAL_WIDTH = ARENA_WIDTH * TOTAL_SCALING_FACTOR # in pixel
FINAL_HEIGHT = ARENA_HEIGHT * TOTAL_SCALING_FACTOR # in pixel

TIME_PER_SEGMENT = 20.0
