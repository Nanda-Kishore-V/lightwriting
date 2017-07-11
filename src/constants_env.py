import os

HOME = os.getcwd()
if '/src' == HOME[-4:]:
    HOME = HOME[:-4]
HOME += '/'

ROS_WS = HOME + '../crazyswarm/ros_ws/src/crazyswarm/'