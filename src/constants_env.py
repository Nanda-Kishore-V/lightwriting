import os

HOME = os.getcwd()
if '/src' == HOME[-4:]:
    HOME = HOME[:-4]
HOME += '/'
