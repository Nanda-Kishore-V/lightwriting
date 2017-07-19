# LightWriting with Crazyswarm

This project generates trajectories for the Crazyswarm to perform light writing. A long exposure photograph will capture the 
given text in the given font.

## Getting started

Clone this repository in the same folder as crazyswarm.

.

├── lightwriting

└── crazyswarm

Create two folders named `traj` and `data` in `./crazyswarm/ros_ws/src/crazyswarm/scripts/`.
Read the modified [crazyswarm repository] for further details.(https://bitbucket.org/nanda-kishore-v/crazyswarm)

### Dependencies

You need to install the following packages in python 2.7: numpy, opencv, scipy, scikit-learn, matplotlib, pypoly, yaml.

Installing pip

```
sudo apt-get install python-pip
```

Installing the packages
```
sudo pip install numpy scipy
sudo pip install -U scikit-learn
sudo pip install matplotlib
sudo pip install PyPolynomial
sudo pip install pyyaml
```

Also install `convert`, a command line utility by imagemagick to convert text to images.

### Usage

Change the TEXT and FONT variables in the `Makefile` and run 

```
make curve_fitting
```

The trajectories will be stored in the traj directory in the crazyswarm workspace.

## Authors

* **Nanda Kishore**
* **Aditya Barve**

