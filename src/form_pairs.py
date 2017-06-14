from __future__ import division

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import operator
import math
import copy

from constants import HOME

class Vector():
    def __init__(self, coordinates_end, coordinates_start=None):
        '''coordinates_end: tuple
            coordinates_start: tuple or None'''
        if coordinates_start is None:
            self.coordinates = coordinates_end
        else:
            self.coordinates = tuple(map(operator.sub, coordinates_end, coordinates_start))

    def __repr__(self):
        return 'Coord: ' + str(self.coordinates) + '\n'

    def norm(self):
        return np.linalg.norm(self.coordinates)

    def normalize(self):
        self.coordinates /= self.norm()

    def unit(self):
        '''Returns a unit vector corresponding to self'''
        u = copy.deepcopy(self)
        u.normalize()
        return u

    @staticmethod
    def angle_between(u, v):
        '''Returns interior angle (< 180 degree) formed by Vector u and Vector v'''
        dot_product = np.dot(u.coordinates, v.coordinates)
        dot_product_normalized = dot_product / (u.norm() * v.norm())
        # why are we clipping value?
        angle = np.arccos(np.clip((dot_product_normalized), -1.0, 1.0))
        return math.degrees(angle)

class Point():
    def __init__(self, coordinates, tangent_at_point=None):
        '''coordinates: tuple
            tangent_at_point: Vector'''
        self.coordinates = coordinates
        self.tangent = tangent_at_point

    def __repr__(self):
        return 'Coord: ' + str(self.coordinates) + '\nTgt: ' + str(self.tangent) + '\n'

    @staticmethod
    def distance(p, q):
        '''Returns Euclidean distance between Point p and Point q'''
        a = np.array(p.coordinates)
        b = np.array(q.coordinates)
        return np.linalg.norm(a - b)

    @staticmethod
    def find_section_point(ratio, a, b):
        '''Returns the section point of line segment with end points as
            Point a and Point b using section ratio from a to b'''
        coordinates1 = np.array(a.coordinates)
        coordinates2 = np.array(b.coordinates)
        section_coordinates = (coordinates1 + ratio * coordinates2) / (1 + ratio)
        return Point(tuple(section_coordinates))

class Segment():
    def __init__(self, state, points):
        '''Boolean state tells whether segment is visible or not
            points is a list of 2 Point'''
        self.state = state
        self.points = points[:]
        self.length = Point.distance(*points) 

    def __repr__(self):
        return 'State: ' + str(self.state) + '\nPts:\n' + str(self.points) + '\n'

    def reverse(self):
        self.points.reverse()

class Path():
    def __init__(self, segments):
        self.segments = segments[:]
        self.length = sum([s.length for s in segments])

    def __repr__(self):
        return 'Length: ' + str(self.length) + '\nSegments:\n' + str(self.segments) + '\n'

    @staticmethod
    def join(p, q, joining_point_p, joining_point_q):
        ''' Joins Path p with Path q and returns a new path'''
        if joining_point_p in p.segments[0].points:
            p.reverse()
        if joining_point_q in q.segments[-1].points:
            q.reverse()
        segments_combined = p.segments + [Segment(False, [joining_point_p, joining_point_p])] + q.segments
        return Path(segments_combined)

    def reverse(self):
        '''Reverses the list of segments and reverses each segment'''
        self.segments.reverse()
        for s in self.segments:
            s.reverse()

    @staticmethod
    def select_pair(p, q, m):
        '''Finds 4 metrics of Path p and Path q using MetricSurface m'''
        paths = [p, q]
        points_end = [[] for p in paths]
        for path_index, path in enumerate(paths):
            for end_point_index in [0, -1]:
                points_end[path_index].append(path.segments[end_point_index].points[end_point_index])


        index_best_p = None
        index_best_q = None
        metric_best = -1
        for index_p in range(2):
            for index_q in range(2):
                metric_curr = m.metric(points_end[0][index_p], points_end[1][index_q])
                if metric_curr > metric_best:
                    metric_best = metric_curr
                    index_best_p = index_p
                    index_best_q = index_q

        return metric_best, (points_end[0][index_best_p], points_end[1][index_best_q])




        points_end = path_smallest.endpoints()
        left[1] = tuple(path_smallest[2:4])
        left[2] = tuple(path_smallest[2])
        left[2] = tuple([-1 * x for x in left[2]]) # tangent is now facing outwards
        right[1] = tuple(path_smallest[4:6])
        right[2] = tuple(path_smallest[4])
        right[2] = tuple([-1 * x for x in right[2]]) # tangent is now facing outwards
        small = left, right


        pass

class Hyperbola():
    @staticmethod
    def find_coefficients(p, q, curvature):
        '''Finds the coefficients of a hyperbola of the form 
            (x - x_offset) * (y - y_offset) = curvature
            passing through Point p and Point q'''
        x2, _, y2 = p.coordinates
        x1, _, y1 = q.coordinates
        b = -1 * (y1 + y2)
        a = 1
        c = y1 * y2 + (curvature * (y1 - y2)) / (x1 - x2)
        y_offset = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        x_offset = (y_offset * (x2 - x1) + x1 * y1 - x2 * y2) / (y1 - y2)
        return x_offset, y_offset

class MetricSurface():
    def __init__(self):
        self.D_MAX = 100
        self.THETA_END = 360
        self.COST_SCALING = 10
        self.LINE_THETA_ZERO = (Point((0, 0, 2 * self.COST_SCALING)), Point((0, self.D_MAX, 1.5 * self.COST_SCALING)))
        self.LINE_THETA_END = (Point((self.THETA_END, 0, 1 * self.COST_SCALING)), Point((self.THETA_END, self.D_MAX, 0 * self.COST_SCALING)))
        self.curvature = 200 # curvature of hyperbolae which form the surface

    def __repr__(self):
        pass

    def metric(self, point_start, point_end):
        dist = Point.distance(point_start, point_end)
        # we want our vectors to be as antiparallel as possible
        theta = 180 - Vector.angle_between(point_start.tangent, point_end.tangent)
        
        ratio = dist / (self.D_MAX - dist)
        point_zero = Point.find_section_point(ratio, *self.LINE_THETA_ZERO)
        point_end = Point.find_section_point(ratio, *self.LINE_THETA_END)
        a, b = Hyperbola.find_coefficients(point_zero, point_end, self.curvature)
        # (theta - a) * (metric - b) = curvature
        return self.curvature/(theta - a) + b 

def main():
    f = open(HOME + "data/tangents.csv", "rb")
    data = np.genfromtxt(f, dtype= float, delimiter=',')
    data = np.delete(data, (0), axis = 0)

    #index, distance between end points, end point 1, tangent 1, end point 2, tangent 2
    paths = []
    for index, d in enumerate(data):
        temp_data = {}
        temp_data['point1'] = tuple(d[1:4])
        temp_data['tangent1'] = Vector(tuple(d[4:7]))
        temp_data['point2'] = tuple(d[7:10])
        temp_data['tangent2'] = Vector(tuple(d[10:13]))

        start = Point(temp_data['point1'], temp_data['tangent1'])
        end = Point(temp_data['point2'], temp_data['tangent2'])
        s = Segment(True, [start, end])
        p = Path([s])
        paths.append(p)

    '''
    print 'paths before sorting:'
    for p in paths:
        print p
    '''
    paths.sort(key = lambda p: p.length)
    '''
    print 'paths after sorting:'
    for p in paths:
        print p
    
    print '-' * 80
    '''

    m = MetricSurface()

    # form pairs now
    n_required_paths = 2

    while len(paths) > n_required_paths:
        path_smallest = paths.pop(0)
        metric_highest = -1
        index_best = None
        points_end_best = None
        for i, p in enumerate(paths):
            metric_curr, points_end_candidate = Path.select_pair(path_smallest, p, m)
            if metric_curr > metric_highest:
                index_best = i
                metric_highest = metric_curr
                points_end_best = points_end_candidate

        path_new = Path.join(path_smallest, paths[index_best], *points_end_best)
        del paths[index_best]
        # optimize later to insert path_new into correct position
        paths.append(path_new)
        paths.sort(key = lambda p: p.length)

    '''
    print 'paths after combining:'
    for p in paths:
        print p
    '''

    print 'about to plot'
    for i, p in enumerate(paths):
        x = []
        y = []
        for s in p.segments:
            x.append(s.points[0].coordinates[0])
            x.append(s.points[1].coordinates[0])
            y.append(s.points[0].coordinates[1])
            y.append(s.points[1].coordinates[1])
        plt.figure(i)
        plt.scatter(x, y, c='r')
        plt.xlim([0, 100])
        plt.ylim([0, 100])
        plt.show()
    print 'done plotting'

if __name__ == "__main__":
    main()
