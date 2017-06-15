from __future__ import division

import numpy as np
import operator
import math
import copy

class Vector():
    def __init__(self, coordinates_end, coordinates_start=None):
        '''coordinates_end: tuple
            coordinates_start: tuple or None'''
        if coordinates_start is None:
            self.coordinates = tuple(coordinates_end)
        else:
            self.coordinates = tuple(map(operator.sub, coordinates_end, coordinates_start))

    def __repr__(self, end=''):
        return 'Coord: ' + str(self.coordinates) + end

    @staticmethod
    def to_dict(v):
        '''Returns a dictionary representation of Vector v which is JSON serializable'''
        if v is None:
            return None
        return {'coordinates': v.coordinates}

    @staticmethod
    def from_dict(v_dict):
        '''Returns a Vector v by converting the dictionary representation of Vector provided by v_dict'''
        if v_dict is None:
            return None
        return Vector(v_dict['coordinates'])

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
        self.coordinates = tuple(coordinates)
        self.tangent = tangent_at_point

    def __repr__(self, end='', sep='\t'):
        return 'Coord: ' + str(self.coordinates) + sep + 'Tgt: ' + str(self.tangent) + end

    @staticmethod
    def to_dict(p):
        '''Returns a dictionary representation of Point p which is JSON serializable'''
        if p is None:
            return None
        return {'coordinates': p.coordinates, 'tangent': Vector.to_dict(p.tangent)}

    @staticmethod
    def from_dict(p_dict):
        '''Returns a Point p by converting the dictionary representation of Point provided by p_dict'''
        if p_dict is None:
            return None
        return Point(p_dict['coordinates'], Vector.from_dict(p_dict['tangent']))

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
        self.length = sum([Point.distance(p, points[i + 1]) for i, p in enumerate(points[:-1])])

    def __repr__(self, end='\n', sep='\t'):
        return 'State: ' + str(self.state) + sep + 'Pts: ' + str(self.points) + end

    @staticmethod
    def to_dict(s):
        '''Returns a dictionary representation of Segment s which is JSON serializable'''
        if s is None:
            return None
        return {'state': s.state, 'points': [Point.to_dict(p) for p in s.points]}

    @staticmethod
    def from_dict(s_dict):
        '''Returns a Segment s by converting the dictionary representation of Segment provided by s_dict'''
        if s_dict is None:
            return None
        return Segment(s_dict['state'], [Point.from_dict(p_dict) for p_dict in s_dict['points']])

    def reverse(self):
        self.points.reverse()

class Path():
    def __init__(self, segments):
        self.segments = segments[:]
        self.length = sum([s.length for s in segments])

    def __repr__(self):
        return 'Length: ' + str(self.length) + '\nSegments:\n' + str(self.segments) + '\n'

    @staticmethod
    def to_dict(p):
        '''Returns a dictionary representation of Path p which is JSON serializable'''
        if p is None:
            return None
        return {'segments': [Segment.to_dict(s) for s in p.segments]}

    @staticmethod
    def from_dict(p_dict):
        '''Returns a Path p by converting the dictionary representation of Path provided by p_dict'''
        if p_dict is None:
            return None
        return Path([Segment.from_dict(s_dict) for s_dict in p_dict['segments']])

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
