from __future__ import division, print_function

from abc import ABCMeta, abstractmethod
import numpy as np
import operator
import math
import copy
import json

from constants import (
    VERBOSE_TEXT,
    MAX_QUADROTOR_VELOCITY,
)

class GeometricEntity():
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def to_dict():
        pass

    @abstractmethod
    def from_dict():
        pass

class Vector(GeometricEntity):
    def __init__(self, coords_end, coords_start=None):
        '''coords_end: tuple
            coords_start: tuple or None'''
        if coords_start is None:
            self.coords = tuple(coords_end)
        else:
            self.coords = tuple(map(operator.sub, coords_end, coords_start))

    def __repr__(self, end=''):
        return 'Coord: ' + str(self.coords) + end

    @staticmethod
    def to_dict(v):
        '''Returns a dictionary representation of Vector v which is JSON serializable'''
        if v is None:
            return None
        return {'coords': v.coords}

    @staticmethod
    def from_dict(v_dict):
        '''Returns a Vector v by converting the dictionary representation of Vector provided by v_dict'''
        if v_dict is None:
            return None
        coords = v_dict.get('coords')
        if coords is None:
            return None
        return Vector(coords)

    def norm(self):
        return np.linalg.norm(self.coords)

    def normalize(self):
        self.coords /= self.norm()

    def unit(self):
        '''Returns a unit vector corresponding to self'''
        u = copy.deepcopy(self)
        u.normalize()
        return u

    @staticmethod
    def angle_between(u, v):
        '''Returns interior angle (< 180 degree) formed by Vector u and Vector v'''
        dot_product = np.dot(u.coords, v.coords)
        dot_product_normalized = dot_product / (u.norm() * v.norm())
        # why are we clipping value?
        angle = np.arccos(np.clip((dot_product_normalized), -1.0, 1.0))
        return math.degrees(angle)

class Point(GeometricEntity):
    def __init__(self, coords, tgt_at_point=None):
        '''coords: tuple
            tgt_at_point: Vector'''
        self.coords = tuple(coords)
        self.tgt = tgt_at_point

    def __repr__(self, end='', sep='\t'):
        my_string = 'Coord: ' + str(self.coords)
        if self.tgt is not None:
            my_string += sep + 'Tgt: ' + str(self.tgt)
        my_string += end
        return my_string

    @staticmethod
    def to_dict(p):
        '''Returns a dictionary representation of Point p which is JSON serializable'''
        if p is None:
            return None
        p_dict = {'coords': p.coords}
        if p.tgt is not None:
            p_dict['tgt'] = Vector.to_dict(p.tgt)
        return p_dict

    @staticmethod
    def from_dict(p_dict):
        '''Returns a Point p by converting the dictionary representation of Point provided by p_dict'''
        if p_dict is None:
            return None
        coords = p_dict.get('coords')
        if coords is None:
            return None
        tgt = p_dict.get('tgt')
        return Point(coords, Vector.from_dict(tgt))

    @staticmethod
    def distance(p, q):
        '''Returns Euclidean distance between Point p and Point q'''
        a = np.array(p.coords)
        b = np.array(q.coords)
        return np.linalg.norm(a - b)

    @staticmethod
    def find_section_point(ratio, a, b):
        '''Returns the section point of line segment with end points as
            Point a and Point b using section ratio from a to b'''
        coords1 = np.array(a.coords)
        coords2 = np.array(b.coords)
        section_coords = (coords1 + ratio * coords2) / (1 + ratio)
        return Point(tuple(section_coords))

    @staticmethod
    def distance_to_line(p, line_end_points):
        '''Returns perpendicular distance between line passing through Points line_end_points and the Point p'''
        x0, y0 = p.coords
        x1, y1 = line_end_points[0].coords
        x2, y2 = line_end_points[1].coords
        distance = abs((y2 - y1) * x0  - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return distance

    @staticmethod
    def to_image(points, width=None, height=None):
        x_coords = [p.coords[0] for p in points]
        y_coords = [p.coords[1] for p in points]
        width = max(x_coords)
        height = max(y_coords)

        image = np.zeros((width, height), dtype=np.uint8)
        for p in points:
            image[p.coords] = WHITE

        show_and_destroy('points as image', image)


class Segment(GeometricEntity):
    def __init__(self, points, state=True, time=None, index=None):
        '''Boolean state tells whether segment is visible or not
            points is a list of Points'''
        self.state = state
        self.points = points[:]
        self.time = (time if time is not None else 0)
        self.index = index

    def __repr__(self, end='\n', sep='\t'):
        return 'Time: ' + str(self.time) + 'Length: ' + str(self.length()) + sep + 'State: ' + str(self.state) + sep + 'Pts: ' + str(self.points) + end

    @staticmethod
    def to_dict(s):
        '''Returns a dictionary representation of Segment s which is JSON serializable'''
        if s is None:
            return None
        s_dict = {'points': [Point.to_dict(p) for p in s.points]}
        if s.state != True:
            s_dict['state'] = s.state
        return s_dict

    @staticmethod
    def from_dict(s_dict):
        '''Returns a Segment s by converting the dictionary representation of Segment provided by s_dict'''
        if s_dict is None:
            return None
        point_dicts = s_dict.get('points')
        if point_dicts is None:
            return None
        points = [Point.from_dict(p_dict) for p_dict in point_dicts]
        state = s_dict.get('state', True)
        return Segment(points, state)

    def reverse(self):
        self.points.reverse()

    def length(self):
        return sum([Point.distance(p, self.points[i + 1]) for i, p in enumerate(self.points[:-1])])

class Path(GeometricEntity):
    def __init__(self, segments):
        self.segments = segments[:]

    def __repr__(self):
        return 'Time:' + str(self.time()) + '\nLength: ' + str(self.length()) + '\nSegments:\n' + str(self.segments) + '\n'

    @staticmethod
    def to_dict(p):
        '''Returns a dictionary representation of Path p which is JSON serializable'''
        if p is None:
            return None
        if p.segments is None:
            return None
        return {'segments': [Segment.to_dict(s) for s in p.segments]}

    @staticmethod
    def from_dict(p_dict):
        '''Returns a Path p by converting the dictionary representation of Path provided by p_dict'''
        if p_dict is None:
            return None
        segments = p_dict.get('segments')
        if segments is None:
            return None
        return Path([Segment.from_dict(s_dict) for s_dict in p_dict['segments']])

    def length(self):
        return sum(s.length() for s in self.segments)

    def time(self):
        return sum(s.time for s in self.segments)

    def pieces(self):
        return sum(len(s.points) - 1 for s in self.segments)

    @staticmethod
    def join(p, q, joining_point_p, joining_point_q):
        ''' Joins Path p with Path q and returns a new path'''
        if joining_point_p in p.segments[0].points:
            p.reverse()
        if joining_point_q in q.segments[-1].points:
            q.reverse()

        distance = Point.distance(joining_point_p, joining_point_q)
        time = distance / MAX_QUADROTOR_VELOCITY
        print('intermediate time: ' + str(time))
        segments_combined = p.segments + [Segment([joining_point_p, joining_point_p], False, time)] + q.segments
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
        #metric_best = -1
        metric_best = float("-inf")
        for index_p in range(2):
            for index_q in range(2):
                metric_curr = m.metric(points_end[0][index_p], points_end[1][index_q])
                if VERBOSE_TEXT: print('metric_curr', metric_curr)
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
        x2, _, y2 = p.coords
        x1, _, y1 = q.coords
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

    def metric(self, point_start, point_end):
        dist = Point.distance(point_start, point_end)
        # we want our vectors to be as antiparallel as possible
        theta = 180 - Vector.angle_between(point_start.tgt, point_end.tgt)
        
        ratio = dist / (self.D_MAX - dist)
        point_zero = Point.find_section_point(ratio, *self.LINE_THETA_ZERO)
        point_end = Point.find_section_point(ratio, *self.LINE_THETA_END)
        a, b = Hyperbola.find_coefficients(point_zero, point_end, self.curvature)
        # (theta - a) * (metric - b) = curvature
        return self.curvature/(theta - a) + b
