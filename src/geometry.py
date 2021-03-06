from __future__ import division, print_function

from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np
import operator
import math
import json

from constants_debug import VERBOSE_TEXT

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
        '''
        coords_end: tuple
        coords_start: tuple
        '''
        assert(coords_end is not None)
        if coords_start is None:
            self.coords = tuple(coords_end)
        else:
            self.coords = tuple(np.subtract(coords_end, coords_start))

    def __repr__(self, end=''):
        return 'Coords: ' + str(self.coords) + end

    @staticmethod
    def to_dict(v):
        '''
        Return dictionary from Vector v
        '''
        if v is None:
            return None
        return {'coords': v.coords}

    @staticmethod
    def from_dict(v_dict):
        '''
        Return Vector from dictionary v_dict
        '''
        if v_dict is None:
            return None
        coords = v_dict.get('coords')
        if coords is None:
            return None
        return Vector(coords)

    def norm(self):
        '''
        Return magnitude of self
        '''
        return np.linalg.norm(self.coords)

    def normalize(self):
        '''
        Return self after normalizing self
        '''
        self.coords /= self.norm()
        return self

    def unit(self):
        '''
        Return unit Vector of self without changing self
        '''
        u = deepcopy(self)
        u.normalize()
        return u

    @staticmethod
    def angle_between(u, v):
        '''
        Return interior angle (< 180 degree) formed by Vectors u and v
        '''
        assert(u is not None)
        assert(v is not None)
        magnitude_product = u.norm() * v.norm()
        assert(magnitude_product != 0)
        dot_product = np.dot(u.coords, v.coords)
        dot_product_normalized = dot_product / magnitude_product
        # why are we clipping value?
        angle = np.arccos(np.clip((dot_product_normalized), -1.0, 1.0))
        return math.degrees(angle)

class Point(GeometricEntity):
    def __init__(self, coords, tgt_at_point=None):
        '''
        coords: tuple
        tgt_at_point: Vector
        '''
        self.coords = tuple(coords)
        self.tgt = tgt_at_point
        if self.tgt is not None:
            assert len(self.coords) == len(self.tgt.coords), \
                'Point and its tangent have different dimensions'

    def __repr__(self, end='\n', sep='\n'):
        my_string = 'Coord: ' + str(self.coords)
        if False and self.tgt is not None:
            my_string += sep + 'Tgt: ' + str(self.tgt)
        my_string += end
        return my_string

    @staticmethod
    def to_dict(p):
        '''
        Return dictionary from Point p
        '''
        if p is None:
            return None
        p_dict = {'coords': p.coords}
        if p.tgt is not None:
            p_dict['tgt'] = Vector.to_dict(p.tgt)
        return p_dict

    @staticmethod
    def from_dict(p_dict):
        '''
        Return Point from dictionary p_dict
        '''
        if p_dict is None:
            return None
        coords = p_dict.get('coords')
        if coords is None:
            return None
        tgt = p_dict.get('tgt')
        return Point(coords, Vector.from_dict(tgt))

    @staticmethod
    def distance(p, q):
        '''
        Return Euclidean distance between Points p and q
        '''
        a = np.array(p.coords)
        b = np.array(q.coords)
        return np.linalg.norm(a - b)

    @staticmethod
    def section_point(ratio, a, b):
        '''
        Return section Point of line 'ab'

        ratio: section ratio directed from a to b
        a: Point
        b: Point
        '''
        coords1 = np.array(a.coords)
        coords2 = np.array(b.coords)
        section_coords = (coords1 + ratio * coords2) / (1 + ratio)
        return Point(section_coords)

    @staticmethod
    def mid_point(a, b):
        '''Return mid Point of line 'ab'

        a: Point
        b: Point
        '''
        return Point.section_point(1, a, b)

    @staticmethod
    def distance_to_line(p, line):
        '''
        Return perpendicular distance of p to line

        p: Point
        line: tuple containing 2 end Points
        '''
        x0, y0 = p.coords
        x1, y1 = line[0].coords
        x2, y2 = line[1].coords
        x_diff = x2 - x1
        y_diff = y2 - y1
        distance_perpendicular = \
            abs( \
                (y_diff * x0  - x_diff * y0 + x2 * y1 - y2 * x1) \
                / Point.distance(*line)
            )
        return distance_perpendicular

    @staticmethod
    def distance_between_lines(line1_endpoints, line2_endpoints):
        '''
        Returns the distance 2 (possible skew) lines described by their
        end points

        line1_endpoints: tuple of 2 end Points of line 1
        line2_endpoints: tuple of 2 end Points of line 2
        '''
        assert(len(line1_endpoints) == 2)
        assert(len(line2_endpoints) == 2)

        A = np.array(line1_endpoints[0].coords)
        B = np.array(line1_endpoints[1].coords)
        C = np.array(line2_endpoints[0].coords)
        D = np.array(line2_endpoints[1].coords)

        assert(len(A) == len(B) == len(C) == len(D))

        direction_line1 = A - B
        direction_line2 = C - D
        normal = np.cross(direction_line1, direction_line2)
        if np.linalg.norm(normal) == 0:
            # line1 and line2 are coincident
            return 0.0
        AC = A - C
        distance = abs(np.dot(AC, normal) / np.linalg.norm(normal))

        return distance

class Segment(GeometricEntity):
    def __init__(
        self,
        points,
        state=True,
        time=None,
        index=None,
        is_reversed=False
        ):
        '''
        points: list of Points
        state: Boolean - whether self is visible or not
        time: time required for quadrotor to fly over self
        index: index of segment in .csv files
        is_reversed: flag to indicate whether segment is reversed
            with respect to its polynomial
        '''
        self.state = state
        self.points = points[:]
        self.time = (time if time is not None else 0)
        self.index = index
        self.is_reversed = is_reversed

    def __repr__(self, sep='\n', end='\n'):
        return 'Index: ' + str(self.index) + sep \
                + 'Time: ' + str(self.time) + sep \
                + 'Is_reversed: ' + str(self.is_reversed) + sep \
                + 'Length: ' + str(self.length()) + sep \
                + 'State: ' + str(self.state) + sep \
                + 'Pts: ' + str(self.points) + end

    @staticmethod
    def to_dict(s):
        '''
        Return dictionary from Segment s
        '''
        if s is None:
            return None
        s_dict = {'points': [Point.to_dict(p) for p in s.points]}
        if s.state != True:
            s_dict['state'] = s.state
        if s.time is not None:
            s_dict['time'] = s.time
        if s.index is not None:
            s_dict['index'] = s.index
        return s_dict

    @staticmethod
    def from_dict(s_dict):
        '''
        Return Segment from dictionary s_dict
        '''
        if s_dict is None:
            return None
        point_dicts = s_dict.get('points')
        if point_dicts is None:
            return None
        points = [Point.from_dict(p_dict) for p_dict in point_dicts]
        state = s_dict.get('state', True)
        time = s_dict.get('time', None)
        index = s_dict.get('index', None)
        return Segment(points, state, time, index)

    def reverse(self):
        '''
        Reverse order of self's Points
        '''
        self.points.reverse()
        if self.is_reversed is not None:
            self.is_reversed = not self.is_reversed

    def length(self):
        return sum(
            Point.distance(p, self.points[i + 1]) \
            for i, p in enumerate(self.points[:-1])
        )

class Path(GeometricEntity):
    def __init__(self, segments):
        self.segments = segments[:]

    def __repr__(self, sep='\n', end='\n'):
        return 'Time:' + str(self.time()) + sep \
            + 'Length: ' + str(self.length()) + sep \
            + 'Pieces: ' + str(self.pieces()) \
            + '\nSegments:\n' + str(self.segments) + end

    @staticmethod
    def to_dict(p):
        '''
        Return dictionary from Path p
        '''
        if p is None:
            return None
        if p.segments is None:
            return None
        return {'segments': [Segment.to_dict(s) for s in p.segments]}

    @staticmethod
    def from_dict(p_dict):
        '''
        Return Path from dictionary p_dict
        '''
        if p_dict is None:
            return None
        s_dicts = p_dict.get('segments')
        if s_dicts is None:
            return None
        return Path([Segment.from_dict(s_dict) for s_dict in s_dicts])

    def length(self):
        '''
        Return total length of self
        '''
        return sum(s.length() for s in self.segments)

    def time(self):
        '''
        Return total time required for quadrotors to fly over self
        '''
        return sum(s.time for s in self.segments)

    def pieces(self):
        '''
        Return total number of piecewise polynomials in self
        '''
        return sum(len(s.points) - 1 for s in self.segments)

    @staticmethod
    def join(p, q, a, b, velocity=None, pause_time=0, min_gap_time=0):
        '''
        Return Path formed by joining Paths p and q
        at joining Points a and b of p and q respectively
        '''
        p_copy = deepcopy(p)
        if a is p.segments[0].points[0]:
            p_copy.reverse()

        q_copy = deepcopy(q)
        if b is q.segments[-1].points[-1]:
            q_copy.reverse()

        distance = Point.distance(a, b)
        time = distance / velocity
        segments_combined = p_copy.segments
        state = True if time < min_gap_time else False
        segments_combined += [Segment([a, a], state, pause_time, is_reversed=None)]
        segments_combined += [Segment([a, b], state, time, is_reversed=None)]
        segments_combined += [Segment([b, b], state, pause_time, is_reversed=None)]
        segments_combined += q_copy.segments
        return Path(segments_combined)

    def reverse(self):
        '''
        Reverses order of self's Points and Segments
        '''
        self.segments.reverse()
        for s in self.segments:
            s.reverse()

    @staticmethod
    def select_pair(p, q, m):
        '''
        Find best of 4 metric combinations using MetricSurface m on
        end Points of Paths p and q

        Return highest metric and the two corresponding end Points
        '''
        paths = [p, q]
        points_end = [[] for p in paths]
        for path_index, path in enumerate(paths):
            for end_point_index in [0, -1]:
                points_end[path_index].append(path.segments[end_point_index].points[end_point_index])

        index_best_p = None
        index_best_q = None
        metric_best = float("-inf")
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
            passing through Point p and Point q
        '''
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
        point_zero = Point.section_point(ratio, *self.LINE_THETA_ZERO)
        point_end = Point.section_point(ratio, *self.LINE_THETA_END)
        a, b = Hyperbola.find_coefficients(point_zero, point_end, self.curvature)
        # (theta - a) * (metric - b) = curvature
        return self.curvature/(theta - a) + b
