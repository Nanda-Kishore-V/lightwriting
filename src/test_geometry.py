from __future__ import division, print_function

from geometry import (
    Vector,
    Point,
    Segment,
    Path,
    MetricSurface,
)

def test_vector():
    a = Vector((3, 5))
    print('a', a)
    a_dict = Vector.to_dict(a)
    print('a_dict', a_dict)

    b = Vector.from_dict(a_dict)
    print('b', b)

    c = Vector((2, 7), (0, 1))
    print('c', c)
    c_dict = Vector.to_dict(c)
    print('c_dict', c_dict)

    d = Vector.from_dict(c_dict)
    print('d', d)
    print('d.norm', d.norm())
    print('d.unit().norm', d.unit().norm())

    x = Vector((1, 0))
    y = Vector((0, 1))
    print('angle between', Vector.angle_between(x, y))

def test_point():
    x = Point((1,2,3), Vector((1,2,1)))
    p = Point((0, 3))
    print('p', p)

    q = Point((4, 0), Vector((-1, 0)))
    print('q', q)
    print('distance', Point.distance(p, q))

    print('distance', Point.distance(Point((1,2,3,4)), Point((2,3,4,5))))

    p_dict = Point.to_dict(p)
    print('p_dict', p_dict)
    p_new = Point.from_dict(p_dict)
    print('p_new', p_new)

    q_dict = Point.to_dict(q)
    print('q_dict', q_dict)
    q_new = Point.from_dict(q_dict)
    print('q_new', q_new)

    print('section_point', Point.section_point(2/3, Point((5, 5)), Point((15, 15))))

    print('mid_point', Point.mid_point(p, q))

    print('distance_to_line', Point.distance_to_line((Point((1, 1))), (Point((0, 0)), Point((2, 0)))))

    points = [
            Point((0, 0)),
            Point((0, 2)),
            Point((2, 2)),
            Point((2, 0)),
            ]
    Point.to_image(points)

def test_segment():
    points = [
            Point((0, 0)),
            Point((0, 2)),
            Point((2, 2)),
            Point((2, 0)),
            ]
    s = Segment(points)
    print('s', s)

    s_dict = Segment.to_dict(s)
    print('s_dict', s_dict)

    s_new = Segment.from_dict(s_dict)
    print('s_new', s_new)

    print('length', s_new.length())

def test_path():
    points = [
            Point((0, 2), Vector((0, -1))),
            Point((0, 0), Vector((0, 1))),
            ]
    s1 = Segment(points, time=4, index=0)
    points = [
            Point((3, 5), Vector((-1, -1))),
            Point((1, 3), Vector((1, 1))),
            ]
    s2 = Segment(points, time=3, index=1)

    p1 = Path([s1])
    print('p1', p1)

    p2 = Path([s2])
    print('p2', p2)

    p1_dict = Path.to_dict(p1)
    print('p1_dict', p1_dict)

    p2_dict = Path.to_dict(p2)
    print('p2_dict', p2_dict)

    p1_new = Path.from_dict(p1_dict)
    print('p1_new', p1_new)

    p2_new = Path.from_dict(p2_dict)
    print('p2_new', p2_new)

    metric, (a, b) = Path.select_pair(p1, p2, MetricSurface())
    p3 = Path.join(p1, p2, a, b)
    print('p3', p3)
    for index,s in enumerate(p3.segments):
        print('index: {}, is_reversed: {}'.format(index, s.is_reversed))

    p3.reverse()
    print('p3 reversed', p3)
    for index,s in enumerate(p3.segments):
        print('index: {}, is_reversed: {}'.format(index, s.is_reversed))

def main():
    # test_vector()
    #test_point()
    # test_segment()
    test_path()
    # print('All tests are commented')

if __name__ == '__main__':
    main()
