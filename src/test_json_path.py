from __future__ import print_function
import json

from constants import HOME
from geometry import Point, Vector, Segment, Path

def test1(filename):
    print('-' * 80)
    print('Test 1')
    print('-' * 80)
    p = Path([Segment(True, [Point((1,2)), Point((3,4))])])
    print('p')
    print(p)
    
    p_dict = Path.to_dict(p)
    print('p_dict')
    print(p_dict)

    with open(filename, 'w') as f:
        json.dump(p_dict, f)

    with open(filename) as f:
        new_p_dict = json.load(f)

    new_p = Path.from_dict(new_p_dict)
    print('new_p')
    print(new_p)

def test2(filename):
    print('-' * 80)
    print('Test 2')
    print('-' * 80)
    s1 = Segment(True, [
                        Point((1, 2), Vector((3, 4))), 
                        Point((5, 2)), 
                        Point((-1, 0))
                        ])
    s2 = Segment(True, [
                        Point((10, 5), Vector((3, 4))), 
                        Point((3, 2), Vector((-1, 1)))
                        ])

    p1 = Path([s1]) 
    p2 = Path([s2]) 
    s1_point = s1.points[0]
    s2_point = s2.points[0]

    p = Path.join(p1, p2, s1_point, s2_point)
    print('p')
    print(p)
p_dict = Path.to_dict(p)

    print('p_dict')
    print(p_dict)
    with open(filename, 'w') as f:
        json.dump(p_dict, f)

    with open(filename) as f:
        new_p_dict = json.load(f)
    new_p = Path.from_dict(new_p_dict)
    print('new_p')
    print(new_p)

def main():
    filename = HOME + 'data/path.json'

    test1(filename)
    test2(filename)


if __name__ == '__main__':
    main()
