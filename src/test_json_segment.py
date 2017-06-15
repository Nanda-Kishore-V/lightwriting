from __future__ import print_function
import json

from constants import HOME
from geometry import Point, Vector, Segment

def test(filename):
    print('-' * 80)
    print('Test')
    print('-' * 80)
    s = Segment(True, [Point((1, 2), Vector((3, 4))), Point((5, 2)), Point((-1, 0))])
    print('segment')
    print(s)

    s_dict = Segment.to_dict(s)

    print('s_dict')
    print(s_dict)
    with open(filename, 'w') as f:
        json.dump(s_dict, f)

    with open(filename) as f:
        new_s_dict = json.load(f)
    new_s = Segment.from_dict(new_s_dict)
    print('new_s')
    print(new_s)

def main():
    filename = HOME + 'data/segment.json'
    test(filename)

if __name__ == '__main__':
    main()
