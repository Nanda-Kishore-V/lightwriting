import cv2

from constants_crazyswarm import WHITE

def show_and_wait(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)

def show_and_destroy(title, image):
    show_and_wait(title, image)
    cv2.destroyWindow(title)

def to_image(points, width=None, height=None):
    '''
    Show points as WHITE pixels on BLACK background

    points: list of Points
    width: width of image generated
    height: height of image generated
    '''
    x_coords = [p.coords[0] for p in points]
    y_coords = [p.coords[1] for p in points]
    width = max(x_coords) + 1 if width is None else width
    height = max(y_coords) + 1 if height is None else height

    assert isinstance(width, (int, long)) and width > 0
    assert isinstance(height, (int, long)) and height > 0

    image = np.zeros((width, height), dtype=np.uint8)
    for p in points:
        image[p.coords] = WHITE

    show_and_destroy('points as image', image)

