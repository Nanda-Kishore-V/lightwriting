import cv2

from constants import HOME

def main():
    image = cv2.imread(HOME + 'data/images/temporary.bmp', 0)
    (thresh, image) = cv2.threshold(image, 128, 255, cv2.THRESH_OTSU)
    cv2.imwrite(HOME + 'data/images/black_text.bmp', image)

if __name__ == '__main__':
    main()
