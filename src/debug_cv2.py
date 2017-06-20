import cv2

def show_and_wait(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)

def show_and_destroy(title, image):
    show_and_wait(title, image)
    cv2.destroyWindow(title)
