import cv2
import numpy as np


def threshold(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return image


def detect_objects(image):
    objects = []
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        objects.append(cv2.boundingRect(contour))
    return objects


def dilate(image):
    kernel = np.ones((w(15), w(15)), np.uint8)
    image = cv2.dilate(image, kernel)
    return image


def put_text(image, text, loc):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, str(text), loc, font, 0.8, (255, 0, 0), 2)


def get_center(rect):
    return (rect[1] + rect[1] + rect[3]) / 2


def get_line(image, axis, axis_value, line, length):
    pixels = 0
    for opposite_axis_value in range(line[0], line[1]):
        point = (axis_value, opposite_axis_value) if axis else (opposite_axis_value, axis_value)
        pixels += (image[point[0]][point[1]] == 255)
        if image[point[0] + 1][point[1]] == 0:
            if pixels > w(length):
                break
            else:
                pixels = 0
    return opposite_axis_value, pixels


def count_rect_pixels(image, rect):
    pixels = 0
    for row in range(rect[1], rect[1] + rect[3]):
        for col in range(rect[0], rect[0] + rect[2]):
            if image[row][col] == 255:
                pixels += 1
    return pixels


def stem_detection(image, rect, length):
    stems = []
    for col in range(rect[0], rect[0] + rect[2]):
        row_range = (rect[1], rect[1] + rect[3])
        row, pixels = get_line(image, False, col, row_range, length)
        if pixels > w(length):
            if len(stems) == 0 or abs(stems[-1][0] + stems[-1][2] - col) > 1:
                stems.append([col, row - pixels, 0, pixels])
            else:
                stems[-1][2] += 1
    return stems


def w(value):
    standard = 10
    return int(value * (standard / 20))