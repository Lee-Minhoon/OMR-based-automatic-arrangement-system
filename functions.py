import cv2
import numpy as np


def threshold(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return image


def get_contours(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def dilate(image):
    kernel = np.ones((w(12), w(12)), np.uint8)
    image = cv2.dilate(image, kernel)
    return image


def put_text(image, text, loc):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, str(text), loc, font, 0.8, (255, 0, 0), 2)


def count_line_pixels(image, axis, axis_value, line, length):
    pixel = 0
    for point in range(line[0], line[1]):
        if axis:
            row = axis_value
            col = point
        else:
            row = point
            col = axis_value
        if image[row][col] == 255:
            pixel += 1
            if image[row + 1][col] == 0:
                if pixel > w(length):
                    break
                else:
                    pixel = 0
    return point, pixel


def count_rect_pixels(image, rect):
    pixel = 0
    for row in range(rect[1], rect[1] + rect[3]):
        for col in range(rect[0], rect[0] + rect[2]):
            if image[row][col] == 255:
                pixel += 1
    return pixel


def stem_detection(image, rect, length):
    stems = []
    # 이미지 가로 길이 (열 탐색)
    for col in range(rect[0], rect[0] + rect[2]):
        row_range = (rect[1], rect[1] + rect[3])
        row, pixels = count_line_pixels(image, False, col, row_range, length)
        # 직선을 검출했다면
        if pixels > w(length):
            # 새로운 직선이라면
            if len(stems) == 0 or abs(stems[-1][0] + stems[-1][2] - col) > 1:
                # 직선 리스트에 추가 (x, y, w, h)
                stems.append([col, row - pixels, 0, pixels])
            # 이전과 같은 직선이라면
            else:
                # 직선의 길이를 업데이트
                stems[-1][2] += 1
    return stems


def w(value):
    standard = 10
    return int(value * (standard / 20))
