# functions.py
import cv2


def threshold(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return image


def detect_objects(image):
    objects = []
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        objects.append(cv2.boundingRect(contour))
    return objects


def put_text(image, text, loc):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, str(text), loc, font, 0.8, (255, 0, 0), 2)


def get_center(y, h):
    return (y + y + h) / 2


def get_line(image, axis, axis_value, line, length):
    pixels = 0
    points = [(axis_value, x) for x in range(line[0], line[1])] if axis else [(x, axis_value) for x in range(line[0], line[1])]
    for point in points:
        pixels += (image[point[0]][point[1]] == 255)
        if image[point[0] + 1][point[1]] == 0:
            if pixels > weighted(length):
                break
            else:
                pixels = 0
    return point[1] if axis else point[0], pixels - 1


def count_rect_pixels(image, rect):
    pixels = 0
    for row in range(rect[1], rect[1] + rect[3]):
        for col in range(rect[0], rect[0] + rect[2]):
            if image[row][col] == 255:
                pixels += 1
    return pixels


def stem_detection(image, stats, length):
    x, y, w, h, area = stats
    stems = []
    for col in range(x, x + w):
        row_range = (y, y + h)
        row, pixels = get_line(image, False, col, row_range, length)
        if pixels > weighted(length):
            if len(stems) == 0 or abs(stems[-1][0] + stems[-1][2] - col) > 1:
                stems.append([col, row - pixels, 0, pixels])
            else:
                stems[-1][2] += 1
    return stems


def weighted(value):
    standard = 10
    return int(value * (standard / 10))
