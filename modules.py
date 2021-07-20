import cv2
import numpy as np
import functions


def remove_noise(image):
    # 그레이스케일 및 이진화
    image = functions.threshold(image)

    # 모든 윤곽선 구하기
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 이미지의 세로와 가로 길이를 가져옴
    height, width = image.shape
    mask = np.zeros(image.shape, np.uint8)

    # 윤곽선에 바운딩된 사각형의 넓이가 이미지 넓이의 70% 이상이라면 오선 영역으로 판단하는 과정
    for contour in contours:
        # 윤곽선을 감싸는 사각형 객체 반환
        rect = cv2.boundingRect(contour)
        # 악보 영상 행 길이의 70%보다 가로로 긴 윤곽선들에 대해 빨간색 박스를 침(오선 영역)
        if rect[2] >= width * 0.7:
            cv2.rectangle(mask, rect, (255, 0, 0), -1)

    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def remove_staves(image):
    height, width = image.shape

    # 오선 좌표가 저장된 리스트 [오선의 마지막 y좌표][오선의 픽셀길이]
    staves = []
    for row in range(height):
        histogram = 0
        for col in range(width):
            # 흰색 픽셀 개수만큼 변수를 증가시킴
            if image[row][col] == 255:
                histogram += 1
        # 픽셀의 개수가 행 길이의 50% 이상이라면 오선으로 판단
        if histogram >= width * 0.5:
            # 새로운 오선이라면
            if len(staves) == 0 or abs(staves[-1][0] - row) > 1:
                staves.append([row, 0])
            # 이전과 같은 오선이라면
            else:
                staves[-1][0] = row
                staves[-1][1] = staves[-1][1] + 1

    # 오선을 제거함
    for staff in range(len(staves)):
        # 오선 최하단 좌표에서 오선 길이를 빼면 최상단 좌표가 나옴
        top_pixel = staves[staff][0] - staves[staff][1]
        bot_pixel = staves[staff][0]
        for col in range(width):
            # 오선 위, 아래로 픽셀이 존재하는지 검사함
            if image[top_pixel - 1][col] == 0 and image[bot_pixel + 1][col] == 0:
                # 존재하지 않는다면 오선 제거
                for row in range(top_pixel, bot_pixel + 1):
                    image[row][col] = 0

    return image, [x[0] for x in staves]


def normalization(image, staves, standard):
    height, width = image.shape

    avg_distance = 0
    lines = int(len(staves) / 5)
    for line in range(lines):
        for staff in range(4):
            staff_above = staves[line * 5 + staff]
            staff_below = staves[line * 5 + staff + 1]
            avg_distance += abs(staff_above - staff_below)
    avg_distance /= len(staves) - lines

    weight = standard / avg_distance
    new_height = int(height * weight)
    new_width = int(width * weight)

    image = cv2.resize(image, (new_width, new_height))
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # for x in staves:
    #     x *= weight
    staves = [x * weight for x in staves]

    return image, staves


def object_detection(image, staves):
    lines = int(len(staves) / 5)

    kernel = np.ones((functions.w(20), functions.w(20)), np.uint8)
    closing_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # closing 연산한 이미지에서 윤곽선 검출
    contours, hierarchy = cv2.findContours(closing_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(image.shape, np.uint8)
    objects = []

    # 원본 이미지에 바운딩
    for contour in contours:
        rect = cv2.boundingRect(contour)
        width_condition = functions.w(200) >= rect[2] >= functions.w(10)
        height_condition = functions.w(160) >= rect[3] >= functions.w(10)
        if width_condition and height_condition:
            center = (rect[1] + rect[1] + rect[3]) / 2
            for line in range(lines):
                # 음표로 가정할 수 있는 조건
                note_width_condition = rect[2] >= functions.w(20)
                note_height_condition = rect[3] >= functions.w(60) or (functions.w(25) >= rect[3] >= functions.w(15))
                if note_width_condition and note_height_condition:
                    top_limit = staves[line * 5] - functions.w(40)
                    bot_limit = staves[(line + 1) * 5 - 1] + functions.w(40)
                else:
                    top_limit = staves[line * 5] - functions.w(10)
                    bot_limit = staves[(line + 1) * 5 - 1] + functions.w(10)
                if top_limit <= center <= bot_limit:
                    cv2.rectangle(mask, rect, (255, 0, 0), -1)
                    objects.append([line, rect])

    masked_image = cv2.bitwise_and(image, mask)
    objects.sort()

    return masked_image, objects


def object_analysis(image, objects):
    # 모든 객체박스를 돌면서
    for obj in objects:
        rect = obj[1]
        stems = []
        # 가로
        for col in range(rect[0], rect[0] + rect[2]):
            histogram = 0
            stem_x = 0
            stem_y = 0
            stem_w = 0
            stem_h = 0
            # 세로
            for row in range(rect[1], rect[1] + rect[3]):
                # 객체가 있으면
                if image[row][col] == 255:
                    # 히스토그램 변수 + 1
                    histogram += 1
                    # 도중에 끊기면
                    if image[row + 1][col] == 0:
                        # 50이상이면 직선 으로 검출하고 이 열은 더이상 탐색 안해도됨
                        if histogram > functions.w(50):
                            stem_x = col
                            stem_y = row - histogram
                            stem_w = 0
                            stem_h = histogram
                            break
                        # 50미만이면 직선이 아닌걸로 판단
                        else:
                            histogram = 0
            if histogram > functions.w(50):
                if len(stems) == 0 or abs(stems[-1][0] + stems[-1][2] - stem_x) > 1:
                    stems.append([stem_x, stem_y, stem_w, stem_h])
                else:
                    stems[-1][2] += 1
        obj.append(stems)

    return image, objects
