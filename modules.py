import cv2
import numpy as np
import functions


# 전처리 과정 1. 오선 영역 밖 노이즈 제거
def remove_noise(image):
    # 그레이스케일 및 이진화
    image = functions.threshold(image)
    # 모든 윤곽선 구하기
    contours = functions.get_contours(image)
    # 이미지의 세로와 가로 길이를 가져옴
    height, width = image.shape
    # 오선 영역만 추출하기 위해 마스크를 준비함
    mask = np.zeros(image.shape, np.uint8)

    # 모든 윤곽선을 돌며
    for contour in contours:
        # 윤곽선을 감싸는 사각형 객체 반환
        rect = cv2.boundingRect(contour)
        # 윤곽선에 바운딩된 사각형의 넓이가 이미지 넓이의 50% 이상이라면
        if rect[2] >= width * 0.5:
            # 마스크에 사각형을 그림
            cv2.rectangle(mask, rect, (255, 0, 0), -1)

    # 이미지와 마스크를 비트 연산해 오선 영역의 이미지만 추출함
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


# 전처리 과정 2. 오선 제거
def remove_staves(image):
    # 이미지의 세로와 가로 길이를 가져옴
    height, width = image.shape
    # 오선 좌표가 저장될 리스트 [오선의 마지막 y좌표][오선의 픽셀 길이]
    staves = []

    # 이미지 세로 길이 (행 탐색)
    for row in range(height):
        histogram = 0
        # 이미지 가로 길이 (열 탐색)
        for col in range(width):
            # 흰색 픽셀 개수만큼 histogram 변수를 증가시킴
            if image[row][col] == 255:
                histogram += 1
        # 픽셀의 개수가 행 길이의 50% 이상이라면 오선으로 판단
        if histogram >= width * 0.5:
            # 새로운 오선이라면
            if len(staves) == 0 or abs(staves[-1][0] - row) > 1:
                # 오선 리스트에 추가
                staves.append([row, 0])
            # 이전과 같은 오선이라면
            else:
                # 오선 좌표와 길이를 업데이트
                staves[-1][0] = row
                staves[-1][1] = staves[-1][1] + 1

    # 오선 리스트를 돌며
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


# 전처리 과정 3. 오선 평균 간격을 구하고 이를 이용해 악보 이미지에 가중치를 곱해줌
def normalization(image, staves, standard):
    # 오선 평균 길이가 저장될 변수
    avg_distance = 0
    # 총 몇개의 오선 영역인지 구함
    lines = int(len(staves) / 5)
    # 오선 영역을 돌며
    for line in range(lines):
        # 오선 간의 간격을 구함
        for staff in range(4):
            # 위 오선
            staff_above = staves[line * 5 + staff]
            # 아래 오선
            staff_below = staves[line * 5 + staff + 1]
            # 간격을 더해줌
            avg_distance += abs(staff_above - staff_below)
    # 간격 개수를 나누어 평균 간격을 구함
    avg_distance /= len(staves) - lines

    # 이미지의 세로와 가로 길이를 가져옴
    height, width = image.shape
    # 기준으로 정한 오선 간격을 이용해 가중치를 구함
    weight = standard / avg_distance
    # 이미지의 세로, 가로 길이에 가중치를 곱해줌
    new_height = int(height * weight)
    new_width = int(width * weight)

    # 기준으로 정한 오선 간격이 되게끔 이미지 리사이징
    image = cv2.resize(image, (new_width, new_height))
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # for x in staves:
    #     x *= weight

    # 오선 리스트에도 가중치를 곱해줌
    staves = [x * weight for x in staves]

    return image, staves


# 객체 검출 과정
def object_detection(image, staves):
    # 이미지 닫힘 연산을 통해 분리된 객체를 합침 (같은 객체임에도 분리되어 검출되는 것을 막기 위함)
    closing_image = functions.closing(image)
    # 닫힘 연산한 이미지에서 윤곽선 검출
    contours = functions.get_contours(closing_image)
    # 총 몇개의 오선 영역인지 구함
    lines = int(len(staves) / 5)
    # 객체 영역만 추출하기 위해 마스크를 준비함
    mask = np.zeros(image.shape, np.uint8)
    # 이미지 내의 모든 객체들을 담을 리스트
    objects = []

    # 모든 윤곽선을 돌며
    for contour in contours:
        # 윤곽선을 감싸는 사각형 객체 반환
        rect = cv2.boundingRect(contour)
        # 악보의 구성요소가 되기 위한 최소 크기 조건
        width_condition = functions.w(200) >= rect[2] >= functions.w(10)
        height_condition = functions.w(160) >= rect[3] >= functions.w(10)
        if width_condition and height_condition:
            # 객체의 중간 y 좌표
            center = (rect[1] + rect[1] + rect[3]) / 2
            for line in range(lines):
                # 음표로 가정할 수 있는 최소 크기 조건
                note_width_condition = rect[2] >= functions.w(20)
                note_height_condition = rect[3] >= functions.w(60) or (functions.w(25) >= rect[3] >= functions.w(15))
                if note_width_condition and note_height_condition:
                    # 음표의 위치 조건
                    top_limit = staves[line * 5] - functions.w(40)
                    bot_limit = staves[(line + 1) * 5 - 1] + functions.w(40)
                else:
                    # 나머지 구성 요소의 위치 조건 (쉼표, 조표 등)
                    top_limit = staves[line * 5] - functions.w(10)
                    bot_limit = staves[(line + 1) * 5 - 1] + functions.w(10)
                # 위치 조건이 만족되면 악보의 구성요소로 판단함
                if top_limit <= center <= bot_limit:
                    # 마스크에 사각형을 그림
                    cv2.rectangle(mask, rect, (255, 0, 0), -1)
                    # 객체 리스트에 라인 번호와 객체 정보를 담음
                    objects.append([line, rect])

    # 이미지와 마스크를 비트 연산해 객체 영역의 이미지만 추출함
    masked_image = cv2.bitwise_and(image, mask)
    # 라인 번호, x 좌표 순서대로 객체들을 정렬
    objects.sort()

    return masked_image, objects


# 객체 분석
def object_analysis(image, objects):
    # 모든 객체를 돌며
    for obj in objects:
        rect = obj[1]
        # 객체 내의 모든 직선들을 담을 리스트
        stems = []
        # 이미지 가로 길이 (열 탐색)
        for col in range(rect[0], rect[0] + rect[2]):
            histogram = 0
            # 이미지 세로 길이 (행 탐색)
            for row in range(rect[1], rect[1] + rect[3]):
                # 객체가 있으면 (흰색 픽셀)
                if image[row][col] == 255:
                    # histgram 변수를 증가시킴
                    histogram += 1
                    # 도중에 끊기면
                    if image[row + 1][col] == 0:
                        # 50이상이면 직선으로 검출하고 해당 열은 더이상 탐색 중지
                        if histogram > functions.w(50):
                            stem_x = col
                            stem_y = row - histogram
                            stem_w = 0
                            stem_h = histogram
                            break
                        # 50미만이면 직선이 아닌걸로 판단
                        else:
                            histogram = 0
            # 직선을 검출했다면
            if histogram > functions.w(50):
                # 새로운 직선이라면
                if len(stems) == 0 or abs(stems[-1][0] + stems[-1][2] - stem_x) > 1:
                    stems.append([stem_x, stem_y, stem_w, stem_h])
                # 이전과 같은 직선이라면
                else:
                    stems[-1][2] += 1
        # 객체 리스트에 직선 리스트를 추가
        obj.append(stems)

    return image, objects


def recognition(image, objects):
    pass