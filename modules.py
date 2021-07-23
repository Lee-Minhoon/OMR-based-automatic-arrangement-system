import cv2
import numpy as np
import functions as fs
import recognition_modules as rs

# 전처리 과정 1. 보표 영역 추출 및 그 외 노이즈 제거
# ==============================================================
'''
윤곽선 검출을 통해 모든 객체를 검출하고 이미지 넓이의 50% 이상인 객체를
보표 영역으로 판단해 그 영역만을 추출하고 나머지는 노이즈로 간주하여 제거
'''
# ==============================================================
def remove_noise(image):
    image = fs.threshold(image)  # 이미지 이진화
    contours = fs.get_contours(image)  # 모든 윤곽선 구하기
    height, width = image.shape  # 이미지의 높이와 넓이
    mask = np.zeros(image.shape, np.uint8)  # 오선 영역만 추출하기 위해 마스크 생성

    for contour in contours:
        rect = cv2.boundingRect(contour)  # 윤곽선을 감싸는 사각형 객체 반환
        if rect[2] >= width * 0.5:  # 사각형의 넓이가 이미지 넓이의 50% 이상이면
            cv2.rectangle(mask, rect, (255, 0, 0), -1)  # 오선 영역으로 판단함 (마스킹)

    masked_image = cv2.bitwise_and(image, mask)  # 오선 영역 추출

    return masked_image


# 전처리 과정 2. 오선 제거
# ==============================================================
'''
오선은 기본적으로 이미지에서 상당히 길게 존재하기 때문에 수평 히스토그램을
이용하여 오선을 제거함. 이 때 다른 객체들의 모양을 훼손하지 않기 위해
오선 위 아래로 픽셀이 있는지 탐색후 제거하게 됨.
'''
# ==============================================================
def remove_staves(image):
    height, width = image.shape  # 이미지의 높이와 넓이
    staves = []  # 오선의 좌표들이 저장될 리스트 [오선의 마지막 y 좌표][오선 높이]

    for row in range(height):
        pixels = 0
        for col in range(width):
            if image[row][col] == 255:
                pixels += 1  # 한 행에 존재하는 픽셀의 개수를 셈
        if pixels >= width * 0.5:  # 픽셀의 개수가 이미지 넓이의 50% 이상이면
            if len(staves) == 0 or abs(staves[-1][0] - row) > 1:  # 새로운 오선이거나 이전에 검출된 오선과 다른 오선
                staves.append([row, 0])
            else:  # 이전에 검출된 오선과 같은 오선
                staves[-1][0] = row  # y 좌표 업데이트
                staves[-1][1] = staves[-1][1] + 1  # 높이 업데이트

    for staff in range(len(staves)):
        top_pixel = staves[staff][0] - staves[staff][1]  # 오선의 최상단 y 좌표 (오선의 최하단 y 좌표 - 오선 높이)
        bot_pixel = staves[staff][0]  # 오선의 최하단 y 좌표
        for col in range(width):
            if image[top_pixel - 1][col] == 0 and image[bot_pixel + 1][col] == 0:  # 오선 위, 아래로 픽셀이 있는지 탐색
                for row in range(top_pixel, bot_pixel + 1):
                    image[row][col] = 0  # 오선을 지움

    return image, [x[0] for x in staves]


# 전처리 과정 3. 오선 평균 간격을 구하고 이를 이용해 악보 이미지에 가중치를 곱해줌
# ==============================================================
'''
각 보표에는 5개의 오선이 존재하므로 오선간의 간격은 보표마다 4개씩 존재함.
이 간격들을 모두 구해 평균치를 낼 수 있음. 이를 이용해 어떤 악보가 입력되던
항상 같은 오선 간격을 가지게끔 할 수 있고, 추후 탐색과정에서 이점으로 작용
'''
# ==============================================================
def normalization(image, staves, standard):
    avg_distance = 0  # 오선 평균 길이
    lines = int(len(staves) / 5)  # 보표의 개수
    for line in range(lines):
        for staff in range(4):
            staff_above = staves[line * 5 + staff]
            staff_below = staves[line * 5 + staff + 1]
            avg_distance += abs(staff_above - staff_below)  # 오선의 간격을 누적해서 더 해줌
    avg_distance /= len(staves) - lines  # 오선 간의 평균 간격

    height, width = image.shape  # 이미지의 높이와 넓이
    weight = standard / avg_distance  # 기준으로 정한 오선 간격을 이용해 가중치를 구함
    new_width = int(width * weight)  # 이미지의 넓이에 가중치를 곱해줌
    new_height = int(height * weight)  # 이미지의 높이에 가중치를 곱해줌

    image = cv2.resize(image, (new_width, new_height))  # 이미지 리사이징
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 이미지 이진화

    # for x in staves:
    #     x *= weight

    staves = [x * weight for x in staves]  # 오선 좌표에도 가중치를 곱해줌

    return image, staves


# 객체 검출 과정
# ==============================================================
'''
이미지에 대한 전처리 과정이 끝나고, 악보 이미지에 존재하는 모든 구성요소들을
검출하는 과정. 악보에 존재할 수 있는 구성요소들을 모두 포함하는 최소 조건을
객체를 대상으로 검사하여 조건에 맞는 객체만을 검출한다.
'''
# ==============================================================
def object_detection(image, staves):
    dilated_image = fs.dilate(image)  # 이미지 팽창 연산 (같은 객체임에도 분리되어 검출되는 것을 방지)
    contours = fs.get_contours(dilated_image)  # 팽창 연산한 이미지에서 윤곽선 검출
    lines = int(len(staves) / 5)  # 보표의 개수
    mask = np.zeros(image.shape, np.uint8)  # 구성요소로 분류된 객체 영역만 추출하기 위해 마스크 생성
    objects = []  # 객체 정보가 저장될 리스트

    for contour in contours:
        rect = cv2.boundingRect(contour)  # 윤곽선을 감싸는 사각형 객체 반환
        if rect[2] >= fs.w(20) and rect[3] >= fs.w(10):  # 악보의 구성요소가 되기 위한 넓이, 높이 조건
            center = (rect[1] + rect[1] + rect[3]) / 2  # 객체의 중간 y 좌표
            for line in range(lines):
                note_width_condition = rect[2] >= fs.w(20)  # 음표의 넓이 조건
                note_height_condition = rect[3] >= fs.w(60) or fs.w(25) >= rect[3] >= fs.w(15)  # 음표의 높이 조건
                if note_width_condition and note_height_condition:
                    area_top = staves[line * 5] - fs.w(20)  # 음표의 위치 조건 (상단)
                    area_bot = staves[(line + 1) * 5 - 1] + fs.w(20)  # 음표의 위치 조건 (하단)
                else:
                    area_top = staves[line * 5]  # 나머지 구성 요소(쉼표, 조표 등)의 위치 조건 (상단)
                    area_bot = staves[(line + 1) * 5 - 1]  # 나머지 구성 요소(쉼표, 조표 등)의 위치 조건 (하단)
                if area_top <= center <= area_bot:  # 위치 조건이 만족되면 악보의 구성요소로 판단
                    cv2.rectangle(mask, rect, (255, 0, 0), -1)  # 마스킹
                    objects.append([line, rect])  # 객체 리스트에 보표 번호와 객체의 정보(위치, 크기)를 담음

    masked_image = cv2.bitwise_and(image, mask)  # 객체 영역 추출
    objects.sort()  # 보표 번호 → x 좌표 순으로 오름차순 정렬

    return masked_image, objects


# 객체 분석 과정
# ==============================================================
'''
객체를 인식하기 이전에, 객체에서 악보의 구성요소들이 가지고 있는 특징점들을
추출한다. 음표는 직선(stem)이 있을 수 있고 방향이 2가지 존재한다.
이 특징점들을 추출해 인식 과정에 사용할 수 있다. 직선이 객체내에서 일정 범위를
탐색하기전에 발견된다면 음표의 방향을 알 수 있다.
'''
# ==============================================================
def object_analysis(image, objects):
    for obj in objects:
        rect = obj[1]
        stems = fs.stem_detection(image, rect, 60)  # 객체 내의 모든 직선들을 검출함
        direction = None
        if len(stems) > 0:  # 직선이 1개 이상 존재함
            if stems[0][0] - rect[0] >= fs.w(10):  # 직선이 나중에 발견 되면
                direction = True  # 정 방향 음표
            else:  # 직선이 일찍 발견 되면
                direction = False  # 역 방향 음표
        obj.append(stems)  # 객체 리스트에 직선 리스트를 추가
        obj.append(direction)  # 객체 리스트에 음표 방향을 추가

    return image, objects


# 객체 인식 과정
# ==============================================================
'''
이전에 추출한 특징점들과 그것을 토대로 추가적인 탐색을 통해 객체를 분류한다.
악보의 구성요소는 크게 다음과 같이 분류된다.
조표(key), 음표(note), 쉼표(rest)
음표로 분류된다면 오선의 좌표와 음표 머리의 좌표를 통해 음의 높낮이를 구한다.
'''
# ==============================================================
def recognition(image, staves, objects):
    beats = []  # 박자 리스트
    notes = []  # 음이름 리스트

    for i in range(len(objects)):
        obj = objects[i]
        line = objects[i][0]
        rect = objects[i][1]
        stems = objects[i][2]
        stem_direction = objects[i][3]
        obj_staves = staves[line * 5: (line + 1) * 5]
        if i == 1:
            key = rs.recognize_key(image, obj_staves, rect)
        else:
            note = rs.recognize_note(image, rect, stems, stem_direction)
            if len(note):
                notes.append(note)
            else:
                notes.append(rs.recognize_rest(image, obj_staves, rect))

            # 분석이 끝난 객체에 박스 바운딩
        cv2.rectangle(image, obj[1], (255, 0, 0), 1)
        fs.put_text(image, i, (obj[1][0], obj[1][1] - fs.w(20)))

    print(key)
    print(notes)

    return image
