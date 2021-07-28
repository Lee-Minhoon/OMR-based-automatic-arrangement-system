import cv2
import numpy as np
import functions as fs
import recognition_modules as rs

# 1. 보표 영역 추출 및 그 외 노이즈 제거
# ======================================================================================================================
'''
0. 원본 이미지가 입력된다.
1. 이미지 이진화 후 윤곽선 검출을 통해 모든 객체들을 검출할 수 있다.
2. 이미지 전체 넓이의 50% 이상인 객체만을 걸러냄으로써 보표 영역을 추출할 수 있다.
3. 추출한 보표 영역만을 마스킹한 마스크 이미지를 준비한다.
4. 마스크 이미지를 이용해 원본 영상에서 보표 영역을 제외한 노이즈를 제거할 수 있다.
5. 노이즈가 제거된 이미지를 반환한다.
- 객체(obj) : [y 좌표, x 좌표, 넓이, 높이]
'''
# ======================================================================================================================
def remove_noise(image):
    image = fs.threshold(image)  # 이미지 이진화
    objects = fs.detect_objects(image)  # 모든 객체 검출하기
    mask = np.zeros(image.shape, np.uint8)  # 보표 영역만 추출하기 위해 마스크 생성

    for obj in objects:
        if obj[2] >= image.shape[1] * 0.5:  # 객체의 넓이가 이미지 넓이의 50% 이상이면
            cv2.rectangle(mask, obj, (255, 0, 0), -1)  # 보표 영역으로 판단함 (마스킹)

    masked_image = cv2.bitwise_and(image, mask)  # 보표 영역 추출

    return masked_image


# 2. 오선 제거
# ======================================================================================================================
'''
0. 보표 영역을 제외한 노이즈가 제거된 이미지가 입력된다.
1. 보표와 마찬가지로 오선도 이미지 내에서 길게 존재한다.
2. 수평 히스토그램을 사용해 오선이 포함된 행을 검출할 수 있다.
3. 다른 객체들의 모양을 훼손하지 않기 위해 오선 위 아래로 픽셀의 존재 여부를 검사한다.
4. 위 아래로 픽셀이 존재하지 않으면 해당 좌표의 픽셀을 제거한다.
5. 오선이 제거된 이미지와 오선 y 좌표가 담긴 리스트를 반환한다.
- 오선 리스트(staves) : [y 좌표, y 좌표, y 좌표 ... ]
'''
# ======================================================================================================================
def remove_staves(image):
    height, width = image.shape
    staves = []  # 오선의 좌표들이 저장될 리스트

    for row in range(height):
        pixels = 0
        for col in range(width):
            pixels += (image[row][col] == 255)  # 한 행에 존재하는 픽셀의 개수를 셈
        if pixels >= width * 0.5:  # 픽셀의 개수가 이미지 넓이의 50% 이상이면
            if len(staves) == 0 or abs(staves[-1][0] + staves[-1][1] - row) > 1:  # 첫 오선이거나 이전에 검출된 오선과 다른 오선
                staves.append([row, 0])  # 오선 추가 [오선의 y 좌표][오선 높이]
            else:  # 이전에 검출된 오선과 같은 오선
                staves[-1][1] += 1  # 높이 업데이트

    for staff in range(len(staves)):
        top_pixel = staves[staff][0]  # 오선의 최상단 y 좌표
        bot_pixel = staves[staff][0] + staves[staff][1]  # 오선의 최하단 y 좌표 (오선의 최상단 y 좌표 + 오선 높이)
        for col in range(width):
            if image[top_pixel - 1][col] == 0 and image[bot_pixel + 1][col] == 0:  # 오선 위, 아래로 픽셀이 있는지 탐색
                for row in range(top_pixel, bot_pixel + 1):
                    image[row][col] = 0  # 오선을 지움

    return image, [x[0] for x in staves]


# 3. 악보 이미지 정규화
# ======================================================================================================================
'''
0. 오선이 제거된 이미지와 오선 리스트, 칸 간격을 입력 받는다.
1. 작은 보표기준 5개의 오선이 존재하고 4개의 칸이 존재한다.
2. 존재하는 모든 칸의 높이를 구해 평균치를 낸다.
3. 칸의 평균 높이를 이용해 어떤 악보가 입력되던 항상 같은 오선 간격(칸)을 가지게 할 수 있다.
4. 이는 추후 탐색과정에서 이점으로 작용된다.
5. 정규화된 이미지와, 오선 y 좌표 리스트를 반환한다.
- 오선 리스트(staves) : [y 좌표, y 좌표, y 좌표 ... ]
'''
# ======================================================================================================================
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
    staves = [x * weight for x in staves]  # 오선 좌표에도 가중치를 곱해줌
    # for x in staves:
    #     x *= weight

    return image, staves


# 4. 객체 검출 과정
# ======================================================================================================================
'''
0. 정규화된 이미지, 오선 리스트를 입력받는다.
1. 윤곽선 검출을 통해 이미지에 존재하는 모든 객체들을 검출할 수 있다.
2. 악보에 존재할 수 있는 구성요소들의 조건을 이용해 객체를 걸러 준다.
3. 거르는 과정에서 해당 구성요소가 몇 번째 보표에 속해있는지 탐색한다.
4. 추출된 객체만 남은 이미지와, 구성요소 리스트를 반환한다.
- 객체(obj) : [y 좌표, x 좌표, 넓이, 높이]
- 구성요소 리스트(components) : [[보표 번호, [객체]], [보표 번호, [객체]], [보표 번호, [객체]] ... ]
'''
# ======================================================================================================================
def object_detection(image, staves):
    dilated_image = fs.dilate(image)  # 이미지 팽창 연산 (같은 객체임에도 분리되어 검출되는 것을 방지)
    objects = fs.detect_objects(dilated_image)  # 팽창 연산한 이미지에서 객체 검출
    lines = int(len(staves) / 5)  # 보표의 개수
    mask = np.zeros(image.shape, np.uint8)  # 구성요소로 분류된 객체 영역만 추출하기 위해 마스크 생성
    components = []  # 구성요소 정보가 저장될 리스트

    for obj in objects:
        if obj[2] >= fs.w(26) and obj[3] >= fs.w(24):  # 악보의 구성요소가 되기 위한 넓이, 높이 조건
            center = fs.get_center(obj)  # 객체의 중간 y 좌표
            for line in range(lines):
                note_conditions = (
                    obj[2] >= fs.w(35) and  # 넓이 조건
                    obj[3] >= fs.w(80) or fs.w(30) >= obj[3] >= fs.w(20)  # 높이 조건
                )
                if note_conditions:
                    area_top = staves[line * 5] - fs.w(20)  # 음표의 위치 조건 (상단)
                    area_bot = staves[(line + 1) * 5 - 1] + fs.w(20)  # 음표의 위치 조건 (하단)
                else:
                    area_top = staves[line * 5]  # 나머지 구성 요소(쉼표, 조표 등)의 위치 조건 (상단)
                    area_bot = staves[(line + 1) * 5 - 1]  # 나머지 구성 요소(쉼표, 조표 등)의 위치 조건 (하단)
                if area_top <= center <= area_bot:  # 위치 조건이 만족되면 악보의 구성요소로 판단
                    cv2.rectangle(mask, obj, (255, 0, 0), -1)  # 마스킹
                    components.append([line, obj])  # 객체 리스트에 보표 번호와 객체의 정보(위치, 크기)를 추가

    masked_image = cv2.bitwise_and(image, mask)  # 객체 영역 추출
    components.sort()  # 보표 번호 → x 좌표 순으로 오름차순 정렬

    return masked_image, components


# 5. 객체 분석 과정
# ======================================================================================================================
'''
0. 객체가 검출된 이미지와 구성요소 리스트를 입력받는다.
1. 객체를 인식하기 이전에 악보의 구성요소들이 가지고 있는 특징점들을 추출한다.
2. 음표는 직선(stem)이 있을 수 있고 직선이 있다면 방향이 2가지 존재한다.
3. 직선은 수직 히스토그램을 통해 추출할 수 있다.
4. 직선이 얼마나 빨리 탐색되냐에 따라 음표의 방향을 알 수 있다.
5. 해당 특징점들을 미리 추출해 추후 인식 과정에 사용할 수 있다.
6. 직선 리스트와 음표 방향이 추가된 객체 리스트를 반환한다.
- 직선(stem) : [x 좌표, y 좌표, 넓이, 높이]
- 직선 리스트(stems) : [직선, 직선, 직선 ... ]
- 구성요소 리스트(components) : [[보표 번호, [객체], [직선 리스트], [음표 방향]] ... ]
'''
# ======================================================================================================================
def object_analysis(image, components):
    for comp in components:
        rect = comp[1]
        stems = fs.stem_detection(image, rect, 60)  # 객체 내의 모든 직선들을 검출함
        direction = None
        if len(stems) > 0:  # 직선이 1개 이상 존재함
            if stems[0][0] - rect[0] >= fs.w(10):  # 직선이 나중에 발견 되면
                direction = True  # 정 방향 음표
            else:  # 직선이 일찍 발견 되면
                direction = False  # 역 방향 음표
        comp.append(stems)  # 객체 리스트에 직선 리스트를 추가
        comp.append(direction)  # 객체 리스트에 음표 방향을 추가

    return image, components


# 6. 객체 인식 과정
# ======================================================================================================================
'''
0. 이미지와 오선 리스트, 구성요소 리스트를 입력받는다.
1. 모든 객체를 인식 알고리즘을 통해 조표, 음표, 쉼표, 기타로 분류한다.
2. 음표로 분류된다면 오선의 좌표와 음표 머리의 좌표를 통해 음의 높낮이를 알 수 있다.
- 구성요소 리스트(components) : [[보표 번호, [객체], [직선 리스트], [음표 방향]] ... ]
'''
# ======================================================================================================================
def recognition(image, staves, components):
    beats = []  # 박자 리스트
    pitches = []  # 음이름 리스트

    for i in range(len(components)):
        component = components[i]
        line = component[0]
        rect = component[1]
        stems = component[2]
        stem_direction = component[3]
        comp_staves = staves[line * 5: (line + 1) * 5]
        if i == 1:  # 구성요소가 첫번째라면 (해당 위치에는 조표 또는 박자가 위치함)
            key = rs.recognize_key(image, comp_staves, rect)
        else:  # 조표가 아니라면 음표 또는 쉼표
            notes, pitch = rs.recognize_note(image, comp_staves, rect, stems, stem_direction)
            if len(notes):  # 음표로 인식 되었다면
                beats.append(notes)
                pitches.append(pitch)
            else:  # 음표로 인식 되지 않았다면
                rest, pitch = rs.recognize_rest(image, comp_staves, rect)
                if rest:  # 쉼표로 인식 되었다면
                    beats.append(rest)
                    pitches.append(pitch)

        cv2.rectangle(image, rect, (255, 0, 0), 1)
        fs.put_text(image, i, (rect[0], rect[1] - fs.w(20)))

    print(key)
    print(beats)
    print(pitches)

    return image, key, beats, pitches
