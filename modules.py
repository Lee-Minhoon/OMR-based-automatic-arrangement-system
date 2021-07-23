import cv2
import numpy as np
import functions


# 과정 1. 오선 영역 밖 노이즈 제거
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


# 과정 2. 오선 제거
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


# 3. 오선 평균 간격을 구하고 이를 이용해 악보 이미지에 가중치를 곱해줌
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


# 4. 객체 검출 과정
def object_detection(image, staves):
    # 이미지 팽창 연산을 통해 분리된 객체를 합침 (같은 객체임에도 분리되어 검출되는 것을 막기 위함)
    dilated_image = functions.dilate(image)
    # 팽창 연산한 이미지에서 윤곽선 검출
    contours = functions.get_contours(dilated_image)
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
        if rect[2] >= functions.w(20) and rect[3] >= functions.w(10):
            # 객체의 중간 y 좌표
            center = (rect[1] + rect[1] + rect[3]) / 2
            for line in range(lines):
                # 음표로 가정할 수 있는 최소 크기 조건
                note_width_condition = rect[2] >= functions.w(20)
                note_height_condition = rect[3] >= functions.w(60) or (functions.w(25) >= rect[3] >= functions.w(15))
                if note_width_condition and note_height_condition:
                    # 음표의 위치 조건
                    top_limit = staves[line * 5] - functions.w(20)
                    bot_limit = staves[(line + 1) * 5 - 1] + functions.w(20)
                else:
                    # 나머지 구성 요소의 위치 조건 (쉼표, 조표 등)
                    top_limit = staves[line * 5]
                    bot_limit = staves[(line + 1) * 5 - 1]
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

    return image, objects


# 5. 객체 분석
def object_analysis(image, objects):
    # 모든 객체를 돌며
    for obj in objects:
        rect = obj[1]
        # 객체 내의 모든 직선들을 검출함
        stems = functions.stem_detection(image, rect, 60)
        # 객체 리스트에 직선 리스트를 추가
        obj.append(stems)
        direction = None
        # 직선이 1개이상 존재함
        if len(stems) > 0:
            # 직선이 나중에 발견 되면
            if stems[0][0] - rect[0] > functions.w(10):
                # 정 방향 음표
                direction = True
            # 직선이 일찍 발견 되면
            else:
                # 역 방향 음표
                direction = False
            # 객체 리스트에 직선 방향을 추가
        obj.append(direction)

    return image, objects


# 6. 객체 인식
def recognition(image, staves, objects):
    # 박자와 음이름 리스트
    beats = []
    notes = []

    # 모든 객체를 돌며
    for i in range(len(objects)):
        obj = objects[i]
        line = objects[i][0]
        rect = objects[i][1]
        stems = objects[i][2]
        stem_direction = objects[i][3]
        obj_staves = staves[line * 5: (line + 1) * 5]
        if i == 1:
            key = recognize_key(image, obj_staves, rect)
        else:
            notes.append(recognize_note(image, rect, stems, stem_direction))
            rests.append(recognize_rest(obj))
            # 분석이 끝난 객체에 박스 바운딩
        cv2.rectangle(image, obj[1], (255, 0, 0), 1)
        functions.put_text(image, i, (obj[1][0], obj[1][1] - functions.w(20)))

    print(key)
    print(notes)

    return image


def recognize_key(image, staves, rect):
    # 조표가 없을 경우 (다장조일 경우) 박자표가 놓이게 되는 것을 검사
    no_key_top_condition = staves[0] + functions.w(5) > rect[1] > staves[0] - functions.w(5)
    no_key_bot_condition = staves[4] + functions.w(5) > rect[1] + rect[3] > staves[4] - functions.w(5)
    no_key_center_condition = staves[2] + functions.w(5) > rect[1] + rect[1] + rect[3] / 2 > staves[2] - functions.w(5)
    if no_key_top_condition and no_key_bot_condition and no_key_center_condition:
        no_key_width_condition = functions.w(35) > rect[2] > functions.w(20)
        no_key_height_condition = functions.w(90) > rect[3] > functions.w(75)
        if no_key_width_condition and no_key_height_condition:
            if functions.count_rect_pixels(image, rect) > functions.w(800):
                key = 0
    # 조표가 있을 경우 (다장조를 제외한 모든 조)
    else:
        # 객체 내의 모든 직선들을 검출함
        stems = functions.stem_detection(image, rect, 30)
        flat_top_condition = staves[0] + functions.w(10) > stems[0][1] > staves[0] - functions.w(10)
        flat_bot_condition = staves[3] > stems[0][1] + stems[0][3] > staves[2]
        # 첫 직선의 위치가 플랫이 처음 놓이게 되는 위치라면
        if flat_top_condition and flat_bot_condition:
            functions.put_text(image, "b" + str(len(stems)), (rect[0], rect[1] + rect[3] + functions.w(60)))
            key = 10 + len(stems)
        # 첫 직선의 위치가 샾이 처음 놓이게 되는 위치라면
        else:
            functions.put_text(image, "#" + str(len(stems)), (rect[0], rect[1] + rect[3] + functions.w(60)))
            key = 20 + len(stems)

    return key


def recognize_note(image, rect, stems, stem_direction):
    note = []
    # 음표로 가정할 수 있는 최소 크기 조건
    if rect[2] > functions.w(20) and rect[3] > functions.w(60):
        # 객체가 가진 모든 직선을 돌며
        for i in range(len(stems)):
            stem = stems[i]
            head_fill, head_pixel = recognize_note_head(image, stem, stem_direction)
            # 음표 머리에 해당하는 부분이 있다고 판단되면
            if head_pixel > functions.w(15):
                tail_cnt = recognize_note_tail(image, i, stem, stem_direction)
                point_exist = recognize_note_point(image, stem, stem_direction)
                half_note_condition = not head_fill and tail_cnt == 0 and point_exist == 0
                dotted_half_note_condition = not head_fill and tail_cnt == 0 and point_exist == 1
                quarter_note_condition = head_fill and tail_cnt == 0 and point_exist == 0
                dotted_quarter_note_condition = head_fill and tail_cnt == 0 and point_exist == 1
                eighth_note_condition = head_fill and tail_cnt == 1 and point_exist == 0
                dotted_eighth_note_condition = head_fill and tail_cnt == 1 and point_exist == 1
                sixteenth_note_condition = head_fill and tail_cnt == 2 and point_exist == 0
                dotted_sixteenth_note_condition = head_fill and tail_cnt == 2 and point_exist == 1
                if half_note_condition:
                    note.append(2)
                elif dotted_half_note_condition:
                    note.append(-2)
                elif quarter_note_condition:
                    note.append(4)
                elif dotted_quarter_note_condition:
                    note.append(-4)
                elif eighth_note_condition:
                    note.append(8)
                elif dotted_eighth_note_condition:
                    note.append(-8)
                elif sixteenth_note_condition:
                    note.append(16)
                elif dotted_sixteenth_note_condition:
                    note.append(-16)
                if len(note):
                    functions.put_text(image, str(note[i]), (stem[0] - functions.w(20), stem[1] + stem[3] + functions.w(60)))

    return note


def recognize_note_head(image, stem, stem_direction):
    # 정 방향 음표
    if stem_direction:
        # 음표 머리가 있는지, 어떤 모양인지 탐색할 위치
        head_area_top = stem[1] + stem[3] - functions.w(20)
        head_area_bot = stem[1] + stem[3] + functions.w(20)
        head_area_left = stem[0] - functions.w(20)
        head_area_right = stem[0]
    # 역 방향 음표
    else:
        # 음표 머리가 있는지, 어떤 모양인지 탐색할 위치 (top, bot)
        head_area_top = stem[1] - functions.w(20)
        head_area_bot = stem[1] + functions.w(20)
        head_area_left = stem[0] + stem[2]
        head_area_right = stem[0] + stem[2] + functions.w(20)
    head_pixel = 0
    head_pixel_max = 0
    head_fill_pixel = 0
    head_fill_pixel_max = 0
    # 머리 탐색 (행)
    for row in range(head_area_top, head_area_bot):
        pixels = 0
        # 머리 탐색 (열)
        for col in range(head_area_left, head_area_right):
            # 흰색 픽셀 개수만큼 histogram 변수를 증가시킴
            if image[row][col] == 255:
                pixels += 1
        # head_pixel = 채워진 머리인지, 빈 머리인지 구분 짓지않음
        if pixels > functions.w(10):
            head_pixel += 1
            head_pixel_max = max(head_pixel_max, pixels)
        col_range = (head_area_left, head_area_right)
        col, pixels = functions.count_line_pixels(image, True, row, col_range, 10)
        # head_fill_pixel = 끊기지 않는 선들만 카운트 (채워진 머리)
        if pixels > functions.w(10):
            head_fill_pixel += 1
            head_fill_pixel_max = max(head_fill_pixel_max, pixels)
    if head_fill_pixel < functions.w(10) and head_pixel_max < functions.w(10) and head_fill_pixel_max < functions.w(5):
        head_fill = 0
    elif head_fill_pixel >= functions.w(10) and head_fill_pixel_max >= functions.w(20):
        head_fill = 1
    else:
        head_fill = -1

    return head_fill, head_pixel


def recognize_note_tail(image, index, stem, stem_direction):
    # 정 방향 음표
    if stem_direction:
        # 음표 꼬리가 있는지, 몇 개인지 탐색할 위치
        wing_area_top = stem[1]
        wing_area_bot = stem[1] + stem[3] - functions.w(20)
    # 역 방향 음표
    else:
        # 음표 꼬리가 있는지, 몇 개인지 탐색할 위치
        wing_area_top = stem[1] + functions.w(20)
        wing_area_bot = stem[1] + stem[3]
    if index:
        wing_area_col = stem[0] - functions.w(7)
    else:
        wing_area_col = stem[0] + stem[2] + functions.w(7)
    pixels = 0
    for row in range(wing_area_top, wing_area_bot):
        if image[row][wing_area_col] == 255:
            pixels += 1
    if pixels < functions.w(6):
        tail_cnt = 0
    elif pixels < functions.w(16):
        tail_cnt = 1
    elif pixels < functions.w(24):
        tail_cnt = 2
    else:
        tail_cnt = -1

    return tail_cnt


def recognize_note_point(image, stem, stem_direction):
    point_area_top = stem[1] + stem[3] - functions.w(10)
    point_area_bot = stem[1] + stem[3] + functions.w(10)
    # 정 방향 음표
    if stem_direction:
        # 점 음표 인지, 아닌지 탐색할 위치
        point_area_left = stem[0] + stem[2] + functions.w(5)
        point_area_right = stem[0] + stem[2] + functions.w(15)
    # 역 방향 음표
    else:
        # 점 음표 인지, 아닌지 탐색할 위치
        point_area_left = stem[0] + stem[2] + functions.w(30)
        point_area_right = stem[0] + stem[2] + functions.w(40)
    point_rect = (
        point_area_left,
        point_area_top,
        point_area_right - point_area_left,
        point_area_bot - point_area_top
    )
    pixels = functions.count_rect_pixels(image, point_rect)
    if pixels < functions.w(14):
        point_exist = 0
    elif pixels < functions.w(40):
        point_exist = 1
    else:
        point_exist = -1

    return point_exist


def recognize_rest(obj):
    pass
