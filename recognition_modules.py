import functions as fs

# 조표 인식 함수
# ==============================================================
'''
조표는 첫 번째 객체인 음자리표 뒤 두 번째 객체로 위치해 있다.
하지만 다 장조의 경우 조표가 없기 때문에 조표가 있을 위치에 박자표가
위치 해있다. 두 번째 객체가 확실하게 박자표라면 해당 악곡은 다장조이다.
'''
# ==============================================================
def recognize_key(image, staves, rect):
    ts_top_condition = staves[0] + fs.w(5) >= rect[1] >= staves[0] - fs.w(5)  # 박자표 상단 조건
    ts_bot_condition = staves[4] + fs.w(5) >= rect[1] + rect[3] >= staves[4] - fs.w(5)  # 박자표 하단 조건
    ts_center_condition = staves[2] + fs.w(5) >= rect[1] + rect[1] + rect[3] / 2 >= staves[2] - fs.w(5)  # 박자표 중앙 조건
    if ts_top_condition and ts_bot_condition and ts_center_condition:
        ts_width_condition = fs.w(35) >= rect[2] >= fs.w(20)  # 박자표 넓이 조건
        ts_height_condition = fs.w(90) >= rect[3] >= fs.w(75)  # 박자표 높이 조건
        if ts_width_condition and ts_height_condition:
            if fs.count_rect_pixels(image, rect) > fs.w(800):  # 박자표 조건에 부합하기 때문에 조표가 아님
                key = 0
    else:  # 조표가 있을 경우 (다장조를 제외한 모든 조)
        stems = fs.stem_detection(image, rect, 30)
        flat_top_condition = staves[0] + fs.w(10) > stems[0][1] > staves[0] - fs.w(10)
        flat_bot_condition = staves[3] > stems[0][1] + stems[0][3] > staves[2]
        if flat_top_condition and flat_bot_condition:  # 첫 직선의 위치가 플랫이 처음 놓이게 되는 위치라면
            fs.put_text(image, "b" + str(len(stems)), (rect[0], rect[1] + rect[3] + fs.w(60)))
            key = 10 + len(stems)

        else:  # 첫 직선의 위치가 샾이 처음 놓이게 되는 위치라면
            fs.put_text(image, "#" + str(int(len(stems) / 2)), (rect[0], rect[1] + rect[3] + fs.w(60)))
            key = 20 + len(stems) / 2

    return key


# 음표 인식 함수
# ==============================================================
'''
음표는 객체에서 추출할 수 있는 여러 특징점들을 조합해 분류할 수 있다.
음표로 추정되는 객체를 분류하기 위해선 4가지 특징점들을 추출해야 한다.
각 머리(head), 기둥(stem), 꼬리(tail), 점(dot)이다.
'''
# ==============================================================
def recognize_note(image, rect, stems, stem_direction):
    note = []
    if rect[2] > fs.w(20) and rect[3] > fs.w(60):  # 음표로 가정할 수 있는 최소 넓이, 높이 조건 (온 음표, 2분 음표 제외)
        for i in range(len(stems)):
            stem = stems[i]
            head_pixel, head_fill = recognize_note_head(image, stem, stem_direction)
            if head_pixel > fs.w(15):  # 음표 머리에 해당하는 부분이 있다고 판단되면
                tail_cnt = recognize_note_tail(image, i, stem, stem_direction)  # 음표 꼬리 개수
                dot_exist = recognize_note_dot(image, stem, stem_direction)  # 점 존재 여부
                if not head_fill and tail_cnt == 0 and dot_exist == 0:  # 2분음표
                    note.append(2)
                elif not head_fill and tail_cnt == 0 and dot_exist == 1:  # 점2분음표
                    note.append(-2)
                elif head_fill and tail_cnt == 0 and dot_exist == 0:  # 4분음표
                    note.append(4)
                elif head_fill and tail_cnt == 0 and dot_exist == 1:  # 점4분음표
                    note.append(-4)
                elif head_fill and tail_cnt == 1 and dot_exist == 0:  # 8분음표
                    note.append(8)
                elif head_fill and tail_cnt == 1 and dot_exist == 1:  # 점8분음표
                    note.append(-8)
                elif head_fill and tail_cnt == 2 and dot_exist == 0:  # 16분음표
                    note.append(16)
                elif head_fill and tail_cnt == 2 and dot_exist == 1:  # 점16분음표
                    note.append(-16)
                if len(note):  # 음표로 분류됨
                    fs.put_text(image, str(note[i]), (stem[0] - fs.w(20), stem[1] + stem[3] + fs.w(60)))

    return note


# 음표 머리 인식 함수
# ==============================================================
'''
음표의 머리는 음표의 기둥(stem) 위치를 이용해 탐색해볼 수 있다.
정 방향 음표는 음표의 기둥 왼쪽 아래, 역 방향 음표는 음표의 기둥 오른쪽 위에
존재한다. 해당 부분을 히스토그램을 통해 탐색한다면 머리가 존재하는지,
존재한다면 채워져 있는지 비었는지 분류할 수 있다.
'''
# ==============================================================
def recognize_note_head(image, stem, stem_direction):
    if stem_direction:  # 정 방향 음표
        head_area_top = stem[1] + stem[3] - fs.w(20)  # 음표 머리를 탐색할 위치 (상단)
        head_area_bot = stem[1] + stem[3] + fs.w(20)  # 음표 머리를 탐색할 위치 (하단)
        head_area_left = stem[0] - fs.w(20)  # 음표 머리를 탐색할 위치 (좌측)
        head_area_right = stem[0]  # 음표 머리를 탐색할 위치 (우측)
    else:  # 역 방향 음표
        head_area_top = stem[1] - fs.w(20)  # 음표 머리를 탐색할 위치 (상단)
        head_area_bot = stem[1] + fs.w(20)  # 음표 머리를 탐색할 위치 (하단)
        head_area_left = stem[0] + stem[2]  # 음표 머리를 탐색할 위치 (좌측)
        head_area_right = stem[0] + stem[2] + fs.w(20)  # 음표 머리를 탐색할 위치 (우측)
    head_pixel = 0  # head_pixel = 채워져있는 머리인지, 비어있는 머리인지 구분 짓지않고 픽셀의 개수를 셈
    head_pixel_max = 0  # head_pixel_max = head_pixel 중 가장 큰 값
    head_fill_pixel = 0  # head_fill_pixel = 끊기지 않고 이어져 있는 선의 픽셀 개수를 셈 (채워진 머리)
    head_fill_pixel_max = 0  # head_fill_pixel_max = head_fill_pixel 중 가장 큰 값
    for row in range(head_area_top, head_area_bot):
        pixels = 0
        for col in range(head_area_left, head_area_right):
            if image[row][col] == 255:
                pixels += 1
        if pixels >= fs.w(10):
            head_pixel += 1
            head_pixel_max = max(head_pixel_max, pixels)
        col_range = (head_area_left, head_area_right)
        col, pixels = fs.count_line_pixels(image, True, row, col_range, 10)
        if pixels >= fs.w(10):
            head_fill_pixel += 1
            head_fill_pixel_max = max(head_fill_pixel_max, pixels)
    if head_fill_pixel < fs.w(10) and head_pixel_max < fs.w(10) and head_fill_pixel_max < fs.w(5):  # 머리가 비어있음
        head_fill = 0
    elif head_fill_pixel >= fs.w(10) and head_fill_pixel_max >= fs.w(20):  # 머리가 채워져있음
        head_fill = 1
    else:  # 머리로 분류할 수 없음
        head_fill = -1

    return head_pixel, head_fill


# 음표 꼬리 인식 함수
# ==============================================================
'''
음표의 꼬리는 음표의 기둥(stem) 위치를 이용해 탐색해볼 수 있다.
정 방향 음표는 음표의 기둥 오른쪽 위에, 역 방향 음표는 음표의 기둥 오른쪽 아래에
존재한다. 해당 부분을 히스토그램을 통해 탐색한다면 꼬리가 존재하는지,
존재한다면 몇개가 있는지 분류할 수 있다.
'''
# ==============================================================
def recognize_note_tail(image, index, stem, stem_direction):
    if stem_direction:  # 정 방향 음표
        tail_area_top = stem[1]  # 음표 꼬리를 탐색할 위치 (상단)
        tail_area_bot = stem[1] + stem[3] - fs.w(20)  # 음표 꼬리를 탐색할 위치 (하단)
    else:  # 역 방향 음표
        tail_area_top = stem[1] + fs.w(20)  # 음표 꼬리를 탐색할 위치 (상단)
        tail_area_bot = stem[1] + stem[3]  # 음표 꼬리를 탐색할 위치 (하단)
    if index:
        tail_area_col = stem[0] - fs.w(7)  # 음표 꼬리를 탐색할 위치 (열)
    else:
        tail_area_col = stem[0] + stem[2] + fs.w(7)  # 음표 꼬리를 탐색할 위치 (열)
    pixels = 0
    for row in range(tail_area_top, tail_area_bot):
        if image[row][tail_area_col] == 255:
            pixels += 1
    if pixels < fs.w(6):  # 꼬리가 발견되지 않음
        tail_cnt = 0
    elif pixels < fs.w(16):  # 꼬리 1개
        tail_cnt = 1
    elif pixels < fs.w(24):  # 꼬리 2개
        tail_cnt = 2
    else:  # 꼬리로 분류할 수 없음
        tail_cnt = -1

    return tail_cnt


# 음표 점 인식 함수
# ==============================================================
'''
음표의 점은 음표의 기둥(stem) 위치를 이용해 탐색해볼 수 있다.
정 방향 음표는 음표의 기둥 오른쪽 아래에, 역 방향 음표는 음표의 기둥 오른쪽 위에
존재한다. 해당 부분을 탐색해 픽셀의 개수를 살펴보면 점이 존재하는지 알 수 있다.
'''
# ==============================================================
def recognize_note_dot(image, stem, stem_direction):
    dot_area_top = stem[1] + stem[3] - fs.w(10)  # 음표 점을 탐색할 위치 (상단)
    dot_area_bot = stem[1] + stem[3] + fs.w(10)  # 음표 점을 탐색할 위치 (하단)
    if stem_direction:  # 정 방향 음표
        dot_area_left = stem[0] + stem[2] + fs.w(5)  # 음표 점을 탐색할 위치 (좌측)
        dot_area_right = stem[0] + stem[2] + fs.w(15)  # 음표 점을 탐색할 위치 (우측)
    else:  # 역 방향 음표
        dot_area_left = stem[0] + stem[2] + fs.w(30)  # 음표 점을 탐색할 위치 (좌측)
        dot_area_right = stem[0] + stem[2] + fs.w(40)  # 음표 점을 탐색할 위치 (우측)
    dot_rect = (
        dot_area_left,
        dot_area_top,
        dot_area_right - dot_area_left,
        dot_area_bot - dot_area_top
    )
    pixels = fs.count_rect_pixels(image, dot_rect)
    if pixels < fs.w(14):  # 점이 발견되지 않음
        dot_exist = 0
    elif pixels < fs.w(40):  # 점이 발견됨
        dot_exist = 1
    else:  # 점으로 분류할 수 없음
        dot_exist = -1

    return dot_exist


# 쉼표 인식 함수
# ==============================================================
'''
쉼표는 보표에서 항상 고정된 위치에 있기 때문에, 이를 이용해 쉽게 분류할 수 있다.
그 외 픽셀의 개수등 다양한 조건을 추가해 좀 더 엄격하게 탐색해볼 수 있다.
'''
# ==============================================================
def recognize_rest(image, staves, rect):
    rest = 0
    center = (rect[1] + rect[1] + rect[3]) / 2  # 객체의 중간 y 좌표
    rest_condition = staves[3] > center > staves[1]
    if rest_condition:
        # fs.put_text(image, fs.count_rect_pixels(image, rect), (rect[0], rect[1] + rect[3] + fs.w(60)))
        # fs.put_text(image, rect[3], (rect[0], rect[1] + rect[3] + fs.w(60)))
        if fs.w(30) >= rect[2] >= fs.w(25) and fs.w(20) >= rect[3] >= fs.w(14):  # 온쉼표와 2분쉼표의 넓이, 높이 조건
            if fs.w(470) >= fs.count_rect_pixels(image, rect) >= fs.w(370):  # 온쉼표와 2분쉼표의 픽셀 개수 조건
                whole_rest_condition = staves[1] + fs.w(10) >= center >= staves[1]
                half_rest_condition = staves[2] >= center >= staves[1] + fs.w(10)
            if whole_rest_condition:
                rest = 1
            if half_rest_condition:
                rest = 2
        elif fs.w(25) >= rect[2] >= fs.w(20) and fs.w(65) >= rect[3] >= fs.w(50):  # 4분쉼표의 넓이, 높이 조건
            if fs.w(800) >= fs.count_rect_pixels(image, rect) >= fs.w(550):  # 4분쉼표의 픽셀 개수 조건
                rest = 4
        elif fs.w(35) >= rect[2] >= fs.w(25) and fs.w(60) >= rect[3] >= fs.w(40):
            if fs.w(150) >= fs.count_rect_pixels(image, rect) >= fs.w(110):
                rest = 8
    if rest:
        fs.put_text(image, "r" + str(rest), (rect[0], rect[1] + rect[3] + fs.w(60)))

    return rest
