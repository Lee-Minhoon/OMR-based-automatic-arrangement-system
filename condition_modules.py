import functions as fs


def note_condition(staves, rect, lines):
    note_width_condition = obj[2] >= fs.w(35)  # 음표의 넓이 조건
    note_height_condition = obj[3] >= fs.w(80) or fs.w(30) >= obj[3] >= fs.w(20)  # 음표의 높이 조건
    return center


def ts_condition(image, staves, rect):
    top = staves[0] + fs.w(5) >= rect[1] >= staves[0] - fs.w(10)  # 박자표 상단 조건
    bot = staves[4] + fs.w(10) >= rect[1] + rect[3] >= staves[4] - fs.w(5)  # 박자표 하단 조건
    center = staves[2] + fs.w(5) >= fs.get_center(rect) >= staves[2] - fs.w(5)  # 박자표 중앙 조건
    width = fs.w(40) >= rect[2] >= fs.w(30)  # 박자표 넓이 조건
    height = fs.w(100) >= rect[3] >= fs.w(90)  # 박자표 높이 조건
    pixel = fs.count_rect_pixels(image, rect) > fs.w(450)  # 박자표 픽셀 개수
    return top and bot and center and width and height and pixel


def flat_condition(staves, stems):  # 첫 직선의 위치가 플랫이 처음 놓이게 되는 위치라면
    top = staves[0] + fs.w(10) > stems[0][1] > staves[0] - fs.w(10)
    bot = staves[3] > stems[0][1] + stems[0][3] > staves[2]
    return top and bot