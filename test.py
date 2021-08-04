# Main.py
import cv2
import os
import numpy as np
import functions as fs
import modules

# 이미지 불러오기
resource_path = os.getcwd() + "/resource/"
image_0 = cv2.imread(resource_path + "music.jpg")

# 1. 보표 영역 추출 및 그 외 노이즈 제거
image_1 = modules.remove_noise(image_0)

# 2. 오선 제거
image_2, staves = modules.remove_staves(image_1)

# 3. 악보 이미지 정규화
image_3, staves = modules.normalization(image_2, staves, 10)

# 4. 객체 검출 과정
image_4, objects = modules.object_detection(image_3, staves)

# 5. 객체 분석 과정
for obj in objects:
    stats = obj[1]
    stems = fs.stem_detection(image_4, stats, 60)  # 객체 내의 모든 직선들을 검출함
    direction = None
    if len(stems) > 0:  # 직선이 1개 이상 존재함
        if stems[0][0] - stats[0] >= fs.w(5):  # 직선이 나중에 발견 되면
            direction = True  # 정 방향 음표
        else:  # 직선이 일찍 발견 되면
            direction = False  # 역 방향 음표
    obj.append(stems)  # 객체 리스트에 직선 리스트를 추가
    obj.append(direction)  # 객체 리스트에 음표 방향을 추가

print(obj)

# 이미지 띄우기
cv2.imshow('image', image_4)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
