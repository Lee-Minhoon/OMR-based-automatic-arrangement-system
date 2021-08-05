# Main.py
import cv2
import os
import numpy as np
import functions as fs
import modules

# 이미지 불러오기
resource_path = os.getcwd() + "/resource/"
image_0 = cv2.imread(resource_path + "nmusic15.jpg")

# 1. 보표 영역 추출 및 그 외 노이즈 제거
image_1 = modules.remove_noise(image_0)

# 2. 오선 제거
image_2, staves = modules.remove_staves(image_1)

# 3. 악보 이미지 정규화
image_3, staves = modules.normalization(image_2, staves, 10)

# 4. 객체 검출 과정
image_4, objects = modules.object_detection(image_3, staves)

# 5. 객체 분석 과정
image_5, objects = modules.object_analysis(image_4, objects)

for i in range(len(objects)):
    obj = objects[i]
    line = obj[0]
    stats = obj[1]
    stems = obj[2]
    direction = obj[3]
    print(line, stats, stems, direction)
    x, y, w, h, area = stats
    cv2.rectangle(image_5, (x, y, w, h), (255, 0, 0), 1)

# 이미지 띄우기
cv2.imshow('image', image_4)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
