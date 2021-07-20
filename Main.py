import copy
from dataclasses import dataclass

import cv2
import numpy as np
import os
import modules

# @dataclass
# class Object:
#     line: list[int]
#     x: list[int]
#     y: list[int]
#     w: list[int]
#     h: list[int]
#
#     def __init__(self, line, rect):
#         self.line = line
#         self.x = rect[0]
#         self.y = rect[1]
#         self.w = rect[2]
#         self.h = rect[3]


# 악보 이미지 로드
resource_path = os.getcwd() + "/resource/"
image = cv2.imread(resource_path + "nmusic3.jpg")

# 전처리 과정 1. 오선 영역 밖 노이즈 제거
image = modules.remove_noise(image)

# 전처리 과정 2. 오선 제거
image, staves = modules.remove_staves(image)

# 전처리 과정 3. 악보 이미지에 가중치를 곱해줌
image, staves = modules.normalization(image, staves, 10)

# 객체 검출
image, objects = modules.object_detection(image, staves)

image, objects = modules.object_analysis(image, objects)
for obj in objects:
    print("line : " + str(obj[0]) + " rect : " + str(obj[1]) + " stems : " + str(obj[2]))

for obj in objects:
    rect = obj[1]
    for row in range(rect[1], rect[1] + rect[3]):
        for col in range(rect[0], rect[0] + rect[2]):
            if image[row][col] == 255:
                print(1, end="")
            else:
                print(image[row][col], end="")
        print()

# 이미지 띄우기
cv2.imshow('image', image)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
