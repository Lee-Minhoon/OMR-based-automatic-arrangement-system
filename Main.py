import copy
import cv2
import numpy as np
import os
import modules

# 악보 이미지 로드
resource_path = os.getcwd() + "/resource/"
image = cv2.imread(resource_path + "nmusic3.jpg")

# 전처리 과정 1. 오선 영역 밖 노이즈 제거
image = modules.remove_noise(image)

# 전처리 과정 2. 오선 제거
image, staves = modules.remove_staves(image)

# 전처리 과정 3. 악보 이미지에 가중치를 곱해줌
image, staves = modules.normalization(image, staves, 8)

# 객체 검출
image = modules.object_detection(image, staves)

# 이미지 띄우기
# image = cv2.resize(image, dsize=(0, 0), fx=0.4, fy=0.4)
cv2.imshow('image', image)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
