# Main.py
import cv2
import os
import modules
from image import Image

# 이미지 불러오기
resource_path = os.getcwd() + "/resource/"
image = Image(resource_path + "music.jpg")  # 15

# 1. 보표 영역 추출 및 그 외 노이즈 제거
image.data = modules.remove_noise(image.data)

# 2. 오선 제거
image.data, image.staves = modules.remove_staves(image.data)

# 3. 악보 이미지 정규화
image_3, staves = modules.normalization(image_2, staves, 10)

# 4. 객체 검출 과정
image_4, objects = modules.object_detection(image_3, staves)
# for obj in objects:
#     print(
#         "[line : " + str(obj[0]) +
#         ", rect : (x : " + str(obj[1][0]) +
#         ", y : " + str(obj[1][1]) +
#         ", w : " + str(obj[1][2]) +
#         ", h : " + str(obj[1][3]) +
#         ")]"
#     )

# 5. 객체 분석 과정
image_5, objects = modules.object_analysis(image_4, objects)
# for obj in objects:
#     print(
#         "[line : " + str(obj[0]) +
#         ", rect : (x : " + str(obj[1][0]) +
#         ", y : " + str(obj[1][1]) +
#         ", w : " + str(obj[1][2]) +
#         ", h : " + str(obj[1][3]) +
#         "), stems : " + str(len(obj[2])) +
#         ", stem direction : " + str(obj[3]) +
#         "]"
#     )

# 6. 객체 인식 과정
image_6, key, beats, pitches = modules.recognition(image_5, staves, objects)

# 이미지 띄우기
image.show()
