import cv2
import numpy as np
import os

# 악보 불러오기
resource_path = os.getcwd() + "/resource/"
img_0 = cv2.imread(resource_path + "music.jpg")

# 그레이스케일 및 이진화
img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
ret, img_0 = cv2.threshold(img_0, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# 모든 컨투어 구하기
height, width = img_0.shape
contours, hierarchy = cv2.findContours(img_0, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img_0 = cv2.cvtColor(img_0, cv2.COLOR_GRAY2RGB)

print(height, width)
print(img_0[1017, 736])

# # 컨투어에 바운딩된 사각형의 넓이가 이미지 넓이의 70% 이상이라면 오선 영역으로 판단
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     if w >= width * 0.7:
#         img_0 = cv2.rectangle(img_0, (x, y), (x + w, y + h), (255, 0, 0), 1)
#
# # 오선 영역이 아닌 부분을 제거
# img_1 = cv2.imread(resource_path + "music.jpg")
# img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
# ret, img_1 = cv2.threshold(img_1, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# for rows in range(height):
#     for cols in range(width):
#         if np.array_equal(img_0[rows, cols], (255, 0, 0)):
#             break
#         else:
#             img_1[rows, cols] = 255
#
# # 오선 제거
# staves = []
# for rows in range(height):
#     histogram = 0
#     for cols in range(width):
#         if img_1[rows, cols] == 0:
#             histogram = histogram + 1
#     if histogram >= width * 0.5:
#         if len(staves) == 0:
#             staves.append([rows, 0])
#         elif abs(staves[-1][0] - rows) > 1:
#             staves.append([rows, 0])
#         else:
#             staves[-1][0] = rows
#             staves[-1][1] = staves[-1][1] + 1
#
# for rows in range(len(staves)):
#     for cols in range(width):
#         topPixel = staves[rows][0] - staves[rows][1]
#         botPixel = staves[rows]
#         if topPixel != 0 and botPixel != 0:
#             for index in range(staves[rows][1]):
#                 img_1[rows - staves[rows][index]][cols] = 255

# 이미지 띄워보기
cv2.imshow('image', img_0)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()