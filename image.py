import cv2 as cv
import numpy as np


class Image:
    def __init__(self, path):
        self._path = path
        self._data = cv.imread(self._path)
        self._staves = []
        self._objects = []

    def show(self):
        cv.imshow('Image', self._data)
        k = cv.waitKey(0)
        if k == 27:
            cv.destroyAllWindows()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def staves(self):
        return self._staves

    @staves.setter
    def staves(self, staves):
        self._staves = staves

    def threshold(self):
        self._data = cv.cvtColor(self._data, cv.COLOR_BGR2GRAY)
        _, self._data = cv.threshold(self._data, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    def remove_noise(self):
        self.threshold()
        mask = np.zeros(self._data.shape, np.uint8)  # 보표 영역만 추출하기 위해 마스크 생성

        cnt, labels, stats, centroids = cv.connectedComponentsWithStats(self._data)  # 레이블링
        for i in range(1, cnt):
            x, y, w, h, area = stats[i]
            if w > self._data.shape[1] * 0.5:  # 보표 영역에만
                cv.rectangle(mask, (x, y, w, h), (255, 0, 0), -1)  # 사각형 그리기

        self._data = cv.bitwise_and(self._data, mask)  # 보표 영역 추출