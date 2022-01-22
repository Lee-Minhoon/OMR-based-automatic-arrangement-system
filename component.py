import cv2 as cv


class Component:
    def __init__(self, path):
        self._path = path
        self._data = cv.imread(self._path)

    def show(self):
        cv.imshow(self._path)
        k = cv.waitKey(0)
        if k == 27:
            cv.destroyAllWindows()
