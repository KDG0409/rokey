# 녹색 필터링 하기

import cv2
import numpy as np

image = cv2.imread('C:/rokey/python/ch21/task/sample.jpg')

green_lower = np.array([35,100,100])
green_upper = np.array([85,255,255])

hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv,green_lower,green_upper)
result = cv2.bitwise_and(image,image,mask=mask1)

cv2.imshow('Green Color Filtering',result)
cv2.waitKey(0)
cv2.destroyAllWindows()