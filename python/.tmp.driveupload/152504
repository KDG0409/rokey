import cv2

people = cv2.imread("C:/rokey/python/ch21/task/sample2.jpg")
resized=cv2.resize(people,(500,500))
gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
edge =cv2.Canny(resized,100,200)
cv2.imshow('people',edge)
cv2.waitKey(0)
cv2.destroyAllWindows()