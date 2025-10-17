import cv2

image = cv2.imread(r'C:\rokey\python\ch21\task\sample.jpg')
resized = cv2.resize(image,(500,500))

(h,w) = resized.shape[:2]
center = (w//2,h//2)
m = cv2.getRotationMatrix2D(center,-90,1.0)
rotated = cv2.warpAffine(resized,m,(w,h))

cv2.imshow('Rotated',rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()