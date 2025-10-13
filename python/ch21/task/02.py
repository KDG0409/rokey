import cv2

# 이미지 로드
image = cv2.imread('C:/rokey/python/ch21/task/sample.jpg')
# cv2.imshow('Loaded Image', image)

# 이미지 크기 조정
resized = cv2.resize(image,(300,300))
# cv2.imshow('Loaded Image', resized)

# 색상_강도 변환
# gray = cv2.cvtColor(resized,cv2.COLOR_BAYER_BG2GRAY)
# cv2.imshow('Grayscale Image', gray)

# 이미지 회전
# (h, w) = resized.shape[:2] # 이미지 크기
# center = (w // 2, h // 2) # 회전 중심 좌표
# M = cv2.getRotationMatrix2D(center, 45, 1.0) # 변환 행렬(각도,확대비율)
# rotated = cv2.warpAffine(resized, M, (w, h)) #(기존이미지,변환행렬,크기)로 이미지 생성
# cv2.imshow('Rotated Image', rotated)

# 엣지 검출(흑백)
# edges = cv2.Canny(resized, 100, 200) #(최소,최대 입계값)
# cv2.imshow('Canny Edge Detection', edges)

# 블러 처리(흐림)
blurred = cv2.GaussianBlur(resized, (15, 15), 0) #(가우시안 커널 크기,표준편차)
cv2.imshow('Gaussian Blur', blurred)

#창 종료 조건
cv2.waitKey(0)
cv2.destroyAllWindows()
