import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

image_path = '/content/drive/MyDrive/두산로보틱스_딥러닝_컴퓨터비전/컴퓨터 비전 심화/교육생제공용/1차시_OpenCV 영상정규화 및  Object Detecion 입문/wafer.jpg'
src = cv2.imread(image_path) # 이미지 로드
dst = cv2.cvtColor(src, cv2.COLOR_BGR2RGB) # 색상 변환

print(f'차원(channel) : {src.ndim}')
print(f'형태(shape): {src.shape}') # (420,420,3) (높이(height), 너비(width), 채널(channel))
print(f'데이터 타입: {src.dtype}') # unit8 unsigned int >> pixel 을 얼마나 자세하게 표현하는가?

dst2 = cv2.add(src, 100)
dst3 = cv2.subtract(src, 100)
dst4 = cv2.multiply(src, 2)
s = 0.5
# s < 1 : 이미지가 더 밝아지고, 대비(contrast)가 강해짐 => 이미지가 뚜렷해짐
# s > 1 : 이미지가 더 어두워지고, 대비가 약해짐
dst5 = cv2.devide(src, s)

dst6 = np.empty(src.shape, src.dtype) # 빈이미지생성
for y in range(src.shape[0]): # 이미지 재생
  for x in range(src.shape[1]): # 이미지 재생
    dst[y,x] = src[y,x] + 50 # 이미지 명도 증가

plt.show(dst)

#도형을 생성함
#np.zeros(크기, 데이터타입) = 0이라는 값으로 채워진 '크기' 만큼의 데이터를 만듦
img1 = np.zeros((200,200,3), dtype=np.uint8)
img2 = np.zeros((200,200,3), dtype=np.uint8)
dst6 = cv2.rectangle(img1,(50,50),(150,150),(125,125,30),-1) # (x1,y1),(x2,y2),(색상),두께:-1은 내부채움을 의미
dst7 = cv2.circle(img2,(100,100),70,(40,170,170),-1)

# AND 연산(교집합) : 겹치는 부분만 그림에 넣는다
dst8 = cv2.bitwise_and(img1, img2)

#도형을 생성함
#np.zeros(크기, 데이터타입) = 0이라는 값으로 채워진 '크기' 만큼의 데이터를 만듦
img1 = np.zeros((200, 200), dtype=np.uint8)
img2 = np.zeros((200, 200), dtype=np.uint8)

cv2.rectangle(img1, (50, 50), (150, 150), (125, 125, 30), -1)
cv2.circle(img2, (100,100), 70, (40,170,70), -1)

fig, axes = plt.subplots(1, 2, figsize=(6,3))

axes[0].imshow(img1)
axes[0].set_title('rectangle')

#AND연산(교집합) -> 겹치는 부분만 그림에 남음

axes[1].imshow(img2)
plt.show()

bit_and = cv2.bitwise_and(img1, img2)
bit_and = cv2.add(bit_and, 100)
fig, axes = plt.subplots(1, 2, figsize=(6,3))

axes[0].imshow(img1, cmap='gray')
axes[0].set_title('rectangle')

axes[1].imshow(bit_and, cmap='gray')
axes[1].set_title('changes')
plt.show()

#OR연산(합집합) -> 두 쪽 중 어느 하나라도 포함되는 픽셀은 표현

img1 = np.zeros((200, 200), dtype=np.uint8)
img2 = np.zeros((200, 200), dtype=np.uint8)

cv2.rectangle(img1, (50, 50), (150, 150), (125, 125, 30), -1)
cv2.circle(img2, (100,100), 70, (40,170,70), -1)

bit_or2 = cv2.bitwise_or(img1, img2)
fig, axes = plt.subplots(1, 2, figsize=(6,3))

axes[0].imshow(img1, cmap='gray')
axes[0].set_title('rectangle')

axes[1].imshow(bit_or2, cmap='gray')
axes[1].set_title('changes')
plt.show()

#xOR연산

img1 = np.zeros((200, 200), dtype=np.uint8)
img2 = np.zeros((200, 200), dtype=np.uint8)

cv2.rectangle(img1, (50, 50), (150, 150), (125, 125, 30), -1)
cv2.circle(img2, (100,100), 70, (40,170,70), -1)

bit_xor = cv2.bitwise_xor(img1, img2)
fig, axes = plt.subplots(1, 2, figsize=(6,3))

axes[0].imshow(img1, cmap='gray')
axes[0].set_title('rectangle')

axes[1].imshow(bit_xor, cmap='gray')
axes[1].set_title('changes')
plt.tight_layout()
plt.show()

#NOT

img1 = np.zeros((200, 200), dtype=np.uint8)
img2 = np.zeros((200, 200), dtype=np.uint8)

cv2.rectangle(img1, (50, 50), (150, 150), (125, 125, 30), -1)

bit_not = cv2.bitwise_not(img1)
fig, axes = plt.subplots(1, 2, figsize=(6,3))

axes[0].imshow(img1, cmap='gray')
axes[0].set_title('rectangle')

axes[1].imshow(bit_not, cmap='gray')
axes[1].set_title('changes')
plt.tight_layout()
plt.show()

# 이미지 히스토그램
image = cv2.imread(image_path)
scr = image.copy()
dst = cv2.cvtColor(scr,cv2.COLOR_BGR2RGB)
dst9 = cv2.cvtColor(scr,cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([dst9],[0],None,[256],[0,256])
# 0번 채널, None: 마스크 설정이 None(전체 이미지 사용)
# [256] bin(계급구간)개수(0-255, 총 256개 구간)
# 'channels'는 분석할 채널 (예: 그레이스케일에서는 [0])을 지정,
# 'histSize'는 히스토그램 구간의 수 (0~255로 256개 구간),
# 'histRange'는 픽셀 값 범위 (0~255)를 지정합니다.

# [0,255] 픽셀의 범위
print(hist[250]) # 밝기 값이 250인 픽셀 개수
print(hist[0,0]) # 행렬로 나타낼 때 첫번째 위치의 값(밝기가 0인 픽셀의 수)

hist_size = [256]
hist_range = [0, 256]
hist = cv2.calcHist([src],[0], None, hist_size, hist_range)
# hist = (bins, 1) → (256, 1)형태
img_back = np.full((100,256),255,dtype = np.uint8)
hist[x,0] # 밝기가 x=0인 픽셀의 수 , 0은 열 인덱스(0밖에 없음)
hist_max = np.max(hist) # 가장 많은 밝기 값의 수

image_background = np.full((100,256),255,dtype=np.uint8)
for x in range(256):
  # 시작점 pt1
  pt1 = (x,100) 
  # 컴퓨터는 y축이 뒤집혀 있어. 높이 100인 그림의 경우 디지털 100 == 사람이 생각하는 0이 됨

  # 끝점 pt2
  pt2 = (x, 100-int(hist[x,0]*100 / hist_max)) # int(hist[x,0]*100 / hist_max)의미는 최댓값 대비 비율*100의 정수값, y좌표
  #cv2.line(이미지, 시작점, 끝점, 색상)
  cv2.line(image_background, pt1, pt2, 0)

plt.imshow(image_background, cmap='gray')
plt.show()

# 마스크 (색상 마스크)
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # HSV(Hue, Satuaration,Value)
lower_blue = np.array([90,50,50])
high_blue = np.array([130,255,255])
mask = cv2.inRange(hsv, lower_blue, high_blue)
result = cv2.bitwise_and(image,image,mask=mask)
dst_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

plt.imshow(dst_rgb)
plt.title('Isolated Blue Areas')
plt.show()

# OpenCV 필터링(블러) : 커널 크기 규칙(홀수만 사용_명확한 중심점이 존재해야함)
image = cv2.imread(image_path)
src = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
dst = cv2.blur(scr,(11,11))
plt.imshow(dst)

dst10 = cv2.GaussianBlur(src, ksize=(3,3), sigmaX=0) # sigmaX = 0 이면 자동계산
plt.imshow(dst10)

# 모폴로지 : 침식(erosion), 팽창(dilate), 열기(opening), 닫기(closing)
#1. 이진화(Binary) - 0, 1 => 흑백 / 임계값(기준값)을 넘으면 255, 모자라면 0
# cv2.threshold(적용할 이미지, 임계값(기준), 기준을 넘으면? : 255로 만들어, 옵션)
result, binary_image = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY) # 127:임계값, 255: 임계값이상 픽셀 변환치
plt.imshow(binary_image)

#2. 커널 만들기 : np.empty(사이즈), np.zeros(사이즈)
# >> 사이즈 크기 만큼 비어있는 객체 생성 / 사이즈 크기 만큼 0으로 채워진 객체 생성
kernel = np.ones((3,3),np.uint8)

# erosion 침식
# >> 흰색 영역(255)의 외곽을 깍아내는 연산
# >> 커널이 완전히 흰색을 포함하는 영역만 유지, 나머지는 검정(0)
erode_image = cv2.erode(binary_image, kernel, iterations=1)

# dialtion 확대(팽창)
# 흰색 영역(255) 넓혀줘요
# 커널이 1개 라도 흰색 만나면 중심 픽셀을 흰색으로 확장
# 객체가 커져요(빈 공간 채워줘요), 끊긴 선 연결(문자, 윤곽선 연결), 구멍 채워줘요
dilate_image = cv2.dilate(binary_image, kernel, iterations=1)

# opening = erosion >> dilation : 노이즈제거 후 모형유지
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# closing = erosion >> dilation : 끊긴 윤곽선 연결, 모형 유지
closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

fig, axes = plt.subplots(1,4, figsize=(6,3))

axes[0].imshow(erode_image)
axes[1].imshow(dilate_image)
axes[2].imshow(opening)
axes[3].imshow(closing)
plt.show()
