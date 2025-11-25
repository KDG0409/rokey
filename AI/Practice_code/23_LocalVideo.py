import cv2
import matplotlib.pyplot as plt
video_path = r'/content/drive/MyDrive/두산로보틱스_딥러닝_컴퓨터비전/컴퓨터 비전 응용/교육생제공용/강의_6기_AI응용_3차시_openCV_영상의 특징 검출/bird.mp4'

# 동영상 객체 생성
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("ERR : 에러 발생(비디오 파일을 찾을 수가 없습니다)")
    exit()

# 동영상 속성 확인
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f'fps: {fps}, width: {width},height: {height},frame_count: {frame_count},')

# 동영상 읽기
while True:
  result, frame  = cap.read() # result: bool T/F 반환, frame: 실제 이미지 데이터(사진 한 장)
  if not result: # 마지막 프레임인 경우 종료
    break        

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 그레이스케일 변환
  cv2.imshow(gray)
  if cv2.waitKey(1) & 0xFF == ord('q'): # 1초동안 'q'키를 누르면 종료
    break
  
cap.release() # 동영상 자원 해제
cv2.destroyAllWindows() # 모든 창 닫기

# 동영상 색상 트래킹
import numpy as np
video_path = '/content/drive/MyDrive/두산로보틱스_딥러닝_컴퓨터비전/컴퓨터 비전 응용/교육생제공용/강의_6기_AI응용_3차시_openCV_영상의 특징 검출/greenball.mp4'

def track(image): # BGR이미지-> 초록색 물체 중심좌표 추적
    blur = cv2.GaussianBlur(image, (5,5),0) # 노이즈 제거 : 가우시안 블러
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) # BGR >> HSV 색 공간 변환(Value기반) : 색 기반 객체 추출에 유리/조명영향 최소화/실시간 영상처리 유리
    lower_green = np.array([40, 70, 100]) # 초록색 범위 설정 HSV
    upper_green = np.array([80, 200, 200])
    mask = cv2.inRange(hsv,lower_green,upper_green) # 초록색 부분만 255(흰색), 나머지 0 -> 이진(0,1)
    bmask = cv2.GaussianBlur(mask,(5,5),0) # 마스크(흑백영상) 다시 블러링(노이즈 제거)
    moments = cv2.moment(bmask) # 모멘트(moment):딕셔너리 형태
    m00 = moments['m00'] # m00:한 영역의 픽셀수(면적), 영역 크기 계산

    centroid_x,centroid_y = None,None

    if m00 !=0:
        centroid_x = int(moments['m10']/m00) # m10: 1차 모멘트(무게중심):흰색 x좌표의 합 /m00->중심 x좌표계산
        centroid_y = int(moments['m01']/m00) # m01: 1차 모멘트(무게중심):흰색 y좌표의 합 /m00->중심 y좌표계산
        # m20, m02 : 2차 모멘트(회전,분산,방향)

if __name__ == '__main__': # 프로그램 시작점(이 파일을 직접 실행할 때만 동작)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print('동영상을 열 수 없습니다. 경로를 확인하세요.')
    else:
      frame_idx = 0

      while True:
        okay, image = capture.read()

        # 비디오가 마지막이라면
        if not okay:
          print('영상 끝까지 재생 완료')
          break

        # 초록색 공 추적
        ctr = track(image)
        print(f'Frame {frame_idx}: centroid = {ctr}')

        frame_idx += 1

      capture.release()
      cv2.destroyAllWindows()

# 이진화 특징 검출
image_path = '/content/drive/MyDrive/두산로보틱스_딥러닝_컴퓨터비전/컴퓨터 비전 응용/교육생제공용/강의_6기_AI응용_3차시_openCV_영상의 특징 검출/copy.png'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
result, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY) # 임계값 기준 이진화
# 50보다 어두우면 검정, 밝으면 흰색으로 처리 # binary: 처리된 이미지

binary_ad = cv2.adaptiveThreshold( # 적응형 이진화 : 영역마다 다른 임계값 사용
    gray, # 적용 이미지
    255,  # 최대값(조건만족시)
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # 옵션(적응형 판단기준):주변 픽셀의 가중평균(가우시안)
    cv2.THRESH_BINARY, # 조건(임계값)기준 결과 반환(검/흰)
    121, # 커널 사이즈 (블록크기:주변 영역 크기)
    4 # 상수 (평균/가중평균에서 빼는 값)
)

# 엣지 검출
# 소벨 필터(sobel filter)
# 1차 미분, slope (gradient) (keyword 기억)
# 영상의 밝기 변화량(gradient) 계산 >> 물체 윤곽선(contour), 경계(edge) 추출하는 필터
# 밝기가 급격하게 변하네 >> edge(경계)네, 변화 없네 >> 배경이네 >> 경계의 방향과 강도 계산
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gx = cv2.Sobel(binary_ad, cv2.CV_32F, 1, 0, ksize=3) # gx : x방향 (가로) 변화 감지 >> 세로선 찾기
gy = cv2.Sobel(binary_ad, cv2.CV_32F, 0, 1, ksize=3) # gy : y방향 (세로) 변화 감지 >> 가로선 찾기
mag = cv2.magnitude(gx, gy)
mag = np.uint8(np.clip(mag, 0,255)) # openCV 이해하도록 unsigned integer 8 bit로 변환
# Canny 필터
result = cv2.Canny(binary_ad, 30, 200)

# 코너 검출
# 해리스 코너 검출기
# 윈도우를 모든 방향으로 움직였을 때, 픽셀 값의 변화가 가장 큰 지점을 코너라고 생각
# 코너랑 두 방향으로 모두 강한 밝기 변화량이 존재하는 곳(특징점 추)
# 코너(선과 선이 교차하는 곳) 저기가 코너일 가능성이 얼마나 높지? 점수(score) 맵(map)
# >> score 가 높을 수록 코너일 가능성이 높음
# blocksize=2 (2*2) 영역씩 검사
# k=0.04 민감도 조절 (작을수록 더 많이 검출)
harris = cv2.cornerHarris(np.float32(gray),blockSize=2, ksize=3, k=0.04)
# ksize : 필터, 마스크의 크기 (홀수만 가능) 3*3, 5*5
# blocksize : adaptiveThreshold 같이 적응형 기반 계산 하기 위함 (주변 영역의 이웃의 크기)
# 즉 평균 구할 영역의 크기 (홀수만 가능)
harris_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# 코너 검출 결과값을 0과 255 사이로 정규화(minmax방식), opencv영상표현을 위한 형태 변환
corner = cv2.cvtColor(binary_ad, cv2.COLOR_BGR2RGB)
# 상위 10% 강한 코너만 빨간색(RGB면 파란색) 표시
corner[harris > 0.9*harris_norm.max()] = (0,0,225)

src = cv2.imread(image_path)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# 알고리즘 적용 : goodFeaturesToTrack

# 추적하기 좋은 특징점 찾기
pts = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=50,     # 최대 50개 특징점
    qualityLevel=0.01, # 상위 1%만
    minDistance=10     # 특징점 간 최소거리 10px
)
if pts is not None:
  pts = pts.astype(np.uint8)
  for i in pts:
    x, y = i.ravel() # ravel(): 2차원 배열 >> 1차원으로 펼치기
    cv2.circle(image, (x,y), 5, (0,0,255), -1)

# 허프변환(Hough Trasnform) : 직선,원 검출 가능
image = cv2.imread('/content/drive/MyDrive/두산로보틱스_딥러닝_컴퓨터비전/컴퓨터 비전 응용/교육생제공용/강의_6기_AI응용_3차시_openCV_영상의 특징 검출/water_coins.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 원(동전) 검출
circles = cv2.HoughCircles( gray,                # 이진화된 이미지
                            cv2.HOUGH_GRADIENT,  # 기울기 사용 원을 찾겠다
                            dp=1.2,              # 해상도 비율 (값이 크면 >> 속도 빨라짐, 성능이 떨어짐))
                            minDist=30,          # 검출된 원 사이 최소거리
                            param1 = 100,        # 높은 임계값
                            param2 = 30,         # 투표(voting) 임계값
                            minRadius = 10,      # 최소 반지름
                            maxRadius = 50       # 최대 반지름
                            )

# 투표(voting) : 많은 점이 원을 지지하면 원으로 인정함 (픽셀관점)
if circles is not None:
   circles = np.around(circles).astype(np.uint16)

   for (x, y, r) in circles[0, :]: # 3차원 형태로 존재함
    # 원 그림에서 배열의 첫번째 차원(보통 1) 무시
    # circles.shape는 3차원 배열(1,N,3): 1은 고정(배치=1처리)
    # 세번째 차원(3) : 3가지 정보(0: 원의 중심 x좌표, 1: 원의 중심 y좌표, 2: 원의 반지름)
      cv2.circle(image, (x,y), r, (0,255,0),2)
      cv2.circle(image, (x,y), 2, (255,0,0),2)

plt.imshow(image)

# 확률적 직선 검출
# >> 실제 길이 선분 형태로 반환(실무)
# cf.  cv2.HoughLines 무한 직선 형태
edges = cv2.Canny(gray,50,150) # 엣지 검출 
lines = cv2.HoughLinesP(edges,         # 엣지의 모음
                        1,             # rho 해상도
                        np.pi/180,     # theta 해상도(각도)
                        threshold=100, # 직선으로 간주될 수 있는 최소값
                        minLineLength=10,  # 내가 검출하려는 직선의 최소 길이
                        maxLineGap=10)     # 직선으로 간주되는 간격
if lines is not None:
  for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1,y1), (x2,y2), (0,0,255), 2)

plt.imshow(image)