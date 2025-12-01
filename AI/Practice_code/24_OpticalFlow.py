import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# 특징자와 매칭

image_path = '/content/drive/MyDrive/두산로보틱스_딥러닝_컴퓨터비전/컴퓨터 비전 응용/교육생제공용/강의_6기_4차시_추적 알고리즘 실습/beau_3.png'
target_path = '/content/drive/MyDrive/두산로보틱스_딥러닝_컴퓨터비전/컴퓨터 비전 응용/교육생제공용/강의_6기_4차시_추적 알고리즘 실습/new_target.png'

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.imread(image_path))
axes[1].imshow(cv2.imread(target_path))
plt.show()

#1.그레이스케일로 변환
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

target = cv2.imread(target_path)
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
target_gray = cv2.resize(target_gray, (200,240)) #타겟의 회전, 크기 변화, 밝기 변화 가 영향을 많이 미침

#2.템플릿의 너비와 높이를 찾음
w, h = target_gray.shape[::-1]

#3.템플릿을 이미지에서 매칭해서 찾아봐!
#매칭 옵션 : SQDIFF(픽셀 제곱차이), CCORR(픽셀 곱의 합), CCOEFF(코사인유사도)  X NORMED(일정한 범위 내로 만듦)
result = cv2.matchTemplate(image_gray, target_gray, cv2.TM_CCOEFF_NORMED)

#매칭 결과에서 최소값, 최대값, 최소값 위치, 최대값 위치를 찾으려고 함
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

#시각화
top_left = max_loc #가능성이 높은 지역의 왼쪽 위 모서리
#top_left = (x, y)형태라서 top_left[0]는 x좌표, top_left[1]는 y좌표
bottom_right = (top_left[0]+w, top_left[1]+h) #왼쪽 위 모서리를 기준으로 +너비, +높이해서 구한 오른쪽 아래 모서리

matched = image.copy()
cv2.rectangle(matched, top_left, bottom_right, (255, 0, 0), 2)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image)
axes[1].imshow(result)
axes[2].imshow(matched)
plt.show()

# 기본설정(배경제거)
# 각 pixel 색상변화를 가우시안 분포(정규분포) 모델링
# 배경(background: bg) 고정되어 있거나 오래 머무는 색상 >> 배경 학습
# 전경(foreground: fg): 갑자기 툭 나타난 색상 : 움직이는 물체로 판단

# 1) 초기 설정
video_path = './newyork.mp4' # 처리할 비디오 파일 경로
cap = cv2.VideoCapture(video_path) # 비디오 캡처 객체 생성
if not cap.isOpened():
    print(f"오류: 비디오 파일 '{video_path}'을 열 수 없습니다.")
    exit() # 열 수 없으면 프로그램 종료

# 2) 배경 제거 객체 생성 : 
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() 

# 3) 비디오 처리 루프
frame_count = 0
MAX_FRAMES_TO_PROCESS = 100 # 테스트를 위해 최대 처리 프레임 수를 100으로 제한
DISPLAY_EVERY_N_FRAMES = 20 # 20 프레임마다 결과 출력

while cap.isOpened() and frame_count < MAX_FRAMES_TO_PROCESS:
    ret, frame = cap.read() # 비디오에서 한 프레임 읽기 (ret: 성공 여부, frame: 실제 이미지)
    
    if not ret: # 프레임을 제대로 읽지 못했거나 (예: 비디오의 끝)
        # print("비디오 스트림 종료") # 디버깅을 위해 출력 가능
        break # 루프 종료

    # 배경 제거 마스크 계산 : frame에서 배경을 제외한 움직이는 객체(전경/Foreground)만 흰색(255)으로 표시된 마스크 생성
    fgmask = fgbg.apply(frame) 

    # 마스크 값
    # 0(검정)>> 배경, 움직이는 않는 부분
    # 255(흰색)>> 전경, 움직이는 개체
    # 127(회색)>> MOG2 가 가진 특별 기능 (그림자)
    
    # 특정 간격의 프레임만 표시 >> 작동 확인
    if frame_count % DISPLAY_EVERY_N_FRAMES == 0:
      # 원본 프레임 표시
      cv2.imshow(frame)
      # 전경 마스크 표시
      cv2.imshow(fgmask)

      time.sleep(1)

    frame_count += 1
    
    if cv2.waitKey(1) & 0xff == 27: # ESC 키를 누르면 루프 종료
        break

# 4) 종료 및 정리
cap.release() # 캡처 객체 반환 및 해제
cv2.destroyAllWindows() # 열린 모든 OpenCV 창 닫기

# OpticalFlow

# 1) 초기 설정 및 변수 정의
video_path = './newyork.mp4' 
cap = cv2.VideoCapture(video_path) 

if not cap.isOpened():
    print("오류: 비디오 파일을 열 수 없습니다.")
    exit()

delay = int(1000/30)  # 프레임 재생 속도 조절을 위한 딜레이 (1000ms / 30fps = 약 33ms)

# 추적 경로를 그리기 위한 랜덤 색상 (200개 코너점에 대응하는 색상)
color = np.random.randint(0, 255, (200, 3)) # 0~255 사이의 3채널(BGR) 색상 200개 생성

lines = None    # 추적 선(이동 경로)을 그릴 이미지 저장 변수 (초기화는 첫 프레임에서 진행)
prevImg = None  # 이전 프레임 저장 변수 (그레이스케일 이미지)

# calcOpticalFlowPyrLK() 중지 요건 설정 (Termination Criteria)
termcriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) # 최대 반복 횟수(10), 오차 임계값(0.03)

# --- 2. 비디오 처리 루프 ---
while cap.isOpened() and frame_count < MAX_FRAMES_TO_PROCESS:
    ret, frame = cap.read() # 비디오에서 한 프레임 읽기
    
    if not ret: # 프레임을 제대로 읽지 못했거나 (예: 비디오 끝)
        break

    # 현재 프레임을 복사하여 추적 결과를 그릴 이미지 준비
    img_draw = frame.copy() 
    # 옵티컬 플로우 계산을 위해 현재 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    # --- 2-1. 최초 프레임 처리 (추적 시작) ---
    if prevImg is None: 
        prevImg = gray # 현재 그레이 이미지를 '이전 이미지'로 저장
        # 추적선을 그릴 검은색 배경 이미지 생성 (원본 프레임과 동일 크기)
        lines = np.zeros_like(frame) 
        # Shi-Tomasi 알고리즘으로 추적을 시작할 코너점 200개 검출
        # (이전 이미지, 최대 코너점 수, 품질 임계값(0.01), 최소 거리(10))
        prevPt = cv2.goodFeaturesToTrack(prevImg, 200, 0.01, 10)

    # --- 2-2. 두 번째 프레임 이후 처리 (추적 진행) ---
    else:
        nextImg = gray # 현재 그레이 이미지를 '다음 이미지'로 설정

        # Lucas-Kanade 옵티컬 플로우 계산
        # prevImg: 이전 이미지, nextImg: 다음 이미지, prevPt: 이전 코너점 목록
        # nextPt: 다음 코너점 (계산 결과), status: 추적 성공 여부 (1=성공, 0=실패), err: 추정 오차
        nextPt, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg,
                                                        prevPt, None, criteria=termcriteria)
        
        # 추적에 성공(status==1)한 코너점만 선별
        prevMv = prevPt[status==1] # 이전 프레임에서 추적 성공한 점
        nextMv = nextPt[status==1] # 현재 프레임에서 대응하는 점

        # 추적 성공한 모든 쌍에 대해 반복
        for i, (p, n) in enumerate(zip(prevMv, nextMv)):
            # 코너점 좌표 추출 (배열 구조 해제)
            px, py = p.ravel()
            nx, ny = n.ravel()

            # 이전 코너(p)와 새로운 코너(n) 사이에 추적 선 그리기 (lines 이미지에 누적)
            # color[i].tolist(): 코너점 i에 할당된 랜덤 색상
            cv2.line(lines, (int(px), int(py)), (int(nx), int(ny)), color[i].tolist(), 2) 
            # 새로운 코너(n)에 원형 점 그리기 (img_draw 이미지에 매 프레임 표시)
            cv2.circle(img_draw, (int(nx), int(ny)), 2, color[i].tolist(), -1)

        # 누적된 추적 선이 그려진 lines 이미지와 현재 프레임(img_draw)을 합성
        # 이로 인해 추적 경로가 비디오 프레임 위에 나타남
        img_draw = cv2.add(img_draw, lines)

        # 다음 루프를 위해 현재 프레임과 코너점을 '이전' 변수로 이월
        prevImg = nextImg
        # prevPt는 nextMv의 형태를 맞춰줘야 함 (N, 1, 2)
        prevPt = nextMv.reshape(-1, 1, 2)


    # 결과 화면 표시
    cv2.imshow('OpticalFlow-LK', img_draw)

    # 키 입력 처리
    key = cv2.waitKey(delay)
    
    if key == 27:    # ESC 키 (ASCII 27): 루프 종료
        break
    elif key == 8:  # Backspace 키 (ASCII 8): 추적 이력 지우기
        # 이전 이미지를 None으로 만들어 다음 루프에서 '최초 프레임' 상태로 되돌림
        prevImg = None 
        # lines 변수는 새로 생성된 lines로 인해 덮어쓰여지므로 명시적 초기화는 불필요하지만,
        # Backspace를 누른 순간 추적 이력을 즉시 지우고 싶다면 lines = np.zeros_like(frame) 추가 가능
        
# --- 3. 종료 및 정리 (수정 완료) ---
# 모든 OpenCV 창 닫기
cv2.destroyAllWindows()
# 비디오 캡처 객체 반환 및 해제
cap.release()

# Visual Optical Flow
# --- 1. 플로 시각화 함수 정의 ---
def drawFlow(img, flow, step=16): # Farneback 옵티컬 플로우 결과를 시각화하는 함수 : 특정 간격(step)마다 화살표를 그려 움직임 벡터를 표시
    h, w = img.shape[:2] # 이미지 크기 [높이,너비,채널] -> [높이,너비]

    # 16픽셀 간격의 그리드 인덱스 구하기 (좌표: y, x)
    # step//2: 시작 지점을 정수 나눗셈으로 안전하게 지정
    idx_y, idx_x = np.mgrid[step//2:h:step, step//2:w:step].astype(np.int32)  # step//2부터 h까지 step간격
    # y, x를 묶어 (N, 2) 형태의 좌표 목록 (x, y 순서)으로 재구성
    indices = np.stack((idx_x, idx_y), axis=-1).reshape(-1, 2)

    for x, y in indices:    # 각 그리드 인덱스 순회 (x: 열, y: 행)
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1) # 1. 각 그리드 인덱스 위치에 시작점(점) 그리기 (녹색)
        
        # 2. 각 그리드 인덱스에 해당하는 플로 결과 값(이동 거리) 얻기
        # flow 배열은 (H, W, 2) 형태이며, (dy, dx) 또는 (dx, dy)를 저장. Farneback은 (dx, dy) 저장.
        # dx, dy는 float 형태이므로 정수형으로 변환
        dx, dy = flow[y, x].astype(np.int32) # 플로 결과 값(이동 거리)
        
        # 3. 각 그리드 인덱스 위치에서 이동한 거리만큼 선(화살표) 그리기
        # 시작점: (x, y), 끝점: (x+dx, y+dy)
        cv2.line(img, (x, y), (x + dx, y + dy), (0, 255, 0), 2, cv2.LINE_AA) 
        
        # 참고: 화살표를 그리려면 cv2.arrowedLine() 함수를 사용하면 더 좋습니다.
        # cv2.line 대신 cv2.arrowedLine() 함수를 사용하여 움직임 벡터를 명확하게 시각화합니다.
        # cv2.arrowedLine(img, (x, y), end_point, arrow_color, 1, cv2.LINE_AA, tipLength=0.3)
        # cv2.LINE_AA : anti-aliasing 부드러운 선 / tipLength: 화살촉크기(전체의0.3)
        # tipLength=0.3

prev = None # 이전 프레임 저장 변수 (그레이스케일)

# --- 2. 비디오 캡처 설정 ---
video_path = './newyork.mp4' # 처리할 비디오 파일 경로
cap = cv2.VideoCapture(video_path) # 비디오 캡처 객체 생성
delay = int(1000/30) # 프레임 재생 속도 조절을 위한 딜레이

# 비디오 파일이 제대로 열렸는지 확인
if not cap.isOpened():
    print("오류: 비디오 파일을 열 수 없습니다.")
    exit()

# --- 3. 비디오 처리 루프 ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 옵티컬 플로우는 그레이스케일 이미지에서 계산됨
    if prev is None: # 최초 프레임의 경우
        prev = gray # 첫 프레임을 '이전 프레임'으로 저장
    else:
        # Farneback 옵티컬 플로우 (Dense Optical Flow) 계산
        # Farneback 알고리즘은 모든 픽셀에 대한 움직임 벡터를 계산합니다.
        # flow 변수는 (H, W, 2) 형태의 NumPy 배열이며, 각 픽셀의 (dx, dy) 벡터를 담고 있습니다.
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 
                                            0.5, # 이미지 피라미드 스케일 (각 단계마다 50% 축소)
                                            3,   # 피라미드 레벨 수
                                            15,  # 윈도우 크기 (평균 이동을 위한 이웃 픽셀 수)
                                            # 크면 부드러워지면서 세밀함 감소 / 작으면 세밀하지만 노이즈 증가
                                            3,   # 반복 횟수
                                            5,   # 다항식 확장 크기
                                            1.1, # 시그마 값
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN) # 플래그
        
        # 계산된 플로 결과 시각화 함수 호출
        drawFlow(frame, flow)
        
        # 다음 루프를 위해 현재 프레임을 '이전 프레임'으로 이월
        prev = gray

    # 결과 화면 표시
    cv2.imshow('OpticalFlow-Farneback', frame)

    # ESC 키 (ASCII 27)를 누르면 루프 종료
    if cv2.waitKey(delay) == 27:
        break

# --- 4. 종료 및 정리 (수정 완료) ---
cap.release()
# 모든 OpenCV 창 닫기 (오류 수정)
cv2.destroyAllWindows()

# meanshift

# --- 1. 변수 초기 설정 ---
roi_hist = None # 추적 대상 객체의 정규화된 히스토그램 저장 변수 (초기값: None)
win_name = 'MeanShift Tracking' # 화면 표시 창 이름

# MeanShift/CamShift 중지 요건 (Termination Criteria)
# (오차(EPS) 또는 반복 횟수(COUNT) 중 하나라도 충족되면 중지)
# 최대 반복 횟수: 10, 허용 오차: 1.0
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# 추적 대상 영역 (바운딩 박스) 좌표를 저장할 전역 변수
# MeanShift가 처음 호출될 때 이 값이 필요하므로, None 대신 0으로 초기화.
x, y, w, h = 0, 0, 0, 0

# --- 2. 비디오 캡처 설정 ---
video_path = './newyork.mp4' # 처리할 비디오 파일 경로
cap = cv2.VideoCapture(video_path) # 비디오 캡처 객체 생성
delay = int(1000/24) # 딜레이 설정 (약 41ms)

# 비디오 파일이 제대로 열렸는지 확인
if not cap.isOpened():
    print("오류: 비디오 파일을 열 수 없습니다.")
    exit()

# --- 3. 비디오 처리 루프 ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    img_draw = frame.copy() # 원본 프레임에 추적 결과를 그리기 위해 복사

    # --- 3-1. 추적 진행 (roi_hist 등록됨) ---
    if roi_hist is not None:
        # 전체 영상 BGR -> HSV 컬러 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 전체 영상에 대해 ROI 히스토그램을 역투영(Back Projection)
        # 현재 픽셀의 h(hue) 값 확인 (예: 30도)
        
        # ROI(관심영역) 히스토그램에서 30도가 얼마나 나오는지(빈도) 확인
        # 그 빈도를 픽셀값으로 설정
        # 그 결과, 밝을 수록 유사한 색상이다라고 간주

        # dst는 각 픽셀이 ROI 색상과 얼마나 유사한지 나타내는 확률 맵이 됨. (0~255)
        # [0]: H(색상) 채널 사용, [0, 180]: H 채널의 범위
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                                 
        # 역투영 결과(dst)와 이전 추적 위치로 평균 이동(Mean Shift) 추적 실행
        # ret: 반복 횟수, (x, y, w, h): 새로운 추적 위치
        ret, (x, y, w, h) = cv2.meanShift(dst, (x, y, w, h), termination)
        
        # 새로운 위치에 초록색 사각형 표시
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 컬러 영상(추적 결과)과 역투영 영상(dst)을 좌우로 통합하여 출력
        # dst는 GRAY이므로 cv2.cvtColor()로 BGR 변환하여 합침
        result = np.hstack((img_draw, cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)))
        
    # --- 3-2. 추적 대기 (roi_hist 등록 안됨) ---
    else:
        # 사용자에게 추적 대상을 설정하라는 안내 텍스트 출력
        cv2.putText(img_draw, 'Hit the Space to set target to track',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        result = img_draw

    # 결과 화면 출력
    cv2.imshow(win_name, result)
    key = cv2.waitKey(delay) & 0xff

    # --- 3-3. 키 입력 처리 ---
    if key == 27:            # ESC 키: 종료
        break
    elif key == ord(' '):    # 스페이스바: ROI 설정
        # 마우스로 추적 영역 설정. 초기 프레임은 원본, 취소 가능(False)
        x_new, y_new, w_new, h_new = cv2.selectROI(win_name, frame, False)
        
        if w_new and h_new:  # ROI가 제대로 설정된 경우
            # 전역 변수 업데이트 (다음 루프의 MeanShift 시작 위치)
            x, y, w, h = x_new, y_new, w_new, h_new
            
            # 1. 초기 추적 대상 영역(ROI) 추출
            roi = frame[y:y + h, x:x + w]
            # 2. ROI를 HSV 컬러로 변경
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 3. ROI의 H(색상) 채널에 대한 히스토그램 계산
            # [0]: H 채널 인덱스, [180]: bin 개수, [0, 180]: H 채널 범위
            roi_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
            # 4. 히스토그램 정규화 (최소 0, 최대 255)
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        else:                # ROI 설정 취소
            roi_hist = None

# --- 4. 종료 및 정리 (수정 완료) ---
cap.release()
# 모든 OpenCV 창 닫기 (오류 수정)
cv2.destroyAllWindows()

# Camshift

# --- 1. 변수 초기 설정 ---
roi_hist = None # 추적 객체 히스토그램 저장 변수
win_name = 'CamShift Tracking'
# CamShift에 필요한 초기 추적 영역 좌표를 전역으로 초기화
x, y, w, h = 0, 0, 0, 0 

# CamShift 중지 요건 (Termination Criteria)
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# --- 2. 비디오 캡처 설정 ---
video_path = './top-down.mp4' # 처리할 비디오 파일 경로
cap = cv2.VideoCapture(video_path) # 비디오 캡처 객체 생성
delay = int(1000/24) # 딜레이 설정

if not cap.isOpened():
    print("오류: 비디오 파일을 열 수 없습니다.")
    exit()

# --- 3. 비디오 처리 루프 ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    img_draw = frame.copy() # 원본 프레임 복사

    # --- 3-1. 추적 진행 (roi_hist 등록됨) ---
    if roi_hist is not None:
        # 1. 전체 영상 BGR -> HSV 컬러 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 2. 마스크 생성: 채도(S)와 명도(V)가 낮은 픽셀을 제외하여 노이즈 제거 (추가된 부분)
        # S가 50보다 크고 V가 50보다 큰 영역만 유효한 색상으로 간주
        target_mask = cv2.inRange(hsv, np.array((0., 50., 50.)), np.array((180., 255., 255.)))
        
        # 3. 전체 영상에 대해 ROI 히스토그램을 역투영(Back Projection)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        
        # 4. 역투영 결과에 마스크를 곱하여 노이즈 영역의 확률을 0으로 만듦 (추가된 부분)
        dst = dst * target_mask 
        
        # 5. 마스크가 적용된 역투영 결과와 이전 추적 위치로 CamShift 추적 실행
        ret, track_window = cv2.CamShift(dst, (x, y, w, h), termination)
        
        # 6. 다음 프레임을 위한 track_window (x, y, w, h) 업데이트
        x, y, w, h = track_window 
        
        # 7. 새로운 위치에 회전된 사각형 표시 (CamShift의 ret 사용)
        pts = cv2.boxPoints(ret) 
        pts = np.int32(pts)      
        cv2.polylines(img_draw, [pts], True, (0, 255, 0), 2)
        
        # 8. 컬러 영상과 역투영 영상을 통합해서 출력
        result = np.hstack((img_draw, cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)))
        
    # --- 3-2. 추적 대기 (roi_hist 등록 안됨) ---
    else:
        # 사용자에게 추적 대상 설정 안내 텍스트 출력
        cv2.putText(img_draw, 'Hit the Space to set target to track',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        result = img_draw
 
    # 결과 화면 출력
    cv2.imshow(win_name, result)
    key = cv2.waitKey(delay) & 0xff
    
    # --- 3-3. 키 입력 처리 ---
    if key == 27:           # ESC 키: 종료
        break
    elif key == ord(' '):   # 스페이스바: ROI 설정
        # 마우스로 추적 영역 설정
        x_new, y_new, w_new, h_new = cv2.selectROI(win_name, frame, False)
        
        if w_new and h_new: # ROI가 제대로 설정됨
            # 전역 변수 업데이트 (다음 루프의 CamShift 시작 위치)
            x, y, w, h = x_new, y_new, w_new, h_new

            # 1. 초기 추적 대상 영역(ROI) 추출 및 HSV 컬러로 변경
            roi = frame[y:y+h, x:x+w]
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 2. 마스크 생성: ROI 히스토그램 계산 시에도 노이즈 픽셀을 제외 (추가된 부분)
            mask = cv2.inRange(roi_hsv, np.array((0., 50., 50.)), np.array((180., 255., 255.)))

            # 3. ROI의 H(색상) 채널에 대한 히스토그램 계산 및 정규화
            # mask를 사용하여 히스토그램 계산
            roi_hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        else:               # ROI 설정 취소
            roi_hist = None
 
# --- 4. 종료 및 정리 ---
cap.release()
cv2.destroyAllWindows()

# Lucas-Kanade

video_path = '/content/drive/MyDrive/두산로보틱스_딥러닝_컴퓨터비전/컴퓨터 비전 응용/교육생제공용/강의_6기_4차시_추적 알고리즘 실습/newyork.mp4'
cap = cv2.VideoCapture(video_path) # 비디오 캡처 객체 생성

# 비디오 파일이 제대로 열렸는지 확인
if not cap.isOpened():
    print(f"오류: 비디오 파일 '{video_path}'을 열 수 없습니다.")
    print("Google Drive가 마운트되었고 경로가 정확한지 확인해 주세요.")
    exit()

# 2. 첫 프레임에서 특징점 찾기
ret, old_frame = cap.read() # 첫 프레임 읽기

# 프레임을 제대로 읽었는지(ret=True) 한 번 더 확인: 이 부분이 기존 오류의 원인일 가능성이 높습니다.
if not ret or old_frame is None:
    print("오류: 첫 번째 프레임을 읽지 못했습니다. 비디오 파일이 손상되었을 수 있습니다.")
    cap.release()
    exit()
# 첫 프레임을 흑백으로 변환
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Shi-Tomasi 코너 검출 알고리즘 사용해 특징점 찾기
corners = cv2.goodFeaturesToTrack(
    old_gray,
    maxCorners=100,      # 최대 특징점 개수
    qualityLevel=0.3,     # 특징점 품질 레벨 (0.0-1.0)
    minDistance=7        # 특징점 간 최소 거리
)

# 3. Lucas-Kanade parameter 설정
lk_params = dict(
            winSize=(15, 15), # 윈도우 사이즈(크기)
            maxLevel=2,       # 피라미드 레벨
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) #종료 조건
        )
# 이미지 피라미드
# 몇 단계까지 사용하지 설정 (0: 원본, 1:1단계 다운 샘플링, 2: 2단계 다운샘플링)

# 4. 비디오 처리 루프 , 광학 흐름(optical flow ) 계산

frame_count = 0
MAX_FRAMES_TO_PROCESS = 150
DISPLAY_EVERY_N_FRAMES = 20

track_frame = None

while cap.isOpened() and frame_count < MAX_FRAMES_TO_PROCESS:
    ret, frame = cap.read()

    if not ret or frame is None:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # optical flow 계산: 이전 프레임(old_gray)의 특징점(corners)이 현재 프레임에서 어디로 이동했는지 추정
    new_corners, status, error = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, corners, None, **lk_params
    )

    # 5. 좋은 점들만 선택 (status = 1 >> 성공적으로 추적된 점)
    good_new = new_corners[status == 1]
    good_old = corners[status == 1]

    # 한번만 초기화
    if track_frame is None:
        track_frame = frame.copy() # first_frame only
    else:
        track_frame = track_frame

    # 6. 움직임 그리기
    # display_frame = frame.copy() # 원본 프레임(사진)에 그릴 복사본 생성

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int) # 현재 위치
        c, d = old.ravel().astype(int) # 이전 위치
        # 픽셀 좌표는 정수여야 하니깐 dtype을 int 로 변환

        # 이전 위치와 현재 위치를 이어주는 초록색 선 그리기
        cv2.line(track_frame, (a, b), (c, d), (0, 255, 0), 2)

        # 현재 위치에 빨간색 점(원) 그리기
        cv2.circle(track_frame, (a, b), 5, (0,0,255), -1)

    if frame_count % DISPLAY_EVERY_N_FRAMES == 0:
        cv2.imshow(track_frame)  # 추적 결과 프레임 표시
        time.sleep(1)

    # 업데이트: 다음 루프를 위해 현재 프레임을 이전 프레임을 설정
    old_gray = frame_gray.copy()
    corners = good_new.reshape(-1, 1, 2) # 자동계산

    frame_count += 1

cap.release()
# cv2.destroyAllWindows()
