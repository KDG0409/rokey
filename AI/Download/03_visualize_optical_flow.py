import numpy as np, cv2

# --- 1. 플로 시각화 함수 정의 ---
def drawFlow(img, flow, step=16):
    """
    Farneback 옵티컬 플로우 결과를 시각화하는 함수.
    특정 간격(step)마다 화살표를 그려 움직임 벡터를 표시합니다.
    """
    h, w = img.shape[:2]

    # 16픽셀 간격의 그리드 인덱스 구하기 (좌표: y, x)
    # step//2: 시작 지점을 정수 나눗셈으로 안전하게 지정
    idx_y, idx_x = np.mgrid[step//2:h:step, step//2:w:step].astype(np.int32)
    # y, x를 묶어 (N, 2) 형태의 좌표 목록 (x, y 순서)으로 재구성
    indices = np.stack((idx_x, idx_y), axis=-1).reshape(-1, 2)

    for x, y in indices:    # 각 그리드 인덱스 순회 (x: 열, y: 행)
        # 1. 각 그리드 인덱스 위치에 시작점(점) 그리기 (녹색)
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        
        # 2. 각 그리드 인덱스에 해당하는 플로 결과 값(이동 거리) 얻기
        # flow 배열은 (H, W, 2) 형태이며, (dy, dx) 또는 (dx, dy)를 저장. Farneback은 (dx, dy) 저장.
        # dx, dy는 float 형태이므로 정수형으로 변환
        dx, dy = flow[y, x].astype(np.int32)
        
        # 3. 각 그리드 인덱스 위치에서 이동한 거리만큼 선(화살표) 그리기
        # 시작점: (x, y), 끝점: (x+dx, y+dy)
        cv2.line(img, (x, y), (x + dx, y + dy), (0, 255, 0), 2, cv2.LINE_AA)
        
        # 참고: 화살표를 그리려면 cv2.arrowedLine() 함수를 사용하면 더 좋습니다.


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
        
    # 옵티컬 플로우는 그레이스케일 이미지에서 계산됨
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 최초 프레임의 경우
    if prev is None:
        prev = gray # 첫 프레임을 '이전 프레임'으로 저장
    else:
        # Farneback 옵티컬 플로우 (Dense Optical Flow) 계산
        # Farneback 알고리즘은 모든 픽셀에 대한 움직임 벡터를 계산합니다.
        # flow 변수는 (H, W, 2) 형태의 NumPy 배열이며, 각 픽셀의 (dx, dy) 벡터를 담고 있습니다.
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 
                                            0.5, # 이미지 피라미드 스케일
                                            3,   # 피라미드 레벨 수
                                            15,  # 윈도우 크기 (평균 이동을 위한 이웃 픽셀 수)
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