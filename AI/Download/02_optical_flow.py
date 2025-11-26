import numpy as np, cv2

# --- 1. 초기 설정 및 변수 정의 ---
video_path = './newyork.mp4' # 처리할 비디오 파일 경로
cap = cv2.VideoCapture(video_path) # 비디오 캡처 객체 생성

# 비디오 파일이 제대로 열렸는지 확인
if not cap.isOpened():
    print("오류: 비디오 파일을 열 수 없습니다.")
    exit()

# 프레임 재생 속도 조절을 위한 딜레이 (1000ms / 30fps = 약 33ms)
delay = int(1000/30) 

# 추적 경로를 그리기 위한 랜덤 색상 (200개 코너점에 대응하는 색상)
# np.random.randint(0, 255, (200, 3)) : 0~255 사이의 3채널(BGR) 색상 200개 생성
color = np.random.randint(0, 255, (200, 3))

lines = None    # 추적 선(이동 경로)을 그릴 이미지 저장 변수 (초기화는 첫 프레임에서 진행)
prevImg = None  # 이전 프레임 저장 변수 (그레이스케일 이미지)

# calcOpticalFlowPyrLK() 중지 요건 설정 (Termination Criteria)
# (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 최대 반복 횟수(10), 오차 임계값(0.03))
termcriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)


# --- 2. 비디오 처리 루프 ---
while cap.isOpened():
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