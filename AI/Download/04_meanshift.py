import numpy as np, cv2

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