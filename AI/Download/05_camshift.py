import numpy as np, cv2

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