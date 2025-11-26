import numpy as np
import cv2 # NumPy와 OpenCV 라이브러리 임포트

# --- 1. 초기 설정 ---
video_path = './newyork.mp4' # 처리할 비디오 파일 경로
cap = cv2.VideoCapture(video_path) # 비디오 캡처 객체 생성

# 비디오 파일이 제대로 열렸는지 확인하는 것이 좋습니다.
if not cap.isOpened():
    print(f"오류: 비디오 파일 '{video_path}'을 열 수 없습니다.")
    exit() # 열 수 없으면 프로그램 종료

# delay = int(1000/30) # 프레임 재생 속도를 맞추기 위한 딜레이 (ms)
# waitKey(1)을 사용하므로 실제 딜레이는 1ms입니다. 이 변수는 사용되지 않습니다.
# 따라서 이 줄은 삭제해도 무방합니다.

# --- 2. 배경 제거 객체 생성 ---
# MOG(Mixture of Gaussians) 알고리즘을 사용한 배경 제거 객체 생성
# cv2.bgsegm 모듈은 'BackgroundSubtractorMOG' 클래스를 제공합니다.
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() 


# --- 3. 비디오 처리 루프 ---
while cap.isOpened():
    ret, frame = cap.read() # 비디오에서 한 프레임 읽기 (ret: 성공 여부, frame: 실제 이미지)
    
    if not ret: # 프레임을 제대로 읽지 못했거나 (예: 비디오의 끝)
        # print("비디오 스트림 종료") # 디버깅을 위해 출력 가능
        break # 루프 종료

    # 배경 제거 마스크 계산
    # frame에서 배경을 제외한 움직이는 객체(전경/Foreground)만 흰색(255)으로 표시된 마스크 생성
    fgmask = fgbg.apply(frame) 
    
    # 결과 화면 표시
    cv2.imshow('Original Frame', frame) # 원본 프레임 표시
    cv2.imshow('Foreground Mask (Background Subtraction)', fgmask) # 전경 마스크 표시
    
    # 키 입력 처리: 1ms 동안 키 입력을 기다림
    # 0xFF는 플랫폼 독립성을 위해 마스크를 씌우는 것이며, 27은 ESC 키의 ASCII 코드입니다.
    if cv2.waitKey(1) & 0xff == 27:
        break # ESC 키를 누르면 루프 종료

 
# --- 4. 종료 및 정리 ---
cap.release() # 캡처 객체 반환 및 해제
cv2.destroyAllWindows() # 열린 모든 OpenCV 창 닫기