# 실시간 비디오 영상 처리

import cv2
cap=cv2.VideoCapture(0)
while True:
    ret,frame = cap.read() #비디오에서 리딩여부,Mat클래스 변수 입력
    if not ret:
        break
    edges = cv2.Canny(frame,100,200) # 실시간 엣지 수행
    cv2.imshow('Edge Detection',edges) 
    if cv2.waitKey(1) == ord('q'): # q클릭 시 종료
        break
cap.release()
cv2.destroyAllWindows()