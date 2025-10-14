# 객체 이미지 검출
import cv2
import matplotlib.pyplot as plt

# 이미지 로드/사전학습 파일
image = cv2.imread("C:/rokey/python/ch21/task/people.jpg")
face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_path)

resized = cv2.resize(image,(300,300))

# 이미지 흑백 전환
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# 얼굴 검출 수행
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))

# 검출된 얼굴에 사각형 그리기
for (x,y,w,h) in faces:
    cv2.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)

# 결과 출력
cv2.imshow('Face Detection',resized)
cv2.waitKey(0)
cv2.destroyAllWindows()