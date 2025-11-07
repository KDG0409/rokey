from ultralytics import YOLO
import cv2

# YOLOv8 모델 로드 (사전 학습된 모델 사용)
model = YOLO('yolov8n.pt')  # n: lightweight 버전, s/m/l도 가능

# 카운트할 대상 클래스 지정
target_class = 'person'  # 원하는 객체 (예: 'car', 'dog', 'cell phone' 등)

# YOLO 모델의 클래스 이름들
class_names = model.names

# 대상 클래스의 ID 찾기
target_id = None
for cls_id, name in class_names.items():
    if name == target_class:
        target_id = cls_id
        break

if target_id is None:
    print(f"[오류] '{target_class}' 클래스는 YOLO 모델에 없습니다.")
    exit()

# 웹캠 열기 (기본 카메라: 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 카운트 변수
count = 0
object_detected = False  # 이전 프레임에서 감지 여부

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추론
    results = model(frame, verbose=False)
    boxes = results[0].boxes

    detected = False

    for box in boxes:
        cls_id = int(box.cls)
        if cls_id == target_id:
            detected = True

            # 박스 시각화
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, target_class, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # 이전 프레임에는 없었는데 이번 프레임에 등장했다면 카운트 +1
    if detected and not object_detected:
        count += 1

    object_detected = detected

    # 화면에 카운트 표시
    cv2.putText(frame, f"{target_class} Count: {count}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('YOLO Webcam Object Counter', frame)

    # 'q' 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()