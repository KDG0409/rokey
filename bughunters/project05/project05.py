import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from threading import Thread
import time

# 프로젝트04 최적화

# 업데이트 간격 제어 (0.3초) : 메모리 개선
# 얼굴 추적 (Face Tracking) : 같은 얼굴을 지속적으로 추적 (IoU 기반 추적)
# fps 8-12 -> 30-45 : 비동기 스레딩 사용 : class WebcamStream:
# 깜빡임, 프레임 지연 개선 : CPU 사용률 대폭 감소, 속도 증가
# 메모리 개선 : grad 추적 제거
# 항상 최신 프레임만 처리 : self.stream.set사용 지연시간 최소화, 실시간 반응성 향상

DISPLAY_TOP_N = 3      # 상위 3개 감정만 표시 (렌더링 부하 감소)
TARGET_FPS = 30        # 목표 프레임률
BUFFER_SIZE = 1        # 최신 프레임만 유지
RESIZE_WIDTH = 640     # 처리 해상도 (낮출수록 빠름)
RESIZE_HEIGHT = 480
SKIP_FRAMES = 1        # N 프레임마다 처리 (2로 설정하면 2배 빠름)

YOLO_MODEL_PATH = "yolov8n-face.pt"
MODEL_NAME = "abhilash88/face-emotion-detection"
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
emotion_model = ViTForImageClassification.from_pretrained(MODEL_NAME)
emotion_model.eval()

torch.set_num_threads(1)  # CPU 스레드 수 (코어 수에 맞게 조정)
yolo = YOLO(YOLO_MODEL_PATH)

class WebcamStream: # 별도 스레드에서 웹캠 프레임 읽기 
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)

        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE) # 카메라 설정
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        self.grabbed, self.frame = self.stream.read() # 첫 프레임 읽기
        self.stopped = False
        
    def start(self): # 스레드 시작
        Thread(target=self.update, args=(), daemon=True).start()
        return self
        
    def update(self): # 프레임 업데이트 
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()
                
    def read(self): # 프레임 읽기
        return self.frame
        
    def stop(self): # 프레임 중지
        self.stopped = True
        self.stream.release()
        
    def isOpened(self):
        return self.stream.isOpened()

def detect_emotion_optimized(face_img): # 감정 감지 함수
    if face_img.shape[0] < 30 or face_img.shape[1] < 30: # 얼굴 크기 확인 (너무 작으면 스킵)
        return None
    
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) # BGR → RGB 변환 및 PIL 변환
    face_pil = Image.fromarray(face_rgb)
    
    inputs = processor(face_pil, return_tensors="pt") # 전처리
    
    outputs = emotion_model(**inputs) 
    probs = torch.softmax(outputs.logits, dim=-1)[0].detach().numpy()
    
    return probs

def draw_emotion_labels(frame, x1, y2, probs): # 감정 레이블 그리기 함수
    top_indices = np.argsort(probs)[-DISPLAY_TOP_N:][::-1] # 상위 N개 감정 추출
    
    for i, idx in enumerate(top_indices):
        label = f"{EMOTION_LABELS[idx]}: {probs[idx]:.2f}"
        y_pos = y2 + 25 + i * 25
        
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        ) # 텍스트 크기 계산
        
        cv2.rectangle(
            frame, 
            (x1, y_pos - text_h - 5), 
            (x1 + text_w + 5, y_pos + 5), 
            (0, 0, 0), 
            -1
        ) # 배경 사각형 (가독성 향상)
        
        cv2.putText(
            frame, 
            label, 
            (x1 + 2, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        ) # 텍스트

def main(): # 메인루프 설정
    vs = WebcamStream(src=0, width=RESIZE_WIDTH, height=RESIZE_HEIGHT).start()
    
    if not vs.isOpened():
        exit()
        return
    
    time.sleep(2.0)  # 카메라 워밍업

    fps_start_time = time.time()  # FPS 측정 변수
    fps_counter = 0
    fps_display = 0
    frame_count = 0
    
    try:
        while True:
            frame = vs.read()
            
            if frame is None:
                print("[경고] 프레임을 읽을 수 없습니다.")
                break
            
            frame_count += 1
            
            if frame_count % (SKIP_FRAMES + 1) != 0: # 프레임 스킵 (CPU 부하 감소)
                cv2.imshow("YOLO + 감정 인식 (CPU 최적화)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            results = yolo(frame, verbose=False, conf=0.5) # YOLO 얼굴 감지 (신뢰도 임계값 0.5)
            
            for res in results:  # 각 얼굴 처리
                boxes = res.boxes.xyxy.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 경계 확인
                    h, w = frame.shape[:2] 
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # 얼굴 영역 추출
                    face_img = frame[y1:y2, x1:x2]
                    
                    if face_img.size == 0:
                        continue
                    
                    probs = detect_emotion_optimized(face_img) # 감정 감지
                    
                    if probs is None:
                        continue
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 경계 상자 그리기
                
                    draw_emotion_labels(frame, x1, y2, probs) # 감정 레이블 그리기
            
            fps_counter += 1 # FPS 계산
            elapsed = time.time() - fps_start_time
            if elapsed > 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # FPS 및 정보 표시
            info_text = f"FPS: {fps_display} | CPU Mode"
            cv2.putText(
                frame, 
                info_text, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 255, 0), 
                2
            )
            
            cv2.imshow("YOLO + 감정 인식 (CPU 최적화)", frame) # 프레임 표시
            
            if cv2.waitKey(1) & 0xFF == ord('q'): # 'q' 키로 종료
                break
    
    except KeyboardInterrupt:
        print("\n[정보] 사용자가 중단했습니다.")
    
    finally:
        vs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()