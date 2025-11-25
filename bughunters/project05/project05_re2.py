import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from threading import Thread
from collections import deque
import time

# ================== 설정 (렉 제거 최적화) ==================
YOLO_MODEL_PATH = "yolov8n-face.pt"
MODEL_NAME = "abhilash88/face-emotion-detection"
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# 초경량 최적화 설정
DISPLAY_TOP_N = 2           # 상위 2개만 (1개 줄임)
RESIZE_WIDTH = 480          # 해상도 낮춤 (640→480)
RESIZE_HEIGHT = 360         # 해상도 낮춤 (480→360)
SKIP_FRAMES = 2             # 3프레임마다 처리 (1→2)
YOLO_SKIP = 2               # YOLO도 3프레임마다 실행
EMOTION_SKIP = 3            # 감정 분석은 4프레임마다
SMOOTHING_FRAMES = 5        # 평활화 프레임 감소 (10→5)
UPDATE_INTERVAL = 0.5       # 업데이트 간격 증가 (0.3→0.5)
MIN_FACE_SIZE = 60          # 최소 얼굴 크기 증가 (30→60)
MAX_FACES = 3               # 최대 얼굴 수 제한

# PyTorch 최적화
torch.set_num_threads(2)    # 스레드 감소 (4→2)
torch.set_grad_enabled(False)  # 전역 그래디언트 비활성화

print("[정보] 초경량 모드 활성화")
print(f"[설정] 해상도: {RESIZE_WIDTH}x{RESIZE_HEIGHT}")
print(f"[설정] 프레임 스킵: YOLO={YOLO_SKIP}, 감정={EMOTION_SKIP}")

# ================== 모델 로드 ==================
print("[정보] 모델 로딩 중...")

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
emotion_model = ViTForImageClassification.from_pretrained(MODEL_NAME)
emotion_model.eval()

# YOLO 설정 최적화
yolo = YOLO(YOLO_MODEL_PATH)
yolo.overrides['verbose'] = False
yolo.overrides['conf'] = 0.6  # 신뢰도 높임 (거짓 탐지 감소)

print("[정보] 모델 로드 완료")

# ================== 경량 감정 평활화 ==================
class LightweightEmotionSmoother:
    """메모리 효율적인 감정 평활화"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.emotion_history = deque(maxlen=window_size)
        self.last_smooth = None
        self.last_update = 0
        self.update_count = 0
        
    def add(self, probs):
        if probs is not None:
            self.emotion_history.append(probs)
            self.update_count += 1
    
    def get(self):
        if len(self.emotion_history) == 0:
            return self.last_smooth
        
        current_time = time.time()
        
        # 업데이트 간격 체크
        if (current_time - self.last_update) >= UPDATE_INTERVAL:
            self.last_smooth = np.mean(self.emotion_history, axis=0)
            self.last_update = current_time
        
        return self.last_smooth
    
    def reset(self):
        self.emotion_history.clear()
        self.last_smooth = None

# ================== 경량 얼굴 추적기 ==================
class LightweightFaceTracker:
    """간단하고 빠른 얼굴 추적"""
    
    def __init__(self, max_faces=3):
        self.max_faces = max_faces
        self.trackers = {}  # face_id: {'box': (x,y,w,h), 'smoother': obj, 'age': int}
        self.next_id = 0
        self.max_age = 30  # 30프레임 미감지시 제거
        
    def distance(self, box1, box2):
        """두 박스 중심점 거리"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        c1_x, c1_y = x1 + w1/2, y1 + h1/2
        c2_x, c2_y = x2 + w2/2, y2 + h2/2
        return np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
    
    def update(self, boxes):
        """박스 업데이트 및 매칭"""
        # 기존 추적기 나이 증가
        for fid in list(self.trackers.keys()):
            self.trackers[fid]['age'] += 1
            if self.trackers[fid]['age'] > self.max_age:
                del self.trackers[fid]
        
        matched = set()
        new_trackers = {}
        
        for box in boxes[:self.max_faces]:  # 최대 얼굴 수 제한
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            new_box = (x1, y1, w, h)
            
            # 가장 가까운 기존 추적기 찾기
            best_id = None
            best_dist = 100  # 임계값
            
            for fid, tracker in self.trackers.items():
                if fid in matched:
                    continue
                dist = self.distance(tracker['box'], new_box)
                if dist < best_dist:
                    best_dist = dist
                    best_id = fid
            
            # 매칭 또는 새 ID 생성
            if best_id is not None:
                fid = best_id
                new_trackers[fid] = self.trackers[fid]
            else:
                fid = self.next_id
                self.next_id += 1
                new_trackers[fid] = {
                    'box': new_box,
                    'smoother': LightweightEmotionSmoother(SMOOTHING_FRAMES),
                    'age': 0
                }
            
            new_trackers[fid]['box'] = new_box
            new_trackers[fid]['age'] = 0
            matched.add(fid)
        
        self.trackers = new_trackers
        return list(self.trackers.keys())

# ================== 초경량 웹캠 스트림 ==================
class FastWebcamStream:
    """최소 오버헤드 웹캠 스트림"""
    
    def __init__(self, src=0, width=480, height=360):
        self.stream = cv2.VideoCapture(src)
        
        # 카메라 최적화
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        
        # MJPEG 코덱 시도 (더 빠름)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            if self.grabbed:
                self.grabbed, self.frame = self.stream.read()
                
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True
        self.stream.release()

# ================== 배치 감정 감지 (더 빠름) ==================
def detect_emotions_batch(face_images):
    """여러 얼굴을 한 번에 처리 (배치)"""
    if len(face_images) == 0:
        return []
    
    try:
        # PIL 변환
        face_pils = []
        for face_img in face_images:
            if face_img.shape[0] < MIN_FACE_SIZE or face_img.shape[1] < MIN_FACE_SIZE:
                face_pils.append(None)
                continue
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_pils.append(Image.fromarray(face_rgb))
        
        # 유효한 얼굴만 처리
        valid_faces = [f for f in face_pils if f is not None]
        if len(valid_faces) == 0:
            return [None] * len(face_images)
        
        # 배치 처리
        inputs = processor(valid_faces, return_tensors="pt")
        outputs = emotion_model(**inputs)
        probs_batch = torch.softmax(outputs.logits, dim=-1).detach().numpy()
        
        # 결과 매핑
        results = []
        valid_idx = 0
        for face_pil in face_pils:
            if face_pil is None:
                results.append(None)
            else:
                results.append(probs_batch[valid_idx])
                valid_idx += 1
        
        return results
    
    except Exception as e:
        return [None] * len(face_images)

# ================== 빠른 렌더링 ==================
def fast_draw_emotions(frame, x1, y1, x2, y2, probs, face_id):
    """최소한의 렌더링"""
    if probs is None:
        return
    
    # 경계 상자 (얇게)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # 상위 N개만
    top_indices = np.argsort(probs)[-DISPLAY_TOP_N:][::-1]
    
    for i, idx in enumerate(top_indices):
        label = f"{EMOTION_LABELS[idx]}: {probs[idx]:.2f}"
        y_pos = y2 + 20 + i * 20
        
        # 배경 없이 텍스트만 (더 빠름)
        cv2.putText(
            frame, label, (x1, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

# ================== 메인 루프 ==================
def main():
    print("\n" + "="*50)
    print("  초경량 실시간 감정 인식")
    print("="*50)
    print(f"[최적화] 해상도: {RESIZE_WIDTH}x{RESIZE_HEIGHT}")
    print(f"[최적화] 최대 얼굴: {MAX_FACES}개")
    print(f"[최적화] 프레임 스킵: {SKIP_FRAMES}")
    print("="*50 + "\n")
    
    vs = FastWebcamStream(src=0, width=RESIZE_WIDTH, height=RESIZE_HEIGHT).start()
    time.sleep(1.5)
    
    tracker = LightweightFaceTracker(max_faces=MAX_FACES)
    
    # 캐싱
    last_boxes = []
    last_yolo_frame = 0
    last_emotion_frame = 0
    
    # FPS
    fps_time = time.time()
    fps_count = 0
    fps_display = 0
    frame_count = 0
    
    print("[정보] 시작! ('q'로 종료)\n")
    
    try:
        while True:
            frame = vs.read()
            if frame is None:
                break
            
            frame_count += 1
            
            # YOLO 스킵
            if frame_count - last_yolo_frame >= (YOLO_SKIP + 1):
                results = yolo.predict(
                    frame, 
                    conf=0.6,
                    verbose=False,
                    device='cpu',
                    half=False,
                    imgsz=320  # YOLO 입력 크기 축소
                )
                
                boxes = []
                for res in results:
                    for box in res.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = map(int, box)
                        # 경계 체크
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        # 최소 크기 필터
                        if (x2 - x1) >= MIN_FACE_SIZE and (y2 - y1) >= MIN_FACE_SIZE:
                            boxes.append((x1, y1, x2, y2))
                
                last_boxes = boxes
                last_yolo_frame = frame_count
                
                # 추적기 업데이트
                face_ids = tracker.update(boxes)
            
            # 감정 감지 스킵
            if frame_count - last_emotion_frame >= (EMOTION_SKIP + 1) and len(last_boxes) > 0:
                face_images = []
                for x1, y1, x2, y2 in last_boxes:
                    face_img = frame[y1:y2, x1:x2]
                    face_images.append(face_img)
                
                # 배치 처리
                probs_list = detect_emotions_batch(face_images)
                
                # 추적기에 추가
                for fid in tracker.trackers.keys():
                    idx = list(tracker.trackers.keys()).index(fid)
                    if idx < len(probs_list):
                        tracker.trackers[fid]['smoother'].add(probs_list[idx])
                
                last_emotion_frame = frame_count
            
            # 렌더링
            for fid, data in tracker.trackers.items():
                x1, y1, w, h = data['box']
                x2, y2 = x1 + w, y1 + h
                smooth_probs = data['smoother'].get()
                
                fast_draw_emotions(frame, x1, y1, x2, y2, smooth_probs, fid)
            
            # FPS 계산 (간단히)
            fps_count += 1
            if time.time() - fps_time > 1:
                fps_display = fps_count
                fps_count = 0
                fps_time = time.time()
            
            # FPS 표시
            cv2.putText(frame, f"FPS: {fps_display}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Light Mode", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n[정보] 중단됨")
    finally:
        print("[정보] 종료 중...")
        vs.stop()
        cv2.destroyAllWindows()
        print("[정보] 종료 완료")

if __name__ == "__main__":
    main()
