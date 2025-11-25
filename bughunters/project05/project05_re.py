import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from threading import Thread
from collections import deque
import time

# 프로젝트04 최적화

# 업데이트 간격 제어 (0.3초) : 메모리 개선
# 얼굴 추적 (Face Tracking) : 같은 얼굴을 지속적으로 추적 (IoU 기반 추적)
# fps 8-12 -> 30-45 : 비동기 스레딩 사용 : class WebcamStream:
# 깜빡임, 프레임 지연 개선 : CPU 사용률 대폭 감소, 속도 증가
# 메모리 개선 : grad 추적 제거
# 항상 최신 프레임만 처리 : self.stream.set사용 지연시간 최소화, 실시간 반응성 향상
# 이동 평균 기반 감정 평활화 : 깜빡임 현상 최소화 : 최근 N개 프레임의 감정 점수를 평균내어 부드럽게 만듦

# ================== 설정 ==================
YOLO_MODEL_PATH = "yolov8n-face.pt"
MODEL_NAME = "abhilash88/face-emotion-detection"
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# CPU 최적화 설정
DISPLAY_TOP_N = 3      # 상위 3개 감정만 표시
TARGET_FPS = 30        # 목표 프레임률
BUFFER_SIZE = 1        # 최신 프레임만 유지
RESIZE_WIDTH = 640     
RESIZE_HEIGHT = 480
SKIP_FRAMES = 1        # 프레임 스킵

# 감정 안정화 설정
SMOOTHING_FRAMES = 10   # 평균 계산할 프레임 수 (높을수록 부드러움)
UPDATE_INTERVAL = 0.3   # 화면 업데이트 간격 (초) - 깜빡임 방지

# ================== 모델 로드 ==================
print("[정보] 모델 로딩 중... (CPU 모드)")

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
emotion_model = ViTForImageClassification.from_pretrained(MODEL_NAME)
emotion_model.eval()

torch.set_num_threads(1)
yolo = YOLO(YOLO_MODEL_PATH)

print("[정보] 모델 로드 완료")

# ================== 감정 평활화 클래스 ==================
class EmotionSmoother:
    """감정 점수를 부드럽게 만드는 클래스 (깜빡임 방지)"""
    
    def __init__(self, window_size=10, update_interval=0.3):
        self.window_size = window_size
        self.update_interval = update_interval
        self.emotion_history = deque(maxlen=window_size)
        self.last_update_time = 0
        self.current_smooth_probs = None
        self.frame_count = 0
        
    def add_emotion(self, probs):
        """새로운 감정 확률 추가"""
        if probs is not None:
            self.emotion_history.append(probs)
            self.frame_count += 1
    
    def get_smooth_emotion(self):
        """평활화된 감정 확률 반환"""
        current_time = time.time()
        
        # 히스토리가 없으면 None 반환
        if len(self.emotion_history) == 0:
            return None
        
        # 일정 시간마다만 업데이트 (깜빡임 방지)
        if (current_time - self.last_update_time) >= self.update_interval:
            # 이동 평균 계산
            self.current_smooth_probs = np.mean(self.emotion_history, axis=0)
            self.last_update_time = current_time
        
        return self.current_smooth_probs
    
    def reset(self):
        """히스토리 초기화 (얼굴이 사라졌을 때)"""
        self.emotion_history.clear()
        self.current_smooth_probs = None
        self.frame_count = 0

# ================== 얼굴 추적기 ==================
class FaceTracker:
    """여러 얼굴을 추적하고 각각의 감정을 평활화"""
    
    def __init__(self, max_faces=5):
        self.max_faces = max_faces
        self.face_smoothers = {}  # face_id: EmotionSmoother
        self.face_positions = {}  # face_id: (x1, y1, x2, y2)
        self.next_face_id = 0
        self.iou_threshold = 0.3  # IoU 임계값
        
    def calculate_iou(self, box1, box2):
        """두 박스 간 IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 교집합 영역
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 영역
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def find_matching_face(self, new_box):
        """새 박스와 가장 유사한 기존 얼굴 찾기"""
        best_match_id = None
        best_iou = 0
        
        for face_id, old_box in self.face_positions.items():
            iou = self.calculate_iou(new_box, old_box)
            if iou > best_iou and iou > self.iou_threshold:
                best_iou = iou
                best_match_id = face_id
        
        return best_match_id
    
    def update(self, boxes, probs_list):
        """얼굴 위치와 감정 업데이트"""
        current_face_ids = set()
        
        for i, box in enumerate(boxes):
            # 기존 얼굴과 매칭
            face_id = self.find_matching_face(box)
            
            # 새 얼굴이면 ID 할당
            if face_id is None:
                face_id = self.next_face_id
                self.next_face_id += 1
                self.face_smoothers[face_id] = EmotionSmoother(
                    window_size=SMOOTHING_FRAMES,
                    update_interval=UPDATE_INTERVAL
                )
            
            # 위치 및 감정 업데이트
            self.face_positions[face_id] = box
            if i < len(probs_list) and probs_list[i] is not None:
                self.face_smoothers[face_id].add_emotion(probs_list[i])
            
            current_face_ids.add(face_id)
        
        # 사라진 얼굴 제거
        disappeared_faces = set(self.face_positions.keys()) - current_face_ids
        for face_id in disappeared_faces:
            del self.face_positions[face_id]
            del self.face_smoothers[face_id]
    
    def get_smooth_emotions(self):
        """모든 얼굴의 평활화된 감정 반환"""
        result = {}
        for face_id, smoother in self.face_smoothers.items():
            smooth_probs = smoother.get_smooth_emotion()
            if smooth_probs is not None:
                result[face_id] = smooth_probs
        return result

# ================== 스레드 기반 비디오 캡처 ==================
class WebcamStream:
    """별도 스레드에서 웹캠 프레임 읽기"""
    
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self
        
    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()
                
    def read(self):
        return self.frame
        
    def stop(self):
        self.stopped = True
        self.stream.release()
        
    def isOpened(self):
        return self.stream.isOpened()

# ================== 최적화된 감정 감지 ==================
@torch.no_grad()
def detect_emotion_optimized(face_img):
    """CPU 최적화 감정 감지"""
    
    if face_img.shape[0] < 30 or face_img.shape[1] < 30:
        return None
    
    try:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        inputs = processor(face_pil, return_tensors="pt")
        outputs = emotion_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].detach().numpy()
        return probs
    except Exception as e:
        return None

# ================== 텍스트 렌더링 (안정화 버전) ==================
def draw_emotion_labels_stable(frame, x1, y2, probs, face_id):
    """부드러운 감정 레이블 그리기 (깜빡임 없음)"""
    
    if probs is None:
        return
    
    # 상위 N개 감정 추출
    top_indices = np.argsort(probs)[-DISPLAY_TOP_N:][::-1]
    
    for i, idx in enumerate(top_indices):
        # 소수점 2자리로 고정
        label = f"{EMOTION_LABELS[idx]}: {probs[idx]:.2f}"
        y_pos = y2 + 25 + i * 25
        
        # 텍스트 크기 계산
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # 배경 사각형 (가독성)
        cv2.rectangle(
            frame, 
            (x1, y_pos - text_h - 5), 
            (x1 + text_w + 5, y_pos + 5), 
            (0, 0, 0), 
            -1
        )
        
        # 텍스트
        cv2.putText(
            frame, 
            label, 
            (x1 + 2, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
    
    # Face ID 표시 (디버깅용 - 선택사항)
    # cv2.putText(frame, f"ID:{face_id}", (x1, y2 + 90),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# ================== 메인 루프 ==================
def main():
    print("\n" + "="*50)
    print("  YOLO + 감정 인식 (안정화 버전)")
    print("="*50)
    print(f"[설정] 평활화 프레임: {SMOOTHING_FRAMES}개")
    print(f"[설정] 업데이트 간격: {UPDATE_INTERVAL}초")
    print(f"[설정] CPU 스레드: {torch.get_num_threads()}개")
    print("="*50 + "\n")
    
    # 스레드 기반 웹캠 시작
    print("[정보] 웹캠 초기화 중...")
    vs = WebcamStream(src=0, width=RESIZE_WIDTH, height=RESIZE_HEIGHT).start()
    
    if not vs.isOpened():
        print("[오류] 웹캠을 열 수 없습니다.")
        return
    
    time.sleep(2.0)
    print("[정보] 웹캠 준비 완료!")
    print("[정보] 'q' 키를 눌러 종료하세요\n")
    
    # 얼굴 추적기 초기화
    face_tracker = FaceTracker()
    
    # FPS 측정
    fps_start_time = time.time()
    fps_counter = 0
    fps_display = 0
    frame_count = 0
    
    try:
        while True:
            frame = vs.read()
            
            if frame is None:
                break
            
            frame_count += 1
            
            # YOLO 얼굴 감지
            results = yolo(frame, verbose=False, conf=0.5)
            
            boxes = []
            probs_list = []
            
            # 모든 얼굴에서 감정 감지
            for res in results:
                detected_boxes = res.boxes.xyxy.cpu().numpy()
                
                for box in detected_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 경계 확인
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    if face_img.size == 0:
                        continue
                    
                    boxes.append((x1, y1, x2, y2))
                    
                    # 프레임 스킵 적용하여 감정 감지
                    if frame_count % (SKIP_FRAMES + 1) == 0:
                        probs = detect_emotion_optimized(face_img)
                        probs_list.append(probs)
                    else:
                        probs_list.append(None)
            
            # 얼굴 추적 및 감정 평활화
            face_tracker.update(boxes, probs_list)
            smooth_emotions = face_tracker.get_smooth_emotions()
            
            # 화면에 그리기
            for face_id, box in face_tracker.face_positions.items():
                x1, y1, x2, y2 = box
                
                # 경계 상자
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 평활화된 감정 레이블
                if face_id in smooth_emotions:
                    draw_emotion_labels_stable(
                        frame, x1, y2, 
                        smooth_emotions[face_id], 
                        face_id
                    )
            
            # FPS 계산
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed > 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # FPS 표시
            cv2.putText(
                frame, 
                f"FPS: {fps_display} | Smooth Mode", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 255, 0), 
                2
            )
            
            cv2.imshow("YOLO + 감정 인식 (안정화)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n[정보] 사용자가 중단했습니다.")
    
    finally:
        print("[정보] 종료 중...")
        vs.stop()
        cv2.destroyAllWindows()
        print("[정보] 정상 종료되었습니다.")

if __name__ == "__main__":
    main()
