import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

# YOLO 얼굴 모델 경로 (이미 설정되어 있다고 가정)
YOLO_MODEL_PATH = "yolov8n-face.pt"

# 감정 모델: Hugging Face ViT FER2013
MODEL_NAME = "abhilash88/face-emotion-detection"
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
emotion_model = ViTForImageClassification.from_pretrained(MODEL_NAME)
emotion_model.eval()

# 감정 레이블 (모델 문서에 맞게)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# YOLO 로드
yolo = YOLO(YOLO_MODEL_PATH)

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠 열기 실패")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame, verbose=False)
    for res in results:
        for box in res.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            # PIL 이미지로 변환 + 전처리
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            inputs = processor(face_pil, return_tensors="pt")

            # 예측
            with torch.no_grad():
                outputs = emotion_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

            # 감정 확률 표시
            for i, prob in enumerate(probs):
                text = f"{EMOTION_LABELS[i]}: {prob:.2f}"
                cv2.putText(frame, text, (x1, y2 + 20 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 얼굴 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("YOLO + Emotion (Fer2013 ViT)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()