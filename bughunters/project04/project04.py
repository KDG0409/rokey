# YOLO 얼굴 탐지 + CNN 감정 분류 (웹캠 실시간)
#  - 웹캠 실시간 감정 확률 출력
#  학습모듈 다운전 버전

import os
import time
import requests
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from ultralytics import YOLO

EMOTION_MODEL_PATH = "emotion_model.pth"
EMOTION_LABELS: List[str] = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]
EMOTION_IMAGE_SIZE = 48
EMOTION_NUM_CHANNELS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 함수 정의
class SimpleEmotionCNN(torch.nn.Module): # CNN으로 특성맵 생성/ 다중 분류
    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        fm = EMOTION_IMAGE_SIZE // 4
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * fm * fm, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def create_emotion_model(path: str, device): #감정 파악 모듈 다운로드
    model = SimpleEmotionCNN(len(EMOTION_LABELS), EMOTION_NUM_CHANNELS)
    model.to(device)
    torch.save(model.state_dict(), path) # 랜덤 초기화된 모델 저장

def load_emotion_model(path: str, device): #감정 파악 모듈 로드
    if not os.path.exists(path):
        create_emotion_model(path, device)

    model = SimpleEmotionCNN(len(EMOTION_LABELS), EMOTION_NUM_CHANNELS)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model

def preprocess_face(img: np.ndarray):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (EMOTION_IMAGE_SIZE, EMOTION_IMAGE_SIZE))

    if EMOTION_NUM_CHANNELS == 1:
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        img_gray = img_gray.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0)
    else:
        img_f = img_resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)

    return tensor

def draw_probabilities(frame, box, probs, labels): # 다중 분류 / 확률로 출력
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    sorted_idx = np.argsort(-probs)
    start_x, start_y = x1, y1 - 10 if y1 > 20 else y2 + 15

    for i, idx in enumerate(sorted_idx):
        text = f"{labels[idx]}: {probs[idx]:.2f}"
        pos = (start_x, start_y + i * 18)

        cv2.rectangle(frame, (pos[0], pos[1] - 15),
                      (pos[0] + 150, pos[1] + 2), (0, 0, 0), -1)
        cv2.putText(frame, text, pos, FONT, 0.45, (255, 255, 255), 1)


# 실습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

yolo = YOLO("yolov8n-face.pt")
emotion_model = load_emotion_model(EMOTION_MODEL_PATH, device)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame, verbose=False)

    for det in results:
        if not hasattr(det, "boxes"):
            continue

        for xyxy in det.boxes.xyxy:
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            tensor = preprocess_face(face).to(device)

            with torch.no_grad():
                out = emotion_model(tensor)
                probs = F.softmax(out, dim=1).cpu().numpy().squeeze()

            draw_probabilities(frame, (x1, y1, x2, y2), probs, EMOTION_LABELS)

    cv2.imshow("Face Emotion", frame)
    key = cv2.waitKey(1) & 0xFF

    if key in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
