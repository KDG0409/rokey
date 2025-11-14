import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

import warnings
warnings.simplefilter('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import urllib.request
import requests
from io import BytesIO

# COCO 클래스 이름 정의 (80개)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# IOU 계산 함수
def compute_iou(box1, box2): # box format: [x1, y1, x2, y2]
    x1 = max(box1[0],box2[0])
    y1 = max(box1[1],box2[1])
    x2 = min(box1[2],box2[2])
    y2 = min(box1[3],box2[3])
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) # box1 면적 
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) # box2 면적
    intersection = max(0,x2-x1) * max(0,y2-y1)
    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0
    return iou

# NMS 함수 구현 : 중복제거(겹치는 박스 중 가장 신뢰도가 높은 박스만 남김)
def simple_nms(boxes, scores, iou_threshold=0.5):
    # boxes: (N, 4) - [x1, y1, x2, y2]
    # scores: (N,) - confidence scores

    indices = np.argsort(scores)[::-1] # 신뢰도(score) 내림차순 정렬
    keep = [] # 가장 높은 신뢰도 점수를 가진 최종 선택 인덱스 저장

    while len(indices)>0:
        current = indices[0] # 신뢰도 가장 높은 박스(첫번째)
        keep.append(current)
        if len(indices) == 1 : # 정지 조건 
            break
        current_box = boxes[current] # 신뢰도 가장 높은 박스(첫번째)
        remaining_boxes = boxes[indices[1:]] # 나머지 박스
        # iou가 threshold 이하인 박스만 유지함
        ious = np.array([compute_iou(current_box,box) for box in remaining_boxes])
        indices = indices[1:][ious <= iou_threshold]

    return keep

# NMS 테스트
test_boxes = np.array([
    [100, 100, 200, 200],
    [110, 110, 210, 210],
    [105, 105, 205, 205],
    [300, 300, 400, 400],
])
test_scores = np.array([0.9, 0.8, 0.85, 0.95])
keep_indices = simple_nms(test_boxes, test_scores, iou_threshold=0.5)

# Faster R-CNN 모델 로드
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
det_model = fasterrcnn_resnet50_fpn(weight=weights).to(device)
det_model.eval()
preprocess = weights.transforms()

# 데이터 이미지 준비
def create_sample_image():
    img = Image.new('RGB', (800, 600), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # 사람 그리기 (간단한 스틱맨) #outline : 윤곽선 ellipse : 원형
    # 머리
    draw.ellipse([150, 100, 250, 200], fill=(255, 220, 180), outline=(0, 0, 0), width=3)
    # 몸통
    draw.rectangle([180, 200, 220, 400], fill=(0, 100, 200), outline=(0, 0, 0), width=3)
    # 팔
    draw.line([180, 250, 120, 300], fill=(0, 100, 200), width=15)
    draw.line([220, 250, 280, 300], fill=(0, 100, 200), width=15)
    # 다리
    draw.line([180, 400, 140, 550], fill=(50, 50, 50), width=15)
    draw.line([220, 400, 260, 550], fill=(50, 50, 50), width=15)

    # 의자 그리기
    draw.rectangle([500, 300, 650, 350], fill=(139, 69, 19), outline=(0, 0, 0), width=3)
    draw.rectangle([520, 350, 540, 500], fill=(139, 69, 19), outline=(0, 0, 0), width=3)
    draw.rectangle([610, 350, 630, 500], fill=(139, 69, 19), outline=(0, 0, 0), width=3)
    draw.rectangle([510, 150, 640, 300], fill=(160, 82, 45), outline=(0, 0, 0), width=3)

    # 책 그리기
    draw.rectangle([350, 450, 450, 550], fill=(200, 50, 50), outline=(0, 0, 0), width=3)
    draw.line([400, 450, 400, 550], fill=(0, 0, 0), width=2)

    # 텍스트 추가
    draw.text((300, 30), "Sample Detection Image", fill=(0, 0, 0))

    return img

def download_image_safe():

    urls = [
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
        "https://ultralytics.com/images/zidane.jpg",
    ]

    for url in urls:
        try:
            print(f"다운로드 시도: {url[:50]}...")
            response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                print(f"이미지 다운로드 성공!")
                return img
        except Exception as e:
            print(f"  실패: {str(e)[:50]}")
            continue

    return None
img = download_image_safe()
if img is None: # 온라인 다운로드 실패 → 샘플 이미지 생성
    img = create_sample_image()

else:
    if max(img.size) > 1000: # 이미지가 너무 크면 리사이즈
        img.thumbnail((1000, 1000))
        print(f"이미지 리사이즈: {img.size}")

# 실제 이미지로 테스트 희망시

# from google.colab import files
# uploaded = files.upload()
# img_path = list(uploaded.keys())[0]
# img = Image.open(img_path).convert("RGB")

# 객체 검출 수행
# 전처리
x = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = det_model(x)
    out = det_model(x)[0] # 리스트를 벗긴 값 >예측값: 딕셔너리형태
    boxes = out['boxes'].cpu().numpy() # 넘파이 변환/CPU변경 : boxes만 출력 /박스 좌표
    labels = out['labels'].cpu().numpy() # 예측 클래스
    scores = out['scores'].cpu().numpy() # 신뢰도(확률)

conf_threshold = 0.5

if len(boxes) > 0:
    mask = scores >= conf_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]

    print(f"\n=== Confidence >= {conf_threshold} 필터링 ===")
    print(f"필터링 후 객체 수: {len(filtered_boxes)}")

    if len(filtered_boxes) > 0:
        # 시각화
        img_np = np.array(img)
        img_draw = img_np.copy()

        # 색상 정의
        colors = plt.cm.tab20(np.linspace(0, 1, 20)) # 20개의 색상 정의

        for i, (box, score, label) in enumerate(zip(filtered_boxes, filtered_scores, filtered_labels)):
            x1, y1, x2, y2 = box.astype(int)
            class_name = COCO_CLASSES[label]

            # 박스 그리기
            color = tuple((np.array(colors[label % 20][:3]) * 255).astype(int).tolist())
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)

            # 레이블 그리기
            text = f'{class_name}: {score:.2f}'
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # getTextSize() : 텍스트 박스(라벨 배경 박스) 크기,스케일,선두께
            cv2.rectangle(img_draw, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            # -1의 의미: 내부를 fill(채움)
            cv2.putText(img_draw, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # 스케일,색상,두께

            print(f"{i+1}. {class_name} (confidence: {score:.3f}), Box: [{x1}, {y1}, {x2}, {y2}]")

        plt.figure(figsize=(12, 8))
        plt.imshow(img_draw)
        plt.axis('off')
        plt.title(f'Faster R-CNN 검출 결과 (Threshold: {conf_threshold})')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Threshold {conf_threshold} 이상인 객체가 없습니다!")
        print("→ Threshold를 낮춰보세요 (예: 0.3)")
else:
    print("검출할 객체가 없어서 시각화를 건너뜁니다.")

# Threshold 비교
if len(boxes) > 0:
    print("\n=== 다양한 Confidence Threshold 비교 ===")
    thresholds = [0.3, 0.5, 0.7, 0.9]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    img_np = np.array(img)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for idx, thresh in enumerate(thresholds):
        mask = scores >= thresh
        img_temp = img_np.copy()

        temp_boxes = boxes[mask]
        temp_scores = scores[mask]
        temp_labels = labels[mask]

        for box, score, label in zip(temp_boxes, temp_scores, temp_labels):
            x1, y1, x2, y2 = box.astype(int)
            class_name = COCO_CLASSES[label]
            color = tuple((np.array(colors[label % 20][:3]) * 255).astype(int).tolist())

            cv2.rectangle(img_temp, (x1, y1), (x2, y2), color, 2)
            text = f'{class_name}: {score:.2f}'
            cv2.putText(img_temp, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        axes[idx].imshow(img_temp)
        axes[idx].axis('off')
        axes[idx].set_title(f'Threshold: {thresh} ({len(temp_boxes)} 객체)', fontsize=12)
        print(f"Threshold {thresh}: {len(temp_boxes)}개 검출")

    plt.tight_layout()
    plt.show()

# 검출 통계 분석
if len(boxes) > 0 and len(boxes[scores >= conf_threshold]) > 0:
    print("\n=== 검출 통계 분석 ===")

    # 코드 작성
    mask = scores >= conf_threshold
    filtered_labels = labels[mask]
    filtered_scores = scores[mask]

    from collections import Counter
    class_counts = Counter(filtered_labels)

    print(f"클래스별 검출 수 (Threshold >= {conf_threshold})")

    for label_id, count in class_counts.most_common():
        class_name = COCO_CLASSES[label_id] 
        avg_conf = filtered_scores[filtered_labels == label_id].mean() # 평균 신뢰도

    # Confidence 분포 시각화
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=conf_threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold: {conf_threshold}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('전체 검출 Confidence 분포')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    class_names = [COCO_CLASSES[l] for l in filtered_labels]
    unique_classes = list(set(class_names))
    class_scores = [filtered_scores[np.array(class_names) == c].mean()
                    for c in unique_classes]
    plt.barh(unique_classes, class_scores, color='skyblue', edgecolor='black')
    plt.xlabel('Average Confidence')
    plt.title('클래스별 평균 Confidence')
    plt.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

print("학습 포인트:")
print("1. IoU 계산으로 검출 정확도 평가")
print("2. NMS로 중복 박스 제거")
print("3. Faster R-CNN으로 객체 검출")
print("4. Confidence threshold 조정으로 검출 민감도 제어")