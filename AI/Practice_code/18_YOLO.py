import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
print(f"모델 파라미터 수: {sum(p.numel() for p in model.model.parameters()):,}")
class_names = model.names # COCO 클래스 이름 (80개)
print(f"\n총 클래스 수: {len(class_names)}")
print(f"일부 클래스: {list(class_names.values())[:10]}") # 10개 : 딕셔너리 value값을 리스트로 출력

# 데이터 준비
def download_image(url): # URL에서 이미지 다운로드
    try:
        response = requests.get(url, timeout=10) # url에서 다운로드, 10초 간격 # 200이면 정상수신 None이면 오류
        img = Image.open(BytesIO(response.content)).convert('RGB') # content->BytesIO로 해석-> Image.open으로 열기
        return img
    except Exception as e:
        print(f"이미지 다운로드 실패: {e}")
        return None
    
test_images = {       # 테스트 이미지 URL 목록
    'street': 'https://ultralytics.com/images/bus.jpg',
    'people': 'https://ultralytics.com/images/zidane.jpg',
    'animals': 'https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=640',
}
images = {}
for name, url in test_images.items():
    img = download_image(url)
    if img is not None:
        images[name] = img

# 데이터 시각화
if images:
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    if len(images) == 1:
        axes = [axes] # img가 1개면 반복문이 돌지 않기 때문에 axes 설정

    for ax, (name, img) in zip(axes, images.items()):
        ax.imshow(img)
        ax.set_title(f'{name.capitalize()} Image', fontsize=12, fontweight='bold') # capitalize(): 첫글자만 대문자
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# 기본 객체 검출

img_name = list(images.keys())[0] # 첫 번째 이미지 검출
test_img = images[img_name]
results = model(test_img) # YOLOv8 추론
result = results[0]  # 첫 번째 이미지 추론 결과
boxes = result.boxes  # 첫 번째 이미지의 Bounding boxes

if len(boxes) > 0:
    print("\n검출 상세:")
    for i, box in enumerate(boxes): # 박스 정보와 인덱스 생성 / 배열로 옮김
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() # x1, y1, x2, y2 <- 텐서에서 넘파이배열로 변경 : cpu로 변경 필요
        conf = box.conf[0].cpu().item() # 신뢰도 점수
        cls = int(box.cls[0].cpu().item()) # 클래스 순서(인덱스)
        class_name = class_names[cls] # 클래스 이름

        print(f"  {i+1}. {class_name}: {conf:.3f} "
              f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        
# 결과 시각화 (기본 내장)
result_img = result.plot()  # BGR 형식(기본)으로 이미지 plot 
plt.figure(figsize=(12, 8)) 
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)) # RGB로 변환
plt.axis('off')
plt.title(f'YOLOv8 Detection Results ({img_name})',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Confidence Threshold 조정

thresholds = [0.25, 0.5, 0.75]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, conf_thresh in zip(axes, thresholds): # threshold 적용하여 추론
    results_thresh = model(test_img, conf=conf_thresh) # threshold 적용
    result_thresh = results_thresh[0] 
    img_plot = result_thresh.plot() # 시각화 
    ax.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)) 
    ax.set_title(f'Confidence ≥ {conf_thresh}\n'
                 f'({len(result_thresh.boxes)} objects)',  
                 fontsize=12, fontweight='bold') 
    ax.axis('off')

plt.tight_layout()
plt.show()

# 여러 이미지 동시 검출

fig, axes = plt.subplots(len(images), 2, figsize=(14, 6*len(images)))
if len(images) == 1:
    axes = axes.reshape(1, -1)

for idx, (name, img) in enumerate(images.items()):
    axes[idx, 0].imshow(img) # 원본 이미지
    axes[idx, 0].set_title(f'{name.capitalize()} - Original',
                           fontsize=12, fontweight='bold')
    axes[idx, 0].axis('off')

    results = model(img, conf=0.5) # 검출 결과
    result_img = results[0].plot()
    axes[idx, 1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[idx, 1].set_title(f'{name.capitalize()} - Detected ({len(results[0].boxes)} objects)',
                           fontsize=12, fontweight='bold')
    axes[idx, 1].axis('off')

plt.tight_layout()
plt.show()

# 클래스별 통계 분석
all_detections = {} # 모든 이미지에서 검출 수행
for name, img in images.items():
    results = model(img, conf=0.5)
    boxes = results[0].boxes

    detections = []
    for box in boxes:
        cls = int(box.cls[0].cpu().item())
        conf = box.conf[0].cpu().item()
        detections.append({
            'class': class_names[cls],
            'confidence': conf
        })
    all_detections[name] = detections

from collections import Counter # 클래스별 카운트

for name, detections in all_detections.items():
    print(f"\n{name.upper()} 이미지:")
    if detections:
        class_counts = Counter([d['class'] for d in detections])
        for cls, count in class_counts.most_common():
            avg_conf = np.mean([d['confidence']
                               for d in detections if d['class'] == cls])
            print(f"  - {cls}: {count}개 (평균 confidence: {avg_conf:.3f})")
    else:
        print("  검출된 객체 없음")

def draw_custom_boxes(image, results, conf_threshold=0.5): # 커스텀 박스 그리기
    img_np = np.array(image).copy()
    boxes = results[0].boxes

    np.random.seed(42)
    colors = {cls: tuple(np.random.randint(0, 255, 3).tolist()) # 색상 팔레트 (클래스별)
              for cls in range(len(class_names))}

    for box in boxes:
        conf = box.conf[0].cpu().item()
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # 박스 정보
        cls = int(box.cls[0].cpu().item())
        class_name = class_names[cls]
        color = colors[cls]

        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 3) # 박스 그리기

        label = f'{class_name} {conf:.2f}' # 레이블 배경
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_np, (x1, y1 - text_h - 10),
                     (x1 + text_w, y1), color, -1)

        cv2.putText(img_np, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # 레이블 텍스트

    return img_np

img_name = list(images.keys())[0] # 커스텀 시각화 적용
test_img = images[img_name]
results = model(test_img)

custom_img = draw_custom_boxes(test_img, results, conf_threshold=0.5)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(test_img)
axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(custom_img)
axes[1].set_title('Custom Visualization', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# 모델 정보 및 성능 비교

model_info = {
    'YOLOv8n': {'params': '3.2M', 'mAP': 37.3, 'speed': '80+ FPS'},
    'YOLOv8s': {'params': '11.2M', 'mAP': 44.9, 'speed': '50+ FPS'},
    'YOLOv8m': {'params': '25.9M', 'mAP': 50.2, 'speed': '30+ FPS'},
    'YOLOv8l': {'params': '43.7M', 'mAP': 52.9, 'speed': '20+ FPS'},
    'YOLOv8x': {'params': '68.2M', 'mAP': 53.9, 'speed': '15+ FPS'},
}
print("\n| 모델 | 파라미터 | mAP@0.5:0.95 | 속도 (T4 GPU) |")
print("|------|----------|--------------|---------------|")
for model_name, info in model_info.items():
    print(f"| {model_name} | {info['params']} | {info['mAP']}% | {info['speed']} |")

# 배치 추론 (여러 이미지 동시 처리)
import time
img_list = list(images.values()) # 이미지 리스트 준비

start_time = time.time() # 단일 추론
for img in img_list:
    _ = model(img, verbose=False)
single_time = time.time() - start_time

start_time = time.time() # 배치 추론
_ = model(img_list, verbose=False)
batch_time = time.time() - start_time

print(f"단일 추론: {single_time:.3f}초 ({len(img_list)}개 이미지)")
print(f"배치 추론: {batch_time:.3f}초 ({len(img_list)}개 이미지)")
print(f"속도 향상: {single_time/batch_time:.2f}배")

# Colab 파일 업로드 기능 (Colab에서 실행가능)
# from google.colab import files
# uploaded = files.upload()

# if uploaded:
#     img_path = list(uploaded.keys())[0] # 업로드된 첫 번째 이미지 사용
#     user_img = Image.open(img_path).convert('RGB')

#     print(f"이미지 업로드 완료: {img_path}")
#     print(f"크기: {user_img.size}")

#     results = model(user_img, conf=0.5) # 검출 수행
#     result_img = results[0].plot()

#     fig, axes = plt.subplots(1, 2, figsize=(16, 8)) # 결과 시각화
#     axes[0].imshow(user_img)
#     axes[0].set_title('Uploaded Image', fontsize=14, fontweight='bold')
#     axes[0].axis('off')

#     axes[1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
#     axes[1].set_title(f'The result : ({len(results[0].boxes)} objects)',
#                      fontsize=14, fontweight='bold')
#     axes[1].axis('off')

#     plt.tight_layout()
#     plt.show()

#     print("\n검출된 객체:") # 검출 상세 정보
#     for i, box in enumerate(results[0].boxes):
#         cls = int(box.cls[0].cpu().item())
#         conf = box.conf[0].cpu().item()
#         print(f"  {i+1}. {class_names[cls]}: {conf:.3f}")
# else:
#     print("업로드된 이미지가 없습니다.")

print("\n학습 포인트:")
print("1. YOLOv8 모델 로드 및 추론")
print("2. Confidence threshold 조정의 영향")
print("3. 검출 결과 시각화 및 분석")
print("4. 배치 추론으로 성능 향상")