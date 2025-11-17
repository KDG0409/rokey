import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from pathlib import Path
from tqdm import tqdm
import zipfile
import urllib.request
import shutil
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 데이터셋 다운로드 및 준비
def download_pennfudan_dataset(): # Penn-Fudan Pedestrian Detection Dataset 다운로드
    data_dir = Path('./PennFudanPed')

    if data_dir.exists():
        print(f"데이터셋이 이미 존재합니다: {data_dir}")
        return data_dir

    print("Penn-Fudan Dataset 다운로드 중...")
    url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    zip_path = "PennFudanPed.zip"

    # 다운로드
    urllib.request.urlretrieve(url, zip_path)
    print(f"다운로드 완료: {zip_path}")

    # 압축 해제
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    print(f"압축 해제 완료: {data_dir}")

    # 압축 파일 삭제
    os.remove(zip_path)

    return data_dir

# 데이터셋 다운로드
data_root = download_pennfudan_dataset()

# 데이터 구조 확인
print("\n=== 데이터셋 구조 ===")
print(f"이미지: {len(list((data_root / 'PNGImages').glob('*.png')))}개")
print(f"마스크: {len(list((data_root / 'PedMasks').glob('*.png')))}개")

# 샘플 이미지 확인
sample_img = Image.open(data_root / 'PNGImages' / 'FudanPed00001.png')
sample_mask = Image.open(data_root / 'PedMasks' / 'FudanPed00001_mask.png')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(sample_img)
axes[0].set_title('Sample Image')
axes[0].axis('off')
axes[1].imshow(sample_mask, cmap='tab20')
axes[1].set_title('Sample Mask (Instance Segmentation)')
axes[1].axis('off')
plt.tight_layout()
plt.show()

# Custom Dataset 클래스 정의
class PennFudanDataset(data.Dataset):

    def __init__(self, root, transforms=None):
        self.root = Path(root)
        self.transforms = transforms

        # 모든 이미지와 마스크 파일 로드
        self.imgs = sorted(list((self.root / 'PNGImages').glob('*.png')))
        self.masks = sorted(list((self.root / 'PedMasks').glob('*.png')))

        print(f"Dataset 초기화: {len(self.imgs)}개 이미지")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 이미지 로드
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # 마스크를 numpy 배열로 변환
        mask = np.array(mask)

        # 각 instance의 고유 ID 추출
        obj_ids = np.unique(mask)
        # 배경 제거 (ID 0)
        obj_ids = obj_ids[1:]

        # 마스크를 binary mask로 분할
        masks = mask == obj_ids[:, None, None]

        # Bounding box 계산
        num_objs = len(obj_ids)
        boxes = []

        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            # 유효한 박스만 추가
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])

        # Tensor로 변환
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # 모두 사람(class 1)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        # Transform 적용
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

# Transform 정의
def get_transform(train=True): # 데이터 Transform
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# 데이터 전처리

# Train/Val 분할
dataset_full = PennFudanDataset(data_root, get_transform(train=True))
indices = torch.randperm(len(dataset_full)).tolist()
split_idx = int(len(dataset_full) * 0.8)
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

# Subset 생성
train_dataset = torch.utils.data.Subset(dataset_full, train_indices)
val_dataset_transforms = PennFudanDataset(data_root, get_transform(train=False))
val_dataset = torch.utils.data.Subset(val_dataset_transforms, val_indices)

# DataLoader
def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn,
    pin_memory=True
)

# 모델 생성
def get_model(num_classes):

    # 사전학습된 모델 로드
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # Classifier head 교체
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# 모델 초기화 (배경 + 사람 = 2 classes)
model = get_model(num_classes=2)
model.to(device)

# 학습 설정
# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

num_epochs = 10

# 학습 함수
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """1 에폭 학습"""
    model.train()

    total_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_objectness = 0
    loss_rpn_box_reg = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{num_epochs}")

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # 통계
        total_loss += losses.item()
        loss_classifier += loss_dict['loss_classifier'].item()
        loss_box_reg += loss_dict['loss_box_reg'].item()
        loss_objectness += loss_dict['loss_objectness'].item()
        loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()

        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'cls': f'{loss_dict["loss_classifier"].item():.3f}',
            'box': f'{loss_dict["loss_box_reg"].item():.3f}'
        })

    n = len(data_loader)
    return {
        'total_loss': total_loss / n,
        'loss_classifier': loss_classifier / n,
        'loss_box_reg': loss_box_reg / n,
        'loss_objectness': loss_objectness / n,
        'loss_rpn_box_reg': loss_rpn_box_reg / n
    }

# 평가 함수
# torch.no_grad()
def evaluate(model, data_loader, device):
    """검증 세트 평가"""
    model.eval()

    total_predictions = 0
    total_targets = 0
    correct_predictions = 0

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            pred_boxes = output['boxes'].cpu()
            pred_scores = output['scores'].cpu()
            target_boxes = target['boxes']

            # Confidence > 0.5인 예측만 사용
            mask = pred_scores > 0.5
            pred_boxes = pred_boxes[mask]

            total_predictions += len(pred_boxes)
            total_targets += len(target_boxes)

            # 간단한 정확도: 예측 수와 타겟 수의 차이
            correct_predictions += min(len(pred_boxes), len(target_boxes))

    accuracy = correct_predictions / max(total_targets, 1)
    precision = correct_predictions / max(total_predictions, 1) if total_predictions > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'total_predictions': total_predictions,
        'total_targets': total_targets
    }

# 학습 루프

history = {
    'train_loss': [],
    'val_accuracy': [],
    'val_precision': [],
    'learning_rate': []
}

best_accuracy = 0.0
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{num_epochs}")
    print(f"{'='*60}")

    # 학습
    train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch)

    # 검증
    val_metrics = evaluate(model, val_loader, device)

    # Learning rate 업데이트
    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # 기록
    history['train_loss'].append(train_metrics['total_loss'])
    history['val_accuracy'].append(val_metrics['accuracy'])
    history['val_precision'].append(val_metrics['precision'])
    history['learning_rate'].append(current_lr)

    # 결과 출력
    print(f"\nEpoch {epoch} 결과:")
    print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
    print(f"    - Classifier: {train_metrics['loss_classifier']:.4f}")
    print(f"    - Box Reg: {train_metrics['loss_box_reg']:.4f}")
    print(f"    - Objectness: {train_metrics['loss_objectness']:.4f}")
    print(f"    - RPN Box Reg: {train_metrics['loss_rpn_box_reg']:.4f}")
    print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Val Precision: {val_metrics['precision']:.4f}")
    print(f"  Learning Rate: {current_lr:.6f}")

    # Best model 저장
    if val_metrics['accuracy'] > best_accuracy:
        best_accuracy = val_metrics['accuracy']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': val_metrics['accuracy'],
            'history': history
        }, 'best_model.pth')
        print(f"  ✓ Best model saved! (Accuracy: {val_metrics['accuracy']:.4f})")

total_time = time.time() - start_time

# 학습 결과 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(history['train_loss'], marker='o', linewidth=2, label='Train Loss')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Accuracy & Precision
axes[1].plot(history['val_accuracy'], marker='o', linewidth=2, label='Accuracy', color='green')
axes[1].plot(history['val_precision'], marker='s', linewidth=2, label='Precision', color='orange')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

# Learning Rate
axes[2].plot(history['learning_rate'], marker='o', linewidth=2, color='purple', label='LR')
axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].set_ylabel('Learning Rate', fontsize=12)
axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[2].set_yscale('log')
axes[2].legend(fontsize=11)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Best 모델 로드
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Best model (Epoch {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.4f}) 로드 완료!")

# 추론 및 시각화 함수
def predict_and_visualize(model, dataset, idx, device, conf_threshold=0.5):
    model.eval()

    # 이미지와 타겟 가져오기
    img, target = dataset[idx]

    # 추론
    with torch.no_grad():
        prediction = model([img.to(device)])[0]

    # CPU로 이동
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()

    pred_boxes = prediction['boxes'].cpu().numpy()
    pred_scores = prediction['scores'].cpu().numpy()
    pred_labels = prediction['labels'].cpu().numpy()

    gt_boxes = target['boxes'].numpy()

    # Confidence threshold 적용
    mask = pred_scores >= conf_threshold
    pred_boxes = pred_boxes[mask]
    pred_scores = pred_scores[mask]

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Ground Truth
    img_gt = img_np.copy()
    for box in gt_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_gt, 'Person', (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    axes[0].imshow(img_gt)
    axes[0].set_title(f'Ground Truth ({len(gt_boxes)} persons)', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Prediction
    img_pred = img_np.copy()
    for box, score in zip(pred_boxes, pred_scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_pred, (x1, y1), (x2, y2), (255, 0, 0), 3)
        text = f'Person: {score:.2f}'
        cv2.putText(img_pred, text, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    axes[1].imshow(img_pred)
    axes[1].set_title(f'Prediction ({len(pred_boxes)} persons, threshold: {conf_threshold})',
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# 여러 샘플 테스트
print("\n=== 추론 결과 샘플 ===\n")

# Validation 세트에서 랜덤 샘플 선택
sample_indices = np.random.choice(len(val_dataset), size=min(5, len(val_dataset)), replace=False)

for i, idx in enumerate(sample_indices, 1):
    print(f"Sample {i}:")
    predict_and_visualize(model, val_dataset, idx, device, conf_threshold=0.5)
    print("-" * 60)

# Confidence Threshold 비교
print("\n=== Confidence Threshold 영향 분석 ===\n")

sample_idx = sample_indices[0]
thresholds = [0.3, 0.5, 0.7, 0.9]

img, target = val_dataset[sample_idx]
img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()

with torch.no_grad():
    prediction = model([img.to(device)])[0]

pred_boxes = prediction['boxes'].cpu().numpy()
pred_scores = prediction['scores'].cpu().numpy()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, thresh in enumerate(thresholds):
    mask = pred_scores >= thresh
    boxes = pred_boxes[mask]
    scores = pred_scores[mask]

    img_temp = img_np.copy()
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_temp, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = f'{score:.2f}'
        cv2.putText(img_temp, text, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    axes[idx].imshow(img_temp)
    axes[idx].set_title(f'Threshold: {thresh} ({len(boxes)} detections)',
                       fontsize=12, fontweight='bold')
    axes[idx].axis('off')

    print(f"Threshold {thresh}: {len(boxes)}개 검출")

plt.tight_layout()
plt.show()

# 최종 성능 평가

final_metrics = evaluate(model, val_loader, device)

print(f"\nValidation Set 최종 결과:")
print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
print(f"  Precision: {final_metrics['precision']:.4f}")
print(f"  Total Predictions: {final_metrics['total_predictions']}")
print(f"  Total Targets: {final_metrics['total_targets']}")

# 최종 모델 저장
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': 2,
    'history': history,
    'final_metrics': final_metrics
}, 'final_model.pth')

# 요약 정리
print("\n\\실습 요약:")
print(f"  • 데이터셋: Penn-Fudan Pedestrian Detection")
print(f"  • Train 샘플: {len(train_dataset)}개")
print(f"  • Val 샘플: {len(val_dataset)}개")
print(f"  • Epochs: {num_epochs}")
print(f"  • Best Validation Accuracy: {best_accuracy:.4f}")
print(f"  • 총 학습 시간: {total_time/60:.1f}분")