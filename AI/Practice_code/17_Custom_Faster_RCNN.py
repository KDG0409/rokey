import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from pathlib import Path
import time
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 함수 정의
# Custom Dataset 클래스 정의
class CustomObjectDetectionDataset(data.Dataset):
    """
    커스텀 객체 검출 데이터셋
    COCO 형식의 annotation 지원
    """
    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = Path(root_dir)
        self.transforms = transforms

        # Annotation 로드
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        # 이미지 ID 리스트
        self.image_ids = [img['id'] for img in self.coco_data['images']]

        # 카테고리 매핑
        self.categories = {cat['id']: cat['name']
                          for cat in self.coco_data['categories']}
        self.num_classes = len(self.categories) + 1  # +1 for background

        # 이미지별 annotation 그룹화 # 주석
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        print(f"Dataset 로드 완료: {len(self.image_ids)}개 이미지, {self.num_classes-1}개 클래스")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx): # 이미지 로드
        img_id = self.image_ids[idx]
        img_info = next(img for img in self.coco_data['images'] if img['id'] == img_id)
        img_path = self.root_dir / img_info['file_name']

        img = Image.open(img_path).convert("RGB")

        # Annotation 가져오기
        anns = self.img_to_anns.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        # Tensor로 변환
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': areas,
            'iscrowd': iscrowd
        }

        # Transform 적용
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
    
# Transform 정의 
def get_transform(train=True): # 학습/검증용 Transform
    transforms_list = []
    transforms_list.append(T.ToTensor()) # PIL Image를 Tensor로 변환
    if train:
        transforms_list.append(T.RandomHorizontalFlip(0.5)) # 학습 시 Data Augmentation
    return T.Compose(transforms_list)

# 모델 생성 함수
def get_model(num_classes, pretrained=True): # Faster R-CNN 모델 생성 및 커스터마이징

    model = fasterrcnn_resnet50_fpn(weights='DEFAULT' if pretrained else None) # 사전학습된 모델 로드
    # Classifier head 교체 (커스텀 클래스 수에 맞게)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# mAP 계산 함수
def calculate_iou_batch(boxes1, boxes2): # 두 박스 세트 간의 IoU 계산 (벡터화)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # :2 x,y
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # 2: w,h

    wh = (rb - lt).clamp(min=0) # max: rb, min: lt -> 교집합 영역 내의 w,h
    # clamp(min=0)의 의미: 교집합이 없을 때 음수가 나오는 것을 방지하고자 최솟값 설정
    inter = wh[:, :, 0] * wh[:, :, 1] # -> 교집합 영역 내의 w*h

    union = area1[:, None] + area2 - inter # None->자동으로 행렬구조 생성 [:, None]->[N,1]
    iou = inter / union

    return iou

# Average Precision 계산
def compute_ap(recall, precision): # 11-point interpolation
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.
    return ap

# mAP 계산
def evaluate_map(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    all_detections = []
    all_ground_truths = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                all_ground_truths.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })
                all_detections.append({
                    'boxes': output['boxes'].cpu(),
                    'scores': output['scores'].cpu(),
                    'labels': output['labels'].cpu()
                })
    num_classes = max([max(gt['labels']) for gt in all_ground_truths]) + 1
    aps = []

    for cls in range(1, num_classes):  # Skip background
        # 해당 클래스만 필터링
        cls_detections = []
        cls_ground_truths = []

        for det, gt in zip(all_detections, all_ground_truths):
            det_mask = det['labels'] == cls
            gt_mask = gt['labels'] == cls

            if det_mask.sum() > 0:
                cls_detections.append({
                    'boxes': det['boxes'][det_mask],
                    'scores': det['scores'][det_mask]
                })
            else:
                cls_detections.append({'boxes': torch.empty(0, 4), 'scores': torch.empty(0)})

            cls_ground_truths.append(gt['boxes'][gt_mask])

        # Score로 정렬
        all_scores = []
        all_tp = []
        num_gt = sum([len(gt) for gt in cls_ground_truths])

        for det, gt in zip(cls_detections, cls_ground_truths):
            if len(det['scores']) == 0:
                continue

            for score, box in zip(det['scores'], det['boxes']):
                all_scores.append(score.item())

                if len(gt) == 0:
                    all_tp.append(0)
                else:
                    ious = calculate_iou_batch(box.unsqueeze(0), gt)
                    max_iou = ious.max().item()
                    all_tp.append(1 if max_iou >= iou_threshold else 0)

        if len(all_scores) == 0 or num_gt == 0:
            continue

        # Precision-Recall 계산
        indices = np.argsort(all_scores)[::-1]
        tp = np.array(all_tp)[indices]
        fp = 1 - tp

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap(recalls, precisions)
        aps.append(ap)

    mAP = np.mean(aps) if len(aps) > 0 else 0.0
    return mAP, aps

# 학습 함수
def train_one_epoch(model, optimizer, data_loader, device, epoch): # 1 에폭 학습
    model.train()
    total_loss = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
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

        total_loss += losses.item()
        pbar.set_postfix({'loss': f'{losses.item():.4f}'})

    return total_loss / len(data_loader)

# 데이터셋 준비 (예시 - 실제 데이터로 교체 필요)
train_dataset = CustomObjectDetectionDataset(
    root_dir='path/to/images',
    annotation_file='path/to/annotations.json',
    transforms=get_transform(train=True)
)

val_dataset = CustomObjectDetectionDataset(
    root_dir='path/to/images',
    annotation_file='path/to/val_annotations.json',
    transforms=get_transform(train=False)
)

# 학습 파이프라인
def train_model(num_classes, train_dataset, val_dataset,
                num_epochs=10, batch_size=4, lr=0.005):
    """
    전체 학습 파이프라인
    """
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda x: tuple(zip(*x))
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # 모델 초기화
    model = get_model(num_classes, pretrained=True)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 학습 기록
    history = {
        'train_loss': [],
        'val_mAP': [],
        'learning_rate': []
    }

    best_mAP = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")

        # 학습
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # 검증
        val_mAP, _ = evaluate_map(model, val_loader, device)

        # Scheduler step
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 기록
        history['train_loss'].append(train_loss)
        history['val_mAP'].append(val_mAP)
        history['learning_rate'].append(current_lr)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val mAP: {val_mAP:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        # Best model 저장
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP': val_mAP,
            }, 'best_model.pth')
            print(f"Best model saved! (mAP: {val_mAP:.4f})")

    return model, history

# 학습 결과 시각화
def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['train_loss'], marker='o', label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # mAP
    axes[1].plot(history['val_mAP'], marker='o', color='green', label='Val mAP')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mAP')
    axes[1].set_title('Validation mAP')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Learning Rate
    axes[2].plot(history['learning_rate'], marker='o', color='orange', label='LR')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# 추론 및 시각화 함수
def predict_and_visualize(model, image_path, device, conf_threshold=0.5,
                         class_names=None):
    model.eval()

    # 이미지 로드
    img = Image.open(image_path).convert("RGB")
    img_tensor = T.ToTensor()(img).unsqueeze(0).to(device)

    # 추론
    with torch.no_grad():
        start_time = time.time()
        predictions = model(img_tensor)[0]
        inference_time = time.time() - start_time

    # 필터링
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()

    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    # 시각화
    img_np = np.array(img)
    img_draw = img_np.copy()

    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        color = tuple((np.array(colors[label % 20][:3]) * 255).astype(int).tolist())

        # 박스
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

        # 레이블
        class_name = class_names[label] if class_names else f"Class {label}"
        text = f'{class_name}: {score:.2f}'
        cv2.putText(img_draw, text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_draw)
    plt.axis('off')
    plt.title(f'검출 결과 ({len(boxes)}개 객체, 추론 시간: {inference_time*1000:.1f}ms)')
    plt.tight_layout()
    plt.show()

    return boxes, scores, labels

# 실무 팁 및 사용 예시
print("""
=== Faster R-CNN Fine-tuning 실무 가이드 ===

1. 데이터 준비
   - COCO 형식의 annotation 권장
   - 클래스당 최소 100개 이상의 샘플
   - Train/Val split: 80/20

2. Hyperparameter 튜닝
   - Learning rate: 0.005 (초기값)
   - Batch size: GPU 메모리에 따라 조정
   - Epochs: 10-50 (early stopping 권장)
   - IoU threshold: 0.5 (기본값)

3. 성능 개선 전략
   - Data Augmentation (flip, crop, color jitter)
   - Anchor box 크기/비율 조정
   - Backbone 변경 (ResNet50 → ResNet101)
   - Learning rate scheduling
   - Mixed precision training (FP16)

4. 평가 및 디버깅
   - mAP로 전체 성능 평가
   - 클래스별 AP로 세부 분석
   - False Positive/Negative 분석
   - Confusion matrix 활용

5. 배포 최적화
   - TorchScript 변환
   - ONNX 내보내기
   - Quantization 적용
   - TensorRT 가속

사용 예시:

# 1. 데이터셋 준비
train_dataset = CustomObjectDetectionDataset(
    root_dir='data/images',
    annotation_file='data/train.json',
    transforms=get_transform(train=True)
)

# 2. 학습
model, history = train_model(
    num_classes=3,  # background + 2 classes
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    num_epochs=20,
    batch_size=4
)

# 3. 결과 시각화
plot_training_history(history)

# 4. 추론
boxes, scores, labels = predict_and_visualize(
    model, 'test.jpg', device,
    class_names=['bg', 'cat', 'dog']
)

""")
