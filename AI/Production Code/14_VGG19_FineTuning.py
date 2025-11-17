import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # Mixed Precision Training
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import time
import os
from datetime import datetime

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 고급 데이터 증강
# 기본 설정
def setup_environment(seed=42): # 환경 설정
    # 시드 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (12, 8)
    
    return device

device = setup_environment(seed=42)
if torch.cuda.is_available():
    print(f'GPU 이름: {torch.cuda.get_device_name(0)}')
    print(f'GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# 데이터 전처리

def get_advanced_transforms(): # 데이터 증강 후 생성
    # 훈련 데이터 변환
    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),  # 랜덤 크롭
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # 랜덤 회전
        transforms.ColorJitter(
            brightness=0.3,  # 밝기 변화
            contrast=0.3,    # 대비 변화
            saturation=0.3,  # 채도 변화
            hue=0.1          # 색조 변화
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))  # CutOut 효과
    ])
    
    # 검증 데이터 변환 (증강 없음)
    val_transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform

# 데이터 로드 함수
def load_data(batch_size=50, num_workers=4): # CIFAR-10 데이터셋 로드
    """
    Args:
        batch_size: 배치 크기
        num_workers: 데이터 로딩 워커 수
    """
    train_transform, val_transform = get_advanced_transforms()
    
    # 데이터셋 생성
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=val_transform
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # GPU 전송 속도 향상
        persistent_workers=True  # 워커 재사용
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

train_loader, val_loader, train_dataset, val_dataset = load_data(
    batch_size=50,
    num_workers=2
)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 모델 함수 정의 및 생성
def create_vgg19_bn(num_classes=10, pretrained=True): # 사전 학습된 VGG-19-BN 불러오기
    if pretrained:
        model = models.vgg19_bn(weights='IMAGENET1K_V1')
    else:
        model = models.vgg19_bn(weights=None)
    
    # classifier 구조 출력
    print('원본 Classifier 구조:')
    print(model.classifier)
    print('\n' + '=' * 60)
    
    # 마지막 FC 레이어 교체
    num_features = model.classifier[6].in_features
    print(f'원본 FC 레이어: Linear(in_features={num_features}, out_features=1000)')
    
    # CIFAR-10의 10개 클래스에 맞게 교체
    model.classifier[6] = nn.Linear(num_features, num_classes)
    print(f'교체된 FC 레이어: Linear(in_features={num_features}, out_features={num_classes})')
    print('=' * 60)
    
    # 모델을 GPU로 이동
    model = model.to(device)
    
    return model

def count_parameters(model): # 모델의 파라미터 수 계산
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

model = create_vgg19_bn(num_classes=10, pretrained=True)
total_params, trainable_params = count_parameters(model)

# 학습 함수 정의 및 생성

def setup_training(model, train_loader, num_epochs=10, lr=0.001):
    """
    Returns:
        criterion: 손실 함수
        optimizer: 최적화 함수
        scheduler: 학습률 스케줄러
        scaler: Mixed Precision용 GradScaler
    """
    # 손실 함수
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label Smoothing
    
    # 최적화 함수
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,  # L2 정규화
        nesterov=True  # Nesterov Momentum
    )
    
    # 학습률 스케줄러 (OneCycleLR)
    # Learning Rate를 삼각파 형태로 변화시켜 빠른 수렴과 좋은 성능 달성
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr * 10,  # 최대 학습률
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # warm-up 비율
        anneal_strategy='cos',  # 코사인 감소
        div_factor=10,  # 초기 학습률 = max_lr / div_factor
        final_div_factor=100  # 최종 학습률 = 초기 학습률 / final_div_factor
    )
    
    # Mixed Precision Training을 위한 GradScaler
    scaler = GradScaler()
    
    return criterion, optimizer, scheduler, scaler

# 학습 설정
num_epochs = 10
criterion, optimizer, scheduler, scaler = setup_training(
    model,
    train_loader,
    num_epochs=num_epochs,
    lr=0.001
)

# TensorBoard 설정

log_dir = f'runs/vgg19_bn_{datetime.now().strftime("%Y%m%d_%H%M%S")}' # 연월일_시분초 저장
writer = SummaryWriter(log_dir=log_dir)

# 학습 과정을 실시간으로 모니터링
# 손실, 정확도, 학습률 등을 시각화
# 모델 그래프 및 히스토그램 확인

# 학습 및 검증 함수 (Mixed Precision) 

# 훈련 함수 (Mixed Precision) # 한 에폭훈련 
def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Training
        with autocast():  # FP16 모드 : 순전파
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # 그래디언트 스케일링 및 역전파
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 학습률 스케줄러 업데이트 (OneCycleLR은 매 배치마다 업데이트)
        scheduler.step()
        
        # 통계 계산
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 진행 상태 업데이트
        current_acc = 100 * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.2f}%',
            'lr': f'{current_lr:.6f}'
        })
        
        # TensorBoard 로깅 (매 100 배치마다)
        if batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss_step', loss.item(), global_step)
            writer.add_scalar('Train/Acc_step', current_acc, global_step)
            writer.add_scalar('Train/LR', current_lr, global_step)
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# 검증 함수
def validate(model, val_loader, criterion, device, epoch):
    """
    검증 데이터로 모델 평가
    
    Returns:
        epoch_loss: 평균 손실
        epoch_acc: 정확도
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Mixed Precision (검증 시에도 적용 가능)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # 통계 계산
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

print('학습 및 검증 함수 정의 완료')

# 모델 체크 포인트 저장
# 체크포인트 저장 함수
# 모델 가중치
# 최적화 함수 상태
# 스케줄러 상태
# 학습 히스토리
def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, history, filename):
    """
    모델 체크포인트 저장
    
    Args:
        filename: 저장할 파일명
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'history': history
    }
    torch.save(checkpoint, filename)
    print(f'체크포인트 저장: {filename} (Val Acc: {val_acc:.2f}%)')

# 체크포인트 로드 함수
def load_checkpoint(model, optimizer, scheduler, filename):
    """
    체크포인트 로드
    
    Returns:
        start_epoch: 시작 에폭
        history: 학습 히스토리
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    history = checkpoint['history']
    
    print(f'체크포인트 로드: {filename}')
    print(f'재시작 에폭: {start_epoch}')
    print(f'이전 검증 정확도: {checkpoint["val_acc"]:.2f}%')
    
    return start_epoch, history

print('체크포인트 함수 정의 완료')

# 전체 학습 실핼

# Mixed Precision Training
# OneCycleLR 스케줄링
# TensorBoard 실시간 로깅
# 최적 모델 자동 저장

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}
best_val_acc = 0.0
best_epoch = 0
start_time = time.time()

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
    )
    val_loss, val_acc = validate(
        model, val_loader, criterion, device, epoch
    )
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # TensorBoard 로깅 (에폭 단위)
    writer.add_scalars('Loss', {
        'train': train_loss,
        'val': val_loss
    }, epoch)

    writer.add_scalars('Accuracy', {
        'train': train_acc,
        'val': val_acc
    }, epoch)

    # 결과 출력
    print(f'\n에폭 {epoch+1} 결과:')
    print(f'  훈련 손실: {train_loss:.4f} | 훈련 정확도: {train_acc:.2f}%')
    print(f'  검증 손실: {val_loss:.4f} | 검증 정확도: {val_acc:.2f}%')
    
    # 최적 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_acc, history,
            'vgg19_bn_best.pth'
        )

    # 주기적으로 체크포인트 저장
    if (epoch + 1) % 5 == 0:
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_acc, history,
            f'vgg19_bn_epoch{epoch+1}.pth'
        )
# 학습 종료
elapsed_time = time.time() - start_time

print(f'총 소요 시간: {elapsed_time/60:.2f}분')
print(f'에폭당 평균 시간: {elapsed_time/num_epochs:.2f}초')
print(f'최고 검증 정확도: {best_val_acc:.2f}% (에폭 {best_epoch})')

writer.close()
checkpoint = torch.load('vgg19_bn_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Grad-CAM 시각화
def visualize_gradcam(model, image, true_label, pred_label, class_names, device):
    """
    Grad-CAM을 사용하여 모델의 주목 영역 시각화
    
    Args:
        model: VGG-19-BN 모델
        image: 입력 이미지 (정규화된 텐서)
        true_label: 실제 레이블
        pred_label: 예측 레이블
        class_names: 클래스 이름 리스트
    """
    # VGG-19-BN의 마지막 Conv 레이어 타겟
    target_layers = [model.features[-1]]
    
    # Grad-CAM 객체 생성
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 입력 이미지 준비
    input_tensor = image.unsqueeze(0).to(device) # (batch,C,H,W)
    
    # Grad-CAM 생성 (예측된 클래스에 대해)
    targets = [ClassifierOutputTarget(pred_label)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # 이미지 역정규화
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) # 채널별로 적용
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) # 채널별로 적용
    img_denorm = image * std + mean
    img_denorm = torch.clamp(img_denorm, 0, 1)
    rgb_img = img_denorm.permute(1, 2, 0).cpu().numpy() # (H,W,C)
    
    # Grad-CAM 오버레이
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 원본 이미지
    axes[0].imshow(rgb_img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Grad-CAM 히트맵
    axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 오버레이
    axes[2].imshow(cam_image)
    color = 'green' if true_label == pred_label else 'red'
    axes[2].set_title(
        f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}',
        color=color,
        fontsize=12,
        fontweight='bold'
    )
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Grad-CAM 실행 (여러 샘플)
def run_gradcam_on_samples(model, val_loader, class_names, device, num_samples=6):
    """
    여러 샘플에 대해 Grad-CAM 시각화
    """
    model.eval()
    
    # 샘플 가져오기
    dataiter = iter(val_loader)
    images, labels = next(dataiter)
    
    # 예측
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
    
    # 샘플별 Grad-CAM 생성
    for i in range(min(num_samples, len(images))):
        print(f'\n샘플 {i+1}:')
        visualize_gradcam(
            model,
            images[i],
            labels[i].item(),
            predicted[i].item(),
            class_names,
            device
        )
run_gradcam_on_samples(model, val_loader, class_names, device, num_samples=6)

# 최종 평가

# 예측 함수
def get_all_predictions(model, loader, device):
    """모든 데이터에 대한 예측 수행"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Predicting'):
            inputs = inputs.to(device)
            
            with autocast():
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

# 혼동 행렬 시각화
def plot_confusion_matrix(y_true, y_pred, class_names):
    """혼동 행렬 시각화 및 분석"""
    cm = confusion_matrix(y_true, y_pred)
    
    # 정규화된 혼동 행렬
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 원본 혼동 행렬
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar_kws={'label': 'Count'}
    )
    axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True', fontsize=12, fontweight='bold')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # 정규화된 혼동 행렬
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar_kws={'label': 'Proportion'},
        vmin=0,
        vmax=1
    )
    axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True', fontsize=12, fontweight='bold')
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 클래스별 성능
    print('\n클래스별 성능:')
    print('=' * 70)
    print(f'{"Class":15s} {"Precision":>10s} {"Recall":>10s} {"F1-Score":>10s} {"Support":>10s}')
    print('-' * 70)
    
    for i, name in enumerate(class_names):
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = cm[i, :].sum()
        
        print(f'{name:15s} {precision:10.4f} {recall:10.4f} {f1:10.4f} {support:10d}')
    
y_pred, y_true, y_probs = get_all_predictions(model, val_loader, device)
# 전체 정확도
accuracy = 100 * (y_pred == y_true).sum() / len(y_true)
print(f'\n최종 테스트 정확도: {accuracy:.2f}%')

# 혼동 행렬
plot_confusion_matrix(y_true, y_pred, class_names)

# 상세 분류 리포트
print('\n\n상세 분류 리포트:')
print('=' * 60)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# 학습 곡선 시각화
def plot_training_history(history):
    """학습 히스토리 시각화"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 손실 곡선
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=13, fontweight='bold')
    axes[0].set_title('Loss Curve', fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=11, loc='upper right')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # 정확도 곡선
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2, markersize=6)
    axes[1].plot(epochs, history['val_acc'], 'r-s', label='Val Acc', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    axes[1].set_title('Accuracy Curve', fontsize=15, fontweight='bold')
    axes[1].legend(fontsize=11, loc='lower right')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    # 최종 통계
    print('\n최종 학습 통계:')
    print(f'최종 훈련 손실: {history["train_loss"][-1]:.4f}')
    print(f'최종 훈련 정확도: {history["train_acc"][-1]:.2f}%')
    print(f'최종 검증 손실: {history["val_loss"][-1]:.4f}')
    print(f'최종 검증 정확도: {history["val_acc"][-1]:.2f}%')
    print(f'\n최고 검증 정확도: {max(history["val_acc"]):.2f}%')
    print(f'최저 검증 손실: {min(history["val_loss"]):.4f}')

# Mixed Precision Training
# FP16과 FP32를 혼합하여 학습 속도 향상
# with autocast():
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)

# scaler.scale(loss).backward()
# scaler.step(optimizer)
# scaler.update()

# OneCycleLR 스케줄러
# scheduler = OneCycleLR(
#     optimizer,
#     max_lr=lr * 10,
#     epochs=num_epochs,
#     steps_per_epoch=len(train_loader)
# )


