import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
import time
import copy

# 기본 설정

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 디바이스: {device}')

torch.manual_seed(42)
np.random.seed(42)

# 데이터 전처리

transform_basic = transforms.Compose([
    transforms.Resize(224),  # ResNet 입력 크기에 맞춤
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 평균
                        std=[0.229, 0.224, 0.225])   # ImageNet 표준편차
])

full_train_dataset = datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform_basic)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform_basic)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(class_names)

# 작은 데이터셋 시나리오: 클래스당 50개씩만 사용 (총 500개)
samples_per_class = 50  # 클래스당 샘플 수

# 각 클래스별로 인덱스 수집
class_indices = {i: [] for i in range(num_classes)}  # 클래스별 인덱스 저장 딕셔너리

# 전체 데이터셋을 순회하며 클래스별로 인덱스 분류
for idx, (_, label) in enumerate(full_train_dataset):
    if len(class_indices[label]) < samples_per_class:  # 클래스당 샘플 수 제한
        class_indices[label].append(idx)
    
# 선택된 인덱스들을 하나의 리스트로 합치기
selected_indices = []
for indices in class_indices.values():
    selected_indices.extend(indices)

# 작은 데이터셋 생성 (Subset 사용)
small_train_dataset = Subset(full_train_dataset, selected_indices)

# 데이터 로더 생성
batch_size = 32  # 배치 크기

train_loader = DataLoader(small_train_dataset, # 학습용 데이터 로더
                         batch_size=batch_size,
                         shuffle=True,  # 데이터 섞기
                         num_workers=2)  # 병렬 처리

test_loader = DataLoader(test_dataset, # 테스트용 데이터 로더
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=2)

# 함수 정의

def show_sample_images(dataset, num_images=5): # 데이터셋에서 샘플 이미지를 보여주는 함수
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        img, label = dataset[i]  # i번째 이미지와 레이블 가져오기

        # 정규화 해제 (시각화를 위해)
        img = img.numpy().transpose((1, 2, 0))  # CHW -> HWC
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean  # 정규화 역변환
        img = np.clip(img, 0, 1)  # 0-1 범위로 클리핑

        # 이미지 표시
        axes[i].imshow(img)
        axes[i].set_title(f'{class_names[label]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def train_one_epoch(model, dataloader, criterion, optimizer, device): # 한 에폭 학습 함수
    model.train()  # 학습 모드로 전환

    running_loss = 0.0  # 누적 손실
    correct = 0  # 맞춘 개수
    total = 0  # 전체 개수

    # 데이터 로더에서 배치 단위로 학습
    for inputs, labels in dataloader:
        inputs = inputs.to(device)  # GPU로 이동
        labels = labels.to(device)  # GPU로 이동

        optimizer.zero_grad()  # 그래디언트 초기화

        outputs = model(inputs)  # 순전파
        loss = criterion(outputs, labels)  # 손실 계산

        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트

        # 통계 업데이트
        running_loss += loss.item() * inputs.size(0) # 평균 손실과 배치 사이즈의 곱
        _, predicted = outputs.max(1)  # 예측값(인덱스)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # 에폭 평균 계산
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, device): # 평가 함수 매우 중요
    model.eval()  # 평가 모드로 전환

    running_loss = 0.0
    correct = 0
    total = 0

    # 그래디언트 계산 비활성화 (평가 시에는 불필요)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  # 순전파만 수행
            loss = criterion(outputs, labels)

            # 통계 업데이트
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # 평균 계산
    eval_loss = running_loss / total
    eval_acc = 100.0 * correct / total

    return eval_loss, eval_acc

# 층별 동결 전략 비교
num_epochs = 10  # 에폭 수 (빠른 실습을 위해 10으로 설정)
learning_rate = 0.001  # 학습률

# Full Freeze (백본 전체 고정)
model_freeze = models.resnet18(pretrained=True)

for param in model_freeze.parameters(): # 모든 파라미터 동결
    param.requires_grad = False  # 그래디언트 계산 비활성화

num_features = model_freeze.fc.in_features  # 원래 fc의 입력 차원
model_freeze.fc = nn.Linear(num_features, num_classes)  # 새로운 분류기 정의

model_freeze = model_freeze.to(device)
criterion = nn.CrossEntropyLoss()  # 분류 손실
optimizer_freeze = optim.Adam(model_freeze.fc.parameters(), lr=learning_rate)  # 분류기만 최적화

history_freeze = {'train_loss': [], 'train_acc': [], 'test_acc': []}
start_time = time.time()

for epoch in range(num_epochs):
    # 한 에폭 학습
    train_loss, train_acc = train_one_epoch(model_freeze, train_loader,
                                           criterion, optimizer_freeze, device)

    # 테스트 데이터로 평가
    _, test_acc = evaluate_model(model_freeze, test_loader, criterion, device)

    # 기록 저장
    history_freeze['train_loss'].append(train_loss)
    history_freeze['train_acc'].append(train_acc)
    history_freeze['test_acc'].append(test_acc)

    # 진행상황 출력
    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train Loss: {train_loss:.4f}, '
          f'Train Acc: {train_acc:.2f}%, '
          f'Test Acc: {test_acc:.2f}%')

elapsed_time_freeze = time.time() - start_time
print(f'\n학습 완료! 소요 시간: {elapsed_time_freeze:.2f}초')
print(f'최종 테스트 정확도: {history_freeze["test_acc"][-1]:.2f}%')

# Partial Fine-tuning (마지막 블록만 해제)
model_partial = models.resnet18(pretrained=True)

for param in model_partial.parameters(): # 먼저 모든 파라미터 동결
    param.requires_grad = False
for param in model_partial.layer4.parameters(): # 마지막 블록(layer4)만 해제
    param.requires_grad = True  # layer4만 학습 가능하게

model_partial.fc = nn.Linear(model_partial.fc.in_features, num_classes)
model_partial = model_partial.to(device)

optimizer_partial = optim.Adam( # layer4 + fc만 최적화
    [{'params': model_partial.layer4.parameters()},
     {'params': model_partial.fc.parameters()}],
    lr=learning_rate
)

print(f'학습 가능한 파라미터 수: {sum(p.numel() for p in model_partial.parameters() if p.requires_grad):,}')
history_partial = {'train_loss': [], 'train_acc': [], 'test_acc': []}
start_time = time.time()

for epoch in range(num_epochs):
    # 한 에폭 학습
    train_loss, train_acc = train_one_epoch(model_partial, train_loader,
                                           criterion, optimizer_partial, device)

    # 테스트 데이터로 평가
    _, test_acc = evaluate_model(model_partial, test_loader, criterion, device)

    # 기록 저장
    history_partial['train_loss'].append(train_loss)
    history_partial['train_acc'].append(train_acc)
    history_partial['test_acc'].append(test_acc)

    # 진행상황 출력
    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train Loss: {train_loss:.4f}, '
          f'Train Acc: {train_acc:.2f}%, '
          f'Test Acc: {test_acc:.2f}%')

elapsed_time_partial = time.time() - start_time
print(f'\n학습 완료! 소요 시간: {elapsed_time_partial:.2f}초')
print(f'최종 테스트 정확도: {history_partial["test_acc"][-1]:.2f}%')

# Full Fine-tuning (모든 층 학습)/ 시험기출(중요)
model_full = models.resnet18(pretrained=True) # 시험기출(중요)

for param in model_full.parameters(): # 시험기출(중요)
    param.requires_grad = True  # 모든 층 학습 가능

model_full.fc = nn.Linear(model_full.fc.in_features, num_classes) # 시험기출(중요)
model_full = model_full.to(device) # 시험기출(중요)

optimizer_full = optim.Adam(model_full.parameters(), lr=learning_rate * 0.1)  # 학습률 낮춤/ 시험기출(중요)

history_full = {'train_loss': [], 'train_acc': [], 'test_acc': []}
start_time = time.time()

for epoch in range(num_epochs):
    # 한 에폭 학습
    train_loss, train_acc = train_one_epoch(model_full, train_loader,
                                           criterion, optimizer_full, device)

    # 테스트 데이터로 평가
    _, test_acc = evaluate_model(model_full, test_loader, criterion, device)

    # 기록 저장
    history_full['train_loss'].append(train_loss)
    history_full['train_acc'].append(train_acc)
    history_full['test_acc'].append(test_acc)

    # 진행상황 출력
    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train Loss: {train_loss:.4f}, '
          f'Train Acc: {train_acc:.2f}%, '
          f'Test Acc: {test_acc:.2f}%')

elapsed_time_full = time.time() - start_time
print(f'\n학습 완료! 소요 시간: {elapsed_time_full:.2f}초')
print(f'최종 테스트 정확도: {history_full["test_acc"][-1]:.2f}%')

# 결과 비교 및 분석 (세 가지 전략 결과 비교 그래프)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

epochs = range(1, num_epochs + 1)

# 1. 학습 정확도 비교
axes[0].plot(epochs, history_freeze['train_acc'], 'b-o', label='Full Freeze', linewidth=2)
axes[0].plot(epochs, history_partial['train_acc'], 'g-s', label='Partial Fine-tune', linewidth=2)
axes[0].plot(epochs, history_full['train_acc'], 'r-^', label='Full Fine-tune', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Train Accuracy (%)', fontsize=12)
axes[0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 2. 테스트 정확도 비교
axes[1].plot(epochs, history_freeze['test_acc'], 'b-o', label='Full Freeze', linewidth=2)
axes[1].plot(epochs, history_partial['test_acc'], 'g-s', label='Partial Fine-tune', linewidth=2)
axes[1].plot(epochs, history_full['test_acc'], 'r-^', label='Full Fine-tune', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Test Accuracy (%)', fontsize=12)
axes[1].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 최종 결과 요약 테이블
results = [
    ['Full Freeze', history_freeze['test_acc'][-1], elapsed_time_freeze,
     history_freeze['train_acc'][-1] - history_freeze['test_acc'][-1]],
    ['Partial Fine-tune', history_partial['test_acc'][-1], elapsed_time_partial,
     history_partial['train_acc'][-1] - history_partial['test_acc'][-1]],
    ['Full Fine-tune', history_full['test_acc'][-1], elapsed_time_full,
     history_full['train_acc'][-1] - history_full['test_acc'][-1]]
]

print(f'{"Strategy":<20} {"Test Acc (%)":<15} {"Time (sec)":<15} {"Overfit Gap (%)":<15}')
print('-'*80)
for result in results:
    print(f'{result[0]:<20} {result[1]:<15.2f} {result[2]:<15.2f} {result[3]:<15.2f}')


# 차등 학습률 (Differential Learning Rate)

# 단일 학습률
model_uniform_lr = models.resnet18(pretrained=True)
for param in model_uniform_lr.parameters():
    param.requires_grad = False
for param in model_uniform_lr.layer4.parameters():
    param.requires_grad = True
model_uniform_lr.fc = nn.Linear(model_uniform_lr.fc.in_features, num_classes)
model_uniform_lr = model_uniform_lr.to(device)
base_lr = 0.001
optimizer_uniform = optim.Adam( # 모두 같은 학습률
    [{'params': model_uniform_lr.layer4.parameters()},
     {'params': model_uniform_lr.fc.parameters()}],
    lr=base_lr 
)

history_uniform_lr = {'train_loss': [], 'train_acc': [], 'test_acc': []}
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model_uniform_lr, train_loader,
                                           criterion, optimizer_uniform, device)
    _, test_acc = evaluate_model(model_uniform_lr, test_loader, criterion, device)

    history_uniform_lr['train_loss'].append(train_loss)
    history_uniform_lr['train_acc'].append(train_acc)
    history_uniform_lr['test_acc'].append(test_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

print(f'\n최종 테스트 정확도: {history_uniform_lr["test_acc"][-1]:.2f}%')

# 차등 학습률 (Differential LR)
model_diff_lr = models.resnet18(pretrained=True)
for param in model_diff_lr.parameters():
    param.requires_grad = False
for param in model_diff_lr.layer4.parameters():
    param.requires_grad = True
model_diff_lr.fc = nn.Linear(model_diff_lr.fc.in_features, num_classes)
model_diff_lr = model_diff_lr.to(device)
backbone_lr = 0.001  # 백본 학습률
head_lr = 0.01       # 헤드 학습률 (10배)
optimizer_diff = optim.Adam(
    [{'params': model_diff_lr.layer4.parameters(), 'lr': backbone_lr},  # 백본
     {'params': model_diff_lr.fc.parameters(), 'lr': head_lr}],         # 헤드
)
history_diff_lr = {'train_loss': [], 'train_acc': [], 'test_acc': []}
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model_diff_lr, train_loader,
                                           criterion, optimizer_diff, device)
    _, test_acc = evaluate_model(model_diff_lr, test_loader, criterion, device)

    history_diff_lr['train_loss'].append(train_loss)
    history_diff_lr['train_acc'].append(train_acc)
    history_diff_lr['test_acc'].append(test_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

# 결과 비교
# 학습률 전략 비교 그래프
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 1. 학습 손실 비교
axes[0].plot(epochs, history_uniform_lr['train_loss'], 'b-o', label='Uniform LR', linewidth=2)
axes[0].plot(epochs, history_diff_lr['train_loss'], 'r-s', label='Differential LR', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Train Loss', fontsize=12)
axes[0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 2. 테스트 정확도 비교
axes[1].plot(epochs, history_uniform_lr['test_acc'], 'b-o', label='Uniform LR', linewidth=2)
axes[1].plot(epochs, history_diff_lr['test_acc'], 'r-s', label='Differential LR', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Test Accuracy (%)', fontsize=12)
axes[1].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 데이터 증강 강도 비교
# 데이터 전처리
transform_weak = transforms.Compose([ # 1. Weak Augmentation (약한 증강)
    transforms.Resize(224),  # 크기 조정
    transforms.RandomHorizontalFlip(),  # 좌우 반전만
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_medium = transforms.Compose([# 2. Medium Augmentation (중간 증강)
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 랜덤 크롭
    transforms.RandomHorizontalFlip(),  # 좌우 반전
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 색상 변형
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_strong = transforms.Compose([ # 3. Strong Augmentation (강한 증강)
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # 더 큰 크롭 범위
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # 회전 추가
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # 강한 색상 변형
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset_weak = datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform_weak)
train_dataset_medium = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_medium)
train_dataset_strong = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_strong)

small_train_weak = Subset(train_dataset_weak, selected_indices)
small_train_medium = Subset(train_dataset_medium, selected_indices)
small_train_strong = Subset(train_dataset_strong, selected_indices)

loader_weak = DataLoader(small_train_weak, batch_size=batch_size,
                        shuffle=True, num_workers=2)
loader_medium = DataLoader(small_train_medium, batch_size=batch_size,
                          shuffle=True, num_workers=2)
loader_strong = DataLoader(small_train_strong, batch_size=batch_size,
                          shuffle=True, num_workers=2)

# Weak Augmentation
model_weak_aug = models.resnet18(pretrained=True) # 모델 준비
for param in model_weak_aug.parameters():
    param.requires_grad = False
for param in model_weak_aug.layer4.parameters():
    param.requires_grad = True
model_weak_aug.fc = nn.Linear(model_weak_aug.fc.in_features, num_classes)
model_weak_aug = model_weak_aug.to(device)

optimizer_weak_aug = optim.Adam(
    [{'params': model_weak_aug.layer4.parameters(), 'lr': 0.001},
     {'params': model_weak_aug.fc.parameters(), 'lr': 0.01}]
)

history_weak_aug = {'train_acc': [], 'test_acc': []}

for epoch in range(num_epochs):
    _, train_acc = train_one_epoch(model_weak_aug, loader_weak,
                                   criterion, optimizer_weak_aug, device)
    _, test_acc = evaluate_model(model_weak_aug, test_loader, criterion, device)

    history_weak_aug['train_acc'].append(train_acc)
    history_weak_aug['test_acc'].append(test_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train: {train_acc:.2f}%, Test: {test_acc:.2f}%')

print(f'최종 결과 - Test: {history_weak_aug["test_acc"][-1]:.2f}%')

# Medium Augmentation
model_medium_aug = models.resnet18(pretrained=True) # 모델 준비
for param in model_medium_aug.parameters():
    param.requires_grad = False
for param in model_medium_aug.layer4.parameters():
    param.requires_grad = True
model_medium_aug.fc = nn.Linear(model_medium_aug.fc.in_features, num_classes)
model_medium_aug = model_medium_aug.to(device)

optimizer_medium_aug = optim.Adam(
    [{'params': model_medium_aug.layer4.parameters(), 'lr': 0.001},
     {'params': model_medium_aug.fc.parameters(), 'lr': 0.01}]
)

history_medium_aug = {'train_acc': [], 'test_acc': []}

for epoch in range(num_epochs):
    _, train_acc = train_one_epoch(model_medium_aug, loader_medium,
                                   criterion, optimizer_medium_aug, device)
    _, test_acc = evaluate_model(model_medium_aug, test_loader, criterion, device)

    history_medium_aug['train_acc'].append(train_acc)
    history_medium_aug['test_acc'].append(test_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train: {train_acc:.2f}%, Test: {test_acc:.2f}%')

print(f'최종 결과 - Test: {history_medium_aug["test_acc"][-1]:.2f}%')

# Strong Augmentation
model_strong_aug = models.resnet18(pretrained=True) # 모델 준비
for param in model_strong_aug.parameters():
    param.requires_grad = False
for param in model_strong_aug.layer4.parameters():
    param.requires_grad = True
model_strong_aug.fc = nn.Linear(model_strong_aug.fc.in_features, num_classes)
model_strong_aug = model_strong_aug.to(device)

optimizer_strong_aug = optim.Adam(
    [{'params': model_strong_aug.layer4.parameters(), 'lr': 0.001},
     {'params': model_strong_aug.fc.parameters(), 'lr': 0.01}]
)

history_strong_aug = {'train_acc': [], 'test_acc': []}

for epoch in range(num_epochs):
    _, train_acc = train_one_epoch(model_strong_aug, loader_strong,
                                   criterion, optimizer_strong_aug, device)
    _, test_acc = evaluate_model(model_strong_aug, test_loader, criterion, device)

    history_strong_aug['train_acc'].append(train_acc)
    history_strong_aug['test_acc'].append(test_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] - '
          f'Train: {train_acc:.2f}%, Test: {test_acc:.2f}%')

print(f'최종 결과 - Test: {history_strong_aug["test_acc"][-1]:.2f}%')

# 결과 비교
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
# 1. 테스트 정확도
axes[0].plot(epochs, history_weak_aug['test_acc'], 'b-o', label='Weak Aug', linewidth=2)
axes[0].plot(epochs, history_medium_aug['test_acc'], 'g-s', label='Medium Aug', linewidth=2)
axes[0].plot(epochs, history_strong_aug['test_acc'], 'r-^', label='Strong Aug', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Test Accuracy (%)', fontsize=12)
axes[0].set_title('Test Accuracy by Augmentation Strength', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 2. 과적합 갭 (Train - Test)
gap_weak = [t - v for t, v in zip(history_weak_aug['train_acc'], history_weak_aug['test_acc'])]
gap_medium = [t - v for t, v in zip(history_medium_aug['train_acc'], history_medium_aug['test_acc'])]
gap_strong = [t - v for t, v in zip(history_strong_aug['train_acc'], history_strong_aug['test_acc'])]

axes[1].plot(epochs, gap_weak, 'b-o', label='Weak Aug', linewidth=2)
axes[1].plot(epochs, gap_medium, 'g-s', label='Medium Aug', linewidth=2)
axes[1].plot(epochs, gap_strong, 'r-^', label='Strong Aug', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Overfitting Gap (%)', fontsize=12)
axes[1].set_title('Overfitting Gap (Train - Test)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

print(f'{"Augmentation":<15} {"Test Acc (%)":<15} {"Overfit Gap (%)":<15}')
print(f'Weak{"":<11} {history_weak_aug["test_acc"][-1]:<15.2f} {gap_weak[-1]:<15.2f}')
print(f'Medium{"":<9} {history_medium_aug["test_acc"][-1]:<15.2f} {gap_medium[-1]:<15.2f}')
print(f'Strong{"":<9} {history_strong_aug["test_acc"][-1]:<15.2f} {gap_strong[-1]:<15.2f}')

# 작은 데이터셋 최적 프로토콜
# Phase 1: 헤드만 학습 (5 epochs)
# Phase 2: 점진적 해제 (5 epochs)

model_2phase = models.resnet18(pretrained=True)
for param in model_2phase.parameters():
    param.requires_grad = False
model_2phase.fc = nn.Linear(model_2phase.fc.in_features, num_classes)
model_2phase = model_2phase.to(device)

# Phase 1: 헤드만 학습
optimizer_phase1 = optim.Adam(model_2phase.fc.parameters(), lr=0.01)
history_2phase = {'train_acc': [], 'test_acc': [], 'phase': []}
phase1_epochs = 5
for epoch in range(phase1_epochs):
    _, train_acc = train_one_epoch(model_2phase, loader_medium,  # Medium 증강 사용
                                   criterion, optimizer_phase1, device)
    _, test_acc = evaluate_model(model_2phase, test_loader, criterion, device)

    history_2phase['train_acc'].append(train_acc)
    history_2phase['test_acc'].append(test_acc)
    history_2phase['phase'].append(1)

    print(f'Phase1 Epoch [{epoch+1}/{phase1_epochs}] - '
          f'Train: {train_acc:.2f}%, Test: {test_acc:.2f}%')

print(f'\nPhase 1 완료 - Test Acc: {history_2phase["test_acc"][-1]:.2f}%')

# Phase 2: 점진적 해제 (Layer4)
for param in model_2phase.layer4.parameters():
    param.requires_grad = True  # Layer4 학습 가능하게
optimizer_phase2 = optim.Adam(
    [{'params': model_2phase.layer4.parameters(), 'lr': 0.0001},  # 백본은 작은 LR
     {'params': model_2phase.fc.parameters(), 'lr': 0.001}],      # 헤드는 중간 LR
)
phase2_epochs = 5

for epoch in range(phase2_epochs):
    _, train_acc = train_one_epoch(model_2phase, loader_medium,
                                   criterion, optimizer_phase2, device)
    _, test_acc = evaluate_model(model_2phase, test_loader, criterion, device)

    history_2phase['train_acc'].append(train_acc)
    history_2phase['test_acc'].append(test_acc)
    history_2phase['phase'].append(2)

    print(f'Phase2 Epoch [{epoch+1}/{phase2_epochs}] - '
          f'Train: {train_acc:.2f}%, Test: {test_acc:.2f}%')

print(f'\nPhase 2 완료 - 최종 Test Acc: {history_2phase["test_acc"][-1]:.2f}%')

# 프로토콜 결과 시각화
# 2단계 학습 과정 시각화
fig, ax = plt.subplots(figsize=(12, 6))

total_epochs = range(1, len(history_2phase['test_acc']) + 1)

# Phase 구분선
phase_boundary = phase1_epochs
ax.axvline(x=phase_boundary, color='gray', linestyle='--', linewidth=2,
           label='Phase Transition', alpha=0.5)

# 학습/테스트 정확도 플롯
ax.plot(total_epochs, history_2phase['train_acc'], 'b-o',
        label='Train Accuracy', linewidth=2, markersize=6)
ax.plot(total_epochs, history_2phase['test_acc'], 'r-s',
        label='Test Accuracy', linewidth=2, markersize=6)

# Phase 영역 표시
ax.axvspan(0, phase_boundary, alpha=0.1, color='blue', label='Phase 1: Head Only')
ax.axvspan(phase_boundary, len(total_epochs), alpha=0.1, color='green',
          label='Phase 2: Head + Layer4')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Two-Phase Training Protocol for Small Dataset',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f'Phase 1 종료 시 정확도: {history_2phase["test_acc"][phase1_epochs-1]:.2f}%')
print(f'Phase 2 종료 시 정확도: {history_2phase["test_acc"][-1]:.2f}%')
print(f'전체 개선: {history_2phase["test_acc"][-1] - history_2phase["test_acc"][0]:.2f}%p')

# 전체 실험 종합 비교
print('\n1. 층별 동결 전략 비교:')
print(f'   - Full Freeze:        {history_freeze["test_acc"][-1]:.2f}%')
print(f'   - Partial Fine-tune:  {history_partial["test_acc"][-1]:.2f}%')
print(f'   - Full Fine-tune:     {history_full["test_acc"][-1]:.2f}%')

print('\n2. 학습률 전략 비교:')
print(f'   - Uniform LR:         {history_uniform_lr["test_acc"][-1]:.2f}%')
print(f'   - Differential LR:    {history_diff_lr["test_acc"][-1]:.2f}%')

print('\n3. 데이터 증강 강도 비교:')
print(f'   - Weak Augmentation:    {history_weak_aug["test_acc"][-1]:.2f}%')
print(f'   - Medium Augmentation:  {history_medium_aug["test_acc"][-1]:.2f}%')
print(f'   - Strong Augmentation:  {history_strong_aug["test_acc"][-1]:.2f}%')

print('\n4. 2단계 프로토콜:')
print(f'   - Two-Phase Training:   {history_2phase["test_acc"][-1]:.2f}%')

# 최종 종합 그래프
fig, ax = plt.subplots(figsize=(14, 7))

strategies = [
    'Full\nFreeze',
    'Partial\nFine-tune',
    'Full\nFine-tune',
    'Uniform\nLR',
    'Differential\nLR',
    'Weak\nAug',
    'Medium\nAug',
    'Strong\nAug',
    '2-Phase\nProtocol'
]

accuracies = [
    history_freeze['test_acc'][-1],
    history_partial['test_acc'][-1],
    history_full['test_acc'][-1],
    history_uniform_lr['test_acc'][-1],
    history_diff_lr['test_acc'][-1],
    history_weak_aug['test_acc'][-1],
    history_medium_aug['test_acc'][-1],
    history_strong_aug['test_acc'][-1],
    history_2phase['test_acc'][-1]
]

# 색상 구분
colors = ['#1f77b4', '#1f77b4', '#1f77b4',  # 실험 1
          '#ff7f0e', '#ff7f0e',              # 실험 2
          '#2ca02c', '#2ca02c', '#2ca02c',   # 실험 3
          '#d62728']                          # 실험 4

bars = ax.bar(strategies, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# 값 표시
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('All Experiments Comparison - Final Test Accuracy',
            fontsize=14, fontweight='bold')
ax.set_ylim([0, max(accuracies) * 1.1])
ax.grid(True, axis='y', alpha=0.3)

# 범례
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', alpha=0.7, label='Exp 1: Freeze Strategy'),
    Patch(facecolor='#ff7f0e', alpha=0.7, label='Exp 2: Learning Rate'),
    Patch(facecolor='#2ca02c', alpha=0.7, label='Exp 3: Augmentation'),
    Patch(facecolor='#d62728', alpha=0.7, label='Exp 4: 2-Phase Protocol')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.xticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.show()


