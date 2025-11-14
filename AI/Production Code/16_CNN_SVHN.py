# CNN 모델 빌드 실습 (SVHN)

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

import torch  # PyTorch 메인 라이브러리
import torch.nn as nn  # 신경망 모듈 (레이어, 손실함수 등)
import torch.optim as optim  # 최적화 알고리즘 (Adam, SGD 등)
import torch.nn.functional as F  # 활성화 함수, Softmax 등
from torch.utils.data import DataLoader  # 데이터 로더 (배치 처리)

import torchvision  # 비전 관련 전체 모듈
import torchvision.transforms as transforms  # 이미지 전처리 및 증강
from torchvision import datasets  # SVHN, CIFAR 등 표준 데이터셋

import numpy as np  # 수치 연산 라이브러리
import matplotlib.pyplot as plt  # 그래프 및 이미지 시각화
from collections import OrderedDict  # 순서가 보장되는 딕셔너리
from tqdm import tqdm  # 진행률 표시줄 (progress bar)

# 기본 설정

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)  # PyTorch 난수 시드 고정
np.random.seed(42)  # NumPy 난수 시드 고정
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)  # CUDA 난수 시드 고정
    torch.backends.cudnn.deterministic = True  # 결정적 알고리즘 사용
    torch.backends.cudnn.benchmark = False  # 벤치마크 비활성화 (재현성 우선)

# 데이터 전처리
transform_train = transforms.Compose([ 
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
transform_test = transforms.Compose([ 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dataset = datasets.SVHN(
    root='./data',  # 데이터를 저장할 디렉토리 경로
    split='train',  # 'train' 또는 'test' 선택
    download=True,  # 데이터가 없으면 자동 다운로드
    transform=transform_train  # 학습용 전처리 적용
)
test_dataset = datasets.SVHN(
    root='./data',  # 동일한 디렉토리 사용
    split='test',  # 테스트 데이터 사용
    download=True,  # 없으면 다운로드
    transform=transform_test  # 테스트용 전처리 적용
)
batch_size = 128
train_loader = DataLoader(
    train_dataset,  
    batch_size=batch_size,  
    shuffle=True,  
    num_workers=2,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,  # 테스트 데이터셋
    batch_size=batch_size,  # 배치 크기 (평가 시에는 더 크게 해도 됨)
    shuffle=False,  # 테스트 시에는 순서를 섞지 않음
    num_workers=2,
    pin_memory=True
)

# 샘플 이미지 시각화
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

dataiter = iter(train_loader)
images,labels = next(iter(train_loader))
def denormalize(tensor): # 역정규화
    return (tensor * 0.5) + 0.5

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)  # 4x4 그리드의 i+1번째 위치
    img = denormalize(images[i]).cpu().numpy().transpose(1, 2, 0) # 이미지를 (C, H, W) -> (H, W, C) 형태로 변환
    plt.imshow(img) # 이미지 표시
    plt.title(f'Label: {class_names[labels[i]]}')  # 레이블 표시
    plt.axis('off')  # 축 숨기기

plt.tight_layout()  # 레이아웃 자동 조정
plt.suptitle('SVHN 학습 데이터 샘플', y=1.02, fontsize=16)
plt.show()

# CNN 모델 정의
class SVHN_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SVHN_CNN, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3,32,3,1) #침력/출력/커널/패딩 b,3,32,32
        self.pool1 = nn.MaxPool2d((2,2)) #커널,스트라이드
        self.conv2 = nn.Conv2d(32,64,3,1) #침력/출력/커널/패딩 b,32,16,16
        self.pool2 = nn.MaxPool2d((2,2))
        self.conv3 = nn.Conv2d(64,128,3,1) #침력/출력/커널/패딩 b,64,8,8
        self.pool3 = nn.MaxPool2d((2,2)) # b,128,4,4
        self.flatten = nn.Flatten() #b,128*4*4 -> b,2048
        self.fc1 = nn.Linear(2048,512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512,10)

        self.features = nn.Sequential(
            self.conv1,self.relu,self.pool1,self.conv2,self.relu,self.pool2,self.conv3,self.relu,self.pool3
        )
        self.classifier = nn.Sequential(
            self.flatten,self.fc1,self.relu,self.dropout,self.fc2
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x
model = SVHN_CNN(num_classes=10)  # 10개 클래스 (0-9)
model = model.to(device)  # 모델을 GPU 또는 CPU로 이동

# 파라미터 계산
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total

def print_parameter_details(model):
    conv1_params = (3 * 3 * 3 + 1) * 32 # (커널크기**2*입력채널수+편향)*출력채널수
    conv2_params = (3 * 3 * 32 + 1) * 64
    conv3_params = (3 * 3 * 64 + 1) * 128
    fc1_params = (2048 + 1) * 512 # (in_features + 1) * out_features
    fc2_params = (512 + 1) * 10
    total = conv1_params + conv2_params + conv3_params + fc1_params + fc2_params
    return total

total_params = count_parameters(model)
calculated_total = print_parameter_details(model)

# Hook 이용 레이어 출력
# Hook: 모델의 중간 레이어 출력을 가로채는 메커니즘

layer_outputs = OrderedDict()
def register_hooks(model):
    handles = []

    def hook_fn(module, input, output):
        layer_name = module.__class__.__name__
        count = sum(1 for k in layer_outputs.keys() if layer_name in k)
        if count > 0:
            layer_name = f"{layer_name}_{count+1}"
        layer_outputs[layer_name] = output.shape

    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and module != model: # 전체 모델 자체는 제외 (하위 레이어만 등록)
            handle = module.register_forward_hook(hook_fn) # register_forward_hook: forward pass 시 hook_fn 실행
            handles.append(handle)
    return handles

hook_handles = register_hooks(model)
dummy_input = torch.randn(1, 3, 32, 32).to(device)
with torch.no_grad():  # 그래디언트 계산 비활성화 (메모리 절약)
    output = model(dummy_input)

for layer_name, shape in layer_outputs.items():
    print(f'{layer_name:<25} {str(tuple(shape)):>30}')

for handle in hook_handles:
    handle.remove()  # hook 등록 해제

# 학습 함수 정의
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0 
    correct = 0 
    total = 0
    pbar = tqdm(train_loader, desc='학습 중', leave=False)

    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input.size(0)
        val,pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.ez(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# 평가 함수 정의

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='평가 중', leave=False)
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# 학습 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),  # 최적화할 파라미터
    lr=0.001,  # 학습률 (learning rate)
    weight_decay=1e-4  # L2 정규화 계수 (과적합 방지)
)
num_epochs = 10
train_losses = []  # 에폭별 학습 손실
train_accs = []  # 에폭별 학습 정확도
test_losses = []  # 에폭별 테스트 손실
test_accs = []  # 에폭별 테스트 정확도
best_acc = 0.0

for epoch in range(num_epochs):
    print(f'\nEpoch [{epoch+1}/{num_epochs}]')

    # 1. 학습 단계
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    # 2. 평가 단계
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    # 3. 결과 저장
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    # 4. 에폭 결과 출력
    print(f'\n학습 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
    print(f'테스트 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')

    if test_acc > best_acc: # 5. 최고 성능 모델 저장
        best_acc = test_acc
        # 모델 가중치 저장
        torch.save(model.state_dict(), 'best_svhn_model.pth')
        # state_dict(): 모델의 모든 파라미터를 딕셔너리로 반환
        print(f'  → 최고 성능 모델 저장! (정확도: {best_acc:.2f}%)')

# 학습 곡선/예측 결과 시각화
epochs_range = range(1, num_epochs + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 왼쪽 그래프: 손실 곡선

ax1.plot(epochs_range, train_losses, 'b-', label='학습 손실', marker='o')
ax1.plot(epochs_range, test_losses, 'r-', label='테스트 손실', marker='s')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('손실 곡선 (Loss Curve)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 오른쪽 그래프: 정확도 곡선

ax2.plot(epochs_range, train_accs, 'b-', label='학습 정확도', marker='o')
ax2.plot(epochs_range, test_accs, 'r-', label='테스트 정확도', marker='s')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('정확도 곡선 (Accuracy Curve)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

model.load_state_dict(torch.load('best_svhn_model.pth'))
model.eval()
dataiter = iter(test_loader) # 테스트 데이터에서 배치 하나 가져오기
images, labels = next(dataiter)

images = images.to(device) # 모델로 예측
with torch.no_grad():
    outputs = model(images)
    _, predicted = outputs.max(1)

images = images.cpu() # CPU로 이동 및 역정규화
predicted = predicted.cpu()
labels = labels.cpu()

plt.figure(figsize=(12, 12)) # 16개 샘플 시각화
for i in range(16):
    plt.subplot(4, 4, i+1)

    # 이미지 역정규화 및 차원 변환
    img = denormalize(images[i]).numpy().transpose(1, 2, 0)
    plt.imshow(img)

    # 정답과 예측 표시
    true_label = class_names[labels[i]]
    pred_label = class_names[predicted[i]]

    # 정답이면 파란색, 오답이면 빨간색
    color = 'blue' if true_label == pred_label else 'red'
    plt.title(f'실제: {true_label} / 예측: {pred_label}', color=color, fontsize=10)
    plt.axis('off')

plt.suptitle('테스트 데이터 예측 결과 (파란색: 정답, 빨간색: 오답)',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# 정확도 계산
correct = (predicted == labels).sum().item()
total = labels.size(0)
print(f'\n현재 배치 정확도: {100.*correct/total:.2f}% ({correct}/{total})')

# 혼동 랭렬
from sklearn.metrics import confusion_matrix
import seaborn as sns

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='혼동 행렬 계산 중'):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

# 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('예측 레이블', fontsize=12)
plt.ylabel('실제 레이블', fontsize=12)
plt.title('혼동 행렬 (Confusion Matrix)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 클래스별 정확도 계산
for i in range(10): # 해당 클래스의 정확도 계산
    class_correct = cm[i, i]  # 대각선 원소 (정답)
    class_total = cm[i].sum()  # 해당 행의 합 (전체)
    class_acc = 100.0 * class_correct / class_total if class_total > 0 else 0
    print(f'{class_names[i]:<10} {class_acc:>14.2f}% {class_total:>15}')

# 요약 정리
print('\n')
print('='*70)
print('CNN 모델 빌드 실습 요약')
print('='*70)

print('\n1. 데이터셋: SVHN (Street View House Numbers)')
print(f'   - 학습 데이터: {len(train_dataset):,}개')
print(f'   - 테스트 데이터: {len(test_dataset):,}개')
print(f'   - 클래스 수: 10개 (0-9 숫자)')

print('\n2. 모델 아키텍처:')
print('   Feature Extractor:')
print('     - Conv1(32) + ReLU + MaxPool')
print('     - Conv2(64) + ReLU + MaxPool')
print('     - Conv3(128) + ReLU + MaxPool')
print('   Classifier:')
print('     - Flatten')
print('     - FC1(2048->512) + ReLU + Dropout(0.5)')
print('     - FC2(512->10)')

print(f'\n3. 총 파라미터 수: {total_params:,}개')

print('\n4. 학습 설정:')
print(f'   - 에폭: {num_epochs}')
print(f'   - 배치 크기: {batch_size}')
print(f'   - 옵티마이저: Adam (lr=0.001)')
print(f'   - 손실 함수: CrossEntropyLoss')

print('\n5. 최종 성능:')
print(f'   - 최고 테스트 정확도: {best_acc:.2f}%')
print(f'   - 최종 테스트 정확도: {test_accs[-1]:.2f}%')
print(f'   - 최종 테스트 손실: {test_losses[-1]:.4f}')

print('\n6. 주요 학습 내용:')
print('   ✓ CNN 아키텍처 설계 (Feature Extractor + Classifier 분리)')
print('   ✓ 파라미터 수 계산 및 모델 크기 파악')
print('   ✓ Hook을 사용한 레이어 출력 shape 추적')
print('   ✓ 데이터 증강을 통한 성능 향상')
print('   ✓ 학습 곡선 및 혼동 행렬 분석')