import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision  # 이미지 처리 도구
import torchvision.transforms as transforms  # 이미지 변환 도구
import torchvision.models as models  # 사전 학습된 모델
import matplotlib.pyplot as plt
import numpy as np  # 숫자 계산
from tqdm import tqdm  # 진행 상태 표시

# 결과를 항상 같게 만들기 (재현성)
torch.manual_seed(42)

# GPU 사용 가능한지 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 처리
train_transform = transforms.Compose([
    # 코드 작성
    transforms.Resize(112), # 크기를 112x112로 조정
    transforms.RandomHorizontalFlip(), # 50% 확률로 좌우 반전
    transforms.ToTensor(), # 이미지를 숫자(텐서)로 변환
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # RGB 평균값으로 정규화
        std=[0.229, 0.224, 0.225]     # RGB 표준편차로 정규화
    )
])

# 테스트용 이미지 변환 (증강 없음)
test_transform = transforms.Compose([
    # 코드 작성
    transforms.Resize(112), # 크기를 112x112로 조정
    transforms.ToTensor(), # 이미지를 숫자(텐서)로 변환
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 훈련 데이터 다운로드 및 준비
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',  # 데이터를 저장할 폴더
    train=True,     # 훈련 데이터 사용
    download=True,  # 없으면 다운로드
    transform=train_transform  # 위에서 정의한 변환 적용
)

# 테스트 데이터 다운로드 및 준비
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,    # 테스트 데이터 사용
    download=True,
    transform=test_transform
)

# 데이터 로더 생성 (배치로 묶어서 제공)
train_loader  = DataLoader(
                train_dataset,
                batch_size = 64,
                shuffle = True,  # 데이터 섞기
                num_workers=2    # 병렬 처리
            )

test_loader  = DataLoader(
               test_dataset,
               batch_size = 64,
               shuffle = False,  # 데이터 섞지 않음
               num_workers=2    # 병렬 처리
            )

classes = ['비행기', '자동차', '새', '고양이', '사슴',
           '개', '개구리', '말', '배', '트럭'] # 클래스 이름 (10가지 물체)

# 함수 정의
def show_sample_images(loader, classes, num_images=8): # 데이터에서 샘플 이미지를 보여주는 함수
    images, labels = next(iter(loader)) # 데이터 가져오기

    # 정규화 해제(다시 원래 이미지로) >> [0,1] 범위로 복원하는 과정
    # 정규화 : x_norm = (x - x_mean) / sigma(std)
    # 정규화 해제: x = x_norm * std + mean
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    # 채널별 평균(mean) 텐서 생성 : [3,1,1] (브로드캐스팅으로 각 픽셀에 채널별 평균을 적용하려고)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    # 채널별 표준편차 텐서 생성 : [3,1,1] (브로드캐스팅으로 각 픽셀에 채널별 표준편차를 적용하려고)
    images = images * std + mean
    # 값 범위 고정(clamp): 시각화/저장 전에 안전하게 [0,1] 구간으로 잘라냄
    # >> 픽셀(pixel) 범위 [0.1] 로 제한
    images = torch.clamp(images, 0, 1)

    # 그리드로 표시
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    # 배치 이미지 텐서를 2×4 격자(grid)로 시각화하는 예시
    axes = axes.ravel()
    # axes는 2×4 배열 형태이므로, 편하게 1차원으로 평탄화
    # print(axes)

    for i in range(num_images):
        # 이미지를 화면에 표시할 수 있는 형태로 변환
        img = images[i].permute(1, 2, 0).numpy()
        # permute(1, 2, 0) : (CHW→HWC) PyTorch 텐서 이미지는 보통 CHW. 축변경 기본(채널,높이,너비)
        axes[i].imshow(img)
        axes[i].set_title(f'{classes[labels[i]]}', fontsize=12)
        axes[i].axis('off')

    plt.show()

# 학습 곡선 그리기
def plot_results(train_losses, train_accs, test_accs): # 학습 결과를 그래프로 그리는 함수
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 손실 그래프
    axes[0].plot(epochs, train_losses, 'b-o', linewidth=2, markersize=8, label='훈련 손실')
    axes[0].set_xlabel('에폭', fontsize=12)
    axes[0].set_ylabel('손실', fontsize=12)
    axes[0].set_title('손실 변화', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 정확도 그래프
    axes[1].plot(epochs, train_accs, 'g-o', linewidth=2, markersize=8, label='훈련 정확도')
    axes[1].plot(epochs, test_accs, 'r-s', linewidth=2, markersize=8, label='테스트 정확도')
    axes[1].set_xlabel('에폭', fontsize=12)
    axes[1].set_ylabel('정확도 (%)', fontsize=12)
    axes[1].set_title('정확도 변화 (클수록 좋음)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 결과 요약
    print('\n학습 결과 요약')
    print('=' * 60)
    print(f'시작 정확도: {test_accs[0]:.2f}%')
    print(f'최종 정확도: {test_accs[-1]:.2f}%')
    print(f'향상도: {test_accs[-1] - test_accs[0]:+.2f}%p')
    print('=' * 60)

    # 해석
    if test_accs[-1] >= 85: # 업계에 따라 기준이 다름(논문 보고서 참조)
        print('훌륭해요! 85% 이상의 정확도를 달성했습니다!')
    elif test_accs[-1] >= 75:
        print('좋아요! 75% 이상의 정확도를 달성했습니다!')
    else:
        print('괜찮아요! 더 많은 에폭으로 학습하면 더 좋아질 거예요!')

# 예측 결과 시각화
def show_predictions(model, loader, classes, device, num_images=12): # 모델의 예측 결과를 시각화하는 함수
    model.eval()

    # 데이터 가져오기
    images, labels = next(iter(loader))
    images = images.to(device)

    # 예측
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # CPU로 이동
    images = images.cpu()
    predicted = predicted.cpu()

    # 정규화 해제
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    # 시각화
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.ravel()

    for i in range(num_images):
        img = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)

        # 정답과 예측 비교
        true_label = classes[labels[i]]
        pred_label = classes[predicted[i]]

        # 맞으면 초록색, 틀리면 빨간색
        if labels[i] == predicted[i]:
            color = 'green'

        else:
            color = 'red'

        axes[i].set_title(
            f'정답: {true_label}\n예측: {pred_label}',
            color=color,
            fontsize=11,
            fontweight='bold'
        )
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # 정확도 계산
    correct = (predicted[:num_images] == labels[:num_images]).sum().item()
    accuracy = 100 * correct / num_images

    print(f'\n이 샘플에서 {num_images}개 중 {correct}개를 맞췄어요!')
    print(f'정확도: {accuracy:.1f}%')

# 클래스별 정확도 계산
def evaluate_per_class(model, loader, classes, device): #클래스별 성능을 평가하는 함수
    model.eval()
    # 클래스별 맞춘 개수와 전체 개수 (초기화)
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    with torch.no_grad(): # 코드 작성
        for images, labels in tqdm(loader, desc='클래스별 평가'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    # 결과 출력 # 코드 작성
    accuracies = []
    for i,class_name in enumerate(classes):
        if class_total[i]>0:
            acc =100 * class_correct[i] / class_total[i]
            accuracies.append(acc)      
            print(f'{class_name:8s} {acc:5.1f}%')

    # 가장 잘 맞추는 것과 어려워하는 것 # 코드 작성
    best_idx = np.argmax(accuracies)
    worst_idx = np.argmin(accuracies)

    print(f'\n가장 잘 맞추는 것: {classes[best_idx]} ({accuracies[best_idx]:.1f}%)')
    print(f'가장 어려워하는 것: {classes[worst_idx]} ({accuracies[worst_idx]:.1f}%)')

# 모델 저장(선택 코딩)


# 모델 호출

model = models.mobilenet_v2(weught='IMAGENET1K_V1')
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features,10)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = optim.SGD(model.parameters(),lr=lr)
num_epochs = 30

# 모델 학습
train_losses = []
train_accs = []
test_accs = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images,labels in tqdm(train_loader,desc='훈련'):
        images = images.to(device)
        labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()

    # loss.item() : 현재 배치 평균 손실 값을 파이썬 숫자로 꺼냄
    running_loss += loss.item() * images.size(0) # size(0) : 배치크기
    # torch.max(outputs, 1) 각 샘플에 대해 가장 큰 로짓값(모델이 예측한 값)과 그 인덱스(예측 클래스) 반환
    _, predicted = torch.max(outputs, 1) # predicted = 각 샘플의 가장 큰 확률값을 가지는 클래스(인덱스)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    # epoch 단위 평균 손실/정확도 계산
    # 에폭 평균 계산
    # running loss >> 배치 손실 + 배치 크기 합 >> total (전체 샘플 수)로 나눠 에폭 평균 손실

epoch_loss = running_loss / total
epoch_accs = 100 * correct / total

train_losses.append(epoch_loss)
train_accs.append(epoch_accs)

model.eval() # 평가 모드
correct = 0
total = 0

with torch.no_grad():
    for images,labels in tqdm(test_loader,desc="테스트"):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    test_accs.append(test_acc)