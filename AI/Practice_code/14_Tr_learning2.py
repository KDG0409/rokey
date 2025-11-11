# 3가지 전이학습 전략을 비교
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import torch  # PyTorch 메인 라이브러리
import torch.nn as nn  # 신경망 모듈
import torch.optim as optim  # 최적화 알고리즘
from torchvision import datasets, transforms, models  # 비전 관련 도구
from torch.utils.data import DataLoader, Subset  # 데이터 로더
import numpy as np  # 수치 연산
import random  # 랜덤 함수

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'사용 디바이스: {device}')
torch.manual_seed(0)  # PyTorch 시드
np.random.seed(0)  # NumPy 시드
random.seed(0)  # Python random 시드

if torch.cuda.is_available():
    torch.cuda.manual_seed(0)  # CUDA 시드
    torch.backends.cudnn.deterministic = True  # 결정적 알고리즘 사용
    torch.backends.cudnn.benchmark = False  # 벤치마크 비활성화

# 데이터 전처리

transform_train = transforms.Compose([
                  transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                  # 이미지의 60-100% 영역을 무작위로 잘라 224*224 이미지로 조정
                  transforms.RandomHorizontalFlip(),

                  transforms.ToTensor(),
                  transforms.Normalize(mean = (0.485, 0.456, 0.406),
                                      std = (0.229, 0.224, 0.225))
              ]
              )
transform_test = transforms.Compose([
                  transforms.Resize(256), # 이미지를 256*256 사이즈로 변형
                  transforms.CenterCrop(224),
                  # 중앙을 224 * 224 이미지로 잘라냄 (증강 없이 고정된 영역 사용)

                  transforms.ToTensor(),
                  transforms.Normalize(mean = (0.485, 0.456, 0.406), # RGB 각 채널의 평균과 표준편차
                                      std = (0.229, 0.224, 0.225))
              ]
              )
dataset_train_full =  datasets.CIFAR10(
                      root = '/tmp/cifar.tl',
                      train = True,
                      download=True,
                      transform = transform_train
                  )

dataset_test       =  datasets.CIFAR10(
                      root = '/tmp/cifar.tl',
                      train = False,
                      download=True,
                      transform = transform_test
                  )
print(f'전체 학습 데이터: {len(dataset_train_full):,}개')
print(f'전체 테스트 데이터: {len(dataset_test):,}개')
print(f'클래스 수: 10개 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)')

# 소규모 데이터셋 사용:전체 50,000개 중 각 클래스 당 300개씩, 총 3,000개만 사용

selected_indices = []
# 각 클래스(0-9) 별로 카운트 저장할 딕셔너리 초기화
class_counts = {i: 0 for i in range(10)} # {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

# 전체 학습 데이터 돌면서 각 클래스 당 300개씩 선택
for idx, (image, label) in enumerate(dataset_train_full):
  # 조건 1: 해당 클래스 현재 개수가 300개 미만이면
  if class_counts[label] < 300:
    selected_indices.append(idx)
    class_counts[label] += 1
  # 조건 2: 중지조건
  if len(selected_indices) >= 3000: # 전체 데이터 수
    break

dataset_train_small = Subset(dataset_train_full, selected_indices)

for class_id, count in class_counts.items():
  class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog','frog','horse','ship','truck'][class_id]
  print(f'{class_id} ({class_name}): {count}개')

train_loader =  DataLoader(
                dataset_train_small,
                batch_size = 64,
                shuffle=True,
                # 한번 학습할 때 마다(매 epoch) 무작위 데이터 순서를 섞어줌 (과적합 방지)
                num_workers=2,
                pin_memory=True
            )

test_loader =   DataLoader(
                dataset_test,
                batch_size = 128,
                # 일반적으로 평가할 때 조금 더 큰 배치 사용 가능
                shuffle=False,
                # 평가시 순서 섞지 않음
                num_workers=2,
                pin_memory=True
            )

# 모델 정의 : 3가지 전략 (freeze, partial, full)

def build_model(stratgy = 'freeze'):
   model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
   # 사전 학습(pre-trained)된 모델의 가중치(weights) 사용

   # 원래 resnet18의 분류기 입력 특징 차원 저장
   in_features = model.fc.in_features # 512차원

   # 분류기를 CIFAR 10 에 맞게 교체 (1000 클래스 >> 10 클래스)
   model.fc = nn.Linear(in_features, 10)

   # 전략에 따라 파라미터 동결 설정
   if stratgy == 'freeze':
     # 전략 1: 백본 전체를 동결 >> 분류기만 학습시키는 전략
      print('[freeze 전략]: 백본 전체를 동결 >> 분류기만 학습')

      # layer1,2,3,4 모두 동결 : 기울기(gradient) 계산 비활성화
      for param in model.layer1.parameters():
        param.requires_grad = False 
      for param in model.layer2.parameters():
        param.requires_grad = False 
      for param in model.layer3.parameters():
        param.requires_grad = False 
      for param in model.layer4.parameters():
        param.requires_grad = False 

      # fc(분류기: classifier) : 새로 생성했기때문에 자동으로 param.requires_grad = True
   elif stratgy == 'partial':
      # 전략 2: 마지막 블록(layer4)과 분류기만 학습
      for param in model.parameters():
        param.requires_grad = False 
      for param in model.layer4.parameters():
        param.requires_grad = True
      for param in model.fc.parameters():
        param.requires_grad = True
     
   elif stratgy == 'full':
      # 전략 3: 모든 층을 학습
       print('[full 전략]: 모든 층을 학습')

       for param in model.parameters():
         param.requires_grad = True

   else:
    raise ValueError('지원되지 않는 전략입니다.')

   model = model.to(device)

   # 학습 가능한 파라미터 수 계산
   trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   total_params = sum(p.numel() for p in model.parameters())
   print(f'학습 가능한 파라미터 : {trainable_params:,} / {total_params:,}')

   return model

# 학습 및 평가 함수 정의
def train_and_evaluate(strategy):
  model = build_model(strategy)

  # 헤드(분류기)와 백본의 파라미터 분리
  head_params = list(model.fc.parameters()) # 분류기 파라미터

  # 백본 파라미터 (fc 아니면서 학습가능한 파라미터)
  backbone_params = [
      param for name, param in model.named_parameters()
      if 'fc' not in name and param.requires_grad
  ]
  # 파라미터 그룹 구성(차등 학습률)
  param_groups = []

  # 백본 파라미터가 있다면
  if backbone_params:
    param_groups.append({
        'params': backbone_params,
        'lr': 1e-4 # 백본은 낮은 학습률(0.0001)
    }
    )

  # 헤드(분류기) 파라미터가 있다면
  if head_params:
    param_groups.append({
        'params': head_params,
        'lr': 1e-3 # 헤드(분류기) 높은 학습률 (0.001, 백본의 10배)
    })

  # AdamW 옵티마이저 사용(가중치 감쇠 포함)
  optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
  criterion = nn.CrossEntropyLoss()
  num_epoch = 10

  for epoch in range(num_epoch):
    model.train()

    running_loss = 0.0 # 에폭별 손실 누적
    correct = 0 # 맞춘 개수
    total = 0   # 전체 샘플 수

    # 배치 별 학습
    for batch_idx, (inputs, labels) in enumerate(train_loader):
       inputs = inputs.to(device)
       labels = labels.to(device)

    # gradient 초기화
    optimizer.zero_grad()

    # 순전파
    outputs = model(inputs)

    # 손실 계산
    loss = criterion(outputs, labels)

    # 역전파
    loss.backward()

    # 가중치 업데이트
    optimizer.step()

    # 통계 업데이트
    running_loss += loss.item() * inputs.size(0)
    _, predicted = outputs.max(1)

    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()

  # 에폭별 결과 출력
  epoch_loss = running_loss / total
  epoch_acc = 100.0 * correct / total
  print(f'Epoch [({epoch+1}/{num_epoch}]'
        f'Loss: {epoch_loss:.4f}, '
        f'Train_Acc: {epoch_acc:.2f}%')

  model.eval() # 평가 모드로 전환

  correct = 0
  total = 0

  # 그래디언트 계산 비활성화(평가시에는 불필요)
  with torch.no_grad():
    for inputs, labels in test_loader:
          inputs = inputs.to(device)
          labels = labels.to(device)

          # 순전파
          outputs = model(inputs)

            # 예측값 계산
          predicted = outputs.argmax(dim=1)

            # 정확도 계산
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

    # 최종 테스트 정확도 계산
  test_accuracy = correct / total

  print(f'테스트 정확도: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')

  return test_accuracy

# 결과 해석
acc_freeze = train_and_evaluate('freeze')
acc_partial = train_and_evaluate('partial')
acc_full = train_and_evaluate('full')

results = {
    'Freeze (백본 전체 동결)': acc_freeze,
    'Partial (layer4, fc 해제)': acc_partial,
    'Full(모든 층 학습)': acc_full
}
for strategy_name, accuracy in results.items():
    print(f'{strategy_name:<30} {accuracy*100:>14.2f}%')

best_strategy = max(results, key=results.get)
print(best_strategy)
best_accuracy = results[best_strategy]
print(best_accuracy)

# Matplotlib 라이브러리 임포트
import matplotlib.pyplot as plt

# 전략 이름과 정확도
strategies = ['Freeze\n(백본 동결)', 'Partial\n(layer4 해제)', 'Full\n(전체 학습)']
accuracies = [acc_freeze * 100, acc_partial * 100, acc_full * 100]

# 막대 그래프 생성
plt.figure(figsize=(10, 6))
bars = plt.bar(strategies, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'],
               alpha=0.8, edgecolor='black', linewidth=1.5)

# 각 막대 위에 정확도 값 표시
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# 그래프 설정
plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Transfer Learning Strategy Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, max(accuracies) * 1.15])  # Y축 범위 설정
plt.grid(True, axis='y', alpha=0.3, linestyle='--')

# 그래프 표시
plt.tight_layout()
plt.show()
