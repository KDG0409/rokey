# 신용카드 사기 거래 탐지 (초중급)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)
import torch
import torch.nn as nn
import torch.optim as optim

# 시각화 설정
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True

# 불균형 데이터 생성

X, y = make_classification(
          n_samples=10000,  # 전체 샘플 개수(행의 수)
          n_features=20,    # 전체 특성(열) 개수
          n_informative=15, # 실제 정답에 기여하는 유의미한 특성
          n_redundant=5,    # 중복특성 >> 제거 대상
          n_classes=2,      # 이진 분류
          weights=[0.95, 0.05], # 클래스 비율(불균형 설정) : 0번(정상) 95% 1번(사기) 5%
          flip_y=0.01,      # 노이즈 (라벨을 무작위로 뒤집는 비율)
          random_state=42
      )

# 클래스 분포 확인
np.unique(y, return_counts=True)
# unique() 중복제거 >> 배열 y에서 중복 제거 >> 고유값 추출
# y = [0,0,1,1,1] >> [0,1]
# return_counts=True
# 각 고유값이 몇 번 등장했냐
unique, counts = np.unique(y, return_counts=True)
print(unique,counts)
print("정상거래(0): ", counts[0])
print("사기거래(1): ", counts[1])

X_train, X_test, y_train, y_test =\
train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)
print(X_train.shape, X_test.shape)

print(f'훈련 데이터 사기 비율: {y_train.sum()/len(y_train)*100: .2f}%')
print(f'평가용(테스트) 데이터 사기 비율: {y_test.sum()/len(y_test)*100: .2f}%')

# 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # 훈련용 데이터
X_test_scaled = scaler.transform(X_test) # 테스트용 데이터
# why? transform() 만 적용한 이유? 테스트 데이터는 새로운 데이터로 간주

print(f'원본 데이터 범위:[{X_train.min():.2f}, {X_train.max():.2f}]')
print(f'표준화된 범위:[{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]')
print(f'표준화된 평균:[{X_train_scaled.mean():.2f}]')
print(f'표준화된 표준편차:[{X_train_scaled.std():.2f}]')
# 표준화 정의 (평균=0, 분산=1)
# >> 표준편차 : 편차란 평균에서 단위 당(unit) 떨어진 정도

# 텐서 변환
X_train_tensor = torch.FloatTensor(X_train_scaled) # (샘플수, 특성수)
y_train_tensor = torch.FloatTensor(y_train).view(-1,1)

X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).view(-1,1)

print(X_train_tensor.shape, y_train_tensor.shape)
print(X_test_tensor.shape, y_test_tensor.shape)

# 3개의 층을 가진 신경망 모델 생성
# self.x 는 학습 대상을 저장하는데 사용함 (인스턴스 변수 저장)

class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = FraudDetectionModel(input_dim = 20)

# pos_weight 사용, 클래스 불균형 조정
pos_weight = torch.tensor([counts[0]/counts[1]]) # array([9461,  539])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # 이진분류 기준
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습

num_epochs = 100
history = {'train_loss': [],'test_loss': [],'test_acc': [] }

for epoch in range(num_epochs):
  model.train()
  optimizer.zero_grad()
  outputs = model(X_train_tensor)  
  loss = criterion(outputs, y_train_tensor)
  loss.backward()
  optimizer.step()

  model.eval()
  with torch.no_grad():  
      test_outputs = model(X_test_tensor)
      test_loss = criterion(test_outputs, y_test_tensor)
      test_pred = (test_outputs >= 0.0).float()
      test_acc = (test_pred == y_test_tensor).float().mean()

  history['train_loss'].append(loss.item())
  history['test_loss'].append(test_loss.item())
  history['test_acc'].append(test_acc.item())

  if (epoch + 1) % 20 == 0:
    print(f'Epoch [{epoch+1} / {num_epochs}] '
          f'Train_loss: {loss.item():.4f} | '
          f'Test_loss: {test_loss.item():.4f} | '
          f'Test_acc: {test_acc.item():.4f} | '
    )

# 학습 곡선 시각화

fig, axes = plt.subplots(1,2, figsize=(14, 5))

axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['test_loss'], label='Test Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Learning Curve - Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history['test_acc'], label='Test Accuracy', color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Learning Curve - Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# 상세 평가 지표 계산

model.eval()
with torch.no_grad():
  test_outputs = model(X_test_tensor)
  test_probs = torch.sigmoid(test_outputs)
  test_pred (test_outputs >=0.0).numpy().astype(int).flatten()
y_test_np = y_test.astype(int)

cm = confusion_matrix(y_test_np, test_pred) # 혼동행렬 (분류문제)

fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(cm, cmap="Blu")
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(['Normal', 'Fraud'])
ax.set_yticklabels(['Normal', 'Fraud'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title("Confusion Matrix")

for i in range(2):
  for j in range(2):
    ax.text(j, i, cm[i,j],
            ha = 'center',
            va ='center',
            color='white' if cm[i,j] >  cm.max()/2 else "black",
            fontsize = 20, fontweight = 'bold'
            )
plt.colorbar(im,ax=ax)
plt.tight_layout()
plt.show()

print('='*50)
print('Confusion Matrix')
print('='*50)
print("                   Predicted")
print("                   0(정상)   1(사기)")
print(f"Actual  0(정상)     {cm[0,0]}   {cm[0,1]}")
print(f"        0(사기)     {cm[1,0]}   {cm[1,1]}")
print()
print('='*50)

tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = precision_score(y_test_np, test_pred)
recall = recall_score(y_test_np, test_pred)
f1 = f1_score(y_test_np, test_pred)

print("="*50)
print("주요 평가지표")
print(f"Accuracy(정확도): {accuracy:.4f}")
print(f"Precision(정밀도): {precision:.4f}")
print(f"Recall(재현율/민감도): {recall:.4f}")
print(f"Accuracy(정확도): {accuracy:.4f}")

fpr , tpr , threholds = roc_curve(y_test_np, test_probs.numpy()) # 넘파이 사용
auc = roc_auc_score(y_test_np, test_probs.numpy())
print(auc) # 아래 면적 크기(=비율) /전체크기가 1임

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve(AUC = {auc:.4f})', linewidth=2)
plt.plot([0,1],[0,1],'k--', label="Random")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()