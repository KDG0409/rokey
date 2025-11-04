import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# BatchNorm

# 재현성을 위한 시드 고정
torch.manual_seed(42)
np.random.seed(42)

# 실습 5-1: Batch Size의 영향 관찰 : "BatchNorm 없는 기본 신경망
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# 데이터셋 준비
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)

print(f"훈련 데이터: {X_train.shape}")

# 다양한 Batch Size로 학습
batch_sizes = [8, 32, 128, 512]
criterion = nn.BCELoss()

batch_size_results = []

for batch_size in batch_sizes:
    # DataLoader 생성
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 모델 초기화
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 학습
    losses = []
    num_epochs = 30

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            model.train()
            optimizer.zero_grad()

            output = model(batch_X)
            loss = criterion(output, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

    batch_size_results.append((batch_size, losses))
    print(f"  최종 손실: {losses[-1]:.4f}")

# 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for batch_size, losses in batch_size_results:
    plt.plot(losses, label=f'Batch={batch_size}', linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Effect of Batch Size', fontsize=14, weight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
final_losses = [losses[-1] for _, losses in batch_size_results]
colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
bars = plt.bar([str(bs) for bs in batch_sizes], final_losses,
               color=colors, edgecolor='black', alpha=0.7)

plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Final Loss', fontsize=12)
plt.title('Final Loss by Batch Size', fontsize=14, weight='bold')
plt.grid(axis='y', alpha=0.3)

# 값 표시
for bar, loss in zip(bars, final_losses):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{loss:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('part5_batch_size_effect.png', dpi=150, bbox_inches='tight')
print("\n저장: part5_batch_size_effect.png")
plt.close()

# 실습 5-2: BatchNorm 없는 네트워크 vs BatchNorm 있는 네트워크
class NetWithoutBN(nn.Module): #BatchNorm 없는 깊은 신경망
    def __init__(self):
        super(NetWithoutBN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class NetWithBN(nn.Module): # BatchNorm 있는 깊은 신경망
    def __init__(self):
        super(NetWithBN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.BatchNorm1d(64),  # BatchNorm 추가
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

print("\nBatchNorm 원리:")
print("  1. 각 배치의 평균과 분산 계산")
print("  2. 정규화: (x - 평균) / sqrt(분산 + epsilon)")
print("  3. 스케일 및 이동: gamma * 정규화값 + beta")
print("  4. gamma, beta는 학습 가능한 파라미터")

# 두 모델 학습
models_to_compare = [
    ('Without BatchNorm', NetWithoutBN()),
    ('With BatchNorm', NetWithBN())
]

comparison_results = []

# DataLoader 생성 (Batch Size 32)
dataset = TensorDataset(X_train_t, y_train_t)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for model_name, model in models_to_compare:
    print(f"\n{model_name} 학습 중...")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []
    num_epochs = 50

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            model.train()
            optimizer.zero_grad()

            output = model(batch_X)
            loss = criterion(output, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    comparison_results.append((model_name, losses))

# 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for model_name, losses in comparison_results:
    plt.plot(losses, label=model_name, linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('BatchNorm Effect on Training', fontsize=14, weight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
for model_name, losses in comparison_results:
    plt.plot(losses, label=model_name, linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss (log scale)', fontsize=12)
plt.title('BatchNorm Effect (Log Scale)', fontsize=14, weight='bold')
plt.yscale('log')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('part5_batchnorm_effect.png', dpi=150, bbox_inches='tight')
print("\n저장: part5_batchnorm_effect.png")
plt.close()

# 실습 5-3: BatchNorm의 Gradient 안정화 효과

# 그래디언트 수집
gradient_history_without_bn = []
gradient_history_with_bn = []

# 새 모델 생성
model_without_bn = NetWithoutBN()
model_with_bn = NetWithBN()

# 그래디언트 기록 Hook
def create_gradient_hook(gradient_list): # 그래디언트를 리스트에 저장하는 Hook 생성
    def hook(grad):
        gradient_list.append(grad.abs().mean().item())
        return grad
    return hook

# Without BN
for name, param in model_without_bn.named_parameters():
    if 'weight' in name:
        param.register_hook(create_gradient_hook(gradient_history_without_bn))

# With BN
for name, param in model_with_bn.named_parameters():
    if 'weight' in name:
        param.register_hook(create_gradient_hook(gradient_history_with_bn))

# 한 번의 Forward/Backward
batch_X = X_train_t[:32]  # 배치 크기 32
batch_y = y_train_t[:32]

# Without BN
output_without = model_without_bn(batch_X)
loss_without = criterion(output_without, batch_y)
loss_without.backward()

# With BN
output_with = model_with_bn(batch_X)
loss_with = criterion(output_with, batch_y)
loss_with.backward()

# 그래디언트 역순 정렬 (Layer 1부터 표시)
gradient_history_without_bn.reverse()
gradient_history_with_bn.reverse()

print("\n각 층의 그래디언트 크기:")
print("\nWithout BatchNorm:")
for i, grad in enumerate(gradient_history_without_bn):
    print(f"  Layer {i+1}: {grad:.6f}")

print("\nWith BatchNorm:")
for i, grad in enumerate(gradient_history_with_bn):
    print(f"  Layer {i+1}: {grad:.6f}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Without BN
ax1 = axes[0]
layers = list(range(1, len(gradient_history_without_bn) + 1))
ax1.bar(layers, gradient_history_without_bn, color='#e74c3c',
        edgecolor='black', alpha=0.7)
ax1.set_xlabel('Layer Number', fontsize=11)
ax1.set_ylabel('Gradient Magnitude', fontsize=11)
ax1.set_title('Without BatchNorm', fontsize=12, weight='bold')
ax1.set_xticks(layers)
ax1.grid(axis='y', alpha=0.3)

# With BN
ax2 = axes[1]
layers = list(range(1, len(gradient_history_with_bn) + 1))
ax2.bar(layers, gradient_history_with_bn, color='#2ecc71',
        edgecolor='black', alpha=0.7)
ax2.set_xlabel('Layer Number', fontsize=11)
ax2.set_ylabel('Gradient Magnitude', fontsize=11)
ax2.set_title('With BatchNorm', fontsize=12, weight='bold')
ax2.set_xticks(layers)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('part5_gradient_stability.png', dpi=150, bbox_inches='tight')
print("\n저장: part5_gradient_stability.png")
plt.close()

print("\n분석:")
print("  - BatchNorm 없음: 층마다 그래디언트 크기 차이가 큼")
print("  - BatchNorm 있음: 모든 층에서 그래디언트가 비슷한 크기")
print("  - 결과: 더 안정적인 학습 가능")

# 실습 5-4: 다양한 Batch Size에서 BatchNorm 효과

batch_sizes_test = [8, 16, 32, 64]
results_comparison = {}

for batch_size in batch_sizes_test:
    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Without BN
    model_no_bn = NetWithoutBN()
    optimizer_no_bn = optim.Adam(model_no_bn.parameters(), lr=0.001)

    losses_no_bn = []
    for epoch in range(30):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            model_no_bn.train()
            optimizer_no_bn.zero_grad()
            output = model_no_bn(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer_no_bn.step()
            epoch_loss += loss.item()
        losses_no_bn.append(epoch_loss / len(dataloader))

    # With BN
    model_with_bn = NetWithBN()
    optimizer_with_bn = optim.Adam(model_with_bn.parameters(), lr=0.001)

    losses_with_bn = []
    for epoch in range(30):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            model_with_bn.train()
            optimizer_with_bn.zero_grad()
            output = model_with_bn(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer_with_bn.step()
            epoch_loss += loss.item()
        losses_with_bn.append(epoch_loss / len(dataloader))

    results_comparison[batch_size] = {
        'without_bn': losses_no_bn,
        'with_bn': losses_with_bn
    }

    print(f"  Without BN 최종 손실: {losses_no_bn[-1]:.4f}")
    print(f"  With BN 최종 손실: {losses_with_bn[-1]:.4f}")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, batch_size in enumerate(batch_sizes_test):
    ax = axes[idx // 2, idx % 2]

    losses_no_bn = results_comparison[batch_size]['without_bn']
    losses_with_bn = results_comparison[batch_size]['with_bn']

    ax.plot(losses_no_bn, label='Without BN', linewidth=2, color='#e74c3c')
    ax.plot(losses_with_bn, label='With BN', linewidth=2, color='#2ecc71')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title(f'Batch Size = {batch_size}', fontsize=12, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('part5_batchnorm_various_batch_sizes.png', dpi=150, bbox_inches='tight')
print("\n저장: part5_batchnorm_various_batch_sizes.png")
plt.close()

# 실습 5-5: BatchNorm 학습 모드 vs 평가 모드

# BatchNorm이 있는 모델 생성 및 학습
model_bn = NetWithBN()
optimizer = optim.Adam(model_bn.parameters(), lr=0.001)

dataset = TensorDataset(X_train_t, y_train_t)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(30):
    for batch_X, batch_y in dataloader:
        model_bn.train()  # 학습 모드
        optimizer.zero_grad()
        output = model_bn(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# 테스트 데이터
X_test_t = torch.FloatTensor(X_test)

print("학습 모드 vs 평가 모드 출력 비교:")

# 학습 모드에서 여러 번 예측
model_bn.train()
predictions_train_mode = []
for _ in range(5):
    with torch.no_grad():
        pred = model_bn(X_test_t[:10])
        predictions_train_mode.append(pred.numpy())

# 평가 모드에서 여러 번 예측
model_bn.eval()
predictions_eval_mode = []
for _ in range(5):
    with torch.no_grad():
        pred = model_bn(X_test_t[:10])
        predictions_eval_mode.append(pred.numpy())

# 예측 분산 계산
variance_train = np.var(predictions_train_mode, axis=0).mean()
variance_eval = np.var(predictions_eval_mode, axis=0).mean()

print(f"학습 모드 예측 분산: {variance_train:.8f}")
print(f"평가 모드 예측 분산: {variance_eval:.8f}")

print("설명:")
print("  - 학습 모드: 각 배치의 통계 사용 -> 예측이 약간 다름")
print("  - 평가 모드: 전체 데이터의 Moving Average 사용 -> 일관된 예측")
print("  - 중요: model.eval()을 반드시 사용해야 함!")
