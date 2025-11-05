# Part 1: 회귀 손실함수 (MSE, MAE, Huber)
# Part 2: 분류 손실함수 (BCE, CrossEntropy)
# Part 3: 라벨 스무딩
# Part 4: 클래스 불균형 대응
# Part 5: 손실 곡면

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

# 재현성
torch.manual_seed(42)
np.random.seed(42)

# 1. 회귀 손실함수

y_true = torch.tensor([10.0, 20.0, 30.0, 40.0])
y_pred = torch.tensor([12.0, 19.0, 35.0, 38.0])

mse = nn.MSELoss()(y_pred, y_true)
mae = nn.L1Loss()(y_pred, y_true)
huber = nn.HuberLoss()(y_pred, y_true)

print(f"\nMSE:   {mse.item():.4f} - 큰 오차에 민감")
print(f"MAE:   {mae.item():.4f} - 모든 오차 동등")
print(f"Huber: {huber.item():.4f} - MSE와 MAE의 절충")

y_pred_outlier = torch.tensor([12.0, 19.0, 100.0, 38.0])  # 하나의 큰 오차

mse_out = nn.MSELoss()(y_pred_outlier, y_true)
mae_out = nn.L1Loss()(y_pred_outlier, y_true)

print(f"\n이상치 포함 시:")
print(f"MSE: {mse.item():.4f} -> {mse_out.item():.4f} ({mse_out/mse:.1f}배 증가)")
print(f"MAE: {mae.item():.4f} -> {mae_out.item():.4f} ({mae_out/mae:.1f}배 증가)")

# 2. 분류 손실 함수

y_true_bin = torch.tensor([1.0, 0.0, 1.0, 0.0])
y_pred_conf = torch.tensor([0.9, 0.1, 0.85, 0.15])  # 확신있는 예측
y_pred_unce = torch.tensor([0.6, 0.4, 0.55, 0.45])  # 불확실한 예측

bce = nn.BCELoss()
loss_conf = bce(y_pred_conf, y_true_bin)
loss_unce = bce(y_pred_unce, y_true_bin)

print(f"\nBCE 손실:")
print(f"확신 있는 예측: {loss_conf.item():.4f}")
print(f"불확실한 예측: {loss_unce.item():.4f}")
print("확신 있을수록 손실 감소")

# BCEWithLogits vs BCE
print("\n중요: BCEWithLogitsLoss 사용 권장")
print("  - 수치적 안정성")
print("  - 출력층에 Sigmoid 제거")

# 3. 라벨 스무딩 간단 비교
X, y = make_classification(n_samples=500, n_features=20,
                          n_classes=3, n_informative=15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)

class SimpleNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# 일반 vs 라벨 스무딩
print("\n라벨 스무딩 비교:")

# 일반
model_no_smooth = SimpleNet(3)
criterion_no_smooth = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_no_smooth.parameters(), lr=0.01)

for _ in range(30):
    optimizer.zero_grad()
    loss = criterion_no_smooth(model_no_smooth(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

# 라벨 스무딩 (PyTorch 1.10+)
try:
    model_smooth = SimpleNet(3)
    criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model_smooth.parameters(), lr=0.01)

    for _ in range(30):
        optimizer.zero_grad()
        loss = criterion_smooth(model_smooth(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    print("  라벨 스무딩 (alpha=0.1) 적용 가능")
    print("  효과: 과신 방지, 일반화 향상")
except:
    print("  PyTorch 1.10 미만: label_smoothing 파라미터 없음")

# 4. 클래스 불균형

# 불균형 데이터 (95:5)
X_imb, y_imb = make_classification(
    n_samples=500, n_features=20,
    weights=[0.95, 0.05], random_state=42
)
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=42
)
scaler_i = StandardScaler()
X_train_i = scaler_i.fit_transform(X_train_i)
X_test_i = scaler_i.transform(X_test_i)

X_train_i_t = torch.FloatTensor(X_train_i)
y_train_i_t = torch.FloatTensor(y_train_i).unsqueeze(1)
X_test_i_t = torch.FloatTensor(X_test_i)

n_0 = np.sum(y_train_i == 0)
n_1 = np.sum(y_train_i == 1)
weight = n_0 / n_1

# Weighted BCE
class BinClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model_weighted = BinClassifier()
criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
optimizer = optim.Adam(model_weighted.parameters(), lr=0.001)

for _ in range(50):
    optimizer.zero_grad()
    loss = criterion_weighted(model_weighted(X_train_i_t), y_train_i_t)
    loss.backward()
    optimizer.step()

# 평가
model_weighted.eval()
with torch.no_grad():
    pred = torch.sigmoid(model_weighted(X_test_i_t))
    pred_bin = (pred > 0.5).float().numpy()

f1 = f1_score(y_test_i, pred_bin, zero_division=0)
print(f"\nWeighted BCE F1-Score: {f1:.4f}")    

# 5. 손실 곡면
print("\n손실 곡면 (Loss Landscape):")
print("  - 가중치 공간에서의 손실값 분포")
print("  - 지형의 모양이 학습에 영향")

print("\n좋은 손실 곡면:")
print("  1. 부드러운 표면 (Smooth)")
print("  2. 넓은 최소값 (Wide Minimum)")
print("  3. 적은 지역 최소값")

print("\n안정적 학습 방법:")
print("  1. 적절한 초기화 (He/Xavier)")
print("  2. BatchNorm 사용")
print("  3. 적절한 학습률")
print("  4. Gradient Clipping")

# 간단한 시각화
def simple_loss(w1, w2):
    return w1**2 + w2**2 + 0.5 * np.sin(w1*3) * np.cos(w2*3)

w = np.linspace(-2, 2, 50)
W1, W2 = np.meshgrid(w, w)
Z = simple_loss(W1, W2)

# 통합 시각화

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 1. 회귀 손실함수 비교
ax1 = axes[0, 0]
errors = np.linspace(-5, 5, 100)
mse_vals = 0.5 * errors**2
mae_vals = np.abs(errors)
huber_vals = []
for e in errors:
    if abs(e) <= 1.0:
        huber_vals.append(0.5 * e**2)
    else:
        huber_vals.append(abs(e) - 0.5)

ax1.plot(errors, mse_vals, label='MSE', linewidth=2, color='#e74c3c')
ax1.plot(errors, mae_vals, label='MAE', linewidth=2, color='#3498db')
ax1.plot(errors, huber_vals, label='Huber', linewidth=2, color='#2ecc71')
ax1.set_xlabel('Prediction Error')
ax1.set_ylabel('Loss Value')
ax1.set_title('Regression Loss Functions', weight='bold')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 10)

# 2. BCE 동작
ax2 = axes[0, 1]
probs = np.linspace(0.01, 0.99, 100)
bce_true1 = -np.log(probs)
bce_true0 = -np.log(1 - probs)

ax2.plot(probs, bce_true1, label='True=1', linewidth=2, color='#2ecc71')
ax2.plot(probs, bce_true0, label='True=0', linewidth=2, color='#e74c3c')
ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('BCE Loss')
ax2.set_title('Binary Cross Entropy', weight='bold')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 5)

# 3. 라벨 스무딩 효과
ax3 = axes[0, 2]
smoothing_alphas = [0.0, 0.05, 0.1, 0.15, 0.2]
test_accs = [0.85, 0.87, 0.88, 0.87, 0.85]  # 예시
ax3.plot(smoothing_alphas, test_accs, 'o-', linewidth=2,
         markersize=8, color='#3498db')
ax3.set_xlabel('Smoothing Alpha')
ax3.set_ylabel('Test Accuracy')
ax3.set_title('Label Smoothing Effect', weight='bold')
ax3.grid(alpha=0.3)
ax3.set_ylim(0.8, 0.9)

# 4. 클래스 불균형 전략
ax4 = axes[1, 0]
methods = ['Baseline', 'Weighted\nCE', 'Focal\nLoss']
f1_scores = [0.20, 0.65, 0.72]  # 예시
colors = ['#e74c3c', '#f39c12', '#2ecc71']

bars = ax4.bar(methods, f1_scores, color=colors, edgecolor='black', alpha=0.7)
ax4.set_ylabel('F1-Score')
ax4.set_title('Imbalance Handling', weight='bold')
ax4.set_ylim(0, 1.0)
ax4.grid(axis='y', alpha=0.3)

for bar, f1 in zip(bars, f1_scores):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{f1:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')

# 5. 손실 곡면 (등고선)
ax5 = axes[1, 1]
contour = ax5.contourf(W1, W2, Z, levels=15, cmap='viridis')
ax5.contour(W1, W2, Z, levels=15, colors='white', linewidths=0.5, alpha=0.3)
ax5.set_xlabel('Weight 1')
ax5.set_ylabel('Weight 2')
ax5.set_title('Loss Landscape', weight='bold')

# 6. 요약
ax6 = axes[1, 2]
ax6.axis('off')

summary = """
손실 함수 요약

회귀:
 MSE  - 표준, 이상치 민감
 MAE  - 강건, 해석 쉬움
 Huber- 절충안

분류:
 BCE  - 이진 분류
 CE   - 다중 분류
 Logits 버전 사용!

고급:
 스무딩 - 과신 방지
 가중치 - 불균형 대응
 Focal  - 극심한 불균형

손실 곡면:
 부드러움 = 안정적
 초기화 + BatchNorm
"""

ax6.text(0.1, 0.9, summary, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         family='monospace')

plt.tight_layout()
plt.savefig('loss_all_summary.png', dpi=150, bbox_inches='tight')
print("저장: loss_all_summary.png")
plt.close()

# 최종 요약
print("\n1. 회귀 손실함수")
print("   MSE: 큰 오차 페널티")
print("   MAE: 이상치에 강건")
print("   Huber: 둘의 장점")

print("\n2. 분류 손실함수")
print("   BCE/BCEWithLogits: 이진")
print("   CrossEntropy: 다중")
print("   Logits 버전 권장")

print("\n3. 라벨 스무딩")
print("   alpha=0.1 표준")
print("   과신 방지")
print("   일반화 향상")

print("\n4. 클래스 불균형")
print("   Weighted CE: 일반")
print("   Focal Loss: 극심")
print("   Recall 중시")

print("\n5. 손실 곡면")
print("   부드러운 곡면")
print("   적절한 초기화")
print("   BatchNorm 사용")

print("\n실전 체크리스트:")
print("  [v] 회귀: MSE 또는 Huber")
print("  [v] 이진: BCEWithLogitsLoss")
print("  [v] 다중: CrossEntropyLoss")
print("  [v] 대규모: label_smoothing=0.1")
print("  [v] 불균형: pos_weight 설정")
print("  [v] 극심한 불균형: Focal Loss")
print("  [v] 안정성: He 초기화 + BatchNorm")


