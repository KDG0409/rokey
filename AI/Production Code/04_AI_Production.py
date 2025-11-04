import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
# 재현성을 위한 시드 고정
torch.manual_seed(42)
np.random.seed(42)
# 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# Part 1: 모델 정의
class BinaryClassifier(nn.Module): # 이진 분류용 다층 퍼셉트론
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()          # 사용할 활성화 함수(은닉층)
        self.dropout = nn.Dropout(0.3) # 과적합 방지 위해
        self.sigmoid = nn.Sigmoid()    # 이진분류(1,0) 위함(출력층)

    # 순전파
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer3(x))
        return x


class Regressor(nn.Module): # 회귀용 다층 퍼셉트론
    def __init__(self, input_dim):
        super(Regressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

print("모델 정의 완료")

# Part 2: 지도학습
# 분류 데이터 생성 및 학습
X_class, y_class = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=5, weights=[0.7, 0.3], random_state=42
)

X_train_c, X_temp_c, y_train_c, y_temp_c = train_test_split(
    X_class, y_class, test_size=0.4, random_state=42, stratify=y_class
)
X_val_c, X_test_c, y_val_c, y_test_c = train_test_split(
    X_temp_c, y_temp_c, test_size=0.5, random_state=42, stratify=y_temp_c
)

# 정규화 (표준화: 평균 0, 분산 1) 스케일링
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_val_c_scaled = scaler_c.transform(X_val_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

# 스케일링 된 데이터를 가지고, 텐서로 변환
X_train_c_t = torch.FloatTensor(X_train_c_scaled)
y_train_c_t = torch.FloatTensor(y_train_c).unsqueeze(1)
X_val_c_t = torch.FloatTensor(X_val_c_scaled)
y_val_c_t = torch.FloatTensor(y_val_c).unsqueeze(1)

model_class = BinaryClassifier(input_dim=20)
criterion = nn.BCELoss()
optimizer = optim.Adam(model_class.parameters(), lr=0.001)

print("분류 모델 학습 중...")
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(100):
    model_class.train()
    # 훈련 모드로 진입
    optimizer.zero_grad()
    # 최적화 초기화
    outputs = model_class(X_train_c_t)
    # 예측값 (모델이 예측 )
    loss = criterion(outputs, y_train_c_t)
    # 손실 계산 (예측값, 실제값)
    loss.backward()
    # 역전파
    optimizer.step()
    # 학습률(learning rate) 활용, 학습 >> 손실 줄이기 위해서
    # w(next step) = w(current step) -lr (diff f(x)/diff(xtr))

    # 평가용 모드 선언(모델의 성능평가)
    model_class.eval()
    with torch.no_grad():
        val_outputs = model_class(X_val_c_t)         # 반드시, 평가용 데이터 활용
        val_loss = criterion(val_outputs, y_val_c_t)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model_class.state_dict()
    else:
        patience_counter += 1

    if patience_counter >= 10:
        break
        # 정지조건

model_class.load_state_dict(best_model_state)

# 회귀 데이터 생성 및 학습
X_reg, y_reg = make_regression(
    n_samples=800, n_features=10, n_informative=8,
    noise=10.0, random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

X_train_r_t = torch.FloatTensor(X_train_r_scaled)
y_train_r_t = torch.FloatTensor(y_train_r).unsqueeze(1)

model_reg = Regressor(input_dim=10)
criterion_reg = nn.MSELoss()
optimizer_reg = optim.Adam(model_reg.parameters(), lr=0.01)

for epoch in range(100):
    model_reg.train()
    optimizer_reg.zero_grad()
    outputs = model_reg(X_train_r_t)
    loss = criterion_reg(outputs, y_train_r_t)
    loss.backward()
    optimizer_reg.step()

# Part 3: 비지도학습과 편향-분산
# K-Means 군집화
X_cluster, y_true_cluster = make_blobs(
    n_samples=300, centers=3, n_features=2,
    cluster_std=1.0, random_state=42
)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred_cluster = kmeans.fit_predict(X_cluster)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_cluster[:, 0], X_cluster[:, 1],
            c=y_true_cluster, cmap='viridis',
            alpha=0.6, edgecolors='k', s=50)
plt.title('True Labels', fontsize=12)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
plt.scatter(X_cluster[:, 0], X_cluster[:, 1],
            c=y_pred_cluster, cmap='plasma',
            alpha=0.6, edgecolors='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=300,
            edgecolors='black', linewidths=2)
plt.title('K-Means Result', fontsize=12)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
cluster_counts = np.bincount(y_pred_cluster)
plt.bar(range(len(cluster_counts)), cluster_counts,
        color=['#440154', '#31688e', '#fde724'], edgecolor='black')
plt.title('Cluster Sizes', fontsize=12)
plt.xlabel('Cluster ID')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('result_clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print("저장: result_clustering.png")

# 편향-분산 트레이드오프
np.random.seed(42)
X_bias = np.sort(np.random.rand(100, 1) * 10, axis=0)
y_bias = np.sin(X_bias).ravel() + np.random.randn(100) * 0.5
X_test_bias = np.linspace(0, 10, 200).reshape(-1, 1)
y_test_bias = np.sin(X_test_bias).ravel()

degrees = [1, 3, 9, 20]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

plt.figure(figsize=(16, 4))

for idx, degree in enumerate(degrees):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_bias)
    X_test_poly = poly.transform(X_test_bias)

    model_bias = Ridge(alpha=0.01)
    model_bias.fit(X_poly, y_bias)

    train_pred = model_bias.predict(X_poly)
    test_pred = model_bias.predict(X_test_poly)

    train_mse = mean_squared_error(y_bias, train_pred)
    test_mse = mean_squared_error(y_test_bias, test_pred)

    plt.subplot(1, 4, idx + 1)
    plt.scatter(X_bias, y_bias, alpha=0.5, s=30,
                color='gray', edgecolors='black')
    plt.plot(X_test_bias, y_test_bias, 'g--', linewidth=2.5)
    plt.plot(X_test_bias, test_pred, color=colors[idx], linewidth=2.5)
    plt.title(f'Degree {degree}\nTrain: {train_mse:.3f} | Test: {test_mse:.3f}')
    plt.ylim(-2.5, 2.5)
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('result_bias_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print("저장: result_bias_variance.png")

# Part 4: K-Fold 교차검증
X_kfold, y_kfold = make_classification(
    n_samples=500, n_features=20, n_informative=15, random_state=42
)

k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_kfold)):
    X_train_fold = X_kfold[train_idx]
    y_train_fold = y_kfold[train_idx]
    X_val_fold = X_kfold[val_idx]
    y_val_fold = y_kfold[val_idx]

    scaler_fold = StandardScaler()
    X_train_fold = scaler_fold.fit_transform(X_train_fold)
    X_val_fold = scaler_fold.transform(X_val_fold)

    X_train_fold_t = torch.FloatTensor(X_train_fold)
    y_train_fold_t = torch.FloatTensor(y_train_fold).unsqueeze(1)
    X_val_fold_t = torch.FloatTensor(X_val_fold)

    model_fold = BinaryClassifier(input_dim=20)
    optimizer_fold = optim.Adam(model_fold.parameters(), lr=0.01)
    criterion_fold = nn.BCELoss()

    for epoch in range(30):
        model_fold.train()
        optimizer_fold.zero_grad()
        outputs = model_fold(X_train_fold_t)
        loss = criterion_fold(outputs, y_train_fold_t)
        loss.backward()
        optimizer_fold.step()

    model_fold.eval()
    with torch.no_grad():
        val_pred_prob = model_fold(X_val_fold_t).numpy().flatten()
        val_pred = (val_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_val_fold, val_pred)
    fold_scores.append(accuracy)

print(f"K-Fold 평균 Accuracy: {np.mean(fold_scores):.4f} (std: {np.std(fold_scores):.4f})")

# Part 5: 평가 지표
# 분류 평가
model_class.eval()
X_test_c_t = torch.FloatTensor(X_test_c_scaled)

with torch.no_grad():
    y_pred_prob_c = model_class(X_test_c_t).numpy().flatten()
    y_pred_c = (y_pred_prob_c > 0.5).astype(int)

cm = confusion_matrix(y_test_c, y_pred_c)
accuracy = accuracy_score(y_test_c, y_pred_c)
precision = precision_score(y_test_c, y_pred_c, zero_division=0)
recall = recall_score(y_test_c, y_pred_c, zero_division=0)
f1 = f1_score(y_test_c, y_pred_c, zero_division=0)
auc = roc_auc_score(y_test_c, y_pred_prob_c)

# acc(정확도), precision(정밀도), recall(재현율), f1-score, auc
print(f"분류 성능: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}")

fig = plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'])
plt.title('Confusion Matrix')

plt.subplot(1, 3, 2)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
values = [accuracy, precision, recall, f1, auc]
plt.barh(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
plt.xlim(0, 1.0)
plt.title('Metrics')
plt.grid(axis='x', alpha=0.3)

plt.subplot(1, 3, 3)
fpr, tpr, _ = roc_curve(y_test_c, y_pred_prob_c)
plt.plot(fpr, tpr, linewidth=3, label=f'AUC={auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlabel('FPR') # false positive rate
plt.ylabel('TPR') # true positive rate
plt.title('ROC Curve')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('result_classification.png', dpi=150, bbox_inches='tight')
plt.close()
print("저장: result_classification.png")

# 회귀 평가
model_reg.eval()
X_test_r_t = torch.FloatTensor(X_test_r_scaled)

with torch.no_grad():
    y_pred_reg = model_reg(X_test_r_t).numpy().flatten()

mae = mean_absolute_error(y_test_r, y_pred_reg)
mse = mean_squared_error(y_test_r, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_r, y_pred_reg)  # 결정계수 : 모델의 설명력(해석)

print(f"회귀 성능: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

fig = plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test_r, y_pred_reg, alpha=0.6, s=50)
plt.plot([y_test_r.min(), y_test_r.max()],
         [y_test_r.min(), y_test_r.max()], 'r--', linewidth=3)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Prediction vs True')
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
residuals = y_test_r - y_pred_reg
# 각 관측치(관측된 포인트)에서 예측치를 뺀 값 >> 잔차
plt.scatter(y_pred_reg, residuals, alpha=0.6, s=50)
plt.axhline(y=0, color='r', linestyle='--', linewidth=3)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', linewidth=3)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('result_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print("저장: result_regression.png")

# Part 6: ML 파이프라인 (통합)
class MLPipeline:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.best_score = 0
        self.baseline_score = 0

    def run(self, X, y):
        print("\nSTEP 1: 문제 정의")
        print("  목표: 고객 이탈 예측 (F1 > 0.80)")

        print("\nSTEP 2: 데이터 준비")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        print("\nSTEP 3: 베이스라인")
        majority_class = np.bincount(y_train).argmax()
        baseline_pred = np.full(len(y_test), majority_class)
        self.baseline_score = f1_score(y_test, baseline_pred, zero_division=0)
        print(f"  베이스라인 F1: {self.baseline_score:.4f}")

        print("\nSTEP 4: 모델 학습")
        self.model = BinaryClassifier(input_dim=X.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(50):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 10:
                break

        self.model.load_state_dict(best_model_state)
        print("  학습 완료")

        print("\nSTEP 5: 최종 평가")
        self.model.eval()
        X_test_t = torch.FloatTensor(X_test)

        with torch.no_grad():
            y_pred_prob = self.model(X_test_t).numpy().flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)

        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        print(f"  테스트 F1: {test_f1:.4f}")
        print(f"  베이스라인 대비: {test_f1 - self.baseline_score:+.4f}")

        if test_f1 > 0.80:
            print("  성공! 목표 달성")
        else:
            print("  목표 미달, 추가 개선 필요")

X_proj, y_proj = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    weights=[0.65, 0.35], random_state=42
)

pipeline = MLPipeline()
pipeline.run(X_proj, y_proj)