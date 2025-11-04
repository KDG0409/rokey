import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine  # Wine 데이터셋 (13개 특성, 3개 클래스)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.family'] = 'DejaVu Sans'

def load_and_prepare_data():
    """
    데이터를 로드하고 전처리하는 함수
    Returns:
        tuple: (x_train, x_val, x_test, y_train, y_val, y_test, scaler)
    """
    # 데이터 로드
    wine = load_wine()
    X, y = wine.data, wine.target
    print(f"전체 데이터 크기: {X.shape}")
    print(f"특성(Feature) 수: {X.shape[1]}")
    print(f"클래스 수: {len(np.unique(y))}")
    print(f"클래스별 분포: {np.bincount(y)}")
    # Train:Val:Test = 60:20:20 비율로 분할
    X_temp, x_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    # 표준화 (Standardization): 평균 0, 표준편차 1로 조정
    # 신경망 학습 시 수렴 속도와 안정성 향상
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    print(f"\n훈련 데이터: {x_train.shape[0]}개")
    print(f"검증 데이터: {x_val.shape[0]}개")
    print(f"테스트 데이터: {x_test.shape[0]}개")

    return x_train, x_val, x_test, y_train, y_val, y_test, scaler

# 데이터 준비
x_train, x_val, x_test, y_train, y_val, y_test, scaler = load_and_prepare_data()

def convert_to_tensors(x_train, x_val, x_test, y_train, y_val, y_test):
    """
    NumPy 배열을 PyTorch 텐서로 변환
    Args:
        x_train, x_val, x_test: 입력 데이터 (numpy array)
        y_train, y_val, y_test: 레이블 데이터 (numpy array)
    Returns:
        tuple: (inputs_train, inputs_val, inputs_test, labels_train, labels_val, labels_test)
    """
    # 입력 데이터는 float32 타입
    inputs_train = torch.tensor(x_train, dtype=torch.float32)
    inputs_val = torch.tensor(x_val, dtype=torch.float32)
    inputs_test = torch.tensor(x_test, dtype=torch.float32)
    # 레이블은 long(int64) 타입 - CrossEntropyLoss 요구사항
    labels_train = torch.tensor(y_train, dtype=torch.long)
    labels_val = torch.tensor(y_val, dtype=torch.long)
    labels_test = torch.tensor(y_test, dtype=torch.long)

    print(f"훈련 입력 텐서 shape: {inputs_train.shape}, dtype: {inputs_train.dtype}")
    print(f"훈련 레이블 텐서 shape: {labels_train.shape}, dtype: {labels_train.dtype}")

    return inputs_train, inputs_val, inputs_test, labels_train, labels_val, labels_test

# 텐서 변환
inputs_train, inputs_val, inputs_test, labels_train, labels_val, labels_test = convert_to_tensors(
    x_train, x_val, x_test, y_train, y_val, y_test
)

class DeepMultiClassNet(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output, dropout_rate=0.3):
        """
        다층 신경망 모델
        구조: Input -> Hidden1 -> ReLU -> Dropout -> Hidden2 -> ReLU -> Dropout -> Output
        Args:
            n_input: 입력 특성 수
            n_hidden1: 첫 번째 은닉층 뉴런 수
            n_hidden2: 두 번째 은닉층 뉴런 수
            n_output: 출력 클래스 수
            dropout_rate: 드롭아웃 비율 (0.0 ~ 1.0)
        """
        super(DeepMultiClassNet, self).__init__()
        # 계층 정의
        self.fc1 = nn.Linear(n_input, n_hidden1)      # 첫 번째 완전연결층
        self.relu1 = nn.ReLU()                        # ReLU 활성화 함수
        self.dropout1 = nn.Dropout(dropout_rate)      # 드롭아웃

        self.fc2 = nn.Linear(n_hidden1, n_hidden2)    # 두 번째 완전연결층
        self.relu2 = nn.ReLU()                        # ReLU 활성화 함수
        self.dropout2 = nn.Dropout(dropout_rate)      # 드롭아웃

        self.fc3 = nn.Linear(n_hidden2, n_output)     # 출력층 (Softmax는 손실함수에 포함)

    def forward(self, x):
        """
        순전파(Forward Propagation)
        Args:
            x: 입력 텐서
        Returns:
            logits: 원본 출력값 (Softmax 적용 전)
        """
        # 첫 번째 은닉층
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # 두 번째 은닉층
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # 출력층 (logits 반환)
        logits = self.fc3(x)
        return logits

# 모델 하이퍼파라미터
n_input = inputs_train.shape[1]   # 13개 특성
n_hidden1 = 64                     # 첫 번째 은닉층: 64개 뉴런
n_hidden2 = 32                     # 두 번째 은닉층: 32개 뉴런
n_output = 3                       # 3개 클래스
dropout_rate = 0.3                 # 30% 드롭아웃

# 모델 생성
model = DeepMultiClassNet(n_input, n_hidden1, n_hidden2, n_output, dropout_rate)

print("=" * 60)
print("모델 구조:")
print("=" * 60)
print(model)
print("=" * 60)

# 총 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n총 파라미터 수: {total_params:,}")
print(f"학습 가능한 파라미터 수: {trainable_params:,}")

criterion = nn.CrossEntropyLoss() # 손실 함수: CrossEntropyLoss (Softmax + NLLLoss 포함)
learning_rate = 0.001 
optimizer = optim.Adam(model.parameters(),lr=learning_rate) # 옵티마이저: Adam (Adaptive Moment Estimation)
# SGD보다 학습률 조정이 자동화되어 더 빠르고 안정적인 학습 가능
print(f"손실 함수: {criterion}")
print(f"옵티마이저: {optimizer.__class__.__name__}")
print(f"학습률: {learning_rate}")

# 모델 평가 함수
def evaluate_model(model, inputs_train, labels_train):
    """
    모델의 손실과 정확도를 계산하는 함수
    Args:
        model: 평가할 모델
        inputs: 입력 텐서
        labels: 정답 레이블 텐서
    Returns:
        tuple: (loss, accuracy)
    """
    model.eval()  # 평가 모드 (Dropout, BatchNorm 등 비활성화)
    with torch.no_grad():
        outputs = model(inputs_train)
        loss = criterion(outputs,labels_train)
        pre = torch.max(outputs,1)[1] # indices
        accuracy = (pre == labels_train).sum().item()/len(labels_train)
    return loss.item(), accuracy

# 학습 루프
def train_model(model, inputs_train, labels_train, inputs_val, labels_val,
                num_epochs=200, print_interval=20):
    """
    모델을 학습시키는 함수
    Args:
        model: 학습할 모델
        inputs_train: 훈련 입력 텐서
        labels_train: 훈련 레이블 텐서
        inputs_val: 검증 입력 텐서
        labels_val: 검증 레이블 텐서
        num_epochs: 학습 에포크 수
        print_interval: 출력 간격
    Returns:
        dict: 학습 이력 (train_losses, val_losses, train_accs, val_accs)
    """
    # 학습 이력 저장용 리스트
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("=" * 70)
    print(f"{'Epoch':^10} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^12} | {'Val Acc':^10}")
    print("=" * 70)

    for epoch in range(num_epochs):
        # --- 학습 단계 ---
        model.train()  # 학습 모드 (Dropout 활성화)

        # 1. 순전파
        outputs = model(inputs_train)

        # 2. 손실 계산
        loss = criterion(outputs, labels_train)

        # 3. 역전파
        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()        # 역전파 (기울기 계산)

        # 4. 가중치 업데이트
        optimizer.step()

        # --- 평가 단계 ---
        # 훈련 데이터 평가
        train_loss, train_acc = evaluate_model(model, inputs_train, labels_train)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 검증 데이터 평가
        val_loss, val_acc = evaluate_model(model, inputs_val, labels_val)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 결과 출력
        if (epoch + 1) % print_interval == 0:
            print(f"{epoch+1:^10} | {train_loss:^12.4f} | {train_acc:^10.4f} | {val_loss:^12.4f} | {val_acc:^10.4f}")

    print("=" * 70)
    print("학습 완료!")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
history = train_model(
    model,
    inputs_train, labels_train,
    inputs_val, labels_val,
    num_epochs=200,
    print_interval=20
)

# 학습 과정 시각화
def plot_training_history(history):
    """
    학습 과정을 시각화하는 함수

    Args:
        history: 학습 이력 딕셔너리
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 손실(Loss) 그래프
    axes[0].plot(history['train_losses'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_losses'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curve', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 정확도(Accuracy) 그래프
    axes[1].plot(history['train_accs'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_accs'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 최종 결과 출력
    print("\n" + "=" * 50)
    print("최종 학습 결과")
    print("=" * 50)
    print(f"훈련 손실: {history['train_losses'][-1]:.4f}")
    print(f"검증 손실: {history['val_losses'][-1]:.4f}")
    print(f"훈련 정확도: {history['train_accs'][-1]:.4f} ({history['train_accs'][-1]*100:.2f}%)")
    print(f"검증 정확도: {history['val_accs'][-1]:.4f} ({history['val_accs'][-1]*100:.2f}%)")
    print("=" * 50)

plot_training_history(history)

# 테스트(정답) 데이터 평가
def test_model(model, inputs_test, labels_test):
    """
    테스트 데이터로 최종 성능 평가
    Args:
        model: 평가할 모델
        inputs_test: 테스트 입력 텐서
        labels_test: 테스트 레이블 텐서
    """
    test_loss, test_acc = evaluate_model(model, inputs_test, labels_test)

    print("\n" + "=" * 50)
    print("테스트 데이터 최종 성능")
    print("=" * 50)
    print(f"테스트 손실: {test_loss:.4f}")
    print(f"테스트 정확도: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("=" * 50)

    # 클래스별 예측 결과
    model.eval()
    with torch.no_grad():
        outputs = model(inputs_test)
        _, predicted = torch.max(outputs, 1)

    # 클래스별 정확도 계산
    for class_idx in range(3):
        class_mask = (labels_test == class_idx)
        class_correct = (predicted[class_mask] == labels_test[class_mask]).sum().item()
        class_total = class_mask.sum().item()
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"클래스 {class_idx} 정확도: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_correct}/{class_total}개")

test_model(model, inputs_test, labels_test)

# 예측 예시

def predict_sample(model, inputs, labels, sample_idx=0):
    """
    개별 샘플에 대한 예측 및 확률 출력
    Args:
        model: 예측할 모델
        inputs: 입력 텐서
        labels: 정답 레이블 텐서
        sample_idx: 예측할 샘플 인덱스
    """
    model.eval()

    # 단일 샘플 선택 (배치 차원 추가)
    sample_input = inputs[sample_idx:sample_idx+1]
    true_label = labels[sample_idx].item()

    with torch.no_grad():
        # 모델 출력 (logits)
        output = model(sample_input)
        # Softmax 적용하여 확률 계산
        probabilities = torch.softmax(output, dim=1)[0]
        # 예측 클래스
        predicted_label = torch.argmax(probabilities).item()

    print("\n" + "=" * 50)
    print(f"샘플 {sample_idx} 예측 결과")
    print("=" * 50)
    print(f"실제 클래스: {true_label}")
    print(f"예측 클래스: {predicted_label}")
    print(f"예측 결과: {'정답 ✓' if predicted_label == true_label else '오답 ✗'}")
    print("\n각 클래스별 확률:")
    for i, prob in enumerate(probabilities):
        print(f"  클래스 {i}: {prob:.4f} ({prob*100:.2f}%)")
    print("=" * 50)

# 테스트 샘플 예측
predict_sample(model, inputs_test, labels_test, sample_idx=0)
predict_sample(model, inputs_test, labels_test, sample_idx=5)