# 1. sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

# 데이터 로드
data = load_iris()  # Iris 꽃 데이터셋 사용
# 아이리스 데이터셋은 150개의 샘플과 4개의 특성(꽃받침 길이, 꽃받침 폭, 꽃잎 길이, 꽃잎 폭)으로 구성되어 있으며, 3개의 클래스를 가진 데이터입니다.
X = data.data
y = data.target

# 데이터 분할 (학습용/테스트용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 초기화 및 학습
model = LogisticRegression(max_iter=200)  # 최대 반복 횟수를 지정 (기본값 100)
model.fit(X_train, y_train)
# LogisticRegression(max_iter=200)는 로지스틱 회귀 모델을 초기화
# 반복 횟수를 200으로 설정하여 학습의 수렴 속도를 높입니다.
# fit() 메서드를 통해 학습용 데이터를 모델에 학습시킵니다. 이 과정을 통해 모델은 데이터 패턴을 학습하고, 입력 특성이 주어졌을 때 특정 클래스를 예측하는 방법을 익힙니다.
predictions = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, predictions)
print(f"모델 정확도: {accuracy * 100:.2f}%")

# 2.이진 분류
import torch  # 딥러닝 프레임워크
import torch.nn as nn  # 신경망 설계를 위한 모듈
import torch.optim as optim  # 최적화 함수 (파라미터 업데이트)
from sklearn.datasets import make_classification  # 데이터를 생성하는 모듈
from sklearn.model_selection import train_test_split  # 데이터 분리 도구

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # 특징 데이터
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 레이블 데이터 (-1, 1로 reshape)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class SimpleNN(nn.Module):  # PyTorch의 nn.Module을 상속받아 정의
    def __init__(self):
        super(SimpleNN, self).__init__()  # 상속받아 초기화
        self.layer = nn.Linear(10, 1)  # 선형 레이어. 입력 10개 -> 출력 1개

    def forward(self, x):
        return torch.sigmoid(self.layer(x))

model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100 
for epoch in range(epochs):  
    model.train() 
    optimizer.zero_grad()  
    y_pred = model(X_train_tensor) 
    loss = criterion(y_pred, y_train_tensor) 
    loss.backward() 
    optimizer.step()
    if epoch % 10 == 0:
        print(f"에포크 {epoch} | 손실: {loss.item():.4f}")  

model.eval()  # 모델을 평가 모드로 전환
with torch.no_grad():  # 평가에서는 기울기를 계산하지 않음
    y_test_pred = model(X_test_tensor)  # 테스트 데이터를 사용해 예측값 계산
    y_test_pred = (y_test_pred > 0.5).float()  # 0.5를 기준으로 0 또는 1로 분류
    # 정확도 계산
    accuracy = (y_test_pred.eq(y_test_tensor).sum().item()) / y_test_tensor.shape[0]
    print(f"테스트 정확도: {accuracy * 100:.2f}%")  # 테스트 정확도 출력

# 3. FashionMNIST

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 데이터 전처리(transform) >> data >> tensor data로 변형 >> normalize (정규화)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. 모델 정의 (간단한 신경망)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()     # 상속
        self.fc1 = nn.Linear(28 * 28, 128)
        # 입력: 28x28 크기의 이미지(784차원) -> 128 차원
        self.fc2 = nn.Linear(128, 10)
        # 출력: 10개의 클래스 (패션 아이템 종류)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 이미지를 1D 벡터로 변환
        x = torch.relu(self.fc1(x))  # 활성화 함수 ReLU
        x = self.fc2(x)              # 출력
        return x
    
# 3. 모델 초기화 및 설정
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류를 위한 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저

# 4. 모델 훈련
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0     # 에폭마다 손실 초기화
        for data, target in train_loader:
            optimizer.zero_grad()  # 기울기 초기화
            output = model(data)   # 모델 예측
            loss = criterion(output, target)  # 손실 계산
            loss.backward()        # 역전파
            optimizer.step()       # 가중치 업데이트

            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}')

# 5. 모델 평가
def test(model, test_loader):
    model.eval()  # 평가 모드로 설정
    correct = 0
    total = 0
    with torch.no_grad():  # 그래디언트 계산을 하지 않음
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            # 가장 높은 확률을 가진 클래스를 예측
            total += target.size(0)
            correct += (predicted == target).sum().item()  # 맞춘 개수 카운트

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# 훈련/테스트
train(model, train_loader, criterion, optimizer, epochs=5)
test(model, test_loader)

# 시각화
def visualize_predictions(model, test_loader, num_images=10):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # 모델로 예측 수행
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)  # 가장 높은 확률을 가진 클래스 선택

    # FashionMNIST 클래스 이름 정의
    classes = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    # 시각화
    plt.figure(figsize=(15, 4))
    for idx in range(num_images):
        ax = plt.subplot(2, num_images // 2, idx + 1)
        ax.imshow(images[idx].numpy().squeeze(), cmap="gray")  # 이미지를 Grayscale 형태로 표시
        ax.set_title(f"Pred: {classes[predicted[idx]]}\nTrue: {classes[labels[idx]]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# 시각화 함수 호출
visualize_predictions(model, test_loader)