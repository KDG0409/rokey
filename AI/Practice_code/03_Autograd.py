import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# 가상의 기울기 a와 y절편 b를 정하기
fake_a=3
fake_b=76

def predict(x):
    return fake_a * x + fake_b

x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])

predict_result = []
for i in range(len(x)):
    predict_result.append(predict(x[i]))

# MSE (함수/클래스/파이토치)

n=len(x)
def mse(y, y_pred):
    return (1/n) * sum((y - y_pred)**2)    

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, targets):
        loss = torch.mean((outputs - targets) ** 2)
        return loss   # 손실함수 값 반환한다

y_true = torch.tensor([3.0, 5.0, 2.0])
y_pred = torch.tensor([2.5, 4.5, 2.0])
mse_loss = nn.MSELoss()
loss = mse_loss(y_pred, y_true)
    
# 교차엔트로피오차(CEE)

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, outputs, targets):
        # 소프트맥스를 적용하여 확률 분포로 변환
        probs = F.softmax(outputs, dim=1)

        # 각 샘플에 대해 정답 클래스의 확률 값을 추출
        # targets 정수 레이블, gather:각 샘플의 정답 클래스 확률만 추출
        target_probs = probs.gather(1, targets.view(-1, 1))

        # 크로스 엔트로피 손실 계산
        loss = -torch.log(target_probs).mean()

        return loss
    
outputs = torch.randn(5, 3)
targets = torch.randint(0, 3, (5,))
loss_fn = CustomCrossEntropyLoss()
loss = loss_fn(outputs, targets)
print(loss)

# 파이토치
outputs = torch.randn(5, 3)
targets = torch.randint(0, 3, (5,))
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(outputs, targets)
print(loss)

# 경사하강법(함수)
model = torch.nn.Linear(10, 2)  # 10개의 입력 특성, 2개의 출력 클래스
criterion = torch.nn.CrossEntropyLoss() # 손실 함수 정의
optimizer = optim.SGD(model.parameters(), lr=0.01) # 옵티마이저 정의 (확률적 경사하강법)

inputs = torch.randn(64, 10)  # 64개의 샘플, 10개의 입력 특성
targets = torch.randint(0, 2, (64,))  # 64개의 샘플, 2개의 클래스 레이블

optimizer.zero_grad()  # 기울기 초기화
outputs = model(inputs)  # 모델의 출력 계산
loss = criterion(outputs, targets)  # 손실 계산
loss.backward()  # 기울기 계산
optimizer.step()  # 파라미터 업데이트

# 경사하강법(응용)

np.random.seed(0)
X_train = np.random.rand(100, 1).astype(np.float32)
y_train = 2 * X_train + 1 + np.random.normal(0, 0.1, (100, 1)).astype(np.float32)
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 선형 계층 정의 (입력 1, 출력 1)
        self.linear = nn.Linear(1, 1)  # y = wx + b

    def forward(self, x):
        return self.linear(x)
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
epochs = 1000
losses = []
for epoch in range(epochs):
    y_pred = model(X_train) #순전파
    loss = criterion(y_pred, y_train) #손실계산
    optimizer.zero_grad() #기울기 초기화
    #   w.grad.zero_() 
    #   b.grad.zero_()
    loss.backward() #역전파
    optimizer.step() #업데이트(수정)
    # with torch.no_grad():
    #   w -= lr * w.grad  # wt+1 = wt - lr * w.grad
    #   b -= lr * b.grad
    losses.append(loss.item()) # 손실기록

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print(f"학습된 기울기 (w): {model.linear.weight.item():.4f}")
print(f"학습된 편향 (b): {model.linear.bias.item():.4f}")

with torch.no_grad():
    predictions = model(X_train)
    plt.scatter(X_train.numpy(), y_train.numpy(), label='True data')
    plt.plot(X_train.numpy(), predictions.numpy(), color='red', label='pred line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# grad 계산 실습

x = torch.randn(5, 5, requires_grad=True)  # 5x5 랜덤 텐서
w = torch.randn(5, 5, requires_grad=True)
y = x * w
z = y.sum()
z.backward()
print(f"x의 기울기: {x.grad}")
print(f"w의 기울기: {w.grad}")

