import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchviz import make_dot
import torch.optim as optim

# 예측 함수
l1 = nn.Linear(784, 128)
l2 = nn.Linear(128, 10)
relu = nn.ReLU(inplace=True)

inputs = torch.randn(100, 784)
m1 = l1(inputs) # 중간 텐서 1 계산
m2 = relu(m1) # 중간 텐서 2 계산
outputs = l2(m2) # 출력 텐서 계산

net2 = nn.Sequential(
    l1,
    relu,   # 활성화 함수(은닉층)
    l2
)
outputs2 = net2(inputs)

# 데이터 생성/시각화/텐서변환
np.random.seed(123)
x = np.random.randn(100, 1) * 2.5
y = x**2 + np.random.randn(100, 1) * 0.8
x_train = x[:50,:]
y_train = y[:50,:]
x_test = x[50:,:]
y_test = y[50:,:]
plt.scatter(x_train, y_train, c='c', label='훈련 데이터')
plt.scatter(x_test, y_test, c='k', marker='x', label='검증 데이터')
plt.legend()
plt.show()
inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).float()
inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).float()

# 선형회귀모델(은닉층 없음,활성화 함수 없음)
class Net(nn.Module):
    def __init__(self):
        #  부모 클래스 nn.Modules 의 초기화
        super().__init__()

        # 출력층 정의
        self.l1 = nn.Linear(1, 1)

    # 예측 함수 정의(순전파)
    def forward(self, x):
        x1 = self.l1(x) # 선형 회귀
        return x1

lr = 0.01
net = Net()
optimizer = optim.SGD(net.parameters(), lr=lr)
criterion = nn.MSELoss()
num_epochs = 10000
history = np.zeros((0,2))

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')

labels_pred = net(inputs_test) # 테스트 데이터로 평가
plt.title('은닉층 없음,활성화 함수 없음')
plt.scatter(inputs_test[:,0].data, labels_pred[:,0].data, c='b', label='예측값')
plt.scatter(inputs_test[:,0].data, labels_test[:,0].data, c='k', marker='x',label='정답')
plt.legend()
plt.show()

# 선형회귀모델(은닉층 있음,활성화 함수 없음)
class Net2(nn.Module):
    def __init__(self):
        #  부모 클래스 nn.Modules 초기화
        super().__init__()

        # 출력층 정의
        self.l1 = nn.Linear(1, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10,1)

    # 예측 함수 정의
    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        return x3
    
lr = 0.01
net2 = Net2()
optimizer = optim.SGD(net2.parameters(), lr=lr)
criterion = nn.MSELoss()
num_epochs = 10000
history = np.zeros((0,2))

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net2(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')

labels_pred = net2(inputs_test) # 테스트 데이터로 평가
plt.title('은닉층 있음,활성화 함수 없음')
plt.scatter(inputs_test[:,0].data, labels_pred[:,0].data, c='b', label='예측값')
plt.scatter(inputs_test[:,0].data, labels_test[:,0].data, c='k', marker='x',label='정답')
plt.legend()
plt.show()

# 선형회귀모델(은닉층 있음,활성화 함수 있음)
class Net3(nn.Module):
    def __init__(self):
        #  부모 클래스 nn.Modules 초기화
        super().__init__()

        # 출력층 정의
        self.l1 = nn.Linear(1, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10,1)
        self.relu = nn.ReLU(inplace=True)

    # 예측 함수 정의
    def forward(self, x):
        x1 = self.relu(self.l1(x))
        x2 = self.relu(self.l2(x1))
        x3 = self.l3(x2)
        return x3

lr = 0.01
net3 = Net3()
optimizer = optim.SGD(net2.parameters(), lr=lr)
criterion = nn.MSELoss()
num_epochs = 10000
history = np.zeros((0,2))

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net3(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')

labels_pred = net3(inputs_test) # 테스트 데이터로 평가
plt.title('은닉층 있음,활성화 함수 없음')
plt.scatter(inputs_test[:,0].data, labels_pred[:,0].data, c='b', label='예측값')
plt.scatter(inputs_test[:,0].data, labels_test[:,0].data, c='k', marker='x',label='정답')
plt.legend()
plt.show()