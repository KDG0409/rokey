import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot

# 입력 :1 출력 :1인 선형 함수

torch.manual_seed(123)
l1 = nn.Linear(1, 1)
nn.init.constant_(l1.weight, 2.0) # weight 텐서의 모든 값을 2.0으로 설정
nn.init.constant_(l1.bias, 1.0)   # bias 텐서의 모든 값을 1.0으로 설정
print("weight:", l1.weight)
print("bias:", l1.bias)

x_np = np.arange(-2.0, 2.1, 1.0)
x = torch.tensor(x_np).float()
x = x.view(-1, 1)
y = l1(x)

# 입력 :2 출력 :1인 선형 함수
l2 = nn.Linear(2, 1)
nn.init.constant_(l2.weight, 1.0)
nn.init.constant_(l2.bias, 2.0)
x2_np = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
x2 = torch.tensor(x2_np).float()
y2 = l2(x2)

# 입력 :2 출력:3인 선형 함수

l3 = nn.Linear(2, 3)
nn.init.constant_(l3.weight[0, :], 1.0)
nn.init.constant_(l3.weight[1, :], 2.0)
nn.init.constant_(l3.weight[2, :], 3.0)
nn.init.constant_(l3.bias, 2.0)
y3 = l3(x2)

# 커스텀 클래스
class Net(nn.Module): # 모든 모델의 부모 클래스인 nn.Module을 상속받습니다.
    def __init__(self, n_input, n_output):
        # 부모 클래스 nn.Module의 초기화 메서드를 반드시 호출해야 합니다.
        super().__init__()

        # 출력층 정의: 입력 특성 수(n_input)와 출력 특성 수(n_output)를 받는 선형 레이어(l1)를 정의합니다.
        self.l1 = nn.Linear(n_input, n_output)

    # 예측 함수 정의: 데이터의 순전파 흐름을 정의합니다.
    def forward(self, x):
        x1 = self.l1(x) # __init__에서 정의한 선형 레이어에 입력 x를 통과시킵니다.
        return x1
    
# 더미 입력 데이터 생성 (100개의 샘플, 1개의 특성)

inputs = torch.ones(100, 1)
n_input = 1
n_output = 1
net = Net(n_input, n_output)
outputs = net(inputs)
criterion = nn.MSELoss()

labels1 = torch.zeros(100, 1) # 정답이 모두 0이라고 가정

# 손실 계산
# "딥러닝을 위한 수학"의 결과와 일치시키기 위해 2로 나눈 값을 손실로 함
loss = criterion(outputs, labels1) / 2.0
loss.backward()

print(f"계산된 손실(loss): {loss.item()}")
print(f"l1.weight.grad: {net.l1.weight.grad}")

# 보스턴 데이터셋
# 데이터 전처리
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+",skiprows=22, header=None)
x_org = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
yt = raw_df.values[1::2, 2]
feature_names = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
                          'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT'])
x = x_org[:, feature_names == 'RM']
plt.scatter(x, yt, s=10, c='b') # x축은 방 개수, y축은 가격
plt.xlabel('방 개수')
plt.ylabel('가격')
plt.title('방 개수와 가격의 산포도')
plt.grid(True)
plt.show()

inputs = torch.tensor(x).float()
labels = torch.tensor(yt).float()
labels1 = labels.view((-1, 1))

# 모델 정의
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
        nn.init.constant_(self.l1.weight, 1.0)
        nn.init.constant_(self.l1.bias, 1.0)

    def forward(self, x): 
        x1 = self.l1(x) 
        return x1
    
n_input = x.shape[1]
n_output = 1
net = Net(n_input, n_output)
criterion = nn.MSELoss()
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)
num_epochs = 5000
history = np.zeros((0, 2))

# 학습
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels1) / 2.0
    loss.backward()
    optimizer.step()

    if (epoch % 1000 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')

# 결과 분석
print(f'초기 손실값: {history[0,1]:.5f}')
print(f'최종 손실값: {history[-1,1]:.5f}')

plt.plot(history[1:,0], history[1:,1], 'b')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.show()

# 회귀 직선 산출
xse = np.array((x.min(), x.max())).reshape(-1,1)
Xse = torch.tensor(xse).float()

with torch.no_grad():
  Yse = net(Xse)

plt.scatter(x, yt, s=10, c='b')
plt.xlabel('방 개수')
plt.ylabel('가격')
plt.plot(Xse.data, Yse.data, c='k')
plt.title('산포도와 회귀 직선')
plt.show()

# 모델 확장 
x_add = x_org[:, feature_names == 'LSTAT']
x2 = np.hstack((x, x_add))
n_input = x2.shape[1]
print(n_input)
net = Net(n_input, n_output)
criterion = nn.MSELoss()
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)
num_epochs = 50000
history = np.zeros((0,2))

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels1) / 2.0
    loss.backward()
    optimizer.step()

    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')

print(f'초기 손실값: {history[0,1]:.5f}')
print(f'최종 손실값: {history[-1,1]:.5f}')

plt.plot(history[1:,0], history[1:,1], 'b')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.show()