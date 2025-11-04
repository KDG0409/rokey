import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 데이터 전처리 (추출/분할/시각화/변환)
# 추출 
iris = load_iris()
x_org, y_org = iris.data, iris.target
x_select = x_org[:,[0,2]]
# 분할
x_train, x_test, y_train, y_test = train_test_split(
    x_select, y_org, train_size=75, test_size=75,
    random_state=123)
# 시각화
x_t0 = x_train[y_train == 0]
x_t1 = x_train[y_train == 1]
x_t2 = x_train[y_train == 2]
plt.scatter(x_t0[:,0], x_t0[:,1], marker='x', c='k', s=50, label='0 (setosa)')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o', c='b', s=50, label='1 (versicolor)')
plt.scatter(x_t2[:,0], x_t2[:,1], marker='+', c='k', s=50, label='2 (virginica)')
plt.xlabel('sepal_length')
plt.ylabel('petal_length')
plt.legend()
plt.show()
# 텐서 변환
inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).long()
inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).long()


# 모델 클래스 정의
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)

        # 초깃값을 모두 1로 함
        # "딥러닝을 위한 수학"과 조건을 맞추기 위한 목적
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = self.l1(x)
        return x1

# 함수 정의
n_input = x_train.shape[1]
n_output = len(list(set(y_train))) #set: 중복 제거   
net = Net(n_input, n_output)
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)
num_epochs = 10000
history = np.zeros((0,5))

for epoch in range(num_epochs):
# 학습(훈련데이터)
    # 경사 초기화
    optimizer.zero_grad()
    # 예측 계산
    outputs = net(inputs)
    # 손실 계산
    loss = criterion(outputs, labels)
    # 경사 계산
    loss.backward()
    # 파라미터 수정
    optimizer.step()
    # 예측 라벨 산출
    predicted = torch.max(outputs, 1)[1]
    # 손실과 정확도 계산
    train_loss = loss.item()
    train_acc = (predicted == labels).sum()  / len(labels)
# 예측(검증데이터)
    # 예측 계산
    outputs_test = net(inputs_test)
    # 손실 계산
    loss_test = criterion(outputs_test, labels_test)
    # 예측 라벨 산출
    predicted_test = torch.max(outputs_test, 1)[1]
    # 손실 정확도 계산
    val_loss =  loss_test.item()
    val_acc =  (predicted_test == labels_test).sum() / len(labels_test)

    if ((epoch) % 1000 == 0):
        print (f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))

# 결과 확인
print(f'초기상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}' )
print(f'최종상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.5f}' )
# Loss 시각화
plt.plot(history[:,0], history[:,1], 'b', label='훈련')
plt.plot(history[:,0], history[:,3], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.legend()
plt.show()
# Acc 시각화
plt.plot(history[:,0], history[:,2], 'b', label='훈련')
plt.plot(history[:,0], history[:,4], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('정확도')
plt.title('학습 곡선(정확도)')
plt.legend()
plt.show()


# 여러 방안
# 1. nn.CrossEntropyLoss() 사용
# 2. 모델에 LogSoftmax 정의 후 손실 함수에 nn.NLLLoss() 사용
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
        # logsoftmax 함수 정의
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.logsoftmax(x1)
        return x2
criterion = nn.NLLLoss()
# 3. 모델에 Softmax정의 후 출력값에 log를 취하고 손실함수에 nn.NLLLoss() 사용
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
        # 소프트맥스 함수 정의
        self.softmax = nn.Softmax(dim=1)

        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.softmax(x1)
        return x2
criterion = nn.NLLLoss()
for epoch in range(num_epochs):
    outputs2 = torch.log(outputs)
    loss = criterion(outputs2, labels)