import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris() 

# 입력 데이터(꽃의 특징)와 정답 데이터(꽃의 종류)를 분리합니다.
x_org, y_org = iris.data, iris.target 

# 원본 데이터의 크기를 확인합니다. (총 150개의 데이터, 4개의 특징)
print('원본 데이터', x_org.shape, y_org.shape) 

# 이진 분류를 위해, 두 종류의 꽃 데이터(100개)와 두 개의 특징(꽃받침 길이/너비)만 사용합니다.
x_data = iris.data[:100, :2] 
y_data = iris.target[:100] 
print('대상 데이터', x_data.shape, y_data.shape) 

# 데이터를 훈련용 70개, 검증용 30개로 나눕니다.
# train_test_split 함수는 데이터를 무작위로 섞어주므로 편향을 방지할 수 있습니다.
# random_state=123으로 설정하여 실행할 때마다 동일한 결과를 얻도록 합니다.

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=70, test_size=30, random_state=123) 
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 입력 특징은 2개(꽃받침 길이, 너비)입니다.
n_input = x_train.shape[1]
n_output = 1

# 2입력 1출력 로지스틱 회귀 모델

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
        self.sigmoid = nn.Sigmoid()
        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = self.l1(x) 
        x2 = self.sigmoid(x1) 
        return x2
    
net = Net(n_input, n_output)
criterion = nn.BCELoss()   
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)
num_epochs = 10000
history = np.zeros((0,5))

inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).float()
labels1 = labels.view((-1,1))
inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).float()
labels1_test = labels_test.view((-1,1))

# 반복 계산 메인 루프

for epoch in range(num_epochs):
    # --- 훈련 페이즈 ---
    optimizer.zero_grad()      
    outputs = net(inputs)      
    loss = criterion(outputs, labels1) 
    loss.backward()            
    optimizer.step()           

    train_loss = loss.item()   
    predicted = torch.where(outputs < 0.5, 0, 1)
    train_acc = (predicted == labels1).sum() / len(y_train)

    # --- 예측(검증) 페이즈 ---
    # 여기서는 경사 계산과 파라미터 수정이 필요 없습니다. (훈련 가중치,편향 대입 계산)
    outputs_test = net(inputs_test) 
    loss_test = criterion(outputs_test, labels1_test) 
    val_loss = loss_test.item()
    predicted_test = torch.where(outputs_test < 0.5, 0, 1)
    val_acc = (predicted_test == labels1_test).sum() / len(y_test) 

    # 1000번에 한 번씩 결과 출력
    if (epoch % 1000 == 0):
        print(f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')

    # 기록
    item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
    history = np.vstack((history, item))

print('--- 최종 결과 ---')
print(f'초기 상태(검증) : 손실: {history[0, 3]:.5f}, 정확도: {history[0, 4]:.5f}')
print(f'최종 상태(검증) : 손실: {history[-1, 3]:.5f}, 정확도: {history[-1, 4]:.5f}')  


# BCEWithLogitsLoss
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)

        self.l1.weight.data.fill_(1.0)
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = self.l1(x)
        return x1
    
lr = 0.01
net = Net(n_input, n_output)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
num_epochs = 10000
history = np.zeros((0,5))

for epoch in range(num_epochs):
    # 훈련 페이즈

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels1)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    predicted = torch.where(outputs < 0.0, 0, 1) # 기준이 0.0
    train_acc = (predicted == labels1).sum() / len(y_train)

    # 예측 페이즈
    outputs_test = net(inputs_test)
    loss_test = criterion(outputs_test, labels1_test)
    val_loss =  loss_test.item()
    predicted_test = torch.where(outputs_test < 0.0, 0, 1)
    val_acc = (predicted_test == labels1_test).sum() / len(y_test)

    if ( epoch % 1000 == 0):
        print (f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))
    
print(f'초기 상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}' )
print(f'최종 상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.5f}' )