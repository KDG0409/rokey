import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ReLU 함수의 그래프

relu = nn.ReLU() # ReLU 함수 인스턴스 생성
x_np = np.arange(-2.0, 2.1, 0.25) # -2.0부터 2.0까지 0.25 간격으로 숫자 생성
x = torch.tensor(x_np).float() # 넘파이 배열을 파이토치 텐서로 변환
y = relu(x) # x 텐서에 ReLU 함수 적용

plt.plot(x.data, y.data)
plt.title('ReLU 함수')
plt.grid(True)
plt.show()

# GPU 사용하기
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
x_np = np.arange(-2.0, 2.1, 0.25)
y_np = np.arange(-1.0, 3.1, 0.25)
x = torch.tensor(x_np).float()
y = torch.tensor(y_np).float()
x = x.to(device)

try:
    z = x * y # 다른 디바이스 간 연산 불가
except RuntimeError as e:
    print(e)

y = y.to(device) # r같은 디바이스 간 연산 가능
z = x * y
print(z)

# Mnist 데이터 전처리
transform = transforms.Compose([
    transforms.ToTensor(), # (1) 데이터를 텐서로 변환
    transforms.Normalize(0.5, 0.5), # (2) 데이터 정규화
    transforms.Lambda(lambda x: x.view(-1)), # (3) 1계 텐서로 변환
])

data_root = './data'

train_set = datasets.MNIST( # Mnist 훈련 데이터 다운로드/설정
    root = data_root, train = True,
    download = True, transform = transform)
test_set = datasets.MNIST( # Mnist 검증 데이터 다운로드/설정
    root = data_root, train = False,
    download = True, transform = transform)

batch_size = 500

train_loader = DataLoader( # Mnist 훈련 데이터 미니배치(셔플o)
    train_set, batch_size = batch_size,
    shuffle = True)
test_loader = DataLoader( # Mnist 검증 데이터 미니배치(셔플x)
    test_set,  batch_size = batch_size,
    shuffle = False)

# Mnist 모델 정의
class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        # 은닉층 정의(은닉층 노드 수 : n_hidden)
        self.l1 = nn.Linear(n_input, n_hidden)

        # 출력층 정의
        self.l2 = nn.Linear(n_hidden, n_output)

        # ReLU 함수 정의
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        return x3
    
# Mnist 함수 정의

for images, labels in train_loader: # 이미지 추출
    break
image = images[0].numpy() # 처음 이미지(28,28)
n_input = image.shape[0] # 처음 이미지 형태 # 784 = 28*28
n_output = len(set(list(labels.data.numpy()))) # 10개로 분류
n_hidden = 128

torch.manual_seed(123) # 난수 고정
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

net = Net(n_input, n_output, n_hidden).to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
num_epochs = 100
history = np.zeros((0,5))

# Mnist 학습

for epoch in range(num_epochs):
    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0
    n_train, n_test = 0, 0

    for inputs, labels in tqdm(train_loader): # 훈련 페이즈
        n_train += len(labels)

        inputs = inputs.to(device) # GPU로 전송
        labels = labels.to(device)

        optimizer.zero_grad() # 경사 초기화
        outputs = net(inputs) # 예측 계산
        loss = criterion(outputs, labels) # 손실 계산
        loss.backward() # 경사 계산
        optimizer.step() # 파라미터 수정
        predicted = torch.max(outputs, 1)[1] # 예측 라벨 산출
        train_loss += loss.item() # 손실과 정확도 계산
        train_acc += (predicted == labels).sum().item()

    for inputs_test, labels_test in test_loader: # 예측 페이즈 (예측/손실/예측라벨 계산)
        n_test += len(labels_test)

        inputs_test = inputs_test.to(device)
        labels_test = labels_test.to(device)

        outputs_test = net(inputs_test)
        loss_test = criterion(outputs_test, labels_test)
        predicted_test = torch.max(outputs_test, 1)[1]
        val_loss +=  loss_test.item()
        val_acc +=  (predicted_test == labels_test).sum().item()

    train_acc = train_acc / n_train # 평가 결과 산출, 기록
    val_acc = val_acc / n_test
    train_loss = train_loss * batch_size / n_train
    val_loss = val_loss * batch_size / n_test
    print (f'Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
    item = np.array([epoch+1 , train_loss, train_acc, val_loss, val_acc])
    history = np.vstack((history, item))

# Mnist 결과 확인
print(f'초기상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}' )
print(f'최종상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.5f}' )

plt.plot(history[:,0], history[:,1], 'b', label='훈련') # 학습 곡선 출력(손실)
plt.plot(history[:,0], history[:,3], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.legend()
plt.show()

plt.plot(history[:,0], history[:,2], 'b', label='훈련') # 학습 곡선 출력(정확도)
plt.plot(history[:,0], history[:,4], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('정확도')
plt.title('학습 곡선(정확도)')
plt.legend()
plt.show()

# 은닉층 추가하기

# Mnist 모델 정의
class Net2(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        # 첫번째 은닉층 정의(은닉층 노드 수: n_hidden)
        self.l1 = nn.Linear(n_input, n_hidden)

        # 두번째 은닉층 정의(은닉층 노드 수: n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)

        # 출력층 정의
        self.l3 = nn.Linear(n_hidden, n_output)

        # ReLU 함수 정의
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        x4 = self.relu(x3)
        x5 = self.l3(x4)
        return x5
    
# Mnist 함수 정의
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

net = Net2(n_input, n_output, n_hidden).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr) 
num_epochs = 100
history2 = np.zeros((0,5))

# Mnist 학습
for epoch in range(num_epochs):
    train_acc = 0
    train_loss = 0
    val_acc = 0
    val_loss = 0
    n_train = 0
    n_test = 0

    for inputs, labels in tqdm(train_loader): # 훈련 페이즈
        n_train += len(labels)

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        predicted = torch.max(outputs, 1)[1]

        train_loss += loss.item()
        train_acc += (predicted == labels).sum().item()

    for inputs_test, labels_test in test_loader: # 예측 페이즈
        n_test += len(labels_test)

        inputs_test = inputs_test.to(device)
        labels_test = labels_test.to(device)

        outputs_test = net(inputs_test)
        loss_test = criterion(outputs_test, labels_test)
        predicted_test = torch.max(outputs_test, 1)[1]
        val_loss +=  loss_test.item()
        val_acc +=  (predicted_test == labels_test).sum().item()

    train_acc = train_acc / n_train # 평가 결과 산출, 기록
    val_acc = val_acc / n_test
    train_loss = train_loss * batch_size / n_train
    val_loss = val_loss * batch_size / n_test
    print (f'Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
    item = np.array([epoch+1 , train_loss, train_acc, val_loss, val_acc])
    history2 = np.vstack((history2, item))

# Mnist 결과 확인
print(f'초기상태 : 손실 : {history2[0,3]:.5f}  정확도 : {history2[0,4]:.5f}' )
print(f'최종상태 : 손실 : {history2[-1,3]:.5f}  정확도 : {history2[-1,4]:.5f}' )

plt.plot(history2[:,0], history2[:,1], 'b', label='훈련') # 학습 곡선 출력(손실)
plt.plot(history2[:,0], history2[:,3], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.legend()
plt.show()

plt.plot(history2[:,0], history2[:,2], 'b', label='훈련') # 학습 곡선 출력(정확도)
plt.plot(history2[:,0], history2[:,4], 'k', label='검증')
plt.xlabel('반복 횟수')
plt.ylabel('정확도')
plt.title('학습 곡선(정확도)')
plt.legend()
plt.show()

# 시그모이드 함수를 활성화 함수로 사용하기 (배치 사이즈 조정)

# Mnist 모델 정의
class Net3(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.l3 = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.sigmoid(x1)
        x3 = self.l2(x2)
        x4 = self.sigmoid(x3)
        x5 = self.l3(x4)
        return x5
    
# Mnist 함수 정의
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):
    base_epochs = len(history)
    for epoch in range(base_epochs, num_epochs+base_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        count = 0 # 훈련 페이즈

        for inputs, labels in tqdm(train_loader):
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # 경사 초기화
            outputs = net(inputs) # 예측 계산
            loss = criterion(outputs, labels) # 손실 계산
            train_loss += loss.item()
            loss.backward() # 경사 계산
            optimizer.step() # 파라미터 수정

            predicted = torch.max(outputs, 1)[1] # 예측 라벨 산출
            train_acc += (predicted == labels).sum().item() # 정답 건수 산출

            avg_train_loss = train_loss / count # 훈련 데이터에 대해 손실과 정확도 계산
            avg_train_acc = train_acc / count

        count = 0 # 예측 페이즈

        for inputs, labels in test_loader:
            count += len(labels)

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs) # 예측 계산
            loss = criterion(outputs, labels) # 손실 계산
            val_loss += loss.item()
            predicted = torch.max(outputs, 1)[1] # 예측 라벨 산출
            val_acc += (predicted == labels).sum().item() # 정답 건수 산출
            avg_val_loss = val_loss / count # 검증 데이터에 대해 손실과 정확도 계산
            avg_val_acc = val_acc / count

        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')
        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        history = np.vstack((history, item))
    return history

# Mnist 학습 (배치 사이즈 조정)

batch_size_train = 50
train_loader = DataLoader(
    train_set, batch_size = batch_size_train,
    shuffle = True)
lr = 0.01
net = Net(n_input, n_output, n_hidden).to(device)
optimizer = optim.SGD(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
num_epochs = 100
history3 = np.zeros((0,5))
history3 = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history3)

# Mnist 결과 확인

plt.plot(history[:,0], history[:,4], label='batch_size=500', c='k', linestyle='-.')
plt.plot(history3[:,0], history3[:,4], label='batch_size=200', c='b', linestyle='-.')
plt.xlabel('반복 횟수')
plt.ylabel('정확도')
plt.title('학습 곡선(정확도)')
plt.legend()
plt.show()