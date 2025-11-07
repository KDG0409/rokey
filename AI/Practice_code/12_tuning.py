# 깊은 모델 > 과적합 방지 > 드롭아웃,배치정규화,데이터 증강 (epoch증가필요,필요시간 증가)
# CNN 모델 설계 : pooling마다 채널수 증가 필요
# 최적화 함수

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from tqdm.auto import tqdm
import warnings
warnings.simplefilter('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_output = len(list(set(classes)))

# 최적화 함수

W = torch.randn(3, 3, requires_grad=True) # SGD
B = torch.randn(3, requires_grad=True)
W.grad = torch.randn(3, 3)
B.grad = torch.randn(3)
lr = 0.001
W.data -= lr * W.grad.data
B.data -= lr * B.grad.data

net_params = [torch.randn(10, 1, requires_grad=True)] # 모멘텀
optimizer = optim.SGD(net_params, lr=lr, momentum=0.9)

net_params = [torch.randn(10, 1, requires_grad=True)] # Adam
optimizer = optim.Adam(net_params)

# 기본 CNN (SGD 사용)

# 데이터 전처리 (transform->dataset>dataloader)

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(0.5, 0.5)
])

data_root = './data'

train_set = datasets.CIFAR10(
    root = data_root, train = True,
    download = True, transform = transform)

test_set = datasets.CIFAR10(
    root = data_root, train = False,
    download = True, transform = transform)

batch_size = 100
train_loader = DataLoader(train_set,
    batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_set,
    batch_size = batch_size, shuffle = False)

# 모델 정의
class CNN_v2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1,1))
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.conv5 = nn.Conv2d(64, 128, 3, padding=(1,1))
        self.conv6 = nn.Conv2d(128, 128, 3, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool2d((2,2))

        self.l1 = nn.Linear(4*4*128, 128)
        self.l2 = nn.Linear(128, num_classes)

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool,
            self.conv3,
            self.relu,
            self.conv4,
            self.relu,
            self.maxpool,
            self.conv5,
            self.relu,
            self.conv6,
            self.relu,
            self.maxpool,
            )

        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.l2
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3

# 함수 정의
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

def evaluate_history(history): #학습 로그
    print(f'초기상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.5f}')
    print(f'최종상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.5f}' )

    num_epochs = len(history)
    unit = num_epochs / 10

    plt.figure(figsize=(9,8)) # 학습 곡선 출력(손실)
    plt.plot(history[:,0], history[:,1], 'b', label='훈련')
    plt.plot(history[:,0], history[:,3], 'k', label='검증')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('반복 횟수')
    plt.ylabel('손실')
    plt.title('학습 곡선(손실)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(9,8)) # 학습 곡선 출력(정확도)
    plt.plot(history[:,0], history[:,2], 'b', label='훈련')
    plt.plot(history[:,0], history[:,4], 'k', label='검증')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('반복 횟수')
    plt.ylabel('정확도')
    plt.title('학습 곡선(정확도)')
    plt.legend()
    plt.show()

def show_images_labels(loader, classes, net, device): #예측 결과 표시
    for images, labels in loader: # 데이터로더에서 처음 1세트를 가져오기
        break
    n_size = min(len(images), 50) # 표시 수는 50개

    if net is not None:
      inputs = images.to(device) # 디바이스 할당
      labels = labels.to(device)

      outputs = net(inputs) # 예측 계산
      predicted = torch.max(outputs,1)[1]
      images = images.to('cpu')

    plt.figure(figsize=(20, 15)) # 처음 n_size개 표시 (참고)
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        if net is not None: # net이 None이 아닌 경우는 예측 결과도 타이틀에 표시함
          predicted_name = classes[predicted[i]]
          if label_name == predicted_name: # 정답인지 아닌지 색으로 구분함
            c = 'k'
          else:
            c = 'b'
          ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
        else: # net이 None인 경우는 정답 라벨만 표시
          ax.set_title(label_name, fontsize=20)
        image_np = images[i].numpy().copy() # 텐서를 넘파이로 변환
        img = np.transpose(image_np, (1, 2, 0)) # 축의 순서 변경 (channel, row, column) -> (row, column, channel)
        img = (img + 1)/2 # 값의 범위를[-1, 1] -> [0, 1]로 되돌림
        plt.imshow(img)  # 결과 표시
        ax.set_axis_off()
    plt.show()

# 학습 (기본 CNN)
lr = 0.01
net = CNN_v2(n_output).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
history = np.zeros((0, 5))
num_epochs = 50
history = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history)

# 평가 (기본 CNN)
evaluate_history(history)

# 학습 (기본 CNN + 모멘텀 설정)
lr = 0.01
net = CNN_v2(n_output).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
history2 = np.zeros((0, 5))
num_epochs = 20
history2 = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history2)

# 평가 (기본 CNN + 모멘텀 설정)
evaluate_history(history2)

# 학습 (기본 CNN + Adam 함수)
lr = 0.01
net = CNN_v2(n_output).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
history3 = np.zeros((0, 5))
num_epochs = 20
history3 = fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history3)

# 평가 (기본 CNN + Adam 함수)
evaluate_history(history3)

# 최적화 함수 비교(검증 데이터의 정확도)
plt.figure(figsize=(9,8))
plt.plot(history[:,0], history[:,4], label='SGD', c='k',ls='dashed' )
plt.plot(history2[:,0], history2[:,4], label='SGD momentum=0.9', c='k')
plt.plot(history3[:,0], history3[:,4], label='Adam', c='b')
plt.title('최적화 함수　비교 결과(검증 데이터의 정확도)')
plt.xlabel('반복 횟수')
plt.ylabel('정확도')
plt.legend()
plt.show()

# 과적합 방지 CNN(Dropout,BatchNorm,데이터증강)
# 데이터 전처리(훈련데이터만 수정)
transform_train = transforms.Compose([
  transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 좌우 반전
  transforms.ToTensor(),
  transforms.Normalize(0.5, 0.5),
  transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])
train_set2 = datasets.CIFAR10(
    root = data_root, train = True,
    download = True, transform = transform_train)
batch_size = 100
train_loader2 = DataLoader(train_set2, batch_size=batch_size, shuffle=True)

# 모델 정의
class CNN_v345(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # ... (이전 conv, relu, maxpool 등 정의는 동일) ...
        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1,1))
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1,1))
        self.conv3 = nn.Conv2d(32, 64, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(64, 64, 3, padding=(1,1))
        self.conv5 = nn.Conv2d(64, 128, 3, padding=(1,1))
        self.conv6 = nn.Conv2d(128, 128, 3, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2,2))
        self.flatten = nn.Flatten()

        self.l1 = nn.Linear(4*4*128, 128)
        self.l2 = nn.Linear(128, 10)

        # 드롭아웃 레이어를 비율을 다르게 하여 3개 정의
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)

        # 각 Conv 레이어의 출력 채널 수에 맞춰 BatchNorm2d 레이어 정의
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.features = nn.Sequential(
            self.conv1, self.bn1, self.relu, # conv1 -> bn1 -> relu
            self.conv2, self.bn2, self.relu, self.maxpool, # conv2 -> bn2 -> relu
            self.dropout1, # 첫 번째 MaxPool 뒤에 추가
            self.conv3, self.bn3, self.relu, # conv3 -> bn3 -> relu
            self.conv4, self.bn4, self.relu, self.maxpool, # conv4 -> bn4 -> relu
            self.dropout2, # 두 번째 MaxPool 뒤에 추가
            self.conv5, self.bn5, self.relu, # conv5 -> bn5 -> relu
            self.conv6, self.bn6, self.relu, self.maxpool, # conv6 -> bn6 -> relu
            self.dropout3, # 세 번째 MaxPool 뒤에 추가
        )
        self.classifier = nn.Sequential(
            self.l1, 
            self.relu,
            self.dropout3, # 분류기의 선형 레이어 사이에 추가
            self.l2
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3
    
# 학습
net = CNN_v345(n_output).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
history4 = np.zeros((0, 5))
num_epochs = 50
history4 = fit(net, optimizer, criterion, num_epochs, train_loader2, test_loader, device, history4) # train_loader2로 변경

# 평가
evaluate_history(history4)