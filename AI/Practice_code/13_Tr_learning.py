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
from torchvision import models
from tqdm.auto import tqdm

import warnings
warnings.simplefilter('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 적응형 풀링 함수

p = nn.AdaptiveAvgPool2d((1,1)) # 출력을 (1, 1) 크기로 만드는 풀링 레이어 정의
print(p)

# 데이터 전처리

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_output = len(classes)

transform_train = transforms.Compose([ # transforms 정의
    transforms.Resize(112),                 # 이미지 크기를 112x112로 조정
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 3채널 이미지에 대한 정규화
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])
transform = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 3채널 이미지에 대한 정규화
])

data_root = './data'

train_set = datasets.CIFAR10( # 데이터 셋
    root = data_root, train = True,
    download = True, transform = transform_train)
test_set = datasets.CIFAR10(
    root = data_root, train = False,
    download = True, transform = transform)

batch_size = 50 # 데이터 로더
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

# 모델 정의
# 전이 학습

# 함수 정의 (고정)
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

# 학습 및 평가
# ResNet18
net = models.resnet18(pretrained = True)
torch.manual_seed(42)
fc_in_features = net.fc.in_features
net.fc = nn.Linear(fc_in_features, n_output)
net = net.to(device)
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
history = np.zeros((0, 5))

num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs,
        train_loader, test_loader, device, history)

evaluate_history(history)
show_images_labels(test_loader, classes, net, device)

# VGG-19-BN
net = models.vgg19_bn(pretrained = True)
torch.manual_seed(42)
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, n_output)

net.features = net.features[:-1] # features 마지막의 MaxPool2d 제거
net.avgpool = nn.Identity() # AdaptiveAvgPool2d 제거

net = net.to(device)
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
history = np.zeros((0, 5))

num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs,
        train_loader, test_loader, device, history)

evaluate_history(history)
show_images_labels(test_loader, classes, net, device)