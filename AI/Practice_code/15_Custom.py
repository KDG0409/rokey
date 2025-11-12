import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import tensor
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

# w = !wget -nc https://download.pytorch.org/tutorial/hymenoptera_data.zip
# w = !unzip -o hymenoptera_data.zip

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

# 데이터 전처리 

# 검증 데이터: 크기 조절, 중앙 잘라내기, 텐서 변환, 정규화
test_transform =  transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
              ])
# 훈련 데이터 : 데이터 증강(Data Augmentation) 기법 추가
train_transform = transforms.Compose([
                  transforms.RandomResizedCrop(224),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                  transforms.RandomErasing(p=0.5, scale = (0.02, 0.33),
                                           ratio=(0.3, 3.3), value=0, inplace=False),

              ])

# 베이스 디렉터리
data_dir = 'hymenoptera_data'

# 훈련 데이터 디렉터리와 검증 데이터 디렉터리 지정
import os
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'val')

# join 함수 결과 확인
print(train_dir, test_dir)

# 분류하려는 클래스의 리스트 작성
classes = ['ants', 'bees']

# 훈련용 (데이터 증강 적용)
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
# 훈련용 (데이터 증강 미적용)
train_data2 = datasets.ImageFolder(train_dir, transform=test_transform)
test_data = datasets.ImageFolder(test_dir, transform=test_transform)

# 데이터 확인
plt.figure(figsize=(15, 4))
for i in range(10):
    # 처음 10개
    ax = plt.subplot(2, 10, i + 1)
    image, label = test_data[i]
    img = (np.transpose(image.numpy(), (1, 2, 0)) + 1) / 2
    plt.imshow(img)
    ax.set_title(classes[label])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 마지막 10개
    ax = plt.subplot(2, 10, i + 11)
    image, label = test_data[-i - 1]
    img = (np.transpose(image.numpy(), (1, 2, 0)) + 1) / 2
    plt.imshow(img)
    ax.set_title(classes[label])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

batch_size = 10

# 훈련용
train_loader = DataLoader(train_data,
           batch_size=batch_size, shuffle=True)
# 검증용
test_loader = DataLoader(test_data,
           batch_size=batch_size, shuffle=False)

# 이미지 출력용
train_loader2 = DataLoader(train_data2,
           batch_size=50, shuffle=True)
test_loader2 = DataLoader(test_data,
           batch_size=50, shuffle=False)

show_images_labels(test_loader2, classes, None, None)

# 모델 학습 Fine Tuning (파인튜닝)

from torchvision import models
net = models.vgg19_bn(pretrained = True)

# 난수 고정
# torch_seed()

in_features = net.classifier[6].in_features # 이진 분류기로 교체
net.classifier[6] = nn.Linear(in_features, 2)

net.avgpool = nn.Identity() # AdaptiveAverage2d 함수 제거

net = net.to(device) # GPU 사용

lr = 0.001 # 학습률

criterion = nn.CrossEntropyLoss() # 손실함수

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9) # 최적화 함수

history = np.zeros((0,5)) # history 파일도 동시에 초기화

# 결과 확인

num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs,
          train_loader, test_loader, device, history)
evaluate_history(history)
# torch_seed()
show_images_labels(test_loader2, classes, net, device)

# 모델 학습 : 전이 학습 (Transfer Learning)

# VGG-19-BN 모델을 학습이 끝난 파라미터와 함께 불러오기
from torchvision import models
net = models.vgg19_bn(pretrained = True)

# [변경점 1] 모든 파라미터의 경사 계산을 OFF로 설정 (가중치 동결)
for param in net.parameters():
    param.requires_grad = False

# 난수 고정
# torch_seed()

# 최종 노드의 출력을 2로 변경
# 이 노드에 대해서만 경사 계산을 수행하게 됨
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 2)

# AdaptiveAvgPool2d 함수 제거
net.avgpool = nn.Identity()

# GPU 사용
net = net.to(device)

# 학습률
lr = 0.001

# 손실 함수로 교차 엔트로피 사용
criterion = nn.CrossEntropyLoss()

# 최적화 함수 정의
# [변경점 2] 최적화 함수에 수정할 파라미터를 최종 노드로 제한
optimizer = optim.SGD(net.classifier[6].parameters(),lr=lr,momentum=0.9)

# history 파일도 동시에 초기화
history = np.zeros((0, 5))

# 결과 확인

num_epochs = 5
history = fit(net, optimizer, criterion, num_epochs,
          train_loader, test_loader, device, history)
evaluate_history(history)
# torch_seed()
show_images_labels(test_loader2, classes, net, device)

# 사용자 정의 데이터 활용 : 시베리안 허스키/늑대 구분

# 데이터 다운로드
# w = !wget https://github.com/makaishi2/pythonlibs/raw/main/images/dog_wolf.zip
# print(w[-2])
# 압축 해제
# !unzip dog_wolf.zip | tail -n 1
# 트리 구조 확인
# !tree dog_wolf

# 데이터 전처리
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    transforms.Normalize(0.5, 0.5)
])
data_dir = 'dog_wolf'

import os
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

classes = ['dog', 'wolf']

train_data = datasets.ImageFolder(train_dir, #증강 존재
            transform=train_transform)
train_data2 = datasets.ImageFolder(train_dir, #증강 미존재
            transform=test_transform)
test_data = datasets.ImageFolder(test_dir, # 검증 데이터(증강x)
            transform=test_transform)

batch_size = 5
# 훈련 데이터
train_loader = DataLoader(train_data,
            batch_size=batch_size, shuffle=True)
# 훈련 데이터, 이미지 출력용
train_loader2 = DataLoader(train_data2,
            batch_size=40, shuffle=False)
# 검증 데이터
test_loader = DataLoader(test_data,
            batch_size=batch_size, shuffle=False)
# 검증데이터, 이미지 출력용
test_loader2 = DataLoader(test_data,
            batch_size=10, shuffle=True)

show_images_labels(train_loader2, classes, None, None)
show_images_labels(test_loader2, classes, None, None)

# 모델 학습
# 사전 학습 모델 불러오기
net = models.vgg19_bn(pretrained = True)

for param in net.parameters():
    param.requires_grad = False

# 난수 고정
# torch_seed()

# 마지막 노드 출력을 2로 변경
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 2)

# AdaptiveAvgPool2d 함수 제거
net.avgpool = nn.Identity()

# GPU 사용
net = net.to(device)

# 학습률
lr = 0.001

# 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# 최적화 함수 정의
# 파라미터 수정 대상을 최종 노드로 제한
optimizer = optim.SGD(net.classifier[6].parameters(),lr=lr,momentum=0.9)

# history 파일도 동시에 초기화
history = np.zeros((0, 5))

num_epochs = 10
history = fit(net, optimizer, criterion, num_epochs,
          train_loader, test_loader, device, history)

evaluate_history(history)
# torch_seed()
show_images_labels(test_loader2, classes, net, device)