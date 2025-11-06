import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import warnings
warnings.simplefilter('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 데이터 전처리

def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

transform1 = transforms.Compose([ # transform1: 1계 텐서화 (전결합형 신경망용)
    transforms.ToTensor(),                   # 데이터를 PyTorch 텐서로 변환
    transforms.Normalize(0.5, 0.5),          # 데이터를 -1 ~ 1 범위로 정규화
    transforms.Lambda(lambda x: x.view(-1)), # 텐서를 1차원으로 평탄화
])

transform2 = transforms.Compose([ # transform2: 정규화만 실시 (CNN용)
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

data_root = './data'

train_set1 = datasets.CIFAR10( # 훈련 데이터셋 (1계 텐서 버전)
    root = data_root, train = True,
    download = True, transform = transform1)

test_set1 = datasets.CIFAR10( # 검증 데이터셋 (1계 텐서 버전)
    root = data_root, train = False,
    download = True, transform = transform1)

train_set2 = datasets.CIFAR10( # 훈련 데이터셋 (3계 텐서 버전)
    root =  data_root, train = True,
    download = True, transform = transform2)

test_set2 = datasets.CIFAR10( # 검증 데이터셋 (3계 텐서 버전)
    root = data_root, train = False,
    download = True, transform = transform2)

image1, label1 = train_set1[0]
image2, label2 = train_set2[0]

batch_size = 100
train_loader1 = DataLoader(train_set1, batch_size=batch_size, shuffle=True) # 훈련용 데이터로더
test_loader1 = DataLoader(test_set1,  batch_size=batch_size, shuffle=False) # 검증용 데이터로더
train_loader2 = DataLoader(train_set2, batch_size=batch_size, shuffle=True) # 훈련용 데이터로더
test_loader2 = DataLoader(test_set2,  batch_size=batch_size, shuffle=False) # 검증용 데이터로더

# 전결합형
# 모델 정의
class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        # 은닉층 정의(은닉층의 노드수 : n_hidden)
        self.l1 = nn.Linear(n_input, n_hidden)

        # 출력층의 정의
        self.l2 = nn.Linear(n_hidden, n_output)

        # ReLU 함수 정의
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
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

# 학습 

for images1, labels1 in train_loader1: # 첫 번째 배치만 확인
    break 

n_input = image1.view(-1).shape[0] #3*32*32=3072
n_output = len(set(list(labels1.data.numpy()))) # 10
n_hidden = 128

net = Net(n_input, n_output, n_hidden).to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)
num_epochs = 50
history = np.zeros((0, 5))
history = fit(net, optimizer, criterion, num_epochs, train_loader1, test_loader1, device, history)

# 평가
evaluate_history(history)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

show_images_labels(test_loader1, classes, net, device) #(참고)

# CNN
# 모델 정의
class CNN(nn.Module):
    def __init__(self, n_output, n_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3) # 입력 채널:3, 출력 채널:32, 커널:3x3
        self.conv2 = nn.Conv2d(32, 32, 3) # 입력 채널:32, 출력 채널:32, 커널:3x3
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2,2))
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(6272, n_hidden) # 입력:6272, 출력:은닉노드 수
        self.l2 = nn.Linear(n_hidden, n_output) # 입력:은닉노드 수, 출력:클래스 수

        self.features = nn.Sequential(  # 역할별로 그룹화
            self.conv1, self.relu, self.conv2, self.relu, self.maxpool
        )
        self.classifier = nn.Sequential(self.l1, self.relu, self.l2)

    def forward(self, x): # '조립 라인': 데이터 흐름 정의
        x1 = self.features(x)   # 특징 추출
        x2 = self.flatten(x1)   # 1차원으로 펼치기
        x3 = self.classifier(x2) # 최종 분류
        return x3
    
# 함수 정의 : 전결합층과 동일
# 학습
for images2, labels2 in train_loader2: # 첫 번째 배치만 확인
    break 

n_input = image2.view(-1).shape[0] #3*32*32=3072, 사용x
n_output = len(set(list(labels1.data.numpy()))) # 10
n_hidden = 128

torch_seed() # 난수 초기화

net = CNN(n_output, n_hidden).to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)
num_epochs = 50
history2 = np.zeros((0, 5)) 

# 학습
history2 = fit(net, optimizer, criterion, num_epochs, train_loader2, test_loader2, device, history2)

# 평가
evaluate_history(history2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

show_images_labels(test_loader2, classes, net, device) #(참고)