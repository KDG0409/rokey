import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# 데이터 전처리

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=100, shuffle=False)

# 모델 빌드
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1), # 3,32,32 -> 32,32,32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # 32,16,16
            nn.Conv2d(32,64,kernel_size=3,padding=1), # 32,16,16 -> 64,16,16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2), # 64,8,8          
        )
        self.classifer = nn.Sequential(
            nn.flatten(),
            nn.Linear(64*8*8,256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifer(x)
        return x
    
model = SimpleCNN().to(device)

# 모델 학습
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

def train(model,loader,criterion,optimizer,epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() 

    print(f'[Epoch {epoch}] 평균 loss: {running_loss /len(loader)}:.4f')

# 함수 정의
def evaluate(model,loader):
    model.eval()
    correct = 0
    total = 0 
    with torch.no_grad():
        for inputs,labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            value,pred = torch.max(outputs,1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

# 모델 학습
num_epochs = 5
train_acc_list = []
test_acc_list = []
for epoch in range(1, num_epochs + 1):
  train(model, train_loader, criterion, optimizer, epoch)
  acc = evaluate(model, test_loader)
  test_acc_list.append(acc)
  print(f'테스트 정확도: {acc:.2f}%')

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(test_acc_list)+1), test_acc_list, marker='o', color='green')
plt.title("테스트 정확도 변화 추이")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.show()

# 성능 시각화
import numpy as np
classes = ('비행기', '자동차','새','고양이','사슴', '개','개구리','말','배')

dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, preds = torch.max(outputs,1)

plt.figure(figsize=(12,6))
for i in range(8):
  plt.subplot(2,4, i+1)
  img = images[i].cpu().permute(1,2,0) * 0.5 + 0.5 # 역정규화
  plt.imshow(img)
  plt.title(f"정답: {classes[labels[i]]}\n예측: {classes[preds[i]]}")
  plt.axis('off')

  plt.figure(figsize=(12,6))
  plt.tight_layout()
  plt.show()
