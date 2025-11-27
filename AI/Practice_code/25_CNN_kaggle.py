import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import os
import time
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

# 함수형 신경망 연산(F) 모듈을 불러옴. 활성화 함수나 풀링 등에 사용함.
import torch.nn.functional as F
import torchvision.transforms as transforms

torch.manual_seed(0)

# LeNet

# LeNet 클래스 정의

class LeNet(nn.Module):
    def __init__(self): # 클래스의 인스턴스를 초기화함. 레이어들을 정의함.
        super(LeNet, self).__init__() # 부모 클래스(nn.Module)의 생성자를 호출함.
        # 입력 채널 3개(RGB 이미지), 출력 특징 맵 6개, 커널 크기 5x5를 사용함.
        self.cn1 = nn.Conv2d(3, 6, 5)
        # 입력 채널 6개, 출력 특징 맵 16개, 커널 크기 5x5를 사용함.
        self.cn2 = nn.Conv2d(6, 16, 5)

        # 입력 크기는 16 * 5 * 5 (이전 레이어의 출력 특징 맵 수 * 공간 크기), 출력 크기는 120임.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 입력 120, 출력 84를 사용함.
        self.fc2 = nn.Linear(120, 84)
        # 입력 84, 최종 출력 클래스 수 10개를 사용함.
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x): # F : 정의 없이 바로 사용
        # cn1을 적용하고 ReLU 활성화 함수를 통과시킴.
        x = F.relu(self.cn1(x))
        # 2x2 크기의 맥스 풀링을 적용함. 공간 크기를 절반으로 줄임.
        x = F.max_pool2d(x, (2, 2))
        # cn2를 적용하고 ReLU 활성화 함수를 통과시킴.
        x = F.relu(self.cn2(x))
        # 2x2 크기의 맥스 풀링을 다시 적용함.
        x = F.max_pool2d(x, (2, 2))
        # 데이터를 평탄화(flatten)함. 배치 차원(-1)을 제외한 모든 차원을 하나의 벡터로 만듦.
        x = x.view(-1, self.flattened_features(x))
        # 첫 번째 완전 연결 레이어와 ReLU를 통과시킴.
        x = F.relu(self.fc1(x))
        # 두 번째 완전 연결 레이어와 ReLU를 통과시킴.
        x = F.relu(self.fc2(x))
        # 최종 출력 레이어를 통과시킴.
        x = self.fc3(x)

        # 최종 결과를 반환함.
        return x

    # 데이터를 평탄화하기 위해 특징들의 총 개수를 계산하는 헬퍼 함수임.
    def flattened_features(self, x):
        size = x.size()[1:] # 배치차원 제외
        num_feats = 1
        # 모든 차원 크기를 곱하여 총 특징 개수를 계산함.
        for s in size:
            num_feats *= s
        # 총 특징 개수를 반환함.
        return num_feats

# LeNet 클래스의 인스턴스를 생성함.
lenet = LeNet()

# AlexNet (colab에서 kaggle data 다운& API설정 후 사용)

# 데이터 전처리
# from google.colab import files
# files.upload()
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/         # api key 인증 키 이동
# !chmod 600 ~/.kaggle/kaggle.json   # 파일접근 권한(본인만 읽고(4) 쓰기(2) 가능하게 권한위임)
# !kaggle datasets download -d ajayrana/hymenoptera-data #데이터 다운
# !unzip -q hymenoptera-data.zip -d . # 압축해제 (. : 현재 작업중인 디렉토리)
ddir = 'hymenoptera_data'
data_transformers = {
    'train': transforms.Compose([
        # 데이터증강 (학습데이터만 적용)
        transforms.RandomResizedCrop(224),
        # 이미지를 무작위로 자르고 크기를 224*224 조정
        transforms.RandomHorizontalFlip(),
        # 이미지를 무작위 수평(좌우반전) 뒤집음
        transforms.ToTensor(),
        # 파이토치 텐서로 변환
        transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221,0.230])]
        # R,G,B 채널 별 평균과 표준편차 사용 >> 텐서를 정규화함
    ),
    'val': transforms.Compose([
        # 데이터증강 (학습데이터만 적용)
        transforms.Resize(256),
        # 이미지를 256*256 조정
        transforms.CenterCrop(224),
        # 이미지 중앙을 224*224 크기로 자름
        transforms.ToTensor(),
        # 파이토치 텐서로 변환
        transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221,0.230])]
        # R,G,B 채널 별 평균과 표준편차 사용 >> 텐서를 정규화함
    )

}
# ImageFolder 데이터 셋 활용, DataLoader 객체 생성
# batch_size = 8. 데이터 섞기 shuffle 활성화, 작업자 수(num_workers = 2) 설정
img_data = {k: datasets.ImageFolder(os.path.join(ddir, k), data_transformers[k]) for k in {'train', 'val'}}
dloaders = {'train': torch.utils.data.DataLoader(img_data['train'], batch_size=8, shuffle=True, num_workers=2),
             'val': torch.utils.data.DataLoader(img_data['val'], batch_size=8, shuffle=False, num_workers=2)}
dset_sizes = {x: len(img_data[x]) for x in {'train', 'val'}}
classes = img_data['train'].classes
dvc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 이미지를 화면에 표시하는 함수 정의 (정규화된 이미지를 역변환함)

def imageshow(img, text=None):
   img = img.numpy().transpose((1,2,0))
   # torch tensor data를 numpy 배열로 변환
   # tensor (chw : 채널, 높이, 너비) >> (높이, 너비, 채널) 전치함.
   # 정규화에 사용했던 R,G,B 채널별 평균(mean) 정의
   avg = np.array([0.490,0.449,0.411])
   # 정규화에 사용했던 R,G,B 채널별 표준편차(stddev) 정의
   stddev = np.array([0.231,0.221,0.230])
   # 역정규화 (denormalization) 수행 : img =  stddev * img + avg
   img =  stddev * img + avg
   # 픽셀 값이 [0,1] 범위를 벗어나는 생길 경우 대비 >> 해당 범위 내로 clip 함
   img = np.clip(img, 0,1)
   plt.imshow(img) # 이미지 표시
   plt.axis('off')
   # 텍스트(제목) 제공되는 경우, 이미지 제목으로 설정하고 싶어
   if text is not None:
      plt.title()

# 학습 데이터로더('train') 이터레이터, 넥스트 가져옴
d_iter = iter(dloaders['train'])
# 이터레이터에서 다음 미니배치(이미지 텐서와 클래스 레이블) 가져옴
imgs, cls = next(d_iter)
# 미니 배치 이미지들을 하나의 격자(grid) 이미지로 만들어 표현
torchvision.utils.make_grid(imgs)
# 미니 배치 이미지들을 하나의 격자(grid) 이미지로 만들어 표현
grid = torchvision.utils.make_grid(imgs)
# 격자 이미지와 해당 레이블(cls) 제목 설정 >> 화면에 표시
imageshow(grid, text=[classes[x] for x in cls])

# 전이학습(transfer learning) 함수 정의
def finetune_model(pretrained_model, loss_func, optim, epochs=10):
  # 학습시간 기록
  start = time.time()

  # 현재 모델의 가중치(state_dict)를 깊은 복사 > 초기상태 저장
  model_weights = copy.deepcopy(pretrained_model.state_dict())

  # 검증 정확도 추적을 위한 변수를 0.0 초기화
  accuracy = 0.0

  # 지정된 epochs 수 만큼 반복하여 학습을 진행함.
  for e in range(epochs):
    # 현재 에폭 진행 상황 출력
    print(f'epoch_number {e}' / {epochs-1})
    print('='*20)

    # 현재 에폭 내에서 학습 데이터 셋과 검증 데이터 셋 순환
    for dset in ['train', 'val']:
      if dset == 'train':
        pretrained_model.train()
        # 모델을 학습 모드로 설정(예: Dropout, BatchNorm 활성화)

      else:
        pretrained_model.eval()
        # 평가 모드 (예: Dropout, BatchNorm 비활성화)(**)

      # 에폭 별 손실과 성공횟수 0.0 초기화
      loss = 0.0
      successes = 0

      # 학습 또는 검증 데이터 로더 순회
      for imgs, tgts in dloaders[dset]:
          # 입력 이미지, 정답 레이블 >> 설정된 device로 이동
          imgs = imgs.to(dvc)
          tgts = tgts.to(dvc)

          optim.zero_grad()

          # 학습 모드('train') 에서만 gradient 변화도 계산을 활성화함
          with torch.set_grad_enabled(dset == 'train'):
              ops = pretrained_model(imgs)
              # 순전파 수행, 예측 결과(ops)얻음
              _, preds = torch.max(ops, 1)
              # 예측 결과에서 모델이 예측한 클래스(preds) 찾음(최대값이 있던 위치 indices)
              # _ : 최대값(value) : 우리 안 쓸거야(필요없어)
              loss_curr = loss_func(ops, tgts)
              # 현재 미니배치에 대한 손실 계산

              # 학습 모드('train')인 경우에만 역전파, 가중치 업데이트 수행함
              if dset == 'train':
                loss_curr.backward()
                optim.step()

          # 배치 손실을 전체 에폭 손실에 누적
          # >> 이미지 개수를 곱해서 평균 손실이 아닌 총 손실을 누적함
          loss += loss_curr.item() * imgs.size(0)
          # loss_curr : 현재 미니배치의 평균 loss 값
          # .item() >> 파이선 숫자(float)
          # imgs.size(0) : batch_size (현재 배치 내 이미지 개)

          # 예측과 정답과 일치하는 개수 세어서 성공 횟수를 누적함
          successes += torch.sum(preds == tgts.data)

      # 에폭이 끝난 후, 전체 손실을 데이터 셋 나누어서 평균 에폭 손실 계산
      loss_epoch = loss / dset_sizes[dset]
      # dset_sizes[dset] 데이터 셋(dset)의 전체 크기(총 샘풀 수)
      # 전체 성공횟수를 데이터 셋 나누어서 에폭 정확도를 계산함
      accuracy_epoch = successes.double() / dset_sizes[dset]
      # .double() 텐서의 데이터 타입을 부동소수점

      print(f'{dset} loss in this epoch: {loss_epoch}, accuracy in this epoch: {accuracy_epoch}')

      # 현재 검증 정확도가 지금까지 최고 정확도 보다 높으면
      if dset == 'val' and accuracy_epoch > accuracy:
          accuracy = accuracy_epoch
          model_weights = copy.deepcopy(pretrained_model.state_dict())
      print()

  # 학습 종료시간 계산 >> 총 소요시간 출력
  time_delta = time.time() - start
  print(f'Training fished in {time_delta // 60}mins {time_delta % 60}secs')
  print(f'Best accuracy: {accuracy}')

  # 최고 성능 보였던 시점의 모델 가중치(model_weights) 를 모델에 로드함
  pretrained_model.load_state_dict(model_weights)

  return pretrained_model

# 모델의 예측결과 시각화 하는 함수 정의
# pretrained_model : 사전학습된 모델, max_num_imgs : 표시할 최대 이미지 수 입력받음

def visualize_predictions(pretrained_model, max_num_imgs=4):
    torch.manual_seed(1) # 난수 생성기 seed 설정

    # 모델의 원래 학습 모드 상태 (True/False) 저장
    was_model_training = pretrained_model.training
    # 모델이 현재 train() 상태인지 eval() 상태인지 기록
    # >> 함수 종료된 뒤에 원래 상태로 복구하기 위해서

    # 모델을 평가 모드로 설정
    pretrained_model.eval()

    # 시각화할 이미지 카운터 0으로 초기화
    imgs_counter = 0

    # 그림 객체 생성
    fig = plt.figure()

    # gradient 계산 비 활성화
    with torch.no_grad():
      # 검증 데이터 로더('val') 순회
      for i, (imgs, tgts) in enumerate(dloaders['val']):
        # 입력 이미지, 정답 레이블 >> 설정된 device로 이동
        imgs = imgs.to(dvc)
        tgts = tgts.to(dvc)

        ops = pretrained_model(imgs)
        _, preds = torch.max(ops,1)

        # 현재 배치 내에서 모든 이미지에 대해 순회
        for j in range(imgs.size()[0]):
          imgs_counter += 1

          ax = plt.subplot(max_num_imgs//2, 2, imgs_counter)
          # (default) max_num_imgs=4 >> (2,2)
          ax.axis('off')

          ax.set_title(f'pred: {classes[preds[j]]} || target: {classes[tgts[j]]}')

          # 역정규화된 이미지를 화면에 표시
          imageshow(imgs.cpu().data[j])

          # 설정된 최대 이미지 수에 도달한다면
          if imgs_counter == max_num_imgs:
             pretrained_model.train(mode=was_model_training)
             #  모델의 모드를 원래 상태로 되돌려라
             return
             # 함수 실행 종료
      pretrained_model.train(mode=was_model_training)
      #  loop를 끝까지 실행했다면 모델의 모드를 원래 상태로 돌려놓아요

# torchvision.models 에서 alexnet 모델 호출
model_finetuned = models.alexnet(pretrained=True)
model_finetuned.classifier[6] = nn.Linear(4096, 2)
loss_func = nn.CrossEntropyLoss()
optim_finetune = optim.SGD(model_finetuned.parameters(), lr=0.0001)
model_finetuned = model_finetuned.to(dvc)
model_finetune = finetune_model(model_finetuned, loss_func, optim_finetune)

visualize_predictions(model_finetune)

# vgg13

# from google.colab import files
# files.upload()
# !pip install kaggle

# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# !kaggle datasets download -d ajayrana/hymenoptera-data
# !unzip -q hymenoptera-data.zip -d .

# 데이터 전처리
ddir = 'hymenoptera_data'
data_transformers = {
    'train': transforms.Compose([
        # 데이터증강 (학습데이터만 적용)
        transforms.RandomResizedCrop(224),
        # 이미지를 무작위로 자르고 크기를 224*224 조정
        transforms.RandomHorizontalFlip(),
        # 이미지를 무작위 수평(좌우반전) 뒤집음

        transforms.ToTensor(),
        # 파이토치 텐서로 변환
        transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221,0.230])]
        # R,G,B 채널 별 평균과 표준편차 사용 >> 텐서를 정규화함
    ),
    'val': transforms.Compose([
        # 데이터증강 (학습데이터만 적용)
        transforms.Resize(256),
        # 이미지를 256*256 조정
        transforms.CenterCrop(224),
        # 이미지 중앙을 224*224 크기로 자름

        transforms.ToTensor(),
        # 파이토치 텐서로 변환
        transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221,0.230])]
        # R,G,B 채널 별 평균과 표준편차 사용 >> 텐서를 정규화함
    )

}
img_data = {k: datasets.ImageFolder(os.path.join(ddir, k), data_transformers[k]) for k in {'train', 'val'}}
dloaders = {'train': torch.utils.data.DataLoader(img_data['train'], batch_size=8, shuffle=True, num_workers=2),
             'val': torch.utils.data.DataLoader(img_data['val'], batch_size=8, shuffle=False, num_workers=2)}
dset_sizes = {x: len(img_data[x]) for x in {'train', 'val'}}
classes = img_data['train'].classes
dvc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import ast # 문자열 객체를 파이썬 문자로 변경
with open('/content/imagenet1000_clsidx_to_labels.txt') as f:
    classes_data = f.read()
classes_dict = ast.literal_eval(classes_data)

# 이미지를 화면에 표시하는 함수를 정의함. (정규화된 이미지를 역변환함)
def imageshow(img, text=None):
    # PyTorch 텐서를 NumPy 배열로 변환하고 채널 순서를 Matplotlib 형식으로 전치함.
    img = img.numpy().transpose((1, 2, 0))

    # 정규화에 사용했던 R, G, B 채널별 평균(mean)을 정의함.
    avg = np.array([0.490, 0.449, 0.411])

    # 정규화에 사용했던 R, G, B 채널별 표준편차(stddev)를 정의함.
    stddev = np.array([0.231, 0.221, 0.230])

    # 역정규화(denormalization)를 수행함.
    img = stddev * img + avg

    # 픽셀 값을 [0, 1] 범위 내로 클립(clip)함.
    img = np.clip(img, 0, 1)

    # Matplotlib을 사용하여 이미지를 화면에 표시함.
    plt.imshow(img)

    # 텍스트가 제공된 경우 이미지의 제목으로 설정함.
    if text is not None:
        plt.title(text)

# 학습된 모델의 예측 결과를 시각화하는 함수를 정의함.
def visualize_predictions(pretrained_model, max_num_imgs=4):
    # 난수 생성기 시드를 설정했음.
    torch.manual_seed(1)

    # 모델의 원래 학습 모드 상태를 저장함.
    was_model_training = pretrained_model.training

    # 모델을 평가 모드로 설정함.
    pretrained_model.eval()

    # 시각화할 이미지 카운터를 0으로 초기화함.
    imgs_counter = 0

    # Matplotlib 그림 객체(figure)를 생성함.
    fig = plt.figure()

    # 변화도(gradient) 계산을 비활성화함.
    with torch.no_grad():
        # 검증 데이터 로더('val')를 순회함.
        for i, (imgs, tgts) in enumerate(dloaders['val']):
            # 이미지와 정답 레이블을 장치(GPU/CPU)로 이동시킴. (이 부분에서 tgts가 사용되지 않고 있음)
            imgs = imgs.to(dvc)
            # tgts = tgts.to(dvc) # 주석: tgts는 현재 사용되지 않으나, GPU로 이동시키는 것이 일반적임.

            # 순전파를 수행하여 예측 결과(ops)를 얻음.
            ops = pretrained_model(imgs)

            # 예측 결과에서 예측 클래스(preds)를 찾음.
            _, preds = torch.max(ops, 1)

            # 현재 배치 내의 모든 이미지에 대해 순회함.
            for j in range(imgs.size()[0]):
                # 이미지 카운터를 증가시킴.
                imgs_counter += 1

                # 서브플롯 위치를 설정함.
                ax = plt.subplot(max_num_imgs//2, 2, imgs_counter)

                # 축(axis) 표시를 끔.
                ax.axis('off')

                # 예측 클래스 이름만 제목으로 설정함. (원래 정답 레이블(target)도 포함했으나 지금은 예측만 표시함)
                # classes_dict는 이전에 정의된 classes 변수와 동일할 것으로 예상함.
                ax.set_title(f'pred: {classes_dict[int(preds[j])]}')

                # 역정규화된 이미지를 화면에 표시함.
                imageshow(imgs.cpu().data[j])

                # 설정된 최대 이미지 수에 도달했다면
                if imgs_counter == max_num_imgs:
                    # 모델의 모드를 원래 상태로 되돌림.
                    pretrained_model.train(mode=was_model_training)
                    # 함수 실행을 종료함.
                    return
        # 루프를 끝까지 실행했다면 모델의 모드를 원래 상태로 되돌림.
        pretrained_model.train(mode=was_model_training)

model = models.vgg13(pretrained=True)
visualize_predictions(model)

# GoogleNet
# 1*1 : 연산량 축소 (세밀한 특징 추출) : 눈, 코, 입 high-level : 병렬 처리 (인셉션: 동시처리)
# 3*3 : 중간 특징 (얼굴 부분) mid-level
# 5*5 : 큰 특징(전체 얼굴) low level
# 계산 효율성
# 일반 55 합성곱 : 5*5*512*512 = 6,553,600 연산
# 인셉션 (1*1 >> 5*5) 1*1*512*24 + 5*5*24*64 = 50,688 연산

# InceptionModule 클래스를 정의함. nn.Module을 상속받아 PyTorch 모듈로 만듦.
class InceptionModule(nn.Module):
    # 인셉션 모듈을 초기화함. 입력 채널 수(input_planes)와 각 브랜치(branch)의 채널 수를 인수로 받음.
    def __init__(self, input_planes, n_channels1x1, n_channels3x3red, n_channels3x3, n_channels5x5red, n_channels5x5, pooling_planes):
        # 부모 클래스(nn.Module)의 생성자를 호출함.
        super(InceptionModule, self).__init__()

        # 첫 번째 브랜치: 1x1 합성곱만 수행함.
        # 빠른 처리, 세부 특징 추출
        self.block1 = nn.Sequential(
            # 1x1 커널 크기의 합성곱 레이어를 정의함.
            nn.Conv2d(input_planes, n_channels1x1, kernel_size=1),
            # 배치 정규화(BatchNorm)를 적용함.
            nn.BatchNorm2d(n_channels1x1),
            # ReLU 활성화 함수를 적용함.
            nn.ReLU(True),
        )

        # 두 번째 브랜치: 1x1 합성곱(채널 축소) -> 3x3 합성곱을 수행함.
        # 중간 크기 패턴 감지
        self.block2 = nn.Sequential(
            # 1x1 합성곱으로 채널을 축소함. (n_channels3x3red)
            nn.Conv2d(input_planes, n_channels3x3red, kernel_size=1),
            nn.BatchNorm2d(n_channels3x3red),
            nn.ReLU(True),
            # 3x3 합성곱을 적용함. padding=1로 출력 해상도를 유지함.
            nn.Conv2d(n_channels3x3red, n_channels3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels3x3),
            nn.ReLU(True),
        )

        # 세 번째 브랜치: 1x1 합성곱(채널 축소) -> 5x5와 동등한 3x3 합성곱 두 번을 수행함.
        self.block3 = nn.Sequential(
            # 1x1 합성곱으로 채널을 축소함. (n_channels5x5red)
            nn.Conv2d(input_planes, n_channels5x5red, kernel_size=1),
            nn.BatchNorm2d(n_channels5x5red),
            nn.ReLU(True),
            # 첫 번째 3x3 합성곱을 적용함.
            nn.Conv2d(n_channels5x5red, n_channels5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels5x5),
            nn.ReLU(True),
            # 두 번째 3x3 합성곱을 적용함. 5x5 합성곱과 유사한 효과를 냄.
            nn.Conv2d(n_channels5x5, n_channels5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels5x5),
            nn.ReLU(True),
        )

        # 네 번째 브랜치: 3x3 맥스 풀링 -> 1x1 합성곱(채널 축소)을 수행함.
        self.block4 = nn.Sequential(
            # 3x3 맥스 풀링을 적용함. stride=1, padding=1로 해상도를 유지함.
            nn.MaxPool2d(3, stride=1, padding=1),
            # 1x1 합성곱으로 채널을 축소함. (pooling_planes)
            nn.Conv2d(input_planes, pooling_planes, kernel_size=1),
            nn.BatchNorm2d(pooling_planes),
            nn.ReLU(True),
        )

    # 데이터가 모듈을 통과하는 순서를 정의함.
    def forward(self, ip):
        # 각 브랜치(block)에 입력(ip)을 통과시켜 출력을 얻음.
        op1 = self.block1(ip)
        op2 = self.block2(ip)
        op3 = self.block3(ip)
        op4 = self.block4(ip)

        # 네 브랜치의 출력을 채널 차원(1번 차원)을 따라 하나로 병합(concatenate)하여 반환함.
        return torch.cat([op1,op2,op3,op4], 1)
    
# GoogLeNet 클래스를 정의함. nn.Module을 상속받아 전체 신경망 구조를 만듦.
class GoogLeNet(nn.Module):
    # 모델의 레이어들을 정의하여 초기화함.
    def __init__(self):
        # 부모 클래스(nn.Module)의 생성자를 호출했음.
        super(GoogLeNet, self).__init__()
        # 초기 특징 추출부(stem)를 정의함. 3x3 합성곱, 배치 정규화, ReLU로 구성됨.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        # 첫 번째 Inception 모듈(im1)을 정의함. 입력 192 채널을 받음.
        self.im1 = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        # 두 번째 Inception 모듈(im2)을 정의함. 입력 256 채널을 받음.
        self.im2 = InceptionModule(256, 128, 128, 192, 32, 96, 64)

        # 해상도 감소를 위한 맥스 풀링 레이어를 정의했음.
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        # 세 번째에서 일곱 번째 Inception 모듈(im3 ~ im7)을 정의함.
        self.im3 = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.im4 = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.im5 = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.im6 = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.im7 = InceptionModule(528, 256, 160, 320, 32, 128, 128)

        # 여덟 번째와 아홉 번째 Inception 모듈(im8, im9)을 정의함.
        self.im8 = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.im9 = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # 최종 특징 추출을 위한 평균 풀링 레이어를 정의했음.
        self.average_pool = nn.AvgPool2d(7, stride=1)
        # 최종 분류를 위한 완전 연결 레이어(1000개 클래스)를 정의함.
        self.fc = nn.Linear(4096, 1000)

    # 데이터의 순전파 경로를 정의함.
    def forward(self, ip):
        # 입력(ip)을 stem 모듈에 통과시켜 op를 얻음.
        op = self.stem(ip)
        # im1 모듈을 통과시켜 out에 저장함. (이후 op 변수가 다시 사용됨)
        op = self.im1(op)
        # im2 모듈을 통과시켜 out에 덮어씀.
        op = self.im2(op)
        # maxpool 레이어를 사용하여 op를 처리함. (주의: self.max_pool 대신 maxpool 변수가 사용됨)
        op = self.maxpool(op)
        # a4 레이어를 통해 op를 처리함. (주의: init에 정의되지 않은 변수임)
        op = self.a4(op)
        # b4 레이어를 통해 op를 처리함. (주의: init에 정의되지 않은 변수임)
        op = self.b4(op)
        # c4 레이어를 통해 op를 처리함. (주의: init에 정의되지 않은 변수임)
        op = self.c4(op)
        # d4 레이어를 통해 op를 처리함. (주의: init에 정의되지 않은 변수임)
        op = self.d4(op)
        # e4 레이어를 통해 op를 처리함. (주의: init에 정의되지 않은 변수임)
        op = self.e4(op)
        # max_pool 레이어를 통과시켜 해상도를 줄였음.
        op = self.max_pool(op)
        # a5 레이어를 통해 op를 처리함. (주의: init에 정의되지 않은 변수임)
        op = self.a5(op)
        # b5 레이어를 통해 op를 처리함. (주의: init에 정의되지 않은 변수임)
        op = self.b5(op)
        # avgerage_pool 레이어를 통과시켰음. (주의: self.average_pool 대신 오타 변수가 사용됨)
        op = self.avgerage_pool(op)
        # 배치 차원을 제외하고 텐서를 평탄화(flatten)했음.
        op = op.view(op.size(0), -1)
        # 최종 완전 연결 레이어(fc)를 통과시켜 최종 출력을 얻음.
        op = self.fc(op)
        # 최종 결과를 반환함.
        return op
  
lenet = GoogLeNet()