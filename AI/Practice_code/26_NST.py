# PyTorch의 핵심 라이브러리를 불러옴.
import torch
# PyTorch의 자동 미분 기능(Autograd)을 위한 Variable 클래스를 불러옴. (최신 PyTorch에서는 텐서가 대체함)
from torch.autograd import Variable
# 신경망 레이어(nn) 모듈을 불러옴.
import torch.nn as nn
# 함수형 신경망 연산(F) 모듈을 불러옴.
import torch.nn.functional as F
# 옵티마이저(optim) 모듈을 불러옴.
from torch import optim
# TorchVision 라이브러리를 불러옴.
import torchvision
# 이미지 변환(transforms) 모듈을 불러옴.
from torchvision import transforms
# PIL 라이브러리의 Image 모듈을 불러옴. 이미지 처리에 사용함.
from PIL import Image
# 순서가 보장되는 딕셔너리(OrderedDict) 클래스를 불러옴. (특정 구조 저장에 유용함)
from collections import OrderedDict

# Gram Matrix 계산하는 파이토치 모듈 정의
class GramMatrix(nn.Module):
  # 순전파 정의
  def forward(self, input):
    b,c,h,w = input.size()
    # input : hidden layer에서 출력한 featu
    # 입력 텐서 배치(b), 채널(c), 높이(h), 너비(w) 크기 추출 (차원정보 추출)
    # (2,64,32,32) b=2, c=64, h=32, w=32
    # 배치(b) 2 : 2장의 이미지
    # 채널(c) 64 : 64개의 feature map
    # h(높이), w(너비) 픽셀
    F = input.view(b, c, h*w)
    # 4차원 >> 3차원 변경 (텐서를 배치 * 채널, 높이 * 너비) 형태로 평탄화(flatten)함
    # 이 구현에서는 배치 차원 유지, 채널차원만 평탄화함 (b,c h*w)
    # (2,64,32,32) >> (2,64,32*32) 2d 이미지를 1줄로 쭉 펴는 것
    # 배치(2) 마다(각각의 이미지마다) 64개 채널(feature map) * 1024개 위치 값
    # 채널 별(색상, 질감, ....)로 길이가 1024개 있는 벡터
    # c * (h*w) >> 공간 차원을 펼친 feature map 생성
    # c * (h*w), c * (h*w) 내적(inner product)하려면 차원을 맞추어 주기 위해 전치행렬이 필요해요
    # 3 * (5*5), 3 * (5*5), 3*25 (3*25).T >> 3*25 25*3 = 3*3 행렬 (c,c)
    # Gram Matrix  각 채널(특성) 끼리(색상, 질감..) 내적
    # channel i = [x_i1, x_i2.....], channel j = [x_j1, x_j2.....] 이 두 벡터의 내적
    # >> channel i와 channel j의 상관정도

    G = torch.bmm(F, F.transpose(1,2))
    # 배치 행렬 곱(bmm) 사용, gram matrix G 계산
    # F와 F의 채널-공간 차원 전치(transpose) 곱해줘요 >> 결과의 크기 (b, c, c)
    # 채널 1(색상), 채널 2(질감) 있다면, 채널 1과 채널 2가 얼마나 함께 활성화 되는가 계산
    # 높으면 두 특징이 자주 함께 나타난다는 의미고 낮으면 두 특징이 잘 안 나타난다는 의미
    # 채널 1.shape (b, n, m) 채널 2.shape (b, m, p)
    # >> (b, n, p) >> 배치 차원 b 개에 대해 각각 행렬 곱 수행
    # (batch_size, channel, h*w)

    G.div_(h*w)
    # gram matrix 를 높이 * 너비(h*w)로 나누어 정규화
    # 왜 해요? 값의 범위를 안정화하기 위해 (정규화 >> 안정)

    return G
  
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]
    
# gram matrix 에 대해 MSE(평균 제곱 오차) 손실 계산하는 모듈 정의
class GramMSELoss(nn.Module):
  # 순전파(forward) 정의함. input: 현재 이미지의 특징, target: 목표로 하는 이미지 gram matrix
  def forward(self, input, target):
    out = nn.MSELoss()(GramMatrix()(input),target)
    # GramMatrix 모듈 통과 > gram matrix 계산 >> 목표(target) gram matix와 비교
    # >> MSE 손실 계산
    # target 은 함수가 아니예요. (tensor) 데이터예요.
    return (out)

vgg = VGG()

# 로드할 이미지 파일 이름들을 정의함. ('vangogh_starry_night.jpg', 'Tuebingen_Neckarfront.jpg')
img1 = "/content/drive/MyDrive/두산로보틱스_딥러닝_컴퓨터비전/컴퓨터 비전 응용/강의용/6차시_신경망 스타일 전이(완료)/Tuebingen_Neckarfront.jpg"
img2 = "/content/drive/MyDrive/두산로보틱스_딥러닝_컴퓨터비전/컴퓨터 비전 응용/강의용/6차시_신경망 스타일 전이(완료)/vangogh_starry_night.jpg"


# 이미지 디렉토리와 파일 이름을 결합하여 PIL Image 객체 리스트로 로드함.
img1 = Image.open(img1)
img2 = Image.open(img2)
imgs = []
imgs.append(img1)
imgs.append(img2)

img_size = 512

prep = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(), # 텐서로 변환
    transforms.Lamda(lambda x: x[torch.LongTensor([2,1,0])]), #chw->whc
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean # 이미지넷 가중치
                                                std=[1,1,1]),
    transforms.Lambda(lambda x: x.mul_(255)), # 픽셀범위 [0,1]->[0,255] 스케일 : 이미지
])
# 각 PIL Image 객체에 사전 정의된 전처리함수(prep)를 적용 >> 텐서로 변환
imgs_torch = [prep(img) for img in imgs] # imgs 이미지 리스트 전체

# GPU(CUDA) 사용이 가능하다면: # 이전 방식
if torch.cuda.is_available():
    # 각 텐서에 배치 차원(unsqueeze(0))을 추가하고 GPU로 이동시킨 후, Variable로 감싸서 저장함.
    imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
# GPU를 사용할 수 없다면:
else:
    # 각 텐서에 배치 차원만 추가하고 Variable로 감싸서 저장함.
    imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]

# 이미지 텐서 리스트를 스타일 이미지와 콘텐츠 이미지 변수에 할당
style_image, content_image = imgs_torch
opt_img = Variable(content_image.data.clone(), requires_grad=True)
# 형태 유지하면서 스타일만 변형되도록 학습(경사하강법) 시작
# >> 이미지 자체가 학습 대상(파라미터)
# (**) 스타일 트랜스퍼(전이) CNN 가중치 학습은 하지 않아요. opt)

# opt_img = content_image.clone().detach().requires_grad_(True) 로 대체가능

# GramMSELoss와 vgg는 정의되어 있다고 가정합니다.
# from your_modules import GramMSELoss, vgg, GramMatrix

# 사용할 GPU를 설정하고, VGG 모델을 GPU로 이동합니다.
if torch.cuda.is_available():
    # VGG 모델을 GPU 메모리로 이동
    vgg = vgg.cuda()
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# 스타일 손실을 계산할 VGG 레이어 이름 정의
style_layers = ['r11','r21','r31','r41', 'r51']
# 콘텐츠 손실을 계산할 VGG 레이어 이름 정의
content_layers = ['r42']

loss_layers = style_layers + content_layers

# 각 스타일에 대해 GramMSELoss 모듈을 사용하고, 콘텐츠에 대해 MSELoss 모듈을 사용하도록 리스트를 정의함.
# GramMSELoss 모듈 : 스타일 손실(loss) 계산
# [GramMSELoss, GramMSELoss, ..., nn.MSELoss] 형태가 됨.
# GramMSELoss가 nn.Module을 상속한다고 가정합니다.
loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)

# GPU (CUDA) 사용이 가능하다면, 모든 손실 함수 모듈을 GPU 메모리로 이동시킴.
if torch.cuda.is_available():
    # loss_fns 리스트의 요소들을 새로운 모듈 인스턴스로 만들고 .cuda()를 적용
    # 리스트 복사를 방지하고 정확하게 GPU로 이동하기 위해 수정
    loss_fns = [GramMSELoss().to(device) for _ in style_layers] + \
               [nn.MSELoss().to(device) for _ in content_layers]
    #  for _ in style_layers : style layers 개수만큼 GramMSELoss() 넣어요 >> 인스턴스
    #  >> 각각 새로 생성
else:
    # CPU 사용 시에도 동일한 로직으로 인스턴스화
    loss_fns = [GramMSELoss().to(device) for _ in style_layers] + \
               [nn.MSELoss().to(device) for _ in content_layers]

# 스타일 레이어 수만큼 스타일 손실 모듈 + 콘텐츠 레이어의 수 만큼 MSELoss 모듈 생심

# 스타일 손실에 적용할 가중치(beta)를 정의
# 깊은 레이어일수록(복잡한 패턴을 추출하는 레이어) 낮은 가중치를 주는 경향이 있음
# 왜? 가중치가 감소되니깐
style_weights = [1e3/n**2 for n in [64,128,256,512,512]]

# 콘텐츠 손실에 부여할 가중치(alpha)를 정의
content_weights = [1e0]

weights = style_weights + content_weights

# 최적화 목표값 (style targets) 계산
# style_image >> VGG 통과 >> 각 스타일 레이어 Gram Matrix 계산
# >> 변화도 추적에서 제외 (detach)
style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
# vgg(style_image, style_layers) 지정한 레이어들의 특징맵(feature map )리스트로 반환
# [A_r11, A_r21.....] 여기서 A.shape (b,c,h,w)
# GramMatrix(A) >> 형태 변환 (b, c, c) >> 스타일 표현
# .detach() 계산 그래프 분리(역전파시 gradient 계산되지 않도록)
# >> 왜? style_targets 고정된 값(ground truth)
# style_targets? 각 스타일 레이어에 대한 스타일 이미지 Gram Matrix 목록

# 최적화 목표값(content targets)을 계산함.
# 콘텐츠 이미지(content_image)를 VGG에 통과시켜 콘텐츠 레이어의 특징 맵을 추출하고 변화도 추적에서 제외(detach)했음.
# content_image 또한 이미 .cuda() 또는 .to(device)로 GPU에 로드되어 있다고 가정합니다.
content_targets = [A.detach() for A in vgg(content_image, content_layers)]

# 최종적으로 사용할 모든 목표값 리스트를 정의함.
targets = style_targets + content_targets

input_image = content_image.clone().requires_grad_(True)

# L-BFGS 옵티마이저: 최적화 단계에서 closure 함수 요구

optimizer = torch.optim.LBFGS([input_image], max_iter=1)

# optimizer = torch.optim.Adam([input_image], lr=0.01)
# Adam 사용 가능

# 최적화 함수 세기 위한 변수
n_iter = 0

def closure():
    global n_iter

    # 이전 기울기를 초기화합니다.
    optimizer.zero_grad()

    # 생성된 이미지(input_image)를 VGG에 통과시켜 특징 맵을 추출합니다.
    # vgg는 이전에 정의되어 GPU로 이동되었다고 가정합니다.
    out = vgg(input_image, loss_layers)

    # 총 손실을 계산합니다.
    layer_losses = []
    total_loss = 0

    for i, weight in enumerate(weights):
        target = targets[i]
        feature = out[i]
        loss_fn = loss_fns[i]

        # GramMatrix 계산이 필요한 스타일 레이어 처리 (GramMSELoss가 GramMatrix를 내부에서 처리한다고 가정)
        # GramMatrix()가 별도 모듈이면, GramMSELoss 내부에 GramMatrix가 포함되어 있어야 합니다.

        loss = weight * loss_fn(feature, target)
        layer_losses.append(loss.item())
        total_loss += loss

    # 역전파를 수행하여 기울기를 계산합니다.
    total_loss.backward()

    # 진행 상황을 출력합니다.
    if n_iter % 50 == 0:
        print(f"Iteration {n_iter}: Total Loss = {total_loss.item():.4f}")
        # print(f"Layer Losses: {layer_losses}") # 디버깅용

    n_iter += 1
    return total_loss

# 최적화 실행 (반복 횟수 지정)
num_iterations = 500 # 원하는 반복 횟수를 설정합니다.
for i in range(num_iterations):
    # L-BFGS는 step() 호출 시마다 클로저를 여러 번 호출할 수 있습니다.
    optimizer.step(closure)

    # 참고: Adam을 사용한다면, 루프는 다음과 같습니다.
    # loss = closure()
    # optimizer.step()

# 최종 결과 이미지 후처리 (옵션)
# 생성된 이미지를 [0, 1] 범위로 클리핑하여 픽셀 값을 보정합니다.
input_image.data.clamp_(0, 1)

# 최적화된 input_image.data가 최종 스타일 트랜스퍼 결과입니다.