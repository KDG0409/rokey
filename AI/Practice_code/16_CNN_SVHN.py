import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import torch                
import torch.nn as nn         
import torch.optim as optim    
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader  
import matplotlib.pyplot as plt
import numpy as np
import random   
import torchvision

# 기본 설정

device = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 0                           # 임의성 제어용 시드
random.seed(SEED)                  # 파이썬 시드 고정
np.random.seed(SEED)               # 넘파이 시드 고정
torch.manual_seed(SEED)            # 파이토치 CPU 시드 고정
if torch.cuda.is_available():      # GPU 사용 시
    torch.cuda.manual_seed_all(SEED)  # 모든 GPU 시드 고정

BATCH_SIZE = 128                   # 배치 크기
EPOCHS = 3                         # 학습 에폭(데모용)
LR = 2e-3                          # 학습률
NUM_WORKERS = 2                    # DataLoader 병렬 워커 수
mean = (0.4377, 0.4438, 0.4728)    # 채널별 평균
std  = (0.1980, 0.2010, 0.1970)    # 채널별 표준편차

# 데이터 전처리

train_tf = transforms.Compose([
    transforms.ToTensor(),        
    transforms.Normalize(mean, std) 
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
train_ds = datasets.SVHN(root='/content/data_svhn', split='train', download=True, transform=train_tf)
test_ds  = datasets.SVHN(root='/content/data_svhn', split='test',  download=True, transform=test_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

images, labels = next(iter(train_loader)) # 샘플 시각화
grid = 16                                      # 표시 개수
plt.figure(figsize=(8,8))                      # 도화지 크기
for i in range(grid):                          # 16개 루프
    plt.subplot(4,4,i+1)                       # 4x4 서브플롯
    img = images[i].permute(1,2,0).cpu().numpy()     # (H,W,C)로 변환
    img = (img * np.array(std) + np.array(mean))     # 정규화 역변환
    img = np.clip(img, 0, 1)                         # 0~1로 클리핑
    plt.imshow(img)                                   # 이미지 표시
    plt.title(int(labels[i]))                         # 레이블 표시
    plt.axis('off')                                   # 축 숨김
plt.tight_layout()                                    # 레이아웃 정리
plt.show()  

# 모델 설계

class ClassCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
        for m in self.modules(): # He(kaiming) 초기화로 안정적 학습 시작
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)     # 특징 추출
        x = self.classifier(x)   # 분류
        return x

cls_model = ClassCNN().to(device)    

# 레이어별 출력 shape 추적(후크)

# Feature Extractor (특징추출) 파트
seq_feature_extractor = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3->32 채널, 3x3 필터
    nn.ReLU(),                                   # 비선형 활성화
    nn.MaxPool2d(2),                              # 공간 크기 절반

    nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32->64 채널
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1),# 64->128 채널
    nn.ReLU(),
    nn.MaxPool2d(2)                               # 32->16->8->4
)
# Classifier (분류기) 파트
seq_classifier = nn.Sequential(
    nn.Flatten(),                                 # (B,128,4,4)->(B,2048)
    nn.Linear(128*4*4, 256),                      # 완전연결층
    nn.ReLU(),                                    # 활성화
    nn.Linear(256, 10)                            # SVHN 10 클래스
)
seq_model = nn.Sequential(seq_feature_extractor, seq_classifier).to(device)
dummy = torch.randn(1, 3, 32, 32).to(device)     # 가짜 이미지
out = seq_model(dummy)                            # 순전파

shapes = []                                       # shape 저장 리스트
def hook(m, i, o):
    if isinstance(o, torch.Tensor):
        shapes.append(tuple(o.shape))             # 출력 shape 기록

hooks = []
for layer in cls_model.features:                  # 특징추출부 레이어 순회
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
        hooks.append(layer.register_forward_hook(hook))  # 후크 등록
_ = cls_model(dummy)                              # 더미 순전파로 후크 실행
for h in hooks: h.remove()                        # 후크 해제:메모리 누수를 방지
print('features 출력 shapes:', shapes)            # 레이어별 shape 출력

# 함수 정의

def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)           # 예측 클래스
    return (preds == targets).float().mean().item()  # 정확도

def train_one_epoch(model, loader, optimizer, criterion): # 1 에폭 학습
    model.train()                           # 학습 모드 
    tot_loss, tot_acc, tot_cnt = 0.0, 0.0, 0
    for x, y in loader:                     # 미니배치 반복
        x, y = x.to(device), y.to(device)   # 디바이스 이동
        optimizer.zero_grad()               # 기울기 초기화
        out = model(x)                      # 순전파
        loss = criterion(out, y)            # 손실 계산
        loss.backward()                     # 역전파
        optimizer.step()                    # 파라미터 갱신
        tot_loss += loss.item() * y.size(0) # 손실 누적
        tot_acc  += (out.argmax(1) == y).float().sum().item() # 정답수 누적
        tot_cnt  += y.size(0)               # 샘플 수 누적
    return tot_loss/tot_cnt, tot_acc/tot_cnt  # 평균 손실/정확도

def evaluate(model, loader, criterion): # 평가
    model.eval()                            # 평가 모드
    tot_loss, tot_acc, tot_cnt = 0.0, 0.0, 0
    with torch.no_grad():                   # 기울기 미계산
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            tot_loss += loss.item() * y.size(0)
            tot_acc  += (out.argmax(1) == y).float().sum().item()
            tot_cnt  += y.size(0)
    return tot_loss/tot_cnt, tot_acc/tot_cnt

# 모델 학습/평가 시각화
criterion = nn.CrossEntropyLoss()                # 다중분류 손실
optimizer = optim.AdamW(seq_model.parameters(), lr=LR)  # AdamW 최적화기

tr_hist, te_hist = [], []                        # 기록용 리스트
for ep in range(1, EPOCHS+1):                    # 에폭 반복
    tr_loss, tr_acc = train_one_epoch(seq_model, train_loader, optimizer, criterion)
    te_loss, te_acc = evaluate(seq_model, test_loader, criterion)
    tr_hist.append((tr_loss, tr_acc))
    te_hist.append((te_loss, te_acc))
    print(f"[Sequential] Epoch {ep}/{EPOCHS} | train {tr_acc:.3f}/{tr_loss:.3f} | test {te_acc:.3f}/{te_loss:.3f}")

# 정확도 곡선
plt.figure(); plt.plot([a for _,a in tr_hist], label='train acc'); plt.plot([a for _,a in te_hist], label='test acc')
plt.legend(); plt.title('정확도 추세(Sequential)'); plt.show()

# 손실 곡선
plt.figure(); plt.plot([l for l,_ in tr_hist], label='train loss'); plt.plot([l for l,_ in te_hist], label='test loss')
plt.legend(); plt.title('손실 추세(Sequential)'); plt.show()

# 하이퍼 파라미터 실험

class SmallExp(nn.Module):
    def __init__(self, ch1=16, ch2=32, k=3, stride=1):
        super().__init__()
        pad = k//2                                 # 출력 사이즈 보존용 패딩
        self.net = nn.Sequential(
            nn.Conv2d(3,  ch1, k, stride=stride, padding=pad), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(ch1, ch2, k, stride=1,      padding=pad), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(ch2*8*8, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self,x): return self.net(x)        # 순전파 정의

def quick_eval(ch1, ch2, k, stride):
    m = SmallExp(ch1, ch2, k, stride).to(device)   # 모델 생성/이동
    opt = optim.AdamW(m.parameters(), lr=LR)       # 최적화기
    crit = nn.CrossEntropyLoss()                   # 손실함수
    for _ in range(2):                             # 데모용 2에폭만
        train_one_epoch(m, train_loader, opt, crit)
    _, acc = evaluate(m, test_loader, crit)        # 테스트 정확도
    return acc

settings = [                                      # 실험 설정들
    {'ch1':16,'ch2':32,'k':3,'stride':1},
    {'ch1':32,'ch2':64,'k':3,'stride':1},
    {'ch1':32,'ch2':64,'k':5,'stride':1}
]

results = []
for s in settings:                                 # 설정 반복
    acc = quick_eval(**s)                          # 설정 실행
    results.append((s, acc))                       # 결과 저장
    print('설정:', s, '| 테스트 정확도:', round(acc,4))

# 막대 그래프 표시
labels = [f"{r[0]['ch1']}/{r[0]['ch2']},k{r[0]['k']},s{r[0]['stride']}" for r in results]
vals   = [r[1] for r in results]
plt.figure(figsize=(8,3)); plt.bar(range(len(vals)), vals)
plt.xticks(range(len(vals)), labels, rotation=30, ha='right')
plt.title('하이퍼파라미터 변화에 따른 정확도(간이)')
plt.show()