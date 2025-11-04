import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# HE초기화,스케쥴러,모니터링
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tfm = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root='/tmp/mnist', train=True, download=True, transform=tfm)
test_ds  = datasets.MNIST(root='/tmp/mnist', train=False, download=True, transform=tfm)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

# MLP: multi layer perceptron 다층 퍼셉트론
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Flatten(), # 1차원 배열로 변형
            nn.Linear(28*28, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
        # He 초기화 (ReLU에 적합)
        for m in self.f:
            if isinstance(m, nn.Linear):  # 선형회귀 레이어(층) 있다면,
                nn.init.kaiming_normal_(m.weight) # He 초기화
                nn.init.zeros_(m.bias)            # bias 편향 0으로 초기화
    def forward(self, x): return self.f(x)

model = MLP().to(device)
opt = optim.AdamW(model.parameters(), lr=3e-3)
# AdamW : adam(adaptive memotum) + weight decay(가중치 감쇠)
sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, steps_per_epoch=len(train_loader), epochs=5)
# OneCycleL : 초반에 학습을 빠르게, 점점 세밀하게 조정
crit = nn.CrossEntropyLoss()

# Grad 히스토그램 수집용(기울기/weight가중치 모니터링)
grads = []
def hook_fn(m, gi, go):
    if isinstance(m, nn.Linear):
        if m.weight.grad is not None:
            grads.append(m.weight.grad.detach().abs().mean().item())
            # detach: 계산 그래프에서 분리(메모리 절약)
            # abs().mean(): 절대값의 평균
            # item() : 텐서 >> 파이썬의 숫자로 변환(**)

hooks = [m.register_full_backward_hook(hook_fn) for m in model.modules() if isinstance(m, nn.Linear)]

def train_epoch():
    model.train(); tot=0; correct=0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x); loss = crit(out, y)
        loss.backward(); opt.step(); sched.step()
        tot += y.size(0); correct += (out.argmax(1)==y).sum().item()
    return loss.item(), correct/tot
    # out.argmax(1): 각 샘플의 최대 확률이 있는 클래스

def eval_epoch():
    model.eval(); tot=0; correct=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            tot += y.size(0); correct += (out.argmax(1)==y).sum().item()
    return correct/tot

hist_grad = []  # 한 번 학습할 때(epoch) gradient(기울기/가중치)
hist_acc  = []  # 정확도
for ep in range(5):
    loss, tr_acc = train_epoch()  # train_epoch[0] = loss 값
    acc = eval_epoch()
    hist_grad.append(sum(grads[-len(train_loader):])/max(1,len(train_loader)))
    # grads[-len(train_loader):] 마지막 epoch 의 gradient 기울기(가중치) 만 사용
    hist_acc.append(acc)
    print(f"EP{ep+1} loss={loss:.3f} test_acc={acc:.3f}")

plt.figure(); plt.plot(hist_grad); plt.title('Average |grad|'); plt.show()
plt.figure(); plt.plot(hist_acc); plt.title('Test ACC'); plt.show()

for h in hooks: h.remove()