import torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# Binary: MSE vs BCEWithLogits

X, y = make_classification(n_samples=4000, n_features=20, n_informative=8, weights=[0.6, 0.4], random_state=0)
Xtr, Xte, ytr, yte = train_test_split(torch.tensor(X, dtype=torch.float32),
                                      torch.tensor(y, dtype=torch.float32).unsqueeze(1), test_size=0.3, random_state=0)

class BinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(20,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self,x): return self.m(x)

def run_binary(loss_fn):
    net = BinNet().to(device)
    opt = optim.AdamW(net.parameters(), lr=1e-3)
    for _ in range(8):
        net.train(); opt.zero_grad()
        out = net(Xtr.to(device))           # 훈련(학습)
        loss = loss_fn(out, ytr.to(device)) # 손실계산
        loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.sigmoid(net(Xte.to(device)))
        acc = ((p>0.5).float()==yte.to(device)).float().mean().item()
    return acc

acc_mse = run_binary(lambda out, y: nn.MSELoss()(torch.sigmoid(out), y))
acc_bce = run_binary(nn.BCEWithLogitsLoss())
print(f"Binary Acc: MSE={acc_mse:.3f} vs BCEWithLogits={acc_bce:.3f}")

# Multiclass: Label Smoothing & Class Weights

tfm = transforms.Compose([transforms.ToTensor()]) # 전처리 #예) ex = float(string-value)
train_ds = datasets.MNIST('/tmp/mnist2', train=True,  download=True, transform=tfm) #60,000 images
test_ds  = datasets.MNIST('/tmp/mnist2', train=False, download=True, transform=tfm) #10,000 images
tr = DataLoader(train_ds, batch_size=256, shuffle=True)
te = DataLoader(test_ds,  batch_size=512, shuffle=False)
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # feature extraction (특징 추출)
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            # 컴퓨터에게 보내기 위해 1차원으로 만들어줌
            nn.Flatten(), nn.Linear(32*7*7, 128), nn.ReLU(), nn.Linear(128,10)
        )
    def forward(self,x): return self.net(x)

def train_eval(criterion):
    model = SmallCNN().to(device)
    opt = optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(3):
        model.train()
        for x,y in tr:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); opt.step()
    model.eval(); correct=0; tot=0
    with torch.no_grad():
        for x,y in te:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item(); tot += y.size(0)
    return correct/tot

# Label smoothing
crit_ls = nn.CrossEntropyLoss(label_smoothing=0.1)
acc_ls = train_eval(crit_ls)

# Class weight 예시(0~9 가중치 다르게, 임의)
weights = torch.tensor([1,1,1,1,1,1.2,1,1.2,1,1.2], dtype=torch.float32).to(device)
crit_w = nn.CrossEntropyLoss(weight=weights)
acc_w = train_eval(crit_w)

print(f"MNIST Acc: LabelSmoothing={acc_ls:.3f} | WeightedCE={acc_w:.3f}")