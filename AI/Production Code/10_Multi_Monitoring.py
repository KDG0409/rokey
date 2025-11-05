# 다중분류 지표

import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_ds = datasets.CIFAR10('/tmp/cifar', train=True, download=True, transform=tfm)
test_ds  = datasets.CIFAR10('/tmp/cifar', train=False, download=True, transform=tfm)
tr = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
te = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2)

class TinyC(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*8*8,256), nn.ReLU(), nn.Linear(256,10)
        )
    def forward(self,x): return self.f(x)

model = TinyC().to(device)
# 클래스 가중치 예시: 임의로 후반 클래스 가중
weights = torch.tensor([1,1,1,1,1,1.2,1.2,1.2,1,1], dtype=torch.float32).to(device)
crit = nn.CrossEntropyLoss(weight=weights)
opt  = optim.AdamW(model.parameters(), lr=2e-3)

for ep in range(3):
    model.train()
    for x,y in tr:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
    print(f"EP{ep+1} done")

# 평가 + 리포트
model.eval(); y_true=[]; y_pred=[]
with torch.no_grad():
    for x,y in te:
        x = x.to(device)
        out = model(x).argmax(1).cpu().numpy()
        y_pred.extend(out); y_true.extend(y.numpy())

print(classification_report(y_true, y_pred, digits=4))
print(confusion_matrix(y_true, y_pred))