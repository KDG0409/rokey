import numpy as np
import torch, math, random
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
from torchviz import make_dot

import warnings
warnings.simplefilter('ignore')

# 데이터 전처리 : 추출>분할>시각화>표준화>텐서변환
sampleData1 = np.array([
    [166, 58.7],
    [176.0, 75.7],
    [171.0, 62.1],
    [173.0, 70.4],
    [169.0, 60.1]
])
x = sampleData1[:,0]
y = sampleData1[:,1]
plt.scatter(x, y, c='k', s=50) 
plt.xlabel('$x$: 신장 (cm)') 
plt.ylabel('$y$: 체중 (kg)') 
plt.title('신장과 체중의 관계')
plt.show()  

X = x - x.mean()
Y = y - y.mean()

X = torch.tensor(X).float()
Y = torch.tensor(Y).float()
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

# 예측함수
def pred(X):
  return W * X + B

# 손실함수
def mse(Yp, Y):
  loss = ((Yp - Y)**2).mean()
  return loss

num_epochs = 500
lr = 0.001
history = np.zeros((0, 2))

for epoch in range(num_epochs):
    Yp = pred(X) # 예측 계산
    loss = mse(Yp, Y) # 손실 계산
    loss.backward() # 경사값 계산
    with torch.no_grad(): # 경사값 업데이트(수정)
        W -= lr * W.grad
        B -= lr * B.grad
    W.grad.zero_() # 경사값 초기화
    B.grad.zero_()
    if (epoch % 10 == 0):  # 손실 기록
        item = np.array([epoch, loss.item()])
        history = np.vstack((history, item))
        print(f'epoch = {epoch} loss = {loss:.4f}')

print('W', W.data.numpy()) # W의 데이터 값(최종)
print('B', B.data.numpy()) # B의 데이터 값(최종)

# 손실 확인
print(f'초기상태: 손실 : {history[0,1]:.4f}') #두 번째 열의 값(초기 손실)을 출력
print(f'최종상태: 손실 : {history[-1,1]:.4f}') #두 번째 열의 값(최종 손실)을 출력

# 시각화
plt.plot(history[:,0], history[:,1], 'b') # X축: history의 모든 행의 첫 번째 열(반복 횟수), Y축: 모든 행의 두 번째 열(손실)
plt.xlabel('반복 횟수') # X축 레이블 설정
plt.ylabel('손실') # Y축 레이블 설정
plt.title('학습 곡선(손실)') # 그래프 제목 설정
plt.show() # 그래프를 화면에 표시

# 산포도와 회귀직선 동시 시각화
X_max = X.max()
X_min = X.min()
X_range = np.array((X_min, X_max)) # x의 범위를 구함(Xrange)
X_range = torch.from_numpy(X_range).float()
print(X_range)
Y_range = pred(X_range) # 이와 대응하는 예측값 y
print(Y_range.data)

plt.scatter(X, Y, c='k', s=50) # 산포도와 상관 직선 동시 출력
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_range.data, Y_range.data, lw=2, c='c') # c : 청녹색
plt.title('신장과 체중의 상관 직선(가공 후)')
plt.show()

# 가공 전 데이터로 시각화
x_range = X_range + x.mean()
yp_range = Y_range + y.mean()

plt.scatter(x,  y,  c='k',  s=50)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.plot(x_range, yp_range.data, lw=2, c='b')
plt.title('신장과 체중의 상관 직선(가공 전)')
plt.show()

# 최적화 함수 사용
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()
num_epochs = 500
lr = 0.001
optimizer = optim.SGD([W, B], lr=lr)
history = np.zeros((0, 2))
for epoch in range(num_epochs):
    optimizer.zero_grad() # 옵티마이저에 연결된 파라미터들의 경사도를 0으로 초기화
    Yp = pred(X)
    loss = mse(Yp, Y)
    loss.backward()
    optimizer.step() # 옵티마이저가 알아서 W와 B를 업데이트함
    if (epoch % 10 == 0): # 손실 기록 (10회마다)
        item = np.array([epoch, loss.item()])
        history = np.vstack((history, item))
        print(f'epoch = {epoch} loss = {loss.item():.4f}')

print('W', W.data.numpy())
print('B', B.data.numpy())
print(f'초기상태: 손실 : {history[0,1]:.4f}')
print(f'최종상태: 손실: {history[-1,1]:.4f}')
plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.show()

# 촤적화 함수 튜닝 (모멘텀 이용)
history_default = history # 사전 학습 결과
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()
optimizer = optim.SGD([W, B], lr=lr, momentum=0.9) # momentum 옵션 추가
history_momentum = np.zeros((0, 2)) # 비교를 위해 새로운 history 변수 생성
for epoch in range(num_epochs):
    Yp = pred(X)
    loss = mse(Yp, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch % 10 == 0):
        item = np.array([epoch, loss.item()])
        history_momentum = np.vstack((history_momentum, item))
        print(f'epoch = {epoch}  loss = {loss:.4f}')
# 비교 시각화      
plt.plot(history_default[:,0], history_default[:,1], 'c', label='기본값 설정') # 기본값 SGD 결과 (청록색)
plt.plot(history_momentum[:,0], history_momentum[:,1], 'k', label='momentum=0.9') # momentum 적용 결과 (검은색)
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.legend() # 범례 표시
plt.title('학습 곡선(손실)')
plt.show()


# 4차시 보강 실습: Bias-Variance 감각 익히기 (편ba 분va )
# 1) 데이터: 사인파 + 잡음 -> 회귀 문제
torch.manual_seed(0)
N = 600
x = torch.linspace(-3*math.pi, 3*math.pi, N).unsqueeze(1)
y = torch.sin(x) + 0.2*torch.randn_like(x)

X_train, X_temp, y_train, y_temp = train_test_split(x.numpy(), y.numpy(), test_size=0.4, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_val,   y_val   = torch.tensor(X_val,   dtype=torch.float32), torch.tensor(y_val,   dtype=torch.float32)
X_test,  y_test  = torch.tensor(X_test,  dtype=torch.float32), torch.tensor(y_test,  dtype=torch.float32)

# 2) 두 모델: 저용량(작은 MLP) vs 고용량(큰 MLP)
def make_mlp(hidden):
    return nn.Sequential(
        nn.Linear(1, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1)
    )

small = make_mlp(hidden=8)   # 고편향/저분산 경향
big   = make_mlp(hidden=128) # 저편향/고분산 경향

def train(model, Xtr, ytr, Xva, yva, epochs=600, lr=1e-3):
    # lr = [0.001,0.01,0.1] for i in lr 가장 성능 좋게 나온 lr >> lr = lr
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    tr_hist, va_hist = [], []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xtr) # 반드시 훈련용 데이터로 학습(**) >> 예측 값
        loss = loss_fn(pred, ytr)
        loss.backward()   # w1 = w0 - lr(diff_y / diff_xtr)
        opt.step()

        # transfer learning(전이 학습)
        # 잘 된 모델(오픈소스 제공 모델)을 활용, 내 모델에 적용하는 것
        model.eval()  # eval : evaluation 평가
        with torch.no_grad():
            v = loss_fn(model(Xva), yva).item()
        tr_hist.append(loss.item()); va_hist.append(v)
    return tr_hist, va_hist

tr_s, va_s = train(small, X_train, y_train, X_val, y_val, epochs=400)
tr_b, va_b = train(big, X_train, y_train, X_val, y_val, epochs=400)

plt.figure(); plt.plot(tr_s, label='small-train'); plt.plot(va_s, label='small-val')
plt.plot(tr_b, label='big-train'); plt.plot(va_b, label='big-val'); plt.legend(); plt.title('Bias-Variance')
plt.show()

# 최종 테스트 MSE
def mse(model, X, y):
    with torch.no_grad():
        return nn.MSELoss()(model(X), y).item()
print("Small Test MSE:", mse(small, X_test, y_test))
print("Big   Test MSE:", mse(big,   X_test, y_test))
