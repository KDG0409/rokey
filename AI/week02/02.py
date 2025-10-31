import torch
import numpy as np

x_train = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y_train = torch.tensor([[3.0],[5.0],[7.0],[9.0]])

w = torch.tensor(1.0,requires_grad=True).float()
b = torch.tensor(1.0,requires_grad=True).float()
learning_rate = 0.01
epochs = 500
history=np.zeros((0,2))

def pred(x):
    return w*x+b
def mse(yp,y):
    return ((yp-y)**2).mean()

for i in range(epochs):
    yp = pred(x_train)
    loss = mse(yp,y_train)
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
    
    if ( i % 100 == 0 ):
        item = np.array([i,loss.item()])
        history = np.vstack((history,item))

print(f"초기상태 손실 : {history[0,1]:.4f}")
print(f"최종상태 손실 : {history[-1,1]:.4f}")