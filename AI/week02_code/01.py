import numpy as np
import matplotlib.pyplot as plt
import torch

# 리스트와 배열의 차이

print([1,2,3] + [4,5,6])
print(np.array([1,2,3]) + np.array([4,5,6]))

# 컨테이너 변수 : copy()사용 복제
x = np.array([5, 7, 9])
y = x
x[1] = -1
print(x)
print(y)

x = np.array([5, 7, 9])
y = x.copy()
x[1] = -1
print(x)
print(y)

# 넘파이 토치 변환
x1 = torch.ones(5)
print(x1)
x2 = x1.data.numpy().copy()
print(x2)
x3 = torch.from_numpy(x2)
print(x3)

# 합성 함수 구현
def f1(x):
    return(x**2)
def f2(x):
    return(x*2)
def f3(x):
    return(x+2)

x = np.arange(-2, 2.1, 0.25)
x1 = f1(x)
x2 = f2(x1)
y = f3(x2)

plt.plot(x, y)
plt.show()

# 수치 미분
