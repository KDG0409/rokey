import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def fdiff(f):
    # 함수 f를 인수로 미분한 결과 함수를 diff로 정의
    def diff(x):
        h = 1e-6
        return (f(x+h) - f(x-h)) / (2*h)

    # fdiff 의 반환은 미분한 결과 함수 diff
    return diff

diff = fdiff(f1)
y_dash = diff(x)
print(y_dash)

# 시그모이드 함수
def g(x):
    return 1 / (1 + np.exp(-x))
y = g(x)
print(y)
diff = fdiff(g)
y_dash = diff(x)
print(y_dash)

# 커스텀 클래스 정의

# Point 클래스 정의

class Point:
    # 인스턴스 생성 시에 두 개의 인수 x와 y를 가짐
    def __init__(self, x, y):
        # 인스턴스 속성 x에 첫 번째 인수를 할당
        self.x = x
        # 인스턴스 속성 y에 두 번째 인수를 할당
        self.y = y

    # draw 함수 정의(인수 없음)
    def draw(self):
        # (x, y)에 점을 그림
        plt.plot(self.x, self.y, marker='o', markersize=10, c='k')

p1 = Point(2,3)
p2 = Point(-1, -2)

# Point의 자식 클래스 Circle 1/2/3 정의

class Circle1(Point):
    def __init__(self, x, y, r):
        super().__init__(x, y)
        self.r = r

c1_1 = Circle1(1, 0, 2)
print(c1_1.x, c1_1.y, c1_1.r)
ax = plt.subplot()

class Circle2(Point):
    def __init__(self, x, y, r):
        super().__init__(x, y)
        self.r = r
    # draw 함수는 자식 클래스만 따로 원을 그림
    def draw(self):
        # 원 그리기
        c = patches.Circle(xy=(self.x, self.y), radius=self.r, fc='b', ec='k')
        ax.add_patch(c)

# Point의 자식 클래스 Circle의 정의 3

class Circle3(Point):
    def __init__(self, x, y, r):
        super().__init__(x, y)
        self.r = r

    # Circle의 draw 함수는 부모의 함수를 호출한 다음, 원 그리기를 독자적으로 수행함
    def draw(self):
        # 부모 클래스의 draw 함수 호출
        super().draw()

        # 원 그리기
        c = patches.Circle(xy=(self.x, self.y), radius=self.r, fc='b', ec='k')
        ax.add_patch(c)

c2_1 = Circle2(1, 0, 2)
c3_1 = Circle3(1, 0, 2)

ax = plt.subplot()
p1.draw()
p2.draw()
c3_1.draw()
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()

# 인스턴스 사용

class H:
    def __call__(self, x):
        return 2*x**2 + 2
x = np.arange(-2, 2.1, 0.25)
h = H()

y = h(x)
print(y)
plt.plot(x, y)
plt.show()

# 리스트 슬라이싱

l = [1, 2, 3, 5, 8, 13]
print(l[:-1])
print(l[-1])
print(l[2:5])
print(l[:3])
print(l[4:])
print(l[-2:]) # 마지막에서 2개 요소
print(l[:]) # 리스트 전체 참조
print(l[::-1]) # 거꾸로 불러오기

# 딕셔너리 item 함수
my_dict = {'yes': 1, 'no': 0}
print(my_dict.items())
for key, value in my_dict.items():
    print(key, ':', value )

# 함수 다중 반환

def squares(x):
    p2 = x * x
    p3= x * x * x
    return (p2, p3)

x1 = 13
p2, p3 = squares(x1)
print(x1, p2, p3)

# 필요하지 않은 warning 출력하지 않기

import warnings
warnings.filterwarnings('ignore')

# 수치의 출력 형식 지정
# .4f : 소수점 이하 네 자리 고정소수점 표시
# 04 : 정수를 0을 포함해 네 자리까지 표시
# 04e : 소수점 이하 네 자리 부동소수점 표시
# #x : 정수를 16진수로 표시