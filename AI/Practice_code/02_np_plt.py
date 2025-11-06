import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=5)

# 배열 요소

n1 = np.array([1, 2, 3, 4, 5, 6, 7])

print(n1)
print(n1.shape)
print(len(n1))

arr_float = np.array([1, 2, 3], dtype=np.float32) # 타입지정
print("float32 배열:", arr_float, ", dtype:", arr_float.dtype)

n2 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10,11,12]
])

print(n2)
print(n2.shape)
print(len(n2))

# 특수 배열 

n3 = np.zeros(5)
print(n3)
print(n3.shape)

n4 = np.ones((2,3))
print(n4)
print(n4.shape)

n5 = np.random.randn(2,3,4)
print(n5)
print(n5.shape)

n6 = np.linspace(-1, 1, 11)
print(n6)

n7 = np.arange(-1, 1.2, 0.2)
print(n7)

# 조작 (추출)

n8 = n2[:,0]
print(n8)

n2_index = np.array([False, True, False, True])
n9 = n2[n2_index]
print(n9)

# 재배열 ( 요소 수 변화x )

n10 = np.array(range(24))
print(n10)

n11 = n10.reshape(3,8)
print(n11)

n12 = n10.reshape(2, -1, 4)
print(n12)

n13 = n10.reshape(1, -1)
print(n10.shape)
print(n13.shape)

# 축 조작

print(n12.shape)
print(n12)

n14 = n2.T
print(n14)

n15 = np.transpose(n12, (1, 2, 0))
print(n15.shape)
print(n15)

# 행렬 연결

n16 = np.array(range(1,7)).reshape(2,3)
n17 = np.array(range(7,13)).reshape(2,3)
n18 = np.array(range(14,17))
n19 = np.array(range(17,19))

n20 = np.vstack([n16, n17])
n21 = np.vstack([n16, n18])
print(n20)
print(n21)

n22 = np.hstack([n16, n17])
n23 = n19.reshape(-1, 1)
n24 = np.hstack([n16, n23])
print(n22)
print(n23)
print(n24)

# 브로드캐스트

print(n1)
n22 = n1 - 4
print(n22)

arr1 = np.array([1, 2, 3])
arr2 = np.array([[10], [20], [30]])
result = arr1 + arr2

# 유니버셜 함수

x = np.linspace(0, 2*np.pi, 25)
y = np.sin(x)

print(x)
print(y)

# 집계 함수(행렬 집계)

print(f'원본 변수 : {n1}')
n23 = np.sum(n1)
n24 = np.mean(n1)
n25 = np.max(n1)
n26 = n1.min()
print(f'합 : {n23}')
print(f'평균 : {n24}')
print(f'최댓값 : {n25}')
print(f'최솟값 : {n26}')

matrix = np.array([[1, 2, 3],[4, 5, 6]])
print("원본 행렬:\n", matrix)

print("전체 합:", np.sum(matrix))
print("행 기준 합(axis=1):", np.sum(matrix, axis=1))
print("열 기준 평균(axis=0):", np.mean(matrix, axis=0))

# 벡터 값을 [0, 1] 범위로 제한

n1_max = n1.max()
n1_min = n1.min()
print(n1_max, n1_min)
n27 = (n1 - n1_min) / (n1_max - n1_min)
print(n27)

# 조건 인덱싱

n28 = n2[:,0] % 2 == 0
print(n28)

n29 = n2[n28]
print(n29)

data = np.array([3, 8, 1, 6, 2, 9])
filtered = data[data > 5]

print("원본 데이터:", data)
print("5보다 큰 값 필터링 결과:", filtered)

# matplotlib (선/분포도)

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label="Sine 함수")
plt.title("기본 선 그래프 (Line Plot)")
plt.xlabel("X 축")
plt.ylabel("Y 축")
plt.legend()
plt.show()

plt.scatter(x, y, label="Sin 데이터 포인트")
plt.title("Scatter Plot 예제")
plt.xlabel("X 축")
plt.ylabel("Y 축")
plt.legend()
plt.show()

# 한 그래프에 여러 함수 표현

x = np.linspace(-5, 5, 200)
y1 = x            # 1차 함수
y2 = x ** 2       # 2차 함수

plt.plot(x, y1, label="y = x")
plt.plot(x, y2, label="y = x^2")
plt.title("1차 함수와 2차 함수 비교")
plt.xlabel("X 축")
plt.ylabel("Y 축")
plt.legend()
plt.grid(True)
plt.show()