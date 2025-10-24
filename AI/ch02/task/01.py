import numpy as np
np.set_printoptions(suppress=True, precision=5)

# 배열 요소

n1 = np.array([1, 2, 3, 4, 5, 6, 7])

print(n1)
print(n1.shape)
print(len(n1))

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

# 유니버셜 함수

x = np.linspace(0, 2*np.pi, 25)
y = np.sin(x)

print(x)
print(y)

# 집계 함수

print(f'원본 변수 : {n1}')
n23 = np.sum(n1)
n24 = np.mean(n1)
n25 = np.max(n1)
n26 = n1.min()
print(f'합 : {n23}')
print(f'평균 : {n24}')
print(f'최댓값 : {n25}')
print(f'최솟값 : {n26}')

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