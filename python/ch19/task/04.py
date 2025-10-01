import numpy as np
import pandas as pd

arr = np.array([1,2,3,4,5])
print(arr)
print(arr.shape)
print(arr.dtype)

# 특정 행렬 생성

print(np.zeros((3,3)))
print(np.ones((2,4)))
print(np.full((2,2),7))
print(np.eye(2,3))
print(np.identity(3))

# 난수 행렬 생성

print(np.random.rand(3,3))
print(np.random.randint(1,10,(3,3)))

# 배열 요소별 연산

print( arr + 5 )
print( arr * 2 )
print(arr.sum())
print(arr.mean())
print(arr.max())
print(arr.min())

# 배열 간 연산

arr1 = np.array([1,2,3])
arr2 = np.array([[1],[2],[3]])
print(arr1+arr2)

# 선형대수 연산 (행렬 곱)

matrix1 = np.array([[1,2],[3,4]])
matrix2 = np.array([[5,6],[7,8]])
print(np.dot(matrix1,matrix2))

# 배열 인덱싱, 슬라이싱

arr = np.array([10,20,30,40])
print(arr[2])

arr2 = np.array([[1,2,3],[4,5,6]])
print(arr2[1,2])

print(arr2[0,:])
print(arr2[:,1])

# 조건부 연산

arr = np.array([1,2,3,4,5])
print(arr[arr>3])

# pandas 통합

df = pd.DataFrame(arr,columns=['Value'])
print(df)