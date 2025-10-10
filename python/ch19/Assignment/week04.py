import numpy as np

arr = np.array(range(1,11))
arr2 = arr * arr
print(f"원본 배열: {arr}")
print(f"제곱 배열: {arr2}")
print(f"평균:{arr2.mean()},최댓값:{arr2.max()},최솟값:{arr2.min()}")