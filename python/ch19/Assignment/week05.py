import numpy as np

arr = np.random.randint(1,10,(3,4))

print(f"원본행렬:{arr}")
print(f"각 행의 최댓값: [{arr[0].max()} {arr[1].max()} {arr[2].max()}]")