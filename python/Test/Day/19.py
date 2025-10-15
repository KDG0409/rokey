#1. b
#2. a
#3. a
#4. a
#5. b
#6/7. 
# import pandas as pd
# data = pd.read_csv("C:/rokey/python/Test/Day/data.csv")

# print(f"Age 평균:{data['Age'].mean()},최댓값:{data['Age'].max()},최솟값:{data['Age'].min()}")
# print(f"Salary 평균:{data['Salary'].mean()},최댓값:{data['Salary'].max()},최솟값:{data['Salary'].min()}")
# print(data[data['Age']>=30][data['Salary']>=60000])
#8.
# import numpy as np
# arr=np.array(list(range(1,11)))
# print(f'원본배열:{arr}')
# print(f'제곱배열:{arr**2}')
# print(f'평균:{arr.mean()},최댓값:{arr.max()},최솟값:{arr.min()}')
#9.
# import numpy as np
# arr=np.random.randint(1,10,(3,4))
# print(f"출력결과:\n원본행렬:\n{arr}")
# max=[]
# for row in arr:
#     max.append(int(row.max()))
# print(f"각 행의 최댓값: {max} ")
#10.
import matplotlib.pyplot as plt

plt.title("line graph")
x=list(range(1,6,1))
y=list(range(2,11,2))
plt.plot(x,y)
plt.xlabel("x_axis:1~5")
plt.ylabel("y_axis:2~10")
plt.grid(True)
plt.show()