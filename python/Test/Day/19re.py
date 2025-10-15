#1. b
#2. a
#3. a
#4. a
#5. b
#6/7.
# import pandas as pd
# data = pd.read_csv("C:/rokey/python/Test/Day/data.csv")
# print(f"Age:{data['Age'].mean()},{data['Age'].max()},{data['Age'].min()}")
# print(f"Salary:{data['Salary'].mean()},{data['Salary'].max()},{data['Salary'].min()}")
# print(data[data['Age']>=30][data['Salary']>=60000])
#8.
# import numpy as np
# arr=np.array(range(1,11))
# arr2=arr**2
# print(arr2)
# print(arr2.mean(),arr2.max(),arr2.min())
#9.
# import numpy as np
# rand=np.random.randint(1,10,(3,4))
# print(rand)
# max=[]
# for line in rand:
#     max.append(int(line.max()))
# print(max)
#10.
# import matplotlib.pyplot as plt
# x=[1,2,3,4,5]
# y=[2,4,6,8,10]
# plt.title('line graph')
# plt.plot(x,y)
# plt.xlabel("x_axis:1~5")
# plt.ylabel("y_axis:2~10")
# plt.grid()
# plt.show()