import matplotlib.pyplot as plt

# 선 그래프

# x1 = [1,2,3,4]
# y1 = [10,20,25,30]
# plt.plot(x1,y1)
# plt.title('Line Plot')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()

# 막대 그래프

# x2 = ['A','B','C','D']
# y2 = [3,7,8,5]
# plt.bar(x2,y2)
# plt.title('Bar Chart')
# plt.show()

# 히스토그램

# data = [1,2,2,3,3,3,4,4,4,4]
# plt.hist(data,bins=4,color='skyblue',edgecolor='black')
# plt.title('Histogram')
# plt.show()

# 산점도

# x3 = [5,7,8,7,2,17,2,9,4,11]
# y3 = [99,86,87,88,100,86,103,87,94,78]
# plt.scatter(x3,y3,color='green')
# plt.title('Scatter Plot')
# plt.show()

# 원 그래프

# sizes = [15,30,45,10]
# labels=['A','B','C','D']
# plt.pie(sizes,labels=labels,autopct='%1.1f%%',startangle=90)
# plt.title('Pie Chart')
# plt.show()

# 박스 플롯

# data = [7,8,5,6,8,9,6,7,5,8]
# plt.boxplot(data)
# plt.title("Box Plot")
# plt.show()

# 커스터마이징

# x = [1,2,3,4]
# y = [10,20,25,30]
# plt.plot(x,y,color='red',linestyle='--',marker='o')
# plt.title('Customized Line Plot')
# plt.show()

# 축 범위와 눈금 설정

# plt.plot(x,y)
# plt.xlim(0,5)
# plt.ylim(0,40)
# plt.xticks(range(1,5,1))
# plt.yticks(range(0,41,10))
# plt.show()

# 범례 추가 및 위치 설정, 이미지로 저장

# x4 = [1, 2, 3, 4]
# y4= [10, 20, 25, 30]
# x5 = [1, 2, 3, 4]
# y5 = [3, 5, 9, 7]

# plt.plot(x4, y4, label="Data 1")
# plt.plot(x5, y5, label="Data 2")
# plt.legend(loc="best")
# plt.savefig("C:/rokey/python/ch19/task/my_plot.png")
# plt.show()

# subplot 활용

x1 = [1,2,3,4]
y1 = [10,20,25,30]
x2 = ['A','B','C','D']
y2 = [3,7,8,5]
x3 = [5,7,8,7,2,17,2,9,4,11]
y3 = [99,86,87,88,100,86,103,87,94,78]
data = [1,2,2,3,3,3,4,4,4,4]

fig,axs = plt.subplots(2,2)
axs[0,0].plot(x1,y1)
axs[0,1].bar(x2,y2)
axs[1,0].scatter(x3,y3)
axs[1,1].hist(data)
plt.tight_layout()
plt.show()