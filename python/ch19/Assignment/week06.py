import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [2,4,6,8,10]

print('X축: 1~5')
print('Y축: 2~10')
line = plt.plot(x,y)
plt.title("line graph")
plt.xlabel("x")
plt.ylabel("y")
plt.grid = True
plt.show()