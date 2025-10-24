import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x,a):
    return 1/(1+np.exp(-a*x))

plt.title("sigmoid function")
x=np.linspace(-3,3,600)
for i in range(0,4):
    y=sigmoid(x,i)
    plt.plot(x,y)
plt.show()