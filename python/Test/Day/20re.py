#1. c
#2. a
#3. a
#4. a
#5. b
#6. 
# import statsmodels.api as sm
# X=[1,2,3,4,5]
# y=[2,4,6,8,10]
# X1=sm.add_constant(X)
# ans=sm.OLS(y,X1).fit()
# print(ans.params[1])
# print(ans.params[0])
#7.
# import matplotlib.pyplot as plt
# plt.title("Linear Regression")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.plot(X,y/'r')
# plt.scatter(X,y)
# plt.show()
#8.
# from sklearn import svm,metrics
# X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
# y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
# X1= [[4.5], [6.5]]
# clf=svm.SVC()
# clf.fit(X,y)
# pre=clf.predict(X1)
# print(pre)
#9.
# from scipy.optimize import root
# def f(x):
#     return (x-2)**2
# ans=root(f,x0=0)
# print(ans.x) 
#10.
# from scipy.stats import describe
# A=[80,85,90,75,95]
# print(describe(A))
            