#1. c
#2. a
#3. a
#4. a
#5. b
##6.
# import statsmodels.api as sm
# X=[1,2,3,4,5]
# y=[2,4,6,8,10]
# X=sm.add_constant(X)
# ans=sm.OLS(y,X).fit()
# print(ans.params[1]) #w
# print(ans.params[0]) #b
#7.
# import matplotlib.pyplot as plt
# plt.plot(X,y,'r')
# plt.scatter(X,y)
# plt.show()
#8.
# from sklearn import svm,metrics
# X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
# y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
# x1 = [[4.5], [6.5]]
# clf=svm.SVC()
# clf.fit(X,y)
# pre=clf.predict(x1)
# print(pre)
#9.
# from scipy.optimize import root
# def f(x):
#     return (x-3)**2
# print(root(f,x0=0).x)
#10.
# from scipy.stats import describe
# data=[80,85,90,75,95]
# print(describe(data))
