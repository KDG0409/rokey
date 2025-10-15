#statsmpdels
# import statsmodels.api as sm
# data=sm.datasets.get_rdataset("mtcars").data
# X=data[[]'hp','wt']]
# y=data['mpg']
# X=sm.add_constant(X)
# ans=sm.OLS(y,X).fit()
# print(ans.summary())

#sklearn
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
# from sklearn import svm,metrics
# iris=load_iris()
# df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
# df['target']=iris.target
# target_names={0:iris.target_names[0],1:iris.target_names[1],2:iris.target_names[2]}
# df['target']=df['target'].map(target_names)
# dataA=df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
# dataB=df['target']
# train_d,test_d,train_l,test_l=train_test_split(dataA,dataB)
# clf=svm.SVC()
# clf.fit(train_d,train_l)
# pre=clf.predict(test_d)
# score=metrics.accuracy_score(test_l,pre)
# print(score)

#scipy
# from scipy.linalg import solve
# x=[[1,2],[3,4]]
# y=[5,8]
# print(solve(x,y))
# from scipy.optimize import minimize,root
# def f(x):
#     return (x-3)**2
# print(minimize(f,x0=0).x)
# print(root(f,x0=0).x)
# from scipy.stats import describe
# data=[2,3,6,9,9,5,3,6,9]
# print(describe(data))
# from scipy.sparse import csr_matrix
# data=[[1,0,0],[0,0,5],[0,7,0]]
# print(csr_matrix(data))
