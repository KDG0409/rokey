# statsmodels
# import statsmodels.api as sm

# r_data = sm.datasets.get_rdataset("mtcars").data
# print(r_data)
# X = r_data[['hp','wt']]
# y = r_data['mpg']
# X=sm.add_constant(X)
# ans=sm.OLS(y,X).fit()
# print(ans.summary)

## sklearn
import sklearn as sk
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm,metrics

# iris = load_iris()
# print(iris)
## df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
## df['target']=iris.target
## target_names={0:iris.target_names[0],1:iris.target_names[1],2:iris.target_names[2]}
# df['target']=df['target'].map(target_names)
# dataA=df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
# dataB=df['target']
# train_d,test_d,train_l,test_l=train_test_split(dataA,dataB)

# clf=svm.SVC()
# clf.fit(train_d,train_l)
# pre=clf.predict(test_d)
# ans=metrics.accuracy_score(test_l,pre)
# print(ans)

# clf=svm.SVC()
# xor_data=[
#     [0,0,0],[0,1,1],[1,0,1],[1,1,0]
# ]
# xor_df=pd.DataFrame(xor_data)
## data=xor_df.loc[:,0:1]
## label=xor_df.loc[:,2]
# clf.fit(data,label)
# pre=clf.predict(data)
# print(pre)
# score=metrics.accuracy_score(label,pre)
# print(score)

# scipy
# from scipy.linalg import solve
# x=[[1,2],[2,3]]
# y=[4,6]
# print(solve(x,y))
# from scipy.optimize import minimize
# def f(x):
#     return x**2 + 4*x + 4
## ans=minimize(f,x0=0)
## print(ans.x)
# from scipy.optimize import root
## ans=root(f,x0=0)
## print(ans.x)
# from scipy.stats import describe
# data=[1,2,3,6,7,9,5,3,6,8,5,3]
# print(describe(data))
# from scipy.sparse import csr_matrix
# data=[[0,0,3],[4,0,0],[0,5,0]]
# print(csr_matrix(data))