import statsmodels.api as sm

data = sm.datasets.get_rdataset("mtcars").data
# print(data.head())

# 선형 회귀 분석
X = data[['hp','wt']]
y = data['mpg']
X = sm.add_constant(X)
model = sm.OLS(y,X).fit()
print(model.summary())