import pandas as pd

data = {
    'ID':[1,2,3],
    'Name':['Alice','Bob','Charlie'],
    'Age':[30,35,25]
}
df = pd.DataFrame(data)
print(df)

# 데이터 필터링,정렬,열추가,행추가,행삭제

print(df['Age'])
print(df[df['Age']>30])
print(df.sort_values(by='Age'))

df['Salary'] = [50000,60000,70000]
print(df)
df.loc[len(df)] = [4,'David',40,80000]
print(df)
df=df.drop(1)
print(df)

# DataFrame 행추가

data2 = {
    'ID':[5,6],
    'Name':['Eve','Frank'],
    'Age':[28,33]
}
df2 = pd.DataFrame(data2)
concated = pd.concat([df,df2])
print(concated)

# DataFrame 열추가

data3 = {
    'ID':[1,2,3,4,5,6],
    'Department':['HR','Engineering','Sales','R&D','Finance','Planning']
}
df3 = pd.DataFrame(data3)
merged = pd.merge(concated,df3)
print(merged)

# 결측치(NaN) 확인 및 채우기(mean)

print(df.isnull().sum())
meanVal = merged['Salary'].mean()
merged['Salary'] = merged['Salary'].fillna(meanVal)
print(merged)

# 중복 데이터 확인 및 제거

data1 = {
    'ID':[1,3],
    'Name':['Alice','Charlie'],
    'Age':[30,25],
    'Salary':[50000,70000],
    'Department':['HR','Sales']
}
df1 = pd.DataFrame(data1)
df1 = pd.concat([merged,df1])
print(df1)
print(df1.duplicated())
print(df1.drop_duplicates())