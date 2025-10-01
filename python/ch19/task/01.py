import pandas as pd

# series 구조

data = [10,20,30]
series = pd.Series(data)
print(series)

data2 = {'a':10, 'b':20, 'c':30}
series2 = pd.Series(data2)
print(series2)

# dataFrame 구조

data3 = [
    [1,'Alice',30],
    [2,'Bob',35],
    [3,'Charlie',25]    
]
df = pd.DataFrame(data3, columns=['Id','Name','Age'])
print(df)

data4 = {
    'ID':[1,2,3],
    'Name':['Alice','Bob','Charlie'],
    'Age':[30,35,25]
}
df = pd.DataFrame(data4)
print(df)