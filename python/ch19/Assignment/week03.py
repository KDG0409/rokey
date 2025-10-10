import pandas as pd

df_csv = pd.read_csv("C:/rokey/python/ch19/Assignment/data.csv")

ANS = df_csv[df_csv['Age']>=30][df_csv['Salary']>=6000]

print(ANS)