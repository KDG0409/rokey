import pandas as pd

df_csv = pd.read_csv("C:/rokey/python/ch19/Assignment/data.csv")

print(f"Age 평균:{df_csv['Age'].mean()},최댓값:{df_csv['Age'].max()},최솟값:{df_csv['Age'].min()}")

print(f"Salary 평균:{df_csv['Salary'].mean()},최댓값:{df_csv['Salary'].max()},최솟값:{df_csv['Salary'].min()}")