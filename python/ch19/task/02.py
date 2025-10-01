import pandas as pd
import openpyxl

df_csv = pd.read_csv("C:/rokey/python/ch19/task/data.csv")
df_xl = pd.read_excel("C:/rokey/python/ch19/task/data.xlsx")

print(df_csv.head())
print(df_csv.tail())
print(df_csv.info())
print(df_csv.describe())
print(df_csv.sample(2))
print(df_csv.sample(frac=0.5))

print(df_xl.head())
print(df_xl.tail())
print(df_xl.info())
print(df_xl.describe())
print(df_xl.sample(2))
print(df_xl.sample(frac=0.5))