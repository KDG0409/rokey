import pandas as pd
import openpyxl

df=pd.read_excel(r".\python\ch22\task\sample.xlsx", sheet_name="Sample Data")
print(df.head())
print(df.info())
print(df.describe())
print(df["나이"])
print(df["나이"].sum())
print(df["나이"].mean())
df["출생년도"] = 2025 - df["나이"]
print(df)

df.to_excel(r".\python\ch22\task\modified_sample.xlsx", index=False)
df.to_csv(r".\python\ch22\task\modified_sample.csv", index=False)