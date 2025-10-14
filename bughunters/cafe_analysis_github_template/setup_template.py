import os
from textwrap import dedent

dirs = ["data", "src", "outputs", "notebooks"]
for d in dirs:
    os.makedirs(d, exist_ok=True)
# https://bigdata.sbiz.or.kr/#/
# === .gitignore ===
gitignore = dedent("""
__pycache__/
*.pyc
.venv/
venv/
data/
outputs/
.ipynb_checkpoints/
""")

# === requirements.txt ===
req = dedent("""
pandas
matplotlib
seaborn
folium
openpyxl
""")

# === README.md ===
readme = dedent("""
#  전국 카페 및 상권 변화 분석 프로젝트

##  개요
공공데이터포털의 상가업소정보를 활용하여 전국 카페 분포 및 상권 변화를 분석하는 팀 프로젝트입니다.

##  실행 방법
```bash
pip install -r requirements.txt
python main.py
""")

#=== main.py ===

main_py = dedent("""
from src.data_preprocess import load_and_clean_data
from src.data_analysis import analyze_by_region, analyze_by_year, analyze_franchise_ratio
from src.data_visualization import plot_yearly_trend, create_cafe_map
import os

def main():
if not os.path.exists("outputs"):
os.makedirs("outputs")

df = load_and_clean_data("data/상가업소정보_전국.csv")
print(f" 데이터 로드 완료: {len(df)}개 카페")

region_counts = analyze_by_region(df)
print("\\n 전국 시도별 카페 수 상위 10개:")
print(region_counts.head(10))

franchise_ratio = analyze_franchise_ratio(df)
print("\\n 프랜차이즈 비율:")
print(franchise_ratio)

yearly_trend = analyze_by_year(df)
plot_yearly_trend(yearly_trend)

create_cafe_map(df, output_path="outputs/cafe_map.html")


if name == "main":
main()
""")

#=== src/data_preprocess.py ===

data_pre = dedent("""
import pandas as pd

def load_and_clean_data(filepath: str) -> pd.DataFrame:
df = pd.read_csv(filepath, encoding='utf-8')
cafe_df = df[df['상권업종중분류명'].str.contains('커피', na=False)].copy()

cols = ['상호명', '시도명', '시군구명', '상권업종중분류명', '위도', '경도', '등록일자']
cafe_df = cafe_df[cols]

cafe_df['등록일자'] = pd.to_datetime(cafe_df['등록일자'], errors='coerce')
cafe_df['연도'] = cafe_df['등록일자'].dt.year
cafe_df.dropna(subset=['위도', '경도'], inplace=True)
return cafe_df


""")

# === src/data_analysis.py ===

data_analysis = dedent("""
def analyze_by_region(df):
return df.groupby(['시도명'])['상호명'].count().sort_values(ascending=False)

def analyze_by_year(df):
return df.groupby('연도')['상호명'].count()

def analyze_franchise_ratio(df):
brands = ['스타벅스', '이디야', '투썸', '커피빈', '빽다방']
df['프랜차이즈'] = df['상호명'].apply(lambda x: '프랜차이즈' if any(b in str(x) for b in brands) else '개인')
return df['프랜차이즈'].value_counts(normalize=True) * 100
""")

# === src/data_visualization.py ===

data_vis = dedent("""
import folium
import matplotlib.pyplot as plt
import seaborn as sns

def plot_yearly_trend(yearly_data):
plt.figure(figsize=(10,6))
sns.lineplot(x=yearly_data.index, y=yearly_data.values, marker='o')
plt.title('연도별 전국 카페 개수 변화')
plt.xlabel('연도')
plt.ylabel('카페 수')
plt.grid(True)
plt.show()

def create_cafe_map(df, output_path='outputs/cafe_map.html'):
center = [df['위도'].mean(), df['경도'].mean()]
m = folium.Map(location=center, zoom_start=7)
for _, row in df.sample(300).iterrows():
folium.CircleMarker(
[row['위도'], row['경도']],
radius=2,
color='brown',
fill=True,
fill_opacity=0.5
).add_to(m)
m.save(output_path)
print(f" 지도 저장 완료: {output_path}")
""")

# === src/utils.py ===

utils = dedent("""
import os

def ensure_dir(path):
if not os.path.exists(path):
os.makedirs(path)
""")

# === notebooks/eda_sample.ipynb (간단 placeholder) ===

eda_placeholder = dedent("""
{
"cells": [
{"cell_type": "markdown", "source": ["# EDA Sample Notebook"], "metadata": {}},
{"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
"source": ["import pandas as pd\n", "df = pd.read_csv('../data/상가업소정보_전국.csv')\n", "df.head()"]}
],
"metadata": {},
"nbformat": 4,
"nbformat_minor": 2
}
""")

# === 파일 작성 ===

files = {
".gitignore": gitignore,
"requirements.txt": req,
"README.md": readme,
"main.py": main_py,
"src/data_preprocess.py": data_pre,
"src/data_analysis.py": data_analysis,
"src/data_visualization.py": data_vis,
"src/utils.py": utils,
"notebooks/eda_sample.ipynb": eda_placeholder
}

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print(" 프로젝트 템플릿 생성 완료! (cafe_analysis_github_template)")
print(" 이제 'data/상가업소정보_전국.csv' 파일을 넣고 'python main.py' 실행하면 바로 분석 및 지도 확인 가능")