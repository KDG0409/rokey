import os
from textwrap import dedent

dirs = ["data", "src", "outputs", "notebooks"]
for d in dirs:
    os.makedirs(d, exist_ok=True)
# https://bigdata.sbiz.or.kr/#/
# === .gitignore ===
gitignore = dedent("""
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.ipynb_checkpoints
.env
.venv/
venv/
.DS_Store

# VSCode
.vscode/

# Data & Outputs
data/
outputs/

# Jupyter
.ipynb_checkpoints/
notebooks/.ipynb_checkpoints/

# OS Files
Thumbs.db
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
# 실행 방법
```bash
pip install -r requirements.txt
python main.py
""")

#=== main.py ===

main_py = dedent("""
from src.data_preprocess import load_and_clean_data
from src.data_analysis import top_dongs_by_year, population_trend, growth_rate
from src.data_visualization import plot_trend, plot_top_dongs_bar, plot_heatmap, plot_multiple_trends
import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'

def main():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # 1 데이터 불러오기
    df = load_and_clean_data('data/datafile.csv')
    
    # 2 특정 동 인구 추이
    dong_name = '강남구'
    trend = population_trend(df, dong_name)
    plot_trend(trend, dong_name)
    
    # 3 특정 연도 인구 상위 동
    year = '2024'
    top_dongs = top_dongs_by_year(df, year)
    plot_top_dongs_bar(top_dongs, year)
    
    # 4 최근 5년 인구 증가율 상위동 (10개)
    growth = growth_rate(df, '2020', '2024')

    # 5 동별 연도별 꺾은선 그래프
    dong_list = df.index.tolist() # '행정동명' = index col
    plot_multiple_trends(df, dong_list)

    # 6 동별 연도별 히트맵
    plot_heatmap(df)

if __name__ == '__main__':
    main()
""")

#=== src/data_preprocess.py ===

data_pre = dedent("""
                  
import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, encoding='utf-8')
    # 문자열 숫자를 숫자로 변환
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)
    return df

""")

# === src/data_analysis.py ===

data_analysis = dedent("""
                       
def top_dongs_by_year(df, year, top_n=10):
    return df[year].sort_values(ascending=False).head(top_n)

def population_trend(df, dong):
    return df.loc[dong]

def growth_rate(df, start_year, end_year):
    return ((df[end_year] - df[start_year]) / df[start_year] * 100).sort_values(ascending=False)
    
""")

# === src/data_visualization.py ===

data_vis = dedent("""

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'

def plot_trend(trend_series, dong):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.figure(figsize=(10,6))
    trend_series.plot(marker='o')
    plt.title(f'{dong} 연도별 인구 추이')
    plt.xlabel('연도')
    plt.ylabel('인구수')
    plt.grid(True)
    plt.show()

def plot_top_dongs_bar(top_dongs, year):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_dongs.values, y=top_dongs.index, palette='coolwarm')
    plt.title(f'{year}년 인구 상위 10개 동')
    plt.xlabel('인구수')
    plt.ylabel('행정동')
    plt.show()

def plot_heatmap(df):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.figure(figsize=(12,10))
    sns.heatmap(df, cmap="YlGnBu", linewidths=0.5)
    plt.title('동별 연도별 인구 히트맵')
    plt.xlabel('연도')
    plt.ylabel('행정동')
    plt.show()

def plot_multiple_trends(df, dong_list):
    plt.figure(figsize=(12,6))
    for dong in dong_list:
        trend = df.loc[dong]
        plt.plot(trend.index, trend.values, marker='o', label=dong)   
    plt.title("동별 연도별 인구 추이 비교")
    plt.xlabel("연도별")
    plt.ylabel("인구수")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()
""")

# === src/utils.py ===

utils = dedent("""
import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
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
}

for path, content in files.items():
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print(" 프로젝트 템플릿 생성 완료! (cafe_analysis_github_template)")