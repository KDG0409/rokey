

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
