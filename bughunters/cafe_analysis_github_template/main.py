from src.data_preprocess import load_and_clean_data
from src.data_analysis import top_dongs_by_year, population_trend, growth_rate
from src.data_visualization import plot_trend, plot_top_dongs_bar, plot_heatmap, plot_multiple_trends
import os
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'

def main():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # 1️ 데이터 불러오기
    df = load_and_clean_data('data/datafile.csv')
    
    # 2️ 특정 동 인구 추이
    dong_name = '강남구'
    trend = population_trend(df, dong_name)
    plot_trend(trend, dong_name)
    
    # 3️ 특정 연도 인구 상위 동
    year = '2024'
    top_dongs = top_dongs_by_year(df, year)
    plot_top_dongs_bar(top_dongs, year)
    
    # 4️ 최근 5년 인구 증가율 상위동 (10개)
    growth = growth_rate(df, '2020', '2024')

    # 5 동별 연도별 꺾은선 그래프
    dong_list = df.index.tolist() # '행정동명' = index col
    plot_multiple_trends(df, dong_list)

    # 6 동별 연도별 히트맵
    plot_heatmap(df)

if __name__ == '__main__':
    main()