def top_dongs_by_year(df, year, top_n=10):
    """특정 연도의 인구 상위 동"""
    return df[year].sort_values(ascending=False).head(top_n)

def population_trend(df, dong):
    """특정 동의 연도별 인구 추이"""
    return df.loc[dong]

def growth_rate(df, start_year, end_year):
    """동별 인구 증가율 계산"""
    return ((df[end_year] - df[start_year]) / df[start_year] * 100).sort_values(ascending=False)