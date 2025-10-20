

def top_dongs_by_year(df, year, top_n=10):
    return df[year].sort_values(ascending=False).head(top_n)

def population_trend(df, dong):
    return df.loc[dong]

def growth_rate(df, start_year, end_year):
    return ((df[end_year] - df[start_year]) / df[start_year] * 100).sort_values(ascending=False)

