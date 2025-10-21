import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 데이터 로드
iris = sns.load_dataset("iris")

# 스타일 설정
sns.set_theme(style="whitegrid",palette='muted')

# 산점도
sns.set_palette('pastel')
sns.scatterplot(data=iris,x='sepal_length',y='sepal_width',hue='species',style='species')
plt.title("Scatter Plot Example")
plt.show()

# 선형회귀선
# sns.lmplot(data=iris,x='sepal_length',y='sepal_width',hue='species',height=6)
# plt.title("Linear Regression Plot")
# plt.show()

# 히스토그램
# sns.histplot(data=iris,x='sepal_length',hue='species',kde=True)
# plt.title("Histogram Example")
# plt.show()

# 박스 플롯
# sns.boxplot(data=iris,x='species',y='sepal_length')
# plt.title("Box Plot Example")
# plt.show

# 바이올린 플롯
# sns.violinplot(data=iris,x='species',y='sepal_length',inner='quart')
# plt.title('Violin Plot Example')
# plt.show()

# 페어 플롯
# sns.pairplot(iris,hue='species')
# plt.show()

# 복합 그래프 작성 (서브플롯)

# g = sns.FacetGrid(iris, col="species", height=4, aspect=1)
# g.map_dataframe(sns.histplot, x="sepal_length", kde=True)
# g.set_titles(col_template="{col_name}")
# plt.show()