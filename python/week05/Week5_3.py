import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

sns.scatterplot(data=iris,x='sepal_length',y='sepal_width',hue='species',style='species')
plt.title("Scatter Plot Example")
plt.show()