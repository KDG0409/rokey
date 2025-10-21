import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = sns.load_dataset("tips")
data=pd.DataFrame(data)
print(data)
datax=data['day']
datay=data['total_bill']
plt.title('BoxPlot')
sns.boxplot(data,x='day',y='total_bill')
plt.show()