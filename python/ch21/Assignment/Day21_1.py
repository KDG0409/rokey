import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(50,10,1000)

plt.title("HISTOGRAM")
sns.histplot(data)
plt.show()