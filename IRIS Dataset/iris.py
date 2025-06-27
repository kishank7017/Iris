import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

sns.set_style()

#Load Data

data_1=datasets.load_iris()

# # print(data_1)
# print(data_1['data'][0])
# print(data_1['target'])

dataframe1=pd.DataFrame(data_1['data'])
dataframe1['target']=pd.DataFrame(data_1['target'])
col=0
df=dataframe1[col].hist()
plt.suptitle(col)
plt.show()
print(dataframe1.describe())
print(dataframe1.head(10))