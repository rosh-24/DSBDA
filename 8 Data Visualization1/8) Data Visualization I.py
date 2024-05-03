

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#load dataset
df=pd.read_csv(r"titanic.csv")
print("TITANIC DATASET LOADED")
df.head()

#display missing values
print(df.isnull().sum())


# In[7]:


#fill the missing values
df['Age'].fillna(df['Age'].median(),inplace=True)
print('Null values are:', df.isnull().sum())

#histogram of 1 variable
fig, axes = plt.subplots(1,2)
fig.suptitle('histogram')
sns.histplot(data=df, x='Age', ax=axes[0])
sns.histplot(data=df, x='Fare', ax=axes[1])
plt.show()

#histogram of 2 variables
fig, axes=plt.subplots(2,2)
fig.suptitle('Histogram of 2 variable')
sns.histplot(data=df, x='Age', hue='Survived', multiple='dodge', ax=axes[0,0])
sns.histplot(data=df, x='Fare', hue='Survived', multiple='dodge', ax=axes[0,1])
sns.histplot(data=df, x='Age', hue='Sex', multiple='dodge', ax=axes[1,0])
sns.histplot(data=df, x='Fare', hue='Sex', multiple='dodge', ax=axes[1,1])
plt.show()
