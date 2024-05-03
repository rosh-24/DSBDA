
# Use iris10 dataset

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

df=pd.read_csv(r"iris.csv")
print('iris loaded')
print(df.head)

#find missing values

misval=df.isnull().sum()
print(misval)

#histogram of 1 variable

#histogram of 1 variable
fig, axes=plt.subplots(2,2)
fig.suptitle('histogram of 1 variable')
sns.histplot(data=df, x='SepalLength', ax=axes[0,0])
sns.histplot(data=df, x='SepalWidth', ax=axes[0,1])
sns.histplot(data=df, x='PetalLength', ax=axes[1,0])
sns.histplot(data=df, x='PetalWidth', ax=axes[1,1])
plt.tight_layout()
plt.show()


#histogram of 2 variable

fig, axes=plt.subplots(2,2)
fig.suptitle('histogram of 2 variables')
sns.histplot(data=df, x='SepalLength', hue='Species', multiple='dodge', ax=axes[0,0])
sns.histplot(data=df, x='SepalWidth', hue='Species', multiple='dodge', ax=axes[0,1])
sns.histplot(data=df, x='PetalLength', hue='Species', multiple='dodge', ax=axes[1,0])
sns.histplot(data=df, x='PetalWidth', hue='Species', multiple='dodge', ax=axes[1,1])
plt.tight_layout()
plt.show()

#boxplot of 1 variable

fig, axes=plt.subplots(2,2)
fig.suptitle('Boxplot of 1 variable')
sns.boxplot(data=df, x='SepalLength', ax=axes[0,0])
sns.boxplot(data=df, x='SepalWidth', ax=axes[0,1])
sns.boxplot(data=df, x='PetalLength', ax=axes[1,0])
sns.boxplot(data=df, x='PetalWidth', ax=axes[1,1])
plt.tight_layout()
plt.show()

#boxplot of 2 variable

fig, axes=plt.subplots(2,2)
sns.boxplot(data=df, x='SepalLength', y='Species', hue='Species', ax=axes[0,0])
sns.boxplot(data=df, x='SepalWidth', y='Species', hue='Species', ax=axes[0,1])
sns.boxplot(data=df, x='PetalLength', y='Species', hue='Species', ax=axes[1,0])
sns.boxplot(data=df, x='PetalWidth', y='Species', hue='Species', ax=axes[1,1])
plt.tight_layout()
plt.show()
