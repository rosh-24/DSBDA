#!/usr/bin/env python
# coding: utf-8

# In[10]:


#importing required library

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

tdata=pd.read_csv(r"titanic.csv")

tdata.head()

tdata.info()

# assigning values ot null

tdata['Age'] = tdata['Age'].fillna(np.mean(tdata['Age']))
tdata['Cabin'] = tdata['Cabin'].fillna(tdata['Cabin'].mode()[0])
tdata['Embarked'] = tdata['Embarked'].fillna(tdata['Embarked'].mode()[0])


tdata.isnull().sum()

# boxplot

sns.boxplot(x=tdata['Sex'], y=tdata["Age"], hue=tdata["Survived"], palette = 'Set2').set_title('Plot for distribution of age with respect to each gender along with the information about whether they survived or not')
plt.show()
