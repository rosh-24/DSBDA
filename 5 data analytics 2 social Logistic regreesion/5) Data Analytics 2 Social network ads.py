#!/usr/bin/env python
# coding: utf-8

# In[1]:




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
     


# In[7]:


df= pd.read_csv(r"Social_Network_Ads.csv")
print(df.head())


# In[8]:


df.shape


# In[9]:


#Drop the column User ID

df.drop(['User ID'],axis=1,inplace=True)


# In[10]:


df.head()


# In[11]:


df.Purchased.value_counts()


# In[12]:




df.Gender.value_counts()
     


# In[13]:




df.dtypes
     


# In[14]:


#Data Preprocessing

df.isnull().sum()


# In[15]:


df.describe()


# In[16]:




g = sns.catplot(x = "Gender",y = "Purchased",data = df,kind = "bar",height = 4)
g.set_ylabels("Purchased Probability")
plt.show
     


# In[17]:




M2 = pd.crosstab(df.Gender, df.Purchased, normalize='index')
print(M2)
M2.plot.bar(figsize=(6,4),stacked=True)
plt.legend(title='Gender vs Purchased', loc='upper right')
plt.show()
     


# In[18]:




corr = df.corr()
print(corr.shape)
plt.figure(figsize=(8,8))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')
     


# In[19]:




X=df.drop(['Gender','Purchased'],axis=1)
Y= df['Purchased']
X.head()
     


# In[20]:


# Split the data into Train set and Test set



from sklearn.model_selection import train_test_split
# Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)

# Success
print("Training and testing split was successful.")
     


# In[21]:


# build the model

from sklearn.linear_model import LogisticRegression
basemodel= LogisticRegression()
basemodel.fit(X_train,y_train)
print("Training accuracy:", basemodel.score(X_train,y_train)*100)


# In[22]:


# make predictions on test data



y_predict= basemodel.predict(X_test)
print("Testing accuracy:", basemodel.score(X_test,y_test)*100)
     


# In[23]:


# Normalize the data using Min Max Normalization or any other technique



from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
     



model= LogisticRegression()
model.fit(X_train,y_train)
y_predict= model.predict(X_test)
print("Training accuracy:", model.score(X_train,y_train)*100)
print("Testing accuracy:", model.score(X_test,y_test)*100)
     


# Measure the perormance using Precision, Recall, Fscore, Support etc



from sklearn.metrics import accuracy_score
Acc=accuracy_score(y_test,y_predict)
print(Acc)
     


# In[29]:




from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_predict)
print(cm)
     


# In[30]:


from sklearn.metrics import precision_recall_fscore_support
prf= precision_recall_fscore_support(y_test,y_predict)
print('precision:',prf[0])
print('Recall:',prf[1])
print('fscore:',prf[2])
print('support:',prf[3])


# In[31]:




from sklearn.metrics import classification_report
cr= classification_report(y_test,y_predict)
print(cr)
     


# In[ ]:




