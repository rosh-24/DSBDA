#!/usr/bin/env python
# coding: utf-8

# In[1]:




# Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt


# In[10]:




data = pd.read_csv(r"C:\Users\prath\OneDrive\Desktop\DATASETS\All Datasets\iris10.csv")
data.head()


# In[11]:


data.shape


# In[12]:


data.info


# In[13]:


data.describe()


# In[14]:


data.isnull().sum()


# In[15]:


#Defining X and Y for the model



X = data.drop(['variety'], axis=1)
y = data.drop(['sepal.length',  'sepal.width',  'petal.length',  'petal.width'], axis=1)
print(X)
print(y)
print(X.shape)
print(y.shape)


# In[16]:




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[17]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)


# In[18]:


y_pred = model.predict(X_test)
model.score(X_test,y_test)


# In[19]:




from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print(accuracy_score(y_test, y_pred))


# In[20]:


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
print("Confusion matrix:")
print(cm)


# In[21]:


disp.plot()
plt.show()


# In[22]:




def get_confusion_matrix_values(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])

TP, FP, FN, TN = get_confusion_matrix_values(y_test, y_pred)
print("TP: ", TP)
print("FP: ", FP)
print("FN: ", FN)
print("TN: ", TN)


# In[23]:




print("The Accuracy is ", (TP+TN)/(TP+TN+FP+FN))
print("The precision is ", TP/(TP+FP))
print("The recall is ", TP/(TP+FN))


# In[ ]:




