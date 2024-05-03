
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"Boston.csv")
data.head()

data.tail()

print("The shape of the data is: ")
data.shape

data.isnull().sum()

data.fillna(0, inplace=True)

#Define the independent and dependent variables from the dataset
X = data.iloc[:,0:13]
y = data.iloc[:,-1]

#Splitting data into traing and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)

#Shapes of the training and testing dataset
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Importing LinearRegression() function
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
model = LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
plt.scatter(y_test,y_pred)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()])
plt.show()
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(mse,"\n",r2)