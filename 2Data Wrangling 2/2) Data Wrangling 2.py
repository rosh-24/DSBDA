
import pandas as pd
import matplotlib.pyplot as plt

# assigning the data
data = {'Name': ['Adi', 'Deeksha', 'Jincy', 'Keerthi', 'Harish', 'Anu', 'Ram'],
        'Age': [17, 17, 18, 17, 18, 17, 17],
        'Gender': ['M', 'F', 'F', 'F', 'M', 'F', 'M'],
        'Marks': [90, 76, 'NAN', 74, 65, 'NAN', 71]}
# converting into the dataframes
df = pd.DataFrame(data)
# displaying the data
print(df)
# replacing the missing values
df = df.replace(to_replace="NAN", value=30)
# displaying the data
print(df)
print("RESHAPING THE DATA")
# Categorize the gender
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1, })
# displaying the data
print(df)
# filter top scoring students
print("FILTERING THE DATA")
df = df[df['Marks'] >= 75]
# remove age row
df = df.drop(['Age'], axis=1)
# displaying the data
print("filter", df)
print("outlier with skewness")
print(df['Marks'].skew())
print(df['Marks'].describe())
print("GRAPH")
plt.boxplot(df["Marks"])
plt.show()
