
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data from CSV file
data = pd.read_csv(r"Iris.csv")
print(data)
group=data.groupby('Species')


# Filter the data for the desired species
setosa_data = group.get_group('Setosa')
versicolor_data = group.get_group('Versicolor')
virginica_data = group.get_group('Virginica')

# Display basic statistical details for each species
print("Summary statistics for Iris-setosa:")
print(setosa_data.describe())

print("\nSummary statistics for Iris-versicolor:")
print(versicolor_data.describe())

print("\nSummary statistics for Iris-virginica:")
print(virginica_data.describe())
for i in range(1,5):
    sns.boxplot(data=data, x='Species', y=i)
    plt.show()

# Create boxplots for each feature
'''sns.boxplot(data=data, x='Species', y='SepalLength')
sns.boxplot(data=data, x='Species', y='SepalWidth')
sns.boxplot(data=data, x='Species', y='PetalLength')
sns.boxplot(data=data, x='Species', y='PetalWidth')'''

