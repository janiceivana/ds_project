import pandas as pd
# Correlation between numerical variables
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Virtual_Reality_in_Education_Impact.csv')

# Display the first few rows
print(df.head())

# Fill missing values in numerical columns with median
numerical_columns = df.select_dtypes(include=['number']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# Fill missing values in non-numerical columns with the mode (most frequent value)
non_numerical_columns = df.select_dtypes(exclude=['number']).columns
for column in non_numerical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Outlier detection and removal using IQR
Q1 = df['Hours_of_VR_Usage_Per_Week'].quantile(0.25)
Q3 = df['Hours_of_VR_Usage_Per_Week'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Hours_of_VR_Usage_Per_Week'] < (Q1 - 1.5 * IQR)) | (df['Hours_of_VR_Usage_Per_Week'] > (Q3 + 1.5 * IQR)))]

print(df.head())

# Summary statistics
print(df.describe())


corr = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

