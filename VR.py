
import pandas as pd
import numpy as np
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


# Calculate the correlation matrix
corr = df.corr()

# Create a figure and axis
plt.figure(figsize=(10, 8))

# Create the heatmap
cax = plt.matshow(corr, cmap='coolwarm', alpha=0.8)

# Add colorbar
plt.colorbar(cax)

# Annotate the heatmap
for (i, j), val in np.ndenumerate(corr):
    plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='black')

# Set ticks and labels
plt.xticks(np.arange(len(corr)), corr.columns, rotation=45)
plt.yticks(np.arange(len(corr)), corr.columns)

# Set the title
plt.title('Correlation Matrix')

# Show the plot
plt.show()

