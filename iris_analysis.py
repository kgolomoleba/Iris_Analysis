# Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Step 2: Load the Iris dataset
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("Dataset loaded successfully!\n")
except Exception as e:
    print("Error loading dataset:", e)

# Step 3: Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Step 4: Explore dataset structure
print("\nDataset Info:")
print(df.info())

# Step 5: Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Step 6: Clean the data if needed (Iris has no missing values, so this is just a check)
# df.dropna(inplace=True)
# df.fillna(method='ffill', inplace=True)

# Step 7: Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Grouping by species and getting the mean of numerical features
print("\nAverage values per species:")
print(df.groupby('species').mean())

# Step 8: Data Visualization

# 1. Line chart: Sepal length over samples (index)
plt.figure(figsize=(8, 4))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title("Sepal Length Over Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Bar chart: Average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram: Distribution of Sepal Width
plt.figure(figsize=(6, 4))
plt.hist(df['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter plot: Sepal length vs Petal length colored by species
plt.figure(figsize=(6, 4))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

print("\nFindings and Observations:")

print("- The Iris dataset has no missing values and clean data.")
print("- Basic statistics show that Sepal length ranges roughly from 4.3 to 7.9 cm.")
print("- Grouping by species reveals distinct average feature differences, for example, Setosa has smaller petal lengths than Virginica.")
print("- The line plot shows the trend of sepal length across samples but has no time component as data is not time-series.")
print("- The bar chart clearly shows Setosa species has the smallest average petal length.")
print("- The histogram of sepal width reveals a fairly normal distribution.")
print("- The scatter plot shows a positive correlation between sepal length and petal length, with species grouping visible by color.")


