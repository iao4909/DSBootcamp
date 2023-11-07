# WEEK 4 PRACTICE PROBLEMS

# TODO
# 4. Create a scatter plot of two heatmap entries that appear to have a very positive correlation.

# TODO
# 5. Now, create a scatter plot of two heatmap entries that appear to have negative correlation.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

housing_csv = open('Week4/data/boston_housing_data.csv')
housing = pd.read_csv(housing_csv)


housing_correlations = housing.corr();

# heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(housing_correlations, vmin=-1, vmax=1)
plt.title('Correlation Heatmap for Housing Data')
plt.show()

# scatterplot POS
positive_correlation_columns = ['TAX', 'CRIM']
plt.figure(figsize=(10, 8))
sns.scatterplot(data=housing, x='TAX', y='CRIM')
plt.xlabel('TAX')
plt.ylabel('CRIM')
plt.title('Scatter Plot with Positive Correlation')
plt.show()

# scatterplot NEG
negative_correlation_columns = ['LSTAT', 'MEDV']
plt.figure(figsize=(10, 8))
sns.scatterplot(data=housing, x='LSTAT', y='MEDV')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.title('Scatter Plot with Negative Correlation')
plt.show()
