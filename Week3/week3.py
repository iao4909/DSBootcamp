# Define two custom numpy arrays, say A and B. Generate two new numpy arrays by stacking A and B vertically and horizontally.

import numpy as np

# 1. Define two custom numpy arrays, say A and B. Generate two new numpy arrays by stacking A and B vertically and horizontally.
A = np.array((8, 9, 10))
B = np.array((1, 3, 7))

C = np.hstack((A,B))
D = np.vstack((A,B))

# 2. Find common elements between A and B. [Hint : Intersection of two sets]
E = np.intersect1d(A, B)

# 3. Extract all numbers from A which are within a specific range. eg between 5 and 10. [Hint: np.where() might be useful or boolean masks]
F = np.where(8 < A, A, 0)

# 4. Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])


filtered = iris_2d[(iris_2d[:, 0] < 5.0) & (iris_2d[:, 2] > 1.5)]


import pandas as pd
# 1. From df filter the 'Manufacturer', 'Model' and 'Type' for every 20th row starting from 1st (row 0).
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
print(df[['Manufacturer', 'Model', 'Type']][::20])


# 2. Replace missing values in Min.Price and Max.Price columns with their respective mean.
minPriceAvg = df['Min.Price'].mean()
maxPriceAvg = df['Max.Price'].mean()

df['Min.Price'] = df['Min.Price'].fillna(minPriceAvg)
df['Max.Price'] = df['Max.Price'].fillna(maxPriceAvg)

# 3. How to get the rows of a dataframe with row sum > 100?
df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
row_sum = df.sum(axis=1)
print(df[row_sum > 100])


