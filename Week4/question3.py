# WEEK 4 PRACTICE PROBLEMS

# TODO
# 3. Create a histogram with pandas for using MEDV in the housing data.

# 	a. Set the bins to 20

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

housing_csv = open('Week4/data/boston_housing_data.csv')
housing = pd.read_csv(housing_csv)


plt.figure(figsize=(11,4))
# set bins to 20
plt.hist(housing['MEDV'], bins=20)
plt.title('Histogram', fontsize=20)
plt.xlabel('MEDV', fontsize=14)
plt.ylabel('Counts', fontsize=14)

plt.show()



