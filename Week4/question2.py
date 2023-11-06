# WEEK 4 PRACTICE PROBLEMS

import pandas as pd;
import matplotlib.pyplot as plt;
import numpy as np;

# TODO
# 2. Create a bar chart using col1 and col2 of dummy data.

# 	a. Give the plot a large title of your choosing.
# 	b. Move the legend to the lower-left corner.
# 	c. Do the same thing but with horizontal bars.
# 	d. Move the legend to the upper-right corner.

# create random number dataframe 
# 2D array with 10 rows 2 columns
dummydata = pd.DataFrame(np.random.randn(10,2),
    columns = ['col1','col2'])


# counts number of contents for each column
# col1_counts = dummydata['col1'].value_counts().sort_values()

# plots one bar chart
# dummydata.col1.value_counts().plot(kind='bar')

# side by side bar chart
# dummydata.groupby('index').value_counts().plot(kind='bar')

# side by side - sorted by column
# dummydata.groupby('columns').value_counts().sort_values().plot(kind='bar')


# using a dataframe:
ax = dummydata.plot(kind='bar', figsize=(11,4))

# title
ax.set_title('Random Data', fontsize=21,y=1)

# move legend to lower left corner
ax.legend(loc="lower left");

# x-axis labels
ax.set_xlabel('Random Numbers Chosen', fontsize=16)

# y-axis labels
ax.set_ylabel('How Often the Number was Chosen', fontsize=16)

plt.show()

# PART 2
# horizontal bars
ax = dummydata.plot(kind='barh', figsize=(11,4))

# move legend to upper right corner
ax.legend(loc='upper right')


plt.show()
