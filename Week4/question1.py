# WEEK 4 PRACTICE PROBLEMS

import pandas as pd;
import matplotlib.pyplot as plt;

# TODO
# 1. Create a line plot of ZN and INDUS in the housing data.

# 	a. For ZN, use a solid green line. For INDUS, use a blue dashed line.
# 	b. Change the figure size to a width of 12 and height of 8.
# 	c. Change the style sheet to something you find https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html.

# found style from link
plt.style.use('bmh')

# increase default figure and font size
# width 12, height 8
plt.rcParams['figure.figsize'] = (12,8)
plt.rcParams['font.size'] = 14

# import and read housing csv
housing_csv = open('Week4/data/boston_housing_data.csv')
housing = pd.read_csv(housing_csv)

# read in housing data
# housing['ZN'] = housing.ZN
# housing['INDUS'] = housing.INDUS

# plot ZN - solid green line
plt.plot(housing['ZN'], label='ZN', color='green', linestyle='-')
# plot INDUS - blue dashed line
plt.plot(housing['INDUS'], label='INDUS', color='blue', linestyle='--')

# display line plot
plt.show()


