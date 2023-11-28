# Linear Regression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 101)

# load dataset
data = pd.read_csv("train.csv")

# drop the columns that dont have an impact on salary
data = data.drop(columns=['id', 'timestamp','country'])

# replace null vals with median
#   median is a safer value to use bc it is less affected by outliers
data.loc[data['hours_per_week'].isna(), 'hours_per_week'] = data['hours_per_week'].median()
data.loc[data['telecommute_days_per_week'].isna(), 'telecommute_days_per_week'] = data['telecommute_days_per_week'].median()

data = data.dropna()

# # joint plots for numeric variables
# cols = ["job_years", "hours_per_week"]
# for c in cols:
#     sns.jointplot(x=c, y="salary", data=data, kind = 'reg', height = 5)
# plt.show()

# # distributions showing job years and hrs per week
# cols = ["job_years", "hours_per_week"]
# for c in cols:
#     sns.distplot(data[c])
#     plt.grid()
#     plt.show()

# # distribution of target variable
# sns.distplot(data['salary'])
# plt.grid()
# plt.title('Distribution of Target Variable in Data')
# plt.show()
# print('max:', np.max(data['salary']))
# print('min:', np.min(data['salary']))

# create copy of data
data_train = data.copy()

# select categorical features
cat_cols = [c for c in data_train.columns if data_train[c].dtype == 'object' 
            and c not in ['is_manager', 'certifications']]
cat_data = data_train[cat_cols]
cat_cols

#Encoding binary variables (Yes=1, No=0)
#   can be problematic for linear models bc of the numerical values associated with them 
#       (encoded as 0-10 but 10 doesn't mean that the item encoded as 10 is higher)
binary_cols = ['is_manager', 'certifications']
for c in binary_cols:
    data_train[c] = data_train[c].replace(to_replace=['Yes'], value=1)
    data_train[c] = data_train[c].replace(to_replace=['No'], value=0)

# perform one hard encoding
#   creates a new column for each job title that existed and the value in that colum is 1 if it falls in to the category
final_data = pd.get_dummies(data_train, columns=cat_cols, drop_first= True)

# Test Split
y = final_data['salary']
X = final_data.drop(columns=['salary'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Normalize Data
#   bring them to one single range 0-1
#   this is bc they all have values in different scales

# select numerical features
num_cols = ['job_years','hours_per_week','telecommute_days_per_week']
num_cols

# Apply standard scaling on numeric data 
scaler = StandardScaler()
scaler.fit(X_train[num_cols])
X_train[num_cols] = scaler.transform(X_train[num_cols])

# Linear Regression
reg = LinearRegression()
reg.fit(X_train,y_train)

# # print coefficients
# reg.coef_
# # print intercept
# reg.intercept_
# mean absolute error
mean_absolute_error(y_train,reg.predict(X_train))
# mean squared error
mean_squared_error(y_train,reg.predict(X_train))**5



# 1. Preprocess Test data and get predictions
X_test[num_cols] = scaler.transform(X_test[num_cols])
y_pred = reg.predict(X_test)
# 2. Compute Mean Abolute Error, Mean Square error for test data
print(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred)**0.5)
# 3. Implement Ridge and Lasso Regression and then compute the following metrics on test data
# Ridge
# alpha= 1 (control parameter is large so coefficients go down)
ridge = Ridge(alpha=1)
ridge.fit(X_train,y_train)
y_pred = ridge.predict(X_test)
print(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred)**0.5)

plt.scatter(np.arange(len(np.sort(y_test))),np.sort(y_test), label='true')
plt.scatter(np.arange(len(np.sort(y_pred))),np.sort(y_pred), label = 'pred')
plt.legend()
ridge.coef_

# Lasso
lasso = Lasso(alpha=1)
lasso.fit(X_train,y_train)
y_pred = lasso.predict(X_test)
print(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred)**0.5)

plt.scatter(np.arange(len(np.sort(y_test))),np.sort(y_test))
plt.scatter(np.arange(len(np.sort(y_pred))),np.sort(y_pred))
lasso.coef_


# Trees

# train Decision Tree regression model
decisiontree = DecisionTreeRegressor(max_depth = 10, min_samples_split = 5)
decisiontree.fit(X_train, y_train)

#evaluating train error
mean_absolute_error(y_train,decisiontree.predict(X_train))

max_depth_list = [2,3,4,5,6,7,8,9,10,11,12,20]
train_error = []
test_error =[]

for md in max_depth_list:

    decisiontree = DecisionTreeRegressor(max_depth = md, min_samples_split = 2)
    decisiontree.fit(X_train, y_train)
    train_error.append(mean_absolute_error(y_train,decisiontree.predict(X_train)))
    test_error.append(mean_absolute_error(y_test,decisiontree.predict(X_test)))

plt.plot(max_depth_list,train_error,label = 'train error')
plt.plot(max_depth_list,test_error,label = 'test error')
plt.legend()

# Fitting a Random Forest Regressor
randomf = RandomForestRegressor(100,)
randomf.fit(X_train, y_train)
mean_absolute_error(y_train,randomf.predict(X_train))

# Compute errors on test sets
mean_absolute_error(y_test,decisiontree.predict(X_test))

# Play with different parameter of decision trees and random forests and see the impact on train and test error
# max_depth_list = [10,11,12,13,14,15,16,17,18,19,20]
max_depth_list = [0,1,2,3,4,5,6,7,8,9,10]
train_error = []
test_error =[]
N_estimator=[20,30,40,50,60,70,80,90,100]
for n in N_estimator:

    decisiontree = RandomForestRegressor(n_estimators=n, max_depth = 12, min_samples_split = 2)
    decisiontree.fit(X_train, y_train)
    train_error.append(mean_absolute_error(y_train,decisiontree.predict(X_train)))
    test_error.append(mean_absolute_error(y_test,decisiontree.predict(X_test)))

plt.plot(N_estimator,train_error,marker='o',label = 'train error')
plt.plot(N_estimator,test_error,marker='o',label = 'test error')
plt.legend()

pd.DataFrame({'feature':X_train.columns, "importance":randomf.feature_importances_*100}).sort_values(by='importance', ascending=False)


# [OPTIONAL] implement cross validation and get best hyperparameters
