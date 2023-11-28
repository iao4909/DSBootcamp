import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

glass = pd.read_csv('glass.csv')


# #####LOGISTIC REGRESSION#########
glass['household'] = glass.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})
glass.household.value_counts()

glass.sort_values( by = 'Al', inplace=True)
X= np.array(glass.Al).reshape(-1,1)
y = glass.household
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X,y)
pred = logreg.predict(X)
logreg.coef_, logreg.intercept_

glass.sort_values( by = 'Al', inplace=True)

logreg.predict_proba(X)[:15]
glass['household_pred_prob'] = logreg.predict_proba(X)[:, 1]

# 1. Try different thresholds for computing predictions. By default it is 0.5. Use predict_proba function to compute probabilities and then try custom thresholds and see their impact on Accuracy, Precision and Recall
#   Change the threshold from 0.5 to 0.8
new_threshold = 0.8
adjusted_predictions = (pred > new_threshold).astype(int)

new_threshold_2 = 0.3
adjusted_predictions_2 = (pred > new_threshold_2).astype(int)

from sklearn import metrics
cm = metrics.confusion_matrix(y_true=y, y_pred=adjusted_predictions)

Accuracy = (cm[0,0]+ cm[1,1])/ (np.sum(cm))
Precision = (cm[1,1])/ (np.sum(cm[: , 1]))

# 2. Do the same analysis for other columns

glass.sort_values( by = 'Mg', inplace=True)
X2= np.array(glass.Al).reshape(-1,1)
y2 = glass.household
logreg.fit(X2,y2)
pred = logreg.predict(X2)

cm = metrics.confusion_matrix(y_true=y, y_pred=adjusted_predictions)

Accuracy = (cm[0,0]+ cm[1,1])/ (np.sum(cm))
Precision = (cm[1,1])/ (np.sum(cm[: , 1]))

# 3. Fit a Logistic Regression Model on all features. Remember to preprocess data(eg. normalization and one hot encoding)
features = glass.drop(['Type', 'household'], axis=1)
logreg_all_features = LogisticRegression()
logreg_all_features.fit(features, y)

# 4. Plot ROC Curves for each model

fpr, tpr, _ = roc_curve(y2, logreg.predict_proba(X2)[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')

# Plot the class predictions.
plt.scatter(glass.Al, glass.household)
plt.plot(glass.Al, pred, color='red', alpha=0.5)
plt.xlabel('al')
plt.ylabel('household')

# Plot the predicted probabilities.
plt.scatter(glass.Al, glass.household)
plt.plot(glass.Al, glass.household_pred_prob, color='red')
plt.xlabel('al')
plt.ylabel('household')


# ######CLUSTERING#########
# 1. Repeat the above exercise for different values of k
#  - How do the inertia and silhouette scores change?
#  - What if you don't scale your features?
#  - Is there a 'right' k? Why or why not?
# there isnt a corredct value of k, It depends on the data and you have to chose it based on that.
# if you don't scale features variables with larger scales may disproportionately influence the resultsand lead to biased model performance.

%matplotlib inline 

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn import cluster, datasets, preprocessing, metrics
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
# Check out the dataset and our target values
df = pd.read_csv("glass.csv")

cols = df.columns[:-1]
sns.pairplot(df[cols])
X_scaled = preprocessing.MinMaxScaler().fit_transform(df[cols])

k = 2
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(X_scaled)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_
metrics.silhouette_score(X_scaled, labels, metric='euclidean')
df['label'] = labels
# 2. Repeat the following exercise for food nutrients dataset