import mglearn

import pandas as pd
from pandas.plotting import scatter_matrix

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()

#  t_t_split helps us to split the dataset into two different samples, one for
# training the model and one for evaluating it, by default it splits into 3: 1
# 3: 1 ratio
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)


# to visualize the data we will be converting the ndarry to pandas dataframe
# labelling columns using strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
iris_dataframe = pd.DataFrame(X_train, columns=chd.columns)
# create a scatter matrix from the dataframe, color by y_train
grr = scatter_matrix(
    iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
    hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)

print("Test set predictions:\n {}".format(y_pred))

print("Test score {:.2f}".format(np.mean(y_pred == y_test)))
