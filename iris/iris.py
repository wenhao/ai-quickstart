import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv('iris_data.csv')
print(data.head())
X = data.drop(['target', 'label'], axis=1)
y = data.loc[:, 'label']

KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X, y)
y_predict = KNN.predict(X)
accuracy = accuracy_score(y, y_predict)

X_norm = StandardScaler().fit_transform(X)

x1_mean = X.loc[:,'sepal length'].mean()
x1_sigma = X.loc[:,'sepal length'].std()
x1_norm_mean = X_norm[:,0].mean()
x1_norm_sigma = X_norm[:,0].std()
print(x1_mean, x1_sigma, x1_norm_mean , x1_norm_sigma)

figure1 = plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.hist(X.loc[:,'sepal length'], bins=100)
plt.subplot(122)
plt.hist(X_norm[:,0], bins=100)
plt.show()

pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_norm)
var_ratio = pca.explained_variance_ratio_

figure2 = plt.figure(figsize=(20, 5))
plt.bar([1,2,3,4], var_ratio)
plt.xticks([1,2,3,4], ['PC1', 'PC2', 'PC3', 'PC4'])
plt.ylabel('Variance Ratio Of Each PC')
plt.show()

pca = PCA(n_components=2)
X_pac = pca.fit_transform(X_norm)

figure3 = plt.figure(figsize=(20, 5))
setosa = plt.scatter(X_pca[:, 0][y == 0], X_pca[:, 1][y == 0])
versicolor = plt.scatter(X_pca[:, 0][y == 1], X_pca[:, 1][y == 1])
virginica = plt.scatter(X_pca[:, 0][y == 2], X_pca[:, 1][y == 2])
plt.legend([setosa, versicolor, virginica], ['setosa', 'versicolor', 'virginica'])
plt.show()

KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_pac, y)
y_predict = KNN.predict(X_pac)
accuracy = accuracy_score(y, y_predict)
print(accuracy)