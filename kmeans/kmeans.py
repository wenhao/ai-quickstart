import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MeanShift, estimate_bandwidth

data = pd.read_csv('data.csv')
X = data.drop(['labels'], axis=1)
Y = data.loc[:, 'labels']

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

cluster_centers_ = kmeans.cluster_centers_

figure = plt.figure()
label0 = plt.scatter(X.loc[:, 'V1'][Y == 0], X.loc[:, 'V2'][Y == 0])
label1 = plt.scatter(X.loc[:, 'V1'][Y == 1], X.loc[:, 'V2'][Y == 1])
label2 = plt.scatter(X.loc[:, 'V1'][Y == 2], X.loc[:, 'V2'][Y == 2])
plt.title('labeled data')
plt.xlabel('V1')
plt.ylabel('V2')
plt.legend((label0, label1, label2), ('label0', 'label1', 'label2'))
plt.scatter(cluster_centers_[:, 0], cluster_centers_[:, 1])
plt.show()

y_predict_test = kmeans.predict([[80, 60]])
print(y_predict_test)

y_predict = kmeans.predict(X)
print(pd.value_counts(y_predict), pd.value_counts(Y))

score = accuracy_score(Y, y_predict)
print(score)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, Y)

y_predict_knn_test = knn.predict([[80, 60]])
knn_predict = knn.predict(X)

print(accuracy_score(Y, knn_predict))

bandwidth = estimate_bandwidth(X, n_samples=500)
mean_shift = MeanShift(bandwidth=bandwidth)
mean_shift.fit(X)

mean_shift_predict = mean_shift.predict(X)
print(pd.value_counts(mean_shift_predict), pd.value_counts(Y))
