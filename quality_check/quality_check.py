import pandas as pd
from matplotlib import pyplot as plt
from sklearn.covariance import EllipticEnvelope

data = pd.read_csv('data_class_raw.csv')
X = data.drop(['y'], axis=1)
y = data.loc[:, 'y']

figure1 = plt.figure(figsize=(10, 10))
bad = plt.scatter(X.loc[:, 'x1'][y == 0], X.loc[:, 'x2'][y == 0])
good = plt.scatter(X.loc[:, 'x1'][y == 1], X.loc[:, 'x2'][y == 1])
plt.legend((good, bad), ['good', 'bad'])
plt.title('raw data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

ad_model = EllipticEnvelope(contamination=0.02)
ad_model.fit(X[y == 0])
y_predict_bad = ad_model.predict(X[y == 0])

figure2 = plt.figure(figsize=(10, 10))
bad = plt.scatter(X.loc[:, 'x1'][y == 0], X.loc[:, 'x2'][y == 0])
good = plt.scatter(X.loc[:, 'x1'][y == 1], X.loc[:, 'x2'][y == 1])
plt.scatter(X.loc[:, 'x1'][y == 0][y_predict_bad == -1], X.loc[:, 'x2'][y == 0][y_predict_bad == -1], marker='x', s=150)
plt.legend((good, bad), ['good', 'bad'])
plt.title('raw data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
