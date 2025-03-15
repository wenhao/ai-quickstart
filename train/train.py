import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

data_train = pd.read_csv('T-R-train.csv')
X_train = data_train.loc[:, 'T']
y_train = data_train.loc[:, 'rate']

figure1 = plt.figure(figsize=(5,5))
plt.scatter(X_train,y_train)
plt.xlabel('temperature')
plt.ylabel('rate')
plt.show()

X_train = np.array(X_train).reshape(-1, 1)
lr1 = LinearRegression()
lr1.fit(X_train,y_train)

data_test = pd.read_csv('T-R-test.csv')
X_test = data_test.loc[:, 'T']
y_test = data_test.loc[:, 'rate']

X_test = np.array(X_test).reshape(-1, 1)

y_train_predict = lr1.predict(X_train)
y_test_predict = lr1.predict(X_test)

r2_train = r2_score(y_train, y_train_predict)
r2_test = r2_score(y_test, y_test_predict)

X_range = np.linspace(40, 90, 300).reshape(-1, 1)
y_range_predict = lr1.predict(X_range)
figure2 = plt.figure(figsize=(10, 10))
plt.plot(X_range, y_range_predict)
plt.scatter(X_train, y_train)
plt.title('prediction Data')
plt.xlabel('temperature')
plt.ylabel('rate')
plt.show()

poly2 = PolynomialFeatures(degree=2)
X_2_train = poly2.fit_transform(X_train)
X_2_test = poly2.fit_transform(X_test)
lr2 = LinearRegression()
lr2.fit(X_2_train,y_train)
y_2_train_predict = lr2.predict(X_2_train)
y_2_test_predict = lr2.predict(X_2_test)
r2_2_train = r2_score(y_train, y_2_train_predict)
r2_2_test = r2_score(y_test, y_2_test_predict)

X_2_range = np.linspace(40, 90, 300).reshape(-1, 1)
X_2_range = poly2.transform(X_2_range)
y_2_range_predict = lr2.predict(X_2_range)
figure3 = plt.figure(figsize=(10, 10))
plt.plot(X_range, y_2_range_predict)
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.title('prediction Data')
plt.xlabel('temperature')
plt.ylabel('rate')
plt.show()

poly5 = PolynomialFeatures(degree=5)
X_5_train = poly5.fit_transform(X_train)
X_5_test = poly5.fit_transform(X_test)
lr5 = LinearRegression()
lr5.fit(X_5_train,y_train)
y_5_train_predict = lr5.predict(X_5_train)
y_5_test_predict = lr5.predict(X_5_test)
r2_5_train = r2_score(y_train, y_5_train_predict)
r2_5_test = r2_score(y_test, y_5_test_predict)
print(r2_5_train, r2_5_test)
X_5_range = np.linspace(40, 90, 300).reshape(-1, 1)
X_5_range = poly5.transform(X_5_range)
y_5_range_predict = lr5.predict(X_5_range)
figure4 = plt.figure(figsize=(10, 10))
plt.plot(X_range, y_5_range_predict)
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.title('prediction Data')
plt.xlabel('temperature')
plt.ylabel('rate')
plt.show()