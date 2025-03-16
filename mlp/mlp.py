import numpy as np
import pandas as pd
from keras.layers import Dense, Activation
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')
X = data.drop(['y'], axis=1)
y = data.loc[:, 'y']

figure1 = plt.figure(figsize=(5, 5))
passed = plt.scatter(X.loc[:, 'x1'][y == 1], X.loc[:, 'x2'][y == 1])
failed = plt.scatter(X.loc[:, 'x1'][y == 0], X.loc[:, 'x2'][y == 0])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('raw data')
plt.legend((passed, failed), ('Passed', 'Failed'))
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

mlp = Sequential()
mlp.add(Dense(units=20, input_dim=2, activation='sigmoid'))
mlp.add(Dense(units=1, activation='sigmoid'))
mlp.summary()
mlp.compile(loss='binary_crossentropy', optimizer='adam')
mlp.fit(X_train, y_train, epochs=10000)

y_train_predict = np.argmax(mlp.predict(X_train), axis=1)
x_train_score = accuracy_score(y_train, y_train_predict)
print('Train Accuracy:', x_train_score)

y_test_predict = np.argmax(mlp.predict(X_test), axis=1)
x_test_score = accuracy_score(y_test, y_test_predict)
print('Test Accuracy:', x_test_score)

xx, yy = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
x_range = np.c_[xx.ravel(), yy.ravel()]
y_range_predict = np.argmax(mlp.predict(x_range), aix=1)

pd.Series(i[0] for i in y_range_predict)
figure2 = plt.figure(figsize=(5, 5))
passed_predict = plt.scatter(x_range[:, 0][y_range_predict == 1], x_range[:, 1][y_range_predict == 1])
failed_predict = plt.scatter(x_range[:, 0][y_range_predict == 0], x_range[:, 1][y_range_predict == 0])
passed = plt.scatter(X.loc[:, 'x1'][y == 1], X.loc[:, 'x2'][y == 1])
failed = plt.scatter(X.loc[:, 'x1'][y == 0], X.loc[:, 'x2'][y == 0])
plt.legend((passed, failed, passed_predict, failed_predict), ('Passed', 'Failed', 'Passed_predict', 'Failed_predict'))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('raw data')
plt.show()
