import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mlp
from scipy.stats import norm
import numpy as np
from sklearn.covariance import EllipticEnvelope

data = pd.read_csv('anomaly_data.csv')

print(data.head())

figure = plt.figure()
plt.scatter(data.loc[:, 'x1'], data.loc[:, 'x2'])
plt.title('anomaly data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

x1 = data.loc[:, 'x1']
x2 = data.loc[:, 'x2']

font2 = {'family': 'Microsoft YaHei', 'weight': 'normal', 'size': '20'}
mlp.rcParams['font.family'] = 'Microsoft YaHei'
mlp.rcParams['axes.unicode_minus'] = False
figure2 = plt.figure(figsize=(20, 7))
plt.subplot(121)
plt.hist(x1, bins=100)
plt.title('x1 数据分布统计', font2)
plt.xlabel('x1', font2)
plt.ylabel('出现次数', font2)
plt.subplot(122)
plt.hist(x2, bins=100)
plt.title('x2 数据分布统计', font2)
plt.xlabel('x2', font2)
plt.ylabel('出现次数', font2)
plt.show()

x1_mean = x1.mean()
x1_sigma = x1.std()
x2_mean = x2.mean()
x2_sigma = x2.std()

print('x1_mean:', x1_mean, 'x1_sigma:', x1_sigma, 'x2_mean:', x2_mean, 'x2_sigma:', x2_sigma)

x1_range = np.linspace(0, 20, 300)
x1_normal = norm.pdf(x1_range, x1_mean, x1_sigma)
x2_range = np.linspace(0, 20, 300)
x2_normal = norm.pdf(x2_range, x2_mean, x2_sigma)
figure3 = plt.figure(figsize=(20, 7))
plt.subplot(121)
plt.plot(x1_range, x1_normal)
plt.title('x1 正态分布', font2)
plt.subplot(122)
plt.plot(x2_range, x2_normal)
plt.title('x2 正态分布', font2)
plt.show()

ad_model = EllipticEnvelope()
ad_model.fit(data)
y_predict = ad_model.predict(data)
print(pd.value_counts(y_predict))

figure4 = plt.figure(figsize=(20, 10))
original_data = plt.scatter(data.loc[:, 'x1'], data.loc[:, 'x2'], marker='x')
anomaly_data = plt.scatter(data.loc[:, 'x1'][y_predict == -1], data.loc[:, 'x2'][y_predict == -1], marker='o',
                      facecolor='none', edgecolor='red', s=150)
plt.title('anomaly data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend((original_data, anomaly_data), ('original data', 'anomaly data'))
plt.show()

ad_model = EllipticEnvelope(contamination=0.02)
ad_model.fit(data)
y_predict = ad_model.predict(data)
figure5 = plt.figure(figsize=(20, 10))
original_data = plt.scatter(data.loc[:, 'x1'], data.loc[:, 'x2'], marker='x')
anomaly_data = plt.scatter(data.loc[:, 'x1'][y_predict == -1], data.loc[:, 'x2'][y_predict == -1], marker='o',
                      facecolor='none', edgecolor='red', s=150)
plt.title('anomaly data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend((original_data, anomaly_data), ('original data', 'anomaly data'))
plt.show()