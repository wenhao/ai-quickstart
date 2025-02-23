import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

class BostonHouse:
    def main(self):
        data = pd.read_csv("usa_housing_price.csv")

        inputs = data.drop(['Price'], axis=1)
        outputs = np.array(data.loc[:, 'Price']).reshape(-1, 1)
        linear_regression = LinearRegression()
        linear_regression.fit(inputs, outputs)

        predict_outputs = linear_regression.predict(inputs)
        score = r2_score(outputs, predict_outputs)
        print(score)

        new_house = np.array([66000,3,6,20000,150]).reshape(1, -1)
        new_house_predict = linear_regression.predict(new_house)
        print("house price: ", new_house_predict)

        plt.figure()
        plt.scatter(outputs, predict_outputs)
        plt.show()

if __name__ == '__main__':
    BostonHouse().main()
