import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class BostonHouse:
    def main(self):
        data = pd.read_csv("USA_Housing.csv")
        real_prices = np.array(data.loc[:, "Price"]).reshape(-1, 1)
        multiple_factors = data.drop(["Price", "Address"], axis=1)

        linear_regression = LinearRegression()
        linear_regression.fit(multiple_factors, real_prices)

        new_house = np.array([79545.45857, 5.682861322, 7.009188143, 6, 23086.8005]).reshape(1, -1)
        new_house_predict = linear_regression.predict(new_house)
        print("house price: ", new_house_predict)

        predict_prices = linear_regression.predict(multiple_factors)
        score = r2_score(real_prices, predict_prices)
        print(score)

if __name__ == '__main__':
    BostonHouse().main()
