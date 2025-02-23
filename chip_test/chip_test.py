import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class ChipTest:
    def main(self):
        data = pd.read_csv('chip_test.csv')
        inputs = data.drop(['pass'], axis=1)
        y = data.loc[:, 'pass']
        x1 = data.loc[:, 'test1']
        x2 = data.loc[:, 'test2']

        x1_2 = x1 * x1
        x2_2 = x2 * x2
        x1_x2 = x1 * x2
        x_new = {'x1': x1, 'x2': x2, 'x1_2': x1_2, 'x2_2': x2_2, 'x1_x2': x1_x2}
        x_new = pd.DataFrame(x_new)

        model = LogisticRegression()
        model.fit(x_new, y)
        y_predict = model.predict(x_new)
        score = accuracy_score(y, y_predict)
        print(score)

        x1_new = x1.sort_values()
        theta0 = model.intercept_[0]
        theta1 = model.coef_[0][0]
        theta2 = model.coef_[0][1]
        theta3 = model.coef_[0][2]
        theta4 = model.coef_[0][3]
        theta5 = model.coef_[0][4]
        a = theta4
        b = theta5 * x1_new + theta2
        c = theta0 + theta1 * x1_new + theta3 * x1_new * x1_new
        x2_new_boundary = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
        print(x2_new_boundary)

        plt.figure()
        mask = data.loc[:, 'pass'] == 1
        passed = plt.scatter(x1[mask], x2[mask])
        failed = plt.scatter(x1[~mask], x2[~mask])
        plt.plot(x1_new, x2_new_boundary)
        plt.title('Chip Test')
        plt.xlabel('Test1')
        plt.ylabel('Test2')
        plt.legend((passed, failed), ('passed', 'failed'))
        plt.show()
    def f(self, theta0, theta1, theta2, theta3, theta4, theta5, x1_new):
        a = theta4
        b = theta5 * x1_new + theta2
        c = theta0 + theta1 * x1_new + theta3 * x1_new * x1_new
        x2_new_boundary1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
        x2_new_boundary2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
        return x2_new_boundary1, x2_new_boundary2

if __name__ == '__main__':
    chip_test = ChipTest()
    chip_test.main()
