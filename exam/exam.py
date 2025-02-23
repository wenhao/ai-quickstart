import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class Exam:
    def main(self):
        data = pd.read_csv('examdata.csv')

        outputs = data.loc[:, 'Pass']
        inputs = data.drop('Pass', axis=1)

        plt.figure()
        mask = data.loc[:, 'Pass'] == 1
        passed = plt.scatter(data.loc[:, 'Exam1'][mask], data.loc[:, 'Exam2'][mask])
        failed = plt.scatter(data.loc[:, 'Exam1'][~mask], data.loc[:, 'Exam2'][~mask])
        plt.title('Exam1-Exam2')
        plt.xlabel('Exam1')
        plt.ylabel('Exam2')
        plt.legend((passed, failed), ('Pass', 'Failed'))

        logistic_regression = LogisticRegression()
        logistic_regression.fit(inputs, outputs)
        predict_outputs = logistic_regression.predict(inputs)
        accuracy = accuracy_score(outputs, predict_outputs)
        print('Accuracy:', accuracy)

        theta0 = logistic_regression.intercept_
        theta1 = logistic_regression.coef_[0][0]
        theta2 = logistic_regression.coef_[0][1]
        x1 = data.loc[:, 'Exam1']
        x2 = data.loc[:, 'Exam2']
        x2_new = -(theta0 + theta1 * x1) / theta2
        plt.plot(x1, x2_new)
        plt.show()

        x1_2 = x1 * x1
        x2_2 = x2 * x2
        x1_x2 = x1 * x2
        x_new = {'x1': x1, 'x2': x2, 'x1_2': x1_2, 'x2_2': x2_2, 'x1_x2': x1_x2}
        x_new = pd.DataFrame(x_new)
        logistic_regression2 = LogisticRegression()
        logistic_regression2.fit(x_new, outputs)
        predict_outputs2 = logistic_regression2.predict(x_new)
        accuracy2 = accuracy_score(outputs, predict_outputs2)

        x1_new = x1.sort_values()
        theta0 = logistic_regression2.intercept_
        theta1 = logistic_regression2.coef_[0][0]
        theta2 = logistic_regression2.coef_[0][1]
        theta3 = logistic_regression2.coef_[0][2]
        theta4 = logistic_regression2.coef_[0][3]
        theta5 = logistic_regression2.coef_[0][4]

        a = theta4
        b = theta5 * x1_new + theta2
        c = theta0 + theta1 * x1_new + theta3 * x1_new * x1_new
        x2_new_boundary = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
        figure = plt.figure()
        passed = plt.scatter(data.loc[:, 'Exam1'][mask], data.loc[:, 'Exam2'][mask])
        failed = plt.scatter(data.loc[:, 'Exam1'][~mask], data.loc[:, 'Exam2'][~mask])
        plt.title('Exam1-Exam2')
        plt.xlabel('Exam1')
        plt.ylabel('Exam2')
        plt.legend((passed, failed), ('Pass', 'Failed'))
        plt.plot(x1_new, x2_new_boundary)
        plt.show()


if __name__ == '__main__':
    exam = Exam()
    exam.main()
