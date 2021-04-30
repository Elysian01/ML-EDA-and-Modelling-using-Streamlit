# from sklearn.model_selection import train_test_split
# from sklearn import datasets
import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.001, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        print("Logistic Regression Training Started...")

        # Gradient Descent
        for _ in range(self.epochs):
            # we take the linear model(y = m*x + c) and then apply sigmoid activation function to squeeze the output 0 and 1
            z = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(z)

            # Calculating derivatives
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

        print("Logistic Regression Training Completed")

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(z)
        y_predicted_class = [1 if y > 0.5 else 0 for y in y_predicted]
        return y_predicted_class

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


if __name__ == '__main__':
    pass
    # bc = datasets.load_breast_cancer()

    # X = bc.data
    # y = bc.target
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=1234)

    # # Model Testing
    # regressor = LogisticRegression(lr=0.0001, epochs=1000)
    # regressor.fit(X_train, y_train)
    # predictions = regressor.predict(X_test)
    # print("Logistic Regression Accuracy: ", accuracy(y_test, predictions))
