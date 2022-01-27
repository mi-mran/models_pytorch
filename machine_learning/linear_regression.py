# THIS IMPLEMENTATION IS USING GRADIENT DESCENT ONLY, NOT OLS

import dataset
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # linear regression prediction
            y_pred = np.dot(X, self.weights) + self.bias

            # change in weights
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            # change in bias
            db = (2 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias

        return y_pred

if __name__ == '__main__':
    # testing with randomly generated dataset
    # note that the dataset here fulfils the assumptions made
    # if using with other datasets, please conduct the necessary tests to ensure assumptions are met

    import sklearn.datasets
    from sklearn.model_selection import train_test_split
    from utils import mse

    X, y = sklearn.datasets.make_regression(n_samples=200, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    predicted_vals = regressor.predict(X_test)
    mse_val = mse(y_test, predicted_vals)

    print("Mean Squared Error: ", mse_val)