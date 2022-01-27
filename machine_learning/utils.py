import numpy as np

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    return accuracy

def mse(y_true, y_pred):
    mse = np.mean(np.square(y_true - y_pred))

    return mse

def rmse(y_true, y_pred):
    rmse = np.sqrt(mse(y_true, y_pred))

    return rmse