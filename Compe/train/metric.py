import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)

def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def MSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def RMSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def MAPE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true)

def MER(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred) / y_true)

def AUC(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

#Binaryにまとめて計算
def Accuracy(y_true, y_pred):
    y_pred = np.round(y_pred, 0)
    return accuracy_score(y_true, y_pred)