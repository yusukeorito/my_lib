import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from Compe.train.metric import *


def fit_lgb(X:pd.DataFrame, y:pd.DataFrame,cv, model_params:dict,fit_params:dict, metric:function, fobj=None, feval=None):

    X, y = X.values, y.values

    """LightGBMをcross validationで学習"""
    models = []
    n_records = len(X)

    oof_pred = np.zeros((n_records, ), dtype=np.float)

    for i, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        tr_X, tr_y = X[tr_idx], y[tr_idx]
        va_X, va_y = X[va_idx], y[va_idx]

        train_data = lgb.Dataset(tr_X, label=tr_y)
        valid_data = lgb.Dataset(va_X, label=va_y)

        model = lgb.train(
                model_params,
                train_data,
               **fit_params,
                valid_names=['train', 'valid'],
                valid_sets=[train_data, valid_data],
                feval=feval #自作の損失関数を適用
            )
        
       
        pred_i = model.predict(va_X, num_iteration=model.best_iteration)
        oof_pred[va_idx] = pred_i
        models.append(model)

        score = metric(va_y, pred_i)
        print(f" - fold{i+1} score - {score:.4f}")

   
    score = metric(y, oof_pred)
    print(f"OOF score - {score:.4f}")   
    return oof_pred, models


def predict_lgb(models, test_X:pd.DataFrame):
    preds = np.array([model.predict(test_X) for model in models])
    pred = np.mean(preds, axis=0) 
    return pred

def make_sub(sub:pd.DataFrame, pred:np.ndarray, Path):
    sub['prediction'] = pred
    sub.to_csv(Path)



if __name__ == '__main__':
    train = pd.read_csv('')
    test = pd.read_csv('')
    sub = pd.read_csv('')
    
    train_X = train.drop(columns=['target'])
    train_y = train['target']
    test_X = test.copy()

    kf = KFold(n_splits=5, shuffle=False, random_state=2022)
    
    lgb_oof, lgb_models = fit_lgb(train_X, train_y, cv=kf, model_params=None, fit_params=None, metric=AUC)
    lgb_pred = predict_lgb(lgb_models, test_X)
    make_sub(sub, lgb_pred)

    