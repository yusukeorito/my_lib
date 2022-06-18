import os
from pathlib import Path
from typing import List, Tuple
from typing_extensions import Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import catboost as cat
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from Compe.train.metric import *

def fit_cat(X:pd.DataFrame, y:pd.DataFrame,cv, model_params:dict,fit_params:dict, metric:function, 
categorical_list:List[str]=None,task:Literal["Regression", "Binary", "Multiclass"]=None, fobj=None,):

    X, y = X.values, y.values

    """LightGBMをcross validationで学習"""
    models = []
    n_records = len(X)
    oof_pred = np.zeros((n_records, ), dtype=np.float)

    for i, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        tr_X, tr_y = X[tr_idx], y[tr_idx]
        va_X, va_y = X[va_idx], y[va_idx]

        train_data = cat.Pool(tr_X, label=tr_y)
        valid_data = cat.Pool(va_X, label=va_y)

        if task=="Regression":
            model = cat.CatBoostRegressor(**model_params)
        elif task=="Binary" or task=="Multiclass":
            model = cat.CatBoostClassifier(**model_params)
        else:
            raise ValueError

        model.fit(
            train_data,
            **fit_params,
            plot=False,
            use_best_model=True,
            eval_set=[valid_data],
        )

        if task=="Regression":
            pred_i = model.predict(va_X)
        elif task == "Binary":
            pred_i = model.predict(X, prediction_type="Probability")[:, 1]
        elif task == "Multiclass":
            pred_i =  model.predict(X, prediction_type="Probability")
        else:
            raise ValueError
        

        oof_pred[va_idx] = pred_i
        models.append(model)

        score = metric(va_y, pred_i)
        print(f" - fold{i+1} score - {score:.4f}")

   
    score = metric(y, oof_pred)
    print(f"OOF score - {score:.4f}")   
    return oof_pred, models

def predict_cat(models, test_X:pd.DataFrame,task:Literal["Regression", "Binary", "Multiclass"]):
    if task=="Regression":
        preds = np.array([model.predict(test_X) for model in models])
        pred = np.mean(preds, axis=0) 
    elif task == "Binary":
        preds = np.array([model.predict(test_X, prediction_type="Probability")[:,1] for model in models])
        pred = np.mean(preds, axis=0) #確率の平均値を算出
    elif task == "Multiclass":
        preds = np.array([model.predict(test_X, prediction_type="Probability") for model in models])
        pred = np.mean(preds, axis=0) #クラスごとに確率の平均値を算出
    return pred