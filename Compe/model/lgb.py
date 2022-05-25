from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from Compe.model.basemodel import GBDTModel
from typing_extensions import Literal



class LGBMModel(GBDTModel):
    def __init__(self, model_params: dict, train_params: dict, cat_cols: List[str]):
        self.model_params = model_params
        self.train_params = train_params
        self.cat_cols = cat_cols

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ):
        train_ds = lgb.Dataset(X_train, y_train, categorical_feature=self.cat_cols)
        valid_ds = lgb.Dataset(X_valid, y_valid, categorical_feature=self.cat_cols)

        model = lgb.train(
            params=self.model_params,
            train_set=train_ds,
            valid_sets=[train_ds, valid_ds],
            valid_names=["train", "valid"],
            **self.train_params,
        )
        return model

    def predict(self, model: lgb.Booster, X: pd.DataFrame):
        return model.predict(X)

    def _get_feature_importances(
        self, model_save_paths: List[str]
    ) -> Tuple[List[np.array], List[List[str]]]:
        feature_importances = []
        for path in model_save_paths:
            model = self.load(path)
            feature_importance_i = model.feature_importance(importance_type="gain")
            feature_importances.append(feature_importance_i)
        columns = model.feature_name()
        columns = [columns] * len(feature_importances)
        return feature_importances, columns
