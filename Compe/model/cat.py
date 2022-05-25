from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing_extensions import Literal


from catboost import CatBoost, Pool
from Compe.model.basemodel import GBDTModel

class CatModel(GBDTModel):
    def __init__(
        self,
        model_params: dict,
        train_params: dict,
        cat_cols: List[str],
        task_type: Literal["regression", "binary", "multiclass"],
    ):
        self.model_params = model_params
        self.train_params = train_params
        self.cat_cols = cat_cols
        self.task_type = task_type

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ):
        train_ds = Pool(X_train, y_train, cat_features=self.cat_cols)
        valid_ds = Pool(X_valid, y_valid, cat_features=self.cat_cols)

        model = CatBoost(self.model_params)
        model.fit(train_ds, eval_set=[valid_ds], **self.train_params)
        return model

    def predict(self, model: CatBoost, X: pd.DataFrame):
        if self.task_type == "regression":
            return model.predict(X)
        elif self.task_type == "binary":
            return model.predict(X, prediction_type="Probability")[:, 1]
        elif self.task_type == "multiclass":
            return model.predict(X, prediction_type="Probability")
        else:
            raise ValueError

    def _get_feature_importances(
        self, model_save_paths: List[str]
    ) -> Tuple[List[np.array], List[List[str]]]:
        feature_importances = []
        columns = []
        for path in model_save_paths:
            model = self.load(path)
            feature_importance_i = model.get_feature_importance(
                type="FeatureImportance"
            )
            columns.append(model.feature_names_)
            feature_importances.append(feature_importance_i)
        return feature_importances, columns