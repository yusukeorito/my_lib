import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import catboost as cat

#特徴量重要度可視化関数
def visualize_importance_cat(models, feat_train_df):
    """Catboost の model 配列の feature importance を plot する
    CVごとのブレを boxen plot として表現します.
    args:
        models:
            List of Catboost models
        feat_train_df:
            学習時に使った DataFrame
    """
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["feature_importance"] = model.get_feature_importance(cat.Pool(feat_train_df), type="PredictionValuesChange")
        _df["column"] = feat_train_df.columns
        _df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], 
                                          axis=0, ignore_index=True)

    order = feature_importance_df.groupby("column")\
        .sum()[["feature_importance"]]\
        .sort_values("feature_importance", ascending=False).index[:50]#上位50個を表示

    fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
    sns.boxenplot(data=feature_importance_df, 
                  x="feature_importance", 
                  y="column", 
                  order=order, 
                  ax=ax, 
                  palette="viridis", 
                  orient="h")
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Importance")
    ax.grid()
    fig.tight_layout()
    return fig, ax