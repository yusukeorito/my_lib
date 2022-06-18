import os
import pandas as pd
import numpy as np
import typing as tp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix



#特徴量重要度を可視化する関数
def visualize_importance(models, feat_train_df):
    """lightGBM の model 配列の feature importance を plot する
    CVごとのブレを boxen plot として表現します.
    args:
        models:
            List of lightGBM models
        feat_train_df:
            学習時に使った DataFrame
    """
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        # _df["feature_importance"] = model.feature_importances_
        _df["feature_importance"] = model.feature_importance(importance_type="gain")
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



#混合行列を描画する関数(multiclass)
#確率ではなくラベルで渡す
def visualize_confusion_matrix(y_true, pred_label,
    ax: tp.Optional[plt.Axes] = None,
    labels: tp.Optional[list] = None,
    conf_options: tp.Optional[dict] = None,
    plot_options: tp.Optional[dict] = None
) -> tp.Tuple[plt.Axes, np.ndarray]:
    """
    visualize confusion matrix
    Args:
        y_true:
            True Label. shape = (n_samples, )
        pred_label:
            Prediction Label. shape = (n_samples, )
        ax:
            matplotlib.pyplot.Axes object.
        labels:
            plot labels
        conf_options:
            option kwrgs when calculate confusion matrix.
            pass to `confusion_matrix` (defined at scikit-learn)
        plot_options:
            option key-words when plot seaborn heatmap
    Returns:
    """

    _conf_options = {
        "normalize": "true",
    }
    if conf_options is not None:
        _conf_options.update(conf_options)

    _plot_options = {
        "cmap": "Blues",
        "annot": True
    }
    if plot_options is not None:
        _plot_options.update(plot_options)

    conf = confusion_matrix(y_true=y_true,
                            y_pred=pred_label,
                            **_conf_options)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf, ax=ax, **_plot_options)
    ax.set_ylabel("Label")
    ax.set_xlabel("Predict")

    if labels is not None:
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        ax.tick_params("y", labelrotation=0)
        ax.tick_params("x", labelrotation=90)

    return ax, conf