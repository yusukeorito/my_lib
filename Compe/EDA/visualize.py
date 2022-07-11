import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



train = pd.DataFrame()
test = pd.DataFrame()

#単一数値変数の分布を確認
fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(train['target'], ax=ax, label='target')
ax.legend()
ax.grid()



#2つのベン図なのでvenn2
from matplotlib_venn import venn2
from typing import List, Set

#trainとtestにおけるuniqueな値の重複の可視化
def get_uniques(input_df:pd.DataFrame, column) -> Set:
    s = input_df[column]
    return set(s.dropna().unique())

#ベン図の可視化関数
def plot_intersection(left:pd.DataFrame, right:pd.DataFrame,target_column:str,ax:plt.Axes=None,set_labels:List[str]=None):
    venn2(subsets=(get_uniques(train, target_column), get_uniques(test, target_column)),
    set_labels=set_labels or ("Train", "Test"),
    ax=ax
    )
    ax.set_title(target_column)

#カテゴリ変数あるいはユニーク数の少ない数値変数をまとめて可視化
cat_cols = []
n_cols = 5
n_rows = -(- len(cat_cols) // n_cols) #切り上げ

fig, axes = plt.subplots(figsize=(4 * n_cols, 3 * n_rows), ncols=n_cols, nrows=n_rows)
for c, ax in zip(cat_cols, np.ravel(axes)):
    plot_intersection(train, test, target_column=c, ax=ax)



