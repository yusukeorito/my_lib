#２変数の関係の可視化

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.DataFrame()
use_cols = []



#カテゴリ変数×数値変数 -> 箱ひげ図(横向き)
fig, ax = plt.subplots(figsize=(8, 15))
sns.boxenplot(data=train,x='numeric', y='category',
              ax=ax, palette='viridis', orient='h')
ax.tick_params(axis='x', rotaion=90)
ax.set_title('numeric')
ax.grid()
fig.tight_layout()


#数値変数×数値変数 -> (ヒートマップ　セル上に相関係数を出力)
fig, ax = plt.subplots(figsize=(10, 10))
corr = train[use_cols].corr()
sns.heatmap(corr, annot=True, ax=ax, fmt=".3f", cmap="viridis",)
"""
args 
annot セルに値を出力
"""
