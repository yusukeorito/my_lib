#クラスタリング分析
import pandas as pd
import numpy as np
import folium
from sklearn import cluster

#階層的クラスタリング分析
from sklearn.cluster import AgglomerativeClustering

# クラスターで色分けするための辞書
color_dic = {0:'red', 1:'blue' ,2:'green', 3:'purple', 4:'orange'}

#階層的クラスタリング分析を用いた地域のグループ分け
def show_cluster(input_df:pd.DataFrame,input_matrix:np.ndarray, use_cols:list,  n_clusters:int, linkage:str, zoom:int=4)->folium.Map:
    '''クラスター分析の結果をfoliumで表示
    input_df クラスターに分けたい地域別のDataFrame(例:国単位のdf)
    input_matrix 距離行列
    use_cols クラスター分けに用いるカラム
    n_clusters : クラスター数
    linkage : 階層的クラスタリングの手法
        手法(linkage)のイメージ
        single (最短距離法): 大きなクラスターができやすい。“Rich get richer”
        complete (最長距離法): 大きなクラスターができにくい
        ward: その中間
    zoom : foliumの縮尺の初期値
    '''
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(input_matrix)
    cluster_df = input_df[use_cols].copy()
    cluster_df['cluster'] = clustering.labels_

    # 以下、foliumで地図上に表示
    f = folium.Figure(width=300, height=300)

    # 表示の中心座標
    center_lat, center_lon =36, 135
    m = folium.Map([center_lat,center_lon], zoom_start=zoom).add_to(f)

    for i in cluster_df.index:
        caption = cluster_df.at[i, 'City']
        folium.CircleMarker(
            location = [cluster_df.at[i, "lat"],cluster_df.at[i, "lon"]],
            popup = caption,
            color = color_dic[cluster_df.at[i, 'cluster']],
            fill = True,
            radius = 6
            ).add_to(m)
    
    return m