# Consolidate all the ML utility functions here

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

import numpy as np
import pandas as pd

from datetime import datetime as dt
import glob
import plotly.graph_objects as go
import plotly.express as px

from sklearn import metrics

def normalize_minus1_plus1(arr):
    return 2.*(arr-np.min(arr))/np.ptp(arr)-1

### GMM ###

def run_gmm(nc,data,covariance_type='full',random_state=2): #should really be **kwargs
    colors=px.colors.qualitative.Bold[:nc]
    gmm = GaussianMixture(n_components=nc,covariance_type=covariance_type,
                          random_state=random_state,max_iter=500)
    model=gmm.fit(data)
    labels = gmm.predict(data)
    gdf['cluster']=labels
    return gdf, colors, model

def compare_clusters(n1,n2,df):
    c12=df.query('cluster==@n1 or cluster ==@n2').dropna(how='all')
    c1=df.where(df.cluster==n1).dropna(how='all')
    c2=df.where(df.cluster==n2).dropna(how='all')
    d12=c12[pkeys]
    return d12,c12

def compute_nonlabel_scores(X,clusters):
    s=metrics.silhouette_score(X, clusters, metric='euclidean')
    ch=metrics.calinski_harabasz_score(X, clusters)
    db=metrics.davies_bouldin_score(X, clusters)
    return s,ch,db

def gmm_notruth_score(nc, df):
    '''df must have column cluster '''
    scores={'n1':[],'n2':[],'s':[],'ch':[],'db':[]}
    for n1 in range(nc):
        k=n1
        for n2 in range(k,nc):
            if n1 != n2:
                d12,c12=compare_clusters(n1,n2,df)
                s,ch,db=compute_nonlabel_scores(d12,c12.cluster)
                scores['n1'].append(n1)
                scores['n2'].append(n2)
                scores['s'].append(s)
                scores['ch'].append(ch)
                scores['db'].append(db)
    sdf=pd.DataFrame(scores)
    sdf.sort_values(by='s',ascending=False,inplace=True)
    return sdf

### get all those Xgboost ones and put them here, put the plots elsewhere ###s


def print_arr_stats(arr, ignore_nan=False):
    if ignore_nan:
        print(f"Mean: {np.nanmean(arr)}\nMin: {np.nanmin(arr)}\nMax: {np.nanmax(arr)}\nStd: {np.nanstd(arr)}\n")
    else:
        print(f"Mean: {np.mean(arr)}\nMin: {np.min(arr)}\nMax: {np.max(arr)}\nStd: {np.std(arr)}\n")

def scale_minus1_plus1(arr,elementwise=False):
    """scale an array to [-1,+1]. elementwise (axis 0) if desired """
    if not elementwise:
        return 2*(arr-np.min(arr))/np.ptp(arr)-1
    else:
        return np.array([2*(a-np.min(a))/np.ptp(a)-1 for a in arr])
