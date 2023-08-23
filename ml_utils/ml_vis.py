# Consolidate all the ML visualization functions here

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

import numpy as np
import pandas as pd

from datetime import datetime as dt
import glob
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pride_colors_plotly import *

from matplotlib.patches import Ellipse

        
def corr_plot(df,vmin=None,vmax=0.3):
    '''Plot correlation heatmap with seaborn '''
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=vmin,vmax=vmax, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return f

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance. from sklearn example"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

def train_val_tf(history,train_key='loss',valkey='val_loss'):
    '''plot the training and validation loss, accuracy, etc, given a Keras/TensorFlow model'''
    fig,ax=plt.subplots(figsize=(8,6))
    ax.plot(history.history[train_key])
    ax.plot(history.history[value_key])
    ax.set_title(f'model {train_key}')
    ax.set_ylabel(f'{train_key}')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid'], loc='upper left')
    return fig

def kmeans_examples(data,clusters,ncluster=0,imshape=(256,256),nexamples=10):
    """Plot examples from each kmeans cluster. specifically for images but can probably work otherwise as well"""
    _,cluster_counts=np.unique(clusters, return_counts=True)
    n_clusters=len(cluster_counts)
    cluster_dict={f"cluster_{i}":list([j.reshape(imshape) for k,j in enumerate(data) if clusters[k]==i]) for i in range(n_clusters)}
    fig,ax=plt.subplots(nexamples,nexamples,figsize=(8,8))
    rng=np.random.default_rng()
    samples=rng.choice(cluster_counts[ncluster],size=nexamples*nexamples,replace=False) #no duplicates
    i=0
    for x in ax:
        for y in x:
            y.imshow(cluster_dict[f"cluster_{ncluster}"][samples[i]],cmap=cm.gray)
            y.axis('off')
            i+=1
    fig.suptitle(f"Cluster {ncluster}")
    return fig

def plot_kmeans_clusters(cluster_centers, cluster_ids):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=160)
    ax.scatter(x[:, 0], x[:, 1], c=cluster_ids, cmap='cool')
    ax.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1],
        c='white',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )
    #fig.axis([-1, 1, -1, 1])
    plt.tight_layout()
    return fig

def plot_confusion_matrix(labels, predictions):
    # change each element of z to type string for annotations
    cm = confusion_matrix(labels, predictions)
    z_text = [[str(y) for y in x] for x in cm]
    x = list(np.unique(labels))
    y = x
    # set up figure 
    fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                    yaxis_title = "Real value", yaxis_range = [max(y)+.5,min(y)-.5]
                    )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    return fig

def norm_label(lab):
    #return (lab - np.nanmean(lab))/np.nanstd(lab)
    p2,p98 = np.nanpercentile(lab,(2,98))
    labarr = np.array(lab)
    labarr[np.where(labarr <=p2)] = p2
    labarr[np.where(labarr >=p98)] = p98
    return (labarr-p2)/(p98-p2)