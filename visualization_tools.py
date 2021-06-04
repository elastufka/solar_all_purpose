import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

import numpy as np
import pandas as pd

from datetime import datetime as dt
import glob
import plotly.graph_objects as go
import plotly
import plotly.colors
import plotly.io as pio
import plotly.express as px

import seaborn as sns

def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
    
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    # noinspection PyUnboundLocalVariable
    return plotly.colors.find_intermediate_color(
        lowcolor=low_color, highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb")
        
def continuous_colors(cscale='Plotly3',ncolors=256):
    plotly3_colors, _ = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.__dict__[cscale])
    colorscale = plotly.colors.make_colorscale(plotly3_colors)
    return [get_continuous_color(colorscale, intermed=i/ncolors) for i in range(ncolors)]

def dark_mode():
    '''activate dark pallette for matplotlib and plotly'''
    plt.style.use('dark_background')
    pio.templates.default = "plotly_dark"#"ggplot2"#"plotly_dark"
    
def default_mode():
    '''activate default pallette for matplotlib and plotly'''
    plt.style.use('default')
    pio.templates.default = "plotly"#"ggplot2"#"plotly_dark"
    
def plot_dem_2D(df,what='DEM'):
    ''' plot stuff in the DEM dataframe'''
    
    lgtkeys=[k for k in df.keys() if k.startswith('dem')]
    lgtkeys=[k for k in lgtkeys if '_m' not in k]
    lgtaxis=[float(k[4:]) for k in lgtkeys] #may not need this...

    fig=go.Figure()
    
    yvec=[df[str(t)+'_mean'].values[0] for t in lgtkeys]
    try:
        errory=dict(type='data', array=[df['edem_'+str(t)+'_mean'].values[0] for t in lgtaxis],visible=True,thickness=0.5)
        errorx=dict(type='data', array=[df['elogt_'+str(t)+'_mean'].values[0] for t in lgtaxis],visible=True,thickness=0.5)
    except KeyError:
        errory=None
        errorx=None
    fig.add_trace(go.Scatter(x=lgtaxis,y=yvec,error_x=errorx,error_y=errory,name=what))
 
    fig.update_layout(yaxis_title='Mean DEM',xaxis_title='Log T (K)',yaxis_type='log')
    return fig


def all_six_AIA(aialist,unmask=True):
    fig = plt.figure(figsize=(20, 7))
    for i, m in enumerate(aialist):
        ax = fig.add_subplot(1,6, i+1, projection=m.wcs)
        if unmask and m.mask.any():
            m.mask=None
        m.plot(axes=ax,title=m.meta['wavelnth'])
        xax = ax.coords[0]
        yax = ax.coords[1]
        if i !=3:
            xax.set_axislabel('')
        if i !=0:
            yax.set_axislabel('')
            ax.set_yticklabels([])
            

def dem_image(df,T):
    '''T is string temperature '''
    fig=px.imshow(df['dem_' + str(T)][0])
    return fig

def corr_plot(df):
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return f

