import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

import numpy as np
import pandas as pd

from astropy import units as u
from datetime import datetime as dt
import glob
import plotly.graph_objects as go
import plotly
import plotly.colors
import plotly.io as pio
import plotly.express as px

import seaborn as sns
from pride_colors import *
        
def continuous_colors(cscale='Plotly3',ncolors=256):
    plotly3_colors, _ = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.__dict__[cscale])
    colorscale = plotly.colors.make_colorscale(plotly3_colors)
    return [get_continuous_color(colorscale, intermed=i/ncolors) for i in range(ncolors)]

def dark_mode(mpl=True,do_plotly=True):
    '''activate dark pallette for matplotlib and plotly'''
    if mpl:
        plt.style.use('dark_background')
    if do_plotly:
        pio.templates.default = "plotly_dark"#"ggplot2"#"plotly_dark"
        
def get_plotly_current_template():
    return pio.templates.default
    
def default_mode(mpl=True,do_plotly=True):
    '''activate default pallette for matplotlib and plotly'''
    if mpl:
        plt.style.use('default')
    if do_plotly:
        pio.templates.default = "plotly"#"ggplot2"#"plotly_dark"

def plotly_figsize(width=800,height=500):
    '''set default figure size for plotly figures'''
    current_template=get_plotly_current_template()
    tt=pio.templates[current_template]
    tt.layout['width']=width
    tt.layout['height']=height
    #tt.layout.paper_bgcolor='gray'
    

def plotly_logscale(xaxis=True,yaxis=True):
    '''set default log scale on indicated axis, with scientific notation'''
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


def all_six_AIA(aialist,unmask=True, use_mask=False,zoom=5, draw_contour=False):
    fig = plt.figure(figsize=(20, 7))
    for i, m in enumerate(aialist):
        ax = fig.add_subplot(1,6, i+1, projection=m.wcs)
        if unmask and m.mask.any():
            m.mask=None
        if type(use_mask) == np.ndarray:
            newdata=~use_mask*m.data
            nzx,nzy=np.where(newdata !=0)
            #nzx=[p[0] for p in nzpx]
            #nzy=[p[1] for p in nzpx]
            bl=m.pixel_to_world((np.min(nzx)-zoom)*u.pixel,(np.min(nzy)-zoom)*u.pixel)
            tr=m.pixel_to_world((zoom+np.max(nzx))*u.pixel,(zoom+np.max(nzy))*u.pixel)
            #print(bl,tr)
            newmap=sunpy.map.Map(newdata,m.meta).submap(bl,tr)
            m=newmap
        m.plot(axes=ax,title=m.meta['wavelnth'])
        if draw_contour != False:
            contour=m.draw_contours(levels=[draw_contour]*u.percent,axes=ax,frame=m.coordinate_frame)
        xax = ax.coords[0]
        yax = ax.coords[1]
        if i !=3:
            xax.set_axislabel('')
        if i !=0:
            yax.set_axislabel('')
            ax.set_yticklabels([])
    return fig
            

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

