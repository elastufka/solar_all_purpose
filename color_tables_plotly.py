"""
convert sunpy's matplotlib colormaps for use in plotly.

sunpy visualization module color_tables.py:

converting matplotlib colorscales to plotly: https://plotly.com/python/v3/matplotlib-colorscales/

"""
import matplotlib.colors as colors
import numpy as np
from sunpy.visualization.colormaps import cm,cmlist

def mplcmap_to_plotly(cmname:str, pl_entries:int=255):
    cmap=cmlist[cmname]
    
    cmap_rgb = []
    norm = colors.Normalize(vmin=0, vmax=255)

    for i in range(0, 255):
           k = colors.colorConverter.to_rgb(cmap(norm(i)))
           cmap_rgb.append(k)

    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap_rgb[k])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale
