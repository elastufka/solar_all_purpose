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
