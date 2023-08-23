
import numpy as np
from datetime import datetime as dt
from datetime import timedelta as td
import os
import pandas as pd
#from sunpy import timeseries as ts
import netCDF4 as nc
from astropy.time import Time
from sunpy.time import parse_time

def netCDF_to_pandas(filename):
    """because sunpy Timeseries can't read GOES background files """
    ds = nc.Dataset(filename)
    bvars = list(ds.variables.keys())
    
    #deal with times
    start_time_str = str(ds.variables['time'])
    start_time_str = start_time_str[start_time_str.find("seconds since")+14:]
    start_time_str = start_time_str[:start_time_str.find("\n")].strip()
    times = Time(parse_time(start_time_str).unix + ds.variables['time'][:], format = 'unix').datetime

    empty_dict = {}
    for b in bvars:
        empty_dict[b]= ds[b][:]
    empty_dict['time'] = times
    aa = pd.DataFrame(empty_dict)
    return aa

def plot_goes_background(filename):
    """Plot the background from a given file"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    ds = netCDF_to_pandas(filename)
    
    htb = "%{x:%d-%b-%Y}<br>1-8 Å flux: %{y:.3e} (%{customdata})"
    hta = "%{x:%d-%b-%Y}<br>0.5-4 Å flux: %{y:.3e} (%{customdata})"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ds.time,y=ds.bkd1d_xrsa_flux, line_shape = 'hv', customdata = np.array([goes_flux_to_class(v) if ~np.isnan(v) else None for v in ds.bkd1d_xrsa_flux.values]), hovertemplate = hta,name = 'XRS bkg short'))
    fig.add_trace(go.Scatter(x=ds.time,y=ds.bkd1d_xrsb_flux, customdata = np.array([goes_flux_to_class(v) if ~np.isnan(v) else None for v in ds.bkd1d_xrsb_flux.values]), hovertemplate = htb, name= 'XRS bkg long'))
    fig.add_trace(go.Scatter(x=ds.time,y=ds.avg1d_xrsa_flux, customdata = np.array([goes_flux_to_class(v) if ~np.isnan(v) else None for v in ds.avg1d_xrsa_flux.values]), hovertemplate = hta,name = 'XRS avg short', visible = 'legendonly'), secondary_y=True)
    fig.add_trace(go.Scatter(x=ds.time,y=ds.avg1d_xrsb_flux, customdata = np.array([goes_flux_to_class(v) if ~np.isnan(v) else None for v in ds.avg1d_xrsb_flux.values]), hovertemplate = htb,name= 'XRS avg long', visible = 'legendonly'))
    
    fig.update_yaxes(type='log', range=[-9,-4], showexponent = 'all', exponentformat = 'e')
    fig.update_yaxes(type='log', range=[-9,-4], tickvals=[3e-8,3e-7,3e-6,3e-5,3e-4],ticktext=["A", "B", "C", "M", "X"], secondary_y=True)
    fig.update_yaxes(showgrid=False,secondary_y=True)
    #fig.update_layout(yaxis2_type='log', yaxis2_ticktext = ["A", "B", "C", "M", "X"])
    fig.update_layout(title='GOES XRS daily background',yaxis_title = 'Flux (Watts m<sup>-2</sup>)')
    
    return fig
