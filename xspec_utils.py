#######################################
#display_aia_dem.py
# Erica Lastufka 15/03/2018  

#Description: Because OSX doesn't play well with XQuartz and IDL sucks
#######################################

#######################################
# Usage:

######################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Markdown

def get_xspec_model_params(model_component, norm=False):
    if not norm:
        return tuple([getattr(model_component,p).values[0] for p in model_component.parameterNames if p!='norm'])
    else:
        return tuple([getattr(model_component,p).values[0] for p in model_component.parameterNames])
        
def get_xspec_model_sigmas(model_component):
    return tuple([getattr(model_component,p).sigma for p in model_component.parameterNames])
        
def show_model(model):
    '''equivalant of pyxspec show() but in Markdown for Jupyter'''
    mdtable="|Model par| Model comp | Component|  Parameter|  Unit |    Value| Sigma |\n |---|---|---|---|---|---| |\n"
    for i,n in enumerate(model.componentNames):
        nprev=0
        if i>0:
            try:
                nprev=len(getattr(model,model.componentNames[i-1]).parameterNames)
            except IndexError:
                nprev=0
        for j,p in enumerate(getattr(model,n).parameterNames):
            param=getattr(getattr(model,n),p)
            if getattr(param,'frozen'):
                plusminus='frozen'
            else:
                plusminus= f"+/-{getattr(param,'sigma'):.2e}"
            mdtable+=f"|{j+1+nprev} |{i+1} | {p}|{n}| {getattr(param,'unit')}| {getattr(param,'values')[0]:.2e} | {plusminus}|\n"
    return Markdown(mdtable)
    
def show_statistic(fit):
    '''input xspec.Fit'''
    return Markdown(f"Fit statistic: {fit.statMethod.capitalize()}   {fit.statistic:.3f} \n Null hypothesis probability of {fit.nullhyp:.2e} with {fit.dof} degrees of freedom")
