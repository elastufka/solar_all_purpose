import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

#from datetime import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Markdown

def get_xspec_model_params(model_component, norm=False):
    '''Returns tuple of current values of xspec model component parameters.
    Input: xspec Component object'''
    if not norm:
        return tuple([getattr(model_component,p).values[0] for p in model_component.parameterNames if p!='norm'])
    else:
        return tuple([getattr(model_component,p).values[0] for p in model_component.parameterNames])
        
def get_xspec_model_sigmas(model_component):
    '''Returns tuple of current values of xspec model component parameter sigmas.
    Input: xspec Component object'''
    return tuple([getattr(model_component,p).sigma for p in model_component.parameterNames])
        
def show_model(model):
    '''equivalant of pyxspec show() but in Markdown for Jupyter
    Input: xspec Model object'''
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
            val=getattr(param,'values')[0]
            fmt=".2e"
            if np.abs(np.log10(val)) < 2:
                fmt=".2f"
            if getattr(param,'frozen'):
                plusminus='frozen'
            else:
                plusminus= f"+/-{getattr(param,'sigma'):{fmt}}"
            mdtable+=f"|{j+1+nprev} |{i+1} | {n}|{p}| {getattr(param,'unit')}| {getattr(param,'values')[0]:{fmt}} | {plusminus}|\n"
    return Markdown(mdtable)
    
def show_error(model):
    '''show parameters and errors. Input: xspec Model object'''
    
    tclout_errs={0:"new minimum found",1:"non-monotonicity detected",2:"minimization may have run into problem",3:"hit hard lower limit",4:"hit hard upper limit",5:"    parameter was frozen",6:"search failed in -ve direction",7:"search failed in +ve direction",8:"reduced chi-squared too high"}

    mdtable="|Model par| Model comp | Component|  Parameter|  Unit | Value| Upper Bound | Lower Bound | Calculation Error |\n |---|---|---|---|---|---|---|---|---|\n"
    for i,n in enumerate(model.componentNames):
        nprev=0
        if i>0:
            try:
                nprev=len(getattr(model,model.componentNames[i-1]).parameterNames)
            except IndexError:
                nprev=0
        for j,p in enumerate(getattr(model,n).parameterNames):
            param=getattr(getattr(model,n),p)
            err= getattr(param,'error')
            fmt=".2e"
            if np.abs(np.log10(getattr(param,'values')[0])) < 2:
                fmt=".2f"
            errcodes="<br>".join([tclout_errs[i] for i,e in enumerate(err[2]) if e=='T' ])
            upper=err[1]
            lower=err[0]
            mdtable+=f"|{j+1+nprev} |{i+1} | {n}|{p}| {getattr(param,'unit')}| {getattr(param,'values')[0]:{fmt}} | {upper:{fmt}}| {lower:{fmt}} | {errcodes}\n"
    return Markdown(mdtable)
    
def show_statistic(fit):
    '''input xspec.Fit'''
    return Markdown(f"Fit statistic: {fit.statMethod.capitalize()}   {fit.statistic:.3f} \n Null hypothesis probability of {fit.nullhyp:.2e} with {fit.dof} degrees of freedom")

def plot_fit(xspec,fitrange=False, dataGroup=1,erange=False,res_range=[-50,50],title=False,annotation=False):
    '''plot data, fit, residuals in PlotLy. Input: xspec global object '''
    xspec.Plot.xAxis = "keV"
    xspec.Plot('data')
    xx=xspec.Plot.x()
    yy=xspec.Plot.y()
    model_name=xspec.AllModels(dataGroup).expression #what if 2 models?
    
    if not fitrange:
        fitrange=[xx[0],xx[-1]]
    if not erange:
        erange=[xx[0],xx[-1]]
    if not title:
        title=f"Fit with {model_name}"
    xl=np.where(np.array(xx) > erange[0])[0][0]
    try:
        xg=np.where(np.array(xx) >= erange[1])[0][0]
    except IndexError:
        xg=len(xx)-1
    yrange=[np.floor(np.log10(yy[xl:xg])).min(),np.ceil(np.log10(yy[xl:xg])).max()]
    
    fig = make_subplots(rows=2, cols=1, start_cell="top-left",shared_xaxes=True,row_heights=[.6,.3],vertical_spacing=.05)
    fig.add_trace(go.Scatter(x=xx,y=yy,mode='markers',name='data',error_y=dict(type='data',array=xspec.Plot.yErr())),row=1,col=1)
    fig.add_trace(go.Scatter(x=xx,y=xspec.Plot.model(),name=model_name),row=1,col=1)

    fig.update_yaxes(type='log',row=1,col=1,showexponent = 'all',exponentformat = 'e',range=yrange)
    fig.add_trace(go.Scatter(x=xx,y=np.subtract(yy,xspec.Plot.model()),mode='markers',marker_color='brown',name='data-model'),row=2,col=1)
    fig.add_vrect(x0=fitrange[0],x1=fitrange[1],annotation_text='fit range',fillcolor='lightgreen',opacity=.25,line_width=0,row=1,col=1)
    fig.add_vrect(x0=fitrange[0],x1=fitrange[1],fillcolor='lightgreen',opacity=.25,line_width=0,row=2,col=1)
    if annotation:
        fig.add_annotation(x=1,y=1,text=annotation,xref='paper',yref='paper')
    fig.update_yaxes(title='Residuals',range=res_range,row=2,col=1)
    fig.update_xaxes(title='Energy (keV)',range=erange,row=2,col=1)
    fig.update_layout(width=500,height=700,title=title)
    return fig
    
def annotate_plot(model,norm=False):
    '''annotations for plot - parameters and confidence intervals if they can be calculated
    Input: xspec Model object
    Output: HTML-formatted string'''
    fittext=""
    for comp in model.componentNames:
        mc=getattr(model,comp)
        for par in getattr(mc,"parameterNames"):
            if par != 'norm':
                p=getattr(mc,par)
                val=p.values[0]
                fmt=".2e"
                if np.abs(np.log10(val)) < 2:
                    fmt=".2f"
                if p.error[2] == "FFFFFFFFF": #error calculated sucessfully
                    errs=f"({p.error[0]:{fmt}}-{p.error[1]:{fmt}})"
                else:
                    errs=""
                fittext +=f"{par}: {val:{fmt}} {errs} {p.unit}<br>"
    return fittext
