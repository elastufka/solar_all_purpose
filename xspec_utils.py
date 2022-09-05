import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

#from datetime import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Markdown
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table

def spectrum_from_time_interval(original_fitsfile, start_time, end_time, out_fitsname=None):
    """write average count rate over selected time interval to new fitsfile for fitting with XSPEC"""
    reference_fits = fits.open(original_fitsfile)
    primary_HDU = reference_fits[0] #header = reference_fits[0].header.copy()
    rate_header = reference_fits[1].header.copy()
    #energy_header = reference_fits[2].header.copy()
    rate_data = reference_fits[1].data.copy()
    #energy_data = reference_fits[2].data.copy()
    energy_HDU = reference_fits[2]
    nchan = len(rate_data['CHANNEL'][0])

    #time axis
    duration_seconds = rate_data['TIMEDEL'] #seconds
    duration_day = duration_seconds/86400
    time_bin_center = rate_data['TIME']
    t_start = Time([bc - d/2. for bc,d in zip(time_bin_center, duration_day)], format='mjd')
    t_end = Time([bc + d/2. for bc,d in zip(time_bin_center, duration_day)], format='mjd')
    t_mean = Time(time_bin_center, format='mjd')
    
    #index times that fall within selected interval
    tstart = Time(start_time)
    tend = Time(end_time)
    tselect = np.where(np.logical_and(time_bin_center >= tstart.mjd,time_bin_center < tend.mjd)) #boolean mask
    #ttimes = time_bin_center[tselect] #actual times
    print(f"tselect {tselect[0][0]} {tselect[0][-1]}")
    #rate data
    total_counts = np.array([np.sum(rate_data['RATE'][tselect],axis=0)]).reshape((1,nchan))
    print(f"max counts: {np.max(total_counts)}")
    
    #average livetime data - same number for each channel
    avg_livetime = np.array([np.mean(rate_data['LIVETIME'][tselect]) for n in range(nchan)]).reshape((1,nchan)) #np.array([np.mean(rate_data['LIVETIME'][tselect])]).reshape((1,))
    
    #error...
    avg_err = np.mean(rate_data['STAT_ERR'][tselect],axis=0).reshape((1,nchan))
    
    # Update keywords that need updating
    #rate_header['DETCHANS'] = self.n_energies
    rate_header.set('NAXIS',1)
    rate_header.set('NAXIS1', 1)
    del rate_header['NAXIS2']
    
    exposure =  np.sum(rate_data['TIMEDEL'][tselect[0]]*rate_data['LIVETIME'][tselect[0]])
    rate_header['EXPOSURE'] = exposure
    rate_header['ONTIME'] = exposure
    print(f"exposure: {exposure}")
    #update times in rate header
    rate_header['TSTARTI'] = int(np.modf(tstart.mjd)[1]) #Integer portion of start time rel to TIMESYS
    rate_header['TSTARTF'] = np.modf(tstart.mjd)[0] #Fractional portion of start time
    rate_header['TSTOPI'] = int(np.modf(tend.mjd)[1])
    rate_header['TSTOPF'] = np.modf(tend.mjd)[0]

    #update rate data
    print(f"max count rate: {np.max(total_counts/exposure)}")
    rate_names = ['RATE', 'STAT_ERR', 'CHANNEL', 'SPEC_NUM', 'LIVETIME', 'TIME', 'TIMEDEL']
    rate_table = Table([(total_counts/exposure).astype('>f8'), avg_err.astype('>f8'), rate_data['CHANNEL'][0].reshape((1,nchan)), [0],avg_livetime.astype('>f8'), np.array([rate_data['TIME'][tselect[0][0]]]), np.array([np.sum(rate_data['TIMEDEL'][tselect])])], names = rate_names) #is spec.counts what we want?

    #primary_HDU = fits.PrimaryHDU(header = primary_header)
    rate_HDU = fits.BinTableHDU(header = rate_header, data = rate_table)
    #energy_HDU = fits.BinTableHDU(header = energy_header, data = energy_data)
    #print(energy_HDU.header)
    hdul = fits.HDUList([primary_HDU, rate_HDU, energy_HDU]) #, att_header, att_table])
    if not out_fitsname:
        out_fitsname=f"{original_fitsfile[:-5]}_{pd.to_datetime(start_time):%H%M%S}-{pd.to_datetime(end_time):%H%M%S}.fits"
    hdul.writeto(out_fitsname)

#def select_background_interval():

def fit_thermal_nonthermal(xspec, ntmodel = 'bknpower', lowErange = [2.0,10.0], highErange = [8.0,30.0], breakEstart = 15, breakEfrozen=False, minCounts=10, statMethod='chi',query='no',renorm=True,nIterations = 1000):
    '''Fit thermal and non-thermal components to spectrum via the following steps:
        1) fit thermal over low energy
        2) fit non-thermal over high-energy with initial break energy frozen (if non-thermal model is bknpow or thick2)
            2a) unfreeze break energy and fit non-thermal again
        3) fit thermal and non-thermal together over entire energy range'''
    breakE = True #assume there's a break energy
    xspec.Xset.abund="felc"
    if ntmodel != 'thick2':
        xspec.AllModels.clear() #for now...
    #settings for fit
    xspec.Fit.statMethod = statMethod #Valid names: 'chi' | 'cstat' | 'lstat' | 'pgstat' | 'pstat' | 'whittle'.
    xspec.Fit.query = query
    xspec.Fit.nIterations = nIterations
        
    #step 1
    m = xspec.Model(f'apec')
    xspec.AllData.ignore(f"0.-{lowErange[0]} {lowErange[1]}-**")
    
    xspec.Fit.renorm()
    xspec.Fit.perform()
    
    mtherm_params = get_xspec_model_params(m.apec, norm = True)
    
    #step2 - fit non-thermal
    xspec.AllModels.clear()
    m = xspec.Model(f'apec+{ntmodel}')
    m_th = m.apec
    set_xspec_model_params(m, 'apec', mtherm_params, frozen = True)
        
    m_nt = getattr(m,ntmodel)
    try: #in thick2 it's eebrk not BreakE...
        breakEindex = getattr(m_nt, 'BreakE')._Parameter__index
        m.setPars({breakEindex:f"{breakEstart} -.5,,,{breakEstart+2}"})
        breakparname = 'BreakE'
        #p = getattr(m_nt,'BreakE')
    except AttributeError:
        try:
            breakEindex = getattr(m_nt, 'eebrk')._Parameter__index
            m.setPars({breakEindex:f"{breakEstart} -.5,,,{breakEstart+2}"})
            breakparname = 'eebrk'
            lowEindex = getattr(m_nt, 'eelow')._Parameter__index
            m.setPars({lowEindex:f"{breakEstart-5} -.5,,,{breakEstart-2}"})
            p = getattr(getattr(m,ntmodel),'eelow')
            p.frozen = False
        except AttributeError:
            breakE = False
 
    xspec.AllData.notice('all')
    #check that count rate at highErange is above minCounts, otherwise adjust highErange and warn
    #TBD
    #warn for negative count rate and zero errors while we're here
    #TBD
    xspec.AllData.ignore(f"0.-{highErange[0]} {highErange[1]}-**")
    
    xspec.Fit.renorm()
    xspec.Fit.perform()

    if breakE and not breakEfrozen: #fit again with unfrozen break E
        p = getattr(getattr(m,ntmodel),breakparname)
        p.frozen = False
        xspec.Fit.renorm()
        xspec.Fit.perform()
        
    #step 3 - fit together, all parameters free
    for param in ['kT','norm']:
        p = getattr(m.apec, param)
        p.frozen = False
        
    xspec.AllData.notice('all')
    xspec.AllData.ignore(f"0.-{lowErange[0]} {highErange[1]}-**")
    
    xspec.Fit.renorm()
    xspec.Fit.perform()
    print(f"Fit statistic: {xspec.Fit.statMethod.capitalize()}   {xspec.Fit.statistic:.3f} \n Null hypothesis probability of {xspec.Fit.nullhyp:.2e} with {xspec.Fit.dof} degrees of freedom")
    xspec.AllData.notice('all')
    return m


def get_xspec_model_params(model_component, norm=False):
    '''Returns tuple of current values of xspec model component parameters.
    Input: xspec Component object'''
    if not norm:
        return tuple([getattr(model_component,p).values[0] for p in model_component.parameterNames if p!='norm'])
    else:
        return tuple([getattr(model_component,p).values[0] for p in model_component.parameterNames])
        
def set_xspec_model_params(model, model_component, component_params, frozen = False):
    '''Sets current values of xspec model parameters.
    Input: xspec Model Component object, tuple of xspec model parameters'''
    mcomp = getattr(model, model_component)
    for param, pval in zip(mcomp.parameterNames, component_params):
        pidx =  getattr(mcomp, param)._Parameter__index
        model.setPars({pidx:f"{pval} -.1,,,{pval}"})
        if not frozen:
            p = getattr(mcomp, param)
            p.frozen = False
        
def get_xspec_model_sigmas(model_component):
    '''Returns tuple of current values of xspec model component parameter sigmas.
    Input: xspec Component object'''
    return tuple([getattr(model_component,p).sigma for p in model_component.parameterNames])
        
def show_model(model, df=False):
    '''equivalant of pyxspec show() but in Markdown for Jupyter
    Input: xspec Model object'''
    mdtable="|Model par| Model comp | Component|  Parameter|  Unit |    Value| Sigma |\n |---|---|---|---|---|---| |\n"
    pdict={'Model par':[], 'Model comp':[], 'Component':[], 'Parameter': [], 'Unit': [], 'Value': [], 'Sigma':[]}
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
                plusminus= f"± {getattr(param,'sigma'):{fmt}}"
            mdtable+=f"|{j+1+nprev} |{i+1} | {n}|{p}| {getattr(param,'unit')}| {getattr(param,'values')[0]:{fmt}} | {plusminus}|\n"
            pdict['Model par'].append(j+1+nprev)
            pdict['Model comp'].append(i+1)
            pdict['Component'].append(n)
            pdict['Parameter'].append(p)
            pdict['Unit'].append(getattr(param,'unit'))
            pdict['Value'].append(f"{getattr(param,'values')[0]:{fmt}}")
            pdict['Sigma'].append(plusminus)
            
    if df: #jupyterlab doesn't work with Mardown for whatever reason, but it will display a dataframe nicely
        return pd.DataFrame(pdict)
    else:
        return Markdown(mdtable)
    
def show_error(model):
    '''show parameters and errors. Input: xspec Model object'''
    
    tclout_errs={0:"new minimum found",1:"non-monotonicity detected",2:"minimization may have run into problem",3:"hit hard lower limit",4:"hit hard upper limit",5:"    parameter was frozen",6:"search failed in -ve direction",7:"search failed in +ve direction",8:"reduced chi-squared too high"}

    mdtable="|Model par| Model comp | Component|  Parameter|  Unit | Value| Lower Bound | Upper Bound | Calculation Error |\n |---|---|---|---|---|---|---|---|---|\n"
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
            mdtable+=f"|{j+1+nprev} |{i+1} | {n}|{p}| {getattr(param,'unit')}| {getattr(param,'values')[0]:{fmt}} | {lower:{fmt}}| {upper:{fmt}} | {errcodes}\n"
    return Markdown(mdtable)
    
def show_statistic(fit):
    '''input xspec.Fit'''
    return Markdown(f"Fit statistic: {fit.statMethod.capitalize()}   {fit.statistic:.3f} \n Null hypothesis probability of {fit.nullhyp:.2e} with {fit.dof} degrees of freedom")

def plot_data(xspec,fitrange=False, dataGroup=1,erange=False, counts=False, title = None):
    '''plot data in PlotLy. Input: xspec global object '''

    xspec.Plot.xAxis = "keV"
    #xspec.Plot('ufspec')
    #model = xspec.Plot.model()
    if not counts: #plot count rate
        xspec.Plot('data')
        ytitle='Counts/s'
        yrange=[-2,4.5]
    else:
        xspec.Plot('counts')
        ytitle='Counts'
        yrange=[1,1e6]
    xx = xspec.Plot.x()
    yy = xspec.Plot.y()
    if not erange:
        erange=[xx[0],xx[-1]]
    xl=np.where(np.array(xx) > erange[0])[0][0]
    try:
        xg=np.where(np.array(xx) >= erange[1])[0][0]
    except IndexError:
        xg=len(xx)-1
    yrange=[np.floor(np.log10(yy[xl:xg])).min(),np.ceil(np.log10(yy[xl:xg])).max()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xx,y=yy,mode='markers',name='data',error_y=dict(type='data',array=xspec.Plot.yErr())))
    fig.update_yaxes(title=ytitle,range=yrange,type='log',showexponent = 'all',exponentformat = 'e') #type='log'
    fig.update_xaxes(title='Energy (keV)',range = erange)
    fig.update_layout(title = title)
    return fig

def plot_fit(xspec, model, fitrange=False, dataGroup=1,erange=False,yrange = [-3, 4], res_range=[-50,50],title=False,annotation=False, plotdata_dict = False, width = 500, height = 700):
    '''plot data, fit, residuals in PlotLy. Input: xspec global object
    Plot from dictionary of plot parameters if xspec = None, model = plotadata '''
    
    if xspec is not None:
        xspec.Plot.xAxis = "keV"
        cnames = model.componentNames
        full_model_name = '+'.join(cnames)
        ncomponents = len(cnames)
        xspec.Plot.add=True
        xspec.Plot('data')
        xx = xspec.Plot.x()
        yy = xspec.Plot.y()
        yErr = xspec.Plot.yErr()
        if ncomponents == 1:
            model_comps = [xspec.Plot.model()]
        else:
            model_comps = []
            for comp in range(ncomponents):
                model_comps.append(xspec.Plot.addComp(comp+1))
            model_comps.append(xspec.Plot.model()) #total
            cnames.append(full_model_name)
        xspec.Plot('delchi')
        res = xspec.Plot.y()
    else:
        xx = model['Energy']
        yy = model['CountRate']
        yErr = model['CountErr']
        cnames = list(model['Fit'].keys())[1:] #first is fitdata
        full_model_name = cnames[-1]
        model_comps = [model['Fit'][c] for c in cnames]
        fitrange = model['Fit']['fitrange']
        res = model['Residuals']
    
    if not fitrange:
        fitrange=[xx[0],xx[-1]]
    if not erange:
        erange=[xx[0],xx[-1]]
    if not title:
        title=f"Fit with {full_model_name}"
    xl=np.where(np.array(xx) > erange[0])[0][0]
    try:
        xg=np.where(np.array(xx) >= erange[1])[0][0]
    except IndexError:
        xg=len(xx)-1

    if not yrange:
        yrange=[np.floor(np.log10(yy[xl:xg])).min(),np.ceil(np.log10(yy[xl:xg])).max()]
    
    fig = make_subplots(rows=2, cols=1, start_cell="top-left",shared_xaxes=True,row_heights=[.6,.3],vertical_spacing=.05)
    fig.add_trace(go.Scatter(x=xx,y=yy,mode='markers',name='data',error_y=dict(type='data',array=yErr)),row=1,col=1)
    for m, model_name in zip(model_comps,cnames):
        if '+' in model_name: #match color to residuals
            fig.add_trace(go.Scatter(x=xx,y=m,name=model_name, line_color = 'black', line_shape = 'hv'),row=1,col=1)
        else:
            fig.add_trace(go.Scatter(x=xx,y=m,name=model_name, line_shape = 'hv'),row=1,col=1)

    #plot residuals
    fig.update_yaxes(type='log',row=1,col=1,showexponent = 'all',exponentformat = 'e',range=yrange, title = 'Count Rate')
    fig.add_trace(go.Scatter(x=xx,y=res,mode = 'lines+markers',marker_color='black',name='residuals',line_shape = 'hv'),row=2,col=1)
    fig.add_vrect(x0=fitrange[0],x1=fitrange[1],annotation_text='fit range',fillcolor='lightgreen',opacity=.25,line_width=0,row=1,col=1)
    fig.add_vrect(x0=fitrange[0],x1=fitrange[1],fillcolor='lightgreen',opacity=.25,line_width=0,row=2,col=1)
    if annotation:
        fig.add_annotation(x=1.25,y=.5,text=annotation,align='left',xref='x domain',yref='paper')
    fig.update_yaxes(title='Residuals',range=res_range,row=2,col=1)
    fig.update_xaxes(title='Energy (keV)',range=erange,row=2,col=1)
    fig.update_layout(width=width,height=height,title=title)
    
    if plotdata_dict: #return plot data in a dictionary
        fitdata = {'fitrange':fitrange}
        for c,m in zip(cnames,model_comps):
            fitdata[c] = m
        plotdata = {'Energy': xx, 'CountRate': yy, 'CountErr': yErr, 'Fit': fitdata, 'Residuals': xspec.Plot.y()}
        return fig, plotdata
    return fig
    
def annotate_plot(model, last_component=False, exclude_parameters = ['norm'], error = False):
    '''annotations for plot - parameters and confidence intervals if they can be calculated
    Input: xspec Model object
    Output: HTML-formatted string'''
    fittext = ""
    if not last_component:
        cnames = model.componentNames[:-1]
    for comp in cnames:
        mc = getattr(model,comp)
        for par in getattr(mc,"parameterNames"):
            if par not in exclude_parameters:
                p = getattr(mc,par)
                val = p.values[0]
                fmt = ".2e"
                if np.abs(np.log10(val)) < 2:
                    fmt = ".2f"
                if p.error[2] == "FFFFFFFFF" and error: #error calculated sucessfully
                    errs = f"({p.error[0]:{fmt}}-{p.error[1]:{fmt}})"
                else:
                 errs = f"±{p.sigma:{fmt}}"#""
                fittext += f"{par}: {val:{fmt}} {errs} {p.unit}<br>"
    return fittext
