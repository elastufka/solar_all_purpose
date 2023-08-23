import numpy as np

from astropy.io import fits
from astropy.time import Time
from datetime import datetime as dt
from datetime import timedelta as td
import plotly.graph_objects as go

def plot_stix_spec(filename, log=False, tickinterval = 100, time_int = None, idx_int = None, mode = 'Heatmap', binning = 'SDC', gridfac = 0.265506, error=True, zmin = None, zmax = None):
    """Plot STIX spectrum converted to XSPEC-compatible format FITS file """
    if isinstance(filename, str):
        spec = fits.open(filename)
        try:
             rate=spec[1].data['RATE']
             rate_err = spec[1].data['STAT_ERR']
             spectime=spec[1].data['TIME']
             emin=list(spec[2].data['E_MIN'])
             emax=list(spec[2].data['E_MAX'])
             header = spec[1].header
             cbar_title = "Background Subtracted<br> Counts s<sup>-1</sup> keV<sup>-1</sup> cm<sup>-2</sup>" #"Counts s<sup>-1</sup>"
        except KeyError: #it's a raw spectrogram
             rate=spec[2].data['counts']
             rate_err = spec[2].data['counts_err']
             if rate.ndim > 2:
                rate = np.sum(np.sum(rate, axis=2),axis=2)
                rate_err = np.sum(np.sum(rate_err, axis=2),axis=2)
             time_bin_center=spec[2].data['time']
             duration = spec[2].data['timedel']
             header = spec[0].header
             start_time = dt.strptime(header['DATE_BEG'],"%Y-%m-%dT%H:%M:%S.%f")
             #print('start_time',start_time)
             factor=1.
             spectime = Time([start_time + td(seconds = bc/factor - d/(2.*factor)) for bc,d in zip(time_bin_center, duration)]).mjd

             emin=list(spec[3].data['e_low'])
             emax=list(spec[3].data['e_high'])
             # timezeri = int(Time(start_time).mjd) - spec[0].header['MJDREF']

             # header.set('TIMEZERO',timezeri)
             # print('TIMEZERO',timezeri)
             cbar_title = 'Counts'
        #rate=spec[1].data['RATE']
        #rate_err = spec[1].data['STAT_ERR']
        #spectime=spec[1].data['TIME']
        #emin=list(spec[2].data['E_MIN'])
        #emax=list(spec[2].data['E_MAX'])
        #header = spec[1].header
        spec.close()
        tformat = 'mjd'
    else: #assume it's a stixpy.processing.spectrogram.spectrogram.Spectrogram
        spec = filename
        rate = spec.rate
        rate_err = spec.stat_err
        if spec.alpha and 'correction' not in spec.history:
            rate = np.sum(rate,axis=1) #sum over detector
        spectime = spec.t_axis.time_mean
        emin = spec.e_axis.low.tolist()
        emax = spec.e_axis.high.tolist()
        header = spec.primary_header
        tformat = None

    tt=Time(spectime, format = tformat)
    if tt.datetime[0].year < 2020 or tt.datetime[0].year > dt.now().year: #find a better way of doing this
        #compare time axis
        tt = Time([Time(header['TIMEZERO']+header['MJDREF'], format='mjd').datetime + td(seconds = t) for t in spectime])
    ylabels=[f"{n:.0f}-{x:.0f}" for n,x in zip(emin,emax)]
    plot_rate = rate.T
    cbar_title = "Background Subtracted<br> Counts s<sup>-1</sup> keV<sup>-1</sup> cm<sup>-2</sup>" #pretty much true, since counts was divided by eff_ewidth during correction
    plot_time = tt

    if log:
        plot_rate = np.log10(plot_rate)
        plot_rate[np.isnan(plot_rate)] = np.nanmin(plot_rate)
        
    #print(plot_rate.shape)
    if time_int: #format HH:MM
        idx_start = tt[0]
        idx_end = tt[-1]
        plot_rate = plot_rate[:,idx_start:idx_end]
        plot_time = plot_time[idx_start:idx_end]
        
    if idx_int:
        idx_start, idx_end = idx_int
        plot_rate = plot_rate[:,idx_start:idx_end]
        plot_time = plot_time[idx_start:idx_end]
        
    fig = go.Figure()
    fig.update_layout(xaxis2=dict(title='Index',tickmode='array',anchor='y',tickvals=np.arange(plot_rate.size/tickinterval)*tickinterval,ticktext=np.arange(1,(plot_rate.size+1)/tickinterval)*tickinterval,tickangle=360,overlaying='x',side='top'))
    if mode.lower() == 'heatmap':
        fig.add_trace(go.Heatmap(x=np.arange(plot_rate.size),z=plot_rate,colorbar_title=cbar_title,xaxis='x2', zauto= False, zmin = zmin, zmax = zmax, opacity = 0))
        fig.add_trace(go.Heatmap(x=plot_time.isot,z=plot_rate,colorbar_title=cbar_title,xaxis='x1', zauto= False, zmin = zmin, zmax = zmax))
        fig.update_yaxes(dict(title='Energy Bin (keV)',tickmode='array',ticktext=ylabels,tickvals=np.arange(len(ylabels))))
        #if zmin:
        #    fig.update_layout(coloraxis_cmin = zmin)
        #if zmax:
        #    fig.update_layout(coloraxis_cmax = zmax)
    elif mode.lower() == 'scatter':
        
        emin.append(emax[-1])
        if binning == 'SDC':
            bins = [(4,10),(10,15),(15,25),(25,50)] #keV
            bin_idx = [[emin.index(l),emin.index(h)] for l,h in bins]
        elif isinstance(binning, list): #bins are a list of tuples
            bins = binning
            bin_idx = [[emin.index(np.float32(l)),emin.index(np.float32(h))] for l,h in bins]
        else: # no binning
            bins = [[l,h] for l,h in zip(emin,emax)]
            bin_idx = [[emin.index(l),emin.index(h)] for l,h in zip(emin,emax)]
        
        #fig.add_trace(go.Scatter(x=np.arange(plot_rate.size),y=np.sum(plot_rate[bin_idx[0][0]:bin_idx[0][1]],axis=0)*gridfac,xaxis='x2',mode='lines',line_shape='hv')) #uneven time bins mess this up...
        for bi,b in zip(bin_idx,bins):
            error_y = None
            if error:
                error_y=dict(type='data',array=np.sum(rate_err[bi[0]:bi[1]],axis=0)*gridfac)
            fig.add_trace(go.Scatter(x=plot_time.isot,y=np.sum(plot_rate[bi[0]:bi[1]],axis=0)*gridfac,error_y=error_y,xaxis='x1',mode='lines',line_shape='hv',name=f"{b[0]:.0f}-{b[1]:.0f} keV")) #plot errors
            fig.update_yaxes(dict(title='Count Rate'))

    fig.update_layout(title=f"Spectrogram {plot_time[0].datetime:%Y-%m-%d %H:%M:%S}")
    return fig
    
def plot_stix_livetime(filename, log=False, tickinterval = 100, time_int = None, idx_int = None):
    """Plot STIX spectrum converted to XSPEC-compatible format FITS file """
    if isinstance(filename,str):
        spec = fits.open(filename)
        ltime=spec[1].data['LIVETIME']
        spectime=spec[1].data['TIME']
        emin=spec[2].data['E_MIN']
        emax=spec[2].data['E_MAX']
        spec.close()
        tformat = 'mjd'
    else: #assume it's a stixpy.processing.spectrogram.spectrogram.Spectrogram
        spec = filename
        try:
            ltime = spec.eff_livetime_fraction
        except AttributeError:
            ltime = np.mean(np.mean(spec.livetime_fraction,axis=0),axis=0)
        spectime = spec.t_axis.time_mean
        emin = spec.e_axis.low
        emax = spec.e_axis.high
        tformat = None
    
    tt=Time(spectime, format = tformat)
    ylabels=[f"{n:.0f}-{x:.0f}" for n,x in zip(emin,emax)]
    plot_rate = ltime.T
    plot_time = tt
    
    if log:
        plot_rate = np.log10(plot_rate)
        plot_rate[np.isnan(plot_rate)] = np.nanmin(plot_rate)
        
    #print(plot_rate.shape)
    if time_int: #format HH:MM
        idx_start = tt[0]
        idx_end = tt[-1]
        plot_rate = plot_rate[idx_start:idx_end]
        plot_time = plot_time[idx_start:idx_end]
        
    if idx_int:
        idx_start, idx_end = idx_int
        plot_rate = plot_rate[idx_start:idx_end]
        plot_time = plot_time[idx_start:idx_end]
        
    fig = go.Figure()
    #fig.add_trace(go.Heatmap(x=np.arange(rate.size),z=rate.T,xaxis='x2',showlegend=False,showscale=False))
    fig.add_trace(go.Scatter(x=plot_time.isot,y=plot_rate,xaxis='x1'))
    fig.update_yaxes(dict(title='Livetime Fraction'))
    fig.update_layout(xaxis2=dict(title='Index',tickmode='array',anchor='y',tickvals=np.arange(plot_rate.size/tickinterval),ticktext=np.arange(1,(plot_rate.size+1)/tickinterval),tickangle=360,overlaying='x',side='top'))
    fig.update_layout(title=f"Livetime fraction {plot_time[0].datetime:%Y-%m-%d %H:%M:%S}")
    return fig
