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
import numpy.ma as ma
import matplotlib.pyplot as plt

import os
from scipy.ndimage.filters import generic_filter as gf
import matplotlib.cm as cm
import matplotlib.colors as colors
import glob
import sunpy.map
from scipy.io import readsav
import astropy.units as u
from astropy.coordinates import SkyCoord
from datetime import datetime as dt
from scipy.interpolate import interp1d
from aia_utils import aia_maps_tint, single_time_indices, timestamp_from_filename
from dn2dem_pos import dn2dem_pos
from sunpy_map_utils import find_centroid_from_map, arcsec_to_cm, scale_skycoord
#how to import python version of demreg? need to add that path to my init.py or python_startup script
import pickle
import plotly.graph_objects as go
from visualization_tools import all_six_AIA

def group6(prepped_files):
    timeinfo=[timestamp_from_filename(f) for f in prepped_files]
    try:
        waveinfo=[int(f[f.find('Z')+2:f.rfind('image')-1]) for f in prepped_files] #for jsoc
    #full_names=[self.path+'/'+f for f in self.prepped_files]
    except ValueError:
        waveinfo=[int(f[-8:-5]) for f in prepped_files]
    ndf=pd.DataFrame({'file':prepped_files,'timestamp':timeinfo,'wavelength':waveinfo})
    ndf.drop(ndf.where(ndf.wavelength==304).dropna().index,inplace=True)
    ndf['date']=[t.date() for t in ndf.timestamp]
    ndf['hour']=[t.hour for t in ndf.timestamp]
    ndf['minute']=[t.round('min').minute for t in ndf.timestamp]
    gdf=ndf.groupby(['date','hour','minute'])
    for name,group in gdf:
        yield name,group
        
def check_datanumbers(prepped_files):
    means,maxima,exptimes,tstamps=[],[],[],[]
    for name,group in group6(prepped_files):
        ff=group.sort_values(by='wavelength')[['file','timestamp']]
        mapf=[sunpy.map.Map(f) for f in ff['file']]
        exptimes.append([m.meta['exptime'] for m in mapf])
        means.append([np.mean(m.data) for m in mapf])
        maxima.append([np.max(m.data) for m in mapf]) #aia_prep_py doesn't update fits keywords for datamean, datamax, etc
        tstamps.append(ff['timestamp'].iloc[0])

    fig=go.Figure()
    for j,c in enumerate([94,131,171,193,211,335]):
        fig.add_trace(go.Scatter(x=tstamps,y=np.array(exptimes)[:,j],name=f"{c} exptime"))
        fig.add_trace(go.Scatter(x=tstamps,y=np.array(means)[:,j],name=f"{c} mean"))
        fig.add_trace(go.Scatter(x=tstamps,y=np.array(maxima)[:,j],name=f"{c} max"))
        fig.add_trace(go.Scatter(x=tstamps,y=np.array(means)[:,j]/np.array(exptimes)[:,j],name=f"{c} mean/exptime"))
        fig.add_trace(go.Scatter(x=tstamps,y=np.array(maxima)[:,j]/np.array(exptimes)[:,j],name=f"{c} max/exptime"))
    return fig


def interp_tresp(trmatrix, logt_in,logt_out):
    '''interpolate temperature response matrix onto new vector'''
    from scipy.interpolate import InterpolatedUnivariateSpline as IUS
    try:
        nx,ny=trmatrix.shape #=np.max(trmatrix.shape)
        #ny=np.min(trmatrix.shape)
    except ValueError:
        nx=len(trmatrix)
        ny=1
        trmatrix=trmatrix.reshape((nx,1))
    nx2=len(logt_out)
    trinterp=np.zeros((nx2,ny))
    #example:
    #ius_lo=IUS(np.round(logt,4),np.log10(ns_tresp[:,0]),k=1)
    #nstrint_lo=10**(ius_lo(aia_tresp_logt))
    for i in range(ny):
        ius_y=IUS(np.round(logt_in,4),np.log10(trmatrix[:,i]),k=1)
        trinterp[:,i]=10**(ius_y(logt_out))
    return trinterp
    
def read_tresp_matrix(plot=False, respfile='/Users/wheatley/Documents/Solar/NuStar/AIA_tresp_20200912.dat'):
    '''from Iian's tutorial

    IDL nonsense required to make valid .dat file for a given date:

    IDL> tresp=aia_get_response(/temp,/dn,/evenorm,timedepend_date='2020-09-12T20:00:00')
    IDL> date=tresp.date
    IDL> effarea_version=tresp.effarea_version
    IDL> channels=tresp.channels
    IDL> remove, 5, channels #get rid of 304
    IDL> units=tresp.units
    IDL> logte=tresp.logt
    IDL> tr=[tresp.a94,tresp.a131,tresp.a171,tresp.a193,tresp.a211,tresp.a335]
    IDL> save, date,effarea_version,channels,units,logt,tr, filename='AIA_tresp_20200912.dat'

    '''
    # Load in the SSWIDL generated response functions
    # Was produced by make_aiaresp_forpy.pro (can't escape sswidl that easily....)
    if not respfile:
        trin=readsav('/Users/wheatley/Documents/Solar/NuStar/demreg/python/aia_tresp_en.dat')
    else:
        trin=readsav(respfile,python_dict=True)

    # Get rid of the b in the string name (byte vs utf stuff....)
    for i in np.arange(len(trin['channels'])):
        trin['channels'][i]=trin['channels'][i].decode("utf-8")
    #print(trin['channels'])

    # Get the temperature response functions in the correct form for demreg
    tresp_logt=np.array(trin['logt'])
    nt=len(tresp_logt)
    nf=len(trin['tr'][:])
    trmatrix=np.zeros((nt,nf))
    for i in range(0,nf):
        try:
            trmatrix[:,i]=trin['tr'][i]
        except ValueError:
            trmatrix[:,i]=trin['tr'][i][3]
    if plot:
        # Setup some AIA colours
        clrs=['darkgreen','darkcyan','gold','sienna','indianred','darkslateblue']

        # Do the plot
        fig = plt.figure(figsize=(8, 7))
        for i in np.arange(6):
            plt.semilogy(tresp_logt,trmatrix[:,i],label=trin['channels'][i],color=clrs[i],lw=4)
        plt.xlabel('$\mathrm{\log_{10}T\;[K]}$')
        plt.ylabel('$\mathrm{AIA\;Response\;[DN\;s^{-1}\;px^{-1}\;cm^5]}$')
        plt.ylim([2e-29,5e-24])
        plt.xlim([5.2,7.6])
        plt.legend(ncol=2,prop={'size': 16})
        plt.rcParams.update({'font.size': 16})
        plt.grid(True,which='both',lw=0.5,color='gainsboro')
        plt.show()

    return nt,nf,trmatrix,tresp_logt

def calc_temp_vars(temps=False,tstart=5.6,tend=6.8,num=42):
    '''this snippet gets used a lot '''
    if type(temps)==bool:
        temps=np.logspace(tstart,tend,num=num)
    dtemps=([temps[i+1]-temps[i] for i in np.arange(0,len(temps)-1)])
    mlogt=([np.mean([(np.log10(temps[i])),np.log10((temps[i+1]))]) \
        for i in np.arange(0,len(temps)-1)])
    return temps, dtemps,mlogt
    
def dem_norm_guess(temps,nx,ny,nt):
    '''calculate our dem_norm guess'''
    nt=len(temps)-1
    off=0.412
    gauss_stdev=12
    dem_norm0=np.zeros([nx,ny,nt]) #what if nx and ny=0? ie lightcurve
    dem_norm_temp=np.convolve(np.exp(-(np.arange(nt)+1-(nt-2)*(off+0.1))**2/gauss_stdev),np.ones(3)/3)[1:-1]
    dem_norm0[:,:,:]=dem_norm_temp
    return dem_norm0

def generate_errors(nx,ny,nf,data):
    '''return error matrix/vector of same shape as input data'''
    serr_per=10.0
    #errors in dn/px/s
    if nx==0:
        npix=1
        edata=np.zeros(nf)
    else:
        npix=4096.**2/(nx*ny)
        edata=np.zeros([nx,ny,nf])
    gains=np.array([18.3,17.6,17.7,18.3,18.3,17.6])
    dn2ph=gains*[94,131,171,193,211,335]/3397.0
    rdnse=1.15*np.sqrt(npix)/npix
    drknse=0.17
    qntnse=0.288819*np.sqrt(npix)/npix
    try:
        for j in np.arange(nf):
            etemp=np.sqrt(rdnse**2.+drknse**2.+qntnse**2.+(dn2ph[j]*abs(data[:,:,j]))/(npix*dn2ph[j]**2))
            esys=serr_per*data[:,:,j]/100.
            edata[:,:,j]=np.sqrt(etemp**2. + esys**2.)
    except (IndexError, TypeError) as e: #data is 1D lightcurve
        for j in np.arange(nf):
            etemp=np.sqrt(rdnse**2.+drknse**2.+qntnse**2.+(dn2ph[j]*abs(data[j]))/(npix*dn2ph[j]**2))
            esys=serr_per*data[j]/100.
            edata[j]=np.sqrt(etemp**2. + esys**2.)
    return edata

def percent_zeros(dem):
    '''quickly calculate % of zeros'''
    return ((np.product(np.shape(dem)) - np.count_nonzero(dem))/np.product(np.shape(dem)))*100.

def fraction_nonzeros(dem):
    '''quickly calculate fraction of nonzeros'''
    return np.count_nonzero(dem)/np.product(np.shape(dem))

def count_nans(dem):
    '''quickly calculate # of NaNs'''
    return np.count_nonzero(np.isnan(dem))
    
def tresp_matrix_from_IDL(datestr):
    ''' ;  tresp=aia_get_response(/temperature,/dn,/eve,timedepend_date='01-Jul-2010')
     ids=[0,1,2,3,4,6]
     channels=tresp.channels[ids]
     logt=tresp.logte
     tr=tresp.all[*,ids]
     units=tresp.units '''
    import pidly
    #datestr=self.get_timedepend_date()
    ids=[0,1,2,3,4,6]
    idl = pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
    idl('ts',datestr)
    idl('ids',ids)
    idl('tresp=aia_get_response(/temperature,/dn,/eve,timedepend_date=ts)')
    idl('channels=tresp.channels[ids]') #do I need this?
    idl('logt=tresp.logte')
    idl('tr=tresp.all[*,ids]')
    idl('units=tresp.units')
    full_trmatrix=idl.tr
    trmatrix_logt=idl.logt
    tresp_units=idl.units
    return full_trmatrix,trmatrix_logt,tresp_units

