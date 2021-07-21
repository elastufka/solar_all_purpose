0 #######################################
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
from aia_utils import aia_maps_tint, single_time_indices
from dn2dem_pos import dn2dem_pos
from sunpy_map_utils import find_centroid_from_map, arcsec_to_cm, scale_skycoord
#how to import python version of demreg? need to add that path to my init.py or python_startup script
import pickle
import plotly.graph_objects as go
from visualization_tools import all_six_AIA

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
