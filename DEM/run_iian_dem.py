 #######################################
#display_aia_dem.py
# Erica Lastufka 15/03/2018  

#Description: Because OSX doesn't play well with XQuartz and IDL sucks
#######################################

#######################################
# Usage:

######################################

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import os
from scipy.ndimage.filters import generic_filter as gf
import matplotlib.cm as cm
import matplotlib.colors as colors
import glob
from sunpy.net import Fido, attrs as a
import sunpy.map
from scipy.io import readsav
import astropy.units as u
from astropy.coordinates import SkyCoord
from datetime import datetime as dt
from scipy.interpolate import interp1d
from aia_dem_batch import bin_images
from aia_utils import aia_maps_tint
#from sunpy_map_utils import int_map
#how to import python version of demreg?
import pickle
#global emcube
#global lgtaxis
#should store all this stuff in common block equivalents...how to deal with the map meta?

def gen_tresp_matrix(plot=False, respfile='/Users/wheatley/Documents/Solar/NuStar/AIA_tresp_20200912.dat'):
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
    
#def interp_tresp(trmatrix, logt_in,logt_out):
#    '''interpolate temperature response matrix onto new vector'''
#    from scipy.interpolate import InterpolatedUnivariateSpline as IUS
#    try:
#        nx,ny=trmatrix.shape
#    except ValueError:
#        nx=len(trmatrix)
#        ny=1
#        trmatrix=trmatrix.reshape((nx,1))
#    nx2=len(logt_out)
#    trinterp=np.zeros((nx2,ny))
#    #example:
#    #ius_lo=IUS(np.round(logt,4),np.log10(ns_tresp[:,0]),k=1)
#    #nstrint_lo=10**(ius_lo(aia_tresp_logt))
#    for i in range(ny):
#        ius_y=IUS(np.round(logt_in,4),np.log10(trmatrix[:,i]),k=1)
#        trinterp[:,i]=10**(ius_y(logt_out))
#    return trinterp
    
#def dem_norm_guess(temps,nx,ny,nt):
#    '''calculate our dem_norm guess'''
#    nt=len(temps)-1
#    off=0.412
#    gauss_stdev=12
#    dem_norm0=np.zeros([nx,ny,nt]) #what if nx and ny=0? ie lightcurve
#    dem_norm_temp=np.convolve(np.exp(-(np.arange(nt)+1-(nt-2)*(off+0.1))**2/gauss_stdev),np.ones(3)/3)[1:-1]
#    dem_norm0[:,:,:]=dem_norm_temp
#    return dem_norm0
#
#def generate_errors(nx,ny,nf,data):
#    '''return error matrix/vector of same shape as input data'''
#    serr_per=10.0
#    #errors in dn/px/s
#    if nx==0:
#        npix=1
#        edata=np.zeros(nf)
#    else:
#        npix=4096.**2/(nx*ny)
#        edata=np.zeros([nx,ny,nf])
#    gains=np.array([18.3,17.6,17.7,18.3,18.3,17.6])
#    dn2ph=gains*[94,131,171,193,211,335]/3397.0
#    rdnse=1.15*np.sqrt(npix)/npix
#    drknse=0.17
#    qntnse=0.288819*np.sqrt(npix)/npix
#    try:
#        for j in np.arange(nf):
#            etemp=np.sqrt(rdnse**2.+drknse**2.+qntnse**2.+(dn2ph[j]*abs(data[:,:,j]))/(npix*dn2ph[j]**2))
#            esys=serr_per*data[:,:,j]/100.
#            edata[:,:,j]=np.sqrt(etemp**2. + esys**2.)
#    except (IndexError, TypeError) as e: #data is 1D lightcurve
#        for j in np.arange(nf):
#            etemp=np.sqrt(rdnse**2.+drknse**2.+qntnse**2.+(dn2ph[j]*abs(data[j]))/(npix*dn2ph[j]**2))
#            esys=serr_per*data[j]/100.
#            edata[j]=np.sqrt(etemp**2. + esys**2.)
#    return edata
    
def rebin_array(arr, new_shape):
    ''' assuming that each dimension of the new shape is a factor of the corresponding dimension in the old one.'''
#    if (shape[0]%binY != 0) and (shape[1]%binX == 0):
#    rest = shape[0]%binY
#    arr1 = Binning2D(array[:shape[0]-rest], binY, binX)
#    arr2 = Binning2D(array[shape[0]-rest:], shape[0]%binY, binX)
#    arr = _numpy.concatenate((arr1,arr2), axis=0)
#    return arr
#    if (shape[1]%binX != 0) and (shape[0]%binY == 0):
#    rest = shape[1]%binX
#    arr1 = Binning2D(array[:,:shape[1]-rest], binY, binX)
#    arr2 = Binning2D(array[:,shape[1]-rest:], binY, shape[1]%binX)
#    arr = _numpy.concatenate((arr1,arr2), axis=1)
#    return arr
    
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

    
def bin_images2(gdat,n=2):
    ''' spatially bin images in n x n bins, crop array if it doesnt fit.g6 is list of file names in order[94,131,171,193,211,335] '''
    g6_binned=[]
    if type(gdat) == np.ndarray:
        gprep=[gdat[:,:,i] for i in range(6)]
    else:
        gprep=gdat
    for g in gprep:
        #gdat=sunpy.map.Map(g).data
        if type(n) == int:
            if n !=1:
                #nx,ny=np.shape(g)
                #new_shape=(g/nx,g/ny)
                gbin=downscale_local_mean(g, (n,n),clip=True)
                #gbin=rebin_array(g,new_shape)
            else:
                gbin=g
            g6_binned.append(gbin)
            g6_arr=np.array(g6_binned)
        elif n =='all':
            #g2=np.nan_to_num(g)
            g6_binned.append(np.nanmean(g)) #mean not sum...
            g6_arr=np.array(g6_binned)
    return g6_arr #but make it an array...
    
def do_2D_dem(dfaia,timestamp, trmatrix,tresp_logt,temps,maxiter=20, binning=False,flat=True,plot=False,mask=False,key='data',datavec=False,emloci=True):
    '''Calculate systematic errors and run demreg.
    Args: dfaia: input dataframe with minimum columns:timestamp, data, wavelength
            timestamp: time that the DEM should be calculated for
            trmatrix: temperaure response matrix of AIA, can be calculated by gen_tresp_matrix()
            tresp_logt: another output of gen_tresp_matrix()
            temps: temperatures DEM should be calculated at (vector)
    kwargs: maxiter: arg for dn_2demreg
            binning: bin the data n x n pixels? or if 'all', treat as lightcurve. Mask argument will be applied before, so if binning='all' and mask=True, only the selected pixels will be counted
            flat: output is flattened dataframe. if false, output is straight from dn_2demreg
            plot: plot the DEM?
            mask: apply mask to data before binning and calculating DEM. If true, must have mask column in dataframe.'''
    #another example -- 2D test
    aia= get_datavec(dfaia,timestamp,key=key)#get from dataframe
    nf=len(aia)
    try:
        nx,ny=np.shape(aia[0])
        #test that all dimensions are the same
        for i in aia[1:]:
            nxi,nyi=np.shape(i)
            if nxi !=nx or nyi !=ny:
                print('dimension mismatch in data!', nxi,nyi, ' do not match ',nx,ny)
                return None
            imdata=np.zeros([nx,ny,nf])
            #convert from our list to an array of data
            for j in np.arange(nf):
                imdata[:,:,j]=aia[j]
            imdata[imdata < 0]=0
    except ValueError: #it's 1d
        nx,ny=0,0
        data=np.array(aia)
    
    if binning:
        #print(np.shape(data))
        data=bin_images2(imdata,n=binning)
        data=data.T #order of the wavelengths IS still preserved with this
        try:
            nx,ny,_=np.shape(data)
        except ValueError: #binning = all
            nx,ny=0,0
        #print(np.shape(data.T))
    else:
        if "data" not in locals():
            if np.shape(imdata) != (nx,ny,nf):
                data=imdata.T
                nx,ny,_=np.shape(data)
            else:
                data=imdata
        
    if mask and 'total_mask_wavelengths' in dfaia.keys():
        immask=np.array(dfaia['total_mask_wavelengths'].iloc[0][0])
        try:
            masked_data=np.array([imdata[:,:,j]*immask for j in range(6)])
        except ValueError:
            print('dimension mismatch in data and mask!', np.shape(imdata[:,:,j]), ' do not match ',np.shape(immask))
            return None
        if type(binning) == int: #bin first...
            binned_mask=downscale_local_mean(immask, (binning,binning),clip=True)
            #filter mask - what's the threshold for calling the new binned vals 1 or 0?
            binned_mask[1:-1,1:-1][binned_mask[1:-1,1:-1] < mask]=0 #ignore the edges...
            #if binned_mask >0, keep as 1
            bool_binned_mask=~binned_mask.astype(bool) #0=False, so have to invert
            #apply mask to data
            masked_data=np.array([data[:,:,j]*bool_binned_mask.T for j in range(6)]) #data already binned
            print("masked data shape",np.shape(masked_data),"mean before mask ", np.mean(data[:,:,j]), "mean after mask", np.mean(masked_data))
            data=masked_data.T
            nx,ny=ny,nx
            #print(data.shape,np.shape(data),nx,ny,nf)
        elif binning == 'all':
            #apply mask before doing bin_images(n='all')
            #replace nans with zeros... I think masked values are non not nan...
            #masked_data[masked_data == np.nan]=0.
            data=bin_images2(masked_data,n='all')
            data=data.T
            print(data)
        else: #masked but no binning
            data=masked_data.T

    if nx !=0:
        #calculate our dem_norm guess
        dem_norm0=dem_norm_guess(temps,nx,ny,nt)
    else:
        dem_norm0=None
        
    if emloci:
        dem_norm0=None
        gloci=1
    else:
        gloci=0 #selfnorm
        
    edata=generate_errors(nx,ny,nf,data)

    dem,edem,elogt,chisq,dn_reg=dn2dem_pos(data,edata,trmatrix,tresp_logt,temps,dem_norm0=dem_norm0,gloci=gloci,max_iter=maxiter)
    if plot:
        quick_plot(dem,np.log10(temps[:-1]))
        
    if flat:
        #flatten by tvec and turn into dataframe
        dfdict={}
        dfdict['timestamp']=pd.Series(timestamp)
        dfdict['dn_reg']=pd.Series([dn_reg])
        dfdict['maxiter']=pd.Series(maxiter)
        dfdict['nx']=pd.Series(nx)
        dfdict['ny']=pd.Series(ny)
        if datavec:
            dfdict['datavec']=pd.Series([data])
        if mask:
            try:
                mflat=np.ma.masked_array(dem[:,:,0],mask=~bool_binned_mask,fill_value=np.nan).flatten() #don't need .T I think...
                dfdict['fraction_nonzero']=pd.Series(np.count_nonzero(mflat)/mflat.count())
            except IndexError: #1D, already masked
                dfdict['fraction_nonzero']=pd.Series(fraction_nonzeros(dem[0]))
            #print(mflat.count()) #number unmasked
            #print(np.count_nonzero(mflat)) #number unmasked and nonzero

        else:
            try:
                dfdict['fraction_nonzero']=pd.Series(fraction_nonzeros(dem[:,:,0])) #if masked, only count the pixels within the mask!
            except IndexError: #1D, already masked
                dfdict['fraction_nonzero']=pd.Series(fraction_nonzeros(dem[0]))
        dfdict['chisq']=pd.Series([chisq])
        dfdict['chisq_mean']=pd.Series(np.mean(chisq[chisq != 0])) #drop zeros
        dfdict['chisq_std']=pd.Series(np.std(chisq[chisq != 0]))

        dfdict['dem_mean']=pd.Series([np.nanmean(np.nanmean(dem,axis=0),axis=0)]) #drop zeros
        dfdict['dem_max']=pd.Series([np.nanmax(np.nanmax(dem,axis=0),axis=0)])

        dfdict['edem_mean']=pd.Series([np.nanmean(np.nanmean(edem,axis=0),axis=0)])
        dfdict['edem_max']=pd.Series([np.nanmax(np.nanmax(edem,axis=0),axis=0)])
        
        dfdict['elogt_mean']=pd.Series([np.nanmean(np.nanmean(elogt,axis=0),axis=0)])
        dfdict['elogt_max']=pd.Series([np.nanmax(np.nanmax(elogt,axis=0),axis=0)])


        for i,t in enumerate(np.log10(temps[:-1])):
            demkey='dem_'+str(t)
            edemkey='edem_'+str(t)
            elogtkey='elogt_'+str(t)
            if nx == 0: #there's got to be a better way of doing this...
                dfdict[demkey]=pd.Series([dem[i]])
                dfdict[edemkey]=pd.Series([edem[i]])
                dfdict[elogtkey]=pd.Series([elogt[i]])
#                dfdict[demkey+'_mean']=pd.Series(np.mean(dem[i][dem[i] !=0]))
#                dfdict[edemkey+'_mean']=pd.Series(np.mean(edem[i][edem[i] !=0]))
#                dfdict[elogtkey+'_mean']=pd.Series(np.mean(elogt[i][elogt[i] !=0]))
#                dfdict[demkey+'_max']=pd.Series(np.max(dem[i][dem[i] !=0]))
#                dfdict[edemkey+'_max']=pd.Series(np.max(edem[i][edem[i] !=0]))
#                dfdict[elogtkey+'_max']=pd.Series(np.max(elogt[i][elogt[i] !=0]))
            else:
                dfdict[demkey]=pd.Series([dem[:,:,i]])
                dfdict[edemkey]=pd.Series([edem[:,:,i]])
                dfdict[elogtkey]=pd.Series([elogt[:,:,i]])
#                dfdict[demkey+'_mean']=pd.Series(np.mean(dem[:,:,i][dem[:,:,i] !=0]))
#                dfdict[edemkey+'_mean']=pd.Series(np.mean(edem[:,:,i][edem[:,:,i] !=0]))
#                dfdict[elogtkey+'_mean']=pd.Series(np.mean(elogt[:,:,i][elogt[:,:,i] !=0]))
#                dfdict[demkey+'_max']=pd.Series(np.max(dem[:,:,i][dem[:,:,i] !=0]))
#                dfdict[edemkey+'_max']=pd.Series(np.max(edem[:,:,i][edem[:,:,i] !=0]))
#                dfdict[elogtkey+'_max']=pd.Series(np.max(elogt[:,:,i][elogt[:,:,i] !=0]))


        if binning:
            dfdict['binning']=pd.Series(binning)
        else:
            dfdict['binning']=pd.Series(None)
    
        df=pd.DataFrame(dfdict)
    
        return df
        
    else:
        return dem,edem,elogt,chisq,dn_reg
        
def mask_data_array(mask,data,flux_only=False):
    '''data is pd.Series, mask is bool array'''
    flux=[]
    for j in data.index:
        tmasked=data[j]*mask
        try:
            tmasked[tmasked == 0] = np.nan
        except ValueError: #it's nan already...
            pass
        flux.append(np.nanmean(tmasked))
    npx=np.sum(~mask)
    if flux_only:
        return flux
    else:
        return flux, npx
        
def lightcurve_df_prep(dfaia, masks=False):
    ''' apply each mask individually, etc'''
    #add same columns as dfaia3...
    if not masks:
        masks=pickle.load(open('all_masks.p','rb'))
    
    all_tmask,all_pmask,all_mmask,tmasks,pmasks,mmasks=masks
        
    dfaia['total_pmask_wavelengths']=[[~all_pmask] for a in dfaia.index]
    dfaia['total_mmask_wavelengths']=[[~all_mmask] for a in dfaia.index]
    dfaia['total_mask_wavelengths']=[[~all_tmask] for a in dfaia.index]

    dfaia['flux_plus_all_lambda']=mask_data_array(all_pmask,dfaia.data, flux_only=True)
    dfaia['flux_minus_all_lambda']=mask_data_array(all_mmask,dfaia.data, flux_only=True)
    dfaia['flux_total_all_lambda']=mask_data_array(all_tmask,dfaia.data, flux_only=True)
    dfaia['flux_outside_all_lambda']=mask_data_array(~all_tmask,dfaia.data, flux_only=True)

    newdf=[]
    for i,w in enumerate([94,131,171,193,211,335]):
        df=dfaia.where(dfaia.wavelength==w).dropna(how='all')
        umflux,bflux,dflux,mflux=[],[],[],[]
        umflux, umpx=mask_data_array(tmasks[i],df.data)
        bflux, bpx=mask_data_array(~pmasks[i],df.data)
        dflux, dpx=mask_data_array(~mmasks[i],df.data)
        mflux, tmpx=mask_data_array(~tmasks[i],df.data)
        
        df['outside_mask_flux']=umflux
        df['outside_mask_px']=[umpx for i in range(len(df.index))]
        df['flux_plus']=bflux
        df['mask_plus_px']=[bpx for i in range(len(df.index))]
        df['flux_minus']=dflux
        df['mask_minus_px']=[dpx for i in range(len(df.index))]
        df['flux_total']=mflux
        df['total_mask_px']=[tpx for i in range(len(df.index))]
        newdf.append(df)
    dfaia_out=pd.concat(newdf)
    return dfaia_out
    
def plot_dem(aia,dem,mlogt):
    fig=plt.figure(figsize=(6,10))
    xylabels=['Log T=' + str(np.round(m,1)) for m in mlogt]
    meta=aia[0].meta
    norm=colors.Normalize(vmin=np.min(dem),vmax=np.max(dem))
    for i in range(nt):
        ax=fig.add_subplot(7,3,i+1)
        demmap=sunpy.map.Map(dem[:,:,i],meta)
        cf=demmap.plot(axes=ax,cmap=cm.rainbow,norm=norm)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.title.set_visible(False)
        ax.annotate(xylabels[i],xy=(demmap.bottom_left_coord.Tx.value+5,demmap.bottom_left_coord.Ty.value+5),xytext=(demmap.bottom_left_coord.Tx.value+5,demmap.bottom_left_coord.Ty.value+5),color='w')#,textcoords='axes points')

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        fig.suptitle('AIA DEM Analysis Results '+meta['date_obs'])
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.875, 0.15, 0.05, 0.7])
        fig.colorbar(cf, cax=cbar_ax)
        fig.show()
        
def quick_plot(dem,mlogt):
    fig=plt.figure(figsize=(6,10))
    xylabels=['Log T=' + str(np.round(m,1)) for m in mlogt]
    norm=colors.Normalize(vmin=np.min(dem),vmax=np.max(dem))
    for i in range(len(mlogt)):
        ax=fig.add_subplot(5,4,i+1)
        cf=ax.imshow(dem[:,:,i],cmap=cm.rainbow,norm=norm)#demmap.plot(axes=ax,cmap=cm.rainbow,norm=norm)
        print(i,np.mean(np.nonzero(dem[:,:,i])),np.mean(dem[:,:,i]))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.title.set_visible(False)
        #ax.annotate(xylabels[i],xy=(demmap.bottom_left_coord.Tx.value+5,demmap.bottom_left_coord.Ty.value+5),xytext=(demmap.bottom_left_coord.Tx.value+5,demmap.bottom_left_coord.Ty.value+5),color='w')#,textcoords='axes points')

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.suptitle('AIA DEM Analysis Results')
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.15, 0.05, 0.7])
    fig.colorbar(cf, cax=cbar_ax)
    fig.show()
        
#def percent_zeros(dem):
#    '''quickly calculate % of zeros'''
#    return ((np.product(np.shape(dem)) - np.count_nonzero(dem))/np.product(np.shape(dem)))*100.
#
#def fraction_nonzeros(dem):
#    '''quickly calculate fraction of nonzeros'''
#    return np.count_nonzero(dem)/np.product(np.shape(dem))
#
#def count_nans(dem):
#    '''quickly calculate # of NaNs'''
#    return np.count_nonzero(np.isnan(dem))
    
def plot_dem_errorbar(mlogt,dem,elogt,edem,yaxis_range=[19,22],title='AIA only - selfnorm'):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=mlogt,y=dem,error_x=dict(type='data',array=elogt),error_y=dict(type='data',array=edem)))
    fig.update_layout(yaxis_type='log',yaxis_range=yaxis_range,title=title,yaxis_title='DEM cm^-5 K^-1',xaxis_title='log_10 T (K)')
    return fig


def DN_in_v_reg(dn_in,dn_reg,chanax=['94','131','171','193','211','335'],title='AIA only - selfnorm'):
    fig=go.Figure()
    for i in range(len(chanax)):
        fig.add_trace(go.Scatter(x=[dn_in[i]],y=[dn_reg[i]],hovertext="ratio: %s" % np.round(dn_reg[i]/dn_in[i],3),mode='markers',marker_symbol='cross',marker_size=10,name=chanax[i]))
    fig.add_trace(go.Scatter(x=np.linspace(.0001,1000,999),y=np.linspace(.0001,1000,999),name=''))
    fig.update_layout(xaxis_type='log',yaxis_type='log',xaxis_title='DN_in',yaxis_title='DN_reg',title=title)
    return fig
    
def calc_emloci_minimum(dn_in,trmatrix,temps,logt,aia_only=True,smth=5):
    ''' calculate EM loci curve minima for use as initial weights to demreg'''
    _,dtemps,mlogt=calc_temp_vars(temps)
    nchans=len(dn_in)
    if aia_only:
        nchans=6
    emloc=dn_in/trmatrix
    
    emloc_int=np.zeros([len(dtemps),nchans])
    for i in range(nchans):
        emloc_int[:,i]=10**(np.interp(mlogt,logt,np.log10(emloc[:,i])))/dtemps
      
    emloc_int_min=np.zeros(len(dtemps))
    for i in np.arange(len(dtemps)):
        emloc_int_min[i]=np.min(emloc_int[i,:])
        
    # Probably best just to smooth it a bit as well....
    emloc_int_min_smth=np.convolve(emloc_int_min[1:-1],np.ones(smth)/smth)[1:-1]
    
    return emloc_int_min_smth/np.max(emloc_int_min_smth)
        
def em_loci_min2D(data,trmatrix,smth=5):
    '''assuming no need to interpolate to different temperature axis!'''
    #The actual values of the normalisation do not matter,only their relative values.
    _,dtemps,mlogt=calc_temp_vars(temps)
    nx,ny,nchans=data.shape
    eml=np.zeros((nx,ny,len(dtemps)))
    emlsmth=np.zeros((nx,ny,len(dtemps)))
    for x in range(nx):
        for y in range(ny):
            emloc=data[x,y,:]/trmatrix
            eml[x,y,:]=np.min(emloc,axis=1)
            emlsmth[x,y,:]=np.convolve(eml[x,y,1:-1],np.ones(smth)/smth)[1:-1]
    #emloc_int_min_smth=np.convolve(emloc_int_min[1:-1],np.ones(smth)/smth)[1:-1]

    #replace nan with mean
    emlmean=np.nanmean(emlsmth)
    #replace zeros too just in case, to make the scale make sense
    emlsmth[emlsmth == 0] = emlmean
    emlsmth=np.abs(np.nan_to_num(emlsmth,nan=emlmean)) #make it positive
    return emlsmth
    
#def calc_temp_vars(temps=False,tstart=5.6,tend=6.8,num=42):
#    '''this snippet gets used a lot '''
#    if type(temps)==bool:
#        temps=np.logspace(tstart,tend,num=num)
#    dtemps=([temps[i+1]-temps[i] for i in np.arange(0,len(temps)-1)])
#    mlogt=([np.mean([(np.log10(temps[i])),np.log10((temps[i+1]))]) \
#        for i in np.arange(0,len(temps)-1)])
#    return temps, dtemps,mlogt
    
#def AIA_XRT_NuSTAR_dem(aiamap,xrtmap,full_trmatrix,trmatrix_logt,tstart=5.6,tend=6.8,numtemps=42,aia_contour=80,aia_mask_channel=211,aia_calc_degs=False,aia_exptime_correct=False,xrt_submap_radius=10,nustar_submap_radius=55,nustar_area='AIA',xrt_fac=1,nustar_fac=1,initial_weights='loci_min',use_AIA=True,use_XRT=True,use_NuSTAR=True,nustar_timerange=False,NuSTARlowE=False,selfnorm=False,verbose=True,plot_submaps=True,plot_dem=False,plot_ratio=False):
#    '''run demreg for given configuration. Basically runs the AIA+XRT+NuSTAR notebook for different configurations
#
#    Input full_trmatrix should NOT have NuSTAR response multiplied by any area factor! '''
#    chanax=['94','131','171','193','211','335','Be-Thin','NuSTAR 1.5-2.5 keV','NuSTAR 2.5-3.5 keV']#np.append(6,filters[1])
#
#    temps,dtemps,mlogt=calc_temp_vars(tstart=tstart,tend=tend,num=numtemps)
#
#    if type(xrtmap) == sunpy.map.sources.hinode.XRTMap: #inputs are NOT dn_in and aia_area
#        dn_in,aia_area,xrt_area=prep_joint_data(aiamap,xrtmap,aia_contour=aia_contour,aia_mask_channel=aia_mask_channel,aia_calc_degs=aia_calc_degs,aia_exptime_correct=aia_exptime_correct,xrt_submap_radius=xrt_submap_radius,nustar_timerange=nustar_timerange,nustar_submap_radius=nustar_submap_radius,plot_submaps=plot_submaps)
#    else:
#        dn_in=aiamap
#        aia_area=xrtmap
#
#
#    trmatrix_use,dn_use,edn_use=choose_data(dn_in,aia_area,full_trmatrix,trmatrix_logt,np.log10(temps),xrt_area=xrt_area,nustar_area=nustar_area,xrt_fac=xrt_fac,nustar_fac=nustar_fac,NuSTARlowE=NuSTARlowE)
#
#
#    #trmatrix_use,dn_use,edn_use=choose_data(dnin,aarea,trmatrix_in,aia_tresp_logt,ltemps,nustar_area='AIA',xrt_fac=1,nustar_fac=1,initial_weights='loci_min',use_AIA=True,use_XRT=True,use_NuSTAR=True)
#    dem,edem,elogt,chisq,dn_reg,aia_ratio,xray_ratio=run_joint_dem(dn_use,edn_use,trmatrix_use,temps,np.log10(temps),initial_weights=initial_weights,selfnorm=selfnorm)
#
#    if verbose:
#        print("DN input: %s" % dn_use)
#        print("chisq: %2f" % chisq)
#        print("AIA DN_reg/DN_in ratio: %s" % aia_ratio)
#        print("Xray DN_reg/DN_in ratio: %s" % xray_ratio)
#
#    if plot_dem:
#        fig=plot_dem_errorbar(mlogt,dem,elogt,edem)
#        fig.show()
#
#    if plot_ratio:
#        fig=DN_in_v_reg(dn_use,dn_reg,chanax)
#        fig.show()
#    df=pd.DataFrame({'aia_contour':aia_contour,'aia_mask_channel':aia_mask_channel,'xrt_submap_radius':xrt_submap_radius,'nustar_submap_radius':nustar_submap_radius,'aia_area':aia_area,'xrt_area':xrt_area,'nustar_area':nustar_area,'xrt_fac':xrt_fac,'nustar_fac':nustar_fac,'initial_weights':initial_weights,'selfnorm':selfnorm,'aia_ratio':aia_ratio,'xray_ratio':xray_ratio,'dem':[[dem]],'edem':[[edem]],'mlogt':[[mlogt]],'elogt':[[elogt]],'chisq':chisq,'dn_reg':[[dn_reg]],'dn_in':[[dn_use]]})
#    return df
    
