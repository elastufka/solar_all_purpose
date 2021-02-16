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

#global emcube
#global lgtaxis
#should store all this stuff in common block equivalents...how to deal with the map meta?

def dem_from_sav(filename):
    dem_dict=readsav(filename,python_dict=True)
    return dem_dict


def get_column(dem_dict,ninterp=100,maskzeros=True,max=False):
    '''Average over spatial dimenions and return one value per log T bin'''
    emcube=dem_dict['emcube']
    colmean=[]
    for em in emcube:
        if maskzeros:
            masked_zeros=ma.masked_less(em,0.00001)#(em == 0.0,em)
            if max:
                colmean.append(masked_zeros.max()) #should only do this for non-zero values?
            else:
                colmean.append(masked_zeros.mean())
        else:
            if max:
                colmean.append(np.max(em)) #should only do this for non-zero values?
            else:
                colmean.append(np.mean(em))

    if ninterp:
        x=np.linspace(0,20,num=ninterp)
        fEM=interp1d(range(0,21),colmean)
        colmean=fEM(x)
        colEM=colmean
    else:
        colEM=colmean
    return colEM


def plot_em_spectrogram(demlist,picklecols=False,EMmin=26,EMmax=32,ninterp=100,maskzeros=False,max=True):
    '''Plot DEM time evolution as a spectrograph'''
    if not picklecols:
        EMmean=[]
        taxis=[]
        for d in demlist:
            ddict=dem_from_sav(d)
            EMmean.append(get_column(ddict,ninterp=ninterp,maskzeros=maskzeros,max=max))
            taxis.append(dt.strptime(d[4:-9],'%Y%m%d_%H%M%S'))
            lgtaxis=ddict['lgtaxis'] #assume it's the same for all
    else:
        taxis,lgtaxis,EMmean=pickle.load(open(picklecols,'rb'))
    if ninterp: #interpolate
        newx=np.linspace(0,20,num=ninterp)
        flgT=interp1d(range(0,21),lgtaxis)
        #print(np.min(lgtaxis),np.max(lgtaxis))
        nlgtaxis=flgT(newx)
        print(np.min(nlgtaxis),np.max(nlgtaxis))
    else:
        nlgtaxis=lgtaxis

    #make axis labels. default 10 per axis
    #print(xlabels)
    #plot
    fig,ax=plt.subplots()
    cf1=ax.imshow(np.transpose(EMmean),origin='lower')#specgram(EMmean,NFFT=21,Fs=1.0,bins=20)
    #make axis labels. default 10 per axis
    ax.set_xlim(0,np.shape(EMmean)[0])
    #ax.set_ylim(40,160)
    fig.canvas.draw()
    cxlabels=ax.get_xticklabels()
    xlabels=[]
    for cx in cxlabels:
        if cx.get_text() != '':
            xlabels.append("{0}".format(taxis[int(cx.get_text())].time())[:-3])
    cylabels=ax.get_yticklabels()
    ylabels=[]
    for cy in cylabels:
        if cy.get_text() != '':
            ylabels.append(np.round(nlgtaxis[int(cy.get_text())],2))
    
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    ax.set_ylabel('log T')

    ax.set_xlabel('Time on '+dt.strftime(taxis[0],'%Y-%b-%d'))
    if max:
        cbar_label='max EM'
    else:
        cbar_label='average EM'
    fig.colorbar(cf1,fraction=.01,pad=.04,label=cbar_label) #need to put some physical units on this!
    #myFmt = DateFormatter('%H:%M')
    #ax.xaxis.set_major_formatter(myFmt)

    fig.show()
    return taxis,lgtaxis,EMmean

def plot_whole_cube(dem_dict=True,picklename=False,meta=False,lgtaxis=False,extent=False,ccmap=False):
    '''what it sounds like. update so that there's only one colobar extending the entire y axis...'''
    try:
        lgtaxis=dem_dict['lgtaxis']
    except KeyError:
        pass
    #if not lgtaxis:
    #    lgt=readsav('lgtaxis.sav')
    #    #global lgtaxis
    #    lgtaxis=lgt['lgtaxis']
    xylabels=['Log T='+str(np.round(lg,2)) for lg in lgtaxis]
    if not picklename:
        emcube=dem_dict['emcube']
        mapcube,mins,maxs=[],[],[]
        if not meta:
            try:
                meta=pickle.load(open('map_meta.p','rb'))
            except IOError:
                #get it from a fitsfile itself
                fitsf=glob.glob('AIA_094*.fits')
                metamap=sunpy.map.Map(fitsf[0])
                meta=metamap.meta
        #convert to sunpy maps
        for n in range(0,21):
            data=emcube[n,:,:]
            data=np.ma.masked_less(data,0.00001)
            mmap=sunpy.map.Map(data,meta)
            if extent:
                mmap=mmap.submap(SkyCoord(extent[0]*u.arcsec,extent[1]*u.arcsec),SkyCoord(extent[2]*u.arcsec,extent[3]*u.arcsec))
            mapcube.append(mmap)
            mins.append(mmap.min())
            maxs.append(mmap.max())
    else:
        mapcube=pickle.load(open(picklename,'rb'))
        mins,maxs=[],[]
        for mmap in mapcube:
            mins.append(mmap.min())
            maxs.append(mmap.max())
        meta=mapcube[0].meta
    if not ccmap:
        ccmap=cm.rainbow

    norm=colors.Normalize(vmin=np.min(mins),vmax=np.max(maxs))
        
    fig=plt.figure(figsize=(6,10))
    for i,im in enumerate(mapcube):
        #ii=i%3
        #jj=i%2
        ax=fig.add_subplot(7,3,i+1)
        cf=im.plot(cmap=ccmap,axes=ax,norm=norm)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.title.set_visible(False)
        #cf=ax[ii][jj].imshow(immasked,cmap=cm.rainbow,norm=norm,origin='lower',extent=extent)
        ax.annotate(xylabels[i],xy=(im.bottom_left_coord.Tx.value+5,im.bottom_left_coord.Ty.value+5),xytext=(im.bottom_left_coord.Tx.value+5,im.bottom_left_coord.Ty.value+5))#,textcoords='axes points')
        #print i,i%3
        #if i%3==2:
        #

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.suptitle('AIA DEM Analysis Results '+meta['date_obs'])
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.15, 0.05, 0.7])
    fig.colorbar(cf, cax=cbar_ax)
    fig.show()
    return mapcube

def get_max_T_EM(vec,lgtaxis):
    maxem=np.max(vec)
    maxx=vec.tolist().index(maxem)
    maxt=lgtaxis[maxx]
    return maxt,maxem

