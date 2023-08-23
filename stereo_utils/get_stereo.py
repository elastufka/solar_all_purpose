 #######################################
# get_stereo.py
# Erica Lastufka 8/5/2017  

#Description: Get STEREO images, etc
#######################################

#######################################
# Usage:

######################################

import numpy as np
import scipy.constants as sc
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
import os
#import data_management as da
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy

import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.wcs

from sunpy.visualization import toggle_pylab, wcsaxes_compat
import sunpy.wcs as wcs
import sunpy.map
import sunpy.coordinates
import sunpy.coordinates.wcs_utils
from sunpy.net import vso

def query(flare,vc,i, AIAwave, all=False,timedelta=5):
    '''Query the VSO client for a particular flare - assume client has already been initialized in main loop'''
    #print 'STEREO_'+flare.Datetimes['RHESSI_datetimes'][i],'\n',dt.strftime(flare.Datetimes['Messenger_datetimes'][i].date(),'%Y-%m-%d'),'\n',dt.strftime(flare.Datetimes['Messenger_datetimes'][i],'%Y-%m-%dT%H:%M:%S'),'\n',dt.strftime(flare.Datetimes['Messenger_datetimes'][i].date()+td(days=1),'%Y-%m-%d')
    res=[]
    inst=flare.Notes[i]
    #print inst
    stereo = (vso.attrs.Source('STEREO_'+inst) &
          vso.attrs.Instrument('EUVI') &
          vso.attrs.Time(dt.strftime(flare.Datetimes['Messenger_datetimes'][i],'%Y-%m-%dT%H:%M:%S'), dt.strftime(flare.Datetimes['Messenger_datetimes'][i]+td(minutes=timedelta),'%Y-%m-%dT%H:%M:%S'))) #using field RHESSI datetimes to store closest stereo satellite
    aia = (vso.attrs.Instrument('AIA') &
    vso.attrs.Sample(24 * u.hour) &
    vso.attrs.Time(dt.strftime(flare.Datetimes['Messenger_datetimes'][i],'%Y-%m-%dT%H:%M:%S'),dt.strftime(flare.Datetimes['Obs_end_time'][i],'%Y-%m-%dT%H:%M:%S')))

    if all:
        aia304 = (vso.attrs.Instrument('AIA') &
        vso.attrs.Sample(24 * u.hour) &
        vso.attrs.Time(dt.strftime(flare.Datetimes['Messenger_datetimes'][i],'%Y-%m-%dT%H:%M:%S'),dt.strftime(flare.Datetimes['Obs_end_time'][i],'%Y-%m-%dT%H:%M:%S')))
        aia171 = (vso.attrs.Instrument('AIA') &
        vso.attrs.Sample(24 * u.hour) &
        vso.attrs.Time(dt.strftime(flare.Datetimes['Messenger_datetimes'][i],'%Y-%m-%dT%H:%M:%S'),dt.strftime(flare.Datetimes['Obs_end_time'][i],'%Y-%m-%dT%H:%M:%S')))
        aia193 = (vso.attrs.Instrument('AIA') &
        vso.attrs.Sample(24 * u.hour) &
        vso.attrs.Time(dt.strftime(flare.Datetimes['Messenger_datetimes'][i],'%Y-%m-%dT%H:%M:%S'),dt.strftime(flare.Datetimes['Obs_end_time'][i],'%Y-%m-%dT%H:%M:%S')))
        wave = vso.attrs.Wave(16.9 * u.nm, 17.2 * u.nm) 
        res.append(vc.query(wave, aia171))
        wave = vso.attrs.Wave(19.1 * u.nm, 19.45 * u.nm)
        res.append(vc.query(wave, aia193))
        wave = vso.attrs.Wave(30 * u.nm, 31 * u.nm) 
        res.append(vc.query(wave, aia304))
        wave = vso.attrs.Wave(19.1 * u.nm, 19.45 * u.nm) 
        res.append(vc.query(wave, stereo))
        print(res)
        
    if AIAwave=='171':
        wave = vso.attrs.Wave(16.9 * u.nm, 17.2 * u.nm) 
        res.append(vc.query(wave, aia))
    elif AIAwave=='193':
        wave = vso.attrs.Wave(19.1 * u.nm, 19.45 * u.nm) 
        res.append(vc.query(wave, aia))
    else: #default to 304
        wave = vso.attrs.Wave(30 * u.nm, 31 * u.nm) 
        res.append(vc.query(wave, aia))
        
    wave = vso.attrs.Wave(19.1 * u.nm, 19.45 * u.nm) 
    res.append(vc.query(wave, stereo))
    return res

def download_files(vc,res,path=False):
    if not path:
        files = vc.get(res,path='/Users/wheatley/Documents/Solar/occulted_flares/data/stereo_pfloops/{file}.fts').wait()
    else:
        files = vc.get(res,path=path+'{file}.fts').wait()        
    print files
    return files
    
def make_maps(files): #should work once AIA is back up...
    #aa=[files[0],files[1][0]]
    aiafiles=[]
    stereofiles=[]
    for f in files:
        if f !=[]:
            if 'aia' in f:
                aiafiles.append(f)
            else:
                stereofiles.append(f)
        #print aiafiles,stereofiles
    try:
        aiafiles.append(stereofiles[-1])
    except IndexError:
        aiafiles.append(stereofiles)
    print aiafiles,stereofiles
    #aiafiles.append(stereofiles[0])
    maps = {m.instrument: m.submap((-1100, 1100) * u.arcsec,
                             (-1100, 1100) * u.arcsec) for m in sunpy.map.Map(aiafiles)}
    return maps

def coords_aia(flare_list,i):
    aia_width = 200 * u.arcsec
    aia_height = 250 * u.arcsec
    locstr=flare_list.Flare_properties['Location'][i]
    locx=int(locstr[locstr.find('[')+1:locstr.find(';')])-100
    locy=int(locstr[locstr.find(';')+1:locstr.rfind(']')])-125
    aia_bottom_left = (locx, locy) * u.arcsec
   
    return aia_bottom_left,aia_width,aia_height

def plot_aia(maps,aia_bottom_left,aia_width,aia_height,flare_list,i,quiet=True,save=False):                             
    #fig = plt.figure(figsize=(6, 5))
    fname='/Users/wheatley/Documents/Solar/occulted_flares/data/stereo-aia/'+dt.strftime(flare_list.Datetimes['Messenger_datetimes'][i].date(),'%Y%m%d')+'AIA_'
    #for i, m in enumerate(maps.values()):
    #    ax = fig.add_subplot(1, 2, i+1, projection=m.wcs)
    #    m.plot(axes=ax)
    
    tags=['AIA 2','AIA 3','AIA 4']
    for m,wave in zip(tags,['193','171','304']):
        try:
            maps[m]
            lastmap=maps[m]
        except KeyError:
            continue
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=maps[m].wcs)
        maps[m].plot(axes=ax)
        maps[m].draw_rectangle(aia_bottom_left, aia_width, aia_height)
        if not quiet: fig.show()
        if save: fig.savefig(fname+'full'+wave+'.png')
        print fname+'full'+wave+'.png'
        fig.clf()
        #subaia0 = maps['AIA']
        #subaia0.peek()

        subaia = maps[m]#.submap(u.Quantity((aia_bottom_left[0],
                         #               aia_bottom_left[0] + aia_width)),
                         #   u.Quantity((aia_bottom_left[1],
                         #               aia_bottom_left[1] + aia_height)))
        if not quiet: subaia.peek()
        #subaia.save(fname+'zoom.png',filetype='png')
        return lastmap

def point_check(maps,flare_list,i,corner=False):
    locstr=flare_list.Flare_properties['Location'][i]
    locx=int(locstr[locstr.find('[')+1:locstr.find(';')])
    locy=int(locstr[locstr.find(';')+1:locstr.rfind(']')])
    hpc_aia = SkyCoord(locx*u.arcsec,locy*u.arcsec, frame=maps['AIA'].coordinate_frame)
    hgs = hpc_aia.transform_to('heliographic_stonyhurst')
    hgs.D0 = maps['SECCHI'].dsun
    hgs.L0 = maps['SECCHI'].heliographic_longitude
    hgs.B0 = maps['SECCHI'].heliographic_latitude
    hpc_B = hgs.transform_to('helioprojective')

    #now plot the map with the point overlaid as a large-ish circle with this point at the center
    fig1 = plt.figure(figsize=(5, 5))
    ma = maps['AIA']
    ax1 = fig1.add_subplot(1, 1, 1, projection=ma.wcs)
    ma.plot(axes=ax1) #do I need the argument here?
    from matplotlib import patches,colors
    #c0=plt.Circle((locx,locy), 5,color='b')
    ma.draw_rectangle((locx,locy)*u.arcsec,100*u.arcsec,100*u.arcsec)
    fig1.show()    
   
    ms=maps['EUVI']
    fig2 = plt.figure(figsize=(5, 5))
    ax2 = fig2.add_subplot(1, 1, 1, projection=ms.wcs)
    coord=hpc_B
    ms.plot(axes=ax2)
    if not corner:
        ms.draw_rectangle((hpc_B.Tx.value,hpc_B.Ty.value)*u.arcsec, 100*u.arcsec,100*u.arcsec)
    else:
        ms.draw_rectangle(corner*u.arcsec,100*u.arcsec,100*u.arcsec)
    #ax2.add_artist(c2)
    fig2.show()    
    return hpc_B

def pick_corner(hgs):
    '''Pick a non-NaN coordinate to use as basis for selection in STEREO'''
    recalc=[]
    for x,y in zip(hgs.Tx.value,hgs.Ty.value):
        if np.isnan(x) or np.isnan(y):
            continue
        else: 
            recalc.append([x,y])
    #determine which one is lower
    if recalc[0][1] - recalc[0][0] > 0.:
        corner=recalc[:][1]
    else:
        corner=recalc[:][0]
    #and whether the rectangle should extend to the left (-) or right (+)...
    sign = 1 if recalc[1][0] - recalc[0][0] < 0 else -1
    #now get coords for bottom left corner if selected corner is bottom right (coords(0)=nan):
    if np.isnan(hgs[0].Tx.value):
        corner[0]=corner[0]-100 #go with 100 arcsec for now
    print corner
    return corner
    
def coord_transform(aia_bottom_left,aia_width,aia_height,maps):

    hpc_aia = SkyCoord((aia_bottom_left,
                    aia_bottom_left + u.Quantity((aia_width, 0 * u.arcsec)),
                    aia_bottom_left + u.Quantity((0 * u.arcsec, aia_height)),
                    aia_bottom_left + u.Quantity((aia_width, aia_height))),
                   frame=maps['AIA 4'].coordinate_frame)

    print(hpc_aia)

    hgs = hpc_aia.transform_to('heliographic_stonyhurst')
    #hgs = hpc_aia.transform_to('heliocentric') #does this give nans?
    print(hgs)

    hgs.D0 = maps['SECCHI'].dsun
    hgs.L0 = maps['SECCHI'].heliographic_longitude
    hgs.B0 = maps['SECCHI'].heliographic_latitude

    hpc_B = hgs.transform_to('helioprojective')
    print(hpc_B)
    return hpc_B

def plot_stereo(hpc_B,maps,aia_width,aia_height,flare_list,i,corner=False,quiet=True,save=False):
    '''deprecated to oplot_limb'''
    fig = plt.figure(figsize=(6, 5))
    #for i, (m, coord) in enumerate(zip([maps['EUVI'], maps['AIA']],
    #                               [hpc_B, hpc_aia])):
    m=maps['SECCHI']
    ax = fig.add_subplot(1, 1, 1, projection=m.wcs)
    #    m.plot(axes=ax)
    coord=hpc_B
    #coord[3] is the top-right corner coord[0] is the bottom-left corner.
    w =200*u.arcsec
    h = 250*u.arcsec
    print coord[0].Tx,coord[0].Ty,w,h #this is bottom right. 2 is top left
    #w = (coord[2].Tx - coord[0].Tx)
    #h = (coord[2].Ty - coord[0].Ty)
    m.plot(axes=ax)
    if not corner:
        m.draw_rectangle((hpc_B.Tx.value,hpc_B.Ty.value)*u.arcsec, w,h,transform=ax.get_transform('world'))
    else:
        m.draw_rectangle(corner*u.arcsec,w,h,transform=ax.get_transform('world'))
    #m.draw_limb()
    #m.draw_limb(axes=ax)
    if not quiet: fig.show()
    fname='/Users/wheatley/Documents/Solar/occulted_flares/data/stereo-aia/'+dt.strftime(flare_list.Datetimes['Messenger_datetimes'][i].date(),'%Y%m%d')+'EUVI_'
    if save:
        fig.savefig(fname+'full.png')
    #print u.Quantity((hpc_B[0].Tx, hpc_B[3].Tx)),u.Quantity((hpc_B[0].Ty, hpc_B[3].Ty))
    subeuvi0 = maps['SECCHI']
    if save:
        subeuvi0.save(fname+'full.fits')
    #subeuvi0.draw_limb(axes=ax)
    #subeuvi0.peek()
    #subeuvi = maps['EUVI'].submap(u.Quantity((hpc_B[0].Tx, hpc_B[2].Tx)),
    #                          u.Quantity((hpc_B[0].Ty, hpc_B[2].Ty)))
    subeuvi0.peek(draw_grid=True)

    #fig = plt.figure(figsize=(15, 5))
    #for i, m in enumerate((subeuvi, subaia)):
    #    ax = fig.add_subplot(1, 2, i+1, projection=m.wcs)
    #    m.plot(axes=ax)

    #@u.quantity_input(grid_spacing=u.deg)
    #[docs]
    
def oplot_AIAlimb(AIAmap,EUVmap,quiet=False,both=False):
    """From Simon Cadair's example"""
    r = AIAmap.rsun_obs.to(u.deg)-1*u.arcsec # remove the one arcsec so it's on disk.
    # Adjust the following range if you only want to plot on STEREO_A
    th = np.linspace(-180*u.deg, 0*u.deg)
    x = r * np.sin(th)
    y = r * np.cos(th)

    coords = SkyCoord(x, y, frame=AIAmap.coordinate_frame)

    hgs = coords.transform_to('heliographic_stonyhurst')
    hgs.D0 = EUVmap.dsun
    hgs.L0 = EUVmap.heliographic_longitude
    hgs.B0 = EUVmap.heliographic_latitude
    coords = hgs.transform_to(EUVmap.coordinate_frame)
    #if quiet: return coords
    if both:
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection=AIAmap)
        AIAmap.plot(axes=ax1)
        #maps['AIA'].draw_limb()

        ax2 = fig.add_subplot(1, 2, 2, projection=EUVmap)
        EUVmap.plot(axes=ax2)
        ax2.plot_coord(coords,'-', color='w')
        if not quiet:
            fig.show()
        else:
            return fig
    else:
        fig = plt.figure(figsize=(10, 8))
        ax2 = fig.add_subplot(1, 1, 1, projection=EUVmap)
        EUVmap.plot(axes=ax2)
        ax2.plot_coord(coords,'-', color='w') #why won't it be a dashed line?
        if not quiet:
            fig.show()
        return fig
            
def one_flare(flare_list,i):
    '''Do everything for one flare'''
    #import data_management2 as d
    #flare_list=d.OCData('flare_lists/list_final.sav')
    files=[]#search_local_fits(flare_list,i)
    #ff=[files[0][0],files[1][0]]
    if files == []:
        vc = vso.VSOClient()
        res=query(flare_list,vc,i,'304')
        if not res:
            pass
        for r in res:
            try:
                files.append(download_files(vc,r))
            except AttributeError:
                continue
        for f in files[0:2]:
            try:
                if not 'aia' in f[0]:
                    continue
            except IndexError:
                continue
    ff=[files[0][0],files[1][0]]
    aia_bottom_left,aia_width,aia_height=coords_aia(flare_list,i)
    maps=make_maps(ff)
    if maps == []:
        pass
    AIAmap=plot_aia(maps,aia_bottom_left,aia_width,aia_height,flare_list,i)
    try:
        foo=oplot_AIAlimb(AIAmap, maps['SECCHI'],quiet=True)
    except KeyError:
        return
    fname='/Users/wheatley/Documents/Solar/occulted_flares/data/stereo-aia/'+dt.strftime(flare_list.Datetimes['Messenger_datetimes'][i].date(),'%Y%m%d')+'EUVI_limb.png'
    foo.savefig(fname)
    
    #ct=coord_transform(aia_bottom_left,aia_width,aia_height,maps)
    #plot_aia(maps,aia_bottom_left,aia_width,aia_height,flare_list,i)
    #try:
    #    corner=pick_corner(ct)
    #except IndexError:
    #    pass
    #plot_stereo(ct,maps,aia_width,aia_height,flare_list,i,corner=corner)

def loop_flares(filen=False):
    import data_management2 as d
    if not filen:
        flare_list=d.OCData('flare_lists/list_final.sav')
    else:
        flare_list=d.OCData('flare_lists/'+filen)
    for i in range(1,len(flare_list.ID)):
        one_flare(flare_list,i)

def download_for_IDL(flare_list):
    for i in range(1,len(flare_list.ID)):
        files=[]#search_local_fits(flare_list,i)
    #ff=[files[0][0],files[1][0]]
        if files == []:
            vc = vso.VSOClient()
            res=query(flare_list,vc,i,'304',timedelta=30)
            if not res:
                pass
            for r in res:
                try:
                    files.append(download_files(vc,r))
                except AttributeError:
                    continue

    
def search_local_fits(flare_list,i,aia=True,euv=True,wave='304'):
    '''see if the file is already available locally in ~/sunpy/data'''
    import glob
    files=[]
    if aia:
        nameaia='aia_lev1_'+wave+'a_'+dt.strftime(flare_list.Datetimes['Messenger_datetimes'][i],'%Y_%m_%d')+'*.fits'
        #print nameaia
        files.append(glob.glob('/Users/wheatley/sunpy/data/'+nameaia))
    if euv:
        nameeuv=dt.strftime(flare_list.Datetimes['Messenger_datetimes'][i],'%Y%m%d')+'_*eub.*fts'
        files.append(glob.glob('/Users/wheatley/sunpy/data/'+nameeuv))
    return files
    
#import data_management2 as d
#flare_list=d.OCData('flare_lists/list_final.sav')
#vc = vso.VSOClient()
#for i in range(1,len(flare_list.ID)):
#    #i=19 #i=8 gets the location wrong...
#    i=11
#    files=[]
#    res=query(flare_list,vc,i)
#    if not res:
#        continue
#    for r in res:
#        try:
#            files.append(download_files(vc,r))
#        except AttributeError:
#            continue
#    for f in files[0:2]:
#        try:
#            if not 'aia' in f[0]:
#                continue
#        except IndexError:
#            continue
#    aia_bottom_left,aia_width,aia_height=coords_aia(flare_list,i)
    files=['/Users/wheatley/sunpy/data/aia_lev1_304a_2012_06_25t18_15_20_12z_image_lev1.4.fits', ['/Users/wheatley/sunpy/data/20120625_181530_n4eub.4.fts']]
    maps=make_maps(files)
    test=get_limb(maps['AIA 4'])
#    if maps == []:
#        continue
    ct=coord_transform(aia_bottom_left,aia_width,aia_height,maps)
    plot_aia(maps,aia_bottom_left,aia_width,aia_height,flare_list,i)
    try:
        corner=pick_corner(ct)
    except IndexError:
        pass
    plot_stereo(ct,maps,aia_width,aia_height,flare_list,i,corner=corner)
    #hgs=point_check(maps, flare_list,i)
    #corner=pick_corner(ct)
    #hgs=point_check(maps, flare_list,i,corner=corner)
#foo=plot_maps(maps,flare_list,i)

