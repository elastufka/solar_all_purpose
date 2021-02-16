 #######################################
# test_limb.py
# Erica Lastufka 12/5/2017  

#Description: Test limb oplot via matplotlib and SkyCoord
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
    
def coords_aia(aia_map):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=aia_map.wcs)
    limb=aia_map.draw_limb(axes=ax,transform=ax.get_transform('world'))   #is this a wcs object? no it's a list containing a matplotlib.patches.circle object
    return ax,limb[0]

def coord_transform(aia_map,ax,limb):

    hpc_aia = SkyCoord((limb._path._vertices, frame=maps['AIA 4'].coordinate_frame)

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
        subaia.peek(draw_grid=True)
        #subaia.save(fname+'zoom.png',filetype='png')
    
def plot_stereo(hpc_B,maps,aia_width,aia_height,flare_list,i,corner=False,quiet=True,save=False):    
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
    
    
#import data_management2 as d
#flare_list=d.Data_Study('flare_lists/list_final.sav')
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

