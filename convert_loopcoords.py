 #######################################
# convert_loopcoords.py
# Erica Lastufka 17/4/18 

#Description: Convert loop coordinates returned by Lucia's loop tracing program to heliographic, hpc_aia or hpc_stereoA or hpc_stereoB
#######################################

import numpy as np
import glob
import pickle
import sunpy.map
from datetime import datetime as dt
import os
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import sunpy.coordinates
import sunpy.coordinates.wcs_utils
from sunpy.coordinates import frames
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import readsav

def hee_to_hgs(hee,date_obs=False, inframe=False):
    '''convert HEE coord to hgs coordinate. HEE coord can be output of run_trace_loops for example. Cartesian has been corrected for Rsun to arcsec. see transform_loop.py'''
    hee_lc=[]
    arc2km=719.684*u.km
    if type(hee) == str:
        data=readsav(hee,python_dict=True)
        coords=data['coords']
        #if data['time'][1] == '-':
        #    data['time']='0'+data['time'] #add a zero in front
        date_obs=dt.strptime(data['time'][:-4],'%d-%b-%Y %H:%M:%S')
        for x,y,z in zip(coords[0],coords[1],coords[2]):
            hee_lc.append(SkyCoord(x*arc2km,y*arc2km,z*arc2km,frame=frames.Heliocentric,obstime=date_obs)) # it doesn't like arcseconds
    elif type(hee) == list:
        for h in hee:
            if type(h) !=SkyCoord:
                hee_lc.append(SkyCoord(coords[0]*u.arcsec,coords[1]*u.arcsec,coords[2]*u.arcsec,frame=frames.Heliocentric))
    else:
        if type(hee) !=SkyCoord:        
            hee_lc=SkyCoord(coords[0]*u.arcsec,coords[1]*u.arcsec,coords[2]*u.arcsec,frame=frames.Heliocentric)
        else:
            hee_lc=hee
            
    hgs_lc=[h.transform_to(frames.HeliographicStonyhurst) for h in hee_lc]
    return hgs_lc


def hgs_to_hpc_aia(hgs,date_obs=True,aia_frame=False,rsun=695508*u.km): #maybe fix rsun later...
    '''convert hgs coord to hpc_aia coordinate. hgs coord can be output of scc_measure for example'''
    #hgs is a list of lat,lon,radius in terms of rsun tuples
    sclist=[]
    if type(hgs) == list:
        for h in hgs:
            if type(h) !=SkyCoord:
                sch=SkyCoord(h[0]*u.deg,h[1]*u.deg,h[2]*rsun,frame=frames.HeliographicStonyhurst,obstime=date_obs)
                sclist.append(sch)
            else:
                sclist.append(h)
    else:
        if type(hgs) !=SkyCoord:
            sch=SkyCoord(hgs[0]*u.deg,hgs[1]*u.deg,hgs[2]*rsun,frame=frames.HeliographicStonyhurst,obstime=date_obs)
            sclist.append(sch)
        else:
            sclist=[hgs]

    hpc_aia=[]
    for sc in sclist:
        if not aia_frame:
            hpc_aia.append(sc.transform_to(frames.Helioprojective))
        else:
            hpc_aia.append(sc.transform_to(frames.aia_frame))
            
    return hpc_aia

'''
def hpcB_to_hpcA(hpcB,date_obs,reverse=False):
    ''convert hpc (stereoB) coord to hpc (stereo A) coordinate. input can be ''
    #hgs is a list of lat,lon,radius tuples
    if type(hgs) = list:
        for h in hgs:
            if type(h) !=SkyCoord:
                sch=SkyCoord(h[0]*u.deg,h[1]*u.deg,h[2]*rsun,frame=frames.Heliographic,obstime=date_obs)
                sclist.append(sch)
            else:
                sclist.append(h)
    else:
        if type(hgs) !=SkyCoord:
            sch=SkyCoord(hgs[0]*u.deg,hgs[1]*u.deg,hgs[2]*rsun,frame=frames.Heliographic,obstime=date_obs)
            sclist.append(sch)
        else:
            sclist=[hgs]


def hpcstereo_to_hgs(hpcstereo,date_obs):
    ''convert hgs coord to hpc_aia coordinate. hgs coord can be output of scc_measure for example''
    #hgs is a list of lat,lon,radius tuples
    if type(hgs) = list:
        for h in hgs:
            if type(h) !=SkyCoord:
                sch=SkyCoord(h[0]*u.deg,h[1]*u.deg,h[2]*rsun,frame=frames.Heliographic,obstime=date_obs)
                sclist.append(sch)
            else:
                sclist.append(h)
    else:
        if type(hgs) !=SkyCoord:
            sch=SkyCoord(hgs[0]*u.deg,hgs[1]*u.deg,hgs[2]*rsun,frame=frames.Heliographic,obstime=date_obs)
            sclist.append(sch)
        else:
            sclist=[hgs]

def hpcaia_to_hgs(hpcaia,date_obs):
    ''convert hgs coord to hpc_aia coordinate. hgs coord can be output of scc_measure for example''
    #hgs is a list of lat,lon,radius tuples
    if type(hgs) = list:
        for h in hgs:
            if type(h) !=SkyCoord:
                sch=SkyCoord(h[0]*u.deg,h[1]*u.deg,h[2]*rsun,frame=frames.HeliographicStonyhurst,obstime=date_obs)
                sclist.append(sch)
            else:
                sclist.append(h)
    else:
        if type(hgs) !=SkyCoord:
            sch=SkyCoord(hgs[0]*u.deg,hgs[1]*u.deg,hgs[2]*rsun,frame=frames.Heliographic,obstime=date_obs)
            sclist.append(sch)
        else:
            sclist=[hgs]
'''

def plot_hgs_3D(hgs,rsun_km=695508.,xran=[67.5,70],yran=[22,27],zran=[0,15]):
    '''plot in 3D the heliographic coordinates. Convert distance to Mm'''
    title=dt.strftime(hgs[0].obstime.value,'%Y-%m-%d %H:%M:%S')
    hgslon=[h.lon.value for h in hgs]
    hgslat=[h.lat.value for h in hgs]
    heightMm=[(h.radius.value-rsun_km)/1000. for h in hgs]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(hgslon, hgslat, heightMm, c='b', marker='o')
    ax.plot(hgslon,heightMm,'r--',zdir='y',zs=yran[0])
    ax.plot(hgslat,heightMm,'m--',zdir='x',zs=xran[0])
    #ax.plot(hgslon,hgslat,'k--',zdir='z',zs=zran[0])
    ax.set_xlabel('X-Longditude (deg)')
    ax.set_ylabel('Y-Latitude (deg)')
    ax.set_zlabel('Z-Height (Mm)')
    ax.set_xlim(xran)
    ax.set_ylim(yran)
    ax.set_zlim(zran)
    ax.set_title(title)

    fig.show()


