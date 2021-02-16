 #######################################
# query_vso.py
# Erica Lastufka 21/05/2018  

#Description: Isolate FeXVIII line in AIA maps and/or lightcurves
#######################################

#######################################
# Usage:
#	here how I made the FeXVIII maps. 
#	read in AIA fits files of 94A, 171A, 211A with your favorite method to get map structures: s094, s171, s211
#	;then co-register the images
#c211=coreg_map(s211,s094)
#c171=coreg_map(s171,s094)
#;FeXVIII structure 
#s18=s094
#s18.id='Fe XVIII:'
#s18.data=s094.data-c211.data/120.-c171.data/450.

######################################

import numpy as np
import scipy.constants as sc
from datetime import datetime as dt
from datetime import timedelta as td
import os
from sunpy.net import Fido, attrs as a
import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord
import pidly
import aia_dem_batch as aia
import make_aligned_movie as mov

def fits2mapIDL(files,coreg=True):
    '''run fits2map in idl. List of files should be: AIA094.fits, AIA171.fits,AIA211.fits'''
    idl = pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
    idl('files',files)
    idl('fits2map,files,maps')
    if coreg:
        idl('refmap=maps[0]')
        idl('c171=coreg_map(maps[1],refmap)')
        idl('c211=coreg_map(maps[2],refmap)')
        coreg_map171= 'AIA171_coreg94_' + files[1][8:-5]#assuming AIA files are in the usual level 1.5 naming convention
        coreg_map211= 'AIA211_coreg94_' +files[2][8:-5]       
        idl('coreg_map171',coreg_map171)
        idl('coreg_map211',coreg_map211)
        idl('map2fits,c171,coreg_map171')
        idl('map2fits,c211,coreg_map211')
        return coreg_map171,coreg_map211

def get_Fe18(map94,map171,map211,submap=False,save2fits=False):
    if type(map94) == str:
        map94=sunpy.map.Map(map94)
    if type(map171) == str:
        map171=sunpy.map.Map(map171) #the coregistered map
    if type(map211) == str:
        map211=sunpy.map.Map(map211) #the coregistered map
    if submap:
        map94=map94.submap(SkyCoord(submap[0]*u.arcsec,submap[1]*u.arcsec,frame=map94.coordinate_frame))
        map171=map171.submap(SkyCoord(submap[0]*u.arcsec,submap[1]*u.arcsec,frame=map171.coordinate_frame))
        map211=map211.submap(SkyCoord(submap[0]*u.arcsec,submap[1]*u.arcsec,frame=map211.coordinate_frame))
    try:
        map18data=map94.data-map211.data/120.-map171.data/450.
    except ValueError: #shapes are different- is 94 always the culprit?
        d94=map94.data[1:,:]
        map18data=d94-map211.data/120.-map171.data/450.
    map18=sunpy.map.Map(map18data,map94.meta)
    #if submap:
    #    map18=map18.submap(SkyCoord(submap[0]*u.arcsec,submap[1]*u.arcsec,frame=map18.coordinate_frame))
    if save2fits:
        filename='AIA_Fe18_'+map18.meta['date-obs']+'.fits'
        map18.save(filename)
    return map18

def Fe18_from_groups(start_index,end_index,moviename,submap=[(-1100,-850),(0,400)],framerate=12,imnames='Fe18_'):
    '''generate the Fe18 images from the DEM groups'''
    preppedfiles=glob.glob('AIA_*.fits')    
    groups=aia.group6(preppedfiles)
    print(len(groups))
    for i,g in enumerate(groups[start_index:end_index]):
        map94=sunpy.map.Map(g[0])
        map171=sunpy.map.Map(g[2])
        map211=sunpy.map.Map(g[4])
        map18=get_Fe18(map94,map171,map211,submap=submap) #save to listl/mapcube? do something
        plt.clf()
        map18.plot()
        plt.savefig(imnames+'{0:03d}'.format(i)+'.png')

    mov.run_ffmpeg(imnames+'%03d.png',moviename,framerate=framerate)
        
        


