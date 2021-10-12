 #######################################
# stereo_angle.py
# Erica Lastufka 31/10/2017  

#Description: Calculate angle between STEREO and Earth from CalTech ASCII files
#######################################

#######################################
# Usage:

######################################

import numpy as np
import scipy.constants as sc
from datetime import datetime as dt
from datetime import timedelta as td
import os
from sunpy_map_utils import find_centroid_from_map

def get_fits_times(flist):
    dlist=[]
    for f in flist:
        dlist.append(dt.strptime(f[f.rfind('/')+1:f.rfind('_')],"%Y%m%d_%H%M%S"))
    return dlist
    
def nitta_method(peak_fits,pre_fits):
    '''nitta method of prepped fulldisk peak image - prepped fulldisk 1-hour preflare image total flux'''
    kmap=sunpy.map.Map(peak_fits)
    pmap=sunpy.map.Map(pre_fits)
    return np.sum(kmap.data - pmap.data)
    
def peak_flux_location(peak_fits,pre_fits,show=False,peak_flux=False,binning=8):
    '''Get the location of the peak in the difference image. Do this by binning and finding location of brightest pixel in binned image (after un-binning) '''
    kmap=sunpy.map.Map(peak_fits)
    pmap=sunpy.map.Map(pre_fits)
    dmap=sunpy.map.Map(kmap.data-pmap.data,kmap.meta)
    bim=downscale_local_mean(dmap.data, (binning,binning),clip=True)
    bmax=np.where(bim==np.max(bim))
    bmax_x=[bmax[0][0]*binning,bmax[0][0]*binning + binning]
    bmax_y=[bmax[1][0]*binning,bmax[1][0]*binning + binning]
    unbinned=dmap.data[int(bmax_x[0]):int(bmax_x[1]),int(bmax_y[0]):int(bmax_y[1])]
    mps=np.where(unbinned==np.max(unbinned))
    hpc_cs=dmap.pixel_to_world(x= (bmax_y[0]+mps[1][0])*u.pixel, y=(bmax_x[0]+mps[0][0])*u.pixel) #remember x-axis is second dim of data array as always
    if peak_flux: #use the largest contour
        return None
    return hpc_cs

def get_stereo_angle(stereo_date, stereo='A'):
    '''Input is date and time in datetime format'''
    os.chdir('/Users/wheatley/Documents/Solar/occulted_flares/data/stereo-aia/')
    year=stereo_date.year
    if stereo == 'A':
        stereo='ahead'
    else:
        stereo='behind'
    pfile='position_'+stereo+'_'+str(year)+'_HEE.txt'
    #day as day in year
    day=dt.strftime(stereo_date,'%-j')
    #time in seconds of day
    hour=stereo_date.hour
    minute=stereo_date.minute
    seconds=hour*3600 + minute*60.
    with open(pfile) as pf:
        for line in pf.readlines():
            #what's the closest time ...
            l=line.split()
            pyear=l[0]
            pday=l[1]
            pseconds=l[2]
            if pday ==day and np.abs(int(pseconds)-int(seconds)) < 2000.:
                #print line
                heex=float(l[4])
                heey0=float(l[5])
                #flag?
                flag=l[3]
                if flag == '0':
                    print('warning: data is not definitive') #check the next one
                #convert to angle
                if np.arctan(heey0/heex) < 0:
                    heey=-1*heey0
                else:
                    heey=heey0
                angle=(90.-np.arctan(heey/heex)*180./np.pi )+90.
                #print heex,heey,angle
                break
    os.chdir('/Users/wheatley/Documents/Solar/occulted_flares')
    return heex,heey0,angle
    

