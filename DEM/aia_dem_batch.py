 #######################################
#aia_dem_batch.py
# Erica Lastufka 29/03/2018  

#Description: Run Sparse DEM for AIA data in given folders, store, clean up
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
#import display_aia_dem as disp
import zipfile
import pidly
from datetime import datetime as dt

from skimage.transform import downscale_local_mean


def dem_from_sav(filename):
    dem_dict=readsav(filename,python_dict=True)
    return dem_dict

def group6(files):
    '''group AIA data into blocks of 6: '''
    #get time tags for each file, put in a dict: {filename:x,time:y,closest_files:[(file1,td1),(file2,td2)]}
    fdict={'094':[],'131':[],'171':[],'193':[],'211':[],'335':[]}
    wavelengths=['094','131','171','193','211','335']
    for f in files:

        try:
            ttag=dt.strptime(f[8:-5],'%Y%m%d_%H%M%S')
            wave=f[4:7]
        except ValueError:
            ttag=dt.strptime(f[4:-10],'%Y%m%d_%H%M%S')
            wave=f[-8:-5]

        #ttag=dt.strptime(f[8:-5],'%Y%m%d_%H%M%S')
        #wave=f[4:7]

        tdict={'filename':f,'time':ttag,'closest_files':[],'tds':[],'group_id':0}
        fdict[wave].append(tdict)
        #find the closest files in other wavelengths
        #for w in fdict.keys():
        #    wavelengths=['094','131','171','193','211','335']
        #    wavelengths.remove(wave) #all the others remain
    for f in fdict['094']: #get the closest file in other wavelengths
        #print f,ttag
        ttag=f['time']
        f['closest_files']=[find_closest_file(ttag,fdict,wave=ww) for ww in wavelengths]
        #don't forget to sort
    groups=[f['closest_files'] for f in fdict['094']]
    #return list of groups ...
    return groups

def find_closest_file(ttag,fdict,wave='094'):
    '''what it sounds like'''
    tvec=[fdict[wave][idx]['time'] for idx in range(0,len(fdict[wave]))]
    fvec=[fdict[wave][idx]['filename'] for idx in range(0,len(fdict[wave]))] #hopefully this doesn't mess up the sorting
    #find closest t in tvec to given ttag
    closest_t= min(tvec, key=lambda x: abs(x - ttag))
    ct_idx=tvec.index(closest_t)
    closest_file=fvec[ct_idx]
    #print closest_t,ct_idx,closest_file
    #return corresponding filename... figure out a way to return timedelta as well
    return closest_file
        

def bin_images(gdat,n=2):
    ''' spatially bin images in n x n bins, crop array if it doesnt fit.g6 is list of file names in order[94,131,171,193,211,335] '''
    g6_binned=[]
    if type(gdat) == np.ndarray:
        gprep=[gdat[:,:,i] for i in range(6)]
    else:
        gprep=gdat
    for g in gprep:
        #gdat=sunpy.map.Map(g).data
        if type(n) == int:
            gbin=downscale_local_mean(g, (n,n),clip=True)
            g6_binned.append(gbin)
            g6_arr=np.array(g6_binned)
        elif n =='all':
            g6_binned.append(np.nanmean(g))
            g6_arr=np.array(g6_binned)
    return g6_arr #but make it an array...
        
def run_sparse_dem(group,timestamp,fov=False,binning=False,flat=True):
    #first trim maps into fov, stick in list
    gdat=[]
    if type(group)==pd.DataFrame: #it's already binned data...
        gdict= get_datavec(group,timestamp,plus=False,minus=False,total=False,data=True,filenames=True)#get from
        gdat=[g['data'] for g in gdict]
        group=[g['filenames'][g['filenames'].rfind('/')+1:] for g in gdict]
        #print(group[0])
        if binning:
            binfac=float(binning)*.3
            submap=bin_images(gdat,n=binning) #do i need to transpose?
            nx,ny,_=np.shape(submap)

    else:
        for g in group:
            gmap=sunpy.map.Map(g)
            print(gmap.meta['wavelnth'],np.shape(gmap.data))
            fbl=SkyCoord(fov[0]*u.arcsec,fov[2]*u.arcsec,frame=gmap.coordinate_frame)
            ftr=SkyCoord(fov[1]*u.arcsec,fov[3]*u.arcsec,frame=gmap.coordinate_frame)
            gdat.append(gmap.submap(fbl,ftr).data)
        #bin if necessary
        if binning:
            #print(np.shape(gdat))
            submap=bin_images(gdat,n=binning)
            print(np.shape(submap))
            binfac=float(binning)*.3
        else:
            submap=np.array(gdat)

    idl = pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
    idl('.compile /Users/wheatley/Documents/Solar/DEM/tutorial_dem_webinar/aia_sparse_em_init.pro')
    idl('.compile /Users/wheatley/Documents/Solar/occulted_flares/code/run_sparse_dem.pro')
    idl('group6', group)
    idl('submap',submap) #now submap is actually the Map.submap.data
    idl('fov',fov)
    #idl('arr_shape',arr_shape)
    if binning:
        idl('binning',binfac)
        idl('result=run_sparse_dem(group6, submap,fov,binning)')
    else:
        idl('result=run_sparse_dem(group6, submap,fov,1)')

    result=idl.result
    #get out the result
    if result == 1:
        print("sparse dem run successfully")
        #fname=strmid(files[0],3,16)+'bin'+strtrim(string(binning),1)
        filename=group[0][3:18]+'_bin'+str(np.round(binfac,1))[:5]+'*.sav'
        print(filename)
        aa=glob.glob(filename)[0]
        demdict=dem_from_sav(aa) #what is the filename? check idl code
        #dict_keys(['lgtaxis', 'status', 'dispimage', 'emcube', 'coeff', 'image'])
        try:
            lgtaxis=demdict['lgtaxis']
            status=demdict['status']
            emcube=demdict['emcube']
            coeff=demdict['coeff']
        except KeyError:
            print('something went wrong, empty sav file ',aa)
            return None
        
    if flat:
        #flatten by tvec and turn into dataframe
        dfdict={}
        dfdict['timestamp']=pd.Series(timestamp)
        dfdict['coeff']=pd.Series([coeff]) #is this in coeff?
        dfdict['coeff_mean']=pd.Series(np.nanmean(coeff[coeff != 0])) #drop zeros
        dfdict['coeff_std']=pd.Series(np.nanstd(coeff[coeff != 0]))
        nx,ny=np.shape(status)
        dfdict['nx']=pd.Series(nx)
        dfdict['ny']=pd.Series(ny)
        #print(np.shape(status),nx,ny) good
        #print(np.shape(lgtaxis[:-1]))
        #print(np.shape(emcube)) #21, 12, 24 ie nT, nx,ny
        dfdict['status']=pd.Series([status])
        cnonzero=np.count_nonzero(status)
        if cnonzero== 0:
            fnonzero=1
        else:
            fnonzero=((nx*ny)-cnonzero)/(nx*ny)
        dfdict['fraction_nonzero']=pd.Series(fnonzero) #0 means yes! whyyyyy

        if binning:
            dfdict['binning']=pd.Series(b)
            dfdict['bin_fac']=pd.Series(binfac)
        else:
            dfdict['binning']=pd.Series(None)
        
        for i,t in enumerate(lgtaxis[:-1]):
            demkey='dem_'+str(t)
            #edemkey='edem_'+str(t)
            #elogtkey='elogt_'+str(t)
            dfdict[demkey]=pd.Series([emcube[i]])
            #dfdict[edemkey]=pd.Series([edem[:,:,i]])
            #dfdict[elogtkey]=pd.Series([elogt[:,:,i]])
            dfdict[demkey+'_mean']=pd.Series(np.nanmean(emcube[i][emcube[i] !=0])) #might be nans not zeros....
            #dfdict[edemkey+'_mean']=pd.Series(np.mean(edem[:,:,i][edem[:,:,i] !=0]))
            #dfdict[elogtkey+'_mean']=pd.Series(np.mean(elogt[:,:,i][elogt[:,:,i] !=0]))
            try:
                dfdict[demkey+'_max']=pd.Series(np.nanmax(emcube[i][emcube[i] !=0]))
            except ValueError: #it's all zeros
                dfdict[demkey+'_max']=pd.Series(0)
            #dfdict[edemkey+'_max']=pd.Series(np.max(edem[:,:,i][edem[:,:,i] !=0]))
            #dfdict[elogtkey+'_max']=pd.Series(np.max(elogt[:,:,i][elogt[:,:,i] !=0]))
            #print(i,np.mean(np.nonzero(dem[:,:,i])))
            #calculate percent zeros cuz why not

    
        df=pd.DataFrame(dfdict)
    
        return df



#if __name__ == '__main__':
#    os.chdir('low_cadence_cutout')
#    #os.chdir('longbefore')
#    files=glob.glob('ssw_cutout*.fts')
#    #files=glob.glob('aia_lev1*.fits')
#    aia_prep(files,zip_old=False)
#    preppedfiles=glob.glob('AIA_*.fits')
#    groups=group6(preppedfiles)
#    #print groups #should be sorted already
#    do_over=[]
#    for g in groups[2:]:
#        res=run_sparse_dem(g,[-1220,-800,0,400])
#        if res !=1:
#            do_over.append(g)
#    pickle.dump(do_over,open('do_over.p','wb'))
