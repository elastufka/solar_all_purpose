import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import re

import sunpy.map
import numpy as np
import numpy.ma as ma
import matplotlib.dates as mdates
import pandas as pd

from datetime import datetime as dt
import glob
import os
import plotly.graph_objects as go
import matplotlib
from matplotlib import cm
import pidly

from sunpy_map_utils import track_region_box, make_submaps
#from aiapy.calibrate import *
#from aiapy.response import Channel
from aiapy.calibrate import normalize_exposure, register, update_pointing, correct_degradation, degradation


def aia_prep_py(files,expnorm=True,tofits=True,path=''):
    """`aia_prep <https://hesperia.gsfc.nasa.gov/ssw/sdo/aia/idl/calibration/aia_prep.pro>` using aiapy instead of IDL.
    See `aiapy <https://aiapy.readthedocs.io/en/latest/generated/gallery/prepping_level_1_data.html>`_.
    
    Args:
        files (iterable): Full paths and filenames of FITS file(s) to be prepped.
        expnorm (bool, optional): Whether or not to perform exposure normalization. Defaults to True.
        tofits (bool, optional): Whether or not to write the prepped maps to FITS files. Defaults to True.
        path (str, optional): Where to write the prepped FITS files to. Defaults to current working directory.
    
    Returns:
        List of FITS files names (if to_fits = True) or list of Sunpy Maps
    """
    maplist=[]
    fnames=[]
    for f in files:
        m= sunpy.map.Map(f)
        try:
            m_updated_pointing = update_pointing(m)
            m_registered = register(m_updated_pointing)
        except ValueError: #not full-disk image
            m_registered=m
        if expnorm:
            m_out = normalize_exposure(m_registered)
        else:
            m_out=m_registered
        if tofits: #save to fitsfile
            if '/' not in f:
                fname=f"{path}{f[:-5]}_prepped.fits"
            else:
                fname=f"{f[:-5]}_prepped.fits"
            m_out.save(fname)
            fnames.append(fname)
        maplist.append(m_out)
    if tofits:
        return fnames
    else:
        #if not tofits:
        return maplist
    
def aia_correct_degradation(maplist):
    """Correct for telescope degradation over time.
    See `aiapy <https://aiapy.readthedocs.io/en/latest/generated/gallery/skip_correct_degradation.html>. It is not very fast.
    
    Args:
        maplist (iterable): Iterable of Sunpy Map objects
    
    Returns:
        list of degradation-corrected Sunpy Maps"""
    from aiapy.calibrate.util import get_correction_table
    correction_table = get_correction_table()
    maps_corrected = [correct_degradation(m, correction_table=correction_table) for m in maplist]
    return maps_corrected
    
def dump_AIA_degradation(obstime,channels=[94,131,171,193,211,335],calibration_version=10,json=True):
    """Compute AIA telescope degradation over time, for a given date. Store in JSON or return as list for use in lower Python or Sunpy versions, or to apply to a large number of maps since aia_correct_degradation is slow.
    
    Args:
        obstime (str, datetime, int, float): Observation time in format readable by astropy.Time.
        channels (list, optional): List of AIA channels to calculate degradation for. Defaults to the EUV DEM channels (all but 304)
        calibration_version (int, optional): Which calibration version to use. Defaults to 10.
        json (bool, optional): If True, store the calculated degradation factors in JSON format. Defaults to True
        
    
    Returns:
        None or list of degradation factors.
        """
    utctime = Time(obstime, scale='utc')

    nc = len(channels)
    degs = np.empty(nc)
    for i in np.arange(nc):
        degs[i] = degradation(channels[i]* u.angstrom,utctime,calibration_version=calibration_version)
    df = pd.DataFrame({'channels':channels,'degradation':degs})
    if json:
        df.to_json('AIA_degradation_%s.json' % obstime)
    else:
        return degs
    
def get_aia_response_py(obstime, channels=[94,131,171,193,211,335]):
    """Get the AIA wavelength response using AIApy instead of ssw (temperature response would be more useful once chiantifix is implemented).
    See `AIApy documentation <https://aiapy.readthedocs.io/en/latest/generated/gallery/calculate_response_function.html>` and also `Mark's implementation <https://gitlab.com/LMSAL_HUB/aia_hub/aiapy/-/issues/23>`.
    
    Args:
        obstime (str, datetime, int, float): Observation time in format readable by astropy.Time.
        channels (list, optional): List of AIA channels to calculate degradation for. Defaults to the EUV DEM channels (all but 304)
    
    Returns:
        List of wavelength responses.
        """
    resp = []
    for c in channels:
        chan = Channel(c*u.angstrom)
        resp.append(chan.wavelength_response(obstime=obstime, include_eve_correction=True))
    return resp

def aia_prep_idl(files,path = '.',zip_old=True,preppedfilenames=False):
    """Run IDL AIA prep on given files, clean up
    Args:
        files (list): List of filenames to prep
        path (str, optional): Where to write the prepped FITS files to. Defaults to current working directory.
        zip_old (bool, optional): Whether or not to zip the level 0 files. Defaults to True.
        preppedfilenames (bool, optional): Whether or not to return the prepped file names. Defaults to False.
    
    Returns:
        None or list of prepped file names"""
    idl = pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
    idl('files',files)
    idl('outdir',outdir)
    idl('aia_prep,files,-1,/do_write_fits,/normalize,outdir=outdir')
    prepped_files=glob.glob(f"{path}/AIA2*.fits")

    if zip_old: #archive level 0 data
        zipf = zipfile.ZipFile('AIALevel0.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir(files, zipf)
        zipf.close()
    if preppedfilenames:
        return prepped_files
        
def timestamp_from_filename(aia_file):
    """Get timestamp from AIA file name.
    
    Args:
        aia_file (str): The AIA file name.
    
    Returns:
        datetime.datetime : timestamp associated with file name."""
    n1 = re.search(r"\d", aia_file).start()
    newstr=aia_file[n1:].replace('-','').replace(':','').replace('_','T')
    try:
        tstamp=pd.to_datetime(newstr[:16])
    except Exception:
        tstamp=timestamp_from_filename(newstr[1:])
    return tstamp

def zipdir(files, ziph):
    """Add files to ZipFile object.
    
    Args:
        files (list): The file names.
        ziph (zipfile.ZipFile) : Zipfile object
    
    Returns:
        None. ZipFile object is modified inplace."""
    # ziph is zipfile handle
    root=os.getcwd()
    for f in files:
        ziph.write(os.path.join(root, f))

def make_Fe18_map(map94,map171,map211,bottom_left_coord=None,top_right_coord=None, save2fits=False):
    """Make an Fe18 map (see reference). Should be run on co-registered maps, see sunpy_map_utils fits2mapIDL(). (Is co-registration done in creating a CompositeMap?)
    
    Args:
        map94 (sunpy.map.Map or str): AIA 94A map or FITS filename
        map171 (sunpy.map.Map or str): AIA 171A map or FITS filename
        map211 (sunpy.map.Map or str): AIA 211A map or FITS filename
        bottom_left_coord (tuple or SkyCoord, optional): Bottom left coordinate of desired submap. Defaults to None.
        top_right_coord (tuple or SkyCoord, optional): Top right coordinate of desired submap. Defaults to None.
        save2fits (bool, optional): Defaults to False.
    Returns:
        sunpy.map.Map : the Fe18 map
    """
    
    if isinstance(map94, str):
        map94=sunpy.map.Map(map94)
    if isinstance(map171, str):
        map171=sunpy.map.Map(map171) #the coregistered map
    if isinstance(map211, str):
        map211=sunpy.map.Map(map211) #the coregistered map
        
    if bottom_left_coord is not None:
        maps = []
        for m in [map94, map171, map211]:
            if not isinstance(bottom_left_coord, SkyCoord):
                bottom_left_coord = SkyCoord(bottom_left_coord[0]*u.arcsec, bottom_left_coord[1]*u.arcsec, frame = m.coordinate_frame)
            if not isinstance(top_right_coord, SkyCoord):
                top_right_coord = SkyCoord(top_right_coord[0]*u.arcsec, top_right_coord[1]*u.arcsec, frame = m.coordinate_frame)
            maps.append(m.submap(bottom_left_coord, top_right = top_right_coord))
        map94, map171, map211 = maps
    
    #check shapes:
    d94 = map94.data
    d171 = map171.data
    d211 = map211.data
    if not d94.shape == d171.shape == d211.shape:
        #trim to smallest shape...
        xmin = np.min([d94.shape[0], d171.shape[0], d211.shape[0]])
        ymin = np.min([d94.shape[1], d171.shape[1], d211.shape[1]])
        d94 = d94[:xmin, :ymin]
        d171 = d171[:xmin, :ymin]
        d211 = d211[:xmin, :ymin]
    
    map18data = d94 - d211/120. - d171/450.
    map18 = sunpy.map.Map(map18data,map94.meta)

    if save2fits:
        filename=f"AIA_Fe18_{map18.meta['date-obs']}.fits"
        map18.save(filename)
    return map18
        
def Fe18_from_groups(start_index,end_index,moviename,submap=[(-1100,-850),(0,400)],framerate=12,imnames='Fe18_'):
    """Make movie using Fe18 images generated from the DEM groups. Assumes files are in current working directory and prefaced with 'AIA'.
    
    Args:
        start_index (int): Index of group to start with
        end_index (int): Index of group to end with
        moviename (str): Name of file to save movie as
        submap (list, optional): Bottom-left and top-right coordinates of submap region to use.
        framerate (int, optional): Movie frame rate. Defaults to 12.
        imnames (str, optional): String with which to preface image names.
    """
    import make_aligned_movie as mov
    from dem_utils import group6
    preppedfiles = glob.glob('AIA_*.fits')
    groups = group6(preppedfiles)
    #print(len(groups))
    for i,g in enumerate(groups[start_index:end_index]):
        map94=sunpy.map.Map(g[0])
        map171=sunpy.map.Map(g[2])
        map211=sunpy.map.Map(g[4])
        map18=get_Fe18(map94,map171,map211,submap=submap) #save to listl/mapcube? do something
        plt.clf()
        map18.plot()
        plt.savefig(imnames+'{0:03d}'.format(i)+'.png')

    mov.run_ffmpeg(imnames+'%03d.png',moviename,framerate=framerate)
    

### some plots etc to do with looking at brightening and dimming regions

def plot_AIA_lightcurves(dfaiam3,rolling=False, group=False, get_traces=False, yrange=[.6,1.75],pcolors=[]):
    mode='lines+markers'
    traces=[]
    fig = make_subplots(rows=3, cols=1, start_cell="top-left",shared_xaxes=True)
    for i,ids in enumerate(dfaiam3.wavelength.unique()):
        df=dfaiam3.where(dfaiam3.wavelength == ids).dropna(how='all')
        df.sort_values('timestamps',inplace=True)
        if rolling:
            fp=df.flux_plus.rolling(rolling).mean()/df.mask_plus_px.max()
            fm=df.flux_minus.rolling(rolling).mean()/df.mask_minus_px.max()
            ft=df.flux_total.rolling(rolling).mean()/df.total_mask_px.max()
            tvec= [df.timestamps.iloc[i] for i in range(len(df.timestamps)) if i % rolling == rolling-1]#1 in every n timestamps, rightmost value
        elif group: #groupby every n indices... not yet implemented
            gdf=df.groupby(df.index // group).mean()
            fp=gdf.flux_plus/df.mask_plus_px.max()
            fm=gdf.flux_minus/df.mask_minus_px.max()
            ft=gdf.flux_total/df.total_mask_px.max()
            tvec= [df.timestamps.iloc[i] for i in range(len(df.timestamps)) if i % group == group-1]#1 in every n timestamps, rightmost value
        else:
            fp=df.flux_plus/df.mask_plus_px.max()
            fm=df.flux_minus/df.mask_minus_px.max()
            ft=df.flux_total/df.total_mask_px.max()
            tvec=df.timestamps #1 in every n timestamps...
        
        #unmasked=df.total_mask_px.max()
        #flux_adj=df.fluxes/unmasked#/df.cutout_shape #don't need when mask is same
        fig.add_trace(go.Scatter(x=tvec,y=fp/np.mean(fp),mode=mode,name='AIA '+str(int(ids))+' brightening',line=dict(color=pcolors[i])),row=1,col=1)
        fig.add_trace(go.Scatter(x=tvec,y=fm/np.mean(fm),mode=mode,name='AIA '+str(int(ids))+' dimming',line=dict(color=pcolors[i])),row=2,col=1)
        ttrace=go.Scatter(x=tvec,y=ft/np.mean(ft),mode=mode,name='AIA '+str(int(ids)),line=dict(color=pcolors[i]))
        traces.append(ttrace)
        fig.add_trace(ttrace,row=3,col=1)

    for r in range(1,4):
        fig.update_yaxes(range=yrange,row=r,col=1)

    fig.update_layout(yaxis_title='Brightening',title='Flux/Mean Flux, Box 2')
    fig.update_yaxes(title='Dimming',row=2,col=1)
    fig.update_yaxes(title='Total',row=3,col=1)
    
    if get_traces:
        return traces
    return fig
    
def plot_aia_masks(maskpickle,tag=False):
    if not tag:
        tag=''
    wavs=[94,131,171,193,211,335]
    masks, mask_plus,mask_minus=pickle.load(open(maskpickle,'rb'))
    mdiffs,qdiffs=[],[]
    for w in wavs:
        mdiff=diff_maps([sunpy.map.Map('AIA/AIA_'+str(w)+'_preflare_'+tag+'00.fits'),sunpy.map.Map('AIA/AIA_'+str(w)+'_flare_'+tag+'00.fits')])[0]
        qdiff=diff_maps([sunpy.map.Map('AIA/AIA_'+str(w)+'_preflare_'+tag+'qs00.fits'),sunpy.map.Map('AIA/AIA_'+str(w)+'_flare_'+tag+'qs00.fits')])[0]
        mdiffs.append(mdiff)
        qdiffs.append(qdiff)
        
    fig,ax=plt.subplots(5,6,figsize=[10,8])
    for i in range(6):
        ax[0][i%6].imshow(mdiffs[i].data,cmap=cm.Greys_r, origin='lower left')
        ax[1][i%6].imshow(qdiffs[i].data,cmap=cm.Greys_r, origin='lower left')
        ax[2][i%6].imshow(masks[i],cmap=cm.Blues, origin='lower left')
        ax[0][i%6].set_title(str(wavs[i]))
        ax[3][i%6].imshow(mask_minus[i],cmap=cm.Greens, origin='lower left')
        #ax[1][i%6].set_title(masked_maps[i].meta['wavelnth'])
        ax[4][i%6].imshow(mask_plus[i],cmap=cm.Reds, origin='lower left')
        #ax[2][i%6].set_title(masked_maps[i].meta['wavelnth'])
    fig.show()
    return fig
    
def make_masked_df(flare_box,masks=False,mask_plus=False,mask_minus=False,tag=None):
    #flare_box=[(-610, 30), (-550, 90)]
    if not tag:
        tag=''
    #if not masks and not mask_minus and not mask_plus:
    #    masks, mask_plus,mask_minus=pickle.load(open('all_masks_source1.p','rb'))

    dfs=[]
    #tdicts,pdicts,mdicts=[],[],[]
    for i,w in enumerate([94,131,171,193,211,335]): #do for all masks too...
        submaps=make_submaps('AIA/AIA'+"{:03d}".format(w)+'maps'+tag+'.p',flare_box[0],flare_box[1])
        ftotal=track_region_box(submaps,mask=masks[i],force_mask=True,mask_data=False)
        fplus=track_region_box(submaps,mask=mask_plus[i],force_mask=True,mask_data=False)
        fminus=track_region_box(submaps,mask=mask_minus[i],force_mask=True,mask_data=False)
        
        df=pd.DataFrame(ftotal)
        df.rename(columns={'fluxes':'flux_total'},inplace=True)
        df['flux_plus']=pd.Series(fplus['fluxes'])
        df['flux_minus']=pd.Series(fminus['fluxes'])
        df['total_mask']=pd.Series([masks[i]])
        df['mask_plus']=pd.Series([mask_plus[i]])
        df['mask_minus']=pd.Series([mask_minus[i]])
        df['total_mask_px']=pd.Series(np.sum(np.logical_not(masks[i])))
        df['mask_plus_px']=pd.Series(np.sum(np.logical_not(mask_plus[i])))
        df['mask_minus_px']=pd.Series(np.sum(np.logical_not(mask_minus[i])))
        
        umflux=[]
        for j in df.index:
            umflux.append(np.nanmean(df.data[j]*~masks[i]))
        umpx=np.sum(masks[i])
        df['outside_mask_flux']=umflux
        df['outside_mask_px']=[umpx for i in range(len(df.index))]
        df.sort_values('timestamps',inplace=True)
        df['wavelength']=w
        dfs.append(df)
        
    dfaiam=pd.concat(dfs)
    dfaiam.reset_index(inplace=True)
    #dfaiam3.to_json('AIA_full_masked_3s.json',default_handler=str)
    return dfaiam

def make_total_masks(df):
    '''sum masks over all wavelengths, fill in empty values in the dataframe if they exist '''
    #get masks
    refill=False
    if len(list(df.total_mask.dropna(how='all'))) < 7: #need to fill in values
        refill=True
        
    for i,w in enumerate([94,131,171,193,211,335]):
        sdf=df.where(df.wavelength==w).dropna(how='all')
        maskt=sdf.total_mask.dropna(how='all').iloc[0]
        maskp=sdf.mask_plus.dropna(how='all').iloc[0]
        maskm=sdf.mask_minus.dropna(how='all').iloc[0]
        #if refill:
            #for t in sdf.index: #this is super slow!
            #    df['total_mask'][t]=maskt
            #     df['mask_plus'][t]=maskp
            #    df['mask_minus'][t]=maskm
        try:
            all_tmask=all_tmask+~np.array(maskt)
            all_pmask=all_pmask+~np.array(maskp)
            all_mmask=all_mmask+~np.array(maskm)
        except NameError:
            all_tmask=~np.array(maskt)
            all_pmask=~np.array(maskp)
            all_mmask=~np.array(maskm)
    
    df['total_mask_wavelengths']=[[~all_tmask] for a in df.index]
    df['total_pmask_wavelengths']=[[~all_pmask] for a in df.index]
    df['total_mmask_wavelengths']=[[~all_mmask] for a in df.index]
    if type(df['timestamps'][0]) == str:
        df['timestamps']=pd.to_datetime(df.timestamps)
    
    return df

def aia_maps_tint(dfaia,timerange=["2020-09-12T20:40:00","2020-09-12T20:41:00"],how=np.nanmean,wavs=[94,131,171,193,211,335]):
    '''Get AIA data for selected timerange, from dataframe containing maps '''
    tstart=dt.strptime(timerange[0],"%Y-%m-%dT%H:%M:%S")
    tend=dt.strptime(timerange[1],"%Y-%m-%dT%H:%M:%S")

    #AIA
    #dfaia.timestamps=pd.to_datetime(dfaia.timestamps)
    tidx=dfaia.query("timestamps >= @tstart and timestamps <= @tend") #dataframe
    if len(tidx.index) == len(wavs): #more than 1 timestamp
        print('only one timestamp')
        tidx.sort_values('wavelength',inplace=True)
        aiamaps=list(tidx.maps)
    else:
        gdf=tidx.groupby('wavelength')
        aiamaps=[]
        for w in wavs:
            gg=gdf.get_group(w).maps
            try:
                meanmap=sunpy.map.Map(np.nanmean([g.data for g in gg],axis=0),gg.iloc[0].meta)
                print('all maps same dimensions')
            except ValueError: #dimension mismatch
                print('warning: dimension missmatch, correcting...')
                smallest_x=np.min([g.data.shape[0] for g in gg])
                smallest_y=np.min([g.data.shape[1] for g in gg])
                trimmed_mapdata=[]
                for g in gg:
                    if g.data.shape[0] != smallest_x:
                        dx=g.data.shape[0]-smallest_x
                        mdata=g.data[dx:,:]
                    else:
                        mdata=g.data
                    if mdata.shape[1] != smallest_y:
                        dy=g.data.shape[1]-smallest_y
                        mmdata=mdata[:,dy:]
                    else:
                        mmdata=mdata
                    trimmed_mapdata.append(mmdata)
                #print(w,np.nanmean(trimmed_mapdata,axis=0).shape,gg.iloc[0].meta['date-obs'])
                meanmap=sunpy.map.Map(how(trimmed_mapdata,axis=0),gg.iloc[0].meta)
            #print(type(meanmap))
            aiamaps.append(meanmap)
    return aiamaps

