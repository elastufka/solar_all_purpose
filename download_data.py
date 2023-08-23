 #######################################
# query_vso.py
# Erica Lastufka 13/11/2017  

#Description: Get maps from VSO
#######################################

import numpy as np
import scipy.constants as sc
from datetime import datetime as dt
from datetime import timedelta as td
import os
from sunpy.net import Fido, attrs as a
import sunpy.map
import astropy.units as u
import pandas as pd
import urllib.request

def download_aia(time_start,time_end,bottom_left_coord=None, top_right_coord=None,jsoc_email=None,wave=171,sample=None,folder_store='.',tracking=False):
    """Download AIA full-disk or cutout via FIDO.

    Args:
        time_start (str, datetime, int, float): Start time in format readable by astropy.Time.
        time_end (str, datetime, int, float): End time in format readable by astropy.Time.
        bottom_left_coord (SkyCoord, optional): Bottom left coordinate of desired cutout. Defaults to None.
        top_right_coord (SkyCoord, optional): Top right coordinate of desired cutout. Defaults to None.
        jsoc_email (str, optional): Email address of registered JSOC user. Defaults to None.
        wave (int, optional): Wavelength in Angstroms. Defaults to 171.
        sample (timedelta, optional): Frequency with which to sample the data. Defaults to None.
        folder_store (str, optional): Path to store the downloaded data. Defaults to '.'
        tracking (bool, optional): Whether or not to track the cutout region as it moves across the solar disk. Defaults to False.

    Returns:
        Fido query result
    """
    wlen = a.Wavelength(wave * u.angstrom) #(wave-.5)* u.angstrom, (wave+.5)* u.angstrom)
    qs=[a.Time(time_start,time_end), wlen]

    if not bottom_left_coord: #not a cutout
        instr= a.Instrument('AIA')
        qs.append(instr)
    else:
        cutout = a.jsoc.Cutout(bottom_left_coord,top_right=top_right_coord,tracking=tracking)
        qs.append(cutout)
        qs.append(a.jsoc.Segment.image)

        if wave in [1600,1700]:
            series=a.jsoc.Series.aia_lev1_uv_24s
        else:
            series=a.jsoc.Series.aia_lev1_euv_12s
        qs.append(series)
        
        if not jsoc_email:
            jsoc_email = os.environ["JSOC_EMAIL"]
        qs.append(a.jsoc.Notify(jsoc_email))

    #if wlen != 'DEM': #gets all euv
    #    qs.append(a.Wavelength(wlen*u.angstrom))

    if sample: #must have units
        qs.append(a.Sample(sample))

    q = Fido.search(*qs)

    files = Fido.fetch(q,path=folder_store)
    return files

#def query_fido(time_int, wave, series='aia_lev1_euv_12s',instrument=False, cutout_coords=False, jsoc_email='erica.lastufka@fhnw.ch',track_region=True, sample=False, source=False, path=False,single_result=True):
#    '''query VSO database for data and download it. Cutout coords will be converted to SkyCoords if not already in those units. For cutout need sunpy 3.0+
#    Series are named here: http://jsoc.stanford.edu/JsocSeries_DataProducts_map.html but with . replaced with _'''
#    if type(time_int[0]) == str:
#        time_int[0]=dt.strptime(time_int[0],'%Y-%m-%dT%H:%M:%S')
#        time_int[1]=dt.strptime(time_int[1],'%Y-%m-%dT%H:%M:%S')
#
#    wave=a.Wavelength(wave*u.angstrom)#(wave-.1)* u.angstrom, (wave+.1)* u.angstrom)
#    #instr= sn.attrs.Instrument(instrument)
#    time = a.Time(time_int[0],time_int[1])
#    qs=[time,wave] #(series & wave)]
#    if series:
#        series=getattr(a.jsoc.Series,series) #is this only needed when using jsoc however?
#        qs.append(series)
#    elif instrument:
#        qs.append(a.Instrument(instrument))
#
#    if cutout_coords != False:
#        if type(cutout_coords[0]) == SkyCoord and type(cutout_coords[1]) == SkyCoord:
#            bottom_left_coord,top_right_coord=cutout_coords
#        else: #convert to skycoord, assume earth-observer at start time. If non-earth observer used, must pass cutout_coords as skycoords in desired frame
#            t0=time_int[0]
#            bottom_left_coord=SkyCoord(cutout_coords[0][0]*u.arcsec, cutout_coords[0][1]*u.arcsec,obstime=t0,frame=sunpy.coordinates.frames.Helioprojective)
#            top_right_coord=SkyCoord(cutout_coords[1][0]*u.arcsec, cutout_coords[1][0]*u.arcsec, obstime=t0,frame=sunpy.coordinates.frames.Helioprojective)
#
#
#        cutout = a.jsoc.Cutout(bottom_left_coord,top_right=top_right_coord,tracking=track_region)
#
#        qs.append(cutout)
#        qs.append(a.jsoc.Notify(jsoc_email))
#        #qs.append(vso.attrs.jsoc.Segment.image)
#        #qs.append(vso.atrs..jsoc.Series.aia_lev1_euv_12s) #this is essential now...
#
#    if source:
#        source=a.Source(source)
#        qs.append(source)
#    if sample: #Quantity
#        sample = a.Sample(sample)
#        qs.append(sample)
#
#    res = Fido.search(*qs)
#    if single_result:
#        res=res[0]
#    #print(qs, path, res)
#    if not path: files = Fido.fetch(res,path='./')
#    else: files = Fido.fetch(res,path=f"{path}/")
#    return res #prints nicely in notebook at any rate

#def query_vso(time_int, instrument, wave, source=False, path=False,save=True):
#        '''method to query VSO database for data and download it. '''
#        #vc = vso.VSOClient()
#        maps=0
#
#        #provider = a.Provider(provider)
#
#        #if type(time_int[0]) == dt:
#        #    time_int[0]=dt.strftime(time_int[0],'%Y-%m-%dT%H:%M:%S')
#        #    time_int[1]=dt.strftime(time_int[1],'%Y-%m-%dT%H:%M:%S')
#
#        if source:
#             source=a.vso.Source(source)
#        while instrument:
#            try:
#                instr= a.Instrument(instrument)
#                break
#            except ValueError:
#                instrument=raw_input('Not a valid instrument! Try again:' )
#
#        #sample = vso.attrs.Sample(24 * u.hour)
#        wl=a.Wavelength((wave-.5)* u.angstrom, (wave+.5)* u.angstrom)
#
#        time = a.Time(time_int[0],time_int[1])
#
#        if source:
#            res=Fido.search(source, wl, time, instr)
#        else:
#            res=Fido.search(time, wl,instr)
#
#        print res
#        #if len(res) != 1:
#        if not path: files = Fido.fetch(res,path='./{file}').wait()
#        else: files = Fido.fetch(res,path=path+'{file}').wait()
#
#        f=sunpy.map.Map(files[0])
#        maps.append({f.instrument: f.submap(SkyCoord((-1100, 1100) * u.arcsec, (-1100, 1100) * u.arcsec,frame=f.coordinate_frame))}) #this field too small for coronographs
#
#        if save: #pickle it?
#            os.chdir(path)
#            newfname=files[0][files[0].rfind('/')+1:files[0].rfind('.')]+'.p'
#            pickle.dump(maps,open(newfname,'wb'))
#        #else:
#        #    print 'no results found! is the server up?'
#        return maps

def query_hek(time_int,event_type='FL',obs_instrument='AIA',small_df=True,single_result=False):
    time = a.Time(time_int[0],time_int[1])
    eventtype = a.hek.EventType(event_type)
    #obsinstrument=sn.attrs.hek.OBS.Instrument(obs_instrument)
    res = Fido.search(time,eventtype,a.hek.OBS.Instrument(obs_instrument))
    tbl = res['hek']
    names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
    df = tbl[names].to_pandas()
    if df.empty:
        return df
    if small_df:
        df = df[['hpc_x','hpc_y','hpc_bbox','frm_identifier','frm_name']]
    if single_result: #select one
        aa = df.where(df.frm_identifier == 'Feature Finding Team').dropna()
        print(aa.index.values)
        if len(aa.index.values) == 1: #yay
            return aa
        elif len(aa.index.values) > 1:
            return pd.DataFrame(aa.iloc[0]).T
        elif aa.empty: #whoops, just take the first one then
            return pd.DataFrame(df.iloc[0]).T
    df.drop_duplicates(inplace=True)
    return df

def download_xrt_from_HEC(textfile,target_dir=False):
    ''' this should absolutely not be necessary so why is it?
    textfile is list from HEC catalog query order creation
    https://www.lmsal.com/cgi-ssw/www_sot_cat.sh
    https://www.lmsal.com/hek/hcr?cmd=search-events
    https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_ObsX2020-09-12T11%3A13%3A24.000.xml
    https://www.lmsal.com/hek/hcr?cmd=view-event&event-id=ivo%3A%2F%2Fsot.lmsal.com%2FVOEvent%23VOEvent_ObsX2020-09-12T18%3A00%3A24.000.xml

    click on observation, then 'get all data', then refine search
    ... why do they have no download all button?
    '''
    with open(textfile) as f:
        lines=[line[:line.rfind('.fits')+5] for line in f.readlines() if line.startswith('http')]
    if target_dir:
        dest=target_dir
    else:
        dest=''
    print("fetching %s files" % len(lines))
    for url in lines:
        #print(url,dest+'/'+url[url.rfind('/')+1:])
        urllib.request.urlretrieve(url, dest+'/'+url[url.rfind('/')+1:])
