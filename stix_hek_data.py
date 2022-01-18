import numpy as np
import pandas as pd

from astropy import units as u
import glob
from astropy.time import Time
from stix_utils import load_SOLO_SPICE,spacecraft_to_earth_time
import hek_event_handler as hek
from stixdcpy.net import JSONRequest as jreq
#from visible_from_earth import is_visible_from_earth
#from sunpy.coordinates.frames import HeliocentricEarthEcliptic
import spiceypy
from datetime import datetime as dt
from datetime import timedelta as td
import os
import logging
from scipy.stats import linregress
from flare_physics_utils import goes_class_to_flux
from stix_goes_fit import STIX_GOES_fit

log = logging.getLogger(__name__)

def get_stix_flares(start_time,end_time,sortedby='time',lighttime_correct=True,local_copy=True):
    flares=jreq.fetch_flare_list(start_time,end_time,sortedby=sortedby) #current limit is 1000, results are returned backwards from end time
    df=pd.DataFrame(flares)
    corrected_times,solo_r=[],[]
    if lighttime_correct:
        for d in df.peak_utc:
            ctime,solor=spacecraft_to_earth_time(d,solo_r=True)
            corrected_times.append(ctime)
            solo_r.append(solor)
        df['peak_utc_corrected']=corrected_times
        df['solo_r']=solo_r
        df['peak_counts_corrected']=df.peak_counts*(df.solo_r)**2
    if local_copy:
        df.to_json(f"data/stix_flares_{dt.strftime(dt.now(),'%Y-%m-%dT%H%M')}.json")
    return df
        
def stix_hek_data(start_time=None,end_time=None,lighttime_correct=True,local_copy=False):
    if not start_time:
        start_time=dt(2020,2,10,5,0,0)
    if not end_time:
        end_time=dt.now()
    load_SOLO_SPICE(dt.now())
    stix_flare_list=get_stix_flares(start_time,end_time,sortedby='time',lighttime_correct=True)
    stix_flare_list['peak_utc']=pd.to_datetime(stix_flare_list.peak_utc)
    nflares=stix_flare_list.peak_utc.count()
    log.info(f"{nflares} STIX flares found")
    if nflares == 1000:
        log.warning(f"{nflares} STIX flares found, there might be more results in the specified date range!")
    #hek event at same time?
    hek_events=[hek.HEKEventHandler(f.peak_utc_corrected).df for _,f in stix_flare_list.iterrows()]
    log.info(f"{len(hek_events)} HEK results found")
    hdf=pd.concat(hek_events)
    hdf['event_peaktime']=pd.to_datetime(hdf.event_peaktime)
    
    #cfl_from_earth=[is_visible_from_earth(f.peak_utc, (f[],f[])) for f in stix_flare_list]
    #locations not returned by jreq yet...
    
    #if CFL location... SOLO event visible from Earth?
    
    #for i,row in df.iterrows():
    #if is_visible_from_earth(row.PEAK_UTC, (row['CFL_LOC_X(arcsec)'],row['CFL_LOC_Y (arcsec)'])) and i > 120:
    #    r=row

    df=stix_flare_list.merge(hdf,left_on='peak_utc_corrected',right_on='date_obs',how='outer')
    return df
    
def update_stix_hek_data(current_data_file='data/stix_hek_full.json'):
    '''update the csv file with the latest flares from STIX and AIA'''
    df=pd.read_json(current_data_file)
    df['peak_utc']=pd.to_datetime(df.peak_utc,unit='ms')
    last_date=df.peak_utc.iloc[0] #it's backwards...
    newdf=stix_hek_data(start_time=last_date)
    udf=pd.concat([newdf,df])
    udf.drop_duplicates(subset=[k for k in udf.keys() if k !='goes'],inplace=True)
    udf.reset_index(inplace=True,drop=True)
    udf.to_json(current_data_file)
    
def prep_and_fit_data(current_data_file='data/stix_hek_full.json',timerange=10,threshold=None):
    '''Prep the data, keep data within given time range and above lower counts threshold (if requested), then fit line to log-log data'''
    df=pd.read_json(current_data_file)
    df.drop_duplicates(subset='_id',inplace=True) #just in case
    df.reset_index(inplace=True,drop=True)
    for k in ['peak_utc','peak_utc_corrected','event_peaktime','date_obs']:
        df[k]=pd.to_datetime(df[k],unit='ms')
    df['STIX_AIA_timedelta']=df.peak_utc_corrected-df.event_peaktime
    df['STIX_AIA_timedelta_abs']=np.abs(df.peak_utc_corrected-df.event_peaktime)
    
    tplus=td(minutes=timerange)
    dfs=df.query("visible_from_SOLO==True and STIX_AIA_timedelta_abs < @tplus").dropna(how='all')
    
    dfgc=dfs.where(dfs.fl_goescls !='').dropna(how='all')#.drop_duplicates(subset='date_obs')
    dfgc['GOES_flux']=[goes_class_to_flux(c) for c in dfgc.fl_goescls]
    
    if threshold:
        dfgc=dfgc[dfgc.peak_counts_corrected > threshold]
    
    dfit=STIX_GOES_fit(dfgc)
    dfit.do_fit(dfgc)
    
    pd.DataFrame(dfit.__dict__,index=[0]).to_json(f"data/fit_result_{dt.now().date()}.json")
    
if __name__ == '__main__':
    update_stix_hek_data()
    prep_and_fit_data()

