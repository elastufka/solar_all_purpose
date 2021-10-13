import pandas as pd
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
from datetime import datetime as dt
from datetime import timedelta as td
import sunpy
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame,pixel_to_skycoord, skycoord_to_pixel
import rotate_coord as rc
from visible_from_earth import *
from rotate_maps import load_SPICE, coordinates_SOLO
#from rotate_maps_utils import rotate_hek_coords

def spacecraft_to_earth_time(date_in,load_spice=False):
    '''convert Solar Orbiter times at spacecraft to times at earth '''
    if type(date_in) != dt:
        date_in=pd.to_datetime(date_in)
    if load_spice:
        load_SPICE(date_in,"/Users/wheatley/Documents/Solar/STIX/solar-orbiter/kernels/mk")
    _,lighttime=coordinates_SOLO(date_in,light_time=True)
    return date_in + td(seconds=lighttime)

def update_all_rotations(df,istart=False,istop=False,verbose=True):
    '''with AIA exact frame instead of earth observer
    this changes basically nothing.... '''
    start_time=dt.now()
    #load spice kernels
    #print(dt.strftime(df.Datetime.iloc[-1],'%Y-%m-%dT%H:%M:%S'))
    load_SPICE(dt.strftime(df.Datetime.iloc[-1],'%Y-%m-%dT%H:%M:%S'), "/Users/wheatley/Documents/Solar/STIX/solar-orbiter/kernels/mk")
    
    hpc_x_rotated,hpc_y_rotated,hpc_lon_rotated,hpc_lat_rotated=[],[],[],[]
    Bproj_x_rotated,Bproj_y_rotated,Bproj_lon_rotated,Bproj_lat_rotated=[],[],[],[]
    CFL_x_rotated,CFL_y_rotated,CFL_lon_rotated,CFL_lat_rotated=[],[],[],[]
    for i, row in df.iterrows():
        if type(istart) == int and i < istart:
            continue
        d=row.Datetime
        Eobs,ewcs=get_AIA_observer(d,wcs=True)
        scE=SkyCoord(row.hpc_x,row.hpc_y,unit=u.arcsec,frame='helioprojective',obstime=d)
        Sobs=get_SO_observer(d)
        so_wcs=get_SO_wcs(d)
        
        #re-rotate hpc to AIA pov. In fact, re-rotate everything...
        aiac=try_rotation(rc.rotate_coord(row.hpc_x,row.hpc_y,Eobs,ewcs,obs_out=Sobs,wcs_out=so_wcs))
        bprojc=try_rotation(rc.rotate_coord(row.Bproj_x,row.Bproj_y,Sobs,so_wcs,obs_out=Eobs,wcs_out=ewcs))
        cflc=try_rotation(rc.rotate_coord(row['CFL_LOC_X(arcsec)'],row['CFL_LOC_Y (arcsec)'],Sobs,so_wcs,obs_out=Eobs,wcs_out=ewcs))
        
        hpc_x_rotated.append(aiac.rotated_x_arcsec)
        hpc_y_rotated.append(aiac.rotated_y_arcsec)
        hpc_lon_rotated.append(aiac.rotated_lon_deg)
        hpc_lat_rotated.append(aiac.rotated_lat_deg)
        
        Bproj_x_rotated.append(bprojc.rotated_x_arcsec)
        Bproj_y_rotated.append(bprojc.rotated_y_arcsec)
        Bproj_lon_rotated.append(bprojc.rotated_lon_deg)
        Bproj_lat_rotated.append(bprojc.rotated_lat_deg)
        
        CFL_x_rotated.append(cflc.rotated_x_arcsec)
        CFL_y_rotated.append(cflc.rotated_y_arcsec)
        CFL_lon_rotated.append(cflc.rotated_lon_deg)
        CFL_lat_rotated.append(cflc.rotated_lat_deg)
        
        #scS=SkyCoord(aiac.hpc_x_rotated,aiac.hpc_y_rotated,unit=u.arcsec,frame='helioprojective',obstime=d)
        #slon,slat=hpc_to_hpr(scS,SkyCoord(0*u.arcsec, 0*u.arcsec,obstime=Sobs.obstime,observer=Sobs,frame='helioprojective'))
        #hpc_lon_rotated.append(slon.value)
        #hpc_lat_rotated.append(slat.value)
        if verbose:
            if i%10==0:
                print(f"{dt.now()-start_time} for {i} queries")
        if type(istop)== int and i==istop:
            break
    #assign back to dataframe
    #for k,v in locals().items():
    #    if 'rotated' in k:
    #        df[k]=v
    
    df['hpc_x_rotated']=hpc_x_rotated
    df['hpc_y_rotated']=hpc_y_rotated
    df['hpc_lon_rotated']=hpc_lon_rotated
    df['hpc_lat_rotated']=hpc_lat_rotated
    
    df['Bproj_x_rotated']=Bproj_x_rotated
    df['Bproj_y_rotated']=Bproj_y_rotated
    df['Bproj_lon_rotated']=Bproj_lon_rotated
    df['Bproj_lat_rotated']=Bproj_lat_rotated
    
    df['CFL_x_rotated']=CFL_x_rotated
    df['CFL_y_rotated']=CFL_y_rotated
    df['CFL_lon_rotated']=CFL_lon_rotated
    df['CFL_lat_rotated']=CFL_lat_rotated
    
    return df
    
def try_rotation(rcoord):
    '''if you want to get past the error thrown by un-rotateable coords '''
    try:
        rcoord.do_rotation()
    except ValueError:
        rcoord.rotated_x_arcsec=None
        rcoord.rotated_y_arcsec=None
        rcoord.rotated_lon_deg=None
        rcoord.rotated_lat_deg=None
    return rcoord
        
def calc_offsets(testdf):
    testdf['STIX-AIA_timedelta']=testdf.Datetime-pd.to_datetime(testdf.event_peaktime)
    testdf['STIX-AIA_timedelta_s']=[t.total_seconds() for t in testdf['STIX-AIA_timedelta']]
#    testdf['Bproj_AIA_lon_diff']=testdf.Bproj_lon-testdf.hpc_lon_rotated
#    testdf['Bproj_AIA_lat_diff']=testdf.Bproj_lat-testdf.hpc_lat_rotated
#    testdf['Bproj_AIA_x_diff']=testdf.Bproj_x-testdf.hpc_x_rotated
#    testdf['Bproj_AIA_y_diff']=testdf.Bproj_y-testdf.hpc_y_rotated
    testdf['Bproj_AIA_lon_diff']=testdf.Bproj_lon-testdf.hpc_rotated_lon_deg
    testdf['Bproj_AIA_lat_diff']=testdf.Bproj_lat-testdf.hpc_rotated_lat_deg
    testdf['Bproj_AIA_x_diff']=testdf.Bproj_x-testdf.hpc_rotated_x_arcsec
    testdf['Bproj_AIA_y_diff']=testdf.Bproj_y-testdf.hpc_rotated_y_arcsec
    testdf['Bproj_CFL_lon_diff']=testdf.Bproj_lon-testdf.CFL_lon
    testdf['Bproj_CFL_lat_diff']=testdf.Bproj_lat-testdf.CFL_lat
    testdf['Bproj_CFL_x_diff']=testdf.Bproj_x-testdf['CFL_LOC_X(arcsec)']
    testdf['Bproj_CFL_y_diff']=testdf.Bproj_y-testdf['CFL_LOC_Y (arcsec)']
#    testdf['CFL_AIA_lon_diff']=testdf.CFL_lon-testdf.hpc_lon_rotated
#    testdf['CFL_AIA_lat_diff']=testdf.CFL_lat-testdf.hpc_lat_rotated
#    testdf['CFL_AIA_x_diff']=testdf['CFL_LOC_X(arcsec)']-testdf.hpc_x_rotated
#    testdf['CFL_AIA_y_diff']=testdf['CFL_LOC_Y (arcsec)']-testdf.hpc_y_rotated
    testdf['CFL_AIA_lon_diff']=testdf.CFL_lon-testdf.hpc_rotated_lon_deg
    testdf['CFL_AIA_lat_diff']=testdf.CFL_lat-testdf.hpc_rotated_lat_deg
    testdf['CFL_AIA_x_diff']=testdf['CFL_LOC_X(arcsec)']-testdf.hpc_rotated_x_arcsec
    testdf['CFL_AIA_y_diff']=testdf['CFL_LOC_Y (arcsec)']-testdf.hpc_rotated_y_arcsec
    return testdf
    
def rotate_pairs(df,key_x='CFL_LOC_X(arcsec)',key_y='CFL_LOC_Y (arcsec)',obs_in='SO',obs_out='AIA'):
    '''do rotations for event coordinates'''
    res=[]
    edict={'x_in':np.nan, 'y_in':np.nan,'x_arcsec':np.nan, 'y_arcsec':np.nan,
        'x_deg':np.nan, 'y_deg':np.nan, 'rsun_apparent':np.nan, 'rotated_x_arcsec':np.nan,
        'rotated_y_arcsec':np.nan, 'rotated_lon_deg':np.nan, 'rotated_lat_deg':np.nan}
    emptydf=pd.DataFrame(edict,index=pd.Index([0]))
    for i,r in df.iterrows():
        d=r.Datetime
        if not pd.isnull(r[key_x]):
            observer_in,wcs_in=get_observer(d,obs=obs_in) #eventually add try/except for loading SPICE kernel if needed
            observer_out,wcs_out=get_observer(d,obs=obs_out)

            pair_rot=rotate_coord(r[key_x],r[key_y],observer_in,wcs_in,obs_out=observer_out,wcs_out=wcs_out)
            pair_rot=try_rotation(pair_rot)

            pdf=pair_rot.to_dataframe()
            pdf['Datetime']=d
            res.append(pdf)
        else:
            emptydf['Datetime']=d
            res.append(emptydf)
        #if i%100==0:
        #    print(i)#break
    hdf=pd.concat(res,ignore_index=True)
    try:
        hdf.drop(columns=['obs_in','wcs_in','obs_out','wcs_out','unit_in','binning','scaling','skycoord','x_px','y_px','rotated_x_px','rotated_y_px','can_rotate'],inplace=True)
    except KeyError:
        pass
    ckey=key_x[:key_x.find('_')+1]
    coldict={}
    for k in hdf.keys():
        if k != 'Datetime':
            coldict[k]=ckey+k
    hdf.rename(columns=coldict,inplace=True)
    #merge back into the original dataframe
    df_out=df.merge(hdf,on='Datetime',how='inner')
    return df_out
    
def get_all_rotations(df):
    '''Query HEK for AIA events, rotate all coordinates. For use with the joined CFL and BackProjection event list. '''
    hek_res=[]
    for i,r in mdf.iterrows():
        d=r.Datetime
        #print(i,d)
        earth_observer=get_Earth_observer(d)
        earth_wcs=get_Earth_wcs(d)
        so_observer=get_SO_observer(d)
        so_wcs=get_SO_wcs(d)
        #print(solar_angular_radius(so_observer))
        qdf=query_hek([d,d+td(minutes=1)]) #if this is empty... what to do?
        qdf['Datetime']=[d for i,_ in qdf.iterrows()]
        if qdf.empty:
            qdf['Datetime']=[d]
            qdf['hpc_x']=np.nan
            qdf['hpc_y']=np.nan
        qdf=rotate_pairs(qdf,key_x='hpc_x',key_y='hpc_y',obs_in='AIA',obs_out='SO')#rotate_hek_coords(qdf,earth_observer,earth_wcs,so_observer,so_wcs)

        
        qdf['CFL_vis_from_Earth']=[is_visible_from_earth(d,(mdf['CFL_LOC_X(arcsec)'][i],mdf['CFL_LOC_Y (arcsec)'][i])) for i,row in qdf.iterrows()]
        qdf['Bproj_vis_from_Earth']=[is_visible_from_earth(d,(mdf['Bproj_x'][i],mdf['Bproj_y'][i])) for i,row in qdf.iterrows()]
        qdf['AIA_vis_from_SO']=[is_visible_from_SO(d,(row['hpc_x'],row['hpc_y'])) for i,row in qdf.iterrows()]

        #rotate bproj and CFL locations as well
        bproj_rot=rotate_coord(r.Bproj_x,r.Bproj_y,so_observer,so_wcs,obs_out=earth_observer,wcs_out=earth_wcs)
        bproj_rot=try_rotation(bproj_rot)
            
        CFL_rot=rotate_coord(r['CFL_LOC_X(arcsec)'],r['CFL_LOC_Y (arcsec)'],so_observer,so_wcs,obs_out=earth_observer,wcs_out=earth_wcs)
        CFL_rot=try_rotation(CFL_rot)
            
        qdf['Bproj_lon']=[bproj_rot.x_deg for i,_ in qdf.iterrows()]
        qdf['Bproj_lat']=[bproj_rot.y_deg for i,_ in qdf.iterrows()]
        qdf['CFL_lon']=[CFL_rot.x_deg for i,_ in qdf.iterrows()]
        qdf['CFL_lat']=[CFL_rot.y_deg for i,_ in qdf.iterrows()]
        qdf['Bproj_rotated_x_arcsec']=[bproj_rot.rotated_x_arcsec for i,_ in qdf.iterrows()]
        qdf['Bproj_rotated_y_arcsec']=[bproj_rot.rotated_y_arcsec for i,_ in qdf.iterrows()]
        qdf['Bproj_rotated_lon_deg']=[bproj_rot.rotated_lon_deg for i,_ in qdf.iterrows()]
        qdf['Bproj_rotated_lat_deg']=[bproj_rot.rotated_lat_deg for i,_ in qdf.iterrows()]
        qdf['CFL_rotated_x_arcsec']=[CFL_rot.rotated_x_arcsec for i,_ in qdf.iterrows()]
        qdf['CFL_rotated_y_arcsec']=[CFL_rot.rotated_y_arcsec for i,_ in qdf.iterrows()]
        qdf['CFL_rotated_lon_deg']=[CFL_rot.rotated_lon_deg for i,_ in qdf.iterrows()]
        qdf['CFL_rotated_lat_deg']=[CFL_rot.rotated_lat_deg for i,_ in qdf.iterrows()]
        hek_res.append(qdf)
        if i%100==0:
            print(i)#break
    hdf=pd.concat(hek_res,ignore_index=True)
    return hdf
    
def query_hek(time_int,event_type='FL',obs_instrument='AIA',small_df=True,single_result=False):
    time = sn.attrs.Time(time_int[0],time_int[1])
    eventtype=sn.attrs.hek.EventType(event_type)
    #obsinstrument=sn.attrs.hek.OBS.Instrument(obs_instrument)
    res=sn.Fido.search(time,eventtype,sn.attrs.hek.OBS.Instrument==obs_instrument)
    tbl=res['hek']
    names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
    df=tbl[names].to_pandas()
    if df.empty:
        return df
    if small_df:
        df=df[['hpc_x','hpc_y','hpc_bbox','frm_identifier','frm_name','fl_goescls','fl_peaktempunit','fl_peakemunit','fl_peakflux','event_peaktime','fl_peakfluxunit','fl_peakem','fl_peaktemp','obs_dataprepurl','gs_imageurl','gs_thumburl']]
        df.drop_duplicates(inplace=True)
    if single_result: #select one
        aa=df.where(df.frm_identifier == 'Feature Finding Team').dropna()
        print(aa.index.values)
        if len(aa.index.values) == 1: #yay
            return aa
        elif len(aa.index.values) > 1:
            return pd.DataFrame(aa.iloc[0]).T
        elif aa.empty: #whoops, just take the first one then
            return pd.DataFrame(df.iloc[0]).T

    return df
    
