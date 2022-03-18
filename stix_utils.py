import pandas as pd
import numpy as np
import os
import glob

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
from sunpy.map.maputils import solar_angular_radius
import spiceypy
import warnings
from spiceypy.utils.exceptions import NotFoundError
#from rotate_maps_utils import rotate_hek_coords

def spacecraft_to_earth_time(date_in,load_spice=False,solo_r=False):
    '''convert Solar Orbiter times at spacecraft to times at earth '''
    if type(date_in) != dt:
        date_in=pd.to_datetime(date_in)
    if load_spice:
        load_SPICE(date_in,os.environ['SPICE'])
    solo_hee,lighttime=coordinates_SOLO(date_in,light_time=True)
    if solo_r: #also return this...
        stix_r=np.sqrt(solo_hee.x.to(u.AU).value**2+solo_hee.y.to(u.AU).value**2+solo_hee.z.to(u.AU).value**2)
        return date_in + td(seconds=lighttime), stix_r
    return date_in + td(seconds=lighttime)
    
def get_rsun_apparent(date_in,observer=False, spacecraft='SO',sc=True):
    if isinstance(observer,bool) and observer == False:
        obs = get_observer(date_in,obs=spacecraft,wcs=False,sc=sc)
    else:
        obs=observer
    return solar_angular_radius(obs)
    
def load_SOLO_SPICE(obs_date, path_kernel=os.environ['SPICE']):
    """
    Load the SPICE kernel that will be used to get the
    coordinates of the different spacecrafts.
    """
    #get cwd
    cwd=os.getcwd()

    # Convert string format to datetime
    if type(obs_date) ==str:
        obs_date = dt.strptime(obs_date, '%Y-%m-%dT%H:%M:%S')

    # Check if path_kernel has folder format
    if path_kernel[-1] != '/':
        path_kernel = path_kernel+'/'

    # Find the MK generation date ...
    MK_date_str = glob.glob(path_kernel+'/solo_*flown-mk_v*.tm')
    # ... and convert it to datetime
    MK_date = dt.strptime(MK_date_str[0][-15:-7], '%Y%m%d')

    # Check which kernel has to be loaded: 'flown' or 'pred'
    if obs_date < MK_date:
        spice_kernel = 'solo_ANC_soc-flown-mk.tm'
    else:
        spice_kernel = 'solo_ANC_soc-pred-mk.tm'
        print()
        print('**********************************************')
        print('The location of Solar Orbiter is a prediction!')
        print('Did you download the most recent SPICE kernel?')
        print('**********************************************')
        print()


    # For STEREO
    #stereo_kernel = spicedata.get_kernel('stereo_a')
    #hespice.furnish(stereo_kernel)

    # Change the CWD to the given path. Necessary to load correctly all kernels
    os.chdir(path_kernel)

    # Load one (or more) SPICE kernel into the program
    spiceypy.spiceypy.furnsh(spice_kernel)

    print()
    print('SPICE kernels loaded correctly')
    print()

    #change back to original working directory
    os.chdir(cwd)

def locations_over_time(start_date,end_date,fn=coordinates_SOLO,body=None,output_unit=u.AU):
    '''return HEE locations of SOLO over given time period'''
    date_range=pd.date_range(start_date,end_date)
    hee=[]
    for d in date_range:
        if body:
            hee.append(coordinates_body(d,body))
        else:
            hee.append(fn(d))
        
    xkm,ykm,zkm=[],[],[]
    #print(f"Units: HEE_x {hee[0].x.unit}, HEE_y {hee[0].y.unit}, HEE_z {hee[0].z.unit}")
    for h in hee:
        xkm.append(h.x.to(output_unit).value)
        ykm.append(h.y.to(output_unit).value)
        zkm.append(h.z.to(output_unit).value)
    return date_range,xkm,ykm,zkm
    
def parse_sunspice_name_py(spacecraft):
    '''python version of sswidl parse_sunspice_name.pro '''
    if type(spacecraft) == str:
        sc = spacecraft.upper()
        n = len(sc)

        #If the string is recognized as one of the STEREO spacecraft, then return the appropriate ID value.
        if 'AHEAD' in sc and 'STEREO' in sc or sc == 'STA':
            return '-234'
        
        if 'BEHIND' in sc and 'STEREO' in sc or sc == 'STB':
            return '-235'

        #If SOHO, then return -21.
        if sc=='SOHO':
            return '-21'

        #If Solar Orbiter then return -144.
        if 'SOLAR' in sc and 'ORBITER' in sc or sc == 'SOLO' or sc == 'ORBITER':
            return '-144'

        #If Solar Probe Plus then return -96.
        if 'PARKER' in sc and 'SOLAR' in sc and 'PROBE' in sc:
            return '-96'
        if 'PROBE' in sc and 'PLUS' in sc:
            return '-96'
        if sc == 'PSP' or sc == 'SPP':
            return '-96'

        #If BepiColombo MPO then return -121.
        if 'BEPICOLOMBO' in sc and 'MPO' in sc:
            return '-121'
        if 'BC' in sc and 'MPO' in sc:
            return '-121'
        #Otherwise, simply return the (trimmed and uppercase) original name.
        return sc
    else:
        raise TypeError("Input spacecraft name must be string")
    
def get_sunspice_roll_py(datestr, spacecraft, system='SOLO_SUN_RTN',degrees=True, radians=False,tolerance=100,kernel_path=os.environ['SPICE']):
    '''Python version of (simpler) sswidl get_sunspice_roll.pro . Assumes spice kernel already furnished'''
    units = float(180)/np.pi
    if radians: units = 1
  
    #Determine which spacecraft was requested, and translate it into the proper input for SPICE.

    inst = 0
    sc_ahead  = '-234'
    sc_behind = '-235'
    sc_psp    = '-96'
    sc = parse_sunspice_name_py(spacecraft)
    if sc == sc_ahead or sc == sc_behind:
        sc_stereo=True
    if sc == '-144':
        sc_frame = -144000

    #Start by deriving the C-matrices.  Make sure that DATE is treated as a vector.

    if system != 'RTN' and type(system) == str:
        system=system.upper()
      
    #don't know why it wants to live in the kernel directory in order to check errons but it does
    pwd=os.getcwd()
    os.chdir(kernel_path)
    et=spiceypy.spiceypy.str2et(datestr)
    sclkdp=spiceypy.spiceypy.sce2c(int(sc), et) #Ephemeris time, seconds past J2000.
    #print(sc,sclkdp, tolerance, system)
    try:
        (cmat,clkout)=spiceypy.spiceypy.ckgp(sc_frame, sclkdp, tolerance, system)
    except NotFoundError:
        warnings.warn("Spice returns not found for function: ckgp, returning roll angle of 0")
        os.chdir(pwd)
        return 0.0

    os.chdir(pwd)
    
    twopi  = 2.*np.pi
    halfpi = np.pi/2.
  
    #sci_frame = (system.upper()== 'SCI') and sc_stereo

    #cspice_m2eul, cmat[*,*,i], 1, 2, 3, rroll, ppitch, yyaw
    rroll,ppitch,yyaw=spiceypy.spiceypy.m2eul(cmat, 1, 2, 3)
    ppitch = -ppitch
    if (sc == sc_ahead) or (sc == sc_behind): rroll = rroll - halfpi
    if sc == sc_behind: rroll = rroll + np.pi
    if abs(rroll) > np.pi: rroll = rroll - sign(twopi, rroll)

    #Correct any cases where the pitch is greater than +/- 90 degrees
    if abs(ppitch) > halfpi:
        ppitch = sign(np.pi,ppitch) - ppitch
        yyaw = yyaw - sign(np.pi, yyaw)
        rroll = rroll - sign(np.pi, rroll)
      
    #Apply the units.
    roll  = units * rroll
    pitch = units * ppitch
    yaw  = units * yyaw

    #Reformat the output arrays to match the input date/time array.

    #if n > 1:
    #    sz = size(date)
    #    dim = [sz[1:sz[0]]]
    #roll  = reform(roll,  dim, /overwrite)
    #pitch = reform(pitch, dim, /overwrite)
    #yaw   = reform(yaw,   dim, /overwrite)
    return roll
    
def rescale_SO_coords(df,key_x='Bproj_x',key_y='Bproj_y',rsun_apparent='rsun_SO'):
    keyname=key_x[:key_x.find('_')]
    lonlat=[hpc_scale_inputs(df[key_x][i],df[key_y][i],df[rsun_apparent][i]) for i in df.index]
    df[keyname+'_lon']=np.array(lonlat)[:,0]
    df[keyname+'_lat']=np.array(lonlat)[:,1]
    return df

def update_all_rotations(df,istart=False,istop=False,verbose=True, AIA=True, Bproj=True,CFL=True):
    '''with AIA exact frame instead of earth observer
    this changes basically nothing.... '''
    start_time=dt.now()
    #load spice kernels
    #print(dt.strftime(df.Datetime.iloc[-1],'%Y-%m-%dT%H:%M:%S'))
    load_SPICE(dt.strftime(df.Datetime.iloc[-1],'%Y-%m-%dT%H:%M:%S'), os.environ['SPICE'])
    if AIA:
        hpc_x_rotated,hpc_y_rotated,hpc_lon_rotated,hpc_lat_rotated=[],[],[],[]
    if Bproj:
        Bproj_x_rotated,Bproj_y_rotated,Bproj_lon_rotated,Bproj_lat_rotated=[],[],[],[]
    if CFL:
        CFL_x_rotated,CFL_y_rotated,CFL_lon_rotated,CFL_lat_rotated=[],[],[],[]
    for i, row in df.iterrows():
        if type(istart) == int and i < istart:
            continue
        d=row.Datetime
        observers=None
#        Eobs,ewcs=get_AIA_observer(d,wcs=True)
#        #scE=SkyCoord(row.hpc_x,row.hpc_y,unit=u.arcsec,frame='helioprojective',obstime=d)
#        Sobs=get_SO_observer(d)
#        so_wcs=get_SO_wcs(d)
        
        #re-rotate hpc to AIA pov. In fact, re-rotate everything...
        if AIA:
            if np.isnan(row.hpc_x) and np.isnan(row.hpc_y):
                hpc_x_rotated.append(np.nan)
                hpc_y_rotated.append(np.nan)
                hpc_lon_rotated.append(np.nan)
                hpc_lat_rotated.append(np.nan)
            else:
                if observers==None:
                    observers=both_observers(d)
                else:
                    Eobs, ewcs,Sobs,so_wcs=observers
                aiac=try_rotation(rc.rotate_coord(row.hpc_x,row.hpc_y,Eobs,ewcs,obs_out=Sobs,wcs_out=so_wcs))
                hpc_x_rotated.append(aiac.rotated_x_arcsec)
                hpc_y_rotated.append(aiac.rotated_y_arcsec)
                hpc_lon_rotated.append(aiac.rotated_lon_deg)
                hpc_lat_rotated.append(aiac.rotated_lat_deg)
        if Bproj:
            if np.isnan(row.Bproj_x) and np.isnan(row.Bproj_y):
                Bproj_x_rotated.append(np.nan)
                Bproj_y_rotated.append(np.nan)
                Bproj_lon_rotated.append(np.nan)
                Bproj_lat_rotated.append(np.nan)
            else:
                if observers==None:
                    observers=both_observers(d)
                else:
                    Eobs, ewcs,Sobs,so_wcs=observers
                bprojc=try_rotation(rc.rotate_coord(row.Bproj_x,row.Bproj_y,Sobs,so_wcs,obs_out=Eobs,wcs_out=ewcs))
                Bproj_x_rotated.append(bprojc.rotated_x_arcsec)
                Bproj_y_rotated.append(bprojc.rotated_y_arcsec)
                Bproj_lon_rotated.append(bprojc.rotated_lon_deg)
                Bproj_lat_rotated.append(bprojc.rotated_lat_deg)
        if CFL:
            if np.isnan(row['CFL_LOC_X(arcsec)']) and np.isnan(row['CFL_LOC_Y (arcsec)']):
                CFL_x_rotated.append(np.nan)
                CFL_y_rotated.append(np.nan)
                CFL_lon_rotated.append(np.nan)
                CFL_lat_rotated.append(np.nan)
            else:
                if observers==None:
                    observers=both_observers(d)
                else:
                    Eobs, ewcs,Sobs,so_wcs=observers
                cflc=try_rotation(rc.rotate_coord(row['CFL_LOC_X(arcsec)'],row['CFL_LOC_Y (arcsec)'],Sobs,so_wcs,obs_out=Eobs,wcs_out=ewcs))
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
    
    if AIA:
        df['hpc_x_rotated']=hpc_x_rotated
        df['hpc_y_rotated']=hpc_y_rotated
        df['hpc_lon_rotated']=hpc_lon_rotated
        df['hpc_lat_rotated']=hpc_lat_rotated
    if Bproj:
        df['Bproj_x_rotated']=Bproj_x_rotated
        df['Bproj_y_rotated']=Bproj_y_rotated
        df['Bproj_lon_rotated']=Bproj_lon_rotated
        df['Bproj_lat_rotated']=Bproj_lat_rotated
    if CFL:
        df['CFL_x_rotated']=CFL_x_rotated
        df['CFL_y_rotated']=CFL_y_rotated
        df['CFL_lon_rotated']=CFL_lon_rotated
        df['CFL_lat_rotated']=CFL_lat_rotated
    
    return df
    
def both_observers(d):
    '''AIA and SO observers for a given datetime '''
    d=row.Datetime
    Eobs,ewcs=get_AIA_observer(d,wcs=True)
    Sobs=get_SO_observer(d)
    so_wcs=get_SO_wcs(d)
    return [Eobs, ewcs,Sobs,so_wcs]

    
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
            if type(obs_in) == 'str':
                observer_in,wcs_in=get_observer(d,obs=obs_in) #eventually add try/except for loading SPICE kernel if needed
            else:
                observer_in,wcs_in=obs_in
            if type(obs_out)==str:
                observer_out,wcs_out=get_observer(d,obs=obs_out)
            else:
                observer_out,wcs_out=obs_out

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
        #earth_observer=get_Earth_observer(d)
        #earth_wcs=get_Earth_wcs(d)
        aia_observer,aia_wcs=get_AIA_observer(d,wcs=True)
        so_observer=get_SO_observer(d)
        so_wcs=get_SO_wcs(d)
        #print(solar_angular_radius(so_observer))
        qdf=query_hek([d,d+td(minutes=1)]) #if this is empty... what to do?
        qdf['Datetime']=[d for i,_ in qdf.iterrows()]
        if qdf.empty:
            qdf['Datetime']=[d]
            qdf['hpc_x']=np.nan
            qdf['hpc_y']=np.nan
        qdf=rotate_pairs(qdf,key_x='hpc_x',key_y='hpc_y',obs_in=[aia_observer,aia_wcs],obs_out=[so_observer,so_wcs])#rotate_hek_coords(qdf,earth_observer,earth_wcs,so_observer,so_wcs)

        
        qdf['CFL_vis_from_Earth']=[is_visible_from_earth(d,(mdf['CFL_LOC_X(arcsec)'][i],mdf['CFL_LOC_Y (arcsec)'][i])) for i,row in qdf.iterrows()]
        qdf['Bproj_vis_from_Earth']=[is_visible_from_earth(d,(mdf['Bproj_x'][i],mdf['Bproj_y'][i])) for i,row in qdf.iterrows()]
        qdf['AIA_vis_from_SO']=[is_visible_from_SO(d,(row['hpc_x'],row['hpc_y'])) for i,row in qdf.iterrows()]

        #rotate bproj and CFL locations as well
        bproj_rot=rotate_coord(r.Bproj_x,r.Bproj_y,so_observer,so_wcs,obs_out=aia_observer,wcs_out=aia_wcs)
        bproj_rot=try_rotation(bproj_rot)
            
        CFL_rot=rotate_coord(r['CFL_LOC_X(arcsec)'],r['CFL_LOC_Y (arcsec)'],so_observer,so_wcs,obs_out=aia_observer,wcs_out=aia_wcs)
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
    
