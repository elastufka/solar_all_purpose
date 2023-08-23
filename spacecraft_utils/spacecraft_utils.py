import pandas as pd
import numpy as np
import os
import glob

from astropy import units as u, constants as const
from astropy.coordinates import SkyCoord
from datetime import datetime as dt
from datetime import timedelta as td
import sunpy
from astropy.wcs import WCS
from astropy.time import Time
#from astropy.wcs.utils import wcs_to_celestial_frame,pixel_to_skycoord, skycoord_to_pixel
from sunpy.coordinates import Helioprojective
from sunpy.coordinates.frames import HeliocentricEarthEcliptic, HeliographicStonyhurst
from sunpy.map.maputils import _verify_coordinate_helioprojective
from sunpy.map import Map, make_fitswcs_header
import drms
import spiceypy
#import warnings
#from spiceypy.utils.exceptions import NotFoundError
#from rotate_maps_utils import rotate_hek_coords
#from sunpy_map_utils import coordinate_behind_limb
import heliopy.data.spice as spicedata
import heliopy.spice as hespice

def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2)            # r
    theta = np.arctan2(z,np.sqrt(x**2+ y**2))     # theta
    phi = np.arctan2(y,x)                        # phi
    return (r, theta, phi)

def furnish_kernels(spacecraft_list=['psp','stereo_a','stereo_a_pred','bepi_pred','psp_pred'],path_kernel="/Users/wheatley/Documents/Solar/STIX/solar-orbiter/kernels/mk"):
    '''allowed names: ['lsk', 'planet_trajectories', 'planet_orientations', 'helio_frames', 'cassini', 'helios1', 'helios2', 'juno', 'stereo_a', 'stereo_b', 'soho', 'ulysses', 'psp', 'solo', 'psp_pred', 'stereo_a_pred', 'stereo_b_pred', 'juno_pred', 'bepi_pred']) '''
    k= spicedata.get_kernel(spacecraft_list[0])
    for sc in spacecraft_list[1:]:
        k+=spicedata.get_kernel(sc)
    cwd=os.getcwd()
    os.chdir(path_kernel)
    hespice.furnish(k)
    os.chdir(cwd)
    
def get_spacecraft_position(start_date,end_date,spacecraft='SPP', path_kernel="/Users/wheatley/Documents/Solar/STIX/solar-orbiter/kernels/mk",sphere=False,hgs=False):
    ''' have to be in the kernel directory while calling generate_positions... this is a pretty annoying thing about spicypy'''
    times=pd.date_range(start_date,end_date)
    cwd=os.getcwd()
    sc = hespice.Trajectory(spacecraft)
    os.chdir(path_kernel)
    sc.generate_positions(times, 'Sun', 'HEE') #is this HEE? ECLIPJ2000
    os.chdir(cwd)
    if not hgs:
        sc.change_units(u.au)
    else:
        cc=SkyCoord(sc.x,sc.y,sc.z,frame=HeliocentricEarthEcliptic,representation='cartesian',obstime=start_date)
        hgs_coord=cc.transform_to(HeliographicStonyhurst)
        return times,hgs_coord.radius.value[0],hgs_coord.lon.value[0],hgs_coord.lat.value[0] #km,deg,deg
    if sphere:
        sc_r, sc_lat, sc_lon=cart2sphere(sc.x,sc.y,sc.z)
        return times,sc_r.value,sc_lat.value,sc_lon.value
    else:
        return times,sc.x.value,sc.y.value,sc.z.value

def estimate_lighttime(hee_x,hee_y,hee_z=None):
    """Given position of spacecraft or body, estimate the light travel time between it and the Sun"""
    if not heez:
        robs=np.sqrt(hee_x**2 + hee_y**2)
    else:
        robs=np.sqrt(hee_x**2 + hee_y**2 + hee_z**2)
    return robs.to_(u.m)/const.c #astropy Quantity
    
def correct_lighttime(time_in,light_travel_time,AU=True):
    """Correct a given datetime for light travel time, assuming a distance of 1AU"""
    if AU:
        ltAU=(1*u.AU).to(u.m)/const.c
    tdelt=ltAU-light_travel_time #seconds, can also be negative
    return time_in + td(seconds=tdelt.value)

def coordinates_body(date_body,body_name,light_time=False):
    """
    Load the kernel needed in order to derive the
    coordinates of the given celestial body and then return them in
    Heliocentric Earth Ecliptic (HEE) coordinates.
    """

    # Observing time
    obstime = spiceypy.datetime2et(date_body)

    # Obtain the coordinates of Solar Orbiter
    if body_name == 'SOLO' or body_name == 'SO' or body_name == 'Solar Orbiter':
        ref_frame = 'SOLO_HEE_NASA'
    else:
        ref_frame = 'HEE'
    hee_spice, lighttimes = spiceypy.spkpos(body_name, obstime,
                                     ref_frame, #  Reference frame of the output position vector of the object
                                     'NONE', 'SUN')
    hee_spice = hee_spice * u.km

    # Convert the coordinates to HEE
    body_hee = HeliocentricEarthEcliptic(hee_spice,
                                          obstime=Time(date_body).isot,
                                          representation_type='cartesian')
    if not light_time:
        # Return the HEE coordinates of the body
        return body_hee
    else:
        return body_hee,lighttimes

def get_observer(date_in,obs='Earth',wcs=True,sc=False,wlen=1600,out_shape=(4096,4096),scale=None,rsun=False,hee=False):
    '''Get observer information. Get WCS object if requested. Return Observer as SkyCoord at (0",0") if requested.'''
    if obs in ['AIA','SDO']:
        observer,wcs_out=get_AIA_observer(date_in,wcs=True,sc=sc,wlen=wlen)
    else:
        if not hee:
            hee = coordinates_body(date_in, obs)
        observer=hee.transform_to(HeliographicStonyhurst(obstime=date_in))
        if sc:
            if rsun: #explicity set rsun
                observer = SkyCoord(0*u.arcsec, 0*u.arcsec,obstime=date_in,observer=observer,rsun=rsun,frame='helioprojective')
            else:
                observer = SkyCoord(0*u.arcsec, 0*u.arcsec,obstime=date_in,observer=observer,frame='helioprojective')

    if wcs == False:
        return observer
    else:
        try:
            return observer,out_wcs
        except UnboundLocalError:
            out_wcs=get_wcs(date_in,observer,out_shape=out_shape,scale=scale)
            return observer,out_wcs


def get_AIA_observer(date_obs,sc=False,wcs=True,wlen=1600):
    '''downloading required keywords from FITS headers. Resulting coordinates can differ from using Earth as observer by as much as 3" (on the limb) but usally will not be more than .5" otherwise.'''
    date_obs_str=dt.strftime(date_obs,"%Y.%b.%d_%H:%M:%S")
    client = drms.Client()
    kstr='CUNIT1,CUNIT2,CRVAL1,CRVAL2,CDELT1,CDELT2,CRPIX1,CRPIX2,CTYPE1,CTYPE2,HGLN_OBS,HGLT_OBS,DSUN_OBS,RSUN_OBS,DATE-OBS,RSUN_REF,CRLN_OBS,CRLT_OBS,EXPTIME,INSTRUME,WAVELNTH,WAVEUNIT,TELESCOP,LVL_NUM,CROTA2'
    if wlen in [1600,1700]:
        qstr=f'aia.lev1_uv_24s[{date_obs_str}/1m@30s]'
    else:
        qstr=f'aia.lev1_euv_12s[{date_obs_str}/1m@30s]'
    df = client.query(qstr, key=kstr)#'aia.lev1_euv_12s[2018.01.01_TAI/1d@12h] #'aia.lev1_euv_12s[2018.01.01_05:00:20/1m@30s]'
    if df.empty: #event not in database yet or other error
        observer=get_observer(date_obs,obs='Earth',sc=sc)
        import warnings
        warnings.warn(f"FITS headers for {date_obs} not available, using Earth observer" )
    else:
        try:
            meta=df.where(df.WAVELNTH==wlen).dropna().iloc[0].to_dict()
        except IndexError:
            meta=df.iloc[0].to_dict()
        if np.isnan(meta['CRPIX1']):
            meta['CRPIX1']=0.
        if np.isnan(meta['CRPIX2']):
            meta['CRPIX2']=0.
        fake_map=Map(np.zeros((10,10)),meta) #could probably do a faster way but this is convenient
        observer=fake_map.coordinate_frame.observer
    
    if sc:
        observer = SkyCoord(0*u.arcsec, 0*u.arcsec,obstime=date_obs,observer=observer,frame='helioprojective')

    if wcs:
        #while we're here and it's convenient...
        if df.empty:
            #wcs=get_wcs(date_obs, 'Earth')
            wcs=observer[1]
            observer=observer[0]
        else:
            wcs=WCS(meta)
        return observer,wcs
    return observer
    
def get_wcs(date_obs,obs_body,out_shape=(4096,4096),scale=None):
    '''generalized way to get WCS'''
    if isinstance(obs_body,str):
        obs_body=get_observer(date_obs, obs=obs_body)
    elif isinstance(obs_body,SkyCoord):
        ref_coord=obs_body
    else:
        ref_coord = SkyCoord(0*u.arcsec,0*u.arcsec,obstime=date_obs,
                              observer=obs_body,frame='helioprojective')
                              
    if scale:
        scale=(1., 1.)*ref_coord.observer.radius/u.AU*u.arcsec/u.pixel
    out_header = make_fitswcs_header(out_shape,ref_coord,scale=scale)
    return WCS(out_header)
    
def coordinate_behind_limb(coord: SkyCoord, pov=None) -> bool:
    '''Check if coordinate is behind the limb as seen by the desired observer. Do this by:
    - converting input coordinate to HGS
    - getting observer (pov) HGS (assume this gives center of frame)
    - check if input HGS is within observer HGS +- 90 degrees longitude '''
    _verify_coordinate_helioprojective(coord)
    if not pov:
        pov = coordinates_body(coord.obstime.to_datetime(),'Earth') #HEE
    #limb is at longitude +- 90 generally. slightly latitude dependent but going to ignore that here
    coord_hgs=coord.transform_to(HeliographicStonyhurst)
    pov_hgs=pov.transform_to(HeliographicStonyhurst(obstime=coord.obstime)) #lon gives center of frame I think
    #do r not lon!!
    if coord_hgs.lon.value > pov_hgs.lon.value -90 and coord_hgs.lon.value < pov_hgs.lon.value + 90: #"on disk" as seen by POV
        return False
    else:
        return True


def is_visible_from(date_obs: dt, flare_loc: tuple, obs_in = 'Earth', obs_out = 'SOLO') -> bool:
    '''Given a flare location seen by one observer, is it visible from a second observer? '''
    
    obs_in_hee = coordinates_body(date_obs, obs_in)
    obs_out_hee = coordinates_body(date_obs, obs_out)

    obs_flare_coord = SkyCoord(flare_loc[0]*u.arcsec,
                              flare_loc[1]*u.arcsec,
                              obstime=date_obs,
                              observer=obs_in,
                              frame='helioprojective')

    obs_out_ref_coord = SkyCoord(0*u.arcsec,
                               0*u.arcsec,
                               obstime=date_obs,
                               observer=obs_out_hee.transform_to(HeliographicStonyhurst(obstime=date_obs)),
                               frame='helioprojective')
                               
    obs_out_flare_coord= obs_flare_coord.transform_to(obs_out_ref_coord.frame)

    #is obs_out_flare_coord on the solar disk?
    if not coordinate_behind_limb(obs_out_flare_coord, pov=obs_out_hee):
        return obs_out_flare_coord
    else:
        raise ValueError(f"The input coordinate {obs_flare_coord} is behind the limb from the view of {obs_out}.")

#def get_observer(date_in,obs='Earth',wcs=True,sc=False,out_shape=(4096,4096)):
#    '''Get observer information. Get WCS object if requested. Return Observer as SkyCoord at (0",0") if requested.'''
#    hee = coordinates_body(date_in, obs)
#    observer=hee.transform_to(HeliographicStonyhurst(obstime=date_in))
#    if sc:
#        observer = SkyCoord(0*u.arcsec, 0*u.arcsec,obstime=date_in,observer=observer,frame='helioprojective')
#
#    if wcs == False:
#        return observer
#    else:
#        if isinstance(observer, SkyCoord):
#            refcoord=observer
#        else:
#            refcoord=SkyCoord(0*u.arcsec, 0*u.arcsec,obstime=date_in,observer=observer,frame='helioprojective')
#
#        out_header = sunpy.map.make_fitswcs_header(out_shape,refcoord,observatory=obs)
#
#        out_wcs = WCS(out_header)
#        return observer,out_wcs
#
#def coordinates_body(date_body,body_name,light_time=False):
#    """
#    Load the kernel needed in order to derive the
#    coordinates of the given celestial body and then return them in
#    Heliocentric Earth Ecliptic (HEE) coordinates.
#    """
#
#    # Observing time
#    obstime = spiceypy.datetime2et(date_body)
#
#    # Obtain the coordinates of Solar Orbiter
#    hee_spice, lighttimes = spiceypy.spkpos(body_name, obstime,
#                                     'SOLO_HEE_NASA', #  Reference frame of the output position vector of the object
#                                     'NONE', 'SUN')
#    hee_spice = hee_spice * u.km
#
#    # Convert the coordinates to HEE
#    body_hee = HeliocentricEarthEcliptic(hee_spice,
#                                          obstime=Time(date_body).isot,
#                                          representation_type='cartesian')
#    if not light_time:
#        # Return the HEE coordinates of the body
#        return body_hee
#    else:
#        return body_hee,lighttimes
