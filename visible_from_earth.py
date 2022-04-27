import astropy.units as u
import numpy as np

from astropy.coordinates import SkyCoord
from datetime import datetime as dt
from sunpy.coordinates import Helioprojective
from sunpy.coordinates.frames import HeliocentricEarthEcliptic, HeliographicStonyhurst
from sunpy.map.maputils import _verify_coordinate_helioprojective #coordinate_is_on_solar_disk #all_coordinates_from_map,
from sunpy.map import Map, make_fitswcs_header
import drms
from astropy.wcs import WCS
import spiceypy
from astropy.time import Time

import rotate_maps as rm

def visible_from_earth(date_solo: dt,flare_loc_solo: tuple, plot_HEE:bool = False) -> SkyCoord:
    '''Given Solar Orbiter flare location, determine if that flare will be visible from an Earth observer (). Return flare coordinates in Earth-observer heiloprojective frame. If no flare location specified, draw SO limb on Earth-observer helioprojective frame.'''
        
    solo_hee = rm.coordinates_SOLO(date_solo)

    solo_flare_coord = SkyCoord(flare_loc_solo[0]*u.arcsec,
                              flare_loc_solo[1]*u.arcsec,
                              obstime=date_solo,
                              observer=solo_hee.transform_to(HeliographicStonyhurst(obstime=date_solo)),
                              frame='helioprojective')

    #transform solo_flare_coord to as seen from Earth:
    # Get the HEE coordinates of the Earth
    earth_hee = rm.coordinates_EARTH(date_solo)
    
    if plot_HEE:
        fig,ax=plt.subplots()
        ax.scatter(solo_hee.x/1e8,solo_hee.y/1e8,color='r')
        ax.annotate('SO',(solo_hee.x.value/1e8,solo_hee.y.value/1e8))
        ax.scatter(earth_hee.x/1e8,earth_hee.y/1e8,color='k')
        ax.annotate('Earth',(earth_hee.x.value/1e8,earth_hee.y.value/1e8))
        ax.scatter(0,0,color='y',s=500) #sun
        ax.set_xlim([-2,2])
        ax.set_ylim([-2,2])
        ax.set_xlabel("HEE_x")
        ax.set_ylabel("HEE_y")
        fig.show()
    
    earth_ref_coord = SkyCoord(0*u.arcsec,
                               0*u.arcsec,
                               obstime=date_solo,
                               observer=earth_hee.transform_to(HeliographicStonyhurst(obstime=date_solo)),
                               frame='helioprojective')
                                                   
    earth_flare_coord= solo_flare_coord.transform_to(earth_ref_coord.frame)
    if not coordinate_behind_limb(earth_flare_coord, pov=earth_hee):
        return earth_flare_coord
    else:
        raise ValueError(f"The input coordinate {solo_flare_coord} is behind the limb from the view of Earth.")
        
def visible_from_SO(date_solo: dt,flare_loc_earth: tuple) -> SkyCoord:
    '''The oppsoite, to make sanity checkng easier using already-available rotated AIA maps plots'''

    earth_hee = rm.coordinates_EARTH(date_solo)
    solo_hee = rm.coordinates_SOLO(date_solo)

    earth_flare_coord = SkyCoord(flare_loc_earth[0]*u.arcsec,
                              flare_loc_earth[1]*u.arcsec,
                              obstime=date_solo,
                              observer='earth',
                              frame='helioprojective')

    solo_ref_coord = SkyCoord(0*u.arcsec,
                               0*u.arcsec,
                               obstime=date_solo,
                               observer=solo_hee.transform_to(HeliographicStonyhurst(obstime=date_solo)),
                               frame='helioprojective')
                               
    solo_flare_coord= earth_flare_coord.transform_to(solo_ref_coord.frame)

    #is earth_flare_coord on SO-view of solar disk?
    if not coordinate_behind_limb(solo_flare_coord, pov=solo_hee):
        return solo_flare_coord
    else:
        raise ValueError(f"The input coordinate {earth_flare_coord} is behind the limb from the view of Solar Orbiter.")
        
def visible_from_STEREO(date_stereo: dt,flare_loc: tuple,obs_instr:str) -> SkyCoord:
    '''is AIA or SO event visible from STEREO-A'''
    if obs_instr in ['Earth','earth']:
        obs_hee = rm.coordinates_EARTH(date_stereo)
    elif obs_instr in ['SO','SOLO','Solar Orbiter','STIX']:
        obs_hee = rm.coordinates_SOLO(date_stereo)
    else:
        raise ValueError('Observing instrument is not valid! Must be one of Earth or SOLO')
        
    stereo_hee = rm.coordinates_STEREO(date_stereo)

    obs_flare_coord = SkyCoord(flare_loc[0]*u.arcsec,
                              flare_loc[1]*u.arcsec,
                              obstime=date_stereo,
                              observer=obs_hee.transform_to(HeliographicStonyhurst(obstime=date_stereo)),
                              frame='helioprojective')

    stereo_ref_coord = SkyCoord(0*u.arcsec,
                               0*u.arcsec,
                               obstime=date_stereo,
                        observer=stereo_hee.transform_to(HeliographicStonyhurst(obstime=date_stereo)),
                               frame='helioprojective')
                               
    stereo_flare_coord= obs_flare_coord.transform_to(stereo_ref_coord.frame)

    #is earth_flare_coord on SO-view of solar disk?
    if not coordinate_behind_limb(stereo_flare_coord, pov=stereo_hee):
        return stereo_flare_coord
    else:
        raise ValueError(f"The input coordinate {stereo_flare_coord} is behind the limb from the view of STEREO.")

def is_visible_from_earth(date_solo: dt,flare_loc_solo: tuple) -> bool:
    '''wrapper returns bool '''
    try:
        _ = visible_from_earth(date_solo,flare_loc_solo,plot_HEE=False)
        return True
    except ValueError:
        return False

def is_visible_from_SO(date_solo: dt,flare_loc_earth: tuple) -> bool:
    '''wrapper returns bool '''
    try:
        _ = visible_from_SO(date_solo,flare_loc_earth)
        return True
    except ValueError:
        return False
        
def is_visible_from_STEREO(date_stereo: dt,flare_loc: tuple, obs_instr: str) -> bool:
    '''wrapper returns bool '''
    try:
        _ = visible_from_STEREO(date_stereo,flare_loc,obs_instr)
        return True
    except ValueError:
        return False
        
def coordinates_body(date_body,body_name,light_time=False):
    """
    Load the kernel needed in order to derive the
    coordinates of the given celestial body and then return them in
    Heliocentric Earth Ecliptic (HEE) coordinates.
    """

    # Observing time
    obstime = spiceypy.datetime2et(date_body)

    # Obtain the coordinates of Solar Orbiter
    hee_spice, lighttimes = spiceypy.spkpos(body_name, obstime,
                                     'HEE', #  Reference frame of the output position vector of the object
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

def coordinate_behind_limb(coord: SkyCoord, pov=None) -> bool:
    '''Check if coordinate is behind the limb as seen by the desired observer. Do this by:
    - converting input coordinate to HGS
    - getting observer (pov) HGS (assume this gives center of frame)
    - check if input HGS is within observer HGS +- 90 degrees longitude '''
    _verify_coordinate_helioprojective(coord)
    if not pov:
        pov = rm.coordinates_EARTH(coord.obstime.to_datetime()) #HEE
    #limb is at longitude +- 90 generally. slightly latitude dependent but going to ignore that here
    coord_hgs=coord.transform_to(HeliographicStonyhurst)
    pov_hgs=pov.transform_to(HeliographicStonyhurst(obstime=coord.obstime)) #lon gives center of frame I think
    
    #print(coord_hgs,(pov_hgs.lon.value -90,pov_hgs.lon.value +90))
    
    #should maintain the units here ideally
    
    #do r not lon!!
    if coord_hgs.lon.value > pov_hgs.lon.value -90 and coord_hgs.lon.value < pov_hgs.lon.value + 90: #"on disk" as seen by POV
        return False
    else:
        return True
        
#def get_observer(date_in,obs='Earth',wcs=True,sc=False):
#    ''' yet another wrapper'''
#    if obs == 'Earth':
#        observer=get_Earth_observer(date_in,sc=sc)
#        if wcs:
#            wcs_out=get_Earth_wcs(date_in)
#    elif obs in ['AIA','SDO']:
#        observer,wcs_out=get_AIA_observer(date_in,wcs=True,sc=sc)
#
#    elif obs in ['SO','SOLO','Solar Orbiter']:
#        observer=get_SO_observer(date_in,sc=sc)
#        if wcs:
#            wcs_out=get_SO_wcs(date_in)
#
#    if wcs == False:
#        return observer
#    else:
#        return observer,wcs_out
        
def get_observer(date_in,obs='Earth',wcs=True,sc=False,wlen=1600,out_shape=(4096,4096),scale=None):
    '''Get observer information. Get WCS object if requested. Return Observer as SkyCoord at (0",0") if requested.'''
    if obs in ['AIA','SDO']:
        observer,wcs_out=get_AIA_observer(date_in,wcs=True,sc=sc,wlen=wlen)
    else:
        hee = coordinates_body(date_in, obs)
        observer=hee.transform_to(HeliographicStonyhurst(obstime=date_in))
        if sc:
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



