import pandas as pd
import numpy as np
import os
import glob

from astropy import units as u
from datetime import datetime as dt
from datetime import timedelta as td
import sunpy
#from astropy.wcs import WCS
#from astropy.wcs.utils import wcs_to_celestial_frame,pixel_to_skycoord, skycoord_to_pixel
#import rotate_coord as rc
from astropy.time import Time
from visible_from_earth import *
#from rotate_maps import load_SPICE, coordinates_SOLO
#from sunpy.map.maputils import solar_angular_radius
import spiceypy
#import warnings
#from spiceypy.utils.exceptions import NotFoundError
#from rotate_maps_utils import rotate_hek_coords
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
    
def get_spacecraft_position(start_date,end_date,spacecraft='SPP', path_kernel="/Users/wheatley/Documents/Solar/STIX/solar-orbiter/kernels/mk",sphere=False):
    ''' have to be in the kernel directory while calling generate_positions... this is a pretty annoying thing about spicypy'''
    times=pd.date_range(start_date,end_date)
    cwd=os.getcwd()
    sc = hespice.Trajectory(spacecraft)
    os.chdir(path_kernel)
    sc.generate_positions(times, 'Sun', 'HEE') #is this HEE? ECLIPJ2000
    os.chdir(cwd)
    sc.change_units(u.au)
    if sphere:
        sc_r, sc_lat, sc_lon=cart2sphere(sc.x,sc.y,sc.z)
        return times,sc_r.value,sc_lat.value,sc_lon.value
    else:
        return times,sc.x.value,sc.y.value,sc.z.value
        
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
