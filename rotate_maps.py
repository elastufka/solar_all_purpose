'''
Created by Andrea Francesco Battaglia
andrea-battaglia@ethz.ch

************************************************************
Any comments and bug reports are really appreciated! Thanks!
************************************************************

Last modification: 06-Sep-2021
'''

############################################################
##### Imports
import astropy.units as u
import glob
import heliopy.data.spice as spicedata
import heliopy.spice as hespice
import numpy as np
import os
import spiceypy as spice
import sunpy
import warnings
warnings.filterwarnings('ignore')

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
from datetime import datetime
from reproject import reproject_adaptive, reproject_interp, reproject_exact
from sunpy.coordinates import Helioprojective
from sunpy.coordinates.frames import HeliocentricEarthEcliptic, HeliographicStonyhurst
from sunpy.map import Map, make_fitswcs_header
from sunpy.map.maputils import all_coordinates_from_map

##### Constants
km2AU = 6.6845871226706e-9*u.AU/u.km
############################################################


############################################################
##### Main functions
def as_seen_by_SOLO(map, date_solo=None, center=None, fov=None, out_shape=None, 
                    pixel_size=None): 
    """
    Return the input map as seen by Solar Orbiter.

    The map has to be a full disk map. If `center` and `fov` 
    are specified, the returned maps include also the cutout
    of the full disk map, centered on `center` and with the
    dimensions given by `fov`. 

    Parameters
    ------------
    map : `sunpy.map.Map`
        Input map to convert as seen by Solar Orbiter.
    path_kernel : string
        String containing the path of the folder in which the
        SPICE kernels are stored.
    date_solo : optional `astropy.time.Time` or 
                `datetime.datetime`
        Date used to obtain the Solar Orbiter position. 
        DEFAULT: the time of the input map
    center : optional array_like (float) or array_like (int)
        Center of the region of interest as seen from Earth, 
        given in arcsec. This region of interest will be the
        center of the returned map, but with coordinates as 
        seen by Solar Orbiter.
        DEFAULT: full disk map, without a region of interest
    fov : optional 'float' or optional array_like (float) or 
                    array_like (int)
        Field of view of the region of interest as seen from
        Earth, given in arcmin.
        If `center` is specified, fov has also to be
        specified, otherwise a default value of 5 arcmin in
        both x and y direction is given.
        DEFAULT: full disk map, without specify the fov
    out_shape : optional array_like (int)
        Dimensions in pixel of the output rotated full disk 
        map. To avoid possible memory problems, sometimes it
        may be good to reduce the pixel size of the map.
        DEFAULT: as in SDO/AIA, i.e., (4096, 4096)
    pixel_size : optional array_like (float)
        Pixel size in arcsec of the rotated map in x 
        (pixel_size[0]) and y (pixel_size[0]) directions.
        DEFAULT: adapted to the distance of Solar Orbiter to
                 the Sun
    """

    # Convert string format to datetime.
    if date_solo == None:
        date_solo = datetime.strptime(map.meta['date-obs'], '%Y-%m-%dT%H:%M:%S.%f')

    # Set the out_shape variable if not specified
    if out_shape == None:
        out_shape = (4096, 4096)
        
    # If center is specified and not the FOV, set default of 5 arcmin
    if center != None and fov == None:
        fov = [5,5]

    # Set the real radius of the Sun (and not the solar disk)
    map.meta['rsun_ref'] = sunpy.sun.constants.radius.to_value('m')

    # Get the HEE coordinates of Solar Orbiter
    solo_hee = coordinates_SOLO(date_solo)
    
    # Set the pixel size, if not specified by the user
    if pixel_size == None:
        dsun_solo = float(np.sqrt(solo_hee.x**2+solo_hee.y**2+solo_hee.z**2)/u.km*1e3)
        dsun_earth = float(map.dsun/u.m)
        factor = dsun_earth / dsun_solo
        pixel_size = [float(map.scale[0]/u.arcsec*u.pixel) * factor, 
                      float(map.scale[1]/u.arcsec*u.pixel) * factor]

    # Mask the off disk data (array type must be float!)
    hpc_coords = all_coordinates_from_map(map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / map.rsun_obs
    map = check_float(map) # maybe there is a better way of doing this
    map.data[r > 1.0] = np.nan
    
    # Set the coordinates of the reference pixel
    solo_ref_coord = SkyCoord(0*u.arcsec, 
                              0*u.arcsec,
                              obstime=date_solo,
                              observer=solo_hee.transform_to(HeliographicStonyhurst(obstime=date_solo)),
                              frame='helioprojective')

    # Create a FITS-WCS header from a coordinate object
    out_header = make_fitswcs_header(out_shape,
                                     solo_ref_coord,
                                     scale=pixel_size*u.arcsec/u.pixel, #(1.2, 1.2)*u.arcsec/u.pixel,
                                     instrument="SOLO-AIA",
                                     observatory="SOLO",
                                     wavelength=map.wavelength)

    # Standard World Coordinate System (WCS) for describing coordinates in a FITS file
    out_wcs = WCS(out_header)

    # Transform to HGS coordinates
    out_wcs.heliographic_observer = solo_hee.transform_to(HeliographicStonyhurst(obstime=date_solo))

    # Image reprojection
    #output, _ = reproject_adaptive(map, out_wcs, out_shape) # Can give memory problems
    output, _ = reproject_interp(map, out_wcs, out_shape) # The fastest algorithm
    #output, _ = reproject_exact(map, out_wcs, out_shape) # The slowest algorithm

    # 2D map as seen from Solar Orbiter
    solo_map = Map((output, out_header))
    
    # If center == None, then return only the full map, otherwise also the sub-map
    if center == None:
        return solo_map    
    else:
        center_solo_hpc = roi_hpc_SOLO(map, center, date_solo, solo_hee)

        # To convert FOV as seen from Solar Orbiter
        ratio_dist = solo_map.dsun/map.dsun

        # Coordinates of the sub-frame
        bl_solo = SkyCoord(center_solo_hpc.Tx - (fov[0]*60/(ratio_dist*2))*u.arcsec, 
                        center_solo_hpc.Ty - (fov[1]*60/(ratio_dist*2))*u.arcsec, 
                        frame=solo_map.coordinate_frame)
        tr_solo = SkyCoord(center_solo_hpc.Tx + (fov[0]*60/(ratio_dist*2))*u.arcsec, 
                        center_solo_hpc.Ty + (fov[1]*60/(ratio_dist*2))*u.arcsec, 
                        frame=solo_map.coordinate_frame)

        # Extract the sub-map
        solo_submap = solo_map.submap(bottom_left=bl_solo, top_right=tr_solo)

        return solo_map, solo_submap
    
    
##########



def as_seen_from_EARTH(map, path_kernel, date_earth=None, center=None, fov=None, out_shape=None, stix_erange=None,
                       pixel_size=None): 
    """
    Return the input map (STIX) as seen from Earth.
    
    See description `as_seen_by_SOLO`

    Parameters
    ------------
    map : `sunpy.map.Map`
        Input map to convert as it would be seen from Earth.
    path_kernel : string
        String containing the path of the folder in which the
        SPICE kernels are stored.
    date_earth : optional `astropy.time.Time` or 
                `datetime.datetime`
        Date used to obtain the Earth position. 
        DEFAULT: the time of the input map
    center : optional array_like (float) or array_like (int)
        Center of the region of interest as seen from Earth, 
        given in arcsec. This region of interest will be the
        center of the returned map, but with coordinates as 
        seen by Solar Orbiter.
        DEFAULT: full disk map, without a region of interest
    fov : optional 'float' or optional array_like (float) or 
                    array_like (int)
        Field of view of the region of interest as seen from
        Earth, given in arcmin.
        If `center` is specified, fov has also to be
        specified, otherwise a default value of 5 arcmin in
        both x and y direction is given.
        DEFAULT: full disk map, without specify the fov
    out_shape : optional array_like (int)
        Dimensions in pixel of the output rotated full disk 
        map. To avoid possible memory problems, sometimes it
        may be good to reduce the pixel size of the map.
        DEFAULT: as in SDO/AIA, i.e., (4096, 4096)
    stix_erange : optional `string`
        String containing the energy range of the STIX map. 
        This will appear in the title of the returned map.
        DEFAULT: string = "map"
    pixel_size : optional array_like (float)
        Pixel size in arcsec of the rotated map in x 
        (pixel_size[0]) and y (pixel_size[0]) directions.
        DEFAULT: adapted to the distance of Solar Orbiter to
                 the Sun
    """
    
    # Convert string format to datetime.
    if date_earth == None:
        date_earth = datetime.strptime(map.meta['date-obs'], '%Y-%m-%dT%H:%M:%S.%f')
    
    # Set the out_shape variable if not specified
    if out_shape == None:
        out_shape = (4096, 4096)
    
    # If center is specified and not the FOV, set default of 5 arcmin
    if center != None and fov == None:
        fov = [5,5]
    
    # Default stix_erange
    if stix_erange == None:
        stix_erange = "map"
    
    # Get the HEE coordinates of the Earth
    earth_hee = coordinates_EARTH(date_earth)
    
    # Set the pixel size, if not specified by the user
    if pixel_size == None:
        dsun_earth = float(np.sqrt(earth_hee.x**2+earth_hee.y**2+earth_hee.z**2)/u.km*1e3)
        dsun_solo = float(map.dsun/u.m)
        factor = dsun_solo / dsun_earth
        pixel_size = [float(map.scale[0]/u.arcsec*u.pixel) * factor, 
                      float(map.scale[1]/u.arcsec*u.pixel) * factor]
    
    # Mask the off disk data (array type must be float!)
    hpc_coords = all_coordinates_from_map(map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / map.rsun_obs
    map = check_float(map) # maybe there is a better way of doing this 
    map.data[r > 1.0] = np.nan
    
    # Get the distance of Earth from the Sun center
    #dist_earth_AU = np.sqrt(earth_hee.x**2+earth_hee.y**2+earth_hee.z**2)*km2AU
    
    # Set the coordinates of the reference pixel
    earth_ref_coord = SkyCoord(0*u.arcsec, 
                               0*u.arcsec,
                               obstime=date_earth,
                               observer=earth_hee.transform_to(HeliographicStonyhurst(obstime=date_earth)),
                               frame='helioprojective')

    # Create a FITS-WCS header from a coordinate object
    out_header_earth = make_fitswcs_header(out_shape,
                                     earth_ref_coord,
                                     scale=pixel_size*u.arcsec/u.pixel, #(1., 1.)*dist_earth_AU/u.AU*u.arcsec/u.pixel,
                                     instrument="STIX "+stix_erange+" (amplitudes only) as seen by Earth",
                                     observatory="Solar Orbiter")
    
    # Standard World Coordinate System (WCS) for describing coordinates in a FITS file
    out_wcs = WCS(out_header_earth)

    # Transform to HGS coordinates
    out_wcs.heliographic_observer = earth_hee.transform_to(HeliographicStonyhurst(obstime=date_earth)) 

    # Image reprojection
    #output, _ = reproject_adaptive(map, out_wcs, out_shape) # Can give memory problems
    output, _ = reproject_interp(map, out_wcs, out_shape) # The fastest algorithm
    #output, _ = reproject_exact(map, out_wcs, out_shape) # The slowest algorithm
    
    # STIX map as seen from Solar Orbiter
    stix_earth_map = Map((output, out_header_earth))
    
    # If center == None, then return only the full map, otherwise also the sub-map
    if center == None:
        return stix_earth_map
    else:
        center_earth_hpc = roi_hpc_EARTH(map, center, date_earth, earth_hee)

        # To convert FOV as seen from Earth
        ratio_dist = stix_earth_map.dsun/map.dsun

        # Coordinates of the sub-frame
        bl_earth = SkyCoord(center_earth_hpc.Tx - (fov[0]*60/(ratio_dist*2))*u.arcsec, 
                        center_earth_hpc.Ty - (fov[1]*60/(ratio_dist*2))*u.arcsec, 
                        frame=stix_earth_map.coordinate_frame)
        tr_earth = SkyCoord(center_earth_hpc.Tx + (fov[0]*60/(ratio_dist*2))*u.arcsec, 
                        center_earth_hpc.Ty + (fov[1]*60/(ratio_dist*2))*u.arcsec, 
                        frame=stix_earth_map.coordinate_frame)

        # Extract the sub-map
        stix_earth_submap = stix_earth_map.submap(bottom_left=bl_earth, top_right=tr_earth)

        return stix_earth_map, stix_earth_submap 

    
##########


def as_seen_by_STEREO(map, path_kernel, date_stereo=None, center=None, fov=None, out_shape=None, 
                      pixel_size=None): 
    """
    Return the input map as seen by STEREO A.

    The map has to be a full disk map. If `center` and `fov` 
    are specified, the returned maps include also the cutout
    of the full disk map, centered on `center` and with the
    dimensions given by `fov`. 

    Parameters
    ------------
    map : `sunpy.map.Map`
        Input map to convert as seen by STEREO A.
    path_kernel : string
        String containing the path of the folder in which the
        SPICE kernels are stored.
    date_stereo : optional `astropy.time.Time` or 
                  `datetime.datetime`
        Date used to obtain the STEREO A position. 
        DEFAULT: the time of the input map
    center : optional array_like (float) or array_like (int)
        Center of the region of interest as seen from Earth, 
        given in arcsec. This region of interest will be the
        center of the returned map, but with coordinates as 
        seen by STEREO A.
        DEFAULT: full disk map, without a region of interest
    fov : optional 'float' or optional array_like (float) or 
                    array_like (int)
        Field of view of the region of interest as seen from
        Earth, given in arcmin.
        If `center` is specified, fov has also to be
        specified, otherwise a default value of 5 arcmin in
        both x and y direction is given.
        DEFAULT: full disk map, without specify the fov
    out_shape : optional array_like (int)
        Dimensions in pixel of the output rotated full disk 
        map. To avoid possible memory problems, sometimes it
        may be good to reduce the pixel size of the map.
        DEFAULT: as in SDO/AIA, i.e., (4096, 4096)
    pixel_size : optional array_like (float)
        Pixel size in arcsec of the rotated map in x 
        (pixel_size[0]) and y (pixel_size[0]) directions.
        DEFAULT: adapted to the distance of Solar Orbiter to
                 the Sun
    """

    # Convert string format to datetime.
    if date_stereo == None:
        date_stereo = datetime.strptime(map.meta['date-obs'], '%Y-%m-%dT%H:%M:%S.%f')

    # Set the out_shape variable if not specified
    if out_shape == None:
        out_shape = (4096, 4096)
        
    # If center is specified and not the FOV, set default of 5 arcmin
    if center != None and fov == None:
        fov = [5,5]

    # Set the real radius of the Sun (and not the solar disk)
    map.meta['rsun_ref'] = sunpy.sun.constants.radius.to_value('m')

    # Get the HEE coordinates of STEREO A
    stereo_hee = coordinates_STEREO(date_stereo)
    
    # Set the pixel size, if not specified by the user
    if pixel_size == None:
        dsun_stereo = float(np.sqrt(stereo_hee.x**2+stereo_hee.y**2+stereo_hee.z**2)/u.km*1e3)
        dsun_earth = float(map.dsun/u.m)
        factor = dsun_earth / dsun_stereo
        pixel_size = [float(map.scale[0]/u.arcsec*u.pixel) * factor, 
                      float(map.scale[1]/u.arcsec*u.pixel) * factor]

    # Mask the off disk data (array type must be float!)
    hpc_coords = all_coordinates_from_map(map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / map.rsun_obs
    map = check_float(map) # maybe there is a better way of doing this
    map.data[r > 1.0] = np.nan
    
    # Set the coordinates of the reference pixel
    stereo_ref_coord = SkyCoord(0*u.arcsec, 
                                0*u.arcsec,
                                obstime=date_stereo,
                                observer=stereo_hee.transform_to(HeliographicStonyhurst(obstime=date_stereo)),
                                frame='helioprojective')

    # Create a FITS-WCS header from a coordinate object
    out_header = make_fitswcs_header(out_shape,
                                     stereo_ref_coord,
                                     scale=pixel_size*u.arcsec/u.pixel,
                                     instrument="STEREO A-AIA",
                                     observatory="STEREO",
                                     wavelength=map.wavelength)

    # Standard World Coordinate System (WCS) for describing coordinates in a FITS file
    out_wcs = WCS(out_header)

    # Transform to HGS coordinates
    out_wcs.heliographic_observer = stereo_hee.transform_to(HeliographicStonyhurst(obstime=date_stereo))

    # Image reprojection
    #output, _ = reproject_adaptive(map, out_wcs, out_shape) # Can give memory problems
    output, _ = reproject_interp(map, out_wcs, out_shape) # The fastest algorithm
    #output, _ = reproject_exact(map, out_wcs, out_shape) # The slowest algorithm

    # 2D map as seen from Solar Orbiter
    stereo_map = Map((output, out_header))
    
    # If center == None, then return only the full map, otherwise also the sub-map
    if center == None:
        return stereo_map    
    else:
        center_stereo_hpc = roi_hpc_STEREO(map, center, date_stereo, stereo_hee)

        # To convert FOV as seen from STEREO
        ratio_dist = stereo_map.dsun/map.dsun

        # Coordinates of the sub-frame
        bl_stereo = SkyCoord(center_stereo_hpc.Tx - (fov[0]*60/(ratio_dist*2))*u.arcsec, 
                             center_stereo_hpc.Ty - (fov[1]*60/(ratio_dist*2))*u.arcsec, 
                             frame=stereo_map.coordinate_frame)
        tr_stereo = SkyCoord(center_stereo_hpc.Tx + (fov[0]*60/(ratio_dist*2))*u.arcsec, 
                             center_stereo_hpc.Ty + (fov[1]*60/(ratio_dist*2))*u.arcsec, 
                             frame=stereo_map.coordinate_frame)

        # Extract the sub-map
        stereo_submap = stereo_map.submap(bottom_left=bl_stereo, top_right=tr_stereo)

        return stereo_map, stereo_submap
############################################################





############################################################
##### Support functions
def load_SPICE(obs_date, path_kernel):
    """
    Load the SPICE kernel that will be used to get the
    coordinates of the different spacecrafts.
    """
    #get cwd
    cwd=os.getcwd()
    
    # Convert string format to datetime
    obs_date = datetime.strptime(obs_date, '%Y-%m-%dT%H:%M:%S')
    
    # Check if path_kernel has folder format
    if path_kernel[-1] != '/': 
        path_kernel = path_kernel+'/'
    
    # Find the MK generation date ...
    MK_date_str = glob.glob(path_kernel+'/solo_*flown-mk_v*.tm')
    # ... and convert it to datetime
    MK_date = datetime.strptime(MK_date_str[0][-15:-7], '%Y%m%d')

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
    stereo_kernel = spicedata.get_kernel('stereo_a')
    hespice.furnish(stereo_kernel)

    # Change the CWD to the given path. Necessary to load correctly all kernels
    os.chdir(path_kernel)

    # Load one (or more) SPICE kernel into the program
    spice.furnsh(spice_kernel)
    
    print()
    print('SPICE kernels loaded correctly')
    print()
    
    #change back to original working directory
    os.chdir(cwd)
    
    
##########
    

    
def coordinates_SOLO(date_solo,light_time=False):
    """
    Load the kernel needed in order to derive the
    coordinates of Solar Orbiter and then return them in
    Heliocentric Earth Ecliptic (HEE) coordinates.
    """

    # Observing time (to get the SOLO coordinates)
    et_solo = spice.datetime2et(date_solo)

    # Obtain the coordinates of Solar Orbiter
    solo_hee_spice, lighttimes = spice.spkpos('SOLO', et_solo, 'SOLO_HEE_NASA', 'NONE', 'SUN')
    solo_hee_spice = solo_hee_spice * u.km

    # Convert the coordinates to HEE
    solo_hee = HeliocentricEarthEcliptic(solo_hee_spice, 
                                         obstime=Time(date_solo).isot, 
                                         representation_type='cartesian')
    
    if not light_time:
        # Return the HEE coordinates of Solar Orbiter
        return solo_hee
    else:
        return solo_hee,lighttimes



##########



def coordinates_EARTH(date_earth,light_time=False):
    """
    Load the kernel needed in order to derive the
    coordinates of the Earth and then return them in
    Heliocentric Earth Ecliptic (HEE) coordinates.
    """

    # Observing time (to get the Earth coordinates)
    et_stix = spice.datetime2et(date_earth)

    # Obtain the coordinates of Solar Orbiter
    earth_hee_spice, lighttimes = spice.spkpos('EARTH', et_stix,
                                     'SOLO_HEE_NASA', #  Reference frame of the output position vector of the object 
                                     'NONE', 'SUN')
    earth_hee_spice = earth_hee_spice * u.km

    # Convert the coordinates to HEE
    earth_hee = HeliocentricEarthEcliptic(earth_hee_spice, 
                                          obstime=Time(date_earth).isot, 
                                          representation_type='cartesian')
    if not light_time:
        # Return the HEE coordinates of the Earth
        return earth_hee
    else:
        return earth_hee,lighttimes



##########



def coordinates_STEREO(date_stereo,light_time=False):
    """
    Load the kernel needed in order to derive the
    coordinates of STEREO A and then return them in
    Heliocentric Earth Ecliptic (HEE) coordinates.
    """
    
    # Get the STEREO kernel and store the trajectory in the 'stereo_kernel' variable
    #stereo_kernel = spicedata.get_kernel('stereo_a')
    #hespice.furnish(stereo_kernel)
    
    # Change directory
    #os.chdir(path_kernel)

    # Load one (or more) SPICE kernel into the program
    #spice.furnsh('solo_ANC_soc-flown-mk.tm')

    # Observing time (to get the Earth coordinates)
    et_stereo = spice.datetime2et(date_stereo)

    # Obtain the coordinates of Solar Orbiter
    stereo_hee_spice, lighttimes = spice.spkpos('STEREO AHEAD', et_stereo,
                                     'SOLO_HEE_NASA', #  Reference frame of the output position vector of the object 
                                     'NONE', 'SUN')
    stereo_hee_spice = stereo_hee_spice * u.km

    # Convert the coordinates to HEE
    stereo_hee = HeliocentricEarthEcliptic(stereo_hee_spice, 
                                          obstime=Time(date_stereo).isot, 
                                          representation_type='cartesian')
    
    if not light_time:
        # Return the HEE coordinates of the STEREO
        return stereo_hee
    else:
        return stereo_hee,lighttimes


##########



def check_float(map):
    """
    Check if the data contained in map are float. If it is
    not the case, change the format.
    """

    if map.data.dtype.kind == 'i' or map.data.dtype.kind == 'u':
        map = Map((map.data.astype('float'), map.meta))
    
    return map



##########



def roi_hpc_SOLO(map, coord, date_solo, solo_hee):
    """
    Takes the coordinates of the region of interest (ROI) as
    seen from Earth (on the surface of the Sun) and
    transform them to hpc coordinates as seen (always on the
    Sun) from Solar Orbiter
    """

    # SkyCoord of the ROI as seen from Earth
    roi_earth_hpc = SkyCoord(coord[0]*u.arcsec, 
                             coord[1]*u.arcsec, 
                             frame=map.coordinate_frame)
    
    # Assume ROI to be on the surface of the Sun
    roi_inter = roi_earth_hpc.transform_to(HeliocentricEarthEcliptic)
    third_dim = 1*u.Rsun

    # ROI location in HEE
    roi_hee = SkyCoord(roi_inter.lon, roi_inter.lat, third_dim, 
                       frame=HeliocentricEarthEcliptic(obstime=date_solo))

    # Since now we have the full 3D coordinate of the ROI position
    # given in HEE, we can now transform that coordinated as seen
    # from Solar Orbiter and give them in Helioprojective coordinates
    roi_solo_hpc = roi_hee.transform_to(Helioprojective(obstime=date_solo,
                                        observer=solo_hee.transform_to(HeliographicStonyhurst(obstime=date_solo))))
    
    return roi_solo_hpc



##########



def roi_hpc_EARTH(map, coord, date_earth, earth_hee):
    """
    Takes the coordinates of the region of interest (ROI) as
    seen from Solar Orbiter (on the surface of the Sun) and
    transform them to hpc coordinates as seen (always on the
    Sun) from Earth
    """

    # SkyCoord of the ROI as seen from Solar Orbiter
    roi_solo_hpc = SkyCoord(coord[0]*u.arcsec, 
                            coord[1]*u.arcsec, 
                            frame=map.coordinate_frame)
    
    # Assume ROI to be on the surface of the Sun
    roi_inter = roi_solo_hpc.transform_to(HeliocentricEarthEcliptic)
    third_dim = 1*u.Rsun

    # ROI location in HEE
    roi_hee = SkyCoord(roi_inter.lon, roi_inter.lat, third_dim, 
                       frame=HeliocentricEarthEcliptic(obstime=date_earth))

    # Since now we have the full 3D coordinate of the ROI position
    # given in HEE, we can now transform that coordinated as seen
    # from Solar Orbiter and give them in Helioprojective coordinates
    roi_earth_hpc = roi_hee.transform_to(Helioprojective(obstime=date_earth,
                                         observer=earth_hee.transform_to(HeliographicStonyhurst(obstime=date_earth))))
    
    return roi_earth_hpc



##########



def roi_hpc_STEREO(map, coord, date_stereo, stereo_hee):
    """
    Takes the coordinates of the region of interest (ROI) as
    seen from Earth (on the surface of the Sun) and
    transform them to hpc coordinates as seen (always on the
    Sun) from STEREO
    """

    # SkyCoord of the ROI as seen from Earth
    roi_earth_hpc = SkyCoord(coord[0]*u.arcsec, 
                             coord[1]*u.arcsec, 
                             frame=map.coordinate_frame)
    
    # Assume ROI to be on the surface of the Sun
    roi_inter = roi_earth_hpc.transform_to(HeliocentricEarthEcliptic)
    third_dim = 1*u.Rsun

    # ROI location in HEE
    roi_hee = SkyCoord(roi_inter.lon, roi_inter.lat, third_dim, 
                       frame=HeliocentricEarthEcliptic(obstime=date_stereo))

    # Since now we have the full 3D coordinate of the ROI position
    # given in HEE, we can now transform that coordinated as seen
    # from STEREO and give them in Helioprojective coordinates
    roi_stereo_hpc = roi_hee.transform_to(Helioprojective(obstime=date_stereo,
                                        observer=stereo_hee.transform_to(HeliographicStonyhurst(obstime=date_stereo))))
    
    return roi_stereo_hpc



##########



def create_STIX_map(path_fits, datetime_map, path_kernel, flare_center, out_shape):
    """
    Converts the input array to a standard Sunpy Solar map, by
    returning the full-disk map containing the STIX source.
    
    At the moment, the function is optimized for the STIX FITS
    files generated by Paolo and Emma, since the first step
    is to convert the FITS they provide in a Sunpy Solar Map.
    """
    
    ###### Firstly, we need to create the STIX submap
    
    # Open the FITS file
    this_fits = fits.open(path_fits)
    
    # In order to properly set the WCS in the header, we need to find 
    # the HEE coordinates of Solar Orbiter and set the proper frame
    
    # Convert the string date and time to datetime.datetime
    datetime_map = datetime.strptime(datetime_map, '%Y-%m-%dT%H:%M:%S.%f')

    # Obtain the HEE coordinates of Solar Orbiter
    solo_hee = coordinates_SOLO(datetime_map, path_kernel)
    
    # Set the coordinates of the reference pixel of the STIX map
    stix_ref_coord = SkyCoord(flare_center[0]*u.arcsec, 
                              flare_center[1]*u.arcsec,
                              obstime=datetime_map,
                              observer=solo_hee.transform_to(HeliographicStonyhurst(obstime=datetime_map)),
                              frame='helioprojective')
    
    # Get the distance of Solar Orbiter from the Sun (center)
    dist_AU = np.sqrt(solo_hee.x**2+solo_hee.y**2+solo_hee.z**2)*km2AU

    # Create a FITS header containing the World Coordinate System (WCS) information
    out_header = make_fitswcs_header(this_fits[0].data,
                                     stix_ref_coord,
                                     scale=(1., 1.)*dist_AU/u.AU*u.arcsec/u.pixel,
                                     instrument="STIX (amplitudes only)",
                                     observatory="Solar Orbiter")

    # Create the STIX map
    #stix_map = Map((np.rot90(this_fits[0].data), out_header))
    stix_map = Map((this_fits[0].data, out_header))
    
    
    ###### Now, we have to create an "empty" Sunpy solar map
    
    # Set the coordinates of the reference pixel of the full-disk map
    stix_fd_ref_coord = SkyCoord(0*u.arcsec, 
                                 0*u.arcsec,
                                 obstime=datetime_map,
                                 observer=solo_hee.transform_to(HeliographicStonyhurst(obstime=datetime_map)),
                                 frame='helioprojective')

    # Create a FITS header containing the World Coordinate System (WCS) information
    # for the full-disk map
    out_header = make_fitswcs_header(out_shape,
                                     stix_fd_ref_coord,
                                     scale=(1., 1.)*dist_AU/u.AU*u.arcsec/u.pixel,
                                     instrument="STIX (amplitudes only)",
                                     observatory="Solar Orbiter")

    # Create the STIX full disk map
    stix_fulldisk = Map((np.zeros(out_shape), out_header))
    
    
    ###### Finally, put the STIX map on the full-disk map
    
    # Create the variable containing the axis on the Sun
    axis_solar_x = np.linspace(float(-dist_AU/u.AU*(out_shape[0]/2.)), 
                               float(dist_AU/u.AU*(out_shape[0]/2.)), 
                               num=out_shape[0])
    axis_solar_y = np.linspace(float(-dist_AU/u.AU*(out_shape[1]/2.)), 
                               float(dist_AU/u.AU*(out_shape[1]/2.)), 
                               num=out_shape[1])
    
    # Find the indices of the center of the axis
    ind_center_x = np.argmin(abs(axis_solar_x-flare_center[0]))
    ind_center_y = np.argmin(abs(axis_solar_y-flare_center[1]))
    
    # Put the STIX map on the full-disk map, where its center matches
    # the center of the axis previously defined
    shape_submap = stix_map.data.shape
    x_min = int(ind_center_x-shape_submap[0]/2)
    x_max = int(ind_center_x+shape_submap[0]/2)
    y_min = int(ind_center_y-shape_submap[1]/2)
    y_max = int(ind_center_y+shape_submap[1]/2)
    stix_fulldisk.data[y_min:y_max,x_min:x_max] = stix_map.data

    return stix_fulldisk
############################################################
