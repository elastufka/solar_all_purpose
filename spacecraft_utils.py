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
from visible_from_earth import *
#from rotate_maps import load_SPICE, coordinates_SOLO
#from sunpy.map.maputils import solar_angular_radius
#import spiceypy
#import warnings
#from spiceypy.utils.exceptions import NotFoundError
#from rotate_maps_utils import rotate_hek_coords
import heliopy.data.spice as spicedata
import heliopy.spice as spice

def furnish_kernels(spacecraft_list=['psp','stereo_a','stereo_a_pred','bepi_pred','psp_pred']):
    k= spicedata.get_kernel(spacecraft_list[0])
    for sc in spacecraft_list[1:]:
        k+=spicedata.get_kernel(sc)
    spice.furnish(k)
    
def get_spacecraft_position(start_date,end_date,spacecraft='SPP', path_kernel="/Users/wheatley/Documents/Solar/STIX/solar-orbiter/kernels/mk"):
    ''' have to be in the kernel directory while calling generate_positions... this is a pretty annoying thing about spicypy'''
    times=pd.date_range(start_date,end_date)
    cwd=os.getcwd()
    sc = spice.Trajectory(spacecraft)
    os.chdir(path_kernel)
    sc.generate_positions(times, 'Sun', 'ECLIPJ2000') #is this HEE?
    os.chdir(cwd)
    sc.change_units(u.au)
    return sc.x.value,sc.y.value,sc.z.value
    
