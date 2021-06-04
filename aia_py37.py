import astropy
import astropy.units as u
import numpy as np
import pandas as pd

from datetime import datetime as dt
import glob

from aiapy.calibrate import degradation
from aiapy.calibrate.util import get_correction_table
from aiapy.calibrate import register, update_pointing

def dump_AIA_degredation(obstime,channels=[94,131,171,193,211,335],calibration_version=10,json=True):
    '''compute AIA degredation and put in json for use in py35. input: obstime as string'''
    utctime=astropy.time.Time(obstime, scale='utc')

    nc=len(channels)
    degs=np.empty(nc)
    for i in np.arange(nc):
        degs[i]=degradation(channels[i]* u.angstrom,utctime,calibration_version=calibration_version)
    df=pd.DataFrame({'channels':channels,'degradation':degs})
    if json:
        df.to_json('AIA_degredation_%s.json' % obstime)
    else:
        return degs
