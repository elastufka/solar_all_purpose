import pandas as pd
import os
import glob
from astropy import units as u

def estimate_goes_flux(r, stix_counts, fit):
    """ r:  distance between the sun and solar orbiter in units of AU
        stix_counts:  background subtracted counts, energy range: 4-10 keV, 4 sec time bin
        fit: Series or dict containing fit parameters and calculated errors"""
    if isinstance(r,u.Quantity):
        rfloat=r.value
    else:
        rfloat=r
    log_corrected_counts=np.log10(stix_counts*rfloat**2)
    lower_bin=np.floor(log_corrected_counts/.5)*.5 # to nearest .5
    log_GOES_flux=fit.slope*log_corrected_counts + fit.intercept
    GOES_flux=10**log_GOES_flux
    one_sigma=fit[f"bin_{lower_bin}_sigma"]
    limits=[10**(log_GOES_flux - one_sigma), 10**(log_GOES_flux + one_sigma)]

    return GOES_flux, limits #in units of W/m^2

def get_latest_fit(fit_path='/home/erica/STIX/data'):
    '''get fit from latest cron run on the STIX test server '''
    fitfiles=glob.glob(f'{fit_path}/fit*.json')
    fitfiles.sort(key=os.path.getmtime)
    fit=pd.read_json(fitfiles[-1])
    return fit.iloc[0]
