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

def estimate_goes_class(peak_counts:float,dsun:float, coeffs:list, error_lut:dict,default_error=0.3):
    """
    estimate goes class
       see https://pub023.cs.technik.fhnw.ch/wiki/index.php?title=GOES_Flux_vs_STIX_counts
    Parameters:
        peak_counts:  float
            STIX background subtracted peak counts
        dsun: float
            distance between solar orbiter and the sun in units of au
        coeff: list
            polynomial function coefficients
        error_lut: dict
            a look-up table contains errors  in log goes flux
            for example, errors_lut={'bins':[[0,0.5],[0.5,2]],'errors':[0.25,0.25]}
            bins contains bin edges and errors the corresponding error of the bin
        default_error: float
            default error in log goes flux
    """
    stix_cnts=counts*dsun**2
    if stix_cnts<=0:
        return {'min':None,'max':None,'max':None}
    x =np.log10(stix_cnts)
    g=lambda y: sum([coeffs[i] * y ** i for i in range(len(coeffs))])
    f=lambda y: goes_flux_to_class(10 ** g(y))
    error=default_error #default error
    try:
        bin_range=errors['bins']
        errors=errors['errors']
        for b,e in zip(bin_range, errors):
            if b[0] <= x <= b[1]:
                error=e
                break
    except (KeyError, IndexError, TypeError):
        pass
    result={'min': f(x -error),  'center': f(x),  'max':f(x+error),
            'parameters':{'coeff':coeffs, 'error':error}
            }
    return result
