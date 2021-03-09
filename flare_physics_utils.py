import matplotlib.pyplot as plt

from run_iian_dem import gen_tresp_matrix
from scipy import constants
#import astropy.units as u
#from astropy.coordinates import SkyCoord
##import wcsaxes
#from astropy.wcs import WCS
#
#import sunpy.map
#import sunpy.coordinates
#import sunpy.coordinates.wcs_utils
#from sunpy.net import vso
import numpy as np
from scipy.special import wofz
#import .ma as ma
#import matplotlib.dates as mdates
#import pandas as pd
#
#from datetime import datetime as dt
#import glob
#import plotly.graph_objects as go
#import matplotlib
#from matplotlib import cm
#import pidly
#from sunpy.physics.differential_rotation import solar_rotate_coordinate, diffrot_map
#
#from flux_in_boxes import track_region_box

def expected_AIA_flux(EM,T,wavelength, size=None,log=False, trmatrix=False,tresp_logt=False):
    '''EM (cm^-3) = F*S/R(T)
    F: flux DN s^-1 px^-1
    S: size cm^2
    R(T): response function @ given T in DN cm^5 s^-1 px^-1
    
    Therefore: F = R(T) * EM / S
    leave out /S if size unknown, then units are: DN cm^2 s^-1 px^-1
    
    Boerner, P. F., Testa, P., Warren, H., Weber, M. A., & Schrijver,
    C. J. 2014, Sol. Phys., 289, 2377'''
    wavs=[94,131,171,193,211,335]
    if not trmatrix.all() or not tresp_logt.all():
        _,_,trmatrix,tresp_logt=gen_tresp_matrix(plot=False)
    widx=wavs.index(wavelength)
    R=trmatrix[:,widx]
    if not log:
        logT=np.log10(T*1e6)
    else:
        logT=T
    tidx=list(tresp_logt).index(find_closest(tresp_logt,logT))
    RT=R[tidx]
    #print(logT,RT)
    if size:
        return tresp_logt,(RT*EM)/size
    else:
        return tresp_logt,(RT*EM)

def plot_AIA_expected_fluxes(EM, T,size_range=[1e3,1e5], show=True, log=False, plotter='matplotlib'):
    wavs=[94,131,171,193,211,335]
    _,_,trmatrix,tresp_logt=gen_tresp_matrix(plot=False)
    sizevec=np.linspace(size_range[0],size_range[1],100)
    
    loci_curves=[]
    for w in wavs:
        loci_curve=[expected_AIA_flux(EM,T,w,size=s,trmatrix=trmatrix,tresp_logt=tresp_logt,log=log)[1] for s in sizevec]
        loci_curves.append(loci_curve)
    
    loci_curves=np.array(loci_curves)

    if show:
        clrs=['darkgreen','darkcyan','gold','sienna','indianred','darkslateblue']
        ylabel='$\mathrm{Flux\;(DN\;s^{-1}\;px^{-1})}$'
        xlabel='$\mathrm{Source\;size\;(cm^{2})}$'
        title="Expected AIA flux for EM=%.2E, T (MK)=%.2f" % (EM,T)
        if plotter=='plotly':
            fig=go.Figure()
            for i,w in enumerate(wavs):
                fig.add_trace(go.Scatter(x=sizevec,y=loci_curves[i],name=w,mode='lines',line=dict(color=clrs[i])))
            fig.update_layout(title=title,yaxis_title="Flux DN/s/px",xaxis_title="Source size cm2")
            #if log:
            fig.update_layout(xaxis_type='log',yaxis_type='log', xaxis_range=[np.log10(size_range[0]),np.log10(size_range[1])],height=570,width=570)
            fig.update_xaxes(showexponent = 'all',exponentformat = 'e')
            fig.update_yaxes(showexponent = 'all',exponentformat = 'e')
            fig.show()
        
        else:
            plt.rcParams.update({'font.size': 10})
            fig,ax=plt.subplots()
            for i,w in enumerate(wavs):
                ax.plot(sizevec,loci_curves[i],label=str(w),color=clrs[i])
            if log:
                ax.set_yscale('log')
                ax.set_xscale('log')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(loc='upper right')
            ax.set_title(title)
            fig.show()
    else:
        return tresp_logt,loci_curves

def thermal_energy_content(n,V,T):
    '''E = 3NkT, N= n/V (units? SI) '''
    return 3*constants.k*(n/V)*T

def find_closest(vec, val, index=False):
    closest_val= min(vec, key=lambda x: abs(x - val))
    if index:
        return vec.index(closest_val)
    else:
        return closest_val


#-----------------------------------------------------------------------------------------------------
#-- from https://github.com/tisobe/Python_scripts/blob/master/voigt_fit.py
#voigt: real part of Faddeeva function.                                                         ---
#-----------------------------------------------------------------------------------------------------

def voigt(x, y):

   """
   The Voigt function is also the real part of
   w(z) = exp(-z^2) erfc(iz), the complex probability function,
   which is also known as the Faddeeva function. Scipy has
   implemented this function under the name wofz()
   Input:   x and y
   Output:  real part of Faddeeva function.
   """

   z = x + 1j*y
   I = wofz(z).real
   return I

#-----------------------------------------------------------------------------------------------------
#-- Voigt: voigt line shape                                                                        ---
#-----------------------------------------------------------------------------------------------------

def Voigt(nu, alphaD, alphaL, nu_0, A, a=0, b=0):

   """
   The Voigt line shape in terms of its physical parameters
   Input:
            nu     --- x-values, usually frequencies.
            alphaD --- Half width at half maximum for Doppler profile
            alphaL --- Half width at half maximum for Lorentz profile
            nu_0   --- Central frequency
            A      --- Area under profile
            a, b   --- Background as in a + b*x
   Output: voigt line profile
   """
   f = np.sqrt(np.log(2))
   x = (nu-nu_0)/alphaD * f
   y = alphaL/alphaD * f
   backg = a + b*nu
   V = A*f/(alphaD*np.sqrt(np.pi)) * voigt(x, y) + backg
   return V

#-----------------------------------------------------------------------------------------------------
#-- funcV: Compose the Voigt line-shape                                                             --
#-----------------------------------------------------------------------------------------------------

def funcV(x,alphaD, alphaL, nu_0, I, a, b):

    """
    Compose the Voigt line-shape
    Input: p       --- parameter list [alphaD, alphaL, nu_0, A, a, b]
           x       --- x value list
    Output: voigt line profile
    """
    #alphaD, alphaL, nu_0, I, a, b = p
    return Voigt(x, alphaD, alphaL, nu_0, I, a, b)

#-----------------------------------------------------------------------------------------------------
#-- funcG: Gaussina Model                                                                          ---
#-----------------------------------------------------------------------------------------------------

def funcG(x,A, mu, sigma, zerolev):

    """
    Model function is a gaussian
    Input: p       --- parameter list [A, mu, sigma, zerolev]
          x       --- x value list
    """
    #A, mu, sigma, zerolev = p
    return( A * np.exp(-(x-mu)*(x-mu)/(2*sigma*sigma)) + zerolev )

#-----------------------------------------------------------------------------------------------------
#-- residualsV: Return weighted residuals of Voigt                                                 ---
#-----------------------------------------------------------------------------------------------------

def residualsV(p, data):
    """
    Return weighted residuals of Voigt
    Input: p       --- parameter list [alphaD, alphaL, nu_0, A, a, b]
           data    --- a list of list (x, y, err)
    """
    x, y, err = data
    return (y-funcV(p,x)) / err

#-----------------------------------------------------------------------------------------------------
#-- residualsG: Return weighted residuals of Gauss                                                 ---
#-----------------------------------------------------------------------------------------------------

def residualsG(p, data):
    """
    Return weighted residuals of Gauss
    Input: p       --- parameter list [A, mu, sigma, zerolev]
           data    --- a list of list (x, y, err)
    """
    x, y, err = data
    return (y-funcG(p,x)) / err

def Lorentzian(x, x0, a, gamma):
    """ Return Lorentzian line shape at x0 with HWHM gamma """
    return a * gamma**2 / ( gamma**2 + ( x - x0 )**2)
    
def fit_metrics(df,key,do_print=False):
    l2norm=np.linalg.norm(df[key],2)
    mse=np.mean(np.sum(df[key]**2))
    if do_print:
        print(l2norm,mse)
    return l2norm,mse
