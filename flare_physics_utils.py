import matplotlib.pyplot as plt

from dem_utils import read_tresp_matrix
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
from sunpy_map_utils import *#cm_to_arcsec
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

def argmax2D(input_array):
    '''return indices of maximum of 2D array'''
    x=input_array.max(axis=1).argmax()
    y=input_array.max(axis=0).argmax()
    return [x,y]
    
def argmin2D(input_array):
    '''return indices of minimum of 2D array'''
    x=input_array.min(axis=1).argmin()
    y=input_array.min(axis=0).argmin()
    return [x,y]
    
def goes_class_to_flux(class_in):
    gdict={'A':-8,'B':-7,'C':-6,'M':-5,'X':-4}
    if len(class_in) > 1:
        goes_flux=float(class_in[1:])*10**gdict[class_in[0]]
    else: #just the letter
        goes_flux=10**gdict[class_in[0]]
    return goes_flux

def goes_flux_to_class(flux_in):
    gdict={-8:'A',-7:'B',-6:'C',-5:'M',-4:'X'}
    try:
        letter=gdict[int(np.log10(flux_in))]
    except KeyError:
        if np.log10(flux_in) < -8:
            letter='A'
        elif np.log10(flux_in) > -4:
            letter='X'
    number=np.round(flux_in/(10**int(np.log10(flux_in))),2)
    return letter+str(number)
    
def get_XRT_resp(tr_logt, date, filter='Be-Thin'):
    '''run tresp_be_thin43 in IDL. from ssw documentation:
    INPUTS:
    ;
    ;       TE     - (float array) log Temperature with a range from 5.0 to 8.0.
    ;       FW1    - (long) Filter No. on filter wheel 1.
    ;       FW2    - (long) Filter No. on filter wheel 2.
    ;       TIME   - (string) Time when you want to calcurate the XRT flux,
    ;                because XRT flux influenced by contamination is a function of time.
    ;                This time is used to derive the thickness of contaminant.
    ;                If you directly give the contamination thickness with CONTAMINATION_THICKNESS
    ;                keyword, you should omit this TIME input.
    ;       VEM        - [Optional input] (float) volume emission measure [cm^-3] of solar plasma in logalithmic scale (e.g., VEM = 44. for 1e44 [cm^-3]).
     OUTPUTS:
    ;
    ;       return - (float array) DN flux [DN sec^-1 pixel^-1] (for CEM = 1 [cm^-5] in default).
    ; NOTES:
    ;
    ;       filter on filter wheel 1
    ;         0: open
    ;         1: thin-Al-poly
    ;         2: C-poly
    ;         3: thin-Be
    ;         4: med-Be
    ;         5: med-Al
    ;
    ;       filter on filter wheel 2
    ;         0: open
    ;         1: thin-Al-mesh
    ;         2: Ti-poly
    ;         3: G-band (optical)
    ;         4: thick-Al
    ;         5: thick-Be
    '''
    idl = pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
    idl('tr_logt',tr_logt)
    idl('date',date)
    if filter == 'Be-Thin':
        fw1=3
        fw2=0
    idl('fw1',fw1)
    idl('fw2',fw2)
    idl('xrt_resp=xrt_flux(tr_logt,fw1,fw2,date,vem=43.)') #units DN s^-1 px^-1
    tresp=idl.xrt_resp
    tresp_cm5=tresp/(1e43)
    return tresp
    

def expected_AIA_flux(EM,T,wavelength, size=None,log=False, trmatrix=False,tresp_logt=False, perpixel=True):
    '''EM (cm^-3) = F*S/R(T)
    F: flux DN s^-1 px^-1
    S: size cm^2
    R(T): response function @ given T in DN cm^5 s^-1 px^-1
    convert px to cm^2
    => RT/pixsize_cm2 has units DN cm^3 s^-1
    Flux in units DN/s
    
    Therefore: F = R(T) * EM / S
    leave out /S if size unknown, then units are: DN cm^2 s^-1 px^-1
    
    Boerner, P. F., Testa, P., Warren, H., Weber, M. A., & Schrijver,
    C. J. 2014, Sol. Phys., 289, 2377'''
    wavs=[94,131,171,193,211,335]
    if type(trmatrix) != np.array or type(tresp_logt) != np.array:
        _,_,trmatrix,tresp_logt=read_tresp_matrix(plot=False)
    widx=wavs.index(wavelength)
    R=trmatrix[:,widx]
    if not log:
        logT=np.log10(T*1e6)
    else:
        logT=T
    tidx=list(tresp_logt).index(find_closest(tresp_logt,logT))
    RT=R[tidx]
    #print(logT,RT)
    if perpixel:
        pixsize_cm2=(arcsec_to_cm(0.6*u.arcsec).value)**2 #assume pixel size is .6"
        flux=EM*(RT/pixsize_cm2)
    if size:
        return tresp_logt,flux/size
    else:
        return tresp_logt,flux

def all_expected_AIA_fluxes(EM,T,size=None,log=False, trmatrix=False,tresp_logt=False, perpixel=True):
    fluxes=[]
    for w in [94,131,171,193,211,335]:
       calc=expected_AIA_flux(EM,T,w,size=size,trmatrix=trmatrix,tresp_logt=tresp_logt,log=log)
       fluxes.append(calc[1])
    return fluxes

def plot_AIA_expected_fluxes(EM, T,size_range=[1e3,1e5], show=True, log=False, plotter='matplotlib',sunits='cm2'):
    wavs=[94,131,171,193,211,335]
    _,_,trmatrix,tresp_logt=read_tresp_matrix(plot=False)
    sizevec=np.linspace(size_range[0],size_range[1],100)
    
    loci_curves=[all_expected_AIA_fluxes(EM,T,size=s,trmatrix=trmatrix,tresp_logt=tresp_logt,log=log) for s in sizevec]
    
    loci_curves=np.array(loci_curves).T #transpose?

    if show:
        clrs=['darkgreen','darkcyan','gold','sienna','indianred','darkslateblue']
        ylabel='$\mathrm{Flux\;(DN\;s^{-1}\;px^{-1})}$'
        if sunits=='arcsec':
            sizevec=[cm_to_arcsec(np.sqrt(s)*u.cm).value**2 for s in sizevec] #is it okay that it's squared?
            size_range=[cm_to_arcsec(np.sqrt(s)*u.cm).value**2 for s in size_range]
            xlabel='arcsec^2'
        else:
            xlabel='$\mathrm{Source\;size\;(cm^{2})}$'
        title="Expected AIA flux for EM=%.2E, T (MK)=%.2f" % (EM,T)
        if plotter=='plotly':
            fig=go.Figure()
            for i,w in enumerate(wavs):
                fig.add_trace(go.Scatter(x=sizevec,y=loci_curves[i],name=w,mode='lines',line=dict(color=clrs[i])))
            fig.update_layout(title=title,yaxis_title="Flux DN/s/px",xaxis_title=xlabel)
            #if log:
            fig.update_layout(xaxis_type='log',yaxis_type='log', xaxis_range=[np.log10(size_range[0]),np.log10(size_range[1])])#,height=570,width=650)
            fig.update_xaxes(showexponent = 'all',exponentformat = 'e')
            fig.update_yaxes(showexponent = 'all',exponentformat = 'e')
            fig.show()
        
        else:
            plt.rcParams.update({'font.size': 10})
            fig,ax=plt.subplots()
            for i,w in enumerate(wavs):
                ax.plot(sizevec,loci_curves[i],label=str(w),color=clrs[i])
            #if log:
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(loc='upper right')
            ax.set_title(title)
            fig.show()
    else:
        return tresp_logt,loci_curves
        
def EM_loci_curves(flux_obs,trmatrix=False):
    '''counterpart to expected_AIA_flux: EM(t) = flux_obs/(tresp/px)
    flux_obs is array of size (6,), tresp has size (6, len(temps)), output has shape(6,len(temps)'''
    if type(trmatrix) !=np.array:
        _,_,trmatrix,tresp_logt=read_tresp_matrix(plot=False)
    pixsize_cm2=(arcsec_to_cm(0.6*u.arcsec).value)**2 #assume pixel size is .6"
    return (flux_obs/(trmatrix/pixsize_cm2)).T #array
    
def plot_EM_loci_curves(flux_obs,trmatrix=False,plotter='matplotlib'):
    if type(trmatrix) !=np.array:
        _,_,trmatrix,tresp_logt=read_tresp_matrix(plot=False)
    EM_loci=EM_loci_curves(flux_obs,trmatrix=trmatrix)
    clrs=['darkgreen','darkcyan','gold','sienna','indianred','darkslateblue']
    wavs=[94,131,171,193,211,335]

    if plotter=='plotly':
        ylabel="EM (cm<sup>3</sup>)"
        xlabel="log<sub>10</sub>(T)"
        fig=go.Figure()
        for i,w in enumerate(wavs):
            fig.add_trace(go.Scatter(x=tresp_logt,y=EM_loci[i],name=w,mode='lines',line=dict(color=clrs[i])))
        fig.update_layout(yaxis_title=ylabel,xaxis_title=xlabel)
        #if log:
        fig.update_layout(yaxis_type='log')#,height=570,width=570)
        fig.update_yaxes(showexponent = 'all',exponentformat = 'e')
        #fig.show()
    
    else:
        ylabel="EM (cm$^3$)"
        xlabel="log$_{10}$(T)"
        plt.rcParams.update({'font.size': 10})
        fig,ax=plt.subplots()
        for i,w in enumerate(wavs):
            ax.plot(tresp_logt,EM_loci[i],label=str(w),color=clrs[i])
        #if log:
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right')
        #ax.set_title(title)
        #fig.show()
        
    return fig
    
def dE_from_DEM(dem, logt, V):
    '''calclulate dE=3n dT/V where n is calculated from dem
    Units:
    E: J
    n: kg m^-3
    T: K
    V: m^3
    '''
    
    dE=[]
    for ldem,T in zip(logt):
        density=dem_to_density(ldem,T)
        dE.append(thermal_energy_content(density,V,10**T))
    return dE
    
def dem_to_density(dem,logt,px_size_cm2):
    '''convert DEM at a given T from units cm−5 K−1 to m^-3 '''
    return (dem*(10**T)*px_size_cm2)*1e-6
    
def img_area_to_volume(area, aia=True, sphere=True, loop=False, arcsec=True):
    '''convert image area (in arcsec or cm^2) to volume '''
    area_conv=area
    if arcsec: #convert area to m^2
        if aia: #use AIA pixel size
            area_conv=area*arcsec_to_cm()
        else: #use input value
            area_conv=aia*area
    radius=np.sqrt(area_conv/np.pi)
    if sphere: #treat as sphere
        return (4/3)*np.pi*radius**3
    if loop: #loop is loop length in compatible units (m)
        return loop*area_conv
        

def thermal_energy_content(n,V,T):
    '''E = 3NkT, N= n/V (units? SI => T in Kelvin, V in m^3, n in kg m^-3) '''
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
