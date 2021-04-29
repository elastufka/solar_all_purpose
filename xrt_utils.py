import urllib.request

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

#from datetime import datetime as dt
import glob
#import plotly.graph_objects as go
#import matplotlib
#from matplotlib import cm
import pidly


def download_xrt_from_HEC(textfile,target_dir=False):
    ''' this should absolutely not be necessary so why is it?
    textfile is list from HEC catalog query order creation
    https://www.lmsal.com/cgi-ssw/www_sot_cat.sh
    
    ... why do they have no download all button?
    '''
    with open(textfile) as f:
        lines=[line[:line.rfind('.fits')+5] for line in f.readlines() if line.startswith('http')]
    if target_dir:
        dest=target_dir
    else:
        dest=''
    print("fetching %s files" % len(lines))
    for url in lines:
        #print(url,dest+'/'+url[url.rfind('/')+1:])
        urllib.request.urlretrieve(url, dest+'/'+url[url.rfind('/')+1:])
    
def xrt_prep(files,grade_map=True):
    '''wrapper for IDL code from Iain: https://github.com/ianan/axn_example/blob/main/make_xrt_for_python.pro
    ; Example from Sep 2020 data
    ff20=file_search('','XRT20200912_204028.8.fits')
    read_xrt,ff20,ind,data,/force
    xrt_prep,ind,data,indp,datap,/float,grade_map=gm,grade_type=1,/coalign
    indp.timesys='UTC'
    filtout=indp.ec_fw1_+'_'+indp.ec_fw2_
    resout=strcompress(string(indp.naxis1),/rem)
    fnameout='XRT_'+break_time(indp.date_obs)+'_'+filtout+'_'+resout+'.fits'
    write_xrt,indp,datap,outdir='',outfile=fnameout,/ver
    gfnameout='gm_XRT_'+break_time(indp.date_obs)+'_'+filtout+'_'+resout+'.fits'
    write_xrt,indp,gm,outdir='',outfile=gfnameout,/ver
     '''
    idl=pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
    idl("xrt_search_network,/enable") #in case it needs to download calibration materials
    idl("outdir",files[0][:files[0].rfind('/')])
    for file in files:
        fnameout=file[file.rfind('/')+1:-5]+'_prepped.fits'
        gfnameout=file[file.rfind('/')+1:-5]+'_grademap.fits' #can make them fancy later
        idl("f",file)
        idl("fnameout",fnameout)
        idl("read_xrt,f, ind,data,/force") #do I have to do this on at a time?
        idl("xrt_prep,ind,data,indp,datap,/float,grade_map=gm,grade_type=1,/coalign")
        idl("print,outdir,fnameout")
        idl("write_xrt,indp,datap,outdir=outdir,outfile=fnameout,/ver")
        if grade_map:
            idl("gfnameout",gfnameout)
            idl("write_xrt,indp,gm,outdir=outdir,outfile=gfnameout,/ver")

        

