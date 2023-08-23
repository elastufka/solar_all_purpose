
import numpy as np
import glob
import pidly
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy_map_utils import fix_units, find_centroid_from_map
    
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

def centroids_from_XRT(ffile='/Users/wheatley/Documents/Solar/NuStar/orbit8/Xray/XRT_Be_orbit8.fits',show=False):
    if not isinstance(ffile,str): #it's a map
        xrts8 = ffile
        _,cf,_ = find_centroid_from_map(xrts8,show=show)
        return cf
    else:
        xrt8 = fix_units(sunpy.map.Map(ffile))
        xbl1 = SkyCoord(-850*u.arcsec,300*u.arcsec,frame=xrt8.coordinate_frame)
        xtr1 = SkyCoord(-750*u.arcsec,400*u.arcsec,frame=xrt8.coordinate_frame)
        xrts8 = xrt8.submap(xbl1,xtr1)
        _,c1f,_ = find_centroid_from_map(xrts8,show=show)
        xbl2 = SkyCoord(-900*u.arcsec,200*u.arcsec,frame=xrt8.coordinate_frame)
        xtr2 = SkyCoord(-850*u.arcsec,300*u.arcsec,frame=xrt8.coordinate_frame)
        xrts2 = xrt8.submap(xbl2,xtr2)
        _,c2f,_ = find_centroid_from_map(xrts2,show=show)
        return c1f,c2f


