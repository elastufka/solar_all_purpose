import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
#import wcsaxes
from astropy.wcs import WCS

import sunpy.map
import sunpy.coordinates
import sunpy.coordinates.wcs_utils
from sunpy.net import Fido
from sunpy.net import attrs as a
import numpy as np
import numpy.ma as ma
import matplotlib.dates as mdates
import pandas as pd

from datetime import datetime as dt
import glob
import plotly.graph_objects as go
import matplotlib
from matplotlib import cm
import pidly
from sunpy.physics.differential_rotation import solar_rotate_coordinate#, diffrot_map
from sunpy.coordinates.frames import HeliocentricEarthEcliptic, HeliographicStonyhurst
from sunpy.map.maputils import solar_angular_radius, _verify_coordinate_helioprojective
from skimage.transform import downscale_local_mean
from scipy.ndimage import sobel
from skimage.measure import find_contours
import pickle
import cmath
#from spacecraft_utils import get_observer, coordinates_body
    
def add_arcsecs():
    """Add arcsecs as astropy Unit. Mainly used to read in STIX FITS files."""
    ar=u.def_unit("arcsecs",1*u.arcsec)
    u.add_enabled_units([ar])
    
def hglt_to_hee(hgln_deg,hglt_deg,obstime):
    """Transform Heligraphic Stonyhurst coordinates to Heliocentric Earth Ecliptic (cartesian) coordinates
    
    Args:
        hgln_deg (float): Heliographic Stonyhurst longitude in degrees
        hglt_deg (float): Heliographic Stonyhurst latitude in degrees
        obstime (datetime, astropy.time.Time): Time of observation
    
    Returns:
        SkyCoord : coordinate in HEE
    """
    sc0=SkyCoord(hgln_deg*u.deg,hglt_deg*u.deg,frame='heliographic_stonyhurst',obstime=obstime)
    sc1=sc0.transform_to(frame='heliocentricearthecliptic')
    sc1.representation_type='cartesian'
    return sc1
    
    
def transform_observer(mcutout,sobs,swcs,scale=None):
   #deal with off-disk... if there are off-disk pixels in input, only consider on-disk
    #refcoord=SkyCoord(0,0,unit=u.arcsec,frame=sobs.frame)#='helioprojective',observer=sobs,obstime=sobs.obstime)
    #stix_ref_coord=mcutout.reference_coordinate.transform_to(sobs.frame)
    #blr=mcutout.bottom_left_coord.transform_to(sobs.frame)
    #trr=mcutout.top_right_coord.transform_to(sobs.frame)
    #if np.isnan([blr.Tx.value,blr.Ty.value]).any() or np.isnan([trr.Tx.value,trr.Ty.value]).any():
    #    blr,trr=rotated_bounds_on_disk(mcutout,sobs.frame)
    
    # Obtain the pixel locations of the edges of the reprojected map
    edges_pix = np.concatenate(sunpy.map.map_edges(mcutout))
    edges_coord = mcutout.pixel_to_world(edges_pix[:, 0], edges_pix[:, 1])
    new_edges_coord = edges_coord.transform_to(sobs)
    new_edges_xpix, new_edges_ypix = swcs.world_to_pixel(new_edges_coord)

    # Determine the extent needed
    left, right = np.min(new_edges_xpix), np.max(new_edges_xpix)
    bottom, top = np.min(new_edges_ypix), np.max(new_edges_ypix)
    
    #if scale:
    #    scale=(1., 1.)*sobs.observer.radius/u.AU*u.arcsec/u.pixel

    # Adjust the CRPIX and NAXIS values
    modified_header = sunpy.map.make_fitswcs_header((1, 1), sobs,scale=scale)
    modified_header['crpix1'] -= left
    modified_header['crpix2'] -= bottom
    modified_header['naxis1'] = int(np.ceil(right - left))
    modified_header['naxis2'] = int(np.ceil(top - bottom))
    
    return modified_header
    
def smart_reproject(mcutout,observatory='Solar Orbiter',scale=None):
    sobs,swcs=get_observer(pd.to_datetime(mcutout.meta['date-obs']),obs=observatory,sc=True, out_shape=(1,1),scale=scale)
    submap_header=transform_observer(mcutout,sobs,swcs,scale=scale)
    rotated_map=mcutout.reproject_to(submap_header)
    return rotated_map

def rotate_all_coords(mcutout,frame):
    return sunpy.map.all_coordinates_from_map(mcutout).transform_to(frame)
    
#def print_arr_stats(arr, ignore_nan=False):
#    if ignore_nan:
#        print(f"Mean: {np.nanmean(arr)}\nMin: {np.nanmin(arr)}\nMax: {np.nanmax(arr)}\nStd: {np.nanstd(arr)}\n")
#    else:
#        print(f"Mean: {np.mean(arr)}\nMin: {np.min(arr)}\nMax: {np.max(arr)}\nStd: {np.std(arr)}\n")

def rotated_bounds_on_disk(smap,frame):
    '''use this to determine extent of rotated shape in case bottom left and top right don't work out...'''
    cc=rotate_all_coords(smap,frame)
    on_disk=sunpy.map.coordinate_is_on_solar_disk(cc)
    on_disk_coordinates=cc[on_disk]
    tx = on_disk_coordinates.Tx.value
    ty = on_disk_coordinates.Ty.value
    return SkyCoord([np.nanmin(tx), np.nanmax(tx)] * u.arcsec,
                    [np.nanmin(ty), np.nanmax(ty)] * u.arcsec,
                    frame=smap.coordinate_frame)
                    
def wcs_to_observer(wcs_in):
    '''Get observer from WCS frame information'''
    if not isinstance(target_wcs, WCS):
        wcs_in = WCS(wcs_in)
    try:
        return sunpy.coordinates.wcs_utils.reference_coordinate_from_frame(sunpy.coordinates.wcs_utils.solar_wcs_frame_mapping(wcs_in))
    except NameError: #different sunpy version
        return SkyCoord(0*u.arcsec,0*u.arcsec,frame=sunpy.coordinates.wcs_utils.solar_wcs_frame_mapping(wcs_in))

def get_limbcoords(aia_map):
    """"""
    r = aia_map.rsun_obs - 1 * u.arcsec  # remove one arcsec so it's on disk.
    # Adjust the following range if you only want to plot on STEREO_A
    th = np.linspace(-180 * u.deg, 0 * u.deg)
    x = r * np.sin(th)
    y = r * np.cos(th)
    limbcoords = SkyCoord(x, y, frame=aia_map.coordinate_frame)
    return limbcoords

def draw_circle(sf,center,r,xoff=0,yoff=0):
    """ """
    r = r* u.arcsec  # remove one arcsec so it's on disk.
    # Adjust the following range if you only want to plot on STEREO_A
    th = np.linspace(-180 * u.deg, 180 * u.deg)
    x = (r * np.sin(th))+(center.Tx+xoff*u.arcsec)
    y = (r * np.cos(th))+(center.Ty+yoff*u.arcsec)
    ccoords = SkyCoord(x, y, frame=sf.coordinate_frame)
    return ccoords

# from make_timeseries_plot_flux.py
def mask_disk(submap,limbcoords=False,plot=False,fac=50.,filled=False,greater=False):
    """ """
    x, y = np.meshgrid(*[np.arange(v.value) for v in submap.dimensions]) * u.pixel
    if limbcoords:
        r=limbcoords.to_pixel(submap.wcs)
    else:
        hpc_coords = submap.pixel_to_world(x, y)
        r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / (submap.rsun_obs+fac*u.arcsec)
    if not greater:
        mask = ma.masked_less_equal(r, 1)
    else:
        mask = ma.masked_greater_equal(r, 1)

    if filled:
        data_arr=np.ma.masked_array(submap.data,mask=mask.mask)
        filled_data=ma.filled(data_arr,0)
        scaled_map = sunpy.map.Map(filled_data, submap.meta)
    else:
        scaled_map = sunpy.map.Map(submap.data, submap.meta, mask=mask.mask)
        palette = submap.plot_settings['cmap']
        palette.set_bad('black')
    if plot:
        fig = plt.figure()
        ax=fig.add_subplot(projection=scaled_map)
        scaled_map.plot(cmap=palette,axes=ax)
        if limbcoords:
            ax.plot_coord(limbcoords, color='w')
        else:
            scaled_map.draw_limb(axes=ax)
        fig.show()
    return scaled_map

def get_corresponding_maps(mtimes,fhead='AIA/AIA20200912_2*_0335.fits'):
    ffiles=glob.glob(fhead)
    ftimes=[dt.strptime(f[-25:-10],'%Y%m%d_%H%M%S')  for f in ffiles]
    matchfiles=[]
    for m in mtimes:
        #find closest t in tvec to given ttag
        closest_t= min(ftimes, key=lambda x: abs(x - m))
        ct_idx=ftimes.index(closest_t)
        closest_file=ffiles[ct_idx]
        matchfiles.append(closest_file)
    matchmaps=[sunpy.map.Map(aa) for aa in matchfiles]
    return matchfiles,matchmaps


def fits2mapIDL_old(files,coreg=True):
    '''run fits2map in idl. List of files should be: AIA094.fits, AIA171.fits,AIA211.fits'''
    idl = pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
    idl('files',files)
    idl('fits2map,files,maps')
    if coreg:
        idl('refmap=maps[0]')
        idl('c171=coreg_map(maps[1],refmap)')
        idl('c211=coreg_map(maps[2],refmap)')
        coreg_map171= 'AIA171_coreg94_' + files[1][8:-5]#assuming AIA files are in the usual level 1.5 naming convention
        coreg_map211= 'AIA211_coreg94_' +files[2][8:-5]
        idl('coreg_map171',coreg_map171)
        idl('coreg_map211',coreg_map211)
        idl('map2fits,c171,coreg_map171')
        idl('map2fits,c211,coreg_map211')
        
def fits2mapIDL(files, reffile,coreg=True):
    '''generalization of old fits2mapIDL to work with any number of input files'''
    idl = pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
    idl('files',files)
    idl('fits2map,files,maps')
    idl('reffile',reffile)
    idl('fits2map,reffile,refmap')
    if coreg:
        for i,f in enumerate(files):
            outfilename=f[:-4]+'_coreg.fits'
            idl('i',i)
            idl('outfilename',outfilename)
            idl('coregmap=coreg_map(maps[i],refmap)')
            idl('map2fits,coregmap,outfilename') #would be nice if the meta was inheirited... in sunpy just reads as genericMap not AIAmap

def coreg_maps_IDL(fits1,fits2,coreg=True,coreg_mapname='coreg_map.fits'):
    '''run fits2map in idl. List of files should be: AIA094.fits, AIA171.fits,AIA211.fits'''
    idl = pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
    idl('fits1',fits1)
    idl('fits2',fits2)
    idl('fits2map,fits1,map1')
    idl('fits2map,fits2,map2')
    if coreg:
        idl('refmap=map1')
        idl('coreg_map=coreg_map(map2,refmap)')
        #idl('help,/st,coreg_map')
        #print(coreg_mapname)
        idl('coreg_mapname',coreg_mapname)
        #idl('print,coreg_mapname')
        idl('map2fits,coreg_map,coreg_mapname')
        
def make_submaps(picklename,bl,tr):
    smaps=[]
    amaps=pickle.load(open(picklename,'rb'))
    for t in amaps:
        bottom_left = SkyCoord(bl[0] * u.arcsec, bl[1] * u.arcsec, frame=t.coordinate_frame)
        top_right = SkyCoord( tr[0]* u.arcsec, tr[1] * u.arcsec, frame=t.coordinate_frame)
        smaps.append(t.submap(bottom_left,top_right))
    return smaps

def int_image(filelist, nint=15,outname='AIA_Fe18_2020-09-12_',how='mean'):
    if type(filelist) == str: #it's a pickle of a list of maps...
        filelist=pickle.load(open(filelist,'rb'))
    #make sure these are sorted by date!
    dates=[m.meta['date-obs'] for m in filelist]
    zipped_lists = zip(dates, filelist)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    _,filelist = [ list(tuple) for tuple in tuples]
    for i,f in enumerate(filelist):
        st=i*nint
        nd=(i+1)*nint
        try:
            foo=filelist[nd]
        except IndexError:
            print(nd,' out of index!')
            break #exits the loop?
        mdata=[]
        mapname=outname+str(i).zfill(2)+'.fits'
        for f in filelist[st:nd]:
            if type(f) == str:
                m=sunpy.map.Map(f)
            else:
                m=f
            if m.meta['exptime'] != 0.0:
                mdata.append(m.data)
        if how == 'mean':
            intdata=np.mean(mdata,axis=0)
        elif how == 'sum':
            intdata=np.sum(mdata,axis=0)
        map_int=sunpy.map.Map(intdata,m.meta)
        map_int.save(mapname)
        print("Integrated image saved in: %s", mapname)
        
def int_map(filelist, nint='all',how='mean'):
    '''time-integrate more than one map'''
    #make sure these are sorted by date!
    dates=[m.meta['date-obs'] for m in filelist]
    zipped_lists = zip(dates, filelist)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    _,filelist = [ list(tuple) for tuple in tuples]
    maps_int=[]
    for i,f in enumerate(filelist):
        if nint !='all':
            st=i*nint
            nd=(i+1)*nint
        else:
            st=0
            nd=-1
        try:
            foo=filelist[nd]
        except IndexError:
            print(nd,' out of index!')
            break #exits the loop?
        mdata=[]
        for f in filelist[st:nd]:
            if type(f) == str:
                m=sunpy.map.Map(f)
            else:
                m=f
            if m.meta['exptime'] != 0.0:
                mdata.append(m.data)
        if how == 'mean':
            intdata=np.mean(mdata,axis=0)
        elif how == 'sum':
            intdata=np.sum(mdata,axis=0)
        map_int=sunpy.map.Map(intdata,m.meta)
        maps_int.append(map_int)
    return maps_int
        
def bin_single_image(im,n=2):
    ''' spatially bin image in n x n bins, crop array if it doesnt fit.'''
    #REWRITE!!
    g6_binned=[]
    if type(gdat) == np.ndarray:
        gprep=[gdat[:,:,i] for i in range(6)]
    else:
        gprep=gdat
    for g in gprep:
        #gdat=sunpy.map.Map(g).data
        if type(n) == int:
            gbin=downscale_local_mean(g, (n,n),clip=True)
            g6_binned.append(gbin)
            g6_arr=np.array(g6_binned)
        elif n =='all':
            g6_binned.append(np.nanmean(g))
            g6_arr=np.array(g6_binned)
    return g6_arr #but make it an array...

#difference maps
def diff_maps(mlist,mask_zeros=False):
    mdifflist=[]
    m0=mlist[0].data
    for m in mlist[1:]:
        try:
            diff=m.data-m0
        except ValueError: #size mismatch!
            m1s=m.data.shape
            m0s=m0.shape
            if m1s[0] > m0s[0]:
                mdata=m.data[1:][:]
            m.data.reshape(m0.shape)
            diff=m.data-m0
        if mask_zeros:
            diff=np.ma.masked_equal(diff,0)
            map_diff=sunpy.map.Map(diff,m.meta,mask=diff.mask)
        else:
            map_diff=sunpy.map.Map(diff,m.meta)
        mdifflist.append(map_diff)
    return mdifflist
    
def integrated_difference(w, aiamaps, bl, tr, timerange1,timerange2, tag=None):
    mlistp= [m.submap(bl,tr) for m in aiamaps if timerange1[0] <= dt.strptime(m.meta['date-obs'],'%Y-%m-%dT%H:%M:%S.%f') <=timerange1[1]]#list of maps in the correct time range
    mlistk= [m.submap(bl,tr) for m in aiamaps if timerange2[0] <= dt.strptime(m.meta['date-obs'],'%Y-%m-%dT%H:%M:%S.%f') <=timerange2[1]]#list of maps
    int_image(mlistp,nint=len(mlistp)-1,outname='AIA/AIA_'+str(w)+'_preflare_'+tag) #sorts the maps
    int_image(mlistk,nint=len(mlistk)-1,outname='AIA/AIA_'+str(w)+'_flare_'+tag) #sorts the maps
    mpre=sunpy.map.Map('AIA/AIA_'+str(w)+'_preflare_'+tag+'00.fits')
    mpeak=sunpy.map.Map('AIA/AIA_'+str(w)+'_flare_'+tag+'00.fits')
    mdiff=diff_maps([mpre,mpeak])[0]
    return mdiff
    
def make_aia_mask(preflare_tr, flare_tr, flare_box, qs_box, tag=None,sigma=3,plot=True):
    '''mask brightening and dimming pixels '''
    if not tag:
        tag=""
    #integrate whole image, then do the box (de-rotation might need to be done tho, test)
    mref=pickle.load(open('AIA/AIA335maps.p','rb'))[0]#
    fbl=SkyCoord(flare_box[0][0]*u.arcsec,flare_box[0][1]*u.arcsec,frame=mref.coordinate_frame)
    ftr=SkyCoord(flare_box[1][0]*u.arcsec,flare_box[1][1]*u.arcsec,frame=mref.coordinate_frame)
    qbl=SkyCoord(qs_box[0][0]*u.arcsec,qs_box[0][1]*u.arcsec,frame=mref.coordinate_frame)
    qtr=SkyCoord(qs_box[1][0]*u.arcsec,qs_box[1][1]*u.arcsec,frame=mref.coordinate_frame)
    if type(preflare_tr[0]) != dt:
        pfs=dt.strptime(preflare_tr[0],'%Y%m%d_%H%M%S')
        pfe=dt.strptime(preflare_tr[1],'%Y%m%d_%H%M%S')
    else:
        pfs,pfe=preflare_tr
    #peak time range
    if type(flare_tr[0]) != dt:
        kfs=dt.strptime(flare_tr[0],'%Y%m%d_%H%M%S')
        kfe=dt.strptime(flare_tr[1],'%Y%m%d_%H%M%S')
    else:
        kfs,kfe=preflare_tr

    qsd,fsd=[],[]
    masked_maps=[]
    mask_plus,mask_minus=[],[]
    masks=[]
    mdiffs,qdiffs=[],[]
    for w in [94,131,171,193,211,335]:
        print(w)
        aiamaps=pickle.load(open('AIA/AIA'+"{:03d}".format(w)+'maps'+tag+'.p','rb')) #need the 0 in 094...
        try:
            mdiff=diff_maps([sunpy.map.Map('AIA/AIA_'+str(w)+'_preflare_'+tag+'00.fits'),sunpy.map.Map('AIA/AIA_'+str(w)+'_flare_'+tag+'00.fits')])[0]
            qdiff=diff_maps([sunpy.map.Map('AIA/AIA_'+str(w)+'_preflare_'+tag+'qs00.fits'),sunpy.map.Map('AIA/AIA_'+str(w)+'_flare_'+tag+'qs00.fits')])[0]
        except ValueError:
            mdiff=integrated_difference(w, aiamaps, fbl, ftr,(pfs,pfe), (kfs,kfe), tag=tag)
            qdiff=integrated_difference(w, aiamaps, qbl, qtr, (pfs,pfe), (kfs,kfe), tag=tag+'qs')
        mdiffs.append(mdiff)
        qdiffs.append(qdiff)
        #print('QS mean diff: ',np.mean(sm.data))
        t=np.std(qdiff.data)
        qsd.append(t)
        print('QS stddev: ',t)
        #print('flare mean diff: ',np.mean(fm.data))
        print('flare stddev: ',np.std(mdiff.data))
        fsd.append(np.std(mdiff.data))
            
        mask_total=np.ma.masked_inside(mdiff.data,-sigma*t,sigma*t)
        mask_plus.append(np.ma.masked_less(mdiff.data,sigma*t).mask)#what about - values? need to do another mask...
        mask_minus.append(np.ma.masked_greater(mdiff.data,-sigma*t).mask) #what about - values? need to do another mask...
        #mask_total=np.logical_and(marr_plus,marr_minus)
        #print(mask_total)#np.shape(marr_plus),np.shape(marr_minus),np.shape(mask_total),np.sum(mask_total))
        #masked_data=np.ma.masked_array(m.data,mask=mask_total)
        #print(np.min(mask_total),np.max(mask_total))
        mmap=sunpy.map.Map(mask_total,mdiff.meta)
        masked_maps.append(mmap)
        masks.append(mask_total.mask)

    if plot:
        fig,ax=plt.subplots(5,6,figsize=[14,10])
        for i in range(6):
            ax[0][i%6].imshow(mdiffs[i].data,cmap=cm.Greys_r, origin='lower left')
            ax[1][i%6].imshow(qdiffs[i].data,cmap=cm.Greys_r, origin='lower left')
            ax[2][i%6].imshow(masks[i],cmap=cm.Blues, origin='lower left')
            ax[0][i%6].set_title(masked_maps[i].meta['wavelnth'])
            ax[3][i%6].imshow(mask_minus[i],cmap=cm.Greens, origin='lower left')
            #ax[1][i%6].set_title(masked_maps[i].meta['wavelnth'])
            ax[4][i%6].imshow(mask_plus[i],cmap=cm.Reds, origin='lower left')
            #ax[2][i%6].set_title(masked_maps[i].meta['wavelnth'])
        fig.show()

    return masks, mask_plus,mask_minus
    
def make_contour_mask(wavelength,submap=False,tag=None,contour=[90],plot=False, diff=True):
    '''make mask from to n% brightest pixels in selected wavelength image '''
    from skimage.draw import polygon
    if not tag:
        tag=""
    #load pre-flare and peak-flare images
    #peakmap=sunpy.map.Map('AIA/AIA_'+str(wavelength)+'_flare_'+tag+'00.fits') #this is the appropriate submap
    try:
        if diff:
            kmap=sunpy.map.Map('AIA/AIA_'+str(wavelength)+'_flare_'+tag+'00.fits')
            pmap=sunpy.map.Map('AIA/AIA_'+str(wavelength)+'_preflare_'+tag+'00.fits')
            mdiff=diff_maps([kmap,pmap])[0]
        else:
            mdiff=sunpy.map.Map('AIA/AIA_'+str(wavelength)+'_flare_'+tag+'00.fits')
    except ValueError:
        print('Could not make difference image, using peak flare image only')
        try:
            mdiff=sunpy.map.Map('AIA/AIA_'+str(wavelength)+'_flare_'+tag+'00.fits') #this is the appropriate submap
        except ValueError:
            mdiff=submap
    if submap:
        if type(submap) == list:
            mdiff=mdiff.submap(submap[0],submap[1])

    mask=find_centroid_from_map(mdiff,levels=contour,show=plot,method='skimage',return_as_mask=True)
    #rr,cc=polygon(contour.allsegs[0][0][:,0],contour.allsegs[0][0][:,1])
    #mask=np.zeros(mdiff.data.T.shape)
    #mask[rr,cc]=1
    return mask #~mask.T.astype(bool)

def fix_units(map_in):
    map_in.meta['cunit1']= 'arcsec'
    map_in.meta['cunit2']= 'arcsec'
    try:
        map_in.meta['date_obs']= dt.strftime(dt.strptime(map_in.meta['date_obs'],'%d-%b-%Y %H:%M:%S.%f'),'%Y-%m-%dT%H:%M:%S.%f')#fix the time string
        map_in.meta['date-obs']= dt.strftime(dt.strptime(map_in.meta['date-obs'],'%d-%b-%Y %H:%M:%S.%f'),'%Y-%m-%dT%H:%M:%S.%f')#fix the time string
    except ValueError: #grrrr
        print(map_in.meta['date_obs'],map_in.meta['date-obs'])
            #map_in.meta['date-obs']= dt.strftime(dt.strptime(map_in.meta['date-obs'],'%d-%m-%YT%H:%M:%S.%f'),'%Y-%m-%dT%H:%M:%S.%f')#fix the time string
    return map_in
    
def hand_coalign(map_in,crpix1_off,crpix2_off):
    map_in.meta['crpix1']=map_in.meta['crpix1']+ crpix1_off
    map_in.meta['crpix2']=map_in.meta['crpix2']+ crpix2_off
    return map_in
    
def scale_skycoord(coord,sf):
    '''scale SkyCoord x and y by given scaling factor. can't modify object in-place (boo astropy) so have to make a new coord based off the old one instead...'''
    cx=sf*coord.Tx
    cy=sf*coord.Ty
    newcoord=SkyCoord(cx,cy,frame=coord.frame)
    return newcoord

#get actual contour centroids...from find_hessi_centroids.py
def center_of_mass(X):
    # calculate center of mass of a closed polygon
    x = X[:,0]
    y = X[:,1]
    g = (x[:-1]*y[1:] - x[1:]*y[:-1])
    A = 0.5*g.sum()
    cx = ((x[:-1] + x[1:])*g).sum()
    cy = ((y[:-1] + y[1:])*g).sum()
    return 1./(6*A)*np.array([cx,cy])
    
def skimage_contour(map,level=90):
    ''' use skimage find_contour instead so I can avoid plotting if need be'''
    contours=find_contours(map.data,(level/100.)*np.max(map.data))
    # Select the largest contiguous contour
    largest_contour = sorted(contours, key=lambda x: len(x))[-1]
    return largest_contour #can do same polygon operations on this now

def find_centroid_from_map(m,levels=[90],idx=0,show=False, return_as_mask=False,method='skimage',transpose=True):
    cs,hpj_cs=[],[]
    ll=np.max(m.data)*np.array(levels)
    
    if method == 'skimage':
        largest_contour=skimage_contour(m,levels[0])
    
    else: #use matplotlib
        fig,ax=plt.subplots()
        ax.imshow(m.data,alpha=.75,cmap=m.plot_settings['cmap'])
        contour=m.draw_contours(levels=levels*u.percent,axes=ax,frame=m.coordinate_frame)
        #print(len(contour.allsegs[-1]))
        largest_contour = sorted(contour.allsegs, key=lambda x: len(x))[-1][0]
        #largest_contour=contour.allsegs[-1][0]
    c =  center_of_mass(largest_contour)
    cs.append(c)
    hpj=m.pixel_to_world(c[1]*u.pixel,c[0]*u.pixel) #does it have to be 0 first then 1 if using matplotlib?
    hpj_cs.append(hpj)
    if show:
        fig,ax=plt.subplots()
        ax.imshow(m.data,alpha=.75,cmap=m.plot_settings['cmap'])
        ax.plot(c[1],c[0], marker="o", markersize=12, color="red")
        ax.plot(largest_contour[:,1],largest_contour[:,0])
        fig.show()
    if return_as_mask:
        from skimage.draw import polygon
        rr,cc=polygon(largest_contour[:,0],largest_contour[:,1])
        #rr,cc=polygon(contour.allsegs[0][0][:,0],contour.allsegs[0][0][:,1])
        if not transpose:
            print("not transposed")
            mask=np.zeros(m.data.shape) #remove .T
            mask[rr,cc]=1
            mout=~mask.astype(bool)
        else: #for legacy sake, keep transpose=True as the default... check if necessary in all cases!
            mask=np.zeros(m.data.T.shape)
            mask[rr,cc]=1
            mout=~mask.T.astype(bool)
        return mout
    else:
        return cs,hpj_cs,largest_contour
        
#def largest_contour_mask_from_map(map_in,contour=[90],show=False):
#    from skimage.draw import polygon
#    cs,hpj_cs,contour=find_centroid_from_map(map_in,levels=contour,show=show)
#    rr,cc=polygon(contour.allsegs[0][0][:,0],contour.allsegs[0][0][:,1])
#    mask=np.zeros(map_in.data.T.shape)
#    mask[rr,cc]=1
#    return ~mask.T.astype(bool)
        
def str_to_SkyCoord(in_str): #for when restoring from json etc
    clist=in_str[in_str.rfind('(')+1:-2].split(',')
    #print(clist)
    Tx=float(clist[0])*u.arcsec
    Ty=float(clist[1])*u.arcsec
    framestr=in_str[in_str.find('(')+1:in_str.find('observer=')]
    fr_name=framestr[:framestr.find(':')].lower().replace(' ','_')
    obstime=framestr[framestr.find('=')+1:framestr.find(',')]
    rsun=float(framestr[framestr.find('rsun=')+5:framestr.find(', obs')-3])*u.km
    obstr=in_str[in_str.find('observer=')+10:in_str.find(')>)')]
    observer=obstr[:obstr.find('(')]
    obscoords=obstr[obstr.rfind('(')+2:].split(',')
    #print(obscoords)
    lon=float(obscoords[0])*u.deg
    lat=float(obscoords[1])*u.deg
    radius=float(obscoords[2])*u.m
    obsloc=SkyCoord(lon,lat,radius,frame=fr_name,obstime=obstime)
    sc_out=SkyCoord(Tx,Ty,frame=obsloc.frame)
    #sc_out.observer.lat=lat
    #sc_out.observer.radius=radius

    return sc_out

def derot_coord(coord, to_time):
    coord_rot=solar_rotate_coordinate(coord, new_observer_time=to_time)
    return coord_rot_all

def plot_and_save(mlist,subcoords=False,outname='STEREO_orbit8_',creverse=True,vrange=False,box=False,diff=False, use_sobel=False):
    '''assume these are de-rotated difference images. bcoords is circle coords'''
    map0=mlist[0]
    palette_name=map0.plot_settings['cmap']
    if creverse and not palette_name.name.endswith('_r'):
        new_cdata=cm.revcmap(palette_name._segmentdata)
    else:
        new_cdata=palette_name._segmentdata
    new_cmap=matplotlib.colors.LinearSegmentedColormap(palette_name.name+'_r',new_cdata)

    if diff:
        mlist=diff_maps(mlist)
        new_cmap='Greys_r'
    if use_sobel:
        #use sobel filter
        nm=[sunpy.map.Map(sobel(m.data),m.meta) for m in mlist]
        mlist=nm

    #new_cmap='Greys_r'
    for i,s in enumerate(mlist):
        fig,ax=plt.subplots(figsize=[6,6])
        #ax=fig.add_subplot(111,projection=s.wcs)
        #if creverse and not palette_name.name.endswith('_r'):
        #    s.plot_settings['cmap']=new_cmap
        if subcoords != False:
            s=s.submap(subcoords[i][0],subcoords[i][1])
            #print(subcoords[i])
        if vrange:
            s.plot_settings['norm']=matplotlib.colors.Normalize(vmin=vrange[0],vmax=vrange[1])
        s.plot(axes=ax,cmap=new_cmap)
        if box:
            bl1=SkyCoord(np.min(box.Tx),np.min(box.Ty),frame=s.coordinate_frame)
            w1=np.max(box.Tx)- np.min(box.Tx)
            h1=np.max(box.Ty)- np.min(box.Ty)
            s.draw_rectangle(bl1, w1,h1,axes=ax,color='m')
        plt.colorbar()
        fig.savefig(outname+str(i).zfill(2)+'.png')
    #return submaps

def get_average_px_size_arcsec(smap):
    ''' what it sounds like, using data size and bottom_left, top_right. returns value in arcsec/px'''
    bl=smap.bottom_left_coord
    tr=smap.top_right_coord
    height=tr.Ty-bl.Ty #arcsec
    width=tr.Tx-bl.Tx #arcsec
    ny,nx=smap.data.shape #x-axis is last
    return(np.mean([height.value/ny,width.value/nx]))
    
def arcsec_to_cm(arcsec,earthsun=False):
    ''' convert arcseconds to cm. return Quantity. angular_diameter_arcsec=radians_to_arcseconds * diameter/distance'''
    #deg_to_m=rsun.value*(np.pi) #rsun is in meters...
    if not earthsun:
        earthsun=u.Quantity(1.5049824e11,u.m)
    sigmaD=arcsec*earthsun.value
    rad2arcsec=(1.*u.rad).to(u.arcsec)
    mout=sigmaD/rad2arcsec *u.m #s= r*theta
    return mout.to(u.cm)
    
def cm_to_arcsec(cm,earthsun=False):
    ''' convert cm to arcseconds'''
    #deg_to_m=rsun.value*(np.pi) #rsun is in meters...
    if not earthsun:
        earthsun=u.Quantity(1.5049824e11,u.m)
    if type(cm) != u.Quantity:
        cm=u.Quantity(cm,u.cm)
    mmeters=cm.to(u.m)
    rad2arcsec=(1.*u.rad).to(u.arcsec)
    sigmaD = mmeters*rad2arcsec #s= r*theta
    arcsec= sigmaD/earthsun
    return arcsec#.to(u.arcsec)

def get_circle_bltr(circle):
    ''' circle is list of skycoords of a circle'''
    xmin=np.min([x.Tx.value for x in circle])
    xmax=np.max([x.Tx.value for x in circle])
    ymin=np.min([x.Ty.value for x in circle])
    ymax=np.max([x.Ty.value for x in circle])
    bl=SkyCoord(xmin*u.arcsec,ymin*u.arcsec,frame=circle.frame)
    tr=SkyCoord(xmax*u.arcsec,ymax*u.arcsec,frame=circle.frame)
    return bl,tr
    
def hpc_scale(hpc_coord,observer):
    '''scale arcsconds to degrees where the limbs and poles are at +- 90'''
    try:
        rsun_arcsec=solar_angular_radius(observer) #note this should ONLY be done on the observer!
    except ValueError:
        rsun_arcsec=cm_to_arcsec(hpc_coord.rsun.to(u.cm))
    rsun_deg=rsun_arcsec.to(u.deg)
    hpc_lon=(hpc_coord.Tx/rsun_arcsec).value*90. #wrong! this is a square with sides at 90 degrees. need a circle
    hpc_lat=(hpc_coord.Ty/rsun_arcsec).value*90.
    return hpc_lon,hpc_lat,rsun_arcsec
    
def hpc_scale_inputs(hpc_x,hpc_y,rsun_arcsec):
    '''assuming correct units: everything in arcsec '''
    if type(rsun_arcsec) != float: #assume it's got units
        rsun_arcsec=rsun_arcsec.value
    hpc_lon=(hpc_x/rsun_arcsec)*90. #wrong! this is a square with sides at 90 degrees. need a circle
    hpc_lat=(hpc_y/rsun_arcsec)*90.
    return hpc_lon,hpc_lat

def hpc_to_hpr(hpc_coord,observer, disk_angle_90=True):
    '''convert from Cartesian to polar coordinates on the solar disk
    
    From Thompson:
    
     From the above, one can derive the conversions between helioprojective-cartesian and helioprojective-radial:
     􏰊􏰁􏰋
     θρ =arg(cosθycosθx, sqrt(cos^2(θy) sin^2(θx) + sin^2(θy))) , 􏰈
     
     ψ = arg(sinθy,−cosθy sinθx) ,
     
     d = d,
     
     See Figure 3
     
     '''
    try:
        rsun_arcsec=solar_angular_radius(observer) #note this should ONLY be done on the observer!
    except ValueError:
        rsun_arcsec=cm_to_arcsec(coord.rsun.to(u.cm))
    rsun_deg=rsun_arcsec.to(u.deg)
    #print(rsun_arcsec)
    thetax=hpc_coord.Tx.to(u.deg) #arcsec - convert to degrees for numpy
    thetay=hpc_coord.Ty.to(u.deg)
    
    hpr_rho=cmath.phase(complex(np.cos(thetay)*np.cos(thetax), np.sqrt(np.cos(thetay)**2 * np.sin(thetax)**2 + np.sin(thetay)**2)))*u.deg
    hpr_phi= cmath.phase(complex(np.sin(thetay), -1*np.cos(thetay)*np.sin(thetax)))*u.deg
    
    #hpc_lon=(hpc_coord.Tx/rsun_arcsec).value*90. #wrong! this is a square with sides at 90 degrees. need a circle
    #hpc_lat=(hpc_coord.Ty/rsun_arcsec).value*90.
    
    if disk_angle_90: #angular size of half disk is 90 degrees
        return (hpr_rho/rsun_deg)*90,(hpr_phi/rsun_deg)*90
    else:
        return hpr_rho,hpr_phi
        
def within_r_of_center(r,hpc_coord,rsun_arcsec=False):
    '''is the input coordinate within a circle of radius r arcsec from the solar center  '''
    if not rsun_arcsec:
        rsun_arcsec=953.489*u.arcsec
    if type(hpc_coord) == tuple or type(hpc_coord) == list: #not a skycoord, assume arcsec
        rho=np.sqrt(hpc_coord[0]**2 + hpc_coord[1]**2)*u.arcsec
    else:
        rho=np.sqrt(hpc_coord.Tx**2 + hpc_coord.Ty**2)
    if rho + r*u.arcsec <= rsun_arcsec:
        return True
    else:
        return False

def hpr_to_hpc(hpr_coord,observer):
    '''convert from radial to Cartesiancoordinates on the solar disk
    
    θx =arg(cosθρ,−sinθρsinψ) ,
    􏰈􏰉
    θy = sin^−1(sinθρ cosψ)
    
    d = d.
    
    See Figure 2
    
    '''
    #try:
    #    rsun_arcsec=solar_angular_radius(observer) #note this should ONLY be done on the observer!
    #except ValueError:
    #    rsun_arcsec=cm_to_arcsec(coord.rsun.to(u.cm))
    #rsun_deg=rsun_arcsec.to(u.deg)
    #print(rsun_arcsec)
    lon=hpr_coord.lon #arcsec - convert to rad
    thetay=hpc_coord.Ty.to(u.rad)

    hpc_lon=cmath.phase(complex(np.cos(thetay)*np.cos(thetax), np.sqrt(np.cos(thetay)**2 * np.sin(thetax)**2 + np.sin(thetay)**2)))*u.deg
    hpc_lat= cmath.phase(complex(np.sin(thetay), -1*np.cos(thetay)*np.sin(thetax)))

    return hpc_lon,hpc_lat,rsun_deg

#### formerly flux_in_boxes.py #########

def flux_from_box(smap,bottom_left,top_right,mask=False, maps=False, expnorm=False, how=np.ma.mean):
    ssmap=smap.submap(bottom_left,top_right)
    data=ssmap.data
    flux=np.ma.sum(data)
    if type(mask) == np.ndarray:
        print('flux_from_box ', np.shape(data),np.shape(mask))
        data=np.ma.masked_array(data,mask=mask) #data.T?
        if expnorm:
            data=data/ssmap.meta['exptime']
        #if how == 'mean':
        flux=how(data) #can I do nanmean too? or is this automatic in ma.mean? i think so...
        #else:
        #    flux=np.ma.sum(data)
    if maps:
        newmap=sunpy.map.Map(data,ssmap.meta)
        #newmap.peek()
        return flux,data,newmap
    return flux,data

def derot_box(bl,tr,start_time, mlist):
    bl_rot_all=[bl]
    tr_rot_all=[tr]
    for i,s in enumerate(mlist[1:]):
        end_time = s.meta['date-obs']
        bl_rot=solar_rotate_coordinate(bl, new_observer_time=end_time,new_observer_location=s.observer_coordinate)
        tr_rot=solar_rotate_coordinate(tr, new_observer_time=end_time,new_observer_location=s.observer_coordinate)
        bl_rot_all.append(bl_rot)
        tr_rot_all.append(tr_rot)
    return bl_rot_all,tr_rot_all

def fit_to_mask(s,bl_rot,tr_rot,mask, mask_data=True, force_mask=False, start_countx=False,start_county=False,how=np.ma.mean):
    nx,ny=np.shape(mask)

    #print(bl_rot)
    #check shape
    ssmap=s.submap(bl_rot,tr_rot)
    data=ssmap.data
    meta=ssmap.meta
    sx,sy=np.shape(data)
    #print('mask shape: ',nx,ny)
    print('before: ',nx,ny, sx,sy)
    if not start_countx:
        countx=0
    else:
        countx=max([start_countx-10,0]) # for some margin
    if not start_county:
        county=0
    else:
        county=max([start_county-10,0]) #just hope this
    if force_mask:
        countmax=county+500
        step=0.01 #not efficient bet whatever
    else:
        countmax=30
        step=0.01
    if sx == nx and sy == ny:
        maskt=mask
        print('no changes to mask or data')
        
    else:
    
        while sx != nx and countx < countmax: #x in array changes Y in SkyCoords
            if sx > nx: #should probably do in a while
                #1 px ~ .6" for AIA
                bl_rot=SkyCoord(bl_rot.Tx,bl_rot.Ty+(step + 1)*u.arcsec,frame=bl_rot.frame)
                tr_rot=SkyCoord(tr_rot.Tx,tr_rot.Ty-(step + 1)*u.arcsec,frame=tr_rot.frame)
            elif sx < nx:
                bl_rot=SkyCoord(bl_rot.Tx,bl_rot.Ty-(step+1)*u.arcsec,frame=bl_rot.frame)
                tr_rot=SkyCoord(tr_rot.Tx,tr_rot.Ty+(step+1)*u.arcsec,frame=tr_rot.frame)
            newdata=s.submap(bl_rot,tr_rot).data
            sx,sy=np.shape(newdata)
            countx+=1
            #print(np.shape(s.submap(bl_rot,tr_rot).data), tr_rot.Tx - bl_rot.Tx)
        while sy !=ny and county < countmax: #y in array changes X in SkyCoords
            if sy > ny:
                bl_rot=SkyCoord(bl_rot.Tx+(step*county+1)*u.arcsec,bl_rot.Ty,frame=bl_rot.frame)
                tr_rot=SkyCoord(tr_rot.Tx-(step*county+1)*u.arcsec,tr_rot.Ty,frame=tr_rot.frame)
            elif sy < ny:
                bl_rot=SkyCoord(bl_rot.Tx-(step*county+1)*u.arcsec,bl_rot.Ty,frame=bl_rot.frame)
                tr_rot=SkyCoord(tr_rot.Tx+(step*county+1)*u.arcsec,tr_rot.Ty,frame=tr_rot.frame)
            newdata=s.submap(bl_rot,tr_rot).data
            sx,sy=np.shape(newdata)
            county+=1
            #print('after: ',sx,sy,tr_rot.Ty - bl_rot.Ty)
        #check for nans!
        if bl_rot.Tx == np.nan or bl_rot.Ty == np.nan or tr_rot.Tx == np.nan or tr_rot.Ty == np.nan:
            print("One of the coordinates is nan!")
        print("new data shape: ", sx,sy,np.shape(newdata), " after ", countx, county," iterations")
        data=newdata
        #print(np.min(data),np.max(data))
        
        maskt=mask
        if not force_mask:
            #what if you need to pad both? grrr
            while sx !=nx: #should be within at least 1 row/col now, trim mask to data size
                try:
                    mask[sx-1]
                    mask2=mask[:sx,:]
                    print('Cropping mask X')
                except IndexError: #sx too big, so pad with True
                    pad=np.ones(ny,dtype=bool)
                    #pad.resize(1,sy)
                    mask2=np.vstack((mask,pad))
                    print('Padding mask X')
                    #print('sx',sx,sy,np.shape(mask2))
                nx,ny=np.shape(mask2)
                maskt=mask2
            while sy !=ny:
                try:
                    mask[:,sy-1]
                    mask2=maskt[:,:sy]
                    #print('sy',sx,sy,np.shape(mask2))
                    print('Cropping mask Y')
                except IndexError: #sx too big, so pad with True
                    pad=np.ones(nx,dtype=bool)
                    pad.resize(1,nx)
                    mask2=np.hstack((maskt,pad.T))
                    print('Padding mask Y')
                    #print('sy',sx,sy,np.shape(mask2))
                nx,ny=np.shape(mask2)
                maskt=mask2
                
    #do flux from box operations here, see if that makes things work
    if mask_data:
        data=np.ma.masked_array(data,mask=maskt) #data.T?
    #if how == 'mean':
    flux=how(np.ma.masked_array(data,mask=maskt)) #always use mask for fluxes, but not always fro data
    #elif how == 'sum':
    #    flux=np.ma.sum(np.ma.masked_array(data,mask=maskt)) #always use mask for fluxes, but not always fro data

    newmap=sunpy.map.Map(data,meta)
    #newmap.peek()
    return flux,data,newmap, countx, county
    #return bl_rot,tr_rot,maskt

def track_region_box(submaps, filenames=False, circle=False, xoff=0.,yoff=0.,start_idx=0,bin=False, mask=False, plot=False, expnorm=False, mask_data=True,force_mask=False,verbose=False,how=np.ma.mean):
    '''add binning option, return timestamps too for greater ease'''
    map0=submaps[0]
    start_time = map0.meta['date-obs']
    timestamps=[]#[start_time]
    
    if circle:
        bl=SkyCoord(np.min(circle.Tx)+xoff*u.arcsec,np.min(circle.Ty)+yoff*u.arcsec,frame=map0.coordinate_frame) #note the +10!
        tr=SkyCoord(np.max(circle.Tx)+xoff*u.arcsec,np.max(circle.Ty)+yoff*u.arcsec,frame=map0.coordinate_frame)
    else: #take it from first map... although then you're gonna get some zeros later on?
        bl=map0.bottom_left_coord
        tr=map0.top_right_coord
        
    bl.set_obstime=start_time
    tr.set_obstime=start_time
    bwidth=tr.Tx-bl.Tx
    bheight=tr.Ty-bl.Ty
    #flux=[flux_from_box(map0,bl,tr,mask=mask)]
    #rotcoords1=[(bl,tr)]
    #data=[map0.submap(bl,tr).data]
    #nx,ny=np.shape(map0.submap(bl,tr).data)
    countx,county=False,False
    
    flux,data,rotcoords1,maps,all_filenames=[],[],[],[],[]
    for i,s in enumerate(submaps):
        if s.meta['exptime'] !=0.0:
            end_time = s.meta['date-obs']
            if verbose:
                print("processing ", s.meta['wavelnth'],"map at time ",end_time)
            timestamps.append(end_time)
            if start_time == end_time: #no need to rotate
                bl_rot=bl
                tr_rot=tr
            else:
                bl_rot=solar_rotate_coordinate(bl, new_observer_time=end_time,new_observer_location=s.observer_coordinate)
                tr_rot=solar_rotate_coordinate(tr, new_observer_time=end_time,new_observer_location=s.observer_coordinate)
            #print(bl_rot, tr_rot)
            if np.isnan(bl_rot.Tx.value): #need to re-define in terms of box width and height
                bl_rot=SkyCoord(tr_rot.Tx - bwidth,tr_rot.Ty - bheight, frame=s.coordinate_frame)

            if type(mask) == np.ndarray:
                fl,dat,current_map, countx, county=fit_to_mask(s,bl_rot, tr_rot,mask,mask_data=mask_data,force_mask=force_mask,start_countx=countx,start_county=county,how=how)
            else:
                fl,dat,current_map=flux_from_box(s,bl_rot,tr_rot,mask=False,maps=True, expnorm=expnorm, how=np.mean)
            #print(nx,ny,sx,sy,np.shape(maskt),np.shape(dat))
            flux.append(fl)
            print(np.min(dat),np.max(dat))
            data.append(dat)
            maps.append(current_map)
            if filenames != False:
                all_filenames.append(filenames[i])
            else:
                all_filenames.append(None)
            rotcoords1.append((bl_rot,tr_rot))

    dout={'timestamps':timestamps,'fluxes':flux,'rotated_coords':rotcoords1,'data':data,'maps':maps,'filenames':all_filenames}
    if bin: #integer number of maps to bin
        newts,newflux,newcoords=[],[],[]
        for j in range(int(len(submaps)/bin)): #+1?
            newts.append(timestamps[j*bin])
            newcoords.append(rotcoords1[j*bin])
            sumflux=np.sum([flux[j*bin+k] for k in range(bin)])
            newflux.append(sumflux)
        dout={'timestamps':newts,'fluxes':newflux,'rotated_coords':newcoords,'data':data,'filenames':all_filenames}
    return dout


def save_boxfig(submaps,rotcoords1,rotcoords2,outname='STEREO_orbit8_',creverse=False,vrange=False):
    palette_name=submaps[0].plot_settings['cmap']
    if creverse and not palette_name.name.endswith('_r'):
        new_cdata=cm.revcmap(palette_name._segmentdata)
        new_cmap=matplotlib.colors.LinearSegmentedColormap(palette_name.name+'_r',new_cdata)
    for i,s in enumerate(submaps):
        fig=plt.figure(figsize=[8,8])
        ax=fig.add_subplot(111,projection=s.wcs)
        if creverse and not palette_name.name.endswith('_r'):
            s.plot_settings['cmap']=new_cmap
        if vrange:
            s.plot_settings['norm']=matplotlib.colors.Normalize(vmin=vrange[0],vmax=vrange[1])
        s.plot(axes=ax,alpha=0.7)
        b1l=rotcoords1[i][0]
        w1=rotcoords1[i][1].Tx-rotcoords1[i][0].Tx
        h1=rotcoords1[i][1].Ty-rotcoords1[i][0].Ty
        b2l=rotcoords2[i][0]
        w2=rotcoords2[i][1].Tx-rotcoords2[i][0].Tx
        h2=rotcoords2[i][1].Ty-rotcoords2[i][0].Ty

        s.draw_rectangle(b1l, w1,h1,axes=ax,color='b')
        s.draw_rectangle(b2l, w2,h2,axes=ax,color='m')
        fig.savefig(outname+str(i).zfill(2)+'.png')
        
