import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
#import wcsaxes
from astropy.wcs import WCS

import sunpy.map
import sunpy.coordinates
import sunpy.coordinates.wcs_utils
from sunpy.net import vso
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
from sunpy.physics.differential_rotation import solar_rotate_coordinate, diffrot_map
from skimage.transform import downscale_local_mean
from scipy.ndimage import sobel
import pickle

def get_limbcoords(aia_map):
    r = aia_map.rsun_obs - 1 * u.arcsec  # remove one arcsec so it's on disk.
    # Adjust the following range if you only want to plot on STEREO_A
    th = np.linspace(-180 * u.deg, 0 * u.deg)
    x = r * np.sin(th)
    y = r * np.cos(th)
    limbcoords = SkyCoord(x, y, frame=aia_map.coordinate_frame)
    return limbcoords

def draw_circle(sf,center,r,xoff=0,yoff=0):
    r = r* u.arcsec  # remove one arcsec so it's on disk.
    # Adjust the following range if you only want to plot on STEREO_A
    th = np.linspace(-180 * u.deg, 180 * u.deg)
    x = (r * np.sin(th))+(center.Tx+xoff*u.arcsec)
    y = (r * np.cos(th))+(center.Ty+yoff*u.arcsec)
    ccoords = SkyCoord(x, y, frame=sf.coordinate_frame)
    return ccoords

# from make_timeseries_plot_flux.py
def mask_disk(submap,limbcoords=False,plot=False,fac=50.,filled=False,greater=False):
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

    palette = submap.plot_settings['cmap']
    palette.set_bad('black')
    if filled:
        data_arr=np.ma.masked_array(submap.data,mask=mask.mask)
        filled_data=ma.filled(data_arr,0)
        scaled_map = sunpy.map.Map(filled_data, submap.meta)
    else:
        scaled_map = sunpy.map.Map(submap.data, submap.meta, mask=mask.mask)
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
def diff_maps(mlist):
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
        map_diff=sunpy.map.Map(diff,m.meta)
        mdifflist.append(map_diff)
    return mdifflist

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

def find_centroid_from_map(m,levels=[90],idx=0,show=False, return_as_mask=False):
    cs,hpj_cs=[],[]
    ll=np.max(m.data)*np.array(levels)

    fig,ax=plt.subplots()
    ax.imshow(m.data,alpha=.75,cmap=m.plot_settings['cmap'])
    contour=m.draw_contours(levels=levels*u.percent,axes=ax,frame=m.coordinate_frame)
    print(len(contour.allsegs[-1]))
    c =  center_of_mass(contour.allsegs[-1][idx])
    cs.append(c)
    hpj=m.pixel_to_world(c[0]*u.pixel,c[1]*u.pixel,origin=0)
    hpj_cs.append(hpj)
    ax.plot(c[0],c[1], marker="o", markersize=12, color="red")
    if show:
        fig.show()
    if return_as_mask:
        from skimage.draw import polygon
        rr,cc=polygon(contour.allsegs[0][0][:,0],contour.allsegs[0][0][:,1])
        mask=np.zeros(m.data.T.shape)
        mask[rr,cc]=1
        return ~mask.T.astype(bool)
    else:
        return cs,hpj_cs,contour
        
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
    ny,nx=smaps[0].data.shape #x-axis is last
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
