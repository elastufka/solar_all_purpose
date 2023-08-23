 #######################################
#stack_plots.py 
# Erica Lastufka 29/05/2018  

#Description: Make AIA/STEREO EUV stack plots so I can see loops rising and falling
#######################################

#######################################
# Usage:

######################################

import numpy as np
import numpy.ma as ma

import os
import scipy.ndimage as ndi
from scipy.ndimage.filters import generic_filter as gf
import matplotlib.cm as cm
import matplotlib.colors as colors
import glob
from sunpy.net import Fido, attrs as a
import sunpy.map
import sunpy.image.coalignment
from scipy.io import readsav
import astropy.units as u
from astropy.coordinates import SkyCoord
from datetime import datetime as dt
from scipy.interpolate import interp1d
import math
from scipy import stats

#global emcube
#global lgtaxis
#should store all this stuff in common block equivalents...how to deal with the map meta?

def coreg_cube(mapcube):
    sunpy.image.coalignment.mapcube_coalign_by_match_template(mapcube,clip=True)
    return mapcube

def trim_added_maps(basemap, other_maps,bl=False, tr=False,test=False):
    bm=pickle.load(open(basemap,'rb'))
    if not bl:
        bl=bm.bottom_left_coord
    if not tr:
        tr=bm.top_right_coord
    for om in other_maps:
        cm=pickle.load(open(om,'rb'))
        sm=cm.submap(bl,tr)
        if test:
            fig=plt.figure()
            ax1=fig.add_subplot(1,2,1)
            bm.plot(axes=ax1)
            ax2=fig.add_subplot(1,2,2)
            cm.plot(axes=ax2)
            fig.show()
        #print bl,tr
        #print sm.bottom_left_coord
        #print sm.top_right_coord
        pickle.dump(sm, open(om,'wb'))

def get_slices(mapcube,slice_line,idx=0,thick=False,plot=False,full=False,polar=True,fat=True):
    '''slice_line is 2 endpoints, either SkyCoords or not.'''
    from numpy import exp, abs, angle
    sl0=slice_line[0]
    sl1=slice_line[1]
    mapbase=mapcube.maps[idx]
    if type(sl0) != SkyCoord:
        sl0=SkyCoord(sl0[0]*u.arcsec,sl0[1]*u.arcsec, frame=mapbase.coordinate_frame)
        sl1=SkyCoord(sl1[0]*u.arcsec,sl1[1]*u.arcsec, frame=mapbase.coordinate_frame)

    if polar:
        slicemask=[]
        #convert all coords to polar
        #rsun=mapbase.rsun_obs
        x, y = np.meshgrid(*[np.arange(v.value) for v in mapbase.dimensions]) * u.pixel
        zarr = x + 1j * y
        hpc_coords = mapbase.pixel_to_data(x, y) #now need to convert these Tx and Ty to polar
        hpc2z=lambda hpc: hpc.Tx.value + 1j*hpc.Ty.value
        hpcz=hpc2z(hpc_coords)
        z2polar=lambda z: (abs(z),angle(z,deg=True))
        rS, thetaS = z2polar(hpcz)
        r0,th0=z2polar(sl0.Tx + 1j*sl0.Ty)
        rf,thf=z2polar(sl1.Tx + 1j*sl1.Ty)
        if thick:
            thf=thf+thick #in degrees
        #return rS,thetaS #use these as maps for the coordinates
        #get all combinations of r and theta between these values
        rpx=np.where(np.logical_and(np.logical_and(rS >r0.value,rS <rf.value),np.logical_and(thetaS >th0,thetaS <thf)))
        #let's hope this is now the magic answer...
        ypix,xpix=rpx
        for m in mapcube.maps:
            smask=np.ones(np.shape(m.data))
            smask[rpx]=0.0
            slicemask.append(smask)
        #return rS,thetaS,hpc_coords,hpcz,rpx,r0,th0,rf,thf
    else:
        #convert from wcs to pixel - do I need to do this for each map in the cube?
        p0=mapbase.world_to_pixel(sl0)
        p1=mapbase.world_to_pixel(sl1)
        #p0=sl0
        #p1=sl1
        print p0,p1
        pixels=interpolate_pixels_along_line(p0[0].value,p0[1].value,p1[0].value,p1[1].value)
        #ppairs=[sunpy.map.mapbase.PixelPair(p[0],p[1]) for p in pixels]
        if thick != False: #get more pixels
            tpix=[]
            for n in range(0,thick):
                p0n=sunpy.map.mapbase.PixelPair((p0[0].value+n+1)*u.pixel,(p0[1].value)*u.pixel)
                p1n=sunpy.map.mapbase.PixelPair((p1[0].value+n+1)*u.pixel,(p1[1].value)*u.pixel)
                newpixels=interpolate_pixels_along_line(p0n[0].value,p0n[1].value,p1n[0].value,p1n[1].value)
                tpix.append(newpixels)
            
    #plot a preview
    if plot:
        #xpix=[p[0] for p in pixels]
        #ypix=[p[1] for p in pixels]
        wcspix=mapbase.pixel_to_world(xpix*u.pixel,ypix*u.pixel)
        fig=plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=mapbase)
        #ax.scatter(lpix)
        mapbase.plot(axes=ax)
        ax.plot_coord(wcspix,color='k')
        #if thick != False:
        #    for n in range(0,thick):
        #        xpix=[p[0] for p in tpix[n]]
        #        ypix=[p[1] for p in tpix[n]]
        #        wcspix=mapbase.pixel_to_world(xpix*u.pixel,ypix*u.pixel)
        #        ax.plot_coord(wcspix,color='k')
        fig.show()
    #return
    #get all data that lies along these pixels
    slices=[]
    taxis=[]
    for m,s in zip(mapcube.maps,slicemask):
        #just use a masked array...
        mdata=np.ma.masked_array(m.data,mask=s) #average it later
        #if not full:
        if fat:
            for i in range(0,fat):
                taxis.append(dt.strptime(m.meta['date-obs'][:-4],'%Y-%m-%dT%H:%M:%S'))
        else:
            taxis.append(dt.strptime(m.meta['date-obs'][:-4],'%Y-%m-%dT%H:%M:%S'))
                
        #if thick:
        #    odata=[[np.transpose(m.data)[int(px),int(py)] for (px,py) in pixels]]
        #    #if full:
        #    #    slices.append(odata)
        #    #print np.shape(odata)
        #    for tp in range(0,thick):
        #        tdata=[np.transpose(m.data)[int(px),int(py)] for (px,py) in tpix[tp]]
        #        #odata.append(tdata)
        #        #print np.shape(odata)
        #        if full:
        #            #mdata=odata
        #            slices.append(tdata)
        #            taxis.append(dt.strptime(m.meta['date-obs'][:-4],'%Y-%m-%dT%H:%M:%S'))
        #    mdata=np.mean(odata,axis=0)
        #    #print np.shape(mdata)
        #else:
        #    mdata=[m.data[int(px),int(py)] for (px,py) in pixels]
        #    #slices.append(mdata)
        #if not full:    
        #    slices.append(mdata)
        #average over r
        if polar: #average over r bins of .5" = 378.5 kM?
            dpoint,dpoints=[],[]
            #bin the r values
            rbins=np.linspace(r0.value,rf.value,200)
            #find out the indexes where the r-values are in each bin
            bin_idx=np.digitize(rS[rpx],rbins)
            #print np.max(bin_idx)
            #zip the bin indexes to the bin values and the (x,y) tuples
            rpxx,rpxy=rpx
            all_zipped=zip(bin_idx,rS[rpx],rpxx,rpxy)
            all_zipped.sort() #will sort by bin_idx
            for az in all_zipped:
                if az[0] == idx:
                    dpoint.append(mdata[az[2]][az[3]])
                else:
                    idx=az[0]
                    if dpoint !=[]:
                        dpoints.append(np.mean(dpoint))
                    dpoint=[]
        if fat:
            for i in range(0,fat):#make the pixel fatter
                slices.append(dpoints)
        else:
            slices.append(dpoints)
            
    #for m in mapcube.maps:
    return slices, taxis

def get_slices_linear(mapcube,slice_line,idx=0,thick=False,plot=False,full=False,polar=True,fat=True):
    '''slice_line is 2 endpoints, either SkyCoords or not.'''
    from numpy import exp, abs, angle
    sl0=slice_line[0]
    sl1=slice_line[1]
    mapbase=mapcube.maps[idx]
    if type(sl0) != SkyCoord:
        sl0=SkyCoord(sl0[0]*u.arcsec,sl0[1]*u.arcsec, frame=mapbase.coordinate_frame)
        sl1=SkyCoord(sl1[0]*u.arcsec,sl1[1]*u.arcsec, frame=mapbase.coordinate_frame)

    #convert from wcs to pixel - do I need to do this for each map in the cube?
    p0=mapbase.world_to_pixel(sl0)
    p1=mapbase.world_to_pixel(sl1)
    #p0=sl0
    #p1=sl1
    #print p0,p1
    pixels=interpolate_pixels_along_line(p0[0].value,p0[1].value,p1[0].value,p1[1].value)
    #ppairs=[sunpy.map.mapbase.PixelPair(p[0],p[1]) for p in pixels]
    if thick != False: #get more pixels
        tpix=[]
        for n in range(0,thick):
            p0n=sunpy.map.mapbase.PixelPair((p0[0].value+n+1)*u.pixel,(p0[1].value)*u.pixel)
            p1n=sunpy.map.mapbase.PixelPair((p1[0].value+n+1)*u.pixel,(p1[1].value)*u.pixel)
            newpixels=interpolate_pixels_along_line(p0n[0].value,p0n[1].value,p1n[0].value,p1n[1].value)
            tpix.append(newpixels)
            
    #plot a preview
    if plot:
        xpix=[p[0] for p in pixels]
        ypix=[p[1] for p in pixels]
        wcspix=mapbase.pixel_to_world(xpix*u.pixel,ypix*u.pixel)
        fig=plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=mapbase)
        #ax.scatter(lpix)
        mapbase.plot(axes=ax)
        ax.plot_coord(wcspix,color='w')
        if thick != False:
            for n in range(0,thick):
                xpix=[p[0] for p in tpix[n]]
                ypix=[p[1] for p in tpix[n]]
                wcspix=mapbase.pixel_to_world(xpix*u.pixel,ypix*u.pixel)
                ax.plot_coord(wcspix,color='w')
        fig.show()
    #return
    #get all data that lies along these pixels
    slices=[]
    taxis=[]
    for m in mapcube.maps:
        #just use a masked array...
        #mdata=np.ma.masked_array(m.data,mask=slicemask) #average it later
        if not full:
            taxis.append(dt.strptime(m.meta['date-obs'][:-4],'%Y-%m-%dT%H:%M:%S'))
                
        if thick:
            odata=[[np.transpose(m.data)[int(px),int(py)] for (px,py) in pixels]]
            for tp in range(0,thick):
                tdata=[np.transpose(m.data)[int(px),int(py)] for (px,py) in tpix[tp]]
                odata.append(tdata)
                #print np.shape(odata)
                if full:
                    #mdata.append(odata)
                    slices.append(tdata)
                    taxis.append(dt.strptime(m.meta['date-obs'][:-4],'%Y-%m-%dT%H:%M:%S'))
            else:
                mdata=np.mean(odata,axis=0)
                
                #print np.shape(mdata)
        else:
            mdata=[m.data[int(px),int(py)] for (px,py) in pixels]
            slices.append(mdata)
        if not full:    
            slices.append(mdata)
        if fat:
            for i in range(0,fat):
                slices.append(mdata)
                taxis.append(dt.strptime(m.meta['date-obs'][:-4],'%Y-%m-%dT%H:%M:%S'))
            
    #for m in mapcube.maps:
    return slices, taxis


def rename_aia(files):
    for f in files:
        wave=f[-8:-5]
        newf=f[:3]+'_'+wave+'_'+f[4:-10]+'.fits'
        os.rename(f,newf)

def make_mapcubes(waves):#=[094,131,171,193,211,335,304]):
    for w in waves:
        files=glob.glob('AIA_*'+str(w)+'_2013*.fits')
        mapcube=sunpy.map.Map(files,cube=True)
        picklename='mapcube'+str(w)+'a.p'
        pickle.dump(mapcube, open(picklename,'wb'))

def interpolate_pixels_along_line(x0, y0, x1, y1):
    """Uses Xiaolin Wu's line algorithm to interpolate all of the pixels along a
    straight line, given two points (x0, y0) and (x1, y1)

    Wikipedia article containing pseudo code that function was based off of:
        http://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm
    """
    pixels = []
    steep = abs(y1 - y0) > abs(x1 - x0)

    # Ensure that the path to be interpolated is shallow and from left to right
    if steep:
        t = x0
        x0 = y0
        y0 = t

        t = x1
        x1 = y1
        y1 = t

    if x0 > x1:
        t = x0
        x0 = x1
        x1 = t

        t = y0
        y0 = y1
        y1 = t

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx  # slope

    # Get the first given coordinate and add it to the return list
    x_end = round(x0)
    y_end = y0 + (gradient * (x_end - x0))
    xpxl0 = x_end
    ypxl0 = round(y_end)
    if steep:
        pixels.extend([(ypxl0, xpxl0), (ypxl0 + 1, xpxl0)])
    else:
        pixels.extend([(xpxl0, ypxl0), (xpxl0, ypxl0 + 1)])

    interpolated_y = y_end + gradient

    # Get the second given coordinate to give the main loop a range
    x_end = round(x1)
    y_end = y1 + (gradient * (x_end - x1))
    xpxl1 = x_end
    ypxl1 = round(y_end)

    # Loop between the first x coordinate and the second x coordinate, interpolating the y coordinates
    for x in range(int(xpxl0) + 1, int(xpxl1)):
        if steep:
            pixels.extend([(math.floor(interpolated_y), x), (math.floor(interpolated_y) + 1, x)])

        else:
            pixels.extend([(x, math.floor(interpolated_y)), (x, math.floor(interpolated_y) + 1)])

        interpolated_y += gradient

    # Add the second given coordinate to the given list
    if steep:
        pixels.extend([(ypxl1, xpxl1), (ypxl1 + 1, xpxl1)])
    else:
        pixels.extend([(xpxl1, ypxl1), (xpxl1, ypxl1 + 1)])

    return pixels
        
    
def calc_spatial_res(mapbase,testpx1,testpx2):
    wcspix1=mapbase.pixel_to_world(testpx1[0]*u.pixel,testpx1[1]*u.pixel)
    wcspix2=mapbase.pixel_to_world(testpx2[0]*u.pixel,testpx2[1]*u.pixel)
    xres=wcspix1.Tx.value-wcspix2.Tx.value
    yres=wcspix1.Ty.value-wcspix2.Ty.value
    return xres,yres
        
def stack_plot(stack,taxis=False,gauss=False,ylim=False,pix2Mm=1.23,bsub=False,oplotline=False,linelim=[50,50],flip=False,norm=False):
    '''Make the stack plot. Default pix2Mm is for STEREO A: xres=1.622"*757km/" -> 1227.854 km = 1.23 Mm. For AIA, pix2Mm=.454'''
    #make axis labels. default 10 per axis
    #print(xlabels)
    #plot
    if type(bsub) == int: #subtract the first row as 'background'
        brow=np.mean(stack[0:bsub],axis=0)
        newstack=[]
        for row in stack[1:]:
            newstack.append(list(np.array(row) - np.array(brow)))
        if taxis:
            taxis=taxis[1:]
        stack=newstack
    elif type(bsub) == np.ndarray:
        newstack=[]
        for row in stack[1:]:
            newstack.append(list(np.array(row) - np.array(bsub)))
            stack=newstack
        
    if gauss:
        stack=ndi.filters.gaussian_filter(stack,gauss)
    if flip:
        stack=np.fliplr(stack)
    plt.clf()
    fig,ax=plt.subplots()
    if ylim:
        stack=np.array(stack)
        stack=stack[:,ylim[0]:ylim[1]]
    if norm:
        pnorm=colors.Normalize(vmin=norm[0],vmax=norm[1])
        cf1=ax.imshow(np.transpose(stack),origin='lower',norm=pnorm)#specgram(EMmean,NFFT=21,Fs=1.0,bins=20)
    else: 
        cf1=ax.imshow(np.transpose(stack),origin='lower')#specgram(EMmean,NFFT=21,Fs=1.0,bins=20)
    if oplotline:
        xvec=range(linelim[0],np.shape(stack)[0]-linelim[1])
        yvec=oplotline[0]*np.array(xvec)+oplotline[1]
        ax.plot(xvec,yvec,linewidth=2,color='w')
        ax.set_xlim([0,np.shape(stack)[0]])
        ax.set_ylim([0,np.shape(stack)[1]])
        print xvec[0],yvec[0],xvec[-1],yvec[-1]
    #if ylim:
    #    ax.set_ylim(ylim)
    #make axis labels. default 10 per axis
    #ax.set_xlim(0,np.shape(stack)[0])
    #ax.set_ylim(40,160)    
    if taxis:
        fig.canvas.draw()
        cxlabels=ax.get_xticklabels()
        xlabels=[]
        for cx in cxlabels:
            #print cx,cx.get_text()
            if cx.get_text() != '':
                try:
                    xlabels.append("{0}".format(taxis[int(cx.get_text())].time())[:-3]) #need to fix this!
                except IndexError:
                    xlabels.append('')
            else:
                xlabels.append('')
        #print xlabels
        ax.set_xticklabels(xlabels)
    cylabels=ax.get_yticklabels()
    ylabels=[] #eventually want these in units of height above limb
    #Mm2pix=
    for cy in cylabels:
        if cy.get_text() != '':
            ylabels.append(np.round(int(cy.get_text())*pix2Mm)) #have to account for radial binning now too.
            #with STEREO lop line:
            # r=89.5 => eff. pixel size = .4477
        else:
            ylabels.append('')
    
    ax.set_yticklabels(ylabels)
    ax.set_ylabel('Height (Mm)')
    if taxis:
        ax.set_xlabel('Time on '+dt.strftime(taxis[0],'%Y-%b-%d'))    
    fig.colorbar(cf1,fraction=.015,pad=.04) #need to put some physical units on this!
    #myFmt = DateFormatter('%H:%M')
    #ax.xaxis.set_major_formatter(myFmt)

    fig.show()
    #return taxis,lgtaxis,EMmean
    #if bsub:
    return stack

def fit_gradient(stack,xlim, ylim,gauss=False,plot=True,factor=1.):
    '''Make line of best fit to gradient within the given boundaries'''
    if gauss:
        stack=ndi.filters.gaussian_filter(stack,gauss)
    #first get the gradients in the stack
    grad=np.gradient(stack)
    #let's look at the gradient in y:
    grady=grad[1]
    #print np.shape(grady)    
    #trim to fit
    grady=grady[xlim[0]:xlim[1],ylim[0]:ylim[1]]
    #print np.shape(grady)
    sigma=np.std(grady)
    gradmean=np.mean(grady)
    #print sigma,gradmean,np.max(grady),np.min(grady)
    top3sig=np.where(grady < gradmean - factor*sigma)
    #only keep the largest y value for any given x
    #print np.shape(top3sig)
    greatesty=[]
    uniqx=set(top3sig[0])
    for ux in uniqx:
        yvals=top3sig[1][np.where(top3sig[0] == ux)]
        greatesty.append(np.max(yvals))

    #print len(greatesty)
    coords=zip(set(top3sig[0]),greatesty)
    slope,intercept,r_value,p_value,std_err=stats.linregress(list(uniqx),greatesty)
    print slope,intercept
    #plot for fun
    if plot:
        fig,ax=plt.subplots(2)
        ax[0].imshow(np.transpose(grady),origin='lower')
        ax[1].scatter(list(uniqx),greatesty,-10.*grady[coords], c='b')
        xvec=range(0,np.max(list(uniqx)))
        yvec=slope*np.array(xvec)+intercept
        #print np.shape(yvec)
        ax[1].plot(xvec,yvec)
        for a in ax:
            a.set_xlim(xlim)
            a.set_ylim(ylim)
        fig.show()
    return top3sig, -10*grady[top3sig]
    

