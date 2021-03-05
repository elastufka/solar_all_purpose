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

from flux_in_boxes import track_region_box

def aia_prep(files,zip_old=True):
    '''run AIA prep on given files, clean up'''
    idl = pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
    idl('files',files)
    idl('aia_prep,files,-1,/do_write_fits,/normalize,/use_ref') #what is input 2? input2 - List of indices of FITS files to read #indgen(size(files,/dim)) <- all files
    #rename files
    prepped_files=glob.glob('AIA2*.fits')
    for pf in prepped_files:
        wave=pf[-8:-5]
        newf=pf[:3]+'_'+wave+'_'+pf[3:-10]+'.fits'
        #print newf
        os.rename(pf,newf)
    if zip_old:
        #archive level 0 data
        zipf = zipfile.ZipFile('AIALevel0.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir(files, zipf)
        zipf.close()

def pickle_maps(bl,tr,outtag='maps'):
    for w in [94,131,171,193,211,335]:
        aiaf=glob.glob('AIA*0%s.fits' % w)
        aiaf.sort()
        aiasmaps=[]
        for f in aiaf:
            t=sunpy.map.Map(f)
            bottom_left = SkyCoord(bl[0] * u.arcsec, bl[1] * u.arcsec, frame=t.coordinate_frame)
            top_right = SkyCoord( tr[0]* u.arcsec, tr[1] * u.arcsec, frame=t.coordinate_frame)
            aiasmaps.append(t.submap(bottom_left,top_right))
        pickle.dump(aiasmaps,open('AIA_'+str(w)+outtag+'.p','wb'))

def zipdir(files, ziph):
    # ziph is zipfile handle
    root=os.getcwd()
    for f in files:
        ziph.write(os.path.join(root, f))

def get_Fe18(map94,map171,map211,submap=False,save2fits=False):
    '''should be run on co-registered maps, see sunpy_map_utils fits2mapIDL() '''
    if type(map94) == str:
        map94=sunpy.map.Map(map94)
    if type(map171) == str:
        map171=sunpy.map.Map(map171) #the coregistered map
    if type(map211) == str:
        map211=sunpy.map.Map(map211) #the coregistered map
    if submap:
        map94=map94.submap(SkyCoord(submap[0]*u.arcsec,submap[1]*u.arcsec,frame=map94.coordinate_frame))
        map171=map171.submap(SkyCoord(submap[0]*u.arcsec,submap[1]*u.arcsec,frame=map171.coordinate_frame))
        map211=map211.submap(SkyCoord(submap[0]*u.arcsec,submap[1]*u.arcsec,frame=map211.coordinate_frame))
    try:
        map18data=map94.data-map211.data/120.-map171.data/450.
    except ValueError: #shapes are different- is 94 always the culprit?
        d94=map94.data[1:,:]
        map18data=d94-map211.data/120.-map171.data/450.
    map18=sunpy.map.Map(map18data,map94.meta)
    #if submap:
    #    map18=map18.submap(SkyCoord(submap[0]*u.arcsec,submap[1]*u.arcsec,frame=map18.coordinate_frame))
    if save2fits:
        filename='AIA_Fe18_'+map18.meta['date-obs']+'.fits'
        map18.save(filename)
    return map18
    
def Fe18_from_groups(start_index,end_index,moviename,submap=[(-1100,-850),(0,400)],framerate=12,imnames='Fe18_'):
    '''generate the Fe18 images from the DEM groups'''
    import make_aligned_movie as mov
    preppedfiles=glob.glob('AIA_*.fits')
    groups=aia.group6(preppedfiles)
    print(len(groups))
    for i,g in enumerate(groups[start_index:end_index]):
        map94=sunpy.map.Map(g[0])
        map171=sunpy.map.Map(g[2])
        map211=sunpy.map.Map(g[4])
        map18=get_Fe18(map94,map171,map211,submap=submap) #save to listl/mapcube? do something
        plt.clf()
        map18.plot()
        plt.savefig(imnames+'{0:03d}'.format(i)+'.png')

    mov.run_ffmpeg(imnames+'%03d.png',moviename,framerate=framerate)
    
def group6(files):
    '''group AIA data into blocks of 6: '''
    #get time tags for each file, put in a dict: {filename:x,time:y,closest_files:[(file1,td1),(file2,td2)]}
    fdict={'094':[],'131':[],'171':[],'193':[],'211':[],'335':[]}
    wavelengths=['094','131','171','193','211','335']
    for f in files:

        try:
            ttag=dt.strptime(f[8:-5],'%Y%m%d_%H%M%S')
            wave=f[4:7]
        except ValueError:
            ttag=dt.strptime(f[4:-10],'%Y%m%d_%H%M%S')
            wave=f[-8:-5]

        #ttag=dt.strptime(f[8:-5],'%Y%m%d_%H%M%S')
        #wave=f[4:7]

        tdict={'filename':f,'time':ttag,'closest_files':[],'tds':[],'group_id':0}
        fdict[wave].append(tdict)
        #find the closest files in other wavelengths
        #for w in fdict.keys():
        #    wavelengths=['094','131','171','193','211','335']
        #    wavelengths.remove(wave) #all the others remain
    for f in fdict['094']: #get the closest file in other wavelengths
        #print f,ttag
        ttag=f['time']
        f['closest_files']=[find_closest_file(ttag,fdict,wave=ww) for ww in wavelengths]
        #don't forget to sort
    groups=[f['closest_files'] for f in fdict['094']]
    #return list of groups ...
    return groups

def find_closest_file(ttag,fdict,wave='094'):
    '''what it sounds like'''
    tvec=[fdict[wave][idx]['time'] for idx in range(0,len(fdict[wave]))]
    fvec=[fdict[wave][idx]['filename'] for idx in range(0,len(fdict[wave]))] #hopefully this doesn't mess up the sorting
    #find closest t in tvec to given ttag
    closest_t= min(tvec, key=lambda x: abs(x - ttag))
    ct_idx=tvec.index(closest_t)
    closest_file=fvec[ct_idx]
    #print closest_t,ct_idx,closest_file
    #return corresponding filename... figure out a way to return timedelta as well
    return closest_file

def plot_and_save(mlist,subcoords=False,outname='STEREO_orbit8_',creverse=True,vrange=False):
    '''assume these are de-rotated difference images. bcoords is circle coords'''
    map0=mlist[0]

    palette_name=map0.plot_settings['cmap']
    #if creverse and not palette_name.name.endswith('_r'):
    #    new_cdata=cm.revcmap(palette_name._segmentdata)
    #else:
    #    new_cdata=palette_name._segmentdata
    #new_cmap=matplotlib.colors.LinearSegmentedColormap(palette_name.name+'_r',new_cdata)
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
        s.plot(axes=ax,cmap='Greys_r')
        plt.colorbar()
        fig.savefig(outname+str(i).zfill(2)+'.png')
    #return submaps

def make_simple_dfaia(submap,timerange=['20120912_09','20120912_'],folder='/Users/wheatley/Documents/Solar/NuStar/orbit8/AIA',to_json=False, mask=False, mask_data=False, force_mask=True,verbose=False):
    '''make the dataframe with just the unmasked data for given timerange'''
    wavs=[94,131,171,193,211,335]
    if type(submap[0]) !=SkyCoord:
        #convert to skycoord
        bl=SkyCoord()
        tr=SkyCoord() #fill in later, need coordframe, for now assume SkyCoord
    else:
        bl,tr=submap
    all_maps,pdicts=[],[]
    for w in wavs:
        afiles=glob.glob(folder+'/AIA'+timerange[0]+'*'+str(w)+'.fits')
        afiles.sort()
        #print(afiles)
        amaps=[sunpy.map.Map(a).submap(bl,tr) for a in afiles]
        all_maps.append(amaps)
        print("Processing ", len(amaps), " maps")
        pdict=track_region_box(amaps,filenames=afiles,circle=False,mask=mask,plot=False, mask_data=mask_data,force_mask=force_mask,verbose=verbose) #currently might fail unless tr is 1 frame
        #try:
            #adf=pd.DataFrame({ key:pd.Series(value) for key, value in pdict.items()})#
        adf=pd.DataFrame(pdict)
        #adf['filenames']=afiles
        adf['wavelength']=w
        pdicts.append(adf)
#        except ValueError:
#            print("Value Error, wavelength ",w, " num files ", len(afiles))
#            print(len(pdicts))
#            print([len(v) for k,v in pdicts[0].items()])
#            print([len(v) for k,v in pdict.items()])
#            break
        

    dfaia=pd.concat(pdicts)
    dfaia['cutout_shape']=[np.product(np.shape(d)) for d in dfaia.data]
    #dfaia.apply() timestamps)
    dfaia.reset_index(inplace=True)
    if to_json != False:
        dfaia.to_json(to_json,default_handler=str)
    return dfaia

def get_datavec(dfaia,timestamp,waves=[94,131,171,193,211,335],plus=False,key="data",filenames=False):
    datavec=[]
    for w in waves:
        wdf=dfaia.where(dfaia.wavelength==w).dropna(how='all')
        datavec.append(get_dataval(wdf,timestamp,key=key,filenames=filenames))
    return datavec

def get_dataval(wdf,timestamp,key="data",filenames=False):
    close_times= np.argmin(np.abs(wdf.timestamps - timestamp))
    #return close_times
    #print(close_times)#,close_times['flux_total_DNs-1'])
    val=wdf[key][close_times]
    #elif minus:
    #    val=wdf['flux_minus_DNs-1'][close_times]
    #elif total:
    #    val=wdf['flux_total_DNs-1'][close_times]
    #elif data:
    #    val=wdf.data[close_times]
    if filenames:
        val={'data':wdf.data[close_times],'filenames':wdf.filenames[close_times]}
    return val
    
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
    
def plot_aia_masks(maskpickle,tag=False):
    if not tag:
        tag=''
    wavs=[94,131,171,193,211,335]
    masks, mask_plus,mask_minus=pickle.load(open(maskpickle,'rb'))
    mdiffs,qdiffs=[],[]
    for w in wavs:
        mdiff=diff_maps([sunpy.map.Map('AIA/AIA_'+str(w)+'_preflare_'+tag+'00.fits'),sunpy.map.Map('AIA/AIA_'+str(w)+'_flare_'+tag+'00.fits')])[0]
        qdiff=diff_maps([sunpy.map.Map('AIA/AIA_'+str(w)+'_preflare_'+tag+'qs00.fits'),sunpy.map.Map('AIA/AIA_'+str(w)+'_flare_'+tag+'qs00.fits')])[0]
        mdiffs.append(mdiff)
        qdiffs.append(qdiff)
        
    fig,ax=plt.subplots(5,6,figsize=[10,8])
    for i in range(6):
        ax[0][i%6].imshow(mdiffs[i].data,cmap=cm.Greys_r, origin='lower left')
        ax[1][i%6].imshow(qdiffs[i].data,cmap=cm.Greys_r, origin='lower left')
        ax[2][i%6].imshow(masks[i],cmap=cm.Blues, origin='lower left')
        ax[0][i%6].set_title(str(wavs[i]))
        ax[3][i%6].imshow(mask_minus[i],cmap=cm.Greens, origin='lower left')
        #ax[1][i%6].set_title(masked_maps[i].meta['wavelnth'])
        ax[4][i%6].imshow(mask_plus[i],cmap=cm.Reds, origin='lower left')
        #ax[2][i%6].set_title(masked_maps[i].meta['wavelnth'])
    fig.show()
    return fig
    
def make_masked_df(flare_box,masks=False,mask_plus=False,mask_minus=False,tag=None):
    #flare_box=[(-610, 30), (-550, 90)]
    if not tag:
        tag=''
    if not masks and not mask_minus and not mask_plus:
        masks, mask_plus,mask_minus=pickle.load(open('all_masks_source1.p','rb'))

    dfs=[]
    #tdicts,pdicts,mdicts=[],[],[]
    for i,w in enumerate([94,131,171,193,211,335]): #do for all masks too...
        submaps=make_submaps('AIA/AIA'+"{:03d}".format(w)+'maps'+tag+'.p',flare_box[0],flare_box[1])
        ftotal=track_region_box(submaps,mask=masks[i],force_mask=True,mask_data=False)
        fplus=track_region_box(submaps,mask=mask_plus[i],force_mask=True,mask_data=False)
        fminus=track_region_box(submaps,mask=mask_minus[i],force_mask=True,mask_data=False)
        
        df=pd.DataFrame(ftotal)
        df.rename(columns={'fluxes':'flux_total'},inplace=True)
        df['flux_plus']=pd.Series(fplus['fluxes'])
        df['flux_minus']=pd.Series(fminus['fluxes'])
        df['total_mask']=pd.Series([masks[i]])
        df['mask_plus']=pd.Series([mask_plus[i]])
        df['mask_minus']=pd.Series([mask_minus[i]])
        df['total_mask_px']=pd.Series(np.sum(np.logical_not(masks[i])))
        df['mask_plus_px']=pd.Series(np.sum(np.logical_not(mask_plus[i])))
        df['mask_minus_px']=pd.Series(np.sum(np.logical_not(mask_minus[i])))
        
        umflux=[]
        for j in df.index:
            umflux.append(np.nanmean(df.data[j]*~masks[i]))
        umpx=np.sum(masks[i])
        df['outside_mask_flux']=umflux
        df['outside_mask_px']=[umpx for i in range(len(df.index))]
        df.sort_values('timestamps',inplace=True)
        df['wavelength']=w
        dfs.append(df)
        
    dfaiam=pd.concat(dfs)
    dfaiam.reset_index(inplace=True)
    #dfaiam3.to_json('AIA_full_masked_3s.json',default_handler=str)
    return dfaiam

def make_total_masks(df):
    '''sum masks over all wavelengths, fill in empty values in the dataframe if they exist '''
    #get masks
    refill=False
    if len(list(df.total_mask.dropna(how='all'))) < 7: #need to fill in values
        refill=True
        
    for i,w in enumerate([94,131,171,193,211,335]):
        sdf=df.where(df.wavelength==w).dropna(how='all')
        maskt=sdf.total_mask.dropna(how='all').iloc[0]
        maskp=sdf.mask_plus.dropna(how='all').iloc[0]
        maskm=sdf.mask_minus.dropna(how='all').iloc[0]
        #if refill:
            #for t in sdf.index: #this is super slow!
            #    df['total_mask'][t]=maskt
            #     df['mask_plus'][t]=maskp
            #    df['mask_minus'][t]=maskm
        try:
            all_tmask=all_tmask+~np.array(maskt)
            all_pmask=all_pmask+~np.array(maskp)
            all_mmask=all_mmask+~np.array(maskm)
        except NameError:
            all_tmask=~np.array(maskt)
            all_pmask=~np.array(maskp)
            all_mmask=~np.array(maskm)
    
    df['total_mask_wavelengths']=[[~all_tmask] for a in df.index]
    df['total_pmask_wavelengths']=[[~all_pmask] for a in df.index]
    df['total_mmask_wavelengths']=[[~all_mmask] for a in df.index]
    if type(df['timestamps'][0]) == str:
        df['timestamps']=pd.to_datetime(df.timestamps)
    
    return df
