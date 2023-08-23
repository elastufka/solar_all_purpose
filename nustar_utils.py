import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
#import wcsaxes
#from astropy.wcs import WCS

import sunpy.map
#import sunpy.coordinates
#import sunpy.coordinates.wcs_utils
from sunpy.net import vso
import numpy as np
import numpy.ma as ma
import matplotlib.dates as mdates
import pandas as pd
import os

import pickle
from datetime import datetime as dt
import glob
import matplotlib
from matplotlib import cm
from matplotlib.colors import LogNorm
from plotly.subplots import make_subplots

from sunpy_map_utils import fix_units, find_centroid_from_map
from flux_in_boxes import track_region_box
from flare_physics_utils import *
# This is from nustar_sac https://github.com/ianan/nustar_sac/blob/master/python/ns_tresp.py
import ns_tresp

def get_obs_params(orbit,tag=None):
    obs_dir='/Users/wheatley/Documents/Solar/NuStar/orbit%s' % orbit
    os.chdir(obs_dir)
    if not tag:
        tag=''
    json_file='orbit%s_params%s.json' % (orbit,tag)
    dfobs=pd.read_json(json_file)
    
    try: #to load coordinate pickles so they are SkyCoord objects
        dfobs['circle']=pd.Series(pickle.load(open('circle80_coords%s.p' % tag,'rb')))
        dfobs['circle20']=pd.Series(pickle.load(open('circle20_coords%s.p' % tag,'rb')))
        dfobs['circle20_off']=pd.Series(pickle.load(open('circle20_offset_coords%s.p' % tag,'rb')))
        dfobs['circ_stereo']=pd.Series(pickle.load(open('circ_stereo%s.p' % tag,'rb')))
    except FileNotFoundError:
        pass
    return dfobs
    
def write_obs_params(obs_params):
    os.chdir(obs_params.obs_dir[0])
    #delete skycoord objects
    orbit=obs_params.orbit[0].astype(str)
    try:
        del obs_params['circle']
        del obs_params['circle20']
        del obs_params['circle20_off']
        del obs_params['circ_stereo']
    except KeyError:
        pass
    try:
        tag=obs_params.tag[0]
    except AttributeError:
        tag=''
    filename='orbit%s_params%s.json' % (orbit,tag)
    obs_params.to_json(filename)
    
def imsav_to_fits(savfile,checksum=True,tag='20s',head='nustar_'):
    ''' convert .sav files from Sam into fits files for use with SunPy'''
    idl = pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
    idl('savfile',savfile)
    idl('head',head)
    idl('restore,savfile')
    idl('nmaps=size(m2_a,/dim)')
    idl("for i=0,nmaps[0]-1 do map2fits,m2_a[i],head+repstr(strmid(m2_a[i].time,12,8),':','')+'_'+repstr(strmid(m2_a[i].id,0,10),' ','_')+'_hi.fits'")
    idl('nmaps=size(m2_b,/dim)')
    idl("for i=0,nmaps[0]-1 do map2fits,m2_b[i],head+repstr(strmid(m2_b[i].time,12,8),':','')+'_'+repstr(strmid(m2_b[i].id,0,10),' ','_')+'_hi.fits'")
    idl('nmaps=size(m_a,/dim)')
    idl("for i=0,nmaps[0]-1 do map2fits,m2_a[i],head+repstr(strmid(m_a[i].time,12,8),':','')+'_'+repstr(strmid(m_a[i].id,0,10),' ','_')+'_lo.fits'")
    idl('nmaps=size(m_b,/dim)')
    idl("for i=0,nmaps[0]-1 do map2fits,m_b[i],head+repstr(strmid(m_b[i].time,12,8),':','')+'_'+repstr(strmid(m_b[i].id,0,10),' ','_')+'_lo.fits'")

def preview_submaps_ROI(mlist,circle, circle2=False,maxim=20,startidx=0,norm=LogNorm(),shape='circle',colors=['m','g']):
    '''plot the submaps and ROI circles'''
    if type(mlist) == str: #it's a pickle file
        mdict=pickle.load(open(mlist,'rb'))
        try:
            mlist=mdict['maps']
        except TypeError:
            mlist=mdict
    fig = plt.figure(figsize=(15, 15))
    for i, m in enumerate(mlist[startidx:startidx+maxim]):
        ax = fig.add_subplot(4, 5, i+1, projection=m.wcs)
        m.plot(axes=ax,norm=norm)
        ax.set_title(m.meta['date-obs'])
        if shape =='circle':
            ax.plot_coord(circle,colors[0])
            if circle2 !=False:
                ax.plot_coord(circle2,colors[1])
        elif shape == 'box' or shape == 'square':
            bl1=SkyCoord(np.min(circle.Tx),np.min(circle.Ty),frame=m.coordinate_frame)
            w1=np.max(circle.Tx)- np.min(circle.Tx)
            h1=np.max(circle.Ty)- np.min(circle.Ty)
            m.draw_rectangle(bl1, w1,h1,axes=ax,color=colors[0])
            if circle2 !=False:
                bl2=SkyCoord(np.min(circle2.Tx),np.min(circle2.Ty),frame=m.coordinate_frame)
                w2=np.max(circle2.Tx)- np.min(circle2.Tx)
                h2=np.max(circle2.Ty)- np.min(circle2.Ty)
                m.draw_rectangle(bl2, w2,h2,axes=ax,color=colors[1])
        #turn of axes labels...
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)
    return fig
    
def make_nustar_dicts(bl,tr,circle, tag=None):
    for t in ['m_a','m_b','m2_a','m2_b']: #'m_a','m_b',
        try:
            nulist=pickle.load(open('Xray/nustar_'+t+'_maps.p','rb'))
        except (FileNotFoundError,EOFError) as e:
            nf=glob.glob('Xray/nustar_'+t+'*.fits')
            nf.sort()
            nulist=[fix_units(sunpy.map.Map(n)) for n in nf]
            pickle.dump(nulist,open('Xray/nustar_'+t+'_maps.p','wb'))
        nsmaps=[]
        for n in nulist:
            bottom_left = SkyCoord(bl[0] * u.arcsec, bl[1] * u.arcsec, frame=n.coordinate_frame)
            top_right = SkyCoord(tr[0]* u.arcsec, tr[1] * u.arcsec, frame=n.coordinate_frame)
            nsmaps.append(n.submap(bottom_left,top_right))
        nsdict=track_region_box(nsmaps, circle=circle,mask_data=False)
        pickle.dump(nsdict,open('Xray/nustar_'+t+'_dict'+tag+'.p','wb'))
        
def make_aia_dicts(bl,tr,circle,tag=None):
    for w in ['094','131','171','193','211','335']:
        amaps=pickle.load(open('AIA/AIA'+w+'maps'+tag+'.p','rb'))
        asmaps=[]
        for n in amaps:
            bottom_left = SkyCoord(bl[0] * u.arcsec, bl[1] * u.arcsec, frame=n.coordinate_frame)
            top_right = SkyCoord(tr[0]* u.arcsec, tr[1] * u.arcsec, frame=n.coordinate_frame)
            asmaps.append(n.submap(bottom_left,top_right))
        asdict=track_region_box(asmaps, circle=circle,mask_data=False)
        pickle.dump(asdict,open('AIA/AIA'+w+'_dict'+tag+'.p','wb'))

def make_event_df(obs_dir,tag=None):
    '''XRT and NuSTAR in /Xray, AIA in /AIA, stereo in /STEREO, dicts or maps stored in pickles. Ncounts and int_fluxes are only meaningful for NuSTAR data '''
    dfs=[]
    for w in ['094','131','171','193','211','335']:
        dicts=pickle.load(open('AIA/AIA'+w+'_dict'+tag+'.p','rb'))
        df=pd.DataFrame(dicts)
        df['wavelength']=int(w)
        df.sort_values('timestamps',inplace=True)
        dfs.append(df)
    for t in ['m_a','m_b','m2_a','m2_b']:
        ndict=pickle.load(open('Xray/nustar_'+t+'_dict'+tag+'.p','rb'))
        df=pd.DataFrame(ndict)
        df['wavelength']=t
        df.sort_values('timestamps',inplace=True)
        dfs.append(df)
    xrtd=pickle.load(open('Xray/xrt_dict'+tag+'.p','rb'))
    df=pd.DataFrame(xrtd)
    df.sort_values('timestamps',inplace=True)
    df['wavelength']='Be-Thin'
    dfs.append(df)
    sdict=pickle.load(open('STEREO/s195dict'+tag+'.p','rb'))
    df=pd.DataFrame(sdict)
    df['wavelength']=195
    #df.sort_values('timestamps',inplace=True)
    dfs.append(df)

    df=pd.concat(dfs)
    cval=0.058556753656769309
    df['ncounts']=[np.sum(m)/cval for m in df['data']]
    df['int_fluxes'] = [int(np.round(ncounts,0)) for ncounts in df.ncounts]
    return df
    
def compare_expected_measured_AIA(dfaia, obs_params):
    '''Compare observed and expected AIA fluxes for spectral fit and given flare_box'''
    return None
    
    
def plot_rmf_diag(nsid='80610208001',fpm='A',orbit='8',time='2039_2042', mdir='/Users/wheatley/Documents/Solar/NuStar/specfiles/'):
    fvth=io.readsav('/Users/wheatley/Documents/Solar/NuStar/nustar_sac/python/fvth_out.dat')
    # in units of keV
    engs=fvth['eng']
    de=engs[1]-engs[0]
    logt=fvth['logt']
    # in units of photons/s/keV/cm2
    phmod=np.array(fvth['fvth'])

    arffile=glob.glob(mdir+'nu'+nsid+fpm+'*'+orbit+'*'+time+'.arf')[0]
    rmffile=glob.glob(mdir+'nu'+nsid+fpm+'*'+orbit+'*'+time+'.rmf')[0]
    print(arffile,rmffile)

    e_lo, e_hi, eff_area = ns_tresp.read_arf(arffile)
    e_lo, e_hi, rmf_mat = ns_tresp.read_rmf(rmffile)
    
    nume=len(engs)
    arf=eff_area[:nume]
    rmf=rmf_mat[0:nume,0:nume]
    
def gen_nustar_tresp(eng_tr=[1.5,2,2.5,3,3.5,4,6,8,10],nsid='80610208001',fpm='both',orbit='8',time='2039_2042', mdir='/Users/wheatley/Documents/Solar/NuStar/specfiles/'
,plot_photon=False,plot_counts=False,plot_tresp=True,plot_loci=True):
    import scipy.io as io
    import ns_tresp
    ''' from nustar_sac repo - basically contents of example_ntresp notebook with some additions
     ; Make a NuSTAR thermal response for given energy ranges and observation
     ;
     ; Note that, For something like AIA/XRT the data is DN/s/px and the response is DN cm^5/s/px
     ; Here uses f_vth.pro which produces ph/s/kev/cm^2 for an EM in cm^-3
     ; The NuSTAR SRM=RMF*ARF is cnts cm^2/photons so SRM#f_vth*dE is cnts/s
     ;
     ; Approach here gives you the Volumetric EM as thermal response in units of cnts cm^3/s,
     ; i.e R(T)=SRM#f_vth(EM,T)*dE/EM  so R(T)*EM[cm^-3] would give the observed cnts/s
     ;
     ; If want/using EM in cm^-5 then need to appropriately divide/multiply by the area of the observation '''
    fvth=io.readsav('/Users/wheatley/Documents/Solar/NuStar/nustar_sac/python/fvth_out.dat')
    # in units of keV
    engs=fvth['eng']
    de=engs[1]-engs[0]
    logt=fvth['logt']
    # in units of photons/s/keV/cm2
    phmod=np.array(fvth['fvth'])
    nume=len(engs)

    if fpm == 'both':
        arffiles=glob.glob(mdir+'nu'+nsid+'*'+orbit+'*'+time+'.arf')[0]
        rmffiles=glob.glob(mdir+'nu'+nsid+'*'+orbit+'*'+time+'.rmf')[0]
        for a,r in zip(arffiles,rmffiles):
            e_lo, e_hi, eff_area = ns_tresp.read_arf(a) #pretty sure elo and ehi are same for both
            e_lo, e_hi, rmf_mat = ns_tresp.read_rmf(r)
            arf=eff_area[:nume]
            rmf=rmf_mat[0:nume,0:nume]
    else:
        arffile=glob.glob(mdir+'nu'+nsid+fpm+'*'+orbit+'*'+time+'.arf')[0]
        rmffile=glob.glob(mdir+'nu'+nsid+fpm+'*'+orbit+'*'+time+'.rmf')[0]
        print(arffile,rmffile)

        e_lo, e_hi, eff_area = ns_tresp.read_arf(arffile)
        e_lo, e_hi, rmf_mat = ns_tresp.read_rmf(rmffile)
    
    
        arf=eff_area[:nume]
        rmf=rmf_mat[0:nume,0:nume]

    srm = np.array([rmf[r, :] * arf[r] for r in range(len(arf))])
    
    n1,n2=phmod.shape
    modrs= np.zeros([n1,n2])
    for t in np.arange(n2):
        modrs[:,t]=(phmod[:,t]@srm)*de
        
    tresp=np.zeros([len(modrs[0,:]),len(eng_tr)-1])

    for i in np.arange(len(eng_tr)-1):
        gd=np.where((e_lo >= eng_tr[i]) & (e_hi < eng_tr[i+1]) )
        mm=np.sum(modrs[gd,:],axis=1)
        tresp[:,i]=mm[0,:]/1e49
    
    phafile=glob.glob(mdir+'nu'+nsid+fpm+'*'+orbit+'*'+time+'.pha')[0]
    engs,cnts,lvtm,ontim=ns_tresp.read_pha(phafile)
    
    rate=np.zeros(len(eng_tr)-1)
    erate=np.zeros(len(eng_tr)-1)
                   
    for i in np.arange(len(eng_tr)-1):
        gd=np.where((engs >= eng_tr[i]) & (engs < eng_tr[i+1]) )
        rate[i]=np.sum(cnts[gd])/lvtm
        erate[i]=np.sqrt(np.sum(cnts[gd]))/lvtm
    
    if plot_photon:
        plt.rcParams.update({'font.size': 18,'font.family':"sans-serif",\
                             'font.sans-serif':"Arial",'mathtext.default':"regular"})
        fig = plt.figure(figsize=(8, 6))
        plt.loglog(engs,phmod[:,11],label=str(round(10**(logt[11])*1e-6,1))+' MK')
        plt.loglog(engs,phmod[:,16],label=str(round(10**(logt[16])*1e-6,1))+' MK')
        plt.loglog(engs,phmod[:,19],label=str(round(10**(logt[19])*1e-6,1))+' MK')
        plt.loglog(engs,phmod[:,25],label=str(round(10**(logt[25])*1e-6,1))+' MK')

        plt.ylim([1e0,1e8])
        plt.xlim([1.,50])
        plt.xlabel('Energy [keV]')
        plt.ylabel('${photons\;s^{-1}\;cm^{-2}\;keV^{-1}}$')
        plt.legend()
        fig.show()
        
    if plot_counts:
        fig = plt.figure(figsize=(8, 6))
        plt.loglog(engs,modrs[:,11],label=str(round(10**(logt[11])*1e-6,1))+' MK')
        plt.loglog(engs,modrs[:,16],label=str(round(10**(logt[16])*1e-6,1))+' MK')
        plt.loglog(engs,modrs[:,19],label=str(round(10**(logt[19])*1e-6,1))+' MK')
        plt.loglog(engs,modrs[:,25],label=str(round(10**(logt[25])*1e-6,1))+' MK')
        plt.ylim([1e-4,1e8])
        plt.xlim([1.,50])
        plt.xlabel('Energy [keV]')
        plt.ylabel('${counts\;s^{-1},\;@\;EM=10^{49}cm^{-3}}$')
        plt.legend()
        fig.show()
        
    if plot_tresp:
        fig = plt.figure(figsize=(8, 6))
        for i in np.arange(len(eng_tr)-1):
            plt.loglog(10**logt,tresp[:,i],label=str(eng_tr[i])+' - '+str(eng_tr[i+1])+ ' keV')

        plt.ylim([1e-56,1e-41])
        plt.xlim([1e6,2e7])
        plt.xlabel('Temperature [K]')
        plt.ylabel('${counts\;s^{-1}\;cm^{3}}$')
        plt.legend()
        fig.show()
        
    if plot_loci:
        fig = plt.figure(figsize=(8, 6))
        clrs=['royalblue','firebrick','teal','orange','black','red','green','cyan','magenta']
        for i in np.arange(len(eng_tr)-3):
            plt.loglog(10**logt,rate[i]/tresp[:,i],label=str(eng_tr[i])+' - '+str(eng_tr[i+1])+ ' keV',color=clrs[i])
        plt.ylim([1e40,1e50])
        plt.xlim([1e6,2e7])
        plt.xlabel('Temperature [K]')
        plt.ylabel('${cm^{-3}}$')
        plt.legend()
        fig.show()
    
    return logt,rate, tresp
    
def read_xspec_results(fitfile='xspec.txt',zero_to_nan=True):
    names = ['energy','denergy','data','data_err','model']
    #should delete fit file if already exists
    df = pd.read_table(fitfile,skiprows=3,names=names, delimiter=' ')
    bidx=df.where(df.energy == 'NO').dropna(how='all').index.values #these are the breaks between
    #if len(bidx)==2:
    df_ld=df.iloc[:bidx[0]]
    df_uf=df.iloc[bidx[0]+1:bidx[1]]
    df_dc=df.iloc[bidx[1]+1:]
    #convert to floats
    for df in [df_ld,df_uf,df_dc]:
        for k in df.keys():
            df[k]=df[k].replace('NO','0')
            df[k]=pd.to_numeric(df[k])
            if zero_to_nan:
                df[k]=df[k].replace(0.0,np.nan)
        
    return df_ld,df_uf,df_dc

    
def plot_nustar_specfit(fitfile='test_model_out.txt',title='',fitstart=0,fitend=5):
    '''for file written from xspec command: iplot ldata ufspec rat '''
    names = ['energy','denergy','data','data_err','model']
    df = pd.read_table(fitfile,skiprows=3,names=names, delimiter=' ')
    bidx=df.where(df.energy == 'NO').dropna(how='all').index.values #these are the breaks between plots..
    #different plots are 'ld','uf','dc' no idea what these mean... uf and ld seem to be the same
    df_ld=df.iloc[:bidx[0]]
    df_uf=df.iloc[bidx[0]+1:bidx[1]]
    df_dc=df.iloc[bidx[1]+1:]
    #convert to floats
    for df in [df_ld,df_uf,df_dc]:
        for k in df.keys():
            df[k]=df[k].replace('NO','0')
            df[k]=pd.to_numeric(df[k])
            
    dfp=df_ld.replace(0.0,np.nan) #don't plot zeros
#    tempfit=np.round(c2.kT.values[0]/kev2mk,2)
#    terr=np.round(c2.kT.error[0]/kev2mk,2)
#    emfit=np.format_float_scientific(c2.norm.values[0]/emfact,precision=2)
#    emerr=np.round(c2.norm.error[0]/emfact,2)
#    title=f"{tempfit} ± {terr} MK, {emfit} ± {emerr} cm<sup>-3</sup>"
    
    #okay but let's do this in Plotly
    fig = make_subplots(rows=2, cols=1, start_cell="top-left",shared_xaxes=True,row_heights=[.6,.3],vertical_spacing=.05)
    fig.add_trace(go.Scatter(x=dfp.energy,y=dfp.data,mode='markers',name='data',error_y=dict(type='data',array=dfp.data_err)),row=1,col=1)
    fig.add_trace(go.Scatter(x=dfp.energy,y=dfp.model,mode='lines',name='fit'),row=1,col=1)
    fig.add_trace(go.Scatter(x=dfp.energy,y=dfp.data-dfp.model,mode='markers',marker_color='brown',name='residuals'),row=2,col=1)
    fig.add_vrect(x0=fitstart,x1=fitend,annotation_text='fit range',fillcolor='lightgreen',opacity=.25,line_width=0,row=1,col=1)
    fig.add_vrect(x0=fitstart,x1=fitend,fillcolor='lightgreen',opacity=.25,line_width=0,row=2,col=1)
    fig.update_yaxes(title='Counts s<sup>-1</sup> keV<sup>-1</sup>',range=[-1.5,1],row=1,col=1,type='log') #type='log'
    fig.update_yaxes(title='Residuals',range=[-.5,.5],row=2,col=1)
    fig.update_xaxes(title='Energy (keV)',row=2,col=1)
    fig.update_layout(width=500,height=600,title=title)
    
    return fig

def plot_nustar_spectrum(timerange=[2039,2042],both=True,erange=[1.5,5]):
    phaf='*sr_*%s_%s.pha' % tuple(timerange) #this will get for both A and B cameras, all CHUs
    phafiles=glob.glob('/Users/wheatley/Documents/Solar/NuStar/specfiles/'+phaf)
    phafiles.sort()
    
#    if not energy_bins: #use native bins/plot each as they come
#
#    else:
#        rate=np.zeros(len(energies))
#        erate=np.zeros(len(energies))
        
    if phafiles != []:
        chu=phafiles[0][phafiles[0].find('chu')+3:phafiles[0].find('chu')+5]
        if both == 'A':
            engs,cnts,lvtm,ontim=ns_tresp.read_pha(phafiles[0])
            ydat=cnts/lvtm
            yerr=dict(type='data',array=np.sqrt(cnts/lvtm))
        elif both == 'B':
            engs,cnts,lvtm,ontim=ns_tresp.read_pha(phafiles[1])
            ydat=cnts/lvtm
            yerr=dict(type='data',array=np.sqrt(cnts/lvtm))
        else:
            ydat=[]
            for p in phafiles: # can eventually adjust this to work with the both='A' or 'B' condition too
                engs,cnts,lvtm,ontim=ns_tresp.read_pha(p)
                #are engs always the same? check...
                ydat.append(cnts/lvtm)
                both = 'A+B'
            ydat=np.sum(ydat,axis=0)
        
        title='NuSTAR spectrum, CHU %s %s, %s-%s' % (chu,both,timerange[0],timerange[1])
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=engs,y=ydat,mode='markers'))
        fig.update_layout(yaxis_type='log',yaxis_title='Counts/s',xaxis_title='E (keV)',xaxis_range=erange,title=title) #is it really counts per second?
        fig.update_layout(yaxis = dict(showexponent = 'all',exponentformat = 'e'))
        
    else:
        print("No .pha file found for given timerange %s-%s!" % tuple(timerange))
        
    return fig

def nustar_dem_prep(bl,tr,timerange=False, contour=.5, tint=True, how=np.sum,return_maps=False, both=True,datadir='/Users/wheatley/Documents/Solar/Nustar/orbit8/Xray/',twenty_seconds=True, use_specfiles=True, energies=[[2.5,4]]):
    ''' get datavec from nustar images. if contour, only take above that value (in each image...or combined ma+mb?)
    submap is in skycoords as usual '''
    if not timerange:
        timerange=[2039,2042]
    nma,mb,m2a,m2b=nustar_get_fits(timerange=timerange,datadir=datadir,twenty_seconds=twenty_seconds)
    
    #check for .pha, .rmf, .arf etc files if available
    phafiles=[]
    if use_specfiles:
        mapa,map2a=None,None
        phaf='*sr_*%s_%s.pha' % tuple(timerange) #this will get for both A and B cameras, all CHUs
        phafiles=glob.glob('/Users/wheatley/Documents/Solar/NuStar/specfiles/'+phaf)
        rate=np.zeros(len(energies))
        erate=np.zeros(len(energies))
        numaps=False
        
        if phafiles != []:
            phafiles.sort()
            rateA=0.
            for p in phafiles: # can eventually adjust this to work with the both='A' or 'B' condition too
                engs,cnts,lvtm,ontim=ns_tresp.read_pha(p)
                # Work out the total count rate and error in the energy bands
                for i,en in enumerate(energies):
                    gd=np.where((engs >= en[0]) & (engs < en[1]))
                    rate[i]+=np.sum(cnts[gd])/lvtm
                    erate[i]+=np.sqrt(np.sum(cnts[gd]))/lvtm
                    #need to add if both! othrerwise just return one!
                if both == 'A':
                    ecounts=rate
                    break
                elif both == 'B':
                    ecounts=rate #but go one
                else:
                    rateA+=rate
            ecounts=rate
            #lowEvals=None
            #highEvals=rate[0]
            #print(rate,erate)
    if use_specfiles == False or phafiles == []:
        print("No specfiles available for timerange %s - %s! Using 20s images instead." % tuple(timerange))
    
        if nma == [] or mb == [] or m2a == [] or m2b == []:
            print('no NuSTAR fits files found for timerange %s!' % timerange)
            return None, None, None, None
        #print("nustar fits files: %s" % nma)
        lowEvals, highEvals=[],[]
        tmlo,tmhi=[],[]
        #get submaps
        for fa,fb,f2a,f2b in zip(nma,mb,m2a,m2b):
            mapa=fix_units(sunpy.map.Map(fa)).submap(bl,tr) #units are counts/s in each pixel!
            mapb=fix_units(sunpy.map.Map(fb)).submap(bl,tr)
            map2a=fix_units(sunpy.map.Map(f2a)).submap(bl,tr)
            map2b=fix_units(sunpy.map.Map(f2b)).submap(bl,tr)
            numaps=[mapa,map2a]
            #print(mapa.bottom_left,mapa.top_right)
            #if contour...
            if both:
                mdatlo=(mapa.data+mapb.data)/2.
                mdathi=(map2a.data+map2b.data)/2.
            elif both == 'A':
                mdatlo=mapa.data
                mdathi=map2a.data
            elif both == 'B':
                mdatlo=mapb.data
                mdathi=map2b.data
            tmlo.append(mdatlo)
            tmhi.append(mdathi)
            lowEvals.append(how(mdatlo))
            highEvals.append(how(mdathi)) #still counts/s in each pixel...
        #print(lowEvals,highEvals)
        if tint:
            nseconds=(timerange[1]-timerange[0])*60
            mapa=sunpy.map.Map(np.mean(tmlo,axis=0),mapa.meta)
            map2a=sunpy.map.Map(np.mean(tmhi,axis=0),map2a.meta)
            lowEvals=np.mean(lowEvals)#/nseconds #these ARE the counts/s maps!
            highEvals=np.mean(highEvals) #/nseconds #units? counts/s ... DEM takes DN/s as input I think, how to convert?
            numaps=[mapa,map2a]
        ecounts=[lowEvals,highEvals]
       
    return ecounts,numaps#lowEvals,highEvals,mapa,map2a
    
def nustar_get_fits(datadir='/Users/wheatley/Documents/Solar/Nustar/orbit8/Xray/',timerange=[2039,2042],twenty_seconds=True):
    ''' get fits files between certain time'''
    ma_out,mb_out,m2a_out,m2b_out=[],[],[],[]
    
    if twenty_seconds:
        ma=glob.glob(datadir+'nustar_*FPMA*lo.fits')
        mb=glob.glob(datadir+'nustar_*FPMB*lo.fits')
        m2a=glob.glob(datadir+'nustar_*FPMA*hi.fits')
        m2b=glob.glob(datadir+'nustar_*FPMB*hi.fits')
        if len(mb)==0: #try other naming convention...
            ma=glob.glob(datadir+'nustar_m_a_*.fits')
            mb=glob.glob(datadir+'nustar_m_b_*.fits')
            m2a=glob.glob(datadir+'nustar_m2_a_*.fits')
            m2b=glob.glob(datadir+'nustar_m2_b_*.fits')

        
    else:
        ma=glob.glob(datadir+'nustar_t_an_*00.fits')
        mb=glob.glob(datadir+'nustar_t_bn_*00.fits')
        m2a=glob.glob(datadir+'nustar_t_an2_*00.fits')
        m2b=glob.glob(datadir+'nustar_t_bn2_*00.fits')

    #get within correct time range
    ma.sort()
    mb.sort()
    m2a.sort()
    m2b.sort()
    
    for fa,fb,f2a,f2b in zip(ma,mb,m2a,m2b): #they'd better all be the same size
        try:
            ts=int(fa[fa.find('2'):fa.find('_FPM')][:-2])
        except ValueError:
            ts=int(fa[fa.rfind('_')+1:fa.rfind('.')][:-2])
        #print(ts)
        if ts >= timerange[0] and ts <= timerange[1]:
            ma_out.append(fa)
            mb_out.append(fb)
            m2a_out.append(f2a)
            m2b_out.append(f2b)
    return ma_out,mb_out,m2a_out,m2b_out

def plot_all_lightcurves(event_df, use_matplotlib=False,shape=None,mode='lines',box=2, title_ext='',legend_fontsize=11,shadecolor='cyan',time_intervals=[["2020-09-12T20:32:00","2020-09-12T20:37:00"],["2020-09-12T20:39:00","2020-09-12T20:42:00"],["2020-09-12T20:50:00","2020-09-12T20:53:00"]]):
    ''' shape='hv' makes it a step plot'''
    nustar_names=['NuSTAR A 1-2 keV','NuSTAR A 2.5-3.5 keV','NuSTAR B 1-2 keV','NuSTAR B 2.5-3.5 keV']
    tvecs,normcurves=[],[]
    idlist=[94,131,171,193,211,335]
    for ids in idlist:
        df=event_df.where(event_df.wavelength == ids).dropna(how='all')
        df.sort_values('timestamps',inplace=True)
        ft=df.flux_total#/df.total_mask_px.max() #don't need if masked, which it is
        tvec=df.timestamps
        tvecs.append(tvec)
        normcurves.append(ft/np.mean(ft))
        
    ntvecs,ndvecs,nevecs=[],[],[]
    if 'ID' in event_df.keys():
        for ids,name in zip(event_df.ID.unique()[1:],nustar_names):
            df=event_df.where(event_df.ID == ids).dropna(how='all')
            df.sort_values('timestamps',inplace=True)
            #df.reset_index(inplace=True,drop=True)
            error=dict(type='data',array=np.sqrt(df.int_fluxes),visible=True)
            #fig.add_trace(go.Scatter(x=df.timestamps,y=df.int_fluxes,mode=mode,name=name, error_y=error),row=2,col=1,secondary_y=False)
            ntvecs.append([df.timestamps])
            ndvecs.append([df.int_fluxes])
            nevecs.append(list(error['array'].values))
    else:
        for ids,name in zip(['m_a', 'm_b', 'm2_a', 'm2_b'],nustar_names):
            df=event_df.where(event_df.wavelength == ids).dropna(how='all')
            df.sort_values('timestamps',inplace=True)
            #df.reset_index(inplace=True,drop=True)
            error=dict(type='data',array=np.sqrt(df.int_fluxes),visible=True)
            #fig.add_trace(go.Scatter(x=df.timestamps,y=df.int_fluxes,mode=mode,name=name, error_y=error),row=2,col=1,secondary_y=False)
            ntvecs.append(df.timestamps.values)
            ndvecs.append(df.int_fluxes.values)
            nevecs.append(list(error['array'].values))
            
    #print(type(nevecs[0]),nevecs[0].shape,len(ndvecs[0][0]))
    
    if use_matplotlib:
        fig,ax=plt.subplots(2,1,sharex=True,figsize=[11,6.5])
        for i,ids in enumerate(idlist):
            ax[0].plot(tvecs[i],normcurves[i],label='AIA '+str(int(ids)))
            
        ax[0].set_ylabel('AIA Flux/Mean Flux')
        ax[0].set_ylim([0.7,1.3])
        ax[0].legend(loc='upper right',fontsize=legend_fontsize)
        #no need for stereo in this one
        for i, name in enumerate(nustar_names):
            ylen=len(ndvecs[i][0])
            #yerr=nevecs[i].reshape((ylen,1))
            #print(yerr.shape)
            #for ne in nevecs[i]:
            #    print(type(ne))
                #if type(ne) !=np.float64:
                #    print(ne,type(ne))
            ax[1].errorbar(ntvecs[i][0],np.array(ndvecs[i][0]),yerr=np.array(nevecs[i]),label=name)#yerr=nevecs[i],
        
        xrtax=ax[1].twinx()
        xdf=event_df.where(event_df.wavelength=='Be-Thin').dropna(how='all').sort_values('timestamps')
        xrtax.plot(xdf['timestamps'],(xdf['fluxes']/np.mean(xdf['fluxes'])),'.-',color='black',label='XRT Be Thin')
        
        trange=(ntvecs[1][0].iloc[0],ntvecs[1][0].iloc[-1])
        tdiff=trange[1]-trange[0]
        #shaded areas
        for tr in time_intervals:
            dttr=[pd.to_datetime(tr[0]),pd.to_datetime(tr[1])]
            for a in ax:
                a.axvline(dttr[0], color=shadecolor, alpha=0.3)
                a.axvline(dttr[1], color=shadecolor, alpha=0.3)
                a.fill_betweenx([0,62], dttr[0], dttr[1],
                            facecolor=shadecolor, alpha=0.3)
                            
        for l,tr,offs in zip(['preflare','X-ray peak','EUV peak'],time_intervals,[.125,0.05,0]):
            xfrac=(pd.to_datetime(tr[0])-trange[0])/tdiff
            #print(xfrac,xfrac+offs)
            xy=(xfrac+offs,.53)
            plt.annotate(l,xy,xycoords='figure fraction')
        
        xrtax.legend(loc='upper right',fontsize=legend_fontsize)
        ax[1].legend(loc='upper left',fontsize=legend_fontsize)
        ax[1].set_ylabel('Counts/s') #is it per second?
        ax[1].set_xlabel('12 September 2020') #is it per second?
        ax[1].set_ylim([0,62])
        xrtax.set_ylabel('XRT Flux/Mean Flux')
        myFmt = mdates.DateFormatter('%H:%M')
        ax[1].xaxis.set_major_formatter(myFmt)
        plt.gcf().autofmt_xdate()
        #print(ntvecs[1][0].iloc[0],ntvecs[1][0].iloc[-1])
        ax[1].set_xlim(trange)
        
    else:
        fig = make_subplots(rows=2, cols=1, start_cell="top-left",shared_xaxes=True,specs=[[{}],[{"secondary_y": True}]])
        #AIA
        for i,ids in enumerate(idlist):
            fig.add_trace(go.Scatter(x=tvecs[i],y=normcurves[i],mode=mode,name='AIA '+str(int(ids))),row=1,col=1)

        try:
            sdf=event_df.where(event_df.Box==box)
        except AttributeError:
            sdf=event_df.where(event_df.wavelength==195)
        fig.add_trace(go.Scatter(x=sdf['timestamps'],y=sdf['fluxes']/np.mean(sdf['fluxes']),mode=mode,name='STEREO 195'),row=1,col=1)

        fig.update_layout(yaxis_title='Flux/Mean Flux',title=str(tvec.iloc[0].date())+title_ext)
        fig.update_layout(yaxis = dict(showexponent = 'all',exponentformat = 'e'))

        for i,name in enumerate(nustar_names):
            fig.add_trace(go.Scatter(x=ntvecs[i],y=ndvecs[i],mode=mode,name=name, error_y=nevecs[i]),row=2,col=1,secondary_y=False)
             
        #add axis on the right for XRT
        xdf=event_df.where(event_df.wavelength=='Be-Thin').dropna(how='all').sort_values('timestamps')
        fig.add_trace(go.Scatter(x=xdf['timestamps'],y=(xdf['fluxes']/np.mean(xdf['fluxes'])),mode=mode,name='XRT Be Thin'),row=2,col=1,secondary_y=True)

        fig.update_yaxes(title='Counts',row=2,col=1)
        fig.update_yaxes(title='Flux/Mean Flux',row=2,col=1,secondary_y=True)
        fig.update_layout(yaxis = dict(showexponent = 'all',exponentformat = 'e'))
    return fig

