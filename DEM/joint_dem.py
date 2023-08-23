0 #######################################
#display_aia_dem.py
# Erica Lastufka 15/03/2018  

#Description: Because OSX doesn't play well with XQuartz and IDL sucks
#######################################

#######################################
# Usage:

######################################

import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt

import os
from scipy.ndimage.filters import generic_filter as gf
import matplotlib.cm as cm
import matplotlib.colors as colors
import glob
import sunpy.map
from scipy.io import readsav
import astropy.units as u
from astropy.coordinates import SkyCoord
from datetime import datetime as dt
from scipy.interpolate import interp1d
from aia_utils import aia_maps_tint, single_time_indices
from dn2dem_pos import dn2dem_pos
from dem_utils import *
from sunpy_map_utils import find_centroid_from_map, arcsec_to_cm, scale_skycoord, cm_to_arcsec, hand_coalign
from nustar_utils import nustar_dem_prep
from pride_colors import get_continuous_color
#how to import python version of demreg? need to add that path to my init.py or python_startup script
import pickle
import plotly.graph_objects as go
from IPython.display import display

class joint_DEM:
    def __init__(self,full_trmatrix=False,trmatrix_logt=False,aiapickle='/Users/wheatley/Documents/Solar/NuStar/orbit8/AIA/dfaia2.p',timerange=["2020-09-12T20:40:00","2020-09-12T20:41:00"],tstart=5.6,tend=6.8,numtemps=42,aia_mask=False,aia_contour=80,aia_mask_channel=211,aia_calc_degs=False,aia_exptime_correct=False,xrt_submap_radius=10,xrt_max=False,nustar_submap_radius=55,nustar_area='AIA',xrt_fac=1,nustar_fac=1,initial_weights='loci_min',min_err=False,use_AIA=True,use_XRT=True,use_NuSTAR=True,selfnorm=False,verbose=True,datadir='/Users/wheatley/Documents/Solar/NuStar/orbit8/Xray/',aia_channels = [94,131,171,193,211,335],nustar_channels=[[2.5,4]], treat_nustar='separate'):
        '''run demreg for given configuration. Basically runs the AIA+XRT+NuSTAR notebook for different configurations

        Input full_trmatrix should NOT have NuSTAR response multiplied by any area factor! '''
        if type(full_trmatrix) !=np.ndarray:
            full_trmatrix,trmatrix_logt=self.load_default_trmatrix()
        self.input_trmatrix=np.copy(full_trmatrix)
        for key, value in locals().items():
            setattr(self, key, value)
        
        self.reset_chanax()
        self.temps,self.dtemps,self.mlogt=calc_temp_vars(tstart=self.tstart,tend=self.tend,num=self.numtemps)

    def load_default_trmatrix(self):
        trmatrix=pickle.load(open('/Users/wheatley/Documents/Solar/NuStar/aia_xrt_nustar_trmatrix_091220.p','rb'))
        aia_tresp_logt=pickle.load(open('/Users/wheatley/Documents/Solar/NuStar/aia_tresp_logt.p','rb'))
        return trmatrix,aia_tresp_logt

    def reset_chanax(self):
        self.chanax=['94','131','171','193','211','335','Be-Thin']
        for nc in self.nustar_channels:
            if self.treat_nustar=='separate':
                self.chanax.append('NuSTAR A %s-%s keV' % tuple(nc))
                self.chanax.append('NuSTAR B %s-%s keV' % tuple(nc))
            else:
                self.chanax.append('NuSTAR %s-%s keV' % tuple(nc))#np.append(6,filters[1])

    def reset_trmatrix(self):
        '''in case the original is modified and needs to be reset'''
        self.full_trmatrix=np.copy(self.input_trmatrix)
        
    def __orbit10_defaults__(self):
        '''to do:set timerange etc to be compatible with this'''
        self.timerange=["2020-09-12T20:40:00","2020-09-12T20:41:00"]
        self.aiapickle="orbit10_df.p"
        
    def __orbit6_defaults__(self):
        '''eventually want this to be the source 2 data...'''
        self.timerange=["2020-09-12T17:15:00","2020-09-12T17:21:00"]
        #self.timerange2=["2020-09-12T17:17:00","2020-09-12T17:20:00"]

        self.datadir='/Users/wheatley/Documents/Solar/NuStar/orbit6/Xray/'
        self.aia_contour=75
        self.xrt_submap_radius=6
        self.min_err=.15
        self.aia_mask_channel=171
        self.initial_weights='loci'
        self.nustar_area='AIA'
        self.aiapickle='/Users/wheatley/Documents/Solar/NuStar/orbit6/orbit6_dfaia.p'

    def run_from_inputs(self):
        self.select_joint_maps()
        self.prep_joint_maps()
        self.generate_dem_input()
        df=self.run_joint_dem()
        return df

    def select_joint_maps(self,preflare_difference=True):
        '''Get AIA,XRT and NuSTAR data for selected timerange '''
        tstart=dt.strptime(self.timerange[0],"%Y-%m-%dT%H:%M:%S")
        tend=dt.strptime(self.timerange[1],"%Y-%m-%dT%H:%M:%S")
        
        #AIA
        dfaia=pickle.load(open(self.aiapickle,'rb'))
        #try:
        dfaia.timestamps=pd.to_datetime(dfaia.timestamps)
        #except AttributeError: #it's the map pickle...
        #    dfaia['timestamps']=[l.meta['date-obs'] for l in dfaia]
        #    dfaia['wavelength']=[l.meta['wavelngth'] for l in dfaia]
        #    dfaia.timestamps=pd.to_datetime(dfaia.timestamps)
        aiamaps=aia_maps_tint(dfaia,timerange=self.timerange,how=np.sum,wavs=self.aia_channels)
        if preflare_difference:
            pftimestamp=dfaia.timestamps[0]
            pfidx=single_time_indices(dfaia,pftimestamp)
            pfmaplist=[dfaia.maps[p] for p in pfidx]
            self.pfmaplist=pfmaplist
        #XRT
        xrtsfmt=dt.strftime(tstart,"%Y%m%d_%H%M")
        xrtefmt=dt.strftime(tend,"%Y%m%d_%H%M")
        all_xrt=glob.glob('XRT_all_maps/XRT*_prepped.fits')
        #for f in all_xrt:
        #    print(f[f.find('/XRT')+4:f.rfind('prepped')-3])
        xrttimes=pd.Series([dt.strptime(f[f.find('/XRT')+4:f.rfind('prepped')-3],"%Y%m%d_%H%M%S") for f in all_xrt])
    
        xrtgt=xrttimes.where(xrttimes >= tstart)
        xrtselect=xrttimes.where(xrtgt <= tend).dropna(how='all').index.values #indices of correct times
        #fgm=glob.glob('XRT_all_maps/XRT'+xrtsfmt+'*.8_grademap.fits')
        #print(fdata)
        fdata=[]
        for i in xrtselect:
            fdata.append(all_xrt[i])
            #print(fdata)
        if len(fdata) == 1:
            xrtmap=sunpy.map.Map(fdata[0])
        else:
            fdata.sort()
            xmaps=[sunpy.map.Map(f) for f in fdata]
            xrt_mapdata=[]
            for m in xmaps:
                dur=m.exposure_time.value
                # What pixel binning per dimension
                chipsum=m.meta['chip_sum']
                #  Get a DN/s/px (non-binned pixels) for the region
                xrt_mapdata.append(m.data/dur/chipsum**2)
            xrtmap=sunpy.map.Map(np.nansum(xrt_mapdata,axis=0),xmaps[0].meta) #full disk...
        #xgmmap=sunpy.map.Map(fgm) #don't use for now...
        
        #NuSTAR timerange
        nustar_timerange=[int(dt.strftime(tstart,"%H%M")),int(dt.strftime(tend,"%H%M"))]
        
        self.aiamaps=aiamaps
        self.xrtmap=xrtmap
        self.nustar_timerange=nustar_timerange
        #return aiamaps,xrtmap,nustar_timerange
        
    def prep_joint_maps(self,plot_contour=False, plot_maps=False, preflare_difference=False,xrtoffset=.1, use_specfiles=True):
        '''Do all map operations here. XRT offset is a scaling factor by which to enlarge the submap, to account for any misalignment'''
        
        xrtsmap=self.xrtmap.submap(scale_skycoord(self.aiamaps[0].bottom_left_coord,(1.-xrtoffset)),scale_skycoord(self.aiamaps[0].top_right_coord,(1.+xrtoffset))) #base on AIA for now, hope they're aligned enough (should be)
        cs,hpj_cs,contour=find_centroid_from_map(xrtsmap,show=plot_contour) #helps if this is a submap to start with
        bl_source=SkyCoord(hpj_cs[0].Tx-self.xrt_submap_radius*u.arcsec,hpj_cs[0].Ty-self.xrt_submap_radius*u.arcsec,frame=self.xrtmap.coordinate_frame)
        tr_source=SkyCoord(hpj_cs[0].Tx+self.xrt_submap_radius*u.arcsec,hpj_cs[0].Ty+self.xrt_submap_radius*u.arcsec,frame=self.xrtmap.coordinate_frame)
        #print(bl_source,tr_source)
        regxmap = self.xrtmap.submap(bottom_left=bl_source, top_right=tr_source)
        #  What is the DN/s/px from the region???
        #dur=regxmap.exposure_time.value
        # What pixel binning per dimension
        #chipsum=regxmap.meta['chip_sum']
        #  Get a DN/s/px (non-binned pixels) for the region
        if self.xrt_max: #just get the single brightest pixel
            xdnspx=np.max(regxmap.data)
            self.xrt_area=arcsec_to_cm(self.xrtmap.meta['cdelt2']*u.arcsec)**2 #check if this is correct -- does this take into account the previous binning? might be larger than expected
        else:
            xdnspx=np.mean(regxmap.data) #np.mean(regxmap.data)/dur/chipsum**2
            self.xrt_area=np.product(regxmap.data.shape)*(arcsec_to_cm(self.xrtmap.meta['cdelt2']*u.arcsec))**2
        
        #AIA
        #self.aia_channels = [94,131,171,193,211,335]
        repmap=self.aiamaps[self.aia_channels.index(self.aia_mask_channel)]
        if preflare_difference:
            #subtract preflare image from repmap
            try:
                for m in self.pfmaplist:
                    if m.meta['wavelnth']==self.aia_mask_channel:
                        pfrmap=m
                        dmap_data=repmap.data-pfrmap.data
                        print('difference image',np.min(dmap_data),np.max(dmap_data))
                        repmap=sunpy.map.Map(dmap_data,repmap.meta)
                        break
            except AttributeError:
                print('no preflare maps selected!')
                
#        fig,ax=plt.subplots()
#        repmap.plot(axes=ax)
#        fig.show()
        if type(self.aia_mask) == bool: #hmmm this is an array otherwise..
            if self.aia_mask == False:
                cmask=find_centroid_from_map(repmap,levels=[self.aia_contour],show=plot_contour,return_as_mask=True,transpose=False) #no contour if plot=False
                self.aia_mask=cmask #else throw error
        npix=np.product(self.aia_mask.shape)-np.sum(self.aia_mask)
        self.aia_area=npix*(arcsec_to_cm(repmap.meta['cdelt2']*u.arcsec))**2

        try:
            unmasked=[t.data*~self.aia_mask for t in self.aiamaps]
        except ValueError: #dimension mismatch...
            cmx,cmy=np.shape(self.aia_mask)
            unmasked=[t.data[:cmx,:cmy]*~self.aia_mask for t in self.aiamaps]
        
        datavec=[np.mean(um[um !=0.]) for um in unmasked] #can't take the whole meoan

        #  Correct the AIA data for the degradation
        if not self.aia_calc_degs:
            #use pre-computed value for Sept 12 2020
            deg_dict={94:0.90317732, 131:0.50719532, 171:0.73993289, 193:0.49304311, 211:0.40223458, 335:0.17221724}
            #degs=np.array([0.90317732, 0.50719532, 0.73993289, 0.49304311, 0.40223458, 0.17221724])
            #what if aia_channels are not the usuals?
            degs=[deg_dict[w] for w in self.aia_channels]
        #else: have to run in py37
        
        adn_in=np.array(datavec)/np.array(degs)
        
        # aia_prep was run with /normalize so this should already be in DN/s /px
        if self.aia_exptime_correct:
            durs = [t.meta['exptime'] for t in tmaps]
            adn_in=adn_in/durs
            
        #nustar - assume XRT source is fairly aligned
        bottom_left=SkyCoord(hpj_cs[0].Tx-self.nustar_submap_radius*u.arcsec,hpj_cs[0].Ty-self.nustar_submap_radius*u.arcsec,frame=hpj_cs[0].frame)
        top_right=SkyCoord(hpj_cs[0].Tx+self.nustar_submap_radius*u.arcsec,hpj_cs[0].Ty+self.nustar_submap_radius*u.arcsec,frame=hpj_cs[0].frame)
        if self.verbose:
            print("nustar cutout coords: ", bottom_left,top_right)
        if self.treat_nustar=='separate': #treat A and B as separate channels
            ecountsA,numapsA=nustar_dem_prep(bottom_left,top_right,both='A',how=np.sum,return_maps=True, timerange=self.nustar_timerange,datadir=self.datadir,twenty_seconds=True, use_specfiles=use_specfiles, energies=self.nustar_channels)
            ecountsB,numapsB=nustar_dem_prep(bottom_left,top_right,both='B',how=np.sum,return_maps=True, timerange=self.nustar_timerange,datadir=self.datadir,twenty_seconds=True, use_specfiles=use_specfiles, energies=self.nustar_channels)
            ecounts=[]
            for i,c in enumerate(self.nustar_channels):
                ecounts.append(ecountsA[i])
                ecounts.append(ecountsB[i]) #should match up with chanax
        else:
            ecounts,numaps=nustar_dem_prep(bottom_left,top_right,how=np.sum,return_maps=True, timerange=self.nustar_timerange,datadir=self.datadir,twenty_seconds=True, use_specfiles=use_specfiles, energies=self.nustar_channels) #contour option not activated yet..
        
        stacklist=[adn_in,xdnspx]
        for ec in ecounts:
            stacklist.append(ec)
        self.dn_in=np.hstack(stacklist)
        
        if self.verbose:
            print(self.dn_in)
        
        if plot_maps:
            titles=['AIA','XRT','NuSTAR low','NuSTAR high']
            aplot=sunpy.map.Map(repmap.data*~self.aia_mask,repmap.meta)
            fig=plt.figure(figsize=(12,6))
            plotlist=[aplot,regxmap,mapa,map2a]
            plotlist = list(filter(None, plotlist))
            for i, m in enumerate(plotlist):
                #print(m.data.shape)
                ax = fig.add_subplot(1,len(plotlist), i+1, projection=m.wcs)
                #if m.mask.any():
                #    m.mask=None
                m.plot(axes=ax,title=titles[i])
                xax = ax.coords[0]
                yax = ax.coords[1]
                xax.set_axislabel('')
                yax.set_axislabel('')
                ax.set_yticklabels([])
            fig.show()
        
    def generate_dem_input(self,xray_err=0.2):
        '''return data used for demreg - non-map-based operations on the data done here, including temperature response matrix modification if necessary '''
        nchans=len(self.aia_channels)
        aia_err=generate_errors(0,0,nchans,self.dn_in[:nchans])
        
        #update trmatrix based on areas
        if self.nustar_area == 'AIA':
            try:
                nustar_area_cm2=self.aia_area.value
            except (TypeError,IndexError) as e:
                nustar_area_cm2=self.aia_area.value
        elif self.nustar_area == 'XRT':
            try:
                nustar_area_cm2=self.xrt_area.value
            except IndexError:
                print("No XRT area value supplied, using AIA area")
                nustar_area_cm2=self.aia_area.value
                
        n_nustar_chans=len(self.nustar_channels)
        if self.treat_nustar =='separate':
            n_nustar_chans=2*n_nustar_chans
        
        for i in range(n_nustar_chans):
            self.full_trmatrix[:,nchans+i+1] *=nustar_area_cm2 #need to do this for all nustar channels...
            
        #assume some percentage error for the rest:
        #self.edn_in=np.hstack([aia_err,xray_err*np.copy(self.dn_in)[6:]])
            
        trmatrix_use,dn_use,edn_use=[],[],[]
        
        #okay this whole bit can be made way more efficient
        if self.use_AIA:
            for j,c in enumerate(self.chanax[:6]):
                try:
                    i=self.aia_channels.index(int(c))
                
                    if np.product(self.trmatrix_logt == self.temps) == 0:
                        #print('interplating from %f to %f' % (trmatrix_logt.shape,temps.shape))
                        vinterp=interp_tresp(self.full_trmatrix[:,i],self.trmatrix_logt,np.log10(self.temps))[:,0]
                    else:
                        vinterp=self.full_trmatrix[:,i]
                    trmatrix_use.append(vinterp)
                    dn_use.append(self.dn_in[i])
                    edn_use.append(aia_err[i])
                except ValueError:
                    del self.chanax[self.chanax.index(c)]
        
        if self.use_XRT:
            if np.product(self.trmatrix_logt == self.temps) == 0:
                vinterp=interp_tresp(self.full_trmatrix[:,nchans],self.trmatrix_logt,np.log10(self.temps))[:,0]
            else:
                vinterp=self.full_trmatrix[:,nchans]
            trmatrix_use.append(vinterp)
            dn_use.append(self.dn_in[nchans]*self.xrt_fac)
            edn_use.append(xray_err*dn_use[nchans])
        else:
            self.chanax.pop(nchans)
        
        if self.use_NuSTAR:
            for i in range(n_nustar_chans):
                if np.product(self.trmatrix_logt == self.temps) == 0:
                    vinterp=interp_tresp(self.full_trmatrix[:,nchans+i+1],self.trmatrix_logt,np.log10(self.temps))[:,0]
                else:
                    vinterp=self.full_trmatrix[:,nchans+i+1]
                trmatrix_use.append(vinterp)
                nsdata=self.dn_in[nchans+i+1]*self.nustar_fac
                dn_use.append(nsdata)
                #print('Nustar Errors: ',nsdata,xray_err*nsdata)
                edn_use.append(xray_err*nsdata) #add in counting errors (sqrt N)
            #erate[i]=np.sqrt(np.sum(cnts[gd]))/lvtm
            # (erate**2+(0.2*rate)**2)**0.5)

        else:
            self.chanax.pop(-1)
        #convert back to arrays
        self.trmatrix_use=np.array(trmatrix_use).T
        self.dn_use=np.array(dn_use)
        
        errtype=type(self.min_err)
        if errtype == np.float or errtype == int: #set minimum error threshold in percent: #min_err as nchannels list
            elist = [self.min_err for i in range(len(self.dn_use))]
        else:
            elist=self.min_err
        for i,(du,err) in enumerate(zip(self.dn_use,elist)):
            if (edn_use[i]/du) < err:
                edn_use[i]=du*err
                
        self.edn_use=np.array(edn_use)
        #print('input errors: ',self.edn_use)

    def run_joint_dem(self, dataframe=True):
        '''run DEMreg'''
        tresp_logt=np.log10(self.temps)

        if self.selfnorm:
           gloci=0
        else:
            gloci=1 #weight by loci min
        self.dem,self.edem,self.elogt,self.chisq,self.dn_reg=dn2dem_pos(self.dn_use,self.edn_use,self.trmatrix_use,tresp_logt,self.temps,gloci=gloci)
        
        ratio=(self.dn_reg/self.dn_use)
        naia=len(self.aia_channels)
        if len(self.dn_use) >naia:
            self.aia_ratio=np.mean(ratio[:naia])
            self.xray_ratio=np.mean(ratio[naia:])
        elif len(self.dn_use)==naia:
            self.aia_ratio=np.mean(ratio)
            self.xray_ratio=None
        else: #only x-ray
            self.xray_ratio=np.mean(ratio)
            self.aia_ratio=None
            
        if self.verbose:
            print("Input data and errors:")
            for c,i,e in zip(self.chanax,self.dn_use,self.edn_use):
                print(c,':    ',"{0:.2f}".format(i),"  {0:.2f}".format(e),"  {0:.0f}".format(100*e/i),'%')
            #print("DN input: %s" % self.dn_use)
            print("chisq: %2f" % self.chisq)
            print("AIA DN_reg/DN_in ratio: %s" % self.aia_ratio)
            print("Xray DN_reg/DN_in ratio: %s" % self.xray_ratio)
    
        if dataframe:
            df=self.to_dataframe()
            return df
            
    def to_dataframe(self):
        '''return dataframe'''
        sd=self.__dict__.copy()
        
        def flatten_array(dict_in,key_in):
            dict_in[key_in]=[[dict_in[key_in]]]
            
        sd['chanax']=[self.chanax]
        sd['nustar_timerange']=[self.nustar_timerange]
        sd['edn_use']=[self.edn_use]
        sd['dn_in']=self.dn_use
        sd['initial_weights']=[self.initial_weights]
        flatkeys=['dem','edem','mlogt','elogt','dn_reg','dn_in','trmatrix_use','full_trmatrix','input_trmatrix','min_err']
        for k in flatkeys:
            flatten_array(sd,k)
        delkeys=['dn_use','trmatrix_logt','timerange','numtemps','verbose','self','use_XRT','use_NuSTAR','temps','dtemps','xrtmap','aiamaps','pfmaplist','aia_mask','aia_channels','nustar_channels','treat_nustar']
        for k in delkeys:
            del sd[k]

        df=pd.DataFrame(sd)
     
        return df
    
    def plot_dem_errorbar(self,yaxis_range=[19,26],oplot_loci=True,sf=6,abs=False,perKEV=True,log=True, plot_xerr=False,use_matplotlib=False):
        title='Joint DEM,%s - %s' % tuple(self.timerange) # make this make sense based on input params
        if abs:
            pval=np.abs(self.dem)
        else:
            pval=self.dem
            
        if not perKEV: #multiply by size of bin in K
            yaxis_title_tag=''
            #have to do the same with the error
            pval=self.dem*self.dtemps
            yerr=self.edem*self.dtemps
            sf=0
        else:
            yaxis_title_tag='K<sup>-1</sup>'
            yerr=self.edem
        if plot_xerr:
            xerr=self.elogt
        else:
            xerr=None
       
        if use_matplotlib:
            fig,ax=plt.subplots(figsize=[9,6])
            ax.errorbar(self.mlogt,pval,xerr=xerr,yerr=yerr,label='DEM')
            if oplot_loci:
                vecs=self.dn_use/self.trmatrix_use
                for i in range(vecs.shape[1]):
                    ax.plot(self.mlogt,vecs[1:,i]/(10**sf),label=self.chanax[i])
            if log:
                ax.set_yscale('log')
            ax.set_ylim([float(10**yaxis_range[0]),float(10**yaxis_range[1])])
            ax.set_title(title)
            if yaxis_title_tag != '':
                yaxis_title_tag='K$^{-1}$'
            ax.set_ylabel('DEM cm$^{-5}$ '+yaxis_title_tag)
            ax.set_xlabel('log$_{10}$ T (K)')
            ax.legend(loc='upper left',fontsize=10)
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.mlogt,y=pval,error_x=dict(type='data',array=xerr),error_y=dict(type='data',array=yerr),name='DEM'))
            if oplot_loci:
                vecs=self.dn_use/self.trmatrix_use
                for i in range(vecs.shape[1]):
                    fig.add_trace(go.Scatter(x=self.mlogt,y=vecs[:,i]/(10**sf),name=self.chanax[i]))
            fig.update_layout(yaxis_range=[10**yaxis_range[0],10**yaxis_range[1]],title=title,yaxis_title='DEM cm<sup>-5</sup> '+yaxis_title_tag,xaxis_title='log<sub>10</sub> T (K)')
            if log:
                fig.update_layout(yaxis_type='log',yaxis_range=yaxis_range)
        return fig
        
    def plot_loci(self,yaxis_range=[23,29]):
        title='EM loci curves,%s - %s' % tuple(self.timerange) # make this make sense based on input params
        fig = go.Figure()
        vecs=self.dn_use/self.trmatrix_use
        #yaxis_range=[19,26]
        for i in range(vecs.shape[1]):
            fig.add_trace(go.Scatter(x=self.mlogt,y=vecs[:,i],name=self.chanax[i]))
        fig.update_layout(yaxis_type='log',yaxis_range=yaxis_range,title=title,yaxis_title='EM cm<sup>-5</sup> K<sup>-1</sup>',xaxis_title='log<sub>10</sub> T (K)')
        return fig

    def DN_in_v_reg(self, use_matplotlib=False):
        title='DN_reg vs DN_in, %s - %s' % tuple(self.timerange)
        lb=int(np.min(np.log10(self.dn_use)))-1
        ub=int(np.max(np.log10(self.dn_use)))+1
        
        if use_matplotlib:
            fig,ax=plt.subplots(figsize=[9,6])
            for i in range(len(self.chanax)):
                ax.scatter(self.dn_use[i],self.dn_reg[i],marker="P",s=150,label=self.chanax[i])
            ax.plot(np.logspace(lb,ub,100),np.logspace(lb,ub,100),'--', label='1:1')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_title(title)
            ax.set_ylabel('DN_reg')
            ax.set_xlabel('DN_in')
            ax.legend(loc='upper left',fontsize=10)
        else:
            fig=go.Figure()
            for i in range(len(self.chanax)):
                fig.add_trace(go.Scatter(x=[self.dn_use[i]],y=[self.dn_reg[i]],hovertext="ratio: %s" % np.round(self.dn_reg[i]/self.dn_use[i],3),mode='markers',marker_symbol='cross',marker_size=10,name=self.chanax[i]))
            fig.add_trace(go.Scatter(x=np.logspace(lb,ub,100),y=np.logspace(lb,ub,100),name='1:1'))
            fig.update_layout(xaxis_type='log',yaxis_type='log',xaxis_title='DN_in',yaxis_title='DN_reg',title=title)
            fig.update_layout(yaxis = dict(showexponent = 'all',exponentformat = 'e'),xaxis = dict(showexponent = 'all',exponentformat = 'e'))
        return fig
        
    def plot_trmatrix(self,yaxis_range=[-30,-22],use_matplotlib=False):
        if use_matplotlib:
            fig,ax=plt.subplots(figsize=[9,6])
        else:
            fig=go.Figure()
        vecs=self.trmatrix_use
        title='Temperature Response Matrix used for DEM calculation'
        #yaxis_range=[19,26]
        if use_matplotlib:
            for i in range(vecs.shape[1]):
                ax.plot(self.mlogt,vecs[1:,i],label=self.chanax[i])
            ax.set_yscale('log')
            ax.set_ylim([10**yaxis_range[0],10**yaxis_range[1]])
            ax.set_title(title)
            ax.set_ylabel('Temperature Response DN s$^{-1}$ px$^{-1}$ cm$^{5}$ K$^{-1}$')
            ax.set_xlabel('log$_{10}$ T (K)')
            ax.legend(loc='upper right',fontsize=10)
        else:
            for i in range(vecs.shape[1]):
                fig.add_trace(go.Scatter(x=self.mlogt,y=vecs[:,i],name=self.chanax[i]))
            fig.update_layout(yaxis_type='log',yaxis_range=yaxis_range,title=title,yaxis_title='Temperature Response DN s<sup>-1</sup> px<sup>-1</sup> cm<sup>5</sup> K<sup>-1</sup>',xaxis_title='log<sub>10</sub> T (K)')
        return fig
        
    def plot_contribution_function(self,log=False, use_matplotlib=False):
        ''' plot DEM*tresp'''
        title='EM Contribution Functions,%s - %s' % tuple(self.timerange) # make this make sense based on input params
        EMval=np.abs(self.dem)#*self.dtemps #try this without the multiplication... units less meaningful but might show shift towards/away high energies
        vecs=[]
        
        for i,c in enumerate(self.chanax):
            vec=EMval*self.trmatrix_use[1:,i]
            vecs.append(vec)
        
        if use_matplotlib:
            fig,ax=plt.subplots(figsize=[9,6])
            for i,c in enumerate(self.chanax):
                ax.plot(self.dtemps,vecs[i],label=c)
            if log:
                ax.set_yscale('log')
            #ax.set_ylim([10**yaxis_range[0],10**yaxis_range[1]])
            ax.set_title(title)
            ax.set_ylabel('DN s$^{-1}$ px$^{-1}$')
            ax.set_xlabel('log$_{10}$ T (K)')
            ax.legend(loc='upper right',fontsize=10)
        else:
            fig = go.Figure()
            for i,c in enumerate(self.chanax):
                fig.add_trace(go.Scatter(x=self.temps,y=vecs[i],name=c)) #try without multiplying by dtemps for now
            fig.update_layout(yaxis = dict(showexponent = 'all',exponentformat = 'e'),xaxis = dict(showexponent = 'all',exponentformat = 'e'),title=title,yaxis_title='DN s<sup>-1</sup> px<sup>-1</sup>',xaxis_title='log<sub>10</sub> T (K)')
            if log:
                fig.update_layout(yaxis_type='log')
        
        if self.verbose:
            df0=pd.DataFrame({'Channel':self.chanax,'Integral':np.sum(vecs,axis=1),'DN_in':self.dn_use,'DN_reg':self.dn_reg})
            df0['DN_reg - DN_in']=df0.DN_reg-df0.DN_in
            df0['Integral - DN_in']=df0.Integral-df0.DN_in
            df0['Integral - DN_reg']=df0.Integral-df0.DN_reg
            display(df0[['Channel','Integral','DN_in','DN_reg','DN_reg - DN_in','Integral - DN_in','Integral - DN_reg']])

        return fig
        
    def plot_four_maps(self,crpix1_off=20,crpix2_off=25):
        '''for overview figure for paper - Nustar high, low, XRT, AIA 193, contour of mask? '''
        a193=self.aiamaps[3]
        xrtmc=hand_coalign(self.xrtmap,crpix1_off,crpix2_off)
        xrtm=xrtmc.submap(a193.bottom_left_coord,a193.top_right_coord)
        width=a193.top_right_coord.Tx-a193.bottom_left_coord.Tx
        height=a193.top_right_coord.Ty-a193.bottom_left_coord.Ty
        _,hpj_cs,_=find_centroid_from_map(xrtm,show=False)
        bottom_left=SkyCoord(hpj_cs[0].Tx-self.nustar_submap_radius*u.arcsec,hpj_cs[0].Ty-self.nustar_submap_radius*u.arcsec,frame=hpj_cs[0].frame)
        top_right=SkyCoord(hpj_cs[0].Tx+self.nustar_submap_radius*u.arcsec,hpj_cs[0].Ty+self.nustar_submap_radius*u.arcsec,frame=hpj_cs[0].frame)
        _,nmaps=nustar_dem_prep(bottom_left,top_right,how=np.sum,return_maps=True, timerange=self.nustar_timerange,datadir=self.datadir,twenty_seconds=True, use_specfiles=False,energies=[[1.5,2.5],[2.5,4]]) #contour option not activated yet..
        nlo,nhi=nmaps

        fig=plt.figure(figsize=(10,11))
        ax0 = fig.add_subplot(221, projection=nlo.wcs)
        clo=nlo.plot(axes=ax0,title='NuSTAR CHU A 1.5-2.5 KeV')
        nlo.draw_rectangle(a193.bottom_left_coord,width,height,axes=ax0)
        fig.colorbar(clo, ax=ax0)
        ax1 = fig.add_subplot(222, projection=nhi.wcs)
        cn=nhi.plot(axes=ax1,title='NuSTAR CHU A 2.5-4 KeV',vmin=nlo.data.min(),vmax=nlo.data.max())
        nhi.draw_rectangle(a193.bottom_left_coord,width,height,axes=ax1)
        fig.colorbar(cn, ax=ax1)
        ax2 = fig.add_subplot(223, projection=xrtm.wcs)
        cxrt=xrtm.plot(axes=ax2,title='XRT Be-Thin')
        fig.colorbar(cxrt, ax=ax2)
        ax3 = fig.add_subplot(224, projection=a193.wcs)
        caia=a193.plot(axes=ax3,title='AIA 193$\AA$')
        fig.colorbar(caia, ax=ax3)
        ax1.yaxis.set_visible(False)
        ax2.yaxis.set_visible(False)
        #for ax in [ax1,ax3]:
            #yax = ax.coords[1]
            #ax.yaxis.set_visible(False)#axislabel('')
            #yax.set_ticklabels([])
            #ax.set_yticklabels([])
        return fig
    
    def parameter_search(self,aia_contours=[75,80,85],xrt_submap_radii=[5,10],xrt_facs=[1,1.5,2,2.5],nustar_facs=[.01,.1,1,2,5,10,15,25],nustar_areas=['AIA','XRT'],min_errs=[.1,.15,.2]):
        ''' find the best possible configuration. start with self=joint_DEM()'''
        start_time=dt.now()
        self.verbose=False
        try:
            aa=self.aiamaps
        except AttributeError:
            self.select_joint_maps()
        results_dfs=[]
        for ac in aia_contours:
            for nr in xrt_submap_radii:
                self.xrt_submap_radius=nr
                self.aia_contour=ac
                self.prep_joint_maps()
                for xf in xrt_facs:
                    for nf in nustar_facs:
                        for na in nustar_areas:
                            for me in min_errs:
                                self.xrt_fac=xf
                                self.nustar_fac=nf
                                self.nustar_area=na
                                self.min_err=me
                                self.generate_dem_input()
                                df=self.run_joint_dem() #not the best of practices yet here we are
                                results_dfs.append(df)
                                self.reset_trmatrix()
                                self.reset_chanax()
        print(dt.now()-start_time)
        results=pd.concat(results_dfs)
        results.reset_index(inplace=True)
        return results
        
    def area_to_volume(self,thirdD,instr='AIA'):
        '''use area of emission to calculate a volume, given a 3rd dimension input (defaults to square root of area, so assuming a cube)'''
        if instr == 'AIA':
            area_cm2=self.aia_area #this accounts fro the number of pixels
            px2=arcsec_to_cm(self.aiamaps[0].meta['cdelt2']*u.arcsec)**2
        elif instr == 'XRT':
            area_cm2=self.xrt_area #this accounts fro the number of pixels
            px2=arcsec_to_cm(self.xrtmap.meta['cdelt2']*u.arcsec)**2
        if not thirdD:
            thirdD=np.sqrt(area_cm2)
        #check that thirdD and area are in same units; if not, convert
        a_units=np.sqrt(1.*area_cm2.unit).unit
        if thirdD.unit != a_units:
            thirdD=thirdD.to(a_units)
        volume=thirdD*area_cm2
        return volume, px2

    def thermal_energy_content(self,thirdD=False,volume_instr='AIA',fill_factor=1.0, plot=False,unit=u.Joule):
        '''E = 3NkT, n= N/V
        E= 3kT sqrt(EM*V*f) #St.Hilaire & Benz 2005 eq 17, EM has units cm^-3
        assume filling factor f=1
        units:
          kT: J/K * K = J  [mass][length]^2[time]^-2
          EM*V*px2: cm^-5 * cm^3 *cm^2 = unitless
        returns E in Joules
        '''
        #T=10**logt
        #tlist=list(self.temps)
        #tidx= tlist.index(min(tlist, key=lambda x: abs(x - T)))
        V,px2=self.area_to_volume(thirdD=thirdD,instr=volume_instr) #V in cm^3, px^2 in cm^2
        dem_scaled=self.dem*self.dtemps*((1*u.cm)**-5)#units cm^-5 #need to do temperature cutoff?
        n_eff=dem_scaled * V * px2 * fill_factor# sqrt([cm^-5 * cm^3 * cm^2]) unitless
        E_joules= 3*constants.k*(u.Joule/u.K) * np.array(self.temps[1:])*u.K * np.sqrt(n_eff)
        #print(V.unit,px2.unit,dem_scaled.unit,n_eff.unit,E_joules.unit)
        if unit != u.Joule:
            E_out=E_joules.to(unit)
        else:
            E_out=E_joules
        
        if plot:
            title="Thermal energy calculated for a " + "{0:.2f}".format((cm_to_arcsec(V**(1/3)))**3) + " volume" #use {0:.2e} for exponential format
            colorscale = plotly.colors.sequential.Plasma
            cdict=[[i/(len(colorscale)-1),c] for i,c in enumerate(colorscale)]
            colors=[get_continuous_color(cdict,en.value/np.max(E_out).value) for en in E_out]
            timetext=[dict(xref='paper',yref='paper',x=0.5, y=1.05,showarrow=False,text ="%s - %s" % tuple(self.timerange))]
            fig=go.Figure()
            fig.add_trace(go.Bar(x=self.temps,y=E_out,name=volume_instr,width=.9*np.array(self.dtemps), marker_color=colors))
            fig.update_layout(xaxis_title='Temperature (K)',yaxis = dict(title='Thermal Energy (%s)' % E_out.unit,showexponent = 'all',exponentformat = 'e'),title=title,annotations=timetext)
            return E_out,fig
        else:
            return E_out #should be a vector
        
    def compare_thermal_energy(self,thirdD=False,fill_factors=[1.0,1.0]):
        ''' compare AIA and Xray contributions to thermal energy content'''
        Eth_AIA=self.thermal_energy_content(thirdD=thirdD,fill_factor=fill_factors[0])
        Eth_Xray=self.thermal_energy_content(volume_instr='XRT',thirdD=thirdD,fill_factor=fill_factors[1])
        fig=go.Figure() #do this as a bar chart... binsize are dtemps
        fig.add_trace(go.Bar(x=self.temps[1:],y=Eth_AIA,name='EUV (AIA)'))
        fig.add_trace(go.Bar(x=self.temps[1:],y=Eth_Xray,name='Xray (XRT & NuSTAR)'))
        fig.update_layout(xaxis_title='log T (K)',yaxis = dict(title='Thermal Energy (%s)' % Eth_AIA.unit,showexponent = 'all',exponentformat = 'e'))
        return fig
