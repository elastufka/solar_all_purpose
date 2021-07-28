0 #######################################
#dem_solve.py
# Erica Lastufka 15/07/2021

#######################################

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
#from scipy.interpolate import interp1d
from skimage.transform import downscale_local_mean
#from aia_utils import aia_maps_tint, single_time_indices
from dn2dem_pos import dn2dem_pos
from dem_utils import *
from sunpy_map_utils import find_centroid_from_map, arcsec_to_cm, scale_skycoord
import pickle
import plotly.graph_objects as go
from visualization_tools import all_six_AIA

class DEM_solve:
    def __init__(self,aiamaps, full_trmatrix=False,trmatrix_logt=False,submap=False,tstart=5.0,tend=6.9,numtemps=42,aia_mask=False,aia_contour=80,aia_mask_channel=211,aia_calc_degs=False,aia_exptime_correct=False,method='demreg',binning=1.,min_err=False,selfnorm=False,verbose=True,datadir='/Users/wheatley/Documents/Solar/NuStar/orbit8/Xray/',aia_channels = [94,131,171,193,211,335]):
        '''run DEM algorithm for given configuration. Input is list of AIA level 1.5 corrected sunpy maps, correctly ordered (for now). If no temperature response matrix is given as input, this is calculated from IDL given the map date. Would be nice once this moves to python since my sswidl installation doesn't have an up-to-date database...'''
            
        for key, value in locals().items():
            setattr(self, key, value)
                   
        if self.method == 'demreg': #don't need this for DeepEM or sparseEM
            if not full_trmatrix or not trmatrix_logt: #also don't need these if usind deepEM
                self.tresp_matrix_from_IDL() #load_default_trmatrix()
                self.og_trmatrix=np.copy(self.full_trmatrix)

        if 'deep' not in self.method:    self.temps,self.dtemps,self.mlogt=calc_temp_vars(tstart=self.tstart,tend=self.tend,num=self.numtemps)
    
    def reset_trmatrix(self):
        self.full_trmatrix=np.copy(self.og_trmatrix)
        
    def get_timedepend_date(self):
        mapdate=self.aiamaps[0].meta['date-obs']
        if type(mapdate)==str: #need as string
            datestr=mapdate[:mapdate.find('T')]
        else:
            datestr=str(mapdate.date())
        return datestr
        
    def dem_units(self):
        if 'reg' in self.method:
            self.dem_units='DEM cm<sup>-5</sup> K<sup>-1</sup>'
        else:
            self.dem_units='DEM cm<sup>-5</sup>'
        
    def tresp_matrix_from_IDL(self):
        ''' ;  tresp=aia_get_response(/temperature,/dn,/eve,timedepend_date='01-Jul-2010')
         ids=[0,1,2,3,4,6]
         channels=tresp.channels[ids]
         logt=tresp.logte
         tr=tresp.all[*,ids]
         units=tresp.units '''
        import pidly
        datestr=self.get_timedepend_date()
        ids=[0,1,2,3,4,6]
        idl = pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
        idl('ts',datestr)
        idl('ids',ids)
        idl('tresp=aia_get_response(/temperature,/dn,/eve,timedepend_date=ts)')
        idl('channels=tresp.channels[ids]') #do I need this?
        idl('logt=tresp.logte')
        idl('tr=tresp.all[*,ids]')
        idl('units=tresp.units')
        self.full_trmatrix=idl.tr
        self.trmatrix_logt=idl.logt
        self.tresp_units=idl.units

    def prep_maps(self,plot_maps=True):
        '''Prepare data within a given submap, contour, etc for use with the DEM inversion. If no mask or contour specified, use a 2D inversion
        
        should i make this do generally all map manipulations, before or after the inversion? to accomodate deepEM'''
        if self.binning != 1: #bin maps
            binned_data=[downscale_local_mean(m.data, (self.binning,self.binning),clip=True) for m in self.aiamaps]
            newmaps=[sunpy.map.Map(d,m.meta) for d,m in zip(binned_data,self.aiamaps)]
            self.aiamaps=newmaps
            
        repmap=self.aiamaps[self.aia_channels.index(self.aia_mask_channel)]
        if not self.aia_contour: # this is okay to then ignore the submap argument later, since if you input a mask it should match the input maps already and further cropping that region would mess things up
            #do 2D dem, so just take a submap if there is one
            try: #top right and bottom left skycoords
                tmaps=[m.submap(self.submap[0],self.submap[1]) for m in self.aiamaps]
            except (IndexError, TypeError) as e:
                tmaps=self.aiamaps
            npix=np.product(tmaps[0].data.shape)
        else:
            cmask=find_centroid_from_map(repmap,levels=[self.aia_contour],show=plot_maps,return_as_mask=True,transpose=False) #no contour if plot=False
            self.aia_mask=cmask #else throw error
            
        if type(self.aia_mask) == np.ndarray:
            npix=np.product(self.aia_mask.shape)-np.sum(self.aia_mask)
            try:
                unmasked=[t.data*~self.aia_mask for t in self.aiamaps]
            except ValueError: #dimension mismatch or no mask?
                cmx,cmy=np.shape(self.aia_mask)
                unmasked=[t.data[:cmx,:cmy]*~self.aia_mask for t in self.aiamaps]

        try:
            datavec=[np.mean(um[um !=0.]) for um in unmasked] #can't take the whole meoan
        except UnboundLocalError: #unmasked not defined
            datavec=[t.data for t in tmaps]
            
        self.aia_area=npix*(arcsec_to_cm(repmap.meta['cdelt2']*self.binning*u.arcsec))**2

        #  Correct the AIA data for the degradation
        if self.aia_calc_degs:
            #use pre-computed value for Sept 12 2020
            deg_dict={94:0.90317732, 131:0.50719532, 171:0.73993289, 193:0.49304311, 211:0.40223458, 335:0.17221724}
            #degs=np.array([0.90317732, 0.50719532, 0.73993289, 0.49304311, 0.40223458, 0.17221724])
            #what if aia_channels are not the usuals?
            degs=[deg_dict[w] for w in self.aia_channels]
            #else: have to run in py37
            datavec=np.array(datavec)/np.array(degs) #what if datavec is list of arrays? will this still work?
        
        # aia_prep was run with /normalize so this should already be in DN/s /px
        if self.aia_exptime_correct:
            durs = [t.meta['exptime'] for t in tmaps]
            datavec=[np.array(dv)/np.array(d) for dv,d in zip(datavec,durs)]
                   
        self.dn_in=np.array(datavec)
        try:
            self.nx=np.array(datavec[0]).shape[0]
            self.ny=np.array(datavec[0]).shape[1]
        except IndexError:
            self.nx=1
            self.ny=1
        
        if self.verbose:
            print(self.dn_in)

        if plot_maps:
            fig=all_six_AIA(self.aiamaps,draw_contour=self.aia_contour,unmask=False)
            fig.show()

        
    def generate_demreg_input(self,xray_err=0.2):
        '''return data used for demreg - non-map-based operations on the data done here, including temperature response matrix modification if necessary '''
        try:
            nf,nx,ny=self.dn_in.shape #dn_in is nf,nx,ny. need to reshape to nx,ny,nf
            self.dn_in=self.dn_in.reshape(nx,ny,nf) #hope this works right
        except ValueError:
            nf=6
            nx,ny=0,0 #does 1,1 work too?
        
        self.edn_in=generate_errors(nx,ny,6,self.dn_in) #input dimensions have to be nx,ny,nf
        trmatrix_use=[]
        if np.product(self.trmatrix_logt == self.temps) == 0:
            vinterp=interp_tresp(self.full_trmatrix.T,self.trmatrix_logt,np.log10(self.temps))
        else:
            vinterp=self.full_trmatrix[:,i]
        print(vinterp.shape)
        self.full_trmatrix=vinterp #shape ntemps, 6
        
        if self.min_err !=False: #set minimum error threshold in percent:
            for i,du in enumerate(self.dn_in):
                if (self.edn_in[i]/du) < self.min_err:
                    self.edn_in[i]=du*self.min_err
       
    def run_dem(self):
        '''wrapper for all the run_method functions basically'''
        if self.method == 'demreg':
            self.generate_demreg_input()
            try:
                self.run_demreg()
            except AttributeError: #no errors were generated so do that
                self.generate_dem_input()
                self.run_demreg()
        elif self.method == 'sparse':
            self.run_sparse()
        elif 'deep' in self.method:
            self.run_deepEM()

    def run_demreg(self, dataframe=True):
        '''run DEMreg'''
        #initial weights:
        tresp_logt=np.log10(self.temps)

        if self.selfnorm:
            gloci=0
        else:
            gloci=1 #weight by loci min
        self.dem,self.edem,self.elogt,self.chisq,self.dn_reg=dn2dem_pos(self.dn_in,self.edn_in,self.full_trmatrix,tresp_logt,self.temps,gloci=gloci) #does this need to be different when using a map?
        
        ratio=(self.dn_reg/self.dn_in)
        self.aia_ratio=np.nanmean(ratio)
            
        if self.verbose:
            if self.dn_in.shape == (6,):
                print("Input data and errors:")
                for c,i,e in zip(self.aia_channels,self.dn_in,self.edn_in):
                    print(c,':    ',"{0:.2f}".format(i),"  {0:.2f}".format(e),"  {0:.0f}".format(100*e/i),'%') #don't do this for 2d
                chisq=self.chisq
            else:
                chisq=np.mean(self.chisq)
            #print("DN input: %s" % self.dn_use)
            print("chisq: %2f" % chisq)
            print("DN_reg/DN_in ratio: %s" % self.aia_ratio)
    
        if dataframe:
            df=self.to_dataframe()
            return df
            
    def run_sparse(self,savname='demtest.sav',adaptive_tolfac=False):
        '''run sparse_dem (basis pursuit). currently only works for 2D maps '''
        #first trim maps into fov, stick in list
        import pidly
        start_time=dt.now()
        idl = pidly.IDL('/Users/wheatley/Documents/Solar/sswidl_py.sh')
        idl('.compile /Users/wheatley/Documents/Solar/DEM/tutorial_dem_webinar/aia_sparse_em_init.pro')
        #idl('.compile /Users/wheatley/Documents/Solar/occulted_flares/code/sparse_dem_prep.pro')
        #idl('group6', group6)
        mapdata=self.dn_in #thise have to be 2d arrays! can be masked (how does IDL bin things?)
        headarr=[{'exptime':m.meta['exptime']} for m in self.aiamaps]
        datestr=self.get_timedepend_date()
        idl('submap',mapdata) #now submap is actually the Map.submap.data
        if type(self.dn_in[0])==np.ndarray: #datavec is already array...
            idl('binning',self.binning)
        else:
            idl('binning',1.)
        idl('lgTmin',self.tstart)   # minimum for lgT axis for inversion
        dlgt=np.mean([np.log10(self.dtemps[i+1])-np.log10(self.dtemps[i]) for i in range(len(self.dtemps[:-1]))])
        idl('dlgT',dlgt)     # width of lgT bin
        idl('nlgT',self.numtemps)    # number of lgT bins
        #idl('tresp',self.full_trmatrix.T) #think it's expecting this size
        idl('datestr',datestr)
        idl('headarr',headarr) #how to get this in the proper format? too long to send to IDL... what part do i actually need
        #only uses exposure time
        
        idl('s={IMG:submap,OINDEX:headarr,BINNING: binning}')
        idl('aia_sparse_em_init, use_lgtaxis=findgen(nlgT)*dlgT+lgTmin, bases_sigmas=[0.0,0.1,0.2],timedepend_date=datestr')
        idl('lgtaxis = aia_sparse_em_lgtaxis()')
        idl('fname',savname) #is this necessary?
        if adaptive_tolfac:
            idl('result=run_sparse_dem(s,lgtaxis,fname,/adaptive_tolfac)')
        else:
            idl('result=run_sparse_dem(s,lgtaxis,fname)')
        
        #outputs are ,emcube,status, image, coeff,lgtaxis
        result=idl.result
        emcube=np.flip(result['emcube']*1e26,axis=0) #needs to be 2d array, else just use reverse of list
        tpassed=dt.now()-start_time
        print('Sparse DEM inversion done in %s' % tpassed)
        #get out the result
        self.dem=emcube #ntemps, nx,ny
        self.status=result['status'] #nx,ny
        self.coeff=result['coeff'] #n?, nx,ny
        
    def run_deepEM(self):
        '''apply trained DeepEM models directly to images (images must be 8x8 binned full disk) - cutouts can be done later'''
        
        #make sure maps are full disk and 8x8 binned
        nx,ny=self.aiamaps[0].data.shape
        if nx != 512 or ny !=512:
            binned_data=np.array([downscale_local_mean(m.data, (8,8),clip=True) for m in self.aiamaps])
            self.datavec=binned_data
            if binned_data.shape[1] and binned_data.shape[2] != 512:
                print("AIA maps must be full disk!")
            return None
        else:
            self.datavec=np.array([m.data for m in self.aiamaps])
        
        if 'reg' in self.method:
            dem_model_file = '/Users/wheatley/Documents/Solar/DEM/DeepEM/DeepEM_CNN_HelioML_reg.pth'
        else:
            dem_model_file = '/Users/wheatley/Documents/Solar/DEM/DeepEM/DeepEM_CNN_HelioML.pth'
        
        import torch
        import torch.nn as nn

        model = nn.Sequential(
            nn.Conv2d(6, 300, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(300, 300, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(300, 18, kernel_size=1)) #do I need this?

        model.load_state_dict(torch.load(dem_model_file,map_location=torch.device('cpu')))

        model.eval()
        dem_pred=model(torch.from_numpy(self.datavec.reshape(1,6,512,512)).type(torch.FloatTensor))
        
        dp=dem_pred.cpu().detach().numpy()
        dp_scaled=1e25*(dp*dp)
        #print(dp_scaled.shape)
        self.dem=dp_scaled
        self.temps=10**np.array([5.5, 5.6, 5.7, 5.8, 5.9, 6. , 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
        6.8, 6.9, 7. , 7.1, 7.2]) #this is what deepEM was trained on
        #do any cropping and mask applications after ....
            
    def to_dataframe(self):
        '''return dataframe'''
        sd=self.__dict__.copy()
        
        def flatten_array(dict_in,key_in):
            try:
                dict_in[key_in]=[[dict_in[key_in]]]
            except KeyError:
                pass
        flatkeys=['dem','edem','mlogt','elogt','dn_reg','dn_in','edn_in','trmatrix_use','full_trmatrix','status','coeff','chisq']
        for k in flatkeys:
            flatten_array(sd,k)
        delkeys=['trmatrix_logt','numtemps','verbose','min_err','self','temps','dtemps','aiamaps','aia_channels','og_trmatrix','aia_calc_degs','aia_exptime_correct']
        for k in delkeys:
            del sd[k]

        df=pd.DataFrame(sd)
     
        return df
        
    def to_dem_inspect(self,targetid=False):
        '''save json to be used with DEM_inspect app'''
        if not targetid:
            #use datetime as target
            targetid=self.aiamaps[0].meta['date-obs']
        
        dd={}
        print(self.dem.shape)
        for i,t in enumerate(np.round(np.log10(self.temps),2)):
            dd['dem_'+str(t)]=[self.dem[i]]
        
        #lb=min(enumerate(self.aia_tresp_logt), key=lambda x: abs(x[1]-np.log10(self.temps)[0]))[0] #index
        #ub=min(enumerate(self.aia_tresp_logt), key=lambda x: abs(x[1]-np.log10(self.temps)[-1]))[0] #index
        #aia_logt_range=self.aia_tresp_logt[lb:ub]
#        for i,t in enumerate(aia_logt_range):
#            for j,w in enumerate(self.aia_channels):
#                #datavec has dimensions nx,ny,6
#                dd['loci_'+str(w)+'_'+str(t)]=[np.array(self.datav[:,:,j])/trmatrix[i,j]] #this makes json files that are too big to be written!
            
        #mean
        dd['dem_mean']=[np.mean(np.mean(self.dem,axis=1),axis=1)]

        if 'edem' in self.__dict__.keys():
            for i,t in enumerate(np.round(np.log10(self.temps),2)):
                dd['edem_'+str(t)]=[self.edem[i]]
        
        if 'elogt' in self.__dict__.keys():
            for i,t in enumerate(np.round(np.log10(self.temps),2)):
                dd['elogt_'+str(t)]=[self.elogt[i]]
        
        if 'status' in self.__dict__.keys():
            dd['status']=[self.status]
        
        if 'coeff' in self.__dict__.keys():
            dd['coeff']=[self.coeff]
        
        df=pd.DataFrame(dd,index=pd.Index([targetid]))
        #errors?
        outfilename=self.method.lower()+'_'+str(self.binning)+'x'+str(self.binning)+'_binned.json'
        df.to_json(self.datadir+outfilename)
        #self.datav.to_json(self.datadir+'datavecs_'+outfilename)
        #should save the trmatrix too, once they're different...
    
    def plot_dem_errorbar(self,yaxis_range=[20,30],oplot_loci=True,abs=False,perKEV=False,log=True):
        datestr=self.get_timedepend_date()
        title='%s DEM inversion,%s' % (self.method,datestr)# make this make sense based on input params
        yaxis_title=self.dem_units
        fig = go.Figure()
        if abs:
            pval=np.abs(self.dem)
        else:
            pval=self.dem
            
        #deal with 2D
        if self.nx !=1 or self.ny!=1:
            pval=np.nanmean(np.nanmean(pval,axis=1),axis=1) #1 or 0?
            try:
                eval=np.nanmean(np.nanmean(self.edem,axis=1),axis=1)
            except KeyError:
                pass
            
        if self.method == 'demreg' and not perKEV: #multiply by size of bin in K
            yaxis_title='DEM cm<sup>-5</sup>'
            #have to do the same with the error
            pval=pval*self.dtemps
            yerr=eval*self.dtemps
            xerr=self.elogt
            #and loci curves?...
        else:
            yerr=np.zeros(len(pval))
            xerr=np.zeros(len(pval))
        fig.add_trace(go.Scatter(x=self.mlogt,y=pval,error_x=dict(type='data',array=xerr),error_y=dict(type='data',array=yerr),name='DEM'))
        if oplot_loci:
            vecs=self.dn_in/self.full_trmatrix
            if self.method == 'demreg' and not perKEV:
                vecs*=self.dtemps
            #yaxis_range=[19,26]
            for i in range(vecs.shape[1]):
                fig.add_trace(go.Scatter(x=self.mlogt,y=vecs[:,i],name=self.aia_channels[i]))
        fig.update_layout(yaxis_range=[10**yaxis_range[0],10**yaxis_range[1]],title=title,yaxis_title=yaxis_title,xaxis_title='log<sub>10</sub> T (K)')
        if log:
            fig.update_layout(yaxis_type='log',yaxis_range=yaxis_range)
        return fig
        
    def plot_loci(self,yaxis_range=[23,29]):
        datestr=self.get_timedepend_date()
        title='EM loci curves,%s' % datestr # make this make sense based on input params
        fig = go.Figure()
        vecs=self.dn_in/self.full_trmatrix
        #yaxis_range=[19,26]
        for i in range(vecs.shape[1]):
            fig.add_trace(go.Scatter(x=self.mlogt,y=vecs[:,i],name=self.aia_channels[i]))
        fig.update_layout(yaxis_type='log',yaxis_range=yaxis_range,title=title,yaxis_title='EM cm<sup>-5</sup> K<sup>-1</sup>',xaxis_title='log<sub>10</sub> T (K)')
        return fig

    def DN_in_v_reg(self):
        datestr=self.get_timedepend_date()
        title='DN_reg vs DN_in, %s' % datestr
        if self.nx !=1 or self.ny !=1:
            dn_in=np.nanmean(np.nanmean(self.dn_in,axis=1),axis=1)
            dn_reg=np.nanmean(np.nanmean(self.dn_reg,axis=1),axis=1)
        else:
            dn_in=self.dn_in
            dn_reg=self.dn_reg
        lb=int(np.min(np.log10(dn_in)))-1
        ub=int(np.max(np.log10(dn_in)))+1
        fig=go.Figure()
        for i in range(6):
            fig.add_trace(go.Scatter(x=[dn_in[i]],y=[dn_reg[i]],hovertext="ratio: %s" % np.round(dn_reg[i]/dn_in[i],3),mode='markers',marker_symbol='cross',marker_size=10,name=self.aia_channels[i]))
        fig.add_trace(go.Scatter(x=np.logspace(lb,ub,100),y=np.logspace(lb,ub,100),name='1:1'))
        fig.update_layout(xaxis_type='log',yaxis_type='log',xaxis_title='DN_in',yaxis_title='DN_reg',title=title)
        fig.update_layout(yaxis = dict(showexponent = 'all',exponentformat = 'e'),xaxis = dict(showexponent = 'all',exponentformat = 'e'))
        return fig
        
    def plot_trmatrix(self,yaxis_range=[-26,-19]):
        fig=go.Figure()
        vecs=self.full_trmatrix
        #yaxis_range=[19,26]
        for i in range(vecs.shape[1]):
            fig.add_trace(go.Scatter(x=self.mlogt,y=vecs[:,i],name=self.aia_channels[i]))
        fig.update_layout(yaxis_type='log',yaxis_range=yaxis_range,title='Temperature Response Matrix used for DEM calculation',yaxis_title='Temperature Response DN s<sup>-1</sup> px<sup>-1</sup> cm<sup>5</sup> K<sup>-1</sup>',xaxis_title='log<sub>10</sub> T (K)')
        return fig
                        
    def area_to_volume(self,thirdD,instr='AIA'):
        '''use area of emission to calculate a volume, given a 3rd dimension input'''
        area_cm2=self[instr+'_area']
        return volume

    def thermal_energy_content(self,logt,thirdD,volume_instr='AIA'):
        '''E = 3NkT, n= N/V (units? SI => T in Kelvin, V in m^3, n in kg m^-3) '''
        T=10**logt
        V=self.area_to_volume(thirdD,instr=volume_instr)
        #get n from the DEM: units are cm^-5 K^-1
        dem_scaled=self.dem*self.dtemps
        #select the correct T bin
        
        #multiply by... area of a single pixel? to get rid of the cm^-2?
        
        #now it's a proper density but it's not necessarily a particle density... hmmm...
        
        return 3*constants.k*n*V*T
        
