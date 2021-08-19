#######################################
#dem_solve.py
# Erica Lastufka 15/07/2021

### IMPORTANT: if not in bashrc, run these commands in the terminal prior to use
#export HEADAS=/Users/wheatley/Documents/HEASOFT/heasoft-6.29/x86_64-apple-darwin19.6.0/
#. $HEADAS/headas-init.sh
#######################################

import numpy as np
import pandas as pd
import xspec

import os
import glob
from datetime import datetime as dt
#import pickle
import plotly.graph_objects as go
#from nustar_utils import plot_nustar_specfit

class fit_with_xspec:
    def __init__(self,datadir='/Users/wheatley/Documents/Solar/NuStar/specfiles/2039_2042',both=True,model='const*apec',fitrange=[2.5,5.0],logfilename='xspec.log',fitfile='xspec_fit.txt',freeze_T=False,freeze_EM=False,abundf='/Users/wheatley/Documents/Solar/NuStar/specfiles/2039_2042/feld92a_coronal0.txt',niter=1000,prange=[1.0,6.0],plot=True):
        '''fit input spectrum with pyXspec'''
        
        global kev2mk
        kev2mk=0.0861733
        global emfact
        emfact=3.5557e-42
        
        logfile = xspec.Xset.openLog(logfilename)
        xspec.Plot.device = '/null'
        xspec.Plot.xAxis = "keV"
        xspec.Xset.abund="file %s" % abundf
        
        self.niter=niter
        self.datadir=datadir
        self.fitfile=fitfile
        
        self.load_data(both=both)
        self.set_fitrange(fitrange)
        m1=self.set_model(model)
        
        if freeze_T != False:
            self.freeze_model_component(m1,'kT',freeze_T,kev2mk)
        if freeze_EM != False:
            self.freeze_model_component(m1,'norm',freeze_EM,emfact)
            
        self.do_fit()
        self.get_T_EM(m1)
        self.expand_plot_range(prange)
        self.write_fit_results()
        
        if plot:
            df_ld,_,_=self.read_fit_results()
            self.plot_fit(df_ld,fitrange,prange)
    
    def load_data(self,both):
        if both== 'A':
            pha=glob.glob(self.datadir+'/'+'*A*chu*N_sr.pha')
            xspec.AllData(pha[0])
        elif both == 'B':
            pha=glob.glob(self.datadir+'/'+'*B*chu*N_sr.pha')
            xspec.AllData(pha[0])
        
        else: #default to using a and b
            pha=glob.glob(self.datadir+'/'+'*chu*N_sr.pha')
            xspec.AllData(pha[0] + ' '+ pha[1])
            xspec.Plot.setGroup("1 2")
            xspec.Plot.add=True
            
    def set_fitrange(self,fitrange):
        xspec.AllData.ignore("0.-%s %s-**" % tuple(fitrange))

    def set_model(self,model):
        m1=xspec.Model(model)
        #1.0 -0.1
        #untie 6  # these commands seem important when using 2 spectra
        #thaw 6
        #m1.untie()
        return m1
        
    def freeze_model_component(self,m1,cname,freezevalue,fac):
        '''assume model is apec for now'''
        par=getattr(m1.apec,cname)
        vals=par.values
        vals[0]=freezevalue*fac
        svals=[str(v) for v in vals]
        newvalues=",".join(svals)
        print(newvalues)
        par.values=newvalues
        par.frozen=True
    
    def do_fit(self):
        xspec.Fit.nIterations=self.niter
        # Stop fit at nIterations and do not query.
        xspec.Fit.query = "no"
        xspec.Fit.statMethod = "cstat"
        xspec.Fit.renorm()
        xspec.Fit.perform()
        xspec.Fit.error("1-4")
        
        self.cstatistic=xspec.Fit.statistic
        self.chisq=xspec.Fit.testStatistic
        
    def get_T_EM(self,m1):
        c2=m1.apec
        self.T=c2.kT.values[0]/kev2mk
        self.T_err=c2.kT.error[0]/kev2mk
        self.EM=c2.norm.values[0]/emfact
        self.EM_err=c2.norm.error[0]/emfact
        
    def expand_plot_range(self,prange):
        xspec.AllData.notice("%s-%s" % tuple(prange))
    
    def write_fit_results(self):
        #how to force overwrite?
        #setplot group 1-2
        xspec.Plot.iplot('ldata', 'ufspec', 'rat') #do I need to do this up front?
        #pdata = 'test_model_out.txt'
        #delete fit file if already exists
        if os.path.isfile(self.datadir+'/'+self.fitfile):
            os.remove(self.datadir+'/'+self.fitfile)
        xspec.Plot.addCommand(f'wd {self.fitfile}')
        xspec.Plot.iplot('ldata', 'ufspec', 'delchi')
        
    def read_fit_results(self):
        ''' number of lines of NO NO NO splist graphs... 1 for each spectra'''
        names = ['energy','denergy','data','data_err','model']
        #should delete fit file if already exists
        df = pd.read_table(self.datadir+'/'+self.fitfile,skiprows=3,names=names, delimiter=' ')
        bidx=df.where(df.energy == 'NO').dropna(how='all').index.values #these are the breaks between plots..
        #different plots are 'ld','uf','dc' no idea what these mean... uf and ld seem to be the same
        
        if len(bidx)==2:
            df_ld=df.iloc[:bidx[0]]
            df_uf=df.iloc[bidx[0]+1:bidx[1]]
            df_dc=df.iloc[bidx[1]+1:]
            
        else: #there are 6 s, so 5 separations
            df_ld1=df.iloc[:bidx[0]]
            df_ld2=df.iloc[bidx[0]+1:bidx[1]]
            df_uf1=df.iloc[bidx[1]+1:bidx[2]]
            df_uf2=df.iloc[bidx[2]+1:bidx[3]]
            df_dc1=df.iloc[bidx[3]+1:bidx[4]]
            df_dc2=df.iloc[bidx[4]+1:]
            df_ld=pd.concat([df_ld1,df_ld2]) #need to sum the data columns? right now this is 2 plots not 1, why?
            df_uf=pd.concat([df_uf1,df_uf2])
            df_dc=pd.concat([df_dc1,df_dc2])
        #convert to floats
        for df in [df_ld,df_uf,df_dc]:
            for k in df.keys():
                df[k]=df[k].replace('NO','0')
                df[k]=pd.to_numeric(df[k])
        return df_ld,df_uf,df_dc
                
    def plot_fit(self,df_ld,fitrange,prange):
        tempfit=np.round(self.T,2)
        terr=np.round(self.T_err,2)
        emfit=np.format_float_scientific(self.EM,precision=2)
        emerr=np.round(self.EM_err,2)
        title=f"{tempfit} ± {terr} MK, {emfit} ± {emerr} cm<sup>-3</sup>"
        
        dfp=df_ld.replace(0.0,np.nan) #don't plot zeros
        fig = make_subplots(rows=2, cols=1, start_cell="top-left",shared_xaxes=True,row_heights=[.6,.3],vertical_spacing=.05)
        fig.add_trace(go.Scatter(x=dfp.energy,y=dfp.data,mode='markers',name='data',error_y=dict(type='data',array=dfp.data_err)),row=1,col=1)
        fig.add_trace(go.Scatter(x=dfp.energy,y=dfp.model,mode='lines',name='fit'),row=1,col=1)
        fig.add_trace(go.Scatter(x=dfp.energy,y=dfp.data-dfp.model,mode='markers',marker_color='brown',name='residuals'),row=2,col=1)
        fig.add_vrect(x0=fitrange[0],x1=fitrange[1],annotation_text='fit range',fillcolor='lightgreen',opacity=.25,line_width=0,row=1,col=1)
        fig.add_vrect(x0=fitrange[0],x1=fitrange[1],fillcolor='lightgreen',opacity=.25,line_width=0,row=2,col=1)
        fig.update_yaxes(title='Counts s<sup>-1</sup> keV<sup>-1</sup>',range=[-1.5,1],row=1,col=1,type='log') #type='log'
        fig.update_yaxes(title='Residuals',range=[-.5,.5],row=2,col=1)
        fig.update_xaxes(title='Energy (keV)',row=2,col=1)
        fig.update_layout(width=500,height=600,title=title)
            
        fig.update_xaxes(range=prange)
        fig.show()

    def __del__(self):
        '''cleanup ... make sure xspec especially is gone '''
        print('destructor called')
        
