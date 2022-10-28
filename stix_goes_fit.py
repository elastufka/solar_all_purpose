import numpy as np
import pandas as pd

import glob
from datetime import datetime as dt
from datetime import timedelta as td
import os
#import logging
from IPython.display import Math
from scipy.stats import linregress
from dataclasses import dataclass

#log = logging.getLogger(__name__)

@dataclass
class STIX_GOES_fit:
    def __init__(self,df, xkey = 'peak_counts_corrected', ykey = 'GOES_flux'):
        self.start_date=df.peak_UTC.min()
        self.end_date=df.peak_UTC.max()
        #self.nflares=df._id.nunique()
        self.timestamp=dt.now()
        self.min_counts=df.peak_counts.min()
        self.bins=np.linspace(2,5,7).tolist() #set by hand for now...
        self.xkey = xkey
        self.ykey = ykey
        
    def __repr__(self):
        return f"{self.ykey} = {self.slope:.3f} ∙  log10({self.xkey}) + {self.intercept:.3f}"
        
    def repr_html(self):
        return f"GOES_flux = {self.slope:.3f} ∙  log<sub>10</sub>(STIX_counts) + {self.intercept:.3f}"
        
    def do_fit(self,df):
        '''Fit all with linear regression'''
        df = df[df[self.ykey] != 0.0] #get rid of places where no GOES flux
        df = df.dropna(subset=[self.xkey, self.ykey])
        df = df[df[self.xkey] > 0.0]
        self.nflares = len(df)
        res=linregress(np.log10(df[self.xkey].values),np.log10(df[self.ykey].values.astype(float)))
        self.slope=res.slope
        self.intercept=res.intercept
        self.rvalue=res.rvalue
        self.pvalue=res.pvalue
        
    def bin_residuals(self,df):
        '''Find upper and lower boundaries based on number of input counts and distribution of residuals'''
        df['calculated_flux']=[self.slope*np.log10(pc)+self.intercept for pc in df[self.xkey]]
        df['residuals']=np.log10(df.GOES_flux)-df.calculated_flux
        for i, bin in enumerate(self.bins):
            ebin=10**bin
            try:
                next_bin=10**self.bins[i+1]
                sigma_bin=df.query("peak_counts_corrected >= @ebin and peak_counts_corrected < @next_bin")['residuals'].dropna().std()
            except IndexError:
                sigma_bin=df.where(df[self.xkey] >= ebin)['residuals'].dropna().std()

            setattr(self,f"bin_{bin}_sigma",sigma_bin)

