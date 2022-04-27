import numpy as np
import pandas as pd

import glob
from datetime import datetime as dt
from datetime import timedelta as td
import os
import logging
from scipy.stats import linregress
from dataclasses import dataclass

log = logging.getLogger(__name__)

@dataclass
class STIX_GOES_fit:
    def __init__(self,df):
        self.start_date=df.peak_utc.min()
        self.end_date=df.peak_utc.max()
        self.nflares=df._id.nunique()
        self.timestamp=dt.now()
        self.min_counts=df.peak_counts.min()
        self.bins=np.linspace(2,5,7).tolist() #set by hand for now...
        
    def do_fit(self,df):
        '''Fit all with linear regression'''
        res=linregress(np.log10(df.peak_counts_corrected.values),np.log10(df.GOES_flux.values))
        self.slope=res.slope
        self.intercept=res.intercept
        self.rvalue=res.rvalue
        self.pvalue=res.pvalue
        
    def bin_residuals(self,df):
        '''Find upper and lower boundaries based on number of input counts and distribution of residuals'''
        df['calculated_flux']=[self.slope*np.log10(pc)+self.intercept for pc in df.peak_counts_corrected]
        df['residuals']=np.log10(df.GOES_flux)-df.calculated_flux
        for i, bin in enumerate(self.bins):
            ebin=10**bin
            try:
                next_bin=10**self.bins[i+1]
                sigma_bin=df.query("peak_counts_corrected >= @ebin and peak_counts_corrected < @next_bin")['residuals'].dropna().std()
            except IndexError:
                sigma_bin=df.where(df.peak_counts_corrected >= ebin)['residuals'].dropna().std()

            setattr(self,f"bin_{bin}_sigma",sigma_bin)

