#import dash_html_components as html

import pandas as pd
import numpy as np
import glob
import os

from datetime import timedelta as td
import sunpy.net as sn
from visible_from_earth import get_observer
from rotate_coord import rotate_coord

class HEKEventHandler():
    def __init__(self, date_obs,obs_out='SO', event_type='FL',obs_instrument='AIA',small_df=True,single_result=False, search_int=td(minutes=5)):
        for k,v in locals().items():
            if k != 'self':
                setattr(self,k,v)
        
        if type(self.date_obs) == str:
            self.date_obs=pd.to_datetime(self.date_obs)
            
        self.time_int=[self.date_obs-self.search_int,self.date_obs+self.search_int]
        df=self.query_hek()
        if not df.empty:
            print("df is not empty")
            self.get_observers_and_wcs()
            df=self.rotate_hek_coords(df)
        self.df=df
        
    def get_observers_and_wcs(self):
        ''' what it sounds like - for use in rotations'''
        self.observer_in,self.wcs_in=get_observer(self.date_obs,obs=self.obs_instrument)
        self.observer_out,self.wcs_out=get_observer(self.date_obs,obs=self.obs_out)
    
    def query_hek(self):
        time = sn.attrs.Time(self.time_int[0],self.time_int[1])
        eventtype=sn.attrs.hek.EventType(self.event_type)
        #obsinstrument=sn.attrs.hek.OBS.Instrument(obs_instrument)
        res=sn.Fido.search(time,eventtype,sn.attrs.hek.OBS.Instrument==self.obs_instrument)
        tbl=res['hek']
        names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
        df=tbl[names].to_pandas()
        if df.empty:
            return df
        if self.small_df:
            df=df[['hpc_x','hpc_y','hpc_bbox','frm_identifier','frm_name','fl_goescls','fl_peaktempunit','fl_peakemunit','fl_peakflux','event_peaktime','fl_peakfluxunit','fl_peakem','fl_peaktemp','obs_dataprepurl','gs_imageurl','gs_thumburl']]
            df.drop_duplicates(inplace=True)
        if self.single_result: #select one
            aa=df.where(df.frm_identifier == 'Feature Finding Team').dropna()
            #print(aa.index.values)
            if len(aa.index.values) == 1: #yay
                return aa
            elif len(aa.index.values) > 1:
                return pd.DataFrame(aa.iloc[0]).T
            elif aa.empty: #whoops, just take the first one then
                return pd.DataFrame(df.iloc[0]).T

        return df
        
    def rotate_hek_coords(self,df,binning=1):
        '''World to pixel and rotate for HEK event coords '''
        dfs=[]
        for i,row in df.iterrows():
            rc=rotate_coord(row.hpc_x,row.hpc_y,self.observer_in,self.wcs_in,obs_out=self.observer_out,wcs_out=self.wcs_out,binning=binning) #.iloc[0]
            try:
                rc.do_rotation()
                #print(rc.rotated_lon_deg,rc.rotated_lat_deg, rc.rotated_x_arcsec,rc.rotated_y_arcsec)
            except ValueError:
                print("Coordinate (%s,%s) could not be rotated!" % (row.hpc_x,row.hpc_y))
                rc.rotated_x_arcsec=None
                rc.rotated_y_arcsec=None
                rc.rotated_x_px=None
                rc.rotated_y_px=None
            rdf=rc.to_dataframe()[['x_arcsec', 'y_arcsec',
            'x_deg', 'y_deg', 'rsun_apparent', 'x_px', 'y_px', 'rotated_x_arcsec',
            'rotated_y_arcsec', 'rotated_lon_deg', 'rotated_lat_deg',
            'rotated_x_px', 'rotated_y_px']]
            dfs.append(rdf)

        hdf=pd.concat(dfs)
        mdf=pd.merge(df,hdf, left_on=['hpc_x','hpc_y'],right_on=['x_arcsec','y_arcsec'])
        #print(mdf.keys())
        return mdf

