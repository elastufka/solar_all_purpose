#import dash_html_components as html

import pandas as pd
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord
#from datetime import datetime as dt
#from datetime import timedelta as td
#import sunpy.map
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame,pixel_to_skycoord, skycoord_to_pixel
from sunpy_map_utils import hpc_scale
#import sunpy.net as sn

class rotate_coord:
    '''Takes coordinate (or list of coordinates) and rotates it, converts to pixels or arcseconds. __init__ will generate SkyCoord from inputs and convert to the other unit (pixels or arcsecords). Rotation can then be done by do_rotation()'''
    def __init__(self, x_in,y_in, obs_in, wcs_in, obs_out=None, wcs_out=None, unit_in='arcsec',binning=1.,scaling=True):
        '''x_in, y_in is a single coord or list of coords. observer is SkyCoord.frame, wcs is WCS or str
        If scaling is desired, the coordinates will be scaled to lon/lat so that the angular size of the solar disk doesn't matter.'''
        #should check that frames and observers are correct types...
        wcs_in=self._check_wcs(wcs_in)
        obs_in=self._check_obs(obs_in)
        if wcs_out is not None:
            wcs_out=self._check_wcs(wcs_out)
        if obs_out is not None:
            obs_out=self._check_obs(obs_out)

        for k,v in locals().items():
            if k != 'self':
                setattr(self,k,v)
                
        #quickly control types on input...
        if type(x_in) == np.ndarray:
            self.x_in=list(x_in)
        if type(y_in) == np.ndarray:
             self.y_in=list(y_in)
                
        self.skycoord=self._to_skycoord()
        #print('original skycoord: ',self.skycoord)
        self.can_rotate=True #in some cases it might not be, see 2021-03-04 07:55:02.128 ValueError with negative distance (behind disk)
        
        #automatically do arcsec->pix for given coord or vice versa. hold off on rotation for now
        if self.unit_in == 'arcsec':
            self.x_arcsec=self.x_in
            self.y_arcsec=self.y_in
            pxx,pxy=self._world_to_pixel()
            if binning > 1:
                pxx=(np.array(pxx)/binning).tolist()
                pxy=(np.array(pxy)/binning).tolist()
            if scaling:
                self.x_deg,self.y_deg,self.rsun_apparent=hpc_scale(self.skycoord,obs_in)
            self.x_px=pxx
            self.y_px=pxy
        if self.unit_in == 'deg':
            self.x_deg=self.x_in #now do I have to rename all my variables...fix this to do things dynamically eventually.
            self.y_deg=self.y_in
            pxx,pxy=self._world_to_pixel()
            if binning > 1:
                pxx=(np.array(pxx)/binning).tolist()
                pxy=(np.array(pxy)/binning).tolist()
            self.x_px=pxx
            self.y_px=pxy
        if self.unit_in in ['px','pixel','pix']:
            self.x_px=x_in
            self.y_px=y_in
            arcx,arcy=self._pixel_to_world()
            self.x_arcsec=arcx
            self.y_arcsec=arcy
            if scaling:
                self.x_deg,self.y_deg=hpc_scale(self.skycoord,obs_in)
        
    def _check_wcs(self,wcs_input):
        '''type control '''
        if type(wcs_input) != WCS:
            try:
                wcs_output=WCS(wcs_input)
            except (IndexError,ValueError) as e: #it's a series
                wcs_output=WCS(wcs_input.iloc[0])
        else:
            wcs_output=wcs_input
        return wcs_output
    
    def _check_obs(self, obs_input):
        '''type control '''
        if type(obs_input) == pd.Series:#SkyCoord:
            obs_output=obs_input.iloc[0]
        else:
            obs_output=obs_input
        return obs_output
        
    def _to_skycoord(self):
        if self.unit_in == 'arcsec':
            coord=SkyCoord(self.x_in,self.y_in,unit=u.arcsec,frame='helioprojective',observer=self.obs_in,obstime=self.obs_in.obstime)
        elif self.unit_in == 'deg':
            coord=SkyCoord(self.x_in,self.y_in,unit=u.deg,frame='helioprojective',observer=self.obs_in,obstime=self.obs_in.obstime)
        else:
            coord=pixel_to_skycoord(np.multiply(self.x_in,self.binning),np.multiply(self.y_in,self.binning),self.wcs_in,obstime=self.obs_in.obstime)
        return coord
        
    def _world_to_pixel(self):
        '''not to be confused with skycoord and wcs functions '''
        if type(self.x_in) == list:
            try:
                pxarr=self.skycoord.to_pixel(wcs=self.wcs_in)
            except ValueError: #distance negative? not that I can tell
                pxarr=(None,None)
                self.can_rotate=False
            
            pxx=list(pxarr[0])
            pxy=list(pxarr[1])
            #pxx,pxy=[],[]
            #for x,y in pxarr:
            #    pxx.append(x)
            #    pxy.append(y)
        else:
            try:
                pxx,pxy=self.skycoord.to_pixel(wcs=self.wcs_in)
            except ValueError: #distance negative? not that I can tell
                pxx=None
                pxy=None
                self.can_rotate=False

            #pxx,pxy=self.skycoord.to_pixel(wcs=self.wcs_in)
        return pxx,pxy

    def _pixel_to_world(self):
        '''not to be confused with skycoord and wcs functions '''
        arcx=self.skycoord.Tx.value
        arcy=self.skycoord.Ty.value #arry if input is list
        if type(arcx)==np.array:
            arcx=arcx.tolist()
            arcy=arcy.tolist()
        return arcx,arcy
        
    def do_rotation(self,to_world=True,to_pixel=True):
        if not self.can_rotate:
            raise ValueError("in ~/anaconda/envs/py37/lib/python3.7/site-packages/sunpy/coordinates/frames.py in make_3d(self): Distance must be >= 0 ie behind limb and HGS can't be transformed to HPC for this coordinate or one of these coordinates.")
        elif not self.obs_out or not self.wcs_out:
            raise TypeError("Output observer and WCS must be provided in order to rotate coordinate!")
            
        testcoord=SkyCoord(0,0,unit=u.arcsec,frame='helioprojective',observer=self.obs_out,obstime=self.obs_out.obstime)
        #print("current skycoord: ", self.skycoord)
        rotcoord_arcsec=self.skycoord.transform_to(testcoord.frame)
        #print("rotated coord: ", rotcoord_arcsec)
        self.rotated_x_arcsec=rotcoord_arcsec.Tx.value.tolist()
        self.rotated_y_arcsec=rotcoord_arcsec.Ty.value.tolist()
        if self.scaling:
            lon,lat,rsun_apparent=hpc_scale(rotcoord_arcsec,observer=SkyCoord(0*u.arcsec, 0*u.arcsec,obstime=self.obs_out.obstime,observer=self.obs_out,frame='helioprojective')) #does this only work one 1 value? check
            self.rotated_lon_deg=lon
            self.rotated_lat_deg=lat
            self.rsun_apparent=rsun_apparent
        if to_pixel:
            rotcoord_pix=rotcoord_arcsec.to_pixel(self.wcs_out) #tuples
            if type(self.x_in) == list:
                #pxx,pxy=[],[]
                #for x,y in rotcoord_pix:
                #    pxx.append(x/self.binning)
                #    pxy.append(y/self.binning)
                pxx=rotcoord_pix[0]/self.binning
                pxy=rotcoord_pix[1]/self.binning
            else:
                pxx=float(rotcoord_pix[0])/self.binning
                pxy=float(rotcoord_pix[1])/self.binning
            self.rotated_x_px=pxx
            self.rotated_y_px=pxy
            
    def to_dataframe(self):
        '''output to dataframe '''
        instr=self.wcs_in.to_header_string()
        outstr=self.wcs_out.to_header_string()
        self.__dict__['wcs_in']=instr
        self.__dict__['wcs_out']=outstr
        if type(self.x_in) == float or type(self.x_in) == int:
            return pd.DataFrame(self.__dict__,index=pd.Index([0]))
        else:
            return pd.DataFrame(self.__dict__,index=pd.Index(list(range(len(self.x_in)))))
        

        
