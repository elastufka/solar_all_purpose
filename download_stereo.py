 #######################################
# query_vso.py
# Erica Lastufka 13/11/2017  

#Description: Get maps from VSO
#######################################

#######################################
# Usage:

######################################

import numpy as np
import scipy.constants as sc
from datetime import datetime as dt
from datetime import timedelta as td
import os
from sunpy.net import Fido, attrs as a
import sunpy.map
import astropy.units as u

def query_vso(time_int, instrument, wave, source=False, path=False,save=True):
        '''method to query VSO database for data and download it. '''
        #vc = vso.VSOClient()
        maps=0

        #provider = a.Provider(provider)
        
        #if type(time_int[0]) == dt:
        #    time_int[0]=dt.strftime(time_int[0],'%Y-%m-%dT%H:%M:%S')
        #    time_int[1]=dt.strftime(time_int[1],'%Y-%m-%dT%H:%M:%S')
            
        if source:
             source=a.vso.Source(source)
        while instrument:
            try:
                instr= a.Instrument(instrument)
                break
            except ValueError:
                instrument=raw_input('Not a valid instrument! Try again:' )
            
        #sample = vso.attrs.Sample(24 * u.hour)
        wl=a.Wavelength((wave-.5)* u.angstrom, (wave+.5)* u.angstrom)
        
        time = a.Time(time_int[0],time_int[1])
        
        if source:
            res=Fido.search(source, wl, time, instr)
        else:
            res=Fido.search(time, wl,instr)

        print res
        #if len(res) != 1:
        if not path: files = Fido.fetch(res,path='./{file}').wait()
        else: files = Fido.fetch(res,path=path+'{file}').wait()

        f=sunpy.map.Map(files[0])
        maps.append({f.instrument: f.submap(SkyCoord((-1100, 1100) * u.arcsec, (-1100, 1100) * u.arcsec,frame=f.coordinate_frame))}) #this field too small for coronographs

        if save: #pickle it? 
            os.chdir(path)
            newfname=files[0][files[0].rfind('/')+1:files[0].rfind('.')]+'.p'
            pickle.dump(maps,open(newfname,'wb'))
        #else:
        #    print 'no results found! is the server up?'
        return maps
