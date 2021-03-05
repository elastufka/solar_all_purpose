 #######################################
# query_vso.py
# Erica Lastufka 13/11/2017  

#Description: Get maps from VSO
#######################################

#######################################
# Usage:

######################################

from datetime import datetime as dt
import os
from sunpy.net import vso
import astropy.units as u

def query_vso(time_int, instrument, wave, sample=False, source=False, path=False):
    '''query VSO database for data and download it. '''
    if type(time_int[0]) == str:
        time_int[0]=dt.strptime(time_int[0],'%Y-%m-%dT%H:%M:%S')
        time_int[1]=dt.strptime(time_int[1],'%Y-%m-%dT%H:%M:%S')
    wave=vso.attrs.Wavelength((wave-.1)* u.angstrom, (wave+.1)* u.angstrom)
    vc = vso.VSOClient()
        
    instr= vso.attrs.Instrument(instrument)
    time = vso.attrs.Time(time_int[0],time_int[1])
    qs=[wave, instr, time]
    if source:
        source=vso.attrs.Source(source)
        qs.append(source)
    if sample: #Quantity
        sample = vso.attrs.Sample(sample)
        qs.append(sample)
            
    res = vc.search(*qs)
    print(qs, path, res)
    if not path: files = vc.fetch(res,path='./{file}').wait()
    else: files = vc.fetch(res).wait()

