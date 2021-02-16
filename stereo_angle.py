 #######################################
# stereo_angle.py
# Erica Lastufka 31/10/2017  

#Description: Calculate angle between STEREO and Earth from CalTech ASCII files
#######################################

#######################################
# Usage:

######################################

import numpy as np
import scipy.constants as sc
from datetime import datetime as dt
from datetime import timedelta as td
import os

def get_stereo_angle(stereo_date, stereo='A'):
    '''Input is date and time in datetime format'''
    os.chdir('/Users/wheatley/Documents/Solar/occulted_flares/data/stereo-aia/')
    year=stereo_date.year
    if stereo == 'A':
        stereo='ahead'
    else:
        stereo='behind'
    pfile='position_'+stereo+'_'+str(year)+'_HEE.txt'
    #day as day in year
    day=dt.strftime(stereo_date,'%-j')
    #time in seconds of day
    hour=stereo_date.hour
    minute=stereo_date.minute
    seconds=hour*3600 + minute*60.
    with open(pfile) as pf:
        for line in pf.readlines():
            #what's the closest time ...
            l=line.split()
            pyear=l[0]
            pday=l[1]
            pseconds=l[2]
            if pday ==day and np.abs(int(pseconds)-int(seconds)) < 2000.:
                #print line
                heex=float(l[4])
                heey0=float(l[5])
                #flag?
                flag=l[3]
                if flag == '0':
                    print 'warning: data is not definitive' #check the next one            
                #convert to angle
                if np.arctan(heey0/heex) < 0:
                    heey=-1*heey0
                else:
                    heey=heey0
                angle=(90.-np.arctan(heey/heex)*180./np.pi )+90.
                #print heex,heey,angle
                break
    os.chdir('/Users/wheatley/Documents/Solar/occulted_flares')
    return heex,heey0,angle
    

