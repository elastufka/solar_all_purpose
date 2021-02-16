 #######################################
# lightcurves.py
# Erica Lastufka 5/4/2017  

#Description: Plot Messenger and RHESSI lightcurves together
#######################################

#######################################
# Usage:

######################################

import numpy as np
import scipy.constants as sc
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
import os
import data_management as da
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy

def import_data(filename):
    '''Import lightcurve data from IDL into a pandas dataframe'''
    from scipy.io import readsav
    #format everything so it's right
    d=readsav(filename, python_dict=True)
    Rdata=d['data']['rdata'] #access via Rdata[0]['UT'], etc
    Mdata=d['data']['mdata']
    Gdata=d['data']['gdata']
    return Rdata[0],Mdata[0],Gdata[0]

def counts_ps(Mdata,n):
    '''Adjust messenger data to counts/s'''
    #get the time bin for each data point
    tim=Mdata['taxis'][0][n]
    mlen=Mdata['len'][0][n]
    nflares=np.shape(Mdata['taxis'][0])[0]/2
    Mtim,cps1,cps2=[],[],[]
    for t in tim: Mtim.append(dt.strptime(t,'%d-%b-%Y %H:%M:%S.%f')) #fix messenger times to datetimes
    M1=Mdata['phigh'][0][n][0:mlen-1]
    M2=Mdata['phigh'][0][n+nflares-1][0:mlen-1]

    for i in range(mlen-1):
        tbin=Mtim[i+1]-Mtim[i] #timedelta object in seconds
        cps1.append(M1[i]/tbin.total_seconds())
        cps2.append(M2[i]/tbin.total_seconds())
    return cps1,cps2

def loop_GM(g,m):
    for n in range(0,26):
        try:
            foo=plot_GM(m,g,n)
        except ValueError: #no data?
            print n,m['taxis'][0][n][0]
            continue

def plot_GM(Mdata,Gdata,n): #will probably have to deal with times to make them all the same...
    import matplotlib.dates as mdates
    tim=Mdata['taxis'][0][n]
    mlen=Mdata['len'][0][n]
    nflares=np.shape(Mdata['taxis'][0])[0]/2 #assume 2 energy ranges for now
    Mtim=[]
    for t in tim: Mtim.append(dt.strptime(t,'%d-%b-%Y %H:%M:%S.%f')) #fix messenger times to datetimes
    cps1,cps2=counts_ps(Mdata,n)
    print type(Mtim),np.shape(Mtim[:-1]),np.shape(cps1)
    #print np.shape(cps1),np.shape(Mtim[0:mlen-1])
    
    gtim=Gdata['taxis'][0][n]
    glen=Gdata['len'][0][n]
    Gtim=[]
    for t in gtim: Gtim.append(dt.strptime(t,'%d-%b-%Y %H:%M:%S.%f')) #fix GOES times to datetimes
    glong=Gdata['ydata'][0][n,1,0:glen-1] #what's up with these data?[1,0:glen-1][n,1,0:glen-1]
    gshort=Gdata['ydata'][0][n,0,0:glen-1]    

    #plt.plot(Mtim[0:mlen-1],Mdata['phigh'][0][n][0:mlen-1],'b') #first energy channel
    #plt.plot(Mtim[0:mlen-1],Mdata['phigh'][0][n+nflares-1][0:mlen-1],'g') #second energy channel I think...check that this is plotting the right thing
    fig,ax1=plt.subplots()
    ax2=ax1.twinx()
    #l1,=ax1.step(Mtim[0:mlen-1],Mdata['phigh'][0][n][0:mlen-1],'b',label='1.5-12.4 keV') #first energy channel
    #l2,=ax1.step(Mtim[0:mlen-1],Mdata['phigh'][0][n+nflares-1][0:mlen-1],'g',label= '3-24.8 keV') #second energy channel I think...
    l1,=ax1.step(Mtim[0:mlen-1],cps1,'b',label='1.5-12.4 keV')
    l2,=ax1.step(Mtim[0:mlen-1],cps2,'g',label= '3-24.8 keV')
    #l1,=ax1.step(Mtim[:-1],cps1,'b',label='1.5-12.4 keV')
    #l2,=ax1.step(Mtim[:-1],cps2,'g',label= '3-24.8 keV')
    
    #plt.axis #add y-axis for GOES flux
    l3,=ax2.plot(Gtim[0:glen-1],gshort,'k',label='GOES 1-8 $\AA$') #goes short - plot with
    l4,=ax2.plot(Gtim[0:glen-1],glong,'m',label='GOES .5-4 $\AA$') #goes long
    myFmt = mdates.DateFormatter('%H:%M')
    ax1.xaxis.set_major_formatter(myFmt)
    plt.gcf().autofmt_xdate()
    #ax1.set_xlabel(dt.strftime(Mtim[0].date(),'%Y-%b-%d'))
    ax1.set_ylabel('Messenger counts $cm^{-2} keV^{-1} s^{-1}$')
    ax1.set_ylim([10**0,10**4])
    ax1.set_yscale('log')
    ax2.set_ylabel('GOES Flux W$m^{-2}$')
    ax2.set_yscale('log')
    
    plt.title(dt.strftime(Mtim[0].date(),'%Y-%b-%d'))
    ax1.set_xlim([Gtim[0],dt.strptime('2012-07-19 05:55:01','%Y-%m-%d %H:%M:%S')])
    #print np.max(glong),np.max(gshort)
    #plt.legend((l1,l2,l3,l4),(l1.get_label(),l2.get_label(),l3.get_label(),l4.get_label()),loc='upper left',prop={'size':12})
    fig.show()
    fname='data/lightcurves/'+dt.strftime(Mtim[0].date(),'%Y-%b-%d')+'MG.png'
    fig.savefig(fname)
    return Mtim[0:mlen-1],cps1

def plotR(Rdata,n):
    import matplotlib.dates as mdates
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    n=n*4
    tim=Rdata['UT'][0][1,:,1]#[n][:,0]#since there are 4 channels per flare
    #rlen=Rdata['len'][0][n]

    Rtim=[]
    nflares=np.shape(Rdata['rate'][0])[0]/4 #assume 4 energy ranges for now
    for t in tim: Rtim.append(dt.strptime(t,'%d-%b-%Y %H:%M:%S.%f'))

    #get the energy bins - or do I need to do this since they should be the same? check first
        
    if np.mean(Rdata['rate'][0][n]) != 0.0:
        ax1.plot(Rtim,Rdata['rate'][0][n],'m',label='4-9 keV') #first energy channel
    if np.mean(Rdata['rate'][0][n+1]) != 0.0:
        print np.shape(Rdata['rate'][0][n+1])
        ax1.plot(Rtim,Rdata['rate'][0][n+1],'g',label='12-18 keV') #second energy channel I think...
    if np.mean(Rdata['rate'][0][n+2]) != 0.0:
        ax1.plot(Rtim,Rdata['rate'][0][n+2],'c',label='18-30 keV') #etc
    if np.mean(Rdata['rate'][0][n+3]) != 0.0:
        ax1.plot(Rtim,Rdata['rate'][0][n+3],'k',label='30-80 keV') #etc
    #ax1.set_xlabel(dt.strftime(Rtim[0].date(),'%Y-%b-%d'))
    ax1.set_yscale('log')
    ax1.set_ylim([0,10**5])
    ax1.legend(loc='upper right')
    myFmt = mdates.DateFormatter('%H:%M')
    ax1.xaxis.set_major_formatter(myFmt)
    plt.gcf().autofmt_xdate()
    plt.title(dt.strftime(Rtim[0].date(),'%Y-%b-%d'))
    plt.show()
    fname='data/lightcurves/'+dt.strftime(Rtim[0].date(),'%Y-%b-%d')+'R.png'
    fig.savefig(fname)
    
def loop_R(r):
    for n in range(0,26):
        foo=plotR(r,n)

        #data,ebins=import_spectrogram('data/testspect.sav')
def import_spectrogram(filename):
    from scipy.io import readsav
    #format everything so it's right
    d=readsav(filename, python_dict=True)
    #foo=np.zeros([23,450],dtype=str)
    data={'UT':0.,'rate':0.,'erate':0.,'ltime':0.,'len':0.}
    #for i,t in enumerate(d['data']['UT']):
    #    j=0
    #    if t[0] != '':
    #        for tim in t:
    #            foo[i,j]=tim#dt.strptime(tim,'%d-%b-%Y %H:%M:%S.000')
    #            j=j+1
    data['UT'] = d['data']['UT']
    data['rate']=d['data']['rate']
    data['erate']=d['data']['erate']
    #ltime=np.zeros(np.shape(d['data']['ltime'])) #whatever side it's supposed to be
    #for i,t in enumerate(data['UT']):
    #    timedelt=data['UT'][i+1]-t
    #data['ltime']=ltime #time offset from start
    data['len']=d['data']['len']
    ebins=d['ebins']
    return data,ebins
  
def loop_spectrogram(data,ebins):
    '''Plot spectrograms for all flares in flare list'''
    nflares=np.shape(data['UT'])[0]
    for i in range(0,nflares):
        try:
            a=sunpy_spectrogram(data,ebins,i)
        except ValueError:
            print dt.strptime(data['UT'][i][0],'%d-%b-%Y %H:%M:%S.000')
            continue

def sunpy_spectrogram(data,ebins,i):
    #from sunpy.spectra import spectrogram as s
    import mySpectrogram as s
    eaxis=ebins[0:-1]
    last=int(data['len'][0]-1)
    time_axis=np.arange(0,last+1)*4.#data[0]['ltime'] #ndarray of time offsets,1D #need the 4 for RHESSI time interval
    start= dt.strptime(data['UT'][i][0],'%d-%b-%Y %H:%M:%S.000') #datetime
    try:
        end= dt.strptime(data['UT'][i][last],'%d-%b-%Y %H:%M:%S.000') #datetime #might want to modify this to be where the data =0
    except ValueError:
        import datetime
        end = start + datetime.timedelta(seconds=time_axis[-1])
        #print dt.strptime(data['UT'][i][last-1],'%d-%b-%Y %H:%M:%S.000')
    drate=np.transpose(np.log10(data['rate'][i][0:last+1])) #get rid of zeros before taking log10?
    #drate=np.nan_to_num(drate)
    drate[drate == -np.inf] = 0.0
    for n,col in enumerate(drate.T):
        if all([c ==0.0 for c in col]):
            drate[:,n] = np.nan
    a=s.Spectrogram(data=drate,time_axis=time_axis,freq_axis=eaxis,start=start,end=end,instruments=['RHESSI'],t_label='',f_label='Energy (keV)',c_label='log(counts/cm^2 s)')
    fig=a.plot()

    outfilename='data/spectrograms/'+s.get_day(a.start).strftime("%d%b%Y")+'sgram.png'
    fig.figure.savefig(outfilename)
    plt.clf()
    return a
    
