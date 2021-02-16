#do_quicklook.py

import webbrowser
import pandas as pd
import os
import urllib
import glob
#for each flare, open the quicklook image in firefox. Limit the number of tabs in each browser to 20.

def do_quicklook(download=False):
    dataurl = 'http://soleil.i4ds.ch/hessidata/metadata/qlook_image_plot'
    #subfolders by year, month,day (except 2001)

    with open('/Users/wheatley/Documents/Solar/occulted_flares/list_observed_1hour.csv') as lf:
        for line in lf:
            c1=line.rfind(',')
            c2=line.find(',')
            newline=line[c2+1:]
            id=line[c2+1:newline.find(',')+c2+1]
            year=line[c1+3:c1+5]
            month=line[c1+6:c1+8]
            day=line[c1+9:c1+11]
            newurl=dataurl+'/20'+year+'/'+month+'/'+day.strip()
            filename='/hsi_qlimg_'+id+'_012025.png' #get the 12-25 keV image
            print c1,c2,newline.find(',')-1,newurl+filename
            image=newurl+filename
            if download:
                #f = open('~/Documents/Solar/MiSolFA/statistics/data/highen/'+filename[1:],'wb')
                #f.write(urllib.urlopen(image).read())
                urllib.urlretrieve(image,'/Users/wheatley/Documents/Solar/occulted_flares/data/round2/'+filename[1:])
                #f.close()
            else:
                webbrowser.open_new_tab(image)
                #print 'foo'

def open_in_RHESSI_browser(filename='/Users/wheatley/Documents/Solar/occulted_flares/list_final_1hour.csv'):
    browserurl = 'http://sprg.ssl.berkeley.edu/~tohban/browser/?show=grth1+qlpcr+qlpg9+qli02+qli03+qli04+synff&date=' #20120917&time=061500'
    #subfolders by year, month,day (except 2001)

    with open(filename) as lf:
        for line in lf:
            if line.startswith('ID'):
                continue
            else:                
                c1=line.rfind(',')
                sec=line[c1-2:c1]
                min=line[c1-5:c1-3]
                hour=line[c1-8:c1-6]
                year=line[c1+1:c1+5]
                month=line[c1+6:c1+8]
                day=line[c1+9:c1+11]
                address=browserurl+ year+month+day + '&time=' + hour+min+sec
                print address         
                webbrowser.open_new_tab(address)
                
def read_folder_contents():
    folder = '~/Documents/Solar/MiSolFA/statistics/data'
    os.chdir(folder)
    files=glob.glob('*.png')
    for f in files:
        #parse the names to get event id 
        id = f[f.rfind('_')+1:f.find('.')]
    #save stuff

#do_quicklook()
#do_quicklook(download=True)
#read_folder_contents()

#or download them to a folder, delete the ones I don't want, use glob to get the ones that are left and parse those into a new .csv?
