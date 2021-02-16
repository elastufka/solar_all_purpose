#transform_loops.py
#use run_trace_loops to get HEE cartesian (in terms of Rsun) coordinates of loop on AIA map 
#transform loop from AIA to STEREO projection

import astropy.units as u
from scipy.io import readsav #for readsav
from astropy.coordinates import SkyCoord
import sunpy.coordinates.wcs_utils
from datetime import datetime as dt
from sunpy.coordinates import frames
import sunpy.map

fl=f.list[0]

#first read the data. Needed: coords, map.time, map.l0,map.b0,map.Rsun
#assume that coords have already been multpilied by map.Rsun in IdL
filename='../data/testloop2.sav'
data=readsav(filename,python_dict=True)
coords=data['coords'] # as % of Rsun
#rsun_km=data['rsun']*719.685*u.km #it really doesn't like arcseconds... need a more exact conversion factor
#z=(coords[0])*rsun_km
#x=(coords[1])*rsun_km
#y=(coords[2])*rsun_km

#format the datetime
if data['time'][1] == '-':
    data['time']='0'+data['time'] #add a zero in front
date=dt.strptime(data['time'][:-4],'%d-%b-%Y %H:%M:%S')

#get AIA map
map_aia=fl.get_AIA(filename='aia_lev1_171a_2010_11_03t12_11_48_34z_image_lev1.fits')

#get stereo map
map_stereo=fl.get_STEREO(filename='20101103_121530_n4eub.fts')

#now make SkyCoord objects out of the coordinates. (what to do with z?)
Tx=coords[1]*map_aia['AIA 3'].rsun_obs
Ty=coords[2]*map_aia['AIA 3'].rsun_obs
hpc_aia_lc=SkyCoord(Tx,Ty,frame=map_aia['AIA 3'].coordinate_frame)
print hpc_aia_lc.Tx[0],hpc_aia_lc.Tx[100]
print hpc_aia_lc.Ty[0],hpc_aia_lc.Ty[100]

#transform to hgs
hgs_lc=hpc_aia_lc.transform_to(frames.HeliographicStonyhurst)

#reset things for STEREO
hgs_lc.D0 = map_stereo[0]['SECCHI'].dsun
hgs_lc.L0 = map_stereo[0]['SECCHI'].heliographic_longitude
hgs_lc.B0 = map_stereo[0]['SECCHI'].heliographic_latitude

#transform to helioprojective (STEREO)
hpc_B = hgs_lc.transform_to(map_stereo[0]['SECCHI'].coordinate_frame)
print hpc_B.Tx[0],hpc_B.Tx[100]
print hpc_B.Ty[0],hpc_B.Ty[100]

#plot things - first plot on AIA to make sure it's right
fig1=plt.figure()
ax1=fig1.add_subplot(111,projection=map_aia['AIA 3'].wcs)
#ax1.plot_coord(HEE,color='r')
map_aia['AIA 3'].plot(axes=ax1)
#ax1.plot_coord(coords[1],coords[2],color='r')
ax1.plot_coord(hpc_aia_lc,color='b')
fig1.show()


fig2=plt.figure()
ax2=fig2.add_subplot(111,projection=map_stereo[0]['SECCHI'].wcs)
#ax2.plot_coord(hpc_B,color='r')
map_stereo[0]['SECCHI'].plot(axes=ax2)
ax2.plot_coord(hpc_B,color='r')
fig2.show()

