 #######################################
# stereo.py
# Erica Lastufka 4/5/2017  

#Description: Copy of SunPy tutorial
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
#import data_management as da
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy

import astropy.units as u
from astropy.coordinates import SkyCoord
import wcsaxes

import sunpy.map
import sunpy.coordinates
import sunpy.coordinates.wcs_utils
from sunpy.net import vso

stereo = (vso.attrs.Source('STEREO_B') &
          vso.attrs.Instrument('EUVI') &
          vso.attrs.Time('2011-01-01', '2011-01-01T00:10:00'))

aia = (vso.attrs.Instrument('AIA') &
       vso.attrs.Sample(24 * u.hour) &
       vso.attrs.Time('2011-01-01', '2011-01-02'))

wave = vso.attrs.Wave(30 * u.nm, 31 * u.nm)


vc = vso.VSOClient()
res = vc.query(wave, aia | stereo)

print(res)

files = vc.get(res).wait()

maps = {m.detector: m.submap((-1100, 1100) * u.arcsec,
                             (-1100, 1100) * u.arcsec) for m in sunpy.map.Map(files)}

fig = plt.figure(figsize=(15, 5))
for i, m in enumerate(maps.values()):
    ax = fig.add_subplot(1, 2, i+1, projection=m.wcs)
    m.plot(axes=ax)

aia_width = 200 * u.arcsec
aia_height = 250 * u.arcsec
aia_bottom_left = (-800, -300) * u.arcsec

m = maps['AIA']
fig = plt.figure()
ax = fig.add_subplot(111, projection=m.wcs)
m.plot(axes=ax)
m.draw_rectangle(aia_bottom_left, aia_width, aia_height)

subaia = maps['AIA'].submap(u.Quantity((aia_bottom_left[0],
                                        aia_bottom_left[0] + aia_width)),
                            u.Quantity((aia_bottom_left[1],
                                        aia_bottom_left[1] + aia_height)))
subaia.peek()

hpc_aia = SkyCoord((aia_bottom_left,
                    aia_bottom_left + u.Quantity((aia_width, 0 * u.arcsec)),
                    aia_bottom_left + u.Quantity((0 * u.arcsec, aia_height)),
                    aia_bottom_left + u.Quantity((aia_width, aia_height))),
                   frame=maps['AIA'].coordinate_frame)

print(hpc_aia)

hgs = hpc_aia.transform_to('heliographic_stonyhurst')
print(hgs)

hgs.D0 = maps['EUVI'].dsun
hgs.L0 = maps['EUVI'].heliographic_longitude
hgs.B0 = maps['EUVI'].heliographic_latitude

hpc_B = hgs.transform_to('helioprojective')
print(hpc_B)

fig = plt.figure(figsize=(15, 5))
for i, (m, coord) in enumerate(zip([maps['EUVI'], maps['AIA']],
                                   [hpc_B, hpc_aia])):
    ax = fig.add_subplot(1, 2, i+1, projection=m.wcs)
    m.plot(axes=ax)

    # coord[3] is the top-right corner coord[0] is the bottom-left corner.
    w = (coord[3].Tx - coord[0].Tx)
    h = (coord[3].Ty - coord[0].Ty)
    m.draw_rectangle(u.Quantity((coord[0].Tx, coord[0].Ty)), w, h,
                     transform=ax.get_transform('world'))

    subeuvi = maps['EUVI'].submap(u.Quantity((hpc_B[0].Tx, hpc_B[3].Tx)),
                              u.Quantity((hpc_B[0].Ty, hpc_B[3].Ty)))
subeuvi.peek()

fig = plt.figure(figsize=(15, 5))
for i, m in enumerate((subeuvi, subaia)):
    ax = fig.add_subplot(1, 2, i+1, projection=m.wcs)
    m.plot(axes=ax)
