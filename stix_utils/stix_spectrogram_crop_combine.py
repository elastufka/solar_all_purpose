import numpy as np

from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
import warnings

from datetime import datetime as dt
from datetime import timedelta as td

def fits_time_corrections(primary_header, tstart, tend):
  creation_date = Time(dt.now()).isot
  date_obs = Time(tstart).isot
  date_beg = date_obs
  date_end = Time(tend).isot
  date_avg = Time(Time(tstart).mjd + (Time(tend).mjd - Time(tstart).mjd)/2., format = 'mjd').isot #average the date
  dateref = date_obs

  date_ear = Time(Time(tstart).datetime + td(seconds = primary_header['EAR_TDEL'])).isot
  date_sun = Time(Time(tstart).datetime - td(seconds = primary_header['SUN_TIME'])).isot

  #OBT_BEG =
  #OBT_END = #what are these?

  #OBT_BEG = '0703714990:39319'
  #OBT_END = '0703736898:43389'
  primary_header.set('DATE',creation_date)
  primary_header.set('DATE_OBS',date_obs)
  primary_header.set('DATE_BEG',date_beg)
  primary_header.set('DATE_END',date_end)
  primary_header.set('DATE_AVG',date_avg)
  primary_header.set('DATEREF',dateref)
  primary_header.set('DATE_EAR',date_ear)
  primary_header.set('DATE_SUN',date_sun)
  primary_header.set('MJDREF', Time(tstart).mjd)
  # Note that MJDREF in these files is NOT 1979-01-01 but rather the observation start time! Affects time calculation later, but further processing of the file corrects this

  return primary_header

def open_spec_fits(filename):
    """Open a L1, L1A, or L4 FITS file and return the HDUs"""
    with fits.open(filename) as hdul:#when to close this?
        primary_header = hdul[0].header.copy()
        control = hdul[1].copy()
        data = hdul[2].copy()
        energy = hdul[3].copy() if hdul[3].name == 'ENERGIES' else hdul[4].copy()
    return primary_header, control, data, energy
       
def fits_time_to_datetime(*args, factor = 1):
    if isinstance(args[0], str):
        primary_header, _, data, _ = open_spec_fits(args[0])
        data_table = data.data
    else:
        primary_header, data_table = args
    time_bin_center=data_table['time']
    duration = data_table['timedel']
    start_time = dt.strptime(primary_header['DATE_BEG'],"%Y-%m-%dT%H:%M:%S.%f")
    spectime = Time([start_time + td(seconds = bc/factor - d/(2.*factor)) for bc,d in zip(time_bin_center, duration)])
    return spectime

def time_select_indices(tstart, tend, primary_header, data_table, factor = 1.):

    spectime = fits_time_to_datetime(primary_header, data_table, factor = factor).mjd
    
    if tstart:
      tstart_mjd = Time(tstart).mjd
    else:
      tstart_mjd = spectime[0]
    if tend:
      tend_mjd = Time(tend).mjd
    else:
      tend_mjd = spectime[-1]

    #get indices for tstart and tend
    tselect = np.where(np.logical_and(spectime > tstart_mjd, spectime <= tend_mjd))[0]
    idx0, idx1 = tselect[0],tselect[-1] #first and last indices
    return idx0, idx1

def spec_fits_crop(fitsfile, tstart,tend, outfilename = None):

    primary_header, control, data, energy = open_spec_fits(fitsfile)

    #get indices for tstart and tend
    idx0, idx1 = time_select_indices(tstart, tend, primary_header, data.data) #first and last indices

    #crop data table
    count_names = data.data.names
    table_data = []
    for n in count_names:
      if data.data[n].ndim >1:
        new_data = data.data[n][idx0:idx1,:]
      else:
        if n == 'time': #this has to be done differently since it is relative to timezero
          timevec = fits_time_to_datetime(primary_header, data.data).mjd #- Time(primary_header['DATE_BEG']).mjd
          timevec -= timevec[idx0] #relative to new start time
          new_data = timevec[idx0:idx1]*86400
        else:
          new_data = data.data[n][idx0:idx1]
      table_data.append(new_data)
    #count_table = Table([data.data[n][idx0:idx1,:] if data.data[n].ndim >1 else data.data[n][idx0:idx1] for n in count_names], names = count_names)
    count_HDU = fits.BinTableHDU(data = Table(table_data, names = count_names))

    #insert other keywords into counts header (do datasum and checksum later)
    count_HDU.header.set('EXTNAME', 'DATA', 'extension name')

    #correct keywords in primary header
    primary_header = fits_time_corrections(primary_header, tstart, tend)
    if not outfilename:
      outfilename = f"{fitsfile[:-5]}_{Time(tstart).datetime:%H%M%S}_{Time(tend).datetime:%H%M%S}.fits"
    primary_header.set('FILENAME', outfilename[outfilename.rfind('/')+1:])
    primary_HDU = fits.PrimaryHDU(header = primary_header)

    hdul = fits.HDUList([primary_HDU, control, count_HDU, energy])
    hdul.writeto(outfilename)

    return outfilename
    
def spec_fits_concatenate(fitsfile1, fitsfile2, tstart = None,tend = None, outfilename = None):
    """Concatenate two spectrogram files. Check to make sure that there is no time gap.
    Overlapping time ranges are handled with priority given to the first fits file (can change this later)"""

    primary_header1, control1, data1, energy1= open_spec_fits(fitsfile1)
    primary_header2, control2, data2, energy2 = open_spec_fits(fitsfile2)

    #check that energy tables are the same. If not, there is an error
    for n in energy1.data.names:
        if not np.allclose(energy1.data[n], energy2.data[n]):
            raise ValueError(f"Values for {n} in energy table are different in {fitsfile1} and {fitsfile2}!")

    #check that detector, pixel, and energy masks masks are the same
    for n in ['pixel_masks','detector_masks','pixel_mask','detector_mask','energy_bin_mask']:
      if n in control1.data.names:
          if not np.allclose(control1.data[n], control2.data[n]):
              raise ValueError(f"Values for {n} in control table are different in {fitsfile1} and {fitsfile2}!")

    #get indices for tstart and tend - assuming tstart is in first file and tend in second.
    # look at this in more detail later
    idx0, idx2 = 0, None #index of start time in file 1, end time in file 2
    if tstart:
      idx0, _ = time_select_indices(tstart, tend, primary_header1, data1.data) #first index
    else:
      tstart = primary_header1['DATE_BEG']

    tend1 = primary_header1['DATE_END'] #end time of first file in case there is ovelap
    if not tend:
      tend = primary_header2['DATE_END']
    idx1, idx2 = time_select_indices(tend1, tend, primary_header2, data2.data) # start time in file 2, end time in file 2

    #check that spectrograms are consecutive in time, ie that time1[-1] + timedel1[-1] == time2[0] within some tolerance
    # no longer accurate if using 2 indices for second data table?
    spec_time_gap = fits_time_to_datetime(primary_header2, data2.data).datetime[0] - fits_time_to_datetime(primary_header1, data1.data).datetime[-1]
    if spec_time_gap.total_seconds() > data1.data['timedel'][-1]:
      warnings.warn(f"Gap of {spec_time_gap.total_seconds() - data1.data['timedel'][-1]:.3f}s between spectrogram files {fitsfile1} and {fitsfile2}")

    #concatenate data tables
    count_names = data1.data.names
    table_data = []
    for n in count_names:
      #rint(n,data1.data[n].shape)
      if data1.data[n].ndim >1:
        new_data = np.concatenate([data1.data[n][idx0:,:], data2.data[n][idx1:idx2,:]])
      else:
        if n == 'time': #this has to be done differently since it is relative to timezero
          timevec1 = fits_time_to_datetime(primary_header1, data1.data).mjd
          new_start_time = timevec1[idx0]
          #print(f"original start time {Time(primary_header1['DATE_BEG']).mjd}, new_start_time {new_start_time}, spec2_start {Time(primary_header2['DATE_BEG']).mjd}")
          timevec1 -= new_start_time #relative to new start time
          timevec2 = fits_time_to_datetime(primary_header2, data2.data).mjd - new_start_time
          new_data = np.concatenate([timevec1[idx0:]*86400, timevec2[idx1:idx2]*86400])
        else:
          new_data = np.concatenate([data1.data[n][idx0:], data2.data[n][idx1:idx2]])
      #print('new data', new_data.shape)
      table_data.append(new_data)

    count_HDU = fits.BinTableHDU(data = Table(table_data, names = count_names))

    #insert other keywords into counts header (do datasum and checksum later)
    count_HDU.header.set('EXTNAME', 'DATA', 'extension name')

    #correct keywords in primary header
    primary_header1 = fits_time_corrections(primary_header1, tstart, tend)
    if not outfilename:
      outfilename = f"{fitsfile1[:-5]}_{Time(tstart).datetime:%H%M%S}_{Time(tend).datetime:%H%M%S}.fits"
    primary_header1.set('FILENAME', outfilename[outfilename.rfind('/')+1:])
    primary_HDU = fits.PrimaryHDU(header = primary_header1)
    #print(primary_header1['DATE_BEG'],primary_header1['TIMEZERO'], primary_header1['MJDREF'])
    hdul = fits.HDUList([primary_HDU, control1, count_HDU, energy1])
    hdul.writeto(outfilename)

    return outfilename

