#import dash_html_components as html

import pandas as pd
import numpy as np
import glob
import os
#from PIL import Image
from matplotlib.colors import Normalize
from matplotlib import cm

from astropy import units as u
from astropy.coordinates import SkyCoord
from plotly.subplots import make_subplots
from datetime import datetime as dt
from datetime import timedelta as td
import plotly.graph_objects as go
from skimage.transform import downscale_local_mean
import sunpy.map
from fake_maps_plotly import *
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame,pixel_to_skycoord, skycoord_to_pixel
import sunpy.net as sn
import plotly.colors
import plotly.express as px
#from sunpy_map_utils import query_hek
#import plotly.express as px
#import dash_html_components as html

def load_data(datapath='/Users/wheatley/Documents/Solar/STIX/data/'):
    cwd=os.getcwd()
    dirs=glob.glob(datapath+"*")
    orig,rot=[],[]
    newhekdata=[]
    hek_json=glob.glob(datapath+'Event_HEK_SO.json')[0]
    dirs=[d for d in dirs if 'HEK' not in d]
    print(dirs)
    hekdf=pd.read_json(hek_json)
    
    for d in dirs:
        os.chdir(d)
        #get maps
        fitsf=glob.glob("*.fits")
        if fitsf == []:
            fitsf=glob.glob("*.json") #temp workaround
        for f in fitsf:
            if f.startswith("AIA"):
                ofile=f
            elif f.startswith("rotated"):
                rfile=f
        #try and find json files of the same names...
        try:
            ojson=glob.glob(ofile[:-4]+'json')[0]
            odf=pd.read_json(ojson)
        except IndexError:
            omap=fake_map(sunpy.map.Map(ofile),binning=8,tickspacing=200)
            omap._cleanup()
            odf=omap.to_dataframe()
        try:
            rjson=glob.glob(rfile[:-4]+'json')[0]
            rdf=pd.read_json(rjson)#just merge them
        except IndexError:
            rmap=fake_map(sunpy.map.Map(rfile),binning=8,tickspacing=200)
            rmap._cleanup()
            rdf=rmap.to_dataframe()

        odf['observer']=reconstruct_observer(odf.iloc[0]) #or can take it directly from object
        rdf['observer']=reconstruct_observer(rdf.iloc[0])

        idx=pd.Index([format_foldername(datapath,d)])
        odf.set_index(idx,inplace=True,drop=True)
        rdf.set_index(idx,inplace=True,drop=True)
        orig.append(odf)
        rot.append(rdf)
        
        if idx.values[0] not in hekdf.Event.values: #query HEK for flares corresponding to this time
            print('Updating HEK dataframe for %s' % idx.values[0])
            idx_dt=dt.strptime(idx.values[0],'%Y-%m-%dT%H:%M:%S')
            qdf=query_hek([idx_dt,idx_dt+td(minutes=1)])
            qdf['Event']=[idx.values[0] for i,_ in qdf.iterrows()]
            #print(qdf)
            qdf=rotate_hek_coords(qdf,odf.observer.to_numpy()[0],odf.wcs.to_numpy()[0],rdf.observer.to_numpy()[0],rdf.wcs.to_numpy()[0])
            if qdf.empty:
                qdf=pd.DataFrame({'Event':idx.values[0],'hpc_x':None,'hpc_y':None,'hpc_bbox':None,'hpc_x_px':None,'hpc_y_px':None,'x_px_rotated':None,'y_px_rotated':None,'frm_identifier':None,'frm_name':None},index=pd.Index([0]))
            newhekdata.append(qdf)
            
    if newhekdata != []: #re-write HEK json
        newhekdata.append(hekdf)
        hdf=pd.concat(newhekdata)
        hdf.reset_index(drop=True,inplace=True)
        hdf.to_json(hek_json)
        hekdf=hdf
        print(f'Writing to {hek_json}')

    os.chdir(cwd)
    all_odf=pd.concat(orig)
    all_rot=pd.concat(rot)
    image_df=all_odf.merge(all_rot,left_index=True,right_index=True,suffixes=('_original','_rotated'))
    return image_df,hekdf
    
def format_foldername(datapath,fname):
    return fname[len(datapath):-4]+':'+fname[-4:-2]+':'+fname[-2:]
    
def load_example_data():
    orig,rot=[],[]
    jsons=glob.glob('data/*.json')
    for j in jsons:
        odf=pd.read_json(j)
        odf['observer']=reconstruct_observer(odf.iloc[0]) #or can take it directly from object
        idx=pd.Index([format_filename(j)])
        odf.set_index(idx,inplace=True,drop=True)

        if j.startswith('data/AIA'):
            orig.append(odf)
        else:
            rot.append(odf)
    
    all_odf=pd.concat(orig)
    all_rot=pd.concat(rot)
    image_df=all_odf.merge(all_rot,left_index=True,right_index=True,suffixes=('_original','_rotated'))
    return image_df
    
def format_filename(j):
    return j[j.find('_')+1:-9]+':'+j[-9:-7]+':'+j[-7:-5]
    
def reconstruct_observer(row):
    '''Use stored values to reconstruct the heliospheric observer object. Works for input of fake_map object or of dataframe row'''
    olon=row.olon*getattr(u,row.olon_unit) #unit
    olat=row.olat*getattr(u,row.olat_unit) #unit
    orad=row.orad*getattr(u,row.orad_unit) #unit
    observer=SkyCoord(olon,olat,orad,frame=row.obsframe,obstime=row.obstime)
    return observer
    
def image_grid(image_df,hdf,target,scale,zmin,zmax,cscale='Blues'):
    '''make the image grid subplots'''
    
    zmin=float(zmin)
    zmax=float(zmax)
    colorsc=getattr(px.colors.sequential,cscale.capitalize())
    ccolors, scal = plotly.colors.convert_colors_to_same_type(colorsc)
    clow=ccolors[0]
    chigh=ccolors[-1]
    #xmin,ymin=0,0
    cNorm = Normalize(vmin=zmin, vmax=zmax)
    scalarMap  = cm.ScalarMappable(norm=cNorm, cmap='gray' ) #think this one is Plasma
    
    if type(target) != list:
        target=[target]

    rows=1
    fig = make_subplots(rows=rows, cols=2,subplot_titles=[f"{target[0]} Earth POV",f"{target[0]} Solar Orbiter POV"])#,shared_xaxes=False,shared_yaxes=False)

    for t in target:
        #xmax,ymax=image_dict[target][binning][target][tidx[0],:,:].shape
        #im_aspect=int(ymax/xmax)
        #img_width=200
        #img_height=200*im_aspect
        #rows_per_method=(int(len(targets)/cols))
        omap=np.array(image_df.binned_data_original[t])
        #print(type(omap),omap.shape)
        omap[omap==0]=np.nan #mask zeros
        fig.add_trace(go.Heatmap(z=omap,name=t,zmin=zmin,zmax=zmax,colorscale=cscale,customdata=image_df.customdata_original[t],hovertemplate='x: %{customdata[0]}"<br>y: %{customdata[1]}"<br>%{z} DN/s <extra></extra>'),row=1,col=1)
        fig.add_trace(go.Heatmap(z=image_df.binned_data_rotated[t],zmin=zmin,zmax=zmax,colorscale=cscale,customdata=image_df.customdata_rotated[t],hovertemplate='x: %{customdata[0]}"<br>y: %{customdata[1]}"<br>%{z} DN/s <extra></extra>'),row=1,col=2)
        
        fig.update_xaxes(title="Helioprojective Longitude (arcsec)",tickmode='array',tickvals=image_df.xtickvals_original[t],ticktext=image_df.ticktextx_original[t],showgrid=False,zeroline=False,row=1,col=1,range=image_df.xlim_original[t])
        fig.update_yaxes(title="Helioprojective Latitude (arcsec)",tickmode='array',tickvals=image_df.ytickvals_original[t],ticktext=image_df.ticktexty_original[t],showgrid=False,zeroline=False,range=image_df.ylim_original[t],scaleanchor = "x",scaleratio = 1,row=1,col=1)
        fig.update_xaxes(title="Helioprojective Longitude (arcsec)",tickmode='array',tickvals=image_df.xtickvals_rotated[t],ticktext=image_df.ticktextx_rotated[t],showgrid=False,zeroline=False,row=1,col=2,range=image_df.xlim_rotated[t])
        fig.update_yaxes(title="Helioprojective Latitude (arcsec)",tickmode='array',tickvals=image_df.ytickvals_rotated[t],ticktext=image_df.ticktexty_rotated[t],showgrid=False,zeroline=False,range=image_df.ylim_rotated[t],scaleanchor = "x2",scaleratio = 1,row=1,col=2)
    
        #overplot HEK events if present
        hek_events=hdf.where(hdf.Event==t).dropna(how='all')
        binning=image_df.binning_original[t]
        if not hek_events.empty:
            #print(hek_events.hpc_x_px,hek_events.hpc_y_px,hek_events.x_px_rotated,hek_events.y_px_rotated)
            marker=dict(size=15,symbol='cross',color=clow,opacity=.5,line=dict(color=chigh,width=2))
            marker2=dict(size=15,symbol='triangle-up',color=chigh,opacity=.5,line=dict(color=clow,width=2))
            cdata0=np.vstack([hek_events.hpc_x,hek_events.hpc_y]) #one too many when CFL involved
            #htemp0='x: %{customdata[0]}"<br>y: %{customdata[1]}" <extra></extra>'
            #cdata1=np.array(hek_events.hpc_x_r,hek_events.hpc_y)
            #test
            #fig.add_trace(go.Scatter(x=[0],y=[0],marker=marker,name='origin',customdata=np.array([100]),hovertemplate='%{customdata}<br>%{customdata[0]} points!'),row=1,col=1)
            fig.add_trace(go.Scatter(x=hek_events.hpc_x_px/binning,y=hek_events.hpc_y_px/binning,mode='markers',name='AIA Flare',marker=marker,customdata=cdata0.T,hovertemplate='x: %{customdata[0]:.1f}"<br>y: %{customdata[1]:.1f}" <extra></extra>'),row=1,col=1)
            fig.add_trace(go.Scatter(x=hek_events.x_px_rotated/binning,y=hek_events.y_px_rotated/binning,mode='markers',name='AIA Flare',marker=marker),row=1,col=2) #need the HPC rotated coords...
            if 'CFL_X_px' in hek_events.keys():
                cdata2=np.hstack([np.array(hek_events['CFL_LOC_X(arcsec)'].iloc[0]),np.array(hek_events['CFL_LOC_Y (arcsec)'].iloc[0])])

                #print(cdata2) #still not the correct labels... something's up with the coordinate rotation. lack of reprojection affecting pixel calcualtion? likely
                if not np.isnan(cdata2).all():
                        fig.add_trace(go.Scatter(x=hek_events.CFL_X_px/binning,y=hek_events.CFL_Y_px/binning,mode='markers',name='SO Flare',marker=marker2,customdata=np.vstack([cdata2,cdata2]),hovertemplate='STIX CFL <br>x: %{customdata[0]:.1f}"<br>y: %{customdata[1]:.1f}" <extra></extra>'),row=1,col=2) #be smarter about picking colors, add hovertext info, move legend
        
        fig.update_layout(legend=dict(
            orientation='h',
            yanchor="top",
            y=1.2,
            xanchor="left",
            x=0.01
        ))
           
            #add polygons too once I remember how to deal with those
    
    return rows,fig

def arr_to_PIL(imdata,scalarMap):
    seg_colors = scalarMap.to_rgba(imdata)
    img = Image.fromarray(np.uint8(seg_colors*255))
    return img
    
#        to_zoom,new_layout=transform_zoom(relayoutData)
#fig['layout'][to_zoom]=new_layout

def _world_to_pixel(x,y,obs_in,wcs_in,rotate_obs=None,rotate_wcs=None):
    '''not to be confused with skycoord and wcs world_to_pixel '''
    #should already do this in the hek dataframe probably... along with rotation.
    wcs0=WCS(wcs_in)
    coord=SkyCoord(x*u.arcsec,y*u.arcsec,frame='helioprojective',observer=obs_in) #should have obstime
    if type(rotate_obs) != type(None) and type(rotate_wcs) != type(None):
        wcs1=WCS(rotate_wcs)
        testcoord=SkyCoord(0*u.arcsec,0*u.arcsec,frame='helioprojective',observer=rotate_obs,obstime=rotate_obs.obstime)
        rotcoord_wcs=coord.transform_to(testcoord.frame)
        return coord.to_pixel(wcs=wcs0),rotcoord_wcs.to_pixel(wcs1)
    return coord.to_pixel(wcs=wcs0)

def transform_zoom(relayoutData,wcs_original,wcs_rotated,observer_original,observer_rotated,binning=8.):
    '''given zoom information for one subplot, figure out the equivalent coordinates and zoom for the other '''
    kk=list(relayoutData.keys())
    zoomed_axis=0 if kk[0][5] == '.' else 2
    wcs0=WCS(wcs_original)
    wcs1=WCS(wcs_rotated)

    if zoomed_axis==0:
        current_wcs=wcs0
        output_wcs=wcs1
        current_observer=observer_original
        output_observer=observer_rotated
        new_layout={'xaxis2':{'range':[]},'yaxis2':{'range':[]}}
        zaxes=['xaxis2','yaxis2']
    else:
        current_wcs=wcs1
        output_wcs=wcs0
        current_observer=observer_rotated
        output_observer=observer_original
        new_layout={'xaxis':{'range':[]},'yaxis':{'range':[]}}
        zaxes=['xaxis','yaxis']
        
    
    outkeys=list(new_layout.keys())
    #transform axis ranges to to WCS
    bottom_left_wcs_noobs=pixel_to_skycoord(relayoutData[kk[0]]*binning,relayoutData[kk[2]]*binning,current_wcs,mode='wcs') #no observer!
    bottom_left_wcs=SkyCoord(bottom_left_wcs_noobs.Tx,bottom_left_wcs_noobs.Ty,frame='helioprojective',observer=current_observer) #add observer
    top_right_wcs_noobs=pixel_to_skycoord(relayoutData[kk[1]]*binning,relayoutData[kk[3]]*binning,current_wcs,mode='wcs')
    top_right_wcs=SkyCoord(top_right_wcs_noobs.Tx,top_right_wcs_noobs.Ty,frame='helioprojective',observer=current_observer) #add observer
    
    #transform from WCS to pixel coords of other subplot
    testcoord=SkyCoord(0*u.arcsec,0*u.arcsec,frame='helioprojective',observer=output_observer,obstime=output_observer.obstime) ##why is observer= None? that messes things up
    #print(testcoord,bottom_left_wcs.obstime)
    bottom_left_wcs_out=bottom_left_wcs.transform_to(testcoord.frame)
    bottom_left_pix=bottom_left_wcs_out.to_pixel(output_wcs)
    top_right_wcs_out=top_right_wcs.transform_to(testcoord.frame)
    top_right_pix=top_right_wcs_out.to_pixel(output_wcs)
    #need to divide by 8....
    
    #print(bottom_left_wcs,top_right_wcs)
    #print(bottom_left_wcs_out,top_right_wcs_out)
    #print(pixel_to_skycoord(bottom_left_pix[0],bottom_left_pix[1],output_wcs),pixel_to_skycoord(top_right_pix[0],top_right_pix[1],output_wcs))
    
    new_layout[outkeys[0]]['range']=[bottom_left_pix[0]//binning,top_right_pix[0]//binning]
    new_layout[outkeys[1]]['range']=[bottom_left_pix[1]//binning,top_right_pix[1]//binning]
    #what to do if it's out of the axis range? or if transformation gives NaN in skycoords? use height/width of rectangle instead
    #print(new_layout)
    
    return zaxes, new_layout
    
def query_hek(time_int,event_type='FL',obs_instrument='AIA',small_df=True,single_result=False):
    time = sn.attrs.Time(time_int[0],time_int[1])
    eventtype=sn.attrs.hek.EventType(event_type)
    #obsinstrument=sn.attrs.hek.OBS.Instrument(obs_instrument)
    res=sn.Fido.search(time,eventtype,sn.attrs.hek.OBS.Instrument==obs_instrument)
    tbl=res['hek']
    names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
    df=tbl[names].to_pandas()
    if df.empty:
        return df
    if small_df:
        df=df[['hpc_x','hpc_y','hpc_bbox','frm_identifier','frm_name']]
        df.drop_duplicates(inplace=True)
    if single_result: #select one
        aa=df.where(df.frm_identifier == 'Feature Finding Team').dropna()
        print(aa.index.values)
        if len(aa.index.values) == 1: #yay
            return aa
        elif len(aa.index.values) > 1:
            return pd.DataFrame(aa.iloc[0]).T
        elif aa.empty: #whoops, just take the first one then
            return pd.DataFrame(df.iloc[0]).T

    return df
    
def rotate_hek_coords(df,observer_in,wcs_in,observer_out,wcs_out):
    '''World to pixel and rotate for HEK event coords '''
    xpx,ypx,xrot,yrot=[],[],[],[]
    for i,row in df.iterrows():
        (pxx,pxy),(xr,yr)=_world_to_pixel(row.hpc_x,row.hpc_y,observer_in,wcs_in,rotate_obs=observer_out,rotate_wcs=wcs_out)
        #do this for each coord in the bounding box too, wow I really hope this isn't slow
        
        xpx.append(float(pxx))
        ypx.append(float(pxy))
        xrot.append(float(xr))
        yrot.append(float(yr))
    df['hpc_x_px']=xpx
    df['hpc_y_px']=ypx
    df['x_px_rotated']=xrot
    df['y_px_rotated']=yrot

    return df
