# Consolidate all the ML utility functions here

import os 

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from PIL import Image

import numpy as np
import pandas as pd

from datetime import datetime as dt
import glob
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from astropy import units as u

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math
import timeit 
import faiss


def normalize_minus1_plus1(arr):
    return 2.*(arr-np.min(arr))/np.ptp(arr)-1

def log10plus_min(arr):
    """Take log10 of array, after making sure all values are positive by adding the minimum to the array, and replacing 0 with the new minimum of non-zero elements"""
    if np.min(arr) < 0:
        arr -= np.min(arr)
        nzarr = arr[arr != 0.0]
        arr[arr == 0.0] = np.min(nzarr)
    return np.log10(arr)

### GMM ###

def run_gmm(nc,data,covariance_type='full',random_state=2): #should really be **kwargs
    colors=px.colors.qualitative.Bold[:nc]
    gmm = GaussianMixture(n_components=nc,covariance_type=covariance_type,
                          random_state=random_state,max_iter=500)
    model=gmm.fit(data)
    labels = gmm.predict(data)
    gdf['cluster']=labels
    return gdf, colors, model

def compare_clusters(n1,n2,df):
    c12=df.query('cluster==@n1 or cluster ==@n2').dropna(how='all')
    c1=df.where(df.cluster==n1).dropna(how='all')
    c2=df.where(df.cluster==n2).dropna(how='all')
    d12=c12[pkeys]
    return d12,c12

def compute_nonlabel_scores(X,clusters):
    s=metrics.silhouette_score(X, clusters, metric='euclidean')
    ch=metrics.calinski_harabasz_score(X, clusters)
    db=metrics.davies_bouldin_score(X, clusters)
    return s,ch,db

def gmm_notruth_score(nc, df):
    '''df must have column cluster '''
    scores={'n1':[],'n2':[],'s':[],'ch':[],'db':[]}
    for n1 in range(nc):
        k=n1
        for n2 in range(k,nc):
            if n1 != n2:
                d12,c12=compare_clusters(n1,n2,df)
                s,ch,db=compute_nonlabel_scores(d12,c12.cluster)
                scores['n1'].append(n1)
                scores['n2'].append(n2)
                scores['s'].append(s)
                scores['ch'].append(ch)
                scores['db'].append(db)
    sdf=pd.DataFrame(scores)
    sdf.sort_values(by='s',ascending=False,inplace=True)
    return sdf

### get all those Xgboost ones and put them here, put the plots elsewhere ###s


def print_arr_stats(arr, ignore_nan=False):
    if ignore_nan:
        arr[np.isinf(arr)] = np.nan
        print(f"Shape: {arr.shape}\nMean: {np.nanmean(arr)}\nMin: {np.nanmin(arr)}\nMax: {np.nanmax(arr)}\nStd: {np.nanstd(arr)}\n")
    else:
        print(f"Shape: {arr.shape}\nMean: {np.mean(arr)}\nMin: {np.min(arr)}\nMax: {np.max(arr)}\nStd: {np.std(arr)}\n")

def scale_minus1_plus1(arr,elementwise=False):
    """scale an array to [-1,+1]. elementwise (axis 0) if desired """
    if not elementwise:
        return 2*(arr-np.min(arr))/np.ptp(arr)-1
    else:
        return np.array([2*(a-np.min(a))/np.ptp(a)-1 for a in arr])

def percent_zeros(arr):
    '''quickly calculate % of zeros'''
    return ((np.product(np.shape(arr)) - np.count_nonzero(arr))/np.product(np.shape(arr)))*100.

def fraction_nonzeros(arr):
    '''quickly calculate fraction of nonzeros'''
    return np.count_nonzero(arr)/np.product(np.shape(arr))

def count_nans(arr):
    '''quickly calculate # of NaNs'''
    return np.count_nonzero(np.isnan(arr))

def parse_timeit(result):
    ''''get times from %%timeit. Have to use %%capture result before %%timeit for this to work'''
    try:
        tstr = result.stdout.split('\n')
        if len(tstr) > 2:
            tstr = tstr[1]
        else:
            tstr = tstr[0]
    except AttributeError:
        tstr = result.__str__() #if using nice_timeit

    tmean=  tstr[tstr.find(':')+1:tstr.find('per loop')].strip()
    tmean, tstd = tmean.split("+-")
    tmean = tmean.strip()
    tstd = tstd.strip()
    munit = getattr(u,tmean[-2:].strip())
    sunit = getattr(u,tstd[-2:].strip()) #what if it's minutes?
    tout = float(tmean[:-2])*munit
    sout = float(tstd[:-2])*sunit
    return tout, sout

def _format_time(timespan, precision=3):
    """Formats the timespan in a human readable form"""
    units = ["s", "ms", "us", "ns"]
    scaling = [1, 1e3, 1e6, 1e9]
    if timespan > 0.0:
        order = min(-int(math.floor(math.log10(timespan)) // 3), 3)
    else:
        order = 3
    scaled_time = timespan * scaling[order]
    unit = units[order]
    return f"{scaled_time:.{precision}g} {unit}"


class TimeitResult(object):
    """
    Object returned by the timeit magic with info about the run.

    Contains the following attributes :

    loops: (int) number of loops done per measurement
    repeat: (int) number of times the measurement has been repeated
    best: (float) best execution time / number
    all_runs: (list of float) execution time of each run (in s)
    compile_time: (float) time of statement compilation (s)
    """

    def __init__(self, loops, repeat, best, worst, all_runs, compile_time, precision):
        self.loops = loops
        self.repeat = repeat
        self.best = best
        self.worst = worst
        self.all_runs = all_runs
        self.compile_time = compile_time
        self._precision = precision
        self.timings = [dt / self.loops for dt in all_runs]

    @property
    def average(self):
        return math.fsum(self.timings) / len(self.timings)

    @property
    def stdev(self):
        mean = self.average
        return (
            math.fsum([(x - mean) ** 2 for x in self.timings]) / len(self.timings)
        ) ** 0.5

    def __str__(self):
        return "{mean} {pm} {std} per loop (mean {pm} std. dev. of {runs} run{run_plural}, {loops} loop{loop_plural} each)".format(
            pm="+-",
            runs=self.repeat,
            loops=self.loops,
            loop_plural="" if self.loops == 1 else "s",
            run_plural="" if self.repeat == 1 else "s",
            mean=_format_time(self.average, self._precision),
            std=_format_time(self.stdev, self._precision),
        )


def nice_timeit(
    stmt="pass",
    setup="pass",
    number=0,
    repeat=None,
    precision=3,
    timer_func=timeit.default_timer,
    globals=None,
):
    """Time execution of a Python statement or expression."""

    if repeat is None:
        repeat = 7 if timeit.default_repeat < 7 else timeit.default_repeat

    timer = timeit.Timer(stmt, setup, timer=timer_func, globals=globals)

    # Get compile time
    compile_time_start = timer_func()
    compile(timer.src, "<timeit>", "exec")
    total_compile_time = timer_func() - compile_time_start

    # This is used to check if there is a huge difference between the
    # best and worst timings.
    # Issue: https://github.com/ipython/ipython/issues/6471
    if number == 0:
        # determine number so that 0.2 <= total time < 2.0
        for index in range(0, 10):
            number = 10 ** index
            time_number = timer.timeit(number)
            if time_number >= 0.2:
                break

    all_runs = timer.repeat(repeat, number)
    best = min(all_runs) / number
    worst = max(all_runs) / number
    timeit_result = TimeitResult(
        number, repeat, best, worst, all_runs, total_compile_time, precision
    )

    # Check best timing is greater than zero to avoid a
    # ZeroDivisionError.
    # In cases where the slowest timing is lesser than a microsecond
    # we assume that it does not really matter if the fastest
    # timing is 4 times faster than the slowest timing or not.
    if worst > 4 * best and best > 0 and worst > 1e-6:
        print(
            f"The slowest run took {worst / best:.2f} times longer than the "
            f"fastest. This could mean that an intermediate result "
            f"is being cached."
        )

    print(timeit_result)

    if total_compile_time > 0.1:
        print(f"Compiler time: {total_compile_time:.2f} s")
    return timeit_result

def imnet_subset(data, labels, ncats = 100, subcats = 3):
    if not isinstance(subcats, list): #choose randomly
        idx = np.random.choice(ncats, size=subcats, replace=False)
    else:
        idx = subcats
    if not isinstance(data, np.ndarray):
        data = np.array(data)
        
    datlist = []
    lablist = []
    for i in idx:
        ix = np.where(labels==i) 
        dat = data[ix,:]
        lab = labels[ix]
        datlist.append(dat)
        lablist.append(lab)
    subarr = np.hstack(datlist).squeeze()
    sublabs = np.hstack(lablist).squeeze()
    return subarr, sublabs

def reindex_labels(labelarr, new_labels = None):
    old_labels = np.unique(labelarr)
    #if not new_labels: #default 0-N
    #    newlabs = np.arange(len(old_labels))
    #else:
    #    newlabs = new_labels
    for i,o in enumerate(old_labels):
        labelarr[labelarr == o] = i
    return labelarr

def fit_pca(data, labels=None, n_components=15):
    pca = PCA(n_components=n_components)
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    components = pipe.fit_transform(data)
    if isinstance(labels,np.ndarray):
        idx = np.array(labels).reshape(len(components),1)
    else:
        idx = np.arange(len(data)).reshape(len(data),1)
    comp = np.hstack([components,idx])
    labels = {str(i): f"PC {i+1} ({var:.1f}%)" for i, var in enumerate(pca.explained_variance_ratio_ * 100)}
    siglabels = sum([1 for i,var in enumerate(pca.explained_variance_ratio_) if var > 0.01])
    #print(np.cumsum(pca.explained_variance_ratio_))
    return comp, labels, siglabels

def n99_variance_explained(data, max_comp = 50):
    pca = PCA(n_components=max_comp)
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    components = pipe.fit_transform(data)
    idx = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.99)
    try:
        return idx[0][0]
    except IndexError:
        print(f"Try with larger max_comp > {max_comp}")

def plot_pca_cutoff(data, max_comp = 50, cutoff = .95, title=None):
    pca = PCA(n_components=max_comp)
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    components = pipe.fit_transform(data)
    cumsum = np.cumsum(pca.explained_variance_ratio_) 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(0,len(cumsum)), y=cumsum, mode = 'lines+markers'))
    fig.add_hline(y=cutoff, name = f"{int(cutoff*100)}%")
    fig.update_layout(title = title, xaxis_title = "Number of components", xaxis_range = [0,max_comp], yaxis_range = [0,1],yaxis_title = "Cumulative sum of explained variance", width=400)
    return fig
    
def plot_pca(comp, labels, siglabels, n_components, max_comp = -1, marker_size=4, xaxis_range= None, yaxis_range=None, **kwargs):
    minn = min(siglabels, max_comp)
    if minn > 0:
        dimensions = range(minn)
    else: 
        dimensions = range(siglabels)#max_comp)
    minn = dimensions[-1]
    #labs = list(labels.values())[:minn]
    #print(labs)
    fig = px.scatter_matrix(
    comp,
    labels = labels,
    dimensions = dimensions,
    hover_data = [n_components],
    color = n_components, **kwargs)
    fig.update_traces(diagonal_visible=False, marker={"size":marker_size})
    if xaxis_range is not None:
        fig.update_layout({"xaxis"+str(i+1): dict(range = xaxis_range) for i in range(minn)})
    if yaxis_range is not None:
        fig.update_layout({"yaxis"+str(i+1): dict(range = yaxis_range) for i in range(minn)})
    fig.update_layout(height=80*minn)
    return fig

def transform_pca(X, n):
    pca = PCA(n_components=n)
    pca.fit(X)
    X_new = pca.inverse_transform(pca.transform(X))
    return X_new

def plot_pca_data(X_scaled, scaled = False, max_comp = 12, marker_size=2, xaxis_range= None, yaxis_range=None, title=None, **kwargs):
    if not scaled:
        X_scaled = StandardScaler().fit_transform(X_scaled)
    dimensions = range(max_comp)#max_comp)
    minn = dimensions[-1]
    rows = 1+ minn//3
    cols = 1+ minn%3
    #print(rows, cols)
    fig = make_subplots(rows = rows, cols = cols, subplot_titles = [f"{d+1} components" for d in dimensions])
    for d in dimensions:
        row = 1+ d//3
        col = 1+ d%3
        #print(d, row, col)
        X_new = transform_pca(X_scaled, d+1)
        fig.add_trace(go.Scatter(x=X_scaled[:,0], y=X_scaled[:,1], marker_color = 'gray', mode='markers'), row = row, col = col)
        fig.add_trace(go.Scatter(x=X_new[:,0], y=X_new[:,1], marker_color='green', mode='markers'), row = row, col = col)
        #fig.update_yaxes(scaleanchor = 'x', scaleratio = 1, row=row, col=col)
    fig.update_traces(marker={"size":marker_size})
    if xaxis_range is not None:
        fig.update_layout({"xaxis"+str(i+1): dict(range = xaxis_range) for i in range(minn)})
    if yaxis_range is not None:
        fig.update_layout({"yaxis"+str(i+1): dict(range = yaxis_range) for i in range(minn)})
    fig.update_layout(height= 80*minn, showlegend=False, title = title)
    return fig

def rename_heads(imdir, prefix):
    pwd = os.getcwd()
    os.chdir(imdir)
    hh = glob.glob("attn-head*.png")
    for h in hh:
        os.rename(h, f"{prefix}_{h}")
    os.rename(f"img.png",f"{prefix}_img.png")
    os.chdir(pwd)

def plot_attn_heads(imdir, prefix, base_img = None, crop_head_titles=True):
    hh = glob.glob(f"{imdir}{prefix}*.png")
    fig, ax = plt.subplots(2,4, figsize = (10,6))
    for i,h in enumerate(sorted(hh)):
        im = Image.open(h).convert("L")
        ax[i//4,i%4].imshow(im,origin='lower')
        if crop_head_titles and "img" not in h:
            head_title = f"attn-head {i}"
        else: 
            head_title = f"{h[h.rfind('/')+1:-4]}"
        ax[i//4,i%4].set_title(head_title, fontsize = 11)
        ax[i//4,i%4].axis('off')
    #earr = np.empty((2,2))
    #earr[:] = np.nan
    if not base_img:
        aa = np.log10(Image.open(hh[-1]).convert("L"))
    else: 
        aa = np.log10(Image.open(base_img).convert("L"))
    ax[1,3].imshow(aa, vmin = np.min(aa), vmax = np.max(aa), origin='lower')
    ax[1,3].set_title("log 10 img")
    ax[1,3].axis("off")
    #fig.colorbar(cbar, ax = ax.ravel().tolist())
    return fig

def faiss_query(d,data, k=5, xq_idx = None):
    index = faiss.IndexFlatL2(d)
    if not xq_idx:
        xq_idx = np.random.randint(len(data))
    #print(xq_idx)
    xq = data[xq_idx].reshape((1,d))
    data_minus_xq = np.delete(data, xq_idx, axis=0)
    index.add(data_minus_xq)
    D, I = index.search(xq, k) 
    return xq_idx, I

def plot_faiss_results(xq_idx, I, img, loader, vmin=None, vmax=None, labels=None):
    k = len(I[0]) + 1
    ncols = (k+1)//2
    
    if type(img) == str: #it's a folder
        aa = sorted(glob.glob(f"{img}/*.npy"))
        new_img = []
        for n in I[0]:
            new_img.append(loader(aa[n]))#Image.open(aa[n]))
        img = new_img
        img.append(loader(aa[xq_idx]))#Image.open(aa[xq_idx]))
        print(np.min(img), np.max(img),np.mean(img))
    fig, ax = plt.subplots(2,ncols, figsize=(10,6))
    for n,i in enumerate(I[0]):
        if i < xq_idx:
            j = i
        else:
            j = i+1 #account for missing index
        if labels is None:
            title = I[0][n]
        else: 
            title = labels[I[0][n]]
        if len(img) == len(I[0]) + 1:
            ax[n//ncols][n%ncols].imshow(img[n], vmin=vmin, vmax=vmax)
            ax[n//ncols][n%ncols].set_title(title)
        else:
            ax[n//ncols][n%ncols].imshow(img[I[0][n]], vmin=vmin, vmax=vmax)
            ax[n//ncols][n%ncols].set_title(title)
        #    ax[n//ncols][n%ncols].set_title(MGCLS_extract_source_name(aa[I[0][n]]))
        ax[n//ncols][n%ncols].axis("off")
        if len(img) == len(I[0]) + 1:
            ax[-1][-1].imshow(img[-1], vmin=vmin, vmax=vmax)
        else:
            ax[-1][-1].imshow(img[xq_idx], vmin=vmin, vmax=vmax)
    #ax[-1][-1].set_title(f"{MGCLS_extract_source_name(aa[xq_idx])}\n(Query)")
    if labels is None:
        qtitle = xq_idx
    else: 
        qtitle = labels[xq_idx]
    ax[-1][-1].set_title(f"{qtitle} Query")
    ax[-1][-1].axis("off")
    return fig

def faiss_query2(d,data, feats, k=5):
    index = faiss.IndexFlatL2(d)
    #if not xq_idx:
    #    xq_idx = np.random.randint(len(data))
    #print(xq_idx)
    xq = data.reshape((1,d))
    #data_minus_xq = np.delete(data, xq_idx, axis=0)
    index.add(feats)
    D, I = index.search(xq, k) 
    return I

def plot_faiss_results2(xq_idx, I, img, loader, vmin=None, vmax=None, labels=None, qname=''):
    k = len(I[0]) + 1
    ncols = (k+1)//2
    
    if type(img) == str: #it's a folder
        aa = sorted(glob.glob(f"{img}/*.npy"))
        new_img = []
        for n in I[0]:
            new_img.append(loader(aa[n]))#Image.open(aa[n]))
        img = new_img
        #img.append(loader(aa[xq_idx]))#Image.open(aa[xq_idx]))
        #print(np.min(img), np.max(img),np.mean(img))
    fig, ax = plt.subplots(2,ncols, figsize=(10,6))
    for n,i in enumerate(I[0]):
        #if i < xq_idx:
        #    j = i
        #else:
        #    j = i+1 #account for missing index
        if labels is None:
            title = I[0][n]
        else: 
            title = labels[I[0][n]]
        #if len(img) == len(I) + 1:
        ax[n//ncols][n%ncols].imshow(img[n], vmin=vmin, vmax=vmax)
        ax[n//ncols][n%ncols].set_title(title)
        #else:
        #    ax[n//ncols][n%ncols].imshow(img[I[0][n]], vmin=vmin, vmax=vmax)
        #    ax[n//ncols][n%ncols].set_title(title)
        #    ax[n//ncols][n%ncols].set_title(MGCLS_extract_source_name(aa[I[0][n]]))
        ax[n//ncols][n%ncols].axis("off")
        if len(img) == len(I) + 1:
            ax[-1][-1].imshow(img[-1], vmin=vmin, vmax=vmax)
        else:
            ax[-1][-1].imshow(np.load(xq_idx), vmin=vmin, vmax=vmax)
    #ax[-1][-1].set_title(f"{MGCLS_extract_source_name(aa[xq_idx])}\n(Query)")
    #print(xq_idx)
    #if labels is None:
    #    qtitle = xq_idx
    #else: 
    #    qtitle = labels[xq_idx]
    ax[-1][-1].set_title(f"{qname} Query")
    ax[-1][-1].axis("off")
    return fig

def n_same_results(I0,I1, n_only = False, indices = True):
    same = set(I0[0]).intersection(set(I1[0]))
    if n_only:
        return len(same)
    elif indices:
        return sorted([list(I0[0]).index(s) for s in same])
    else:
        return same