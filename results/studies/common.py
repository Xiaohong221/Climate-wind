"""Shared read-only data access, spherical geometry and train-only preprocessing."""
from pathlib import Path
import os,json,tempfile
os.environ.setdefault('OMP_NUM_THREADS','3')
os.environ.setdefault('OPENBLAS_NUM_THREADS','3')
os.environ.setdefault('MPLCONFIGDIR','/tmp/climate-study-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/climate-study-xdg')
import numpy as np,pandas as pd
from scipy.ndimage import uniform_filter1d
EARTH=6371.0088
ROOT=Path(os.environ.get('CLIMATE_DATA_ROOT','/workspace/Climate-wind'))
BASE=Path(__file__).resolve().parent
CACHE=BASE/'cache';CACHE.mkdir(exist_ok=True)
FULL=pd.date_range('2000-01-01','2022-12-31')
SEED=20261028

def matrix(path):
    path=Path(path);out=CACHE/(path.stem+'.npy');meta=out.with_suffix('.source.json')
    st=path.stat();fingerprint={'path':str(path.resolve()),'size':st.st_size,'mtime_ns':st.st_mtime_ns}
    same=False
    if out.exists() and meta.exists():
        try:same=json.loads(meta.read_text())==fingerprint
        except (ValueError,OSError):pass
    if not same:
        a=np.loadtxt(path,dtype=np.float32)
        with tempfile.NamedTemporaryFile(dir=CACHE,suffix='.npy',delete=False) as f:
            np.save(f,a);tmp=f.name
        os.replace(tmp,out)
        with tempfile.NamedTemporaryFile(dir=CACHE,mode='w',suffix='.json',delete=False) as f:
            json.dump(fingerprint,f);tmp=f.name
        os.replace(tmp,meta)
    return np.load(out,mmap_mode='r')

def hourly(step=3):
    h=ROOT/'NA_5N45N_250E350E_Hourly'
    dates=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(h/f'NA_5N45N_250E350E_datelist_2000_2022_{step}H.csv').Date))
    files=sorted((h/f'{step}H').glob('*.txt'))
    a=np.concatenate([matrix(p) for p in files])
    assert a.shape==(len(dates),160)
    assert dates.equals(pd.date_range('2000-01-01','2022-12-31 23:00' if step==1 else '2022-12-31 21:00',freq=f'{step}h'))
    assert np.isfinite(a).all()
    grid=pd.read_csv(h/'NA_5N45N_250E350E_nodelist_5deg_160n.csv')
    assert np.allclose(grid.target_lat.to_numpy().reshape(8,20)[:,0],np.arange(7.5,45,5))
    assert np.allclose(grid.target_lon.to_numpy().reshape(8,20)[0],np.arange(252.5,350,5))
    return a,dates,grid

def tracks():
    d=pd.read_csv(ROOT/'data_hurricanes/ibtracs.NA.list.v04r01.csv',skiprows=[1],keep_default_na=False,low_memory=False)
    d=d[d.TRACK_TYPE.eq('main')].copy();d['t']=pd.to_datetime(d.ISO_TIME,format='mixed')
    for c in ['LAT','LON','USA_WIND','USA_PRES']:d[c]=pd.to_numeric(d[c],errors='coerce')
    d['x']=d.LON%360
    d=d.drop_duplicates(['SID','t']).sort_values(['SID','t'])
    return d

def inside(d):return d.LAT.between(5,45)&d.x.between(250,350)&d.BASIN.eq('NA')

def genesis(d):
    g=d[d.USA_STATUS.isin(['TD','TS','HU'])].groupby('SID',sort=False).head(1).copy()
    first=d.groupby('SID').t.min();g['left_censored']=g.apply(lambda z:z.t==first[z.SID],axis=1)
    return g[inside(g)&g.t.between(FULL.min(),FULL.max()+pd.Timedelta(hours=23))].copy()

def distance(lat,lon,y,x):
    lat,lon,y,x=map(np.deg2rad,[lat,lon,y,x]);dl=(x-lon+np.pi)%(2*np.pi)-np.pi
    return 2*EARTH*np.arcsin(np.sqrt(np.clip(np.sin((y-lat)/2)**2+np.cos(lat)*np.cos(y)*np.sin(dl/2)**2,0,1)))

def tangent(lat,lon,y,x):
    r=distance(lat,lon,y,x);la,lo,ya,xx=map(np.deg2rad,[lat,lon,y,x]);dl=(xx-lo+np.pi)%(2*np.pi)-np.pi
    bearing=np.arctan2(np.sin(dl)*np.cos(ya),np.cos(la)*np.sin(ya)-np.sin(la)*np.cos(ya)*np.cos(dl))
    return np.column_stack([r*np.sin(bearing),r*np.cos(bearing)])

def destination(lat,lon,en):
    la=np.deg2rad(np.asarray(lat));lo=np.deg2rad(np.asarray(lon));en=np.asarray(en);r=np.linalg.norm(en,axis=1)/EARTH;b=np.arctan2(en[:,0],en[:,1])
    y=np.arcsin(np.clip(np.sin(la)*np.cos(r)+np.cos(la)*np.sin(r)*np.cos(b),-1,1))
    x=lo+np.arctan2(np.sin(b)*np.sin(r)*np.cos(la),np.cos(r)-np.sin(la)*np.sin(y))
    return np.rad2deg(y),np.rad2deg(x)%360

def daynumber(dates):
    dates=pd.DatetimeIndex(dates)
    # Fixed 365-day climatological calendar; leap-day values use Feb 28's bin.
    return dates.dayofyear.to_numpy()-1-(dates.is_leap_year & (dates.month>2)).astype(int)-(dates.is_leap_year & (dates.month==2)&(dates.day==29)).astype(int)

def anomaly(a,dates):
    a=np.asarray(a,dtype=float);doy=daynumber(dates);train=pd.DatetimeIndex(dates).year<=2015
    sums=np.zeros((365,a.shape[1]));counts=np.zeros_like(sums)
    for k in range(365):
        v=a[train&(doy==k)];sums[k]=np.nansum(v,axis=0);counts[k]=np.isfinite(v).sum(axis=0)
    smooth=uniform_filter1d(sums,size=31,axis=0,mode='wrap')/np.maximum(uniform_filter1d(counts,size=31,axis=0,mode='wrap'),1e-12)
    z=a-smooth[doy];std=np.nanstd(z[train],axis=0);std=np.where(std>1e-12,std,1)
    return z/std,{'training_end':'2015-12-31','calendar_bins':365,'smoothing_days':31}

def split(t):
    y=pd.DatetimeIndex(t).year.to_numpy();return np.where(y<=2015,'train',np.where(y<=2018,'validation','test'))

def bootstrap_mean_difference(per_group,repeats=2000,seed=SEED):
    v=np.asarray(per_group);rng=np.random.default_rng(seed)
    means=np.array([v[rng.integers(0,len(v),len(v))].mean() for _ in range(repeats)])
    return np.quantile(means,[.025,.975]).tolist()
