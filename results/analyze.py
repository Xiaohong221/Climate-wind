from pathlib import Path
import json, os
os.environ.setdefault('MPLCONFIGDIR', str(Path('/tmp')/'climate-mpl-cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path('/tmp')/'climate-xdg-cache'))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(os.environ.get('CLIMATE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))).expanduser().resolve(); O=Path(__file__).resolve().parent; rng=np.random.default_rng(221)
# Original data are read only; generated caches and outputs live in results/.
def load(p):
 cache=O/(p.stem+'.npy')
 if not cache.exists(): np.save(cache,np.loadtxt(p).astype('float32'))
 return np.load(cache,mmap_mode='r')
H=R/'NA_5N45N_250E350E_Hourly'
times=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(H/'NA_5N45N_250E350E_datelist_2000_2022_3H.csv').Date))
a=np.concatenate([load(p) for p in sorted((H/'3H').glob('*.txt'))])
assert a.shape==(len(times),160)
grid=pd.read_csv(H/'NA_5N45N_250E350E_nodelist_5deg_160n.csv')
lat=grid.target_lat.to_numpy(); lon=grid.target_lon.to_numpy()
d=pd.read_csv(R/'data_hurricanes/ibtracs.NA.list.v04r01.csv',skiprows=[1],low_memory=False,keep_default_na=False)
d=d[d.TRACK_TYPE.eq('main')].copy();d['t']=pd.to_datetime(d.ISO_TIME,format="mixed")
for c in ['LAT','LON','USA_WIND']:d[c]=pd.to_numeric(d[c],errors='coerce')
d['x']=d.LON%360;d=d.sort_values(['SID','t'])
alltracks=d.copy(); d=d[d.t.between('2000-01-01','2022-12-31 23:59:59')].copy()
def inarea(f):return f.LAT.between(5,45)&f.x.between(250,350)
def node(y,x):return int(np.argmin((lat-y)**2+((lon-x)*np.cos(np.deg2rad(y)))**2))
def dist(y,x,y2,x2):
 y,x,y2,x2=map(np.deg2rad,[y,x,y2,x2]);return 6371*2*np.arcsin(np.sqrt(np.clip(np.sin((y2-y)/2)**2+np.cos(y)*np.cos(y2)*np.sin((x2-x)/2)**2,0,1)))
# Displacements from actual 6-hourly positions, excluding extra interpolated 3h observations.
s=d[d.t.dt.hour.mod(6).eq(0)&d.t.dt.minute.eq(0)].copy()
g=s.groupby('SID');s['dt']=(g.t.shift(-1)-s.t).dt.total_seconds()/3600
s['lat2']=g.LAT.shift(-1);s['lon2']=g.x.shift(-1)
s['speed_kmh']=dist(s.LAT,s.x,s.lat2,s.lon2)/s.dt
s['north']=s.lat2-s.LAT;s['east']=(s.lon2-s.x+180)%360-180
s=s[inarea(s)&s.BASIN.eq('NA')&s.dt.eq(6)&s.USA_STATUS.isin(['TD','TS','HU'])].copy()
s['lat_band']=pd.cut(s.LAT,[5,15,25,35,45.00001],right=False,labels=['5–15','15–25','25–35','35–45'])
s['stage']=np.where(s.USA_STATUS.eq('HU'),'HU','TD/TS')
s.to_csv(O/'movement_segments.csv',index=False)
summary=s.groupby('lat_band',observed=True).agg(segments=('SID','size'),storms=('SID','nunique'),median_speed_kmh=('speed_kmh','median'),westward_fraction=('east',lambda v:(v<0).mean()),northward_fraction=('north',lambda v:(v>0).mean()))
summary.to_csv(O/'movement_summary.csv')
# First recorded TD/TS/HU per storm, using full tracks before restricting years/area.
gen=alltracks[alltracks.USA_STATUS.isin(['TD','TS','HU'])].groupby('SID',sort=False).head(1)
gen=gen[gen.t.between(times.min(),times.max())&inarea(gen)&gen.BASIN.eq('NA')].copy()
# Exclude left-censored cases already tropical at very first record? Keep flag visible.
first=alltracks.groupby('SID').t.min();gen['first_record_is_tropical']=gen.apply(lambda z:z.t==first[z.SID],axis=1)
# Contamination mask: exclude control dates if ANY recorded center within 10 degrees
# (distance <=1112 km) of genesis location within +-24 hours.
tracks=d.dropna(subset=['LAT','x']); tt=tracks.t.to_numpy(); yy=tracks.LAT.to_numpy();xx=tracks.x.to_numpy()
leads=np.array([-72,-48,-24,0,24,48]); rows=[]; controls=[]
for _,z in gen.iterrows():
 n=node(z.LAT,z.x); i=times.get_indexer([z.t])[0]
 if i<24 or i+16>=len(times):continue
 candidate=[]
 for days in [-28,-21,-14,14,21,28]:
  ct=z.t+pd.Timedelta(days=days);j=times.get_indexer([ct])[0]
  if j<24 or j+16>=len(times):continue
  near=np.abs(tt-np.datetime64(ct))<=np.timedelta64(24,'h')
  if np.any(dist(z.LAT,z.x,yy[near],xx[near])<=1112):continue
  candidate.append(j)
 if len(candidate)<2:continue
 event=a[i+leads//3,n].astype(float)
 control=np.nanmean(np.array([a[j+leads//3,n] for j in candidate]),axis=0)
 if not np.isfinite(event).all() or not np.isfinite(control).all():continue
 rows.append(dict(SID=z.SID,time=str(z.t),lat=z.LAT,lon=z.x,node=n,controls=len(candidate),first_record_is_tropical=bool(z.first_record_is_tropical),**{f'event_{h}h':v for h,v in zip(leads,event)},**{f'control_{h}h':v for h,v in zip(leads,control)}))
res=pd.DataFrame(rows);res.to_csv(O/'genesis_matched_samples.csv',index=False)
event=res[[f'event_{h}h' for h in leads]].to_numpy();control=res[[f'control_{h}h' for h in leads]].to_numpy();delta=event-control
boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean(axis=0) for _ in range(2000)])
comp=pd.DataFrame(dict(lead_hours=leads,n_storms=len(res),event_mean=event.mean(axis=0),control_mean=control.mean(axis=0),difference=delta.mean(axis=0),ci_low=np.quantile(boot,.025,axis=0),ci_high=np.quantile(boot,.975,axis=0)))
comp.to_csv(O/'genesis_composite.csv',index=False)
# 2020 hourly versus 3-hourly sampling; do not presume different products are identical.
h=load(H/'1H/VO850_2020_2020_1H_5deg_160n.txt'); t1=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(H/'NA_5N45N_250E350E_datelist_2000_2022_1H.csv').Date));t1=t1[t1.year==2020]
assert len(h)==len(t1)
b=a[times.year==2020]; sampled=h[::3];good=np.isfinite(b)&np.isfinite(sampled)
quality={'3h_shape':list(a.shape),'3h_nonfinite':int((~np.isfinite(a)).sum()),'2020_1h_shape':list(h.shape),'1h_3h_cotimed_mae':float(np.mean(np.abs(b[good]-sampled[good]))),'1h_3h_cotimed_correlation':float(np.corrcoef(b[good],sampled[good])[0,1]),'movement_storms':int(s.SID.nunique()),'movement_segments':len(s),'genesis_candidates':len(gen),'genesis_matched':len(res),'genesis_left_censored':int(res.first_record_is_tropical.sum())}
(O/'metrics.json').write_text(json.dumps(quality,indent=2))
fig,ax=plt.subplots(1,3,figsize=(17,4.5))
for sid,f in s.groupby('SID'):ax[0].plot(f.x-360,f.LAT,lw=.4,alpha=.2,color='steelblue')
ax[0].set(xlabel='Longitude',ylabel='Latitude',title='North Atlantic tropical tracks, 2000-2022',xlim=(-110,-10),ylim=(5,45))
summary.median_speed_kmh.plot.bar(ax=ax[1],color='steelblue');ax[1].set(title='Median 6-hour translation speed',ylabel='km/h',xlabel='Latitude band');ax[1].tick_params(axis='x',rotation=25)
ax[2].plot(leads,event.mean(axis=0)*1e5,'o-',label='First TD/TS/HU events');ax[2].plot(leads,control.mean(axis=0)*1e5,'o-',label='Season-matched controls');ax[2].set(xlabel='Hours from first tropical classification',ylabel='Vorticity file value × 1e5',title=f'Fixed genesis grid cell (n={len(res)})');ax[2].legend(fontsize=8)
fig.tight_layout();fig.savefig(O/'overview.png',dpi=170);plt.close(fig)
print(json.dumps(quality,indent=2));print(summary.to_string());print(comp.to_string(index=False))
