import os
from pathlib import Path
import numpy as np,pandas as pd,json
O=Path(__file__).resolve().parent;R=Path(os.environ.get('CLIMATE_DATA_ROOT', str(Path(__file__).resolve().parents[1]))).expanduser().resolve();rng=np.random.default_rng(222)
a=np.concatenate([np.load(p,mmap_mode='r') for p in sorted(O.glob('VO850_*_3H_5deg_160n.npy'))]);times=pd.date_range('2000-01-01',periods=len(a),freq='3h')
grid=pd.read_csv(R/'NA_5N45N_250E350E_Hourly/NA_5N45N_250E350E_nodelist_5deg_160n.csv');y=grid.target_lat.to_numpy();x=grid.target_lon.to_numpy()
s=pd.read_csv(O/'movement_segments.csv');s['t']=pd.to_datetime(s.t);lookup=s.set_index(['SID','t']);rows=[]
def distance(y1,x1,y2,x2):
 y1,x1,y2,x2=map(np.deg2rad,[y1,x1,y2,x2]);return 6371*2*np.arcsin(np.sqrt(np.clip(np.sin((y2-y1)/2)**2+np.cos(y1)*np.cos(y2)*np.sin((x2-x1)/2)**2,0,1)))
def peak(t,lat,lon):
 j=times.get_indexer([t])[0]
 if j<0:return None
 ids=np.flatnonzero(distance(lat,lon,y,x)<=800)
 if not len(ids):return None
 n=ids[np.argmax(a[j,ids])]
 if a[j,n]<=0:return None
 return n
for z in s.itertuples():
 key=(z.SID,z.t+pd.Timedelta(hours=24))
 if key not in lookup.index:continue
 v=lookup.loc[key];n=peak(z.t,z.LAT,z.x);m=peak(key[1],v.LAT,v.x)
 if n is None or m is None:continue
 true_e=(v.x-z.x)*np.cos(np.deg2rad(z.LAT));true_n=v.LAT-z.LAT
 peak_e=(x[m]-x[n])*np.cos(np.deg2rad(z.LAT));peak_n=y[m]-y[n]
 rows.append(dict(SID=z.SID,t=z.t,center_error_km=distance(z.LAT,z.x,y[n],x[n]),true_e=true_e,true_n=true_n,peak_e=peak_e,peak_n=peak_n,true_displacement_km=distance(z.LAT,z.x,v.LAT,v.x),peak_displacement_km=distance(y[n],x[n],y[m],x[m])))
pd.DataFrame(rows).to_csv(O/'vortex_tracking_diagnostic.csv',index=False);q=pd.DataFrame(rows)
metrics={'description':'Retrospective, IBTrACS-guided strongest positive cell within 800 km at BOTH times; not independent tracking or a forecast','segments_24h':len(q),'storms':int(q.SID.nunique()),'median_localization_error_km':float(q.center_error_km.median()),'stationary_peak_fraction':float(((q.peak_e==0)&(q.peak_n==0)).mean()),'east_displacement_correlation':float(q.true_e.corr(q.peak_e)),'north_displacement_correlation':float(q.true_n.corr(q.peak_n))}
(O/'vortex_tracking_metrics.json').write_text(json.dumps(metrics,indent=2))
# Sensitivity to already-tropical first records, one sample per storm.
g=pd.read_csv(O/'genesis_matched_samples.csv');out=[]
for name,f in [('all',g),('prior_nontropical_record',g[~g.first_record_is_tropical])]:
 for h in [-72,-48,-24,0,24,48]:
  diff=(f[f'event_{h}h']-f[f'control_{h}h']).to_numpy();boot=np.mean(diff[rng.integers(0,len(diff),(2000,len(diff)))],axis=1)
  out.append(dict(subset=name,n=len(f),lead_hours=h,difference=diff.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
pd.DataFrame(out).to_csv(O/'genesis_sensitivity.csv',index=False)
# Storm-cluster bootstrap for movement summaries: do not treat all segments as independent.
ci=[]
for band,f in s.groupby('lat_band'):
 groups=[v[['speed_kmh','east','north']] for _,v in f.groupby('SID')];boot=[]
 for _ in range(1000):
  b=pd.concat([groups[i] for i in rng.integers(0,len(groups),len(groups))],ignore_index=True)
  boot.append([b.speed_kmh.median(),(b.east<0).mean(),(b.north>0).mean()])
 low,high=np.quantile(boot,[.025,.975],axis=0)
 ci.append(dict(lat_band=band,speed_ci_low=low[0],speed_ci_high=high[0],west_ci_low=low[1],west_ci_high=high[1],north_ci_low=low[2],north_ci_high=high[2]))
pd.DataFrame(ci).to_csv(O/'movement_cluster_ci.csv',index=False)
print(json.dumps(metrics,indent=2));print(pd.DataFrame(out).to_string(index=False));print(pd.DataFrame(ci).to_string(index=False))
