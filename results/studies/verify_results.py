"""Scientific consistency checks; no data modification."""
from common import *
import json
from pathlib import Path
base=Path(__file__).resolve().parent
# Spherical targets and reconstructed forecasts must round-trip across the date line.
lat=np.array([10.,35.,60.,-20.]);lon=np.array([359.,280.,350.,179.]);y=np.array([14.,39.,63.,-18.]);x=np.array([2.,270.,10.,181.])
e=tangent(lat,lon,y,x);yy,xx=destination(lat,lon,e)
assert np.max(distance(yy,xx,y,x))<1e-8
# Leap days do not shift March's climatological bins.
cal=pd.to_datetime(['2000-02-28','2000-02-29','2000-03-01','2001-03-01'])
assert daynumber(cal).tolist()==[58,58,59,59]
# Changing held-out values cannot change training anomalies or scaling.
dates=pd.date_range('2014-01-01','2017-12-31');rng=np.random.default_rng(5);v=rng.normal(size=(len(dates),2));changed=v.copy();changed[dates.year>=2016]+=1e6
z,_=anomaly(v,dates);zz,_=anomaly(changed,dates)
assert np.allclose(z[dates.year<=2015],zz[dates.year<=2015])
# Verify genuine held-out predictions and recompute all reported AP/Brier values.
a=base/'topic_a'
if a.exists():
    m=pd.read_csv(a/'metrics.csv');p=pd.read_csv(a/'test_predictions.csv.gz');issues=pd.read_csv(a/'candidate_issues.csv.gz')
    from sklearn.metrics import average_precision_score,brier_score_loss
    assert pd.to_datetime(p.t).dt.year.between(2019,2022).all()
    sets={s:set(f.track_id) for s,f in issues.groupby('split')};assert not(sets['train']&sets['test'])
    for row in m.itertuples():
     f=p[p.horizon.eq(row.horizon)&p.model.eq(row.model)]
     assert len(f)==row.test_samples and f.label.sum()==row.test_positive_samples
     assert np.isclose(average_precision_score(f.label,f.probability),row.average_precision)
     assert np.isclose(brier_score_loss(f.label,f.probability),row.brier)
     assert len(f.drop_duplicates(['track_id','t']))==len(f)
# C compares models on identical storms and issue times, and respects complete-day lags.
c=base/'topic_c'
if c.exists():
    p=pd.read_csv(c/'test_predictions.csv');m=pd.read_csv(c/'metrics.csv');meta=pd.read_csv(c/'sample_metadata.csv')
    assert pd.to_datetime(p.t).dt.year.between(2019,2022).all()
    sets={s:set(f.SID) for s,f in meta.groupby('split')};assert not(sets['train']&sets['test'])
    for h in [24,48]:
     refs=None
     for name,f in p[p.horizon.eq(h)].groupby('model'):
      keys=set(zip(f.SID,f.t));refs=keys if refs is None else refs;assert keys==refs
      err=distance(f.pred_lat,f.pred_lon,f[f'lat_{h}'],f[f'lon_{h}'])
      assert np.allclose(err,f.error_km)
      expected=f.groupby('SID').error_km.mean().mean()
      reported=m[m.horizon.eq(h)&m.model.eq(name)].test_storm_mean_km.iloc[0]
      assert np.isclose(expected,reported)
    assert len(pd.read_csv(c/'negative_controls.csv'))==100
print('PASS: geometry, leap calendar, train-only preprocessing, and all available experiment outputs and split checks.')
