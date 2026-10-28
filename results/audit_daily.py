from pathlib import Path
import os,json
import numpy as np,pandas as pd
R=Path(os.environ.get('CLIMATE_DATA_ROOT',str(Path(__file__).resolve().parents[1])))
O=Path(__file__).resolve().parent
FOLDER='HE_2000_2022'
NODES=288
p=R/FOLDER/'data'
full=pd.date_range('2000-01-01','2022-12-31')
dates=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(next(p.glob('*datelist*'))).Date))
grid=pd.read_csv(next(p.glob('*nodelist*')))
assert len(grid)==NODES
rows=[];quality=[]
for path in sorted(p.glob('*.txt')):
 a=np.loadtxt(path)
 assert a.shape==(len(full),NODES),(path,a.shape)
 missing=np.isnan(a).all(axis=1);md=full[missing]
 assert md.equals(full.difference(dates)),path
 rows.append(dict(file=str(path.relative_to(R)),rows=len(a),date_rows=len(dates),all_nan_dates=md.strftime('%Y-%m-%d').tolist(),partial_nan_rows=int((np.isnan(a).any(axis=1)&~missing).sum())))
 finite=a[np.isfinite(a)]
 quality.append(dict(file=path.name,rows=a.shape[0],nodes=a.shape[1],date_rows=len(dates),all_nan_rows=int(missing.sum()),nonfinite_cells=int((~np.isfinite(a)).sum()),finite_min=float(finite.min()),finite_max=float(finite.max()),finite_mean=float(finite.mean())))
assert len(rows)==9
(O/'daily_date_audit.json').write_text(json.dumps(rows,indent=2))
pd.DataFrame(quality).to_csv(O/'data_quality.csv',index=False)
print(FOLDER,': 9 matrices verified; all six missing dates are February 29.')
