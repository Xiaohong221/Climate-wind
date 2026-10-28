"""Topic C: held-out test of remote HE information for TC displacement."""
from pathlib import Path
import sys,os,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent
PLAN=json.loads((O/'analysis_plan.json').read_text())

def log(s):print(s,flush=True)
def fitprep(x,mask):
    im=SimpleImputer(strategy='median',add_indicator=True);sc=StandardScaler()
    tr=im.fit_transform(x[mask]);sc.fit(tr)
    return sc.transform(im.transform(x))
def equal_storm(error,sids):return pd.DataFrame({'e':error,'SID':sids}).groupby('SID').e.mean().mean()
def errors(meta,disp,h):
    y,x=destination(meta.LAT.to_numpy(),meta.x.to_numpy(),disp)
    e=distance(y,x,meta[f'lat_{h}'].to_numpy(),meta[f'lon_{h}'].to_numpy())
    return e,y,x

def main():
    log('Loading inputs and building train-only climatologies')
    a,t3,grid=hourly(3);d=tracks();keep=d.t.between('2000-01-08','2022-12-31 23:00')&inside(d)&d.USA_STATUS.isin(['TD','TS','HU'])&d.t.dt.hour.mod(6).eq(0)&d.t.dt.minute.eq(0)
    q=d[keep].copy();lookup=d.set_index(['SID','t'])
    def pastfuture(hours,col):
        ix=pd.MultiIndex.from_arrays([q.SID,q.t+pd.Timedelta(hours=hours)],names=['SID','t'])
        return lookup[col].reindex(ix).to_numpy()
    for hh in [-12,-6,24,48]:
        q[f'lat_{hh}']=pastfuture(hh,'LAT');q[f'lon_{hh}']=pastfuture(hh,'x')
    # Purge storms spanning partition boundaries, and future targets crossing them.
    tp=d[d.t.between('2000-01-01','2022-12-31 23:59')][['SID','t']].copy();tp['split']=split(tp.t)
    crossing=set(tp.groupby('SID').split.nunique().loc[lambda x:x>1].index)
    q=q[~q.SID.isin(crossing)&q[['lat_-12','lon_-12','lat_-6','lon_-6']].notna().all(axis=1)].reset_index(drop=True)
    q['split']=split(q.t);p=t3.get_indexer(q.t);assert np.all(p>=0)
    day=FULL.get_indexer(q.t.dt.floor('D'));assert np.all(day>=7)
    assert np.all((FULL[day-1]+pd.Timedelta(days=1)).to_numpy()<=q.t.to_numpy())
    rr=np.clip(np.rint((q.LAT.to_numpy()-7.5)/5).astype(int),0,7);cc=np.clip(np.rint((q.x.to_numpy()-252.5)/5).astype(int),0,19);node=rr*20+cc
    q['node']=node
    velocity=-tangent(q.LAT,q.x,q['lat_-12'],q['lon_-12'])/12
    v6=-tangent(q.LAT,q.x,q['lat_-6'],q['lon_-6'])/6
    previous=tangent(q['lat_-12'],q['lon_-12'],q['lat_-6'],q['lon_-6'])/6
    season=2*np.pi*daynumber(q.t)/365;la=(q.LAT.to_numpy()-25)/20;lo=(q.x.to_numpy()-300)/50
    motion=np.column_stack([la,lo,la**2,lo**2,la*lo,np.sin(season),np.cos(season),q.USA_WIND.to_numpy(),velocity,v6,previous,v6-previous])
    names=['lat','lon','lat2','lon2','lat_lon','sin_season','cos_season','USA_WIND','ve12','vn12','ve6','vn6','ve_previous6','vn_previous6','ae','an']
    localparts=[motion];localnames=names.copy()
    for lag in [0,6,12,24]:
        vals=a[p-lag//3].reshape(-1,8,20);patch=np.stack([vals[np.arange(len(q)),np.clip(rr+dy,0,7),np.clip(cc+dx,0,19)] for dy in [-1,0,1] for dx in [-1,0,1]],axis=1)
        f=np.column_stack([a[p-lag//3,node],patch.mean(axis=1),patch.max(axis=1),patch.min(axis=1),patch.std(axis=1),(patch[:,5]-patch[:,3])/10,(patch[:,7]-patch[:,1])/10])
        localparts.append(f);localnames.extend([f'NA_{v}_{lag}h' for v in ['center','mean','max','min','std','gx','gy']])
    na=matrix(ROOT/'NA_5N45N_250E350E_Daily/data/Vorticity850_2000_2022_daily_mean_5deg_160n.txt')
    wind=matrix(ROOT/'NA_5N45N_250E350E_Daily/data/WindSpeed850_2000_2022_daily_mean_5deg_160n.txt')
    he=matrix(ROOT/'HE_2000_2022/data/Vorticity850_2000_2022_daily_mean_7p5deg_288n.txt')
    hg=pd.read_csv(ROOT/'HE_2000_2022/data/HE_nodelist_7p5deg_288n.csv')
    naz,_=anomaly(na,FULL);hez,_=anomaly(he,FULL);train_days=FULL.year<=2015
    npca=PCA(n_components=8,svd_solver='full').fit(np.nan_to_num(naz[train_days]))
    npc=npca.transform(np.nan_to_num(naz))
    for lag in PLAN['completed_day_lags']:
        localparts.append(npc[day-lag]);localnames.extend([f'NA_PC{k+1}_day{lag}' for k in range(8)])
        localparts.append(np.isnan(na[day-lag]).all(axis=1)[:,None].astype(float));localnames.append(f'NA_missing_day{lag}')
    localparts.append(wind[day-1,node,None]);localnames.append('NA_prior_completed_day_windspeed')
    local=np.column_stack(localparts)
    remote=~(hg.target_lat.between(-2.5,52.5)&hg.target_lon.between(242.5,357.5)).to_numpy()
    rg=hg[remote].copy().reset_index(drop=True);rz=hez[:,remote]
    hpca=PCA(n_components=8,svd_solver='full').fit(np.nan_to_num(rz[train_days]))
    far=distance(q.LAT.to_numpy()[:,None],q.x.to_numpy()[:,None],rg.target_lat.to_numpy()[None,:],rg.target_lon.to_numpy()[None,:])>=1500
    assert far.shape==(len(q),len(rg))
    def remote_features(k,yearmap=None):
        parts=[]
        for lag in PLAN['completed_day_lags']:
            idx=day-lag
            if yearmap is not None:
                dates=FULL[idx];mapped=[]
                for dt in dates:
                    yr=yearmap[dt.year]
                    try:mapped.append(dt.replace(year=yr))
                    except ValueError:mapped.append(dt.replace(year=yr,day=28))
                idx=FULL.get_indexer(pd.DatetimeIndex(mapped));assert np.all(idx>=0)
            z=rz[idx];masked=np.where(far,np.nan_to_num(z),0)
            parts.append((masked-hpca.mean_)@hpca.components_[:k].T)
            parts.append(np.isnan(z).all(axis=1)[:,None].astype(float))
        parts.append(far.mean(axis=1)[:,None])
        return np.column_stack(parts)
    remote_x={k:remote_features(k) for k in [4,8]}
    rg['remote_static']=True;rg.to_csv(O/'remote_nodes.csv',index=False)
    q.to_csv(O/'sample_metadata.csv',index=False)
    np.savez_compressed(O/'preprocessing.npz',HE_pca=hpca.components_,HE_mean=hpca.mean_,NA_pca=npca.components_,NA_mean=npca.mean_)
    pd.DataFrame({'feature':localnames}).to_csv(O/'local_feature_names.csv',index=False)
    metrics=[];predrows=[];searchrows=[];permutation_rows=[];gainrows=[];splitrows=[]
    rng=np.random.default_rng(SEED)
    for horizon in [24,48]:
        valid=q[[f'lat_{horizon}',f'lon_{horizon}']].notna().all(axis=1).to_numpy()
        valid &= split(q.t+pd.Timedelta(hours=horizon))==q.split.to_numpy()
        meta=q[valid].reset_index(drop=True);sp=meta.split.to_numpy();tr=sp=='train';va=sp=='validation';te=sp=='test'
        assert not(set(meta.SID[tr])&set(meta.SID[te]))
        for part in ['train','validation','test']:
            f=meta[sp==part];splitrows.append(dict(horizon=horizon,split=part,samples=len(f),storms=f.SID.nunique(),first=str(f.t.min()),last=str(f.t.max())))
        y=tangent(meta.LAT,meta.x,meta[f'lat_{horizon}'],meta[f'lon_{horizon}']);persist=velocity[valid]*horizon;target=y-persist
        weights=1/meta.SID[tr].map(meta.SID[tr].value_counts()).to_numpy();weights*=tr.sum()/weights.sum()
        def fitselect(raw,label):
            z=fitprep(raw[valid],tr);best=None
            for alpha in PLAN['ridge_alphas']:
                model=Ridge(alpha=alpha).fit(z[tr],target[tr],sample_weight=weights)
                pred=persist[va]+model.predict(z[va]);e,_,_=errors(meta[va],pred,horizon);score=equal_storm(e,meta.SID[va])
                if not label.startswith('null'):searchrows.append(dict(horizon=horizon,model=label,alpha=alpha,validation_error_km=score))
                if best is None or score<best['score']:best=dict(score=score,alpha=alpha,model=model,z=z)
            return best
        chosen={'persistence':None,'motion':fitselect(motion,'motion'),'local':fitselect(local,'local')}
        candidates=[(k,fitselect(np.column_stack([local,remote_x[k]]),f'remote_K{k}')) for k in [4,8]]
        bestk,best=min(candidates,key=lambda x:x[1]['score']);chosen['local_plus_remote']=best
        output={}
        for name,item in chosen.items():
            pred=persist[te] if item is None else persist[te]+item['model'].predict(item['z'][te])
            e,py,px=errors(meta[te],pred,horizon);output[name]=e
            metrics.append(dict(horizon=horizon,model=name,test_storm_mean_km=equal_storm(e,meta.SID[te]),test_sample_mean_km=float(e.mean()),test_median_km=float(np.median(e)),test_storms=meta.SID[te].nunique(),test_samples=te.sum(),selected_alpha=None if item is None else item['alpha'],remote_components=bestk if name=='local_plus_remote' else None,validation_storm_mean_km=None if item is None else item['score']))
            frame=meta.loc[te,['SID','t','LAT','x',f'lat_{horizon}',f'lon_{horizon}']].copy();frame['horizon']=horizon;frame['model']=name;frame['pred_lat']=py;frame['pred_lon']=px;frame['error_km']=e;predrows.append(frame)
        paired=pd.DataFrame({'SID':meta.SID[te],'gain_km':output['local']-output['local_plus_remote']}).groupby('SID').gain_km.mean()
        gain=float(paired.mean());ci=bootstrap_mean_difference(paired.to_numpy());gainrows.append(dict(horizon=horizon,gain_km=gain,ci_low=ci[0],ci_high=ci[1],relative_gain_pct=100*gain/equal_storm(output['local'],meta.SID[te]),test_storms=len(paired)))
        pd.DataFrame({'SID':paired.index,'gain_km':paired.values,'horizon':horizon}).to_csv(O/f'per_storm_gain_{horizon}h.csv',index=False)
        log(f'Horizon {horizon}: local {equal_storm(output["local"],meta.SID[te]):.2f} km, remote {equal_storm(output["local_plus_remote"],meta.SID[te]):.2f} km; gain {gain:.2f}, CI {ci}; remote K={bestk}')
        # Negative-control permutations retain calendar season and entire remote yearly evolution.
        for b in range(50):
            mapping={}
            for years in [np.arange(2000,2016),np.arange(2016,2019),np.arange(2019,2023)]:
                perm=rng.permutation(years)
                # Identity permutations in small partitions are allowed as part of the exact null.
                mapping.update(dict(zip(years.tolist(),perm.tolist())))
            nullbest=None
            for k in [4,8]:
                candidate=fitselect(np.column_stack([local,remote_features(k,mapping)]),f'null_{b}_{k}')
                if nullbest is None or candidate['score']<nullbest['score']:nullbest=candidate
            pred=persist[te]+nullbest['model'].predict(nullbest['z'][te]);e,_,_=errors(meta[te],pred,horizon)
            permutation_rows.append(dict(horizon=horizon,iteration=b,null_gain_km=equal_storm(output['local'],meta.SID[te])-equal_storm(e,meta.SID[te])))
            if (b+1)%10==0:log(f'{horizon}h negative control {b+1}/50')
        null=np.array([r['null_gain_km'] for r in permutation_rows if r['horizon']==horizon]);gainrows[-1]['permutation_p']=(1+int((null>=gain).sum()))/(1+len(null))
    pd.DataFrame(metrics).to_csv(O/'metrics.csv',index=False);pd.DataFrame(gainrows).to_csv(O/'incremental_gain.csv',index=False)
    pd.DataFrame(searchrows).to_csv(O/'validation_search.csv',index=False);pd.DataFrame(splitrows).to_csv(O/'split_counts.csv',index=False)
    pd.concat(predrows,ignore_index=True).to_csv(O/'test_predictions.csv',index=False);pd.DataFrame(permutation_rows).to_csv(O/'negative_controls.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for h in [24,48]:
        f=pd.DataFrame(metrics);f=f[f.horizon==h];ax[0].plot(f.model,f.test_storm_mean_km,'o-',label=f'{h}h')
    ax[0].set(ylabel='Storm-mean position error (km)',title='Independent test: 2019-2022');ax[0].tick_params(axis='x',rotation=25);ax[0].legend()
    gg=pd.DataFrame(gainrows);ax[1].errorbar(gg.horizon,gg.gain_km,yerr=np.vstack([gg.gain_km-gg.ci_low,gg.ci_high-gg.gain_km]),fmt='o',capsize=5);ax[1].axhline(0,color='gray',ls='--');ax[1].set(xlabel='Lead time (hours)',ylabel='Local error minus remote-added error (km)',title='Paired storm bootstrap 95% CI')
    fig.tight_layout();fig.savefig(O/'topic_c_results.png',dpi=180);plt.close(fig)
    (O/'checks.json').write_text(json.dumps({'PCA_and_climatology_training_end':'2015-12-31','test_used_for_selection':False,'HE_days_end_before_issue':True,'raw_HE_nodes':288,'remote_HE_nodes':len(rg),'remote_minimum_distance_km':1500,'purged_cross_partition_storms':sorted(crossing),'same_test_rows_for_all_models':True,'primary_horizon_hours':24},indent=2))
    log('Topic C complete')
if __name__=='__main__':main()
