"""Topic A: online candidate catalog and controlled temporal-information benchmark."""
from pathlib import Path
import sys,json,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent
PLAN=json.loads((O/'analysis_plan.json').read_text())
def log(s):print(s,flush=True)

def build_catalog(a,grid):
    cache=O/'catalog_all.npz'
    if cache.exists():
        c=np.load(cache);return pd.DataFrame({k:c[k] for k in c.files})
    fields=a.reshape(-1,8,20)
    peaks=(fields==maximum_filter(fields,size=(1,3,3),mode='constant',cval=-np.inf))&(fields>=1e-5)
    ys=grid.target_lat.to_numpy();xs=grid.target_lon.to_numpy();ids=[];steps=[];nodes=[];ages=[];lastnodes=None;lastids=None;birth={};serial=0
    for step in range(len(a)):
        now=np.flatnonzero(peaks[step].reshape(-1));assigned=np.full(len(now),-1,dtype=int)
        if lastnodes is not None and len(now) and len(lastnodes):
            dist=distance(ys[lastnodes,None],xs[lastnodes,None],ys[now][None,:],xs[now][None,:])
            amplitude=.15*np.abs(np.log(np.maximum(a[step-1,lastnodes],1e-12)[:,None]/np.maximum(a[step,now],1e-12)[None,:]))
            cost=dist/800+amplitude;cost[dist>800]=1e6
            ri,ci=linear_sum_assignment(cost)
            for r,c in zip(ri,ci):
                if dist[r,c]<=800:assigned[c]=lastids[r]
        for i in range(len(now)):
            if assigned[i]<0:assigned[i]=serial;birth[serial]=step;serial+=1
        ids.extend(assigned.tolist());steps.extend([step]*len(now));nodes.extend(now.tolist());ages.extend([3*(step-birth[x]) for x in assigned]);lastnodes=now;lastids=assigned
        if step%16000==0:log(f'Online catalog step {step}/{len(a)}; candidate tracks {serial}')
    c=pd.DataFrame({'track_id':np.array(ids,dtype=np.int32),'step':np.array(steps,dtype=np.int32),'node':np.array(nodes,dtype=np.int16),'age_hours':np.array(ages,dtype=np.int16)})
    np.savez_compressed(cache,**{k:c[k].to_numpy() for k in c});return c

def main():
    log('Loading hourly and 3-hourly vorticity')
    a3,t3,grid=hourly(3);a1,t1,_=hourly(1)
    diff=np.abs(a1[::3]-a3);assert diff.max()<1e-10
    rawdaily=matrix(ROOT/'NA_5N45N_250E350E_Daily/data/Vorticity850_2000_2022_daily_mean_5deg_160n.txt')
    derived=a1.reshape(len(FULL),24,160).mean(axis=1,dtype=np.float64);valid=np.isfinite(rawdaily)
    (O/'cross_resolution_checks.json').write_text(json.dumps({'1h_vs_3h_max_absolute_difference':float(diff.max()),'daily_vs_hourly_derived_mean_absolute_difference':float(np.abs(rawdaily[valid]-derived[valid]).mean()),'daily_vs_hourly_derived_max_absolute_difference':float(np.abs(rawdaily[valid]-derived[valid]).max()),'original_daily_all_nan_dates':FULL[np.isnan(rawdaily).all(axis=1)].strftime('%Y-%m-%d').tolist(),'hourly_contains_finite_leap_days':bool(np.isfinite(derived).all())},indent=2));del diff
    c=build_catalog(a3,grid);d=tracks();g=genesis(d)
    log(f'Catalog: {len(c)} observations, {c.track_id.nunique()} tracks; genesis events {len(g)}')
    bystep={int(k):v.index.to_numpy() for k,v in c.groupby('step',sort=False)}
    ys=grid.target_lat.to_numpy();xs=grid.target_lon.to_numpy();mapping={};matches=[]
    for step,events in g.assign(step=t3.get_indexer(g.t)).groupby('step'):
        if step<0 or step not in bystep:continue
        available=c.loc[bystep[step]];available=available[~available.track_id.isin(mapping)]
        if len(available)==0:continue
        dist=distance(events.LAT.to_numpy()[:,None],events.x.to_numpy()[:,None],ys[available.node.to_numpy()][None,:],xs[available.node.to_numpy()][None,:])
        cost=np.where(dist<=800,dist,1e9);ri,ci=linear_sum_assignment(cost)
        for r,k in zip(ri,ci):
            if dist[r,k]>800:continue
            ev=events.iloc[r];ca=available.iloc[k];tid=int(ca.track_id);mapping[tid]={'SID':ev.SID,'t':ev.t,'left_censored':bool(ev.left_censored)}
            matches.append(dict(SID=ev.SID,genesis_time=str(ev.t),track_id=tid,distance_km=dist[r,k],candidate_age_at_genesis=int(ca.age_hours),left_censored=bool(ev.left_censored)))
    pd.DataFrame(matches).to_csv(O/'matched_genesis.csv',index=False)
    # Common issue times: 00Z; complete history ending at issue, no future observations.
    obs=c[(c.step%8==0)&(c.age_hours>=12)&(c.step>=24)&(c.step<=(len(t3)-25))].copy();obs['t']=t3[obs.step.to_numpy()];obs['split']=split(obs.t)
    obs['lat']=ys[obs.node.to_numpy()];obs['lon']=xs[obs.node.to_numpy()]
    bounds=c.groupby('track_id').step.agg(['min','max']);spmin=split(t3[bounds['min'].to_numpy()]);spmax=split(t3[bounds['max'].to_numpy()]);cross=set(bounds.index[spmin!=spmax]);obs=obs[~obs.track_id.isin(cross)].copy()
    obs=obs[split(obs.t+pd.Timedelta(hours=72))==obs.split.to_numpy()].copy()
    obs['genesis_sid']=obs.track_id.map({k:v['SID'] for k,v in mapping.items()}).fillna('')
    obs['genesis_time']=pd.to_datetime(obs.track_id.map({k:v['t'] for k,v in mapping.items()}))
    obs=obs[obs.genesis_time.isna()|(obs.genesis_time>obs.t)].copy()
    # Remove centers of already-classified tropical/subtropical cyclones using current records only.
    active=d[d.USA_STATUS.isin(['TD','TS','HU','SD','SS'])].groupby('t')
    existing={k:v[['LAT','x']].dropna().to_numpy() for k,v in active}
    exclude=np.zeros(len(obs),dtype=bool)
    obs=obs.reset_index(drop=True)
    for t,ix in obs.groupby('t').groups.items():
        if t not in existing:continue
        pts=existing[t];ix=np.asarray(ix)
        near=distance(obs.loc[ix,'lat'].to_numpy()[:,None],obs.loc[ix,'lon'].to_numpy()[:,None],pts[:,0][None,:],pts[:,1][None,:])
        exclude[ix]=(near<=800).any(axis=1)
    excluded_existing=int(exclude.sum());obs=obs[~exclude].reset_index(drop=True)
    sp=obs.split.to_numpy();tr=sp=='train';va=sp=='validation';te=sp=='test'
    assert not(set(obs.track_id[tr])&set(obs.track_id[te]))
    idx=obs.step.to_numpy()*3;node=obs.node.to_numpy();assert np.array_equal(t1[idx].to_numpy(),obs.t.to_numpy());assert idx.min()>=72
    current=a1[idx,node];fields=a3[obs.step.to_numpy()].reshape(-1,8,20);rr=node//20;cc=node%20
    patch=np.stack([fields[np.arange(len(obs)),np.clip(rr+dy,0,7),np.clip(cc+dx,0,19)] for dy in [-1,0,1] for dx in [-1,0,1]],axis=1)
    season=2*np.pi*daynumber(obs.t)/365;lat=(obs.lat.to_numpy()-25)/20;lon=(obs.lon.to_numpy()-300)/50
    baseline=np.column_stack([lat,lon,lat**2,lon**2,np.sin(season),np.cos(season),np.log1p(obs.age_hours),current,patch.mean(axis=1),patch.max(axis=1),patch.std(axis=1)])
    # Eulerian histories at the candidate's issue-time cell; not the future genesis location.
    hist=a1[idx[:,None]-np.arange(72,0,-1)[None,:],node[:,None]]
    variants={'current_only':baseline};vinfo={'current_only':(0,'baseline')}
    for res in [1,3,6,12,24]:
        seq=a1[idx[:,None]-np.arange(72,-1,-res)[None,:],node[:,None]];tt=-np.arange(72,-1,-res,dtype=float)
        for representation,series,tv in [('sample',seq,tt)]+([] if res==1 else [('mean',hist.reshape(len(obs),72//res,res).mean(axis=2),np.arange(-72,0,res)+(res-1)/2)]):
            centered=tv-tv.mean();slope=(series@centered)/(centered@centered)
            high=series>=1e-5
            persistence=np.zeros(len(obs))
            for j in range(series.shape[1]-1,-1,-1):
                alive=high[:,j]&(persistence==(series.shape[1]-1-j)*res);persistence[alive]+=res
            summary=np.column_stack([series.mean(axis=1),series.min(axis=1),series.max(axis=1),series.std(axis=1),slope,series[:,-1]-series[:,0],high.mean(axis=1),persistence])
            key=f'{res}h_{representation}';variants[key]=np.column_stack([baseline,summary]);vinfo[key]=(res,representation)
    lead=(obs.genesis_time-obs.t).dt.total_seconds()/3600
    for h in [24,48,72]:obs[f'label_{h}h']=((lead>0)&(lead<=h)).astype(int)
    obs.to_csv(O/'candidate_issues.csv.gz',index=False,compression='gzip')
    splitcounts=[]
    for part in ['train','validation','test']:
        f=obs[sp==part];splitcounts.append(dict(split=part,samples=len(f),candidate_tracks=f.track_id.nunique(),positive_24h=int(f.label_24h.sum()),positive_48h=int(f.label_48h.sum()),positive_72h=int(f.label_72h.sum()),matched_event_storms=f.loc[f.label_72h.eq(1),'genesis_sid'].nunique()))
    pd.DataFrame(splitcounts).to_csv(O/'split_counts.csv',index=False)
    log('Split counts: '+str(splitcounts))
    np.savez_compressed(O/'feature_cache.npz',**{key:value.astype('float32') for key,value in variants.items()})
    if '--prepare-only' in sys.argv:
        log('Feature cache prepared; original metrics retained');return
    metrics=[];predictions=[];coefs=[];testprobs={};test_y=None
    all_test_events=g[g.t.dt.year.between(2019,2022)].SID.nunique()
    for horizon in [24,48,72]:
        y=obs[f'label_{horizon}h'].to_numpy();yt=y[te]
        assert y[tr].sum()>0 and yt.sum()>0
        for name,x in variants.items():
            model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,max_iter=3000,solver='lbfgs'))
            model.fit(x[tr],y[tr]);pv=model.predict_proba(x[va])[:,1];pt=model.predict_proba(x[te])[:,1]
            threshold=float(np.quantile(pv[y[va]==0],.99,method='higher'));alert=pt>threshold
            tp=int(((yt==1)&alert).sum());fp=int(((yt==0)&alert).sum());pos=int(yt.sum());neg=int((yt==0).sum())
            tests=obs[te].copy();detected=tests.loc[alert&(yt==1),'genesis_sid'].unique();lead_detected=[]
            for sid in detected:
                mask=alert&(yt==1)&tests.genesis_sid.eq(sid).to_numpy();lead_detected.append(float(lead[te].to_numpy()[mask].max()))
            metrics.append(dict(horizon=horizon,model=name,resolution_hours=vinfo[name][0],representation=vinfo[name][1],test_samples=len(yt),test_positive_samples=pos,test_base_rate=float(yt.mean()),average_precision=float(average_precision_score(yt,pt)),roc_auc=float(roc_auc_score(yt,pt)),brier=float(brier_score_loss(yt,pt)),threshold=threshold,test_recall=tp/pos,test_precision=tp/max(1,tp+fp),test_false_positive_rate=fp/neg,true_positive_issues=tp,false_positive_issues=fp,detected_storms=len(detected),all_test_genesis_storms=all_test_events,end_to_end_event_recall=len(detected)/all_test_events,median_first_alert_lead_hours=float(np.median(lead_detected)) if lead_detected else np.nan))
            f=tests[['track_id','t','node','genesis_sid','genesis_time']].copy();f['horizon']=horizon;f['model']=name;f['label']=yt;f['probability']=pt;f['alert']=alert;predictions.append(f)
            for j,w in enumerate(model.named_steps['logisticregression'].coef_[0]):coefs.append(dict(horizon=horizon,model=name,feature_index=j,standardized_coefficient=w))
            if horizon==72:testprobs[name]=pt;test_y=yt
        log(f'Completed {horizon}h models')
    pd.DataFrame(metrics).to_csv(O/'metrics.csv',index=False);pd.concat(predictions).to_csv(O/'test_predictions.csv.gz',index=False,compression='gzip');pd.DataFrame(coefs).to_csv(O/'coefficients.csv',index=False)
    # Cluster-paired uncertainty: each resample retains all issue times from a disturbance.
    tests=obs[te];groups=[np.flatnonzero(tests.track_id.to_numpy()==k) for k in tests.track_id.unique()];rng=np.random.default_rng(SEED);boot=[]
    for b in range(500):
        sample=np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))]);yy=test_y[sample]
        if yy.sum()==0:continue
        fine=average_precision_score(yy,testprobs['1h_sample'][sample])
        boot.append([fine-average_precision_score(yy,testprobs[n][sample]) for n in ['24h_sample','24h_mean']])
    boot=np.array(boot);intervals=[]
    for j,name in enumerate(['24h_sample','24h_mean']):
        delta=average_precision_score(test_y,testprobs['1h_sample'])-average_precision_score(test_y,testprobs[name]);ci=np.quantile(boot[:,j],[.025,.975]);intervals.append(dict(horizon=72,comparison='1h_sample minus '+name,AP_difference=delta,ci_low=ci[0],ci_high=ci[1],bootstrap_unit='candidate track',bootstrap_repeats=len(boot)))
    pd.DataFrame(intervals).to_csv(O/'paired_bootstrap.csv',index=False)
    fig,ax=plt.subplots(1,2,figsize=(12,4.5));met=pd.DataFrame(metrics)
    for h in [24,48,72]:
        f=met[(met.horizon==h)&met.representation.eq('sample')].sort_values('resolution_hours');ax[0].plot(f.resolution_hours,f.average_precision,'o-',label=f'{h}h horizon')
    ax[0].set(xlabel='Sampling interval (hours)',ylabel='Average precision',title='Held-out 2019-2022, common 3h catalog');ax[0].legend()
    for name in ['current_only','1h_sample','3h_sample','24h_sample','24h_mean']:
        pp,rr,_=precision_recall_curve(test_y,testprobs[name]);ax[1].plot(rr,pp,label=name)
    ax[1].axhline(test_y.mean(),ls='--',color='gray');ax[1].set(xlabel='Recall',ylabel='Precision',title='72h genesis prediction');ax[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(O/'topic_a_results.png',dpi=180);plt.close(fig)
    summary={'all_catalog_observations':len(c),'all_candidate_tracks':int(c.track_id.nunique()),'study_genesis_events':len(g),'matched_to_catalog_at_genesis':len(matches),'matched_left_censored_events':sum(r['left_censored'] for r in matches),'all_test_genesis_events':int(all_test_events),'test_events_with_eligible_positive_issue':int(tests.loc[test_y==1,'genesis_sid'].nunique()),'existing_TC_candidate_issues_removed':excluded_existing,'cross_partition_tracks_removed':len(cross),'catalog_uses_future':False,'features_use_future':False,'event_labels_use_future_as_ground_truth':True,'daily_model_uses_common_3h_candidate_catalog':True,'feature_history_uses_issue_cell_not_future_genesis_cell':True,'threshold_chosen_on_validation_only':True,'predictor_count_baseline':baseline.shape[1],'predictor_count_temporal_models':variants['1h_sample'].shape[1]}
    (O/'catalog_summary.json').write_text(json.dumps(summary,indent=2));log(json.dumps(summary));log('Topic A complete')
if __name__=='__main__':main()
