"""Explicitly post-primary label sensitivity and matched-control diagnostics.
These do not replace or tune the prespecified held-out primary experiment.
"""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
O=Path(__file__).resolve().parent

def main():
    obs=pd.read_csv(O/'candidate_issues.csv.gz');obs['t']=pd.to_datetime(obs.t);obs['genesis_time']=pd.to_datetime(obs.genesis_time);obs['genesis_sid']=obs.genesis_sid.fillna('')
    cache=np.load(O/'feature_cache.npz');matched=pd.read_csv(O/'matched_genesis.csv');left=set(matched.loc[matched.left_censored,'track_id']);clean=~obs.track_id.isin(left).to_numpy()
    y=obs.label_72h.to_numpy();tr=obs.split.eq('train').to_numpy()&clean;va=obs.split.eq('validation').to_numpy()&clean;te=obs.split.eq('test').to_numpy()&clean
    output=[]
    for name in ['current_only','1h_sample','24h_sample','24h_mean']:
        x=cache[name];model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,max_iter=3000))
        model.fit(x[tr],y[tr]);pv=model.predict_proba(x[va])[:,1];pt=model.predict_proba(x[te])[:,1];threshold=np.quantile(pv[y[va]==0],.99,method='higher');alert=pt>threshold
        output.append(dict(model=name,horizon=72,test_samples=int(te.sum()),test_positive_samples=int(y[te].sum()),test_positive_storms=int(obs.loc[te&(y==1),'genesis_sid'].nunique()),average_precision=average_precision_score(y[te],pt),test_recall=float(alert[y[te]==1].mean()),test_false_positive_rate=float(alert[y[te]==0].mean())))
    pd.DataFrame(output).to_csv(O/'uncensored_sensitivity.csv',index=False)
    # One case per event, with an issue 24-72h beforehand closest to 48h.
    lead=(obs.genesis_time-obs.t).dt.total_seconds()/3600;raw=cache['current_only'];fine=cache['1h_sample'];doy=daynumber(obs.t);pairs=[];diffs=[];rng=np.random.default_rng(SEED)
    descriptors={'history72_std':14,'history72_max_minus_mean':None,'history72_trend_per_hour':15,'history72_net_change':16,'history72_high_fraction':17,'history72_threshold_persistence_hours':18}
    used=set()
    for part in ['train','validation','test']:
        pos=obs[obs.split.eq(part)&obs.label_72h.eq(1)&lead.between(24,72)].copy();pos['from48']=(lead[pos.index]-48).abs();pos=pos.sort_values(['from48','t']).groupby('genesis_sid',sort=False).head(1).sort_values('t')
        neg=np.flatnonzero(obs.split.eq(part).to_numpy()&obs.label_72h.eq(0).to_numpy()&obs.genesis_sid.eq('').to_numpy())
        for ci,case in pos.iterrows():
            dd=distance(case.lat,case.lon,obs.loc[neg,'lat'].to_numpy(),obs.loc[neg,'lon'].to_numpy());season=np.minimum(np.abs(doy[neg]-doy[ci]),365-np.abs(doy[neg]-doy[ci]));amplitude=np.abs(np.log(raw[neg,7]/raw[ci,7]));age=np.abs(np.log((obs.age_hours.to_numpy()[neg]+3)/(case.age_hours+3)))
            ok=(dd<=750)&(season<=30)&(amplitude<=np.log(2))&(age<=np.log(3))
            order=np.argsort(dd/750+season/30+amplitude/np.log(2)+age/np.log(3));selected=[];seen=set()
            for k in order:
                ni=int(neg[k]);tid=int(obs.track_id.iloc[ni])
                if ok[k] and tid not in used and tid not in seen and tid!=case.track_id:
                    selected.append(ni);seen.add(tid)
                if len(selected)==3:break
            if not selected:continue
            used.update(seen)
            for ni in selected:pairs.append(dict(split=part,SID=case.genesis_sid,case_index=ci,control_index=ni,case_track=int(case.track_id),control_track=int(obs.track_id.iloc[ni]),case_time=str(case.t),control_time=str(obs.t.iloc[ni]),case_lead_hours=float(lead[ci]),case_left_censored=int(case.track_id) in left,distance_km=float(distance(case.lat,case.lon,obs.lat.iloc[ni],obs.lon.iloc[ni])),season_day_difference=int(min(abs(doy[ci]-doy[ni]),365-abs(doy[ci]-doy[ni]))),current_vorticity_ratio=float(raw[ni,7]/raw[ci,7])))
            for name,col in descriptors.items():
                values=fine[:,13]-fine[:,11] if col is None else fine[:,col]
                diffs.append(dict(split=part,SID=case.genesis_sid,feature=name,difference=float(values[ci]-values[selected].mean()),controls=len(selected)))
    pd.DataFrame(pairs).to_csv(O/'matched_control_pairs.csv',index=False);f=pd.DataFrame(diffs);rows=[]
    for (part,name),g in f.groupby(['split','feature']):
        ci=bootstrap_mean_difference(g.difference.to_numpy(),1000);rows.append(dict(split=part,feature=name,case_storms=g.SID.nunique(),case_minus_controls=g.difference.mean(),ci_low=ci[0],ci_high=ci[1]))
    pd.DataFrame(rows).to_csv(O/'matched_control_diagnostics.csv',index=False)
    (O/'supplement_scope.json').write_text(json.dumps({'added_after_primary_results':True,'purpose':'label sensitivity and descriptive matched contrasts; no new forecast model tuning','matching':'same partition, <=750km, <=30 climatological days, amplitude ratio 0.5-2, (age+3h) ratio 1/3-3, up to 3 controls from distinct candidate tracks, controls not reused','controls':'candidate tracks never assigned a genesis event by this catalog; not independently certified physical nondevelopers','matching_features_use_issue_or_past_only':True,'cases_selected_using_future_labels_for_retrospective_diagnostics':True,'matched_interval_multiplicity':'pointwise 95% intervals, no family-wise adjustment'},indent=2))
    print(pd.DataFrame(output).to_string(index=False));print(pd.DataFrame(rows).query("split=='test'").to_string(index=False))
if __name__=='__main__':main()
