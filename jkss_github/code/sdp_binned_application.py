"""Independently tuned exact nonnegative PSD baseline for binned application."""
from pathlib import Path
import json,time,warnings,hashlib
import numpy as np
import pandas as pd
import cvxpy as cp
from threadpoolctl import threadpool_limits
from certified_binned import bin_design

ROOT=Path(__file__).resolve().parents[1];GRID=[0,1,2,3,4,5,7,9,11]


def estimate(y,r):
    H=len(y)
    if r==0:return np.full(H,1/H),dict(status='analytic',raw_min_eigenvalue=0.,repair_norm=0.,objective=0.)
    B=bin_design(r,H);Q=cp.Variable((r+1,r+1),hermitian=True)
    a=cp.hstack([cp.sum(cp.diag(Q,k=k)) for k in range(1,r+1)])
    z=cp.hstack([cp.real(a),cp.imag(a)]);q=1/H+B@z
    prob=cp.Problem(cp.Minimize(.5*cp.sum_squares(q-y)),[Q>>0,cp.trace(Q)==1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        prob.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=300)
    if prob.status not in ('optimal','optimal_inaccurate'):raise RuntimeError(f'SDP status {prob.status}')
    raw=(Q.value+Q.value.conj().T)/2;v,U=np.linalg.eigh(raw);clean=np.maximum(v,0)
    if clean.sum()<=0:raise RuntimeError('PSD trace repair failed.')
    fixed=(U*clean)@U.conj().T/clean.sum()
    repair=float(np.linalg.norm(fixed-raw));mineig=float(v.min())
    if repair>1e-6:raise RuntimeError(f'PSD repair too large: {repair}')
    coeff=np.array([np.diag(fixed,k=k).sum() for k in range(1,r+1)])
    zz=np.r_[coeff.real,coeff.imag];out=1/H+B@zz
    if out.min()<-1e-10 or abs(out.sum()-1)>1e-10:raise RuntimeError('Invalid reconstructed bin probabilities.')
    return out,dict(status=prob.status,raw_min_eigenvalue=mineig,repair_norm=repair,objective=float(.5*np.sum((out-y)**2)))


def main():
    folder=ROOT/'results'/'binned_sdp_baseline';folder.mkdir(exist_ok=False)
    df=pd.read_csv(ROOT/'data'/'processed'/'complete_days_hourly.csv',parse_dates=['dteday'])
    counts=df.pivot(index='dteday',columns='hr',values='cnt').sort_index()
    shares=counts.div(counts.sum(axis=1),axis=0);wd=df.groupby('dteday').workingday.first().reindex(shares.index)
    started=dict(protocol_sha256=hashlib.sha256((ROOT/'research'/'20_binned_sdp_protocol.md').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        candidates=GRID,stage='Post-hoc sensitivity baseline on same observed application.')
    (folder/'started.json').write_text(json.dumps(started,indent=2),encoding='utf-8')
    tuning=[];forecasts=[];daily=[];start_time=time.perf_counter();all_status=[]
    try:
        for start in pd.date_range('2012-01-01','2012-12-01',freq='MS'):
            end=start+pd.offsets.MonthBegin(1);cutoff=start-pd.Timedelta(days=35);window=start-pd.Timedelta(days=180)
            for workingday in [0,1]:
                subset=shares[wd==workingday];train=subset[(subset.index>=window)&(subset.index<cutoff)]
                val=subset[(subset.index>=cutoff)&(subset.index<start)];refit=subset[(subset.index>=window)&(subset.index<start)]
                test=subset[(subset.index>=start)&(subset.index<end)]
                if min(len(train),len(val),len(test))==0:continue
                y=train.mean().to_numpy();scores=[]
                for r in GRID:
                    q,diag=estimate(y,r);all_status.append(diag['status'])
                    score=float(np.mean(np.sum((val.to_numpy()-q)**2,axis=1)));scores.append(score)
                    tuning.append(dict(month=str(start.date()),workingday=workingday,method='sdp_bins',parameter=r,score=score,
                        train_max=str(train.index.max().date()),validation_max=str(val.index.max().date()),test_min=str(test.index.min().date()),**diag))
                r=GRID[int(np.argmin(scores))];q,diag=estimate(refit.mean().to_numpy(),r);all_status.append(diag['status'])
                for h,value in enumerate(q):forecasts.append(dict(month=str(start.date()),workingday=workingday,
                    method='sdp_bins',parameter=r,hour=h,probability=value))
                for date,row in test.iterrows():
                    obs=row.to_numpy();peak=abs(int(np.argmax(q))-int(np.argmax(obs)))
                    daily.append(dict(date=str(date.date()),month=str(start.date()),workingday=workingday,method='sdp_bins',parameter=r,
                        squared_share_error=float(np.sum((obs-q)**2)),log_score=float(-obs@np.log(np.maximum(q,1e-12))),
                        peak_hour_error=min(peak,24-peak),morning_share_error=abs(float(q[7:10].sum()-obs[7:10].sum())),
                        evening_share_error=abs(float(q[16:19].sum()-obs[16:19].sum())),**diag))
            print(f'SDP baseline completed {start.date()}',flush=True)
    except Exception as exc:
        pd.DataFrame(tuning).to_csv(folder/'tuning.csv',index=False);pd.DataFrame(daily).to_csv(folder/'daily_scores.csv',index=False)
        (folder/'failure.json').write_text(json.dumps(dict(error=repr(exc),month=str(start.date()),workingday=int(workingday),r=r),indent=2),encoding='utf-8')
        raise
    t=pd.DataFrame(tuning);d=pd.DataFrame(daily);f=pd.DataFrame(forecasts)
    t.to_csv(folder/'tuning.csv',index=False);d.to_csv(folder/'daily_scores.csv',index=False);f.to_csv(folder/'forecasts.csv',index=False)
    prev=pd.read_csv(ROOT/'results'/'binned_certificate_application'/'daily_scores.csv')
    old=pd.read_csv(ROOT/'results'/'application_pilot'/'daily_scores.csv')
    full=pd.concat([prev,old,d],ignore_index=True);pp=full.pivot(index=['date','month'],columns='method',values='squared_share_error')
    if pp.isna().any().any():raise RuntimeError('Unmatched test days.')
    full.groupby('method').agg(test_days=('date','count'),mean_squared_share_error=('squared_share_error','mean'),
        mean_log_score=('log_score','mean'),mean_peak_hour_error=('peak_hour_error','mean')).reset_index().to_csv(folder/'summary.csv',index=False)
    diff=pp.certified_bins-pp.sdp_bins
    diff.groupby(level='month').mean().to_csv(folder/'monthly_certified_minus_sdp.csv')
    meta=dict(started,completed=True,test_days=int(d.date.nunique()),tuning_fits=len(t),
        status_counts=pd.Series(all_status).value_counts().to_dict(),maximum_psd_repair=float(max(t.repair_norm.max(),d.repair_norm.max())),
        selected_r11=int((f.groupby(['month','workingday']).parameter.first()==11).sum()),
        candidate_minus_sdp=float(diff.mean()),seconds=time.perf_counter()-start_time)
    (folder/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
