"""Rolling application of the weighted-bin certificate, alongside old baselines."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from certified_binned import fit

ROOT=Path(__file__).resolve().parents[1]
GRID=[0,1,2,3,4,5,7,9,11]


def main():
    out=ROOT/'results'/'binned_certificate_application';out.mkdir(exist_ok=False)
    src=ROOT/'data'/'processed'/'complete_days_hourly.csv'
    data=pd.read_csv(src,parse_dates=['dteday'])
    counts=data.pivot(index='dteday',columns='hr',values='cnt').sort_index()
    shares=counts.div(counts.sum(axis=1),axis=0)
    wd=data.groupby('dteday').workingday.first().reindex(shares.index)
    old=pd.read_csv(ROOT/'results'/'application_pilot'/'daily_scores.csv')
    methods={'certified_bins':'adaptive','fixed_2r_bins':'fixed_2r'}
    daily=[];tuning=[];forecasts=[];start_time=time.perf_counter()
    meta=dict(protocol_sha256=hashlib.sha256((ROOT/'research'/'18_binned_application_protocol.md').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        engine_sha256=hashlib.sha256((ROOT/'code'/'certified_binned.py').read_bytes()).hexdigest(),
        old_data_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),degree_candidates=GRID,error_tolerance=1e-6,
        target='Equal-day 24-hour share profiles, conditional on workingday, not exact trip times.',completed=False)
    (out/'started.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    try:
        for start in pd.date_range('2012-01-01','2012-12-01',freq='MS'):
            end=start+pd.offsets.MonthBegin(1);cutoff=start-pd.Timedelta(days=35);window=start-pd.Timedelta(days=180)
            for status in [0,1]:
                subset=shares[wd==status]
                train=subset[(subset.index>=window)&(subset.index<cutoff)]
                val=subset[(subset.index>=cutoff)&(subset.index<start)]
                refit=subset[(subset.index>=window)&(subset.index<start)]
                test=subset[(subset.index>=start)&(subset.index<end)]
                if min(len(train),len(val),len(test))==0:continue
                y=train.mean().to_numpy()
                for method,mode in methods.items():
                    scores=[]
                    for r in GRID:
                        q,a,d=fit(y,r,mode)
                        score=float(np.mean(np.sum((val.to_numpy()-q)**2,axis=1)))
                        scores.append(score)
                        tuning.append(dict(month=str(start.date()),workingday=status,method=method,parameter=r,
                            score=score,train_days=len(train),validation_days=len(val),
                            train_max=str(train.index.max().date()),validation_max=str(val.index.max().date()),
                            test_min=str(test.index.min().date()),**{k:v for k,v in d.items() if k!='r'}))
                    chosen=int(np.argmin(scores));r=GRID[chosen]
                    q,a,d=fit(refit.mean().to_numpy(),r,mode)
                    for h,value in enumerate(q):
                        forecasts.append(dict(month=str(start.date()),workingday=status,method=method,parameter=r,
                            D=d['D'],hour=h,probability=value))
                    for date,row in test.iterrows():
                        observed=row.to_numpy();peak=abs(int(np.argmax(q))-int(np.argmax(observed)))
                        daily.append(dict(date=str(date.date()),month=str(start.date()),workingday=status,method=method,
                            parameter=r,D=d['D'],squared_share_error=float(np.sum((observed-q)**2)),
                            log_score=float(-observed@np.log(np.maximum(q,1e-12))),
                            mass_below_log_floor=float(np.sum(q<1e-12)),peak_hour_error=min(peak,24-peak),
                            morning_share_error=abs(float(q[7:10].sum()-observed[7:10].sum())),
                            evening_share_error=abs(float(q[16:19].sum()-observed[16:19].sum())),
                            error_bound=d['error_bound'],min_weight=d['min_weight'],stationarity=d['stationarity']))
            print(f'binned application completed {start.date()}',flush=True)
    except Exception as exc:
        pd.DataFrame(tuning).to_csv(out/'tuning.csv',index=False)
        pd.DataFrame(daily).to_csv(out/'daily_scores.csv',index=False)
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),month=str(start.date()),status=int(status),method=method,r=r),indent=2),encoding='utf-8')
        raise
    new=pd.DataFrame(daily);t=pd.DataFrame(tuning);f=pd.DataFrame(forecasts)
    new.to_csv(out/'daily_scores.csv',index=False);t.to_csv(out/'tuning.csv',index=False);f.to_csv(out/'forecasts.csv',index=False)
    both=pd.concat([old,new],ignore_index=True)
    pivot=both.pivot(index=['date','month'],columns='method',values='squared_share_error')
    if pivot.isna().any().any():raise RuntimeError('Baseline and new methods have unmatched dates.')
    diffs=pivot.subtract(pivot.certified_bins,axis=0).groupby(level='month').mean()
    diffs.to_csv(out/'monthly_method_minus_certified.csv')
    summary=both.groupby('method').agg(test_days=('date','count'),mean_squared_share_error=('squared_share_error','mean'),
        mean_log_score=('log_score','mean'),mean_peak_hour_error=('peak_hour_error','mean'),
        mean_morning_share_error=('morning_share_error','mean'),mean_evening_share_error=('evening_share_error','mean')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    selected=t.loc[t.groupby(['month','workingday','method'],sort=False).score.idxmin()]
    forecast_degree=f.groupby(['month','workingday','method']).parameter.first()
    assert all(row.parameter==forecast_degree.loc[(row.month,row.workingday,row.method)] for row in selected.itertuples())
    assert (f.groupby(['month','workingday','method']).probability.sum()-1).abs().max()<1e-10
    assert f.probability.min()>=-1e-10
    cert=new[new.method=='certified_bins'];assert cert.error_bound.max()<=1e-6+1e-12
    assert cert.min_weight.min()>=-1e-10
    meta.update(completed=True,test_days=int(new.date.nunique()),new_final_forecast_profiles=len(f)//24,
        new_tuning_fits=len(t),min_weight=float(cert.min_weight.min()),max_selected_error_bound=float(cert.error_bound.max()),
        max_stationarity=float(new.stationarity.max()),seconds=time.perf_counter()-start_time,
        limitations=['Binned output error, not full continuous density error.',
        'Tuning epsilon is computational, no iid day-risk claim.',
        'Complete-day selection may be informative; no trip-level inference.',
        'Old baseline methods loaded from prior result, matched on all test dates.'])
    (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(json.dumps(meta,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
