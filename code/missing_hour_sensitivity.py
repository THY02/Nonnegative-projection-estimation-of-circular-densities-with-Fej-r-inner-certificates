"""Sensitivity on observed hours of incomplete days; no zero imputation."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'results'/'missing_hour_sensitivity';out.mkdir(exist_ok=False)
    files=[ROOT/'results'/'application_pilot'/'forecasts.csv',
           ROOT/'results'/'binned_certificate_application'/'forecasts.csv',
           ROOT/'results'/'binned_sdp_baseline'/'forecasts.csv']
    f=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    assert not f.duplicated(['month','workingday','method','hour']).any()
    assert (f.groupby(['month','workingday','method']).size()==24).all()
    assert (f.groupby(['month','workingday','method']).probability.sum()-1).abs().max()<1e-10
    raw=pd.read_csv(ROOT/'data'/'raw'/'hour.csv',parse_dates=['dteday'])
    assert not raw.duplicated(['dteday','hr']).any() and (raw.cnt>=0).all()
    audit=raw.groupby('dteday').agg(hours=('hr','nunique'),total=('cnt','sum'),statuses=('workingday','nunique'))
    eligible=audit[(audit.index.year==2012)&(audit.hours<24)]
    assert (eligible.total>0).all() and (eligible.statuses==1).all()
    rows=[];coverage=[]
    for day,condition in eligible.iterrows():
        z=raw[raw.dteday==day].sort_values('hr');hours=z.hr.to_numpy(int);obs=z.cnt.to_numpy(float);obs/=obs.sum()
        month=str(day.to_period('M').start_time.date());status=int(z.workingday.iloc[0]);available=set(hours.tolist())
        missing=[i for i in range(24) if i not in available]
        coverage.append(dict(date=str(day.date()),month=month,workingday=status,observed_hours=len(hours),
            missing_hours=','.join(map(str,missing)),recorded_total=int(condition.total)))
        sub=f[(f.month==month)&(f.workingday==status)]
        if sub.method.nunique()!=8:raise RuntimeError(f'Missing predictions for {day}.')
        for method,g in sub.groupby('method'):
            q=g.sort_values('hour').probability.to_numpy(float)[hours];den=q.sum()
            if den<=0:raise RuntimeError(f'zero conditional mass for {day} {method}')
            q=q/den
            rows.append(dict(date=str(day.date()),month=month,workingday=status,method=method,
                observed_hours=len(hours),omitted_hours=24-len(hours),observed_mass=float(den),
                squared_conditional_share_error=float(np.sum((obs-q)**2)),
                conditional_log_score=float(-obs@np.log(np.maximum(q,1e-12)))))
    score=pd.DataFrame(rows);coverage=pd.DataFrame(coverage)
    if len(score)!=len(coverage)*8 or score.duplicated(['date','method']).any():raise RuntimeError('Unmatched predictions.')
    score.to_csv(out/'daily_conditional_scores.csv',index=False);coverage.to_csv(out/'day_coverage.csv',index=False)
    summary=[]
    for label,subset in [('observed_at_least_20',score[score.observed_hours>=20]),('all_incomplete',score)]:
        for method,g in subset.groupby('method'):
            summary.append(dict(scope=label,method=method,days=len(g),
                squared_error_mean=g.squared_conditional_share_error.mean(),log_score_mean=g.conditional_log_score.mean()))
    summary=pd.DataFrame(summary);summary.to_csv(out/'summary.csv',index=False)
    meta=dict(completed=True,stage='Post-hoc sensitivity; not independently registered.',
        protocol_sha256=hashlib.sha256((ROOT/'research'/'27_missing_hour_protocol.md').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        raw_sha256=hashlib.sha256((ROOT/'data'/'raw'/'hour.csv').read_bytes()).hexdigest(),
        prediction_hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        incomplete_days=len(coverage),at_least_20_hours=int((coverage.observed_hours>=20).sum()),
        days_with_single_hour=int((coverage.observed_hours==1).sum()),
        no_imputation=True,interpretable_target='Conditional share distribution across hours actually recorded, not complete-day distribution.')
    (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(summary.to_string(index=False));print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
