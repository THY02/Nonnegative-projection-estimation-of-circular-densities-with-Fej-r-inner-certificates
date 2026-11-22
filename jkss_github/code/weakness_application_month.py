"""Post-hoc paired month-level examination; no independence claims."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/weakness_application_month'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    names=['results/application_pilot/daily_scores.csv','results/binned_certificate_application/daily_scores.csv',
           'results/binned_sdp_baseline/daily_scores.csv']
    methods=['certified_bins','sdp_bins','circular_smooth','histogram','fixed_2r_bins']
    data=pd.concat([pd.read_csv(ROOT/name) for name in names],ignore_index=True)
    data=data[data.method.isin(methods)]
    assert len(data)==350*len(methods) and not data.duplicated(['date','method']).any()
    p=data.pivot(index=['date','month'],columns='method',values='squared_share_error')
    assert len(p)==350 and not p.isna().any().any() and p.index.get_level_values('month').nunique()==12
    rows=[]
    for (date,month),v in p.iterrows():
        for baseline in methods[1:]:
            rows.append({'date':date,'month':month,'baseline':baseline,
                         'certified_minus_baseline':v.certified_bins-v[baseline]})
    daily=pd.DataFrame(rows);daily.to_csv(OUT/'daily_differences.csv',index=False)
    monthly=daily.groupby(['baseline','month']).agg(days=('date','size'),mean_difference=('certified_minus_baseline','mean')).reset_index()
    assert len(monthly)==48 and (monthly.groupby('baseline').days.sum()==350).all()
    monthly.to_csv(OUT/'monthly_differences.csv',index=False)
    summary=[]
    for method,g in monthly.groupby('baseline'):
        direct=daily[daily.baseline==method].certified_minus_baseline.mean()
        weighted=np.average(g.mean_difference,weights=g.days)
        assert abs(direct-weighted)<1e-15
        summary.append({'baseline':method,'days':int(g.days.sum()),'months':len(g),
                        'daily_weighted_mean':direct,'negative_months':int((g.mean_difference<0).sum()),
                        'positive_months':int((g.mean_difference>0).sum()),'zero_months':int((g.mean_difference==0).sum()),
                        'minimum_monthly_mean':g.mean_difference.min(),'maximum_monthly_mean':g.mean_difference.max()})
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'summary.csv',index=False)
    paths=names+['research/weakness_application_month_protocol.md','code/weakness_application_month.py']
    meta={'completed':True,'days':len(p),'months':12,'baselines':methods[1:],
          'inputs_sha256':{name:sha(ROOT/name) for name in paths},
          'note':'Post-hoc descriptive paired data; no independent-month or causal inference.'}
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
