"""Reconstruct the 480 published samples and compare the two certificate bounds."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from certified_experiment import NAMES,draw,truth
from circular import moments
from certified_projection import fit
from fair_baselines import RGRID

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/certificate_bound_diagnostics'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    prior_path=ROOT/'results/precision_tradeoff/trials.csv'
    prior=pd.read_csv(prior_path)
    prior=prior[prior.eta==0.1].set_index(['scenario','n','rep'],verify_integrity=True)
    assert len(prior)==480 and (prior.status=='ok').all()
    manifest={'seed':20261104,'cases':480,'stage':'post-hoc replay of observed samples',
      'protocol_sha256':sha(ROOT/'research/42_certificate_diagnostics_protocol.md'),
      'source_sha256':sha(Path(__file__)),'engine_sha256':sha(ROOT/'code/certified_projection.py'),
      'prior_sha256':sha(prior_path),'ties':'absolute difference <= 1e-12 times larger bound'}
    (OUT/'started.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    records=[];max_ise_diff=0.;max_bound_diff=0.
    try:
        for si,name in enumerate(NAMES):
            true=truth(name)
            for n in (100,500):
                for rep in range(40):
                    old=prior.loc[name,n,rep];r=int(old.r)
                    rng=np.random.default_rng(np.random.SeedSequence([20261104,si,n,rep]))
                    x=draw(name,n,rng);c=moments(x,max(RGRID))
                    a,info=fit(c,r,n,eta=0.1,max_ratio=256)
                    padded=np.pad(a,(0,len(true)-len(a)))
                    ise=float(np.sum(abs(padded[1:]-true[1:])**2)/np.pi)
                    d_ise=abs(ise-float(old.ise));d_bound=abs(info['l2_squared_bound']-float(old.l2_squared_bound))
                    max_ise_diff=max(max_ise_diff,d_ise);max_bound_diff=max(max_bound_diff,d_bound)
                    assert info['D']==int(old.D) and d_ise<1e-10 and d_bound<1e-10
                    row={'scenario':name,'n':n,'rep':rep,'r':r,'D':info['D'],
                      'ise':ise,'bound':info['l2_squared_bound'],'target':info['target'],
                      'minimum_weight':info['minimum_weight'],'selected':'r0','ratio':np.nan,
                      'direct_bound':np.nan,'ball_bound':np.nan}
                    if r:
                        d=info['direct_bound'];b=info['ball_bound'];tol=1e-12*max(abs(d),abs(b),1e-300)
                        winner='tie' if abs(d-b)<=tol else ('direct' if d<b else 'ball')
                        row.update(selected=winner,ratio=info['D']/r,direct_bound=d,ball_bound=b)
                        assert abs(info['l2_squared_bound']-min(d,b))<1e-15
                    records.append(row)
    except Exception as e:
        pd.DataFrame(records).to_csv(OUT/'partial.csv',index=False)
        (OUT/'failure.json').write_text(json.dumps({'exception':repr(e),'scenario':name,'n':n,'rep':rep},indent=2),encoding='utf-8')
        raise
    cases=pd.DataFrame(records);cases.to_csv(OUT/'cases.csv',index=False)
    rows=[]
    for (name,n),g in cases.groupby(['scenario','n'],sort=True):
        active=g[g.r>0];counts=active.selected.value_counts();N=len(active)
        rows.append({'scenario':name,'n':int(n),'cases':len(g),'r0_count':int((g.r==0).sum()),
          'active':N,'ratio_q1':active.ratio.quantile(.25),'ratio_median':active.ratio.median(),
          'ratio_q3':active.ratio.quantile(.75),'direct_wins':int(counts.get('direct',0)),
          'ball_wins':int(counts.get('ball',0)),'ties':int(counts.get('tie',0)),
          'direct_fraction':int(counts.get('direct',0))/N if N else np.nan,
          'ball_fraction':int(counts.get('ball',0))/N if N else np.nan,
          'tie_fraction':int(counts.get('tie',0))/N if N else np.nan})
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'summary.csv',index=False)
    manifest.update(completed=True,max_replayed_ise_difference=max_ise_diff,max_replayed_bound_difference=max_bound_diff,
      active_cases=int((cases.r>0).sum()),r0_cases=int((cases.r==0).sum()))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
