"""Post-hoc expand r<=24 on the same narrow samples/folds; no truth tuning."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
import fair_baselines as fb
from circular import moments,l2score
from certified_experiment import NAMES,draw,truth,H
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
RGRID=[0,1,2,3,4,6,8,12,16,24]


def main():
    out=ROOT/'results'/'narrow_fair_sensitivity';out.mkdir(exist_ok=False)
    original=pd.read_csv(ROOT/'results'/'fair_baselines_20261104'/'tuning.csv')
    original=original[(original.scenario=='narrow')&(original.method.isin(['certified','sdp_cv']))]
    meta=dict(seed=fb.SEED,repetitions=40,scenario='narrow',rgrid=RGRID,
        protocol_sha256=hashlib.sha256((ROOT/'research'/'16b_narrow_sensitivity_protocol.md').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Post-hoc sensitivity using the same samples/folds, not an independent experiment.')
    (out/'started.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    fb.RGRID=RGRID;sdp=fb.SDPSolver();rows=[];tuning=[];start=time.perf_counter()
    try:
        si=NAMES.index('narrow');true=truth('narrow')
        for n in [100,500]:
            for rep in range(40):
                rng=np.random.default_rng(np.random.SeedSequence([fb.SEED,si,n,rep]))
                x=draw('narrow',n,rng);folds=np.array_split(rng.permutation(n),5)
                cv=[]
                for ids in folds:
                    train=np.setdiff1d(np.arange(n),ids)
                    cv.append((moments(x[train],24),moments(x[ids],24),len(ids),len(train)))
                full=moments(x,24)
                for method in ['certified','sdp_cv']:
                    old=original[(original.n==n)&(original.rep==rep)&(original.method==method)]
                    if len(old)!=8 or old.r.duplicated().any():raise RuntimeError('Missing original CV results')
                    scores={int(row.r):float(row.cv_score) for row in old.itertuples()}
                    for r in [16,24]:
                        score=0.
                        for train,val,nv,nt in cv:
                            a,diagnostics=fb.candidate(method,train,r,0,nt,sdp)
                            score+=nv/n*l2score(a,val)
                        scores[r]=float(score)
                    r=min(RGRID,key=lambda x:scores[x]);a,diagnostics=fb.candidate(method,full,r,0,n,sdp)
                    padded=np.pad(a,(0,H+1-len(a)));ise=float(np.sum(abs(padded[1:]-true[1:])**2)/np.pi)
                    rows.append(dict(scenario='narrow',n=n,rep=rep,method=method,r=r,ise=ise,
                        cv_score=scores[r],r_upper_boundary=r==24,**{k:v for k,v in diagnostics.items() if k!='history' and k!='r'}))
                    tuning.extend(dict(scenario='narrow',n=n,rep=rep,method=method,r=r,cv_score=scores[r]) for r in RGRID)
            pd.DataFrame(rows).to_csv(out/'trials.csv',index=False);pd.DataFrame(tuning).to_csv(out/'tuning.csv',index=False)
            print(f'narrow n={n} extended r completed {time.perf_counter()-start:.1f}s',flush=True)
    except Exception as exc:
        pd.DataFrame(rows).to_csv(out/'trials.csv',index=False);pd.DataFrame(tuning).to_csv(out/'tuning.csv',index=False)
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),n=n,rep=rep,method=method,r=r),indent=2),encoding='utf-8')
        raise
    df=pd.DataFrame(rows);summary=[]
    for n,g in df.groupby('n'):
        ref=g[g.method=='certified'].set_index('rep').ise
        for method,z in g.groupby('method'):
            values=z.set_index('rep').ise;delta=ref-values
            summary.append(dict(n=n,method=method,mise=values.mean(),mcse=values.std(ddof=1)/np.sqrt(len(values)),
                certified_minus_method=delta.mean(),paired_mcse=delta.std(ddof=1)/np.sqrt(len(delta)),
                r_boundary_fraction=z.r_upper_boundary.mean()))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    meta.update(completed=True,final_fits=len(rows),candidate_scores=len(tuning),failures=0,
        sdp_calls=sdp.calls,sdp_analytic=sdp.analytic,sdp_feasible_shortcut=sdp.cert_shortcut,
        sdp_status=sdp.status,sdp_max_psd_repair=sdp.max_psd_repair,sdp_max_solver_residual=sdp.max_residual,
        seconds=time.perf_counter()-start)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
