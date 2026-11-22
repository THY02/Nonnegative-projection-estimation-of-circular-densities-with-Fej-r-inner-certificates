"""New-sample, paired sensitivity for r and grid-repair mesh size."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import fair_baselines as fb
from circular import moments, l2score
from certified_experiment import NAMES, draw, truth, H

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'weakness_independent_narrow'
RG=[0,1,2,3,4,6,8,12,16,24]
RR=[2,4,8,16,32]
SEED=2026102719
REPS=20
METHODS=['certified','grid_repair_cv','sdp_cv']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def diagnostics(d):
    return {key:d.get(key,np.nan) for key in ['D','delta','stationarity','grid_slack','l2_squared_bound','target','minimum_weight','status','psd_repair','min_eigenvalue','residual']}

def main():
    OUT.mkdir(exist_ok=False)
    paths=['research/weakness_independent_narrow_protocol.md','code/weakness_independent_narrow.py',
           'code/fair_baselines.py','code/certified_projection.py','code/certified_experiment.py',
           'code/circular.py','code/separated_degree.py']
    meta={'seed':SEED,'scenario':'narrow','repetitions':REPS,'sample_sizes':[100,500],'folds':5,
          'rgrid':RG,'repair_ratios':RR,'eta':.1,'inputs_sha256':{p:sha(ROOT/p) for p in paths},
          'stage':'New random sample internal replication; design chosen after prior results.'}
    (OUT/'started.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    fb.RGRID=RG
    sdp=fb.SDPSolver(); true=truth('narrow');si=NAMES.index('narrow')
    rows=[];cvrows=[];folds_rows=[];fullrows=[]; start=time.perf_counter()
    try:
        for n in [100,500]:
            for rep in range(REPS):
                rng=np.random.default_rng(np.random.SeedSequence([SEED,si,n,rep]))
                x=draw('narrow',n,rng);ids=np.array_split(rng.permutation(n),5)
                cv=[]
                for val in ids:
                    tr=np.setdiff1d(np.arange(n),val)
                    cv.append((moments(x[tr],24),moments(x[val],24),len(val),len(tr)))
                full=moments(x,24)
                for method in METHODS:
                    choices=[(r,ratio) for r in RG for ratio in ([0] if r==0 or method!='grid_repair_cv' else RR)]
                    scores={}
                    for r,ratio in choices:
                        score=0.
                        for fold,(train,val,nv,nt) in enumerate(cv):
                            a,d=fb.candidate(method,train,r,ratio,nt,sdp)
                            score+=nv/n*l2score(a,val)
                            if method=='certified':
                                assert d['l2_squared_bound']<=d['target']+1e-12 and d['minimum_weight']>=-1e-10
                            if method=='grid_repair_cv':
                                assert d['grid_slack']>=-1e-10 and d['delta']>=0
                            folds_rows.append({'n':n,'rep':rep,'method':method,'r':r,'ratio':ratio,'fold':fold,**diagnostics(d)})
                        scores[r,ratio]=float(score)
                        cvrows.append({'n':n,'rep':rep,'method':method,'r':r,'ratio':ratio,'score':score})
                    cache={}
                    for design in ['restricted','expanded']:
                        selectable=[(r,ratio) for r,ratio in choices if design=='expanded' or (r<=12 and ratio<=16)]
                        r,ratio=min(selectable,key=lambda pair:scores[pair])
                        if (r,ratio) not in cache:cache[(r,ratio)]=fb.candidate(method,full,r,ratio,n,sdp)
                        a,d=cache[(r,ratio)]
                        if method=='certified':
                            assert d['l2_squared_bound']<=d['target']+1e-12 and d['minimum_weight']>=-1e-10
                        if method=='grid_repair_cv':
                            assert d['grid_slack']>=-1e-10 and d['delta']>=0
                        padded=np.pad(a,(0,H+1-len(a)))
                        ise=float(np.sum(abs(padded[1:]-true[1:])**2)/np.pi)
                        rows.append({'n':n,'rep':rep,'method':method,'design':design,'r':r,'ratio':ratio,
                            'ise':ise,'score':scores[r,ratio],'r_boundary':r==(12 if design=='restricted' else 24),
                            'ratio_boundary':bool(method=='grid_repair_cv' and ratio==(16 if design=='restricted' else 32))})
                        fullrows.append({'n':n,'rep':rep,'method':method,'design':design,'r':r,'ratio':ratio,**diagnostics(d)})
                if rep%5==4:print(f'n={n}, samples={rep+1}/{REPS}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
    except Exception as exc:
        for name,data in [('partial_trials.csv',rows),('partial_tuning.csv',cvrows),('partial_fold_diagnostics.csv',folds_rows),('partial_final_diagnostics.csv',fullrows)]:
            pd.DataFrame(data).to_csv(OUT/name,index=False)
        (OUT/'failure.json').write_text(json.dumps({'error':repr(exc),'n':n,'rep':rep,'method':method,
            'r':r,'ratio':ratio,'fold':fold},indent=2),encoding='utf-8')
        raise
    df=pd.DataFrame(rows);ts=pd.DataFrame(cvrows);fd=pd.DataFrame(folds_rows);f=pd.DataFrame(fullrows)
    for name,data in [('trials.csv',df),('tuning.csv',ts),('fold_diagnostics.csv',fd),('final_diagnostics.csv',f)]:data.to_csv(OUT/name,index=False)
    summary=[]
    for (n,design),group in df.groupby(['n','design'],sort=True):
        ref=group[group.method=='certified'].set_index('rep').ise
        for method,z in group.groupby('method',sort=True):
            val=z.set_index('rep').ise;delta=ref-val
            summary.append({'n':int(n),'design':design,'method':method,'cases':len(z),'mise':val.mean(),
                'mcse':val.std(ddof=1)/np.sqrt(len(z)),'certified_minus_method':delta.mean(),
                'paired_mcse':delta.std(ddof=1)/np.sqrt(len(z)),'r_upper_fraction':z.r_boundary.mean(),
                'ratio_upper_fraction':z.ratio_boundary.mean()})
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    meta.update(completed=True,sample_cases=2*REPS,final_fits=len(rows),unique_candidate_scores=len(cvrows),
        fold_fits=len(folds_rows),failures=0,seconds=time.perf_counter()-start,sdp_calls=sdp.calls,
        sdp_analytic=sdp.analytic,sdp_feasible_shortcuts=sdp.cert_shortcut,sdp_status=sdp.status,
        sdp_max_psd_repair=sdp.max_psd_repair,sdp_max_residual=sdp.max_residual)
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
