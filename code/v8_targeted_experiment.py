"""Post-hoc matched Fejer KDE and new-seed targeted simulations for v8."""
from pathlib import Path
import hashlib, json, time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from circular import moments, l2score, multiplier
from certified_experiment import NAMES, draw, truth, H, FEJER
from fair_baselines import RGRID, RATIOS, METHODS, SDPSolver, candidate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/v8_targeted'
NAMES_SELECTED = ['bimodal','narrow']
NEW_SEED = 2026102821
OLD_SEED = 20261104
REPS = 20
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    sources = ['research/55_v8_validation_protocol.md','code/v8_targeted_experiment.py',
               'code/fair_baselines.py','code/certified_projection.py',
               'code/certified_experiment.py','code/circular.py','code/simulation.py',
               'results/fair_baselines_20261104/trials.csv']
    meta = dict(seed=NEW_SEED, old_seed=OLD_SEED, new_reps=REPS, scenarios=NAMES_SELECTED,
                n_values=[100,500], rgrid=RGRID, kde_grid=FEJER, folds=5,
                source_sha256={p:sha(ROOT/p) for p in sources},
                scope='Post-hoc design; old Fejer fits are reconstructed on saved 40 samples, new batch has 20 per cell.')
    (OUT/'started.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    old=pd.read_csv(ROOT/'results/fair_baselines_20261104/trials.csv')
    old=old[old.scenario.isin(NAMES_SELECTED)].copy()
    old['batch']='old40';old['status']='ok';old=old.rename(columns={'r':'parameter'})
    rows=old.to_dict('records'); scores=[]; diagnostics=[]; samples={}; failures=[]
    solver=SDPSolver(); clock=time.perf_counter()
    for name in NAMES_SELECTED:
      si=NAMES.index(name); true=truth(name)
      for n in [100,500]:
       for batch,seed,reps in [('old40',OLD_SEED,40),('new20',NEW_SEED,REPS)]:
        for rep in range(reps):
         try:
          rng=np.random.default_rng(np.random.SeedSequence([seed,si,n,rep]))
          x=draw(name,n,rng); folds=np.array_split(rng.permutation(n),5)
          samples[f'{batch}_{name}_{n}_{rep}']=x
          samples[f'{batch}_{name}_{n}_{rep}_fold']=np.array(folds)
          max_degree=max(FEJER)
          cv=[(moments(x[np.setdiff1d(np.arange(n),ids)],max_degree),
               moments(x[ids],max_degree),len(ids),n-len(ids)) for ids in folds]
          full=moments(x,max_degree)
          methods=['fejer_kde'] if batch=='old40' else METHODS+['fejer_kde']
          for method in methods:
           choices=([(d,0) for d in FEJER] if method=='fejer_kde' else
                    [(r,ratio) for r in RGRID for ratio in ([0] if r==0 or method!='grid_repair_cv' else RATIOS)])
           values=[]
           for par,ratio in choices:
            value=0.
            for ct,valid,nv,nt in cv:
             if method=='fejer_kde': a=multiplier(par)*ct[:par+1]
             else: a,_=candidate(method,ct,par,ratio,nt,solver)
             value+=nv/n*l2score(a,valid)
            values.append(float(value))
            scores.append(dict(batch=batch,scenario=name,n=n,rep=rep,method=method,
                               parameter=par,ratio=ratio,score=value))
           j=int(np.argmin(values));par,ratio=choices[j]
           if method=='fejer_kde': a=multiplier(par)*full[:par+1]; diag={}
           else: a,diag=candidate(method,full,par,ratio,n,solver)
           if method=='certified' and (diag['l2_squared_bound']>diag['target']+1e-12 or diag['minimum_weight'] < -1e-10):
            raise ValueError('certificate diagnostic failed')
           if method=='grid_repair_cv' and diag['grid_slack'] < -1e-10: raise ValueError('grid diagnostic failed')
           ise=float(np.sum(np.abs(np.pad(a,(0,H+1-len(a)))[1:]-true[1:])**2)/np.pi)
           rows.append(dict(batch=batch,scenario=name,n=n,rep=rep,method=method,status='ok',
                            parameter=par,ratio=ratio,ise=ise,cv_score=values[j],
                            r_boundary=par==(max(FEJER) if method=='fejer_kde' else max(RGRID)),
                            ratio_boundary=method=='grid_repair_cv' and ratio==max(RATIOS)))
           diagnostics.append(dict(batch=batch,scenario=name,n=n,rep=rep,method=method,
                                   **{k:v for k,v in diag.items() if k!='history'}))
         except Exception as ex:
          failures.append(dict(batch=batch,scenario=name,n=n,rep=rep,error=repr(ex)))
          pd.DataFrame(rows).to_csv(OUT/'partial_trials.csv',index=False)
          (OUT/'failure.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
          raise
        print(f'{name} n={n} {batch} complete at {time.perf_counter()-clock:.1f}s',flush=True)
    df=pd.DataFrame(rows); pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
    df.to_csv(OUT/'trials.csv',index=False);pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics.csv',index=False)
    np.savez_compressed(OUT/'samples.npz',**samples)
    summary=[]
    for name in NAMES_SELECTED:
     for n in [100,500]:
      for batch in ['old40','new20','pooled60']:
       z=df[(df.scenario==name)&(df.n==n)]
       if batch!='pooled60':z=z[z.batch==batch]
       for method in METHODS+['fejer_kde']:
        v=z[z.method==method].set_index(['batch','rep']).sort_index()
        ref=z[z.method=='certified'].set_index(['batch','rep']).sort_index()
        assert len(v)==len(ref) and v.index.equals(ref.index)
        diff=v.ise-ref.ise
        summary.append(dict(scenario=name,n=n,batch=batch,method=method,cases=len(v),
          mise=v.ise.mean(),mcse=v.ise.std(ddof=1)/np.sqrt(len(v)),
          method_minus_certified=diff.mean(),paired_mcse=diff.std(ddof=1)/np.sqrt(len(diff)),
          upper_fraction=v.r_boundary.mean(),ratio_upper_fraction=v.ratio_boundary.mean()))
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    meta.update(completed=True, failures=0, seconds=time.perf_counter()-clock,
      sdp_calls=solver.calls,sdp_min_raw_eigenvalue=solver.min_eigenvalue,
      sdp_max_psd_repair=solver.max_psd_repair,sdp_max_solver_residual=solver.max_residual,
      rows=len(df), candidate_scores=len(scores))
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=1):main()
