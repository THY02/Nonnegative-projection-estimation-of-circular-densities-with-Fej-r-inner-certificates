"""New-seed replication for CV fold-specific envelope penalties."""
from pathlib import Path
import hashlib, json, time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from circular import moments,l2score
from certified_experiment import NAMES,draw,truth
from certified_projection import fit
from fair_baselines import RGRID

R=Path(__file__).resolve().parents[1];OUT=R/'results/v7_envelope_expansion'
SEED=2026102807;K=5;N=257;ETA=.1;REPS=100
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    meta={'seed':SEED,'scenarios':NAMES,'n_values':[100,500],'reps_per_cell':REPS,
      'K':K,'rgrid':RGRID,'eta':ETA,'max_ratio':256,'grid_N':N,
      'stage':'new random samples, post-hoc design after old results; not blinded preregistration',
      'hashes':{p:sha(R/p) for p in ['research/52_v7_expansion_protocol.md',
       'code/v7_envelope_expansion.py','code/certified_projection.py','code/certified_experiment.py',
       'code/circular.py','code/fair_baselines.py','code/simulation.py']}}
    (OUT/'started.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    grids=np.exp(1j*np.outer(np.arange(1,max(RGRID)+1),2*np.pi*np.arange(N)/N))
    slopes=np.arange(1,max(RGRID)+1)**2
    rows=[];candidate_rows=[];diag_rows=[];failure_rows=[];saved=[];keys=[];clock=time.perf_counter()
    for si,name in enumerate(NAMES):
      true=truth(name)
      for n in [100,500]:
        assert n%K==0
        t=np.log(K*len(RGRID)**2*n**2);m=n/K
        for rep in range(REPS):
          r=-1;k=-1;stage='generation'
          try:
            rng=np.random.default_rng(np.random.SeedSequence([SEED,si,n,rep]));x=draw(name,n,rng)
            folds=np.array_split(rng.permutation(n),K)
            val=[moments(x[idx],max(RGRID)) for idx in folds]
            train=[moments(x[np.setdiff1d(np.arange(n),idx)],max(RGRID)) for idx in folds]
            coeff=np.zeros((len(RGRID),K,max(RGRID)+1),dtype=complex)
            scores=np.zeros(len(RGRID));caps=np.empty((len(RGRID),K))
            for j,r in enumerate(RGRID):
              for k in range(K):
                stage='fold_fit';a,d=fit(train[k],r,n-len(folds[k]),eta=ETA,max_ratio=256)
                coeff[j,k,:len(a)]=a;scores[j]+=l2score(a,val[k])/K
                q=1+2*np.real(coeff[j,k,1:]@grids)
                beta=(np.pi/N)**2*np.sum(slopes*np.abs(coeff[j,k,1:]))
                bound=(np.max(q)+beta+1e-12)/(2*np.pi)
                caps[j,k]=min((r+1)/(2*np.pi),bound)
                if not np.isfinite(caps[j,k]) or caps[j,k]<=0:raise ValueError('Invalid envelope')
                diag_rows.append({'scenario':name,'n':n,'rep':rep,'r':r,'fold':k,'D':d['D'],
                    'bound':d['l2_squared_bound'],'target':d['target'],'minimum_weight':d['minimum_weight'],
                    'envelope':caps[j,k]})
            fixed=np.array([2*t*(r+1)/(2*np.pi*3*m) for r in RGRID]);local=2*t*caps.mean(axis=1)/(3*m)
            if np.any(local>fixed+1e-12):raise ValueError('Local bound exceeds uniform bound')
            for j,r in enumerate(RGRID):
              candidate_rows.append({'scenario':name,'n':n,'rep':rep,'r':r,'cv_score':scores[j],
                'fixed_penalty':fixed[j],'local_penalty':local[j],
                'envelope_mean':caps[j].mean(),'envelope_max':caps[j].max()})
            for method,p in [('ordinary',np.zeros(len(RGRID))),('fixed',fixed),('local',local)]:
              j=int(np.argmin(scores+p));avg=coeff[j].mean(axis=0)
              padded=np.pad(avg,(0,len(true)-len(avg)))
              ise=float(np.sum(np.abs(padded[1:]-true[1:])**2)/np.pi)
              rows.append({'scenario':name,'n':n,'rep':rep,'method':method,'status':'ok','r':RGRID[j],
                'ise':ise,'cv_score':scores[j],'penalty':p[j],'criterion':scores[j]+p[j]})
            keys.append([si,n,rep]);saved.append(coeff)
          except Exception as exc:
            failure_rows.append({'scenario':name,'n':n,'rep':rep,'stage':stage,'r':r,'fold':k,'error':repr(exc)})
            for method in ['ordinary','fixed','local']:
              rows.append({'scenario':name,'n':n,'rep':rep,'method':method,'status':'failed'})
        print(f'finished {name} n={n} in {time.perf_counter()-clock:.1f}s',flush=True)
    trials=pd.DataFrame(rows);candidates=pd.DataFrame(candidate_rows);diag=pd.DataFrame(diag_rows)
    trials.to_csv(OUT/'trials.csv',index=False);candidates.to_csv(OUT/'candidates.csv',index=False)
    diag.to_csv(OUT/'fold_diagnostics.csv',index=False)
    pd.DataFrame(failure_rows,columns=['scenario','n','rep','stage','r','fold','error']).to_csv(OUT/'failures.csv',index=False)
    np.savez_compressed(OUT/'fold_coefficients.npz',coefficients=np.stack(saved) if saved else np.empty((0,len(RGRID),K,max(RGRID)+1),complex),keys=np.array(keys).reshape(-1,3))
    summary=[]
    for name in NAMES:
      for n in [100,500]:
        z=trials[(trials.scenario==name)&(trials.n==n)]
        for method in ['ordinary','fixed','local']:
          g=z[(z.method==method)&(z.status=='ok')].set_index('rep').sort_index()
          ordinary=z[(z.method=='ordinary')&(z.status=='ok')].set_index('rep').sort_index()
          reference=z[(z.method=='fixed')&(z.status=='ok')].set_index('rep').sort_index()
          row={'scenario':name,'n':n,'method':method,'attempted':REPS,'success':len(g),'failures':REPS-len(g)}
          if len(g):
            row.update(mise=g.ise.mean(),mcse=g.ise.std(ddof=1)/np.sqrt(len(g)),
                r0_fraction=(g.r==0).mean(),rmax_fraction=(g.r==max(RGRID)).mean())
            for label,ref in [('ordinary',ordinary),('fixed',reference)]:
              delta=g.ise-ref.loc[g.index].ise
              row[label+'_delta']=delta.mean();row[label+'_paired_mcse']=delta.std(ddof=1)/np.sqrt(len(delta))
          summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    meta.update(completed=True,attempted=REPS*len(NAMES)*2,successful_cases=len(saved),failed_cases=len(failure_rows),
        trials=len(trials),candidate_rows=len(candidates),fold_fits=len(diag),seconds=time.perf_counter()-clock,
        limitation='100 samples per cell; design followed inspection of prior results; floating-point envelope is not a formal interval certificate.')
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
