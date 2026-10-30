"""Post-hoc independent-CV comparison of Pfister--Bresler sample certificate.

The positivity condition is Corollary 2 of Pfister--Bresler (2019), with the
simple upper bound C=(1-2r/N)^(-1/2). The optimization and checks use binary64
arithmetic and are not interval-certified.
"""
from pathlib import Path
from functools import lru_cache
import hashlib,json,time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from circular import moments,l2score
from certified_experiment import NAMES,draw,truth
from fair_baselines import RGRID

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/pfister_cv_comparison_v2'
SEED=20261104; RATIOS=[2,4,8,16]; REPS=8
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

@lru_cache(None)
def matrices(r,ratio):
    D=r*ratio; N=2*D+1; grid=2*np.pi*np.arange(N)/N
    e=np.exp(1j*np.outer(grid,np.arange(1,r+1)))
    B=2*np.c_[e.real,-e.imag]
    C=(1-2*r/N)**(-.5); t=(C-1)/(C+1)
    # q - t*a >= 0 and a - q >= 0, q=1+Bz
    A=np.vstack([np.c_[B,-t*np.ones(N)],np.c_[-B,np.ones(N)]])
    b=np.r_[np.ones(N),-np.ones(N)]
    return D,N,B,C,t,A,b

def estimate(c,r,ratio):
    if r==0:
        return np.array([1+0j]),{'D':0,'N':1,'slack':1.,'iterations':0,'repair':1.,'status':'analytic'}
    D,N,B,C,t,A,b=matrices(r,ratio)
    target=np.r_[c[1:r+1].real,c[1:r+1].imag]
    qt=1+B@target
    if qt.min()-t*qt.max()>=1e-10:
        z=target.copy(); iterations=0; status='target_feasible'
    else:
        w0=np.r_[np.zeros(2*r),(1+1/t)/2] if t else np.r_[np.zeros(2*r),2.]
        objective=lambda w:.5*np.sum((w[:-1]-target)**2)
        gradient=lambda w:np.r_[w[:-1]-target,0.]
        res=minimize(objective,w0,jac=gradient,method='SLSQP',
            constraints=[{'type':'ineq','fun':lambda w:A@w+b,'jac':lambda w:A}],
            options={'ftol':1e-11,'maxiter':800})
        if not res.success:raise RuntimeError(f'SLSQP {res.message}; r={r}, ratio={ratio}')
        if (A@res.x+b).min()<-1e-8:raise RuntimeError(f'infeasible optimizer {r} {ratio}')
        z=res.x[:-1]; iterations=int(res.nit); status='optimal_tolerance'
    # conservative numerical repair toward uniform; finite precision only.
    q=1+B@z; d=(q.min()-1)-t*(q.max()-1); factor=1.
    if (1-t)+d<1e-10:
        factor=min(1.,(1-t-1e-10)/(-d)); z=z*factor
    q=1+B@z; slack=float(q.min()-t*q.max())
    if slack<-1e-12:raise RuntimeError(f'post-repair slack {slack}')
    a=np.r_[1+0j,z[:r]+1j*z[r:]]
    return a,{'D':D,'N':N,'C':float(C),'slack':slack,'iterations':iterations,'repair':float(factor),'status':status}

def main():
    OUT.mkdir(exist_ok=False)
    meta={'stage':'Post-hoc independent CV on 96 previously observed samples','seed':SEED,'replications_per_cell':REPS,
        'folds':5,'rgrid':RGRID,'ratios':RATIOS,
        'protocol_sha256':sha(ROOT/'research/36_pfister_baseline_protocol.md'),
        'source_sha256':sha(Path(__file__)),
        'previous_trials_sha256':sha(ROOT/'results/fair_baselines_20261104/trials.csv')}
    (OUT/'started.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    previous=pd.read_csv(ROOT/'results/fair_baselines_20261104/trials.csv')
    refs=previous[previous.method.isin(['certified','grid_repair_cv','sdp_cv'])].set_index(['scenario','n','rep','method'])
    records=[]; tuning=[]; diagnostics=[]; failures=[]; start=time.perf_counter()
    for si,name in enumerate(NAMES):
        target_true=truth(name)
        for n in [100,500]:
            for rep in range(REPS):
                rng=np.random.default_rng(np.random.SeedSequence([SEED,si,n,rep]))
                x=draw(name,n,rng); folds=np.array_split(rng.permutation(n),5)
                cv=[]
                for val in folds:
                    train=np.setdiff1d(np.arange(n),val)
                    cv.append((moments(x[train],max(RGRID)),moments(x[val],max(RGRID)),len(val)))
                candidates=[];failed=False
                for r in RGRID:
                    for ratio in ([0] if r==0 else RATIOS):
                        score=0.
                        for k,(train,valid,nval) in enumerate(cv):
                            try:a,info=estimate(train,r,ratio)
                            except Exception as exc:
                                failures.append({'scenario':name,'n':n,'rep':rep,'stage':'cv','r':r,'ratio':ratio,'fold':k,'error':repr(exc)})
                                failed=True;break
                            score+=nval/n*l2score(a,valid)
                            diagnostics.append({'scenario':name,'n':n,'rep':rep,'r':r,'ratio':ratio,'fold':k,**info})
                        if failed:break
                        candidates.append((score,r,ratio))
                        tuning.append({'scenario':name,'n':n,'rep':rep,'r':r,'ratio':ratio,'cv_score':score})
                    if failed:break
                if failed:
                    records.append({'scenario':name,'n':n,'rep':rep,'status':'failed_cv'});continue
                score,r,ratio=min(candidates,key=lambda row:row[0]); full=moments(x,max(RGRID))
                try:a,info=estimate(full,r,ratio)
                except Exception as exc:
                    failures.append({'scenario':name,'n':n,'rep':rep,'stage':'refit','r':r,'ratio':ratio,'fold':-1,'error':repr(exc)})
                    records.append({'scenario':name,'n':n,'rep':rep,'status':'failed_refit'});continue
                padding=np.pad(a,(0,len(target_true)-len(a)))
                ise=float(np.sum(abs(padding[1:]-target_true[1:])**2)/np.pi)
                solver_status=info.pop('status')
                entry={'scenario':name,'n':n,'rep':rep,'status':'ok','solver_status':solver_status,'r':r,'ratio':ratio,'ise':ise,
                    'cv_score':score,'max_r':r==max(RGRID),'max_ratio':ratio==max(RATIOS),**info}
                for method in ['certified','grid_repair_cv','sdp_cv']:
                    entry[method+'_ise']=float(refs.loc[(name,n,rep,method)].ise)
                records.append(entry)
            print(f'completed {name} n={n}, {time.perf_counter()-start:.1f}s, failures={len(failures)}',flush=True)
            pd.DataFrame(records).to_csv(OUT/'partial_trials.csv',index=False)
            pd.DataFrame(failures).to_csv(OUT/'partial_failures.csv',index=False)
    trials=pd.DataFrame(records);pd.DataFrame(tuning).to_csv(OUT/'tuning.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'fold_diagnostics.csv',index=False)
    trials.to_csv(OUT/'trials.csv',index=False);pd.DataFrame(failures,columns=['scenario','n','rep','stage','r','ratio','fold','error']).to_csv(OUT/'failures.csv',index=False)
    summaries=[]
    for (name,n),g in trials.groupby(['scenario','n'],sort=True):
        ok=g[g.status=='ok']; row={'scenario':name,'n':int(n),'attempted':len(g),'success':len(ok),'failures':len(g)-len(ok)}
        if len(ok):
            row.update(mise=ok.ise.mean(),mcse=ok.ise.std(ddof=1)/np.sqrt(len(ok)),
                r_boundary=ok.max_r.mean(),ratio_boundary=ok.max_ratio.mean())
            for method in ['certified','grid_repair_cv','sdp_cv']:
                d=ok.ise-ok[method+'_ise'];row[method+'_delta']=d.mean();row[method+'_paired_mcse']=d.std(ddof=1)/np.sqrt(len(d))
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    meta.update(completed=True,attempted=len(trials),successful=int((trials.status=='ok').sum()),
        failures=len(failures),candidate_rows=len(tuning),fold_fits=len(diagnostics),seconds=time.perf_counter()-start)
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
