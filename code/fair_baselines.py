"""Independently tune Fejer, globally repaired grid, and full SDP baselines.

Protocol: research/16_fair_baselines_protocol.md. Result directory refuses
overwriting; numerical failures leave a failure record and partial CSV files.
"""
from pathlib import Path
import hashlib,json,time,warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from threadpoolctl import threadpool_limits
from circular import TAU,moments,l2score
from certified_experiment import NAMES,draw,truth,H
from certified_projection import fit,outer_matrix,project_polytope
from separated_degree import certificate_matrix

ROOT=Path(__file__).resolve().parents[1]
RGRID=[0,1,2,3,4,6,8,12]
RATIOS=[2,4,8,16]
METHODS=['certified','grid_repair_cv','sdp_cv']
REPEATS=40
SEED=20261104


class SDPSolver:
    def __init__(self):
        self.problems={};self.calls=0;self.analytic=0;self.cert_shortcut=0
        self.status={};self.max_psd_repair=0.;self.min_eigenvalue=0.;self.max_residual=0.
        for r in RGRID:
            if r<2:continue
            Q=cp.Variable((r+1,r+1),hermitian=True)
            a=cp.hstack([cp.sum(cp.diag(Q,k=l)) for l in range(1,r+1)])
            c=cp.Parameter(r,complex=True)
            problem=cp.Problem(cp.Minimize(.5*cp.sum_squares(cp.abs(a-c))),[Q >> 0,cp.trace(Q)==1])
            self.problems[r]=(problem,Q,c)

    def __call__(self,c,r):
        self.calls+=1
        if r==0:
            self.analytic+=1;return np.array([1.+0j]),dict(status='analytic',psd_repair=0.,min_eigenvalue=0.,residual=0.)
        if r==1:
            self.analytic+=1
            a=c[1]*min(1.,.5/abs(c[1])) if abs(c[1]) else c[1]
            return np.array([1.+0j,a]),dict(status='analytic',psd_repair=0.,min_eigenvalue=0.,residual=0.)
        target=np.r_[c[1:r+1].real,c[1:r+1].imag]
        if np.min(1+certificate_matrix(r,2*r)@target)>=1e-11:
            self.cert_shortcut+=1
            return c[:r+1].copy(),dict(status='feasible_shortcut',psd_repair=0.,min_eigenvalue=0.,residual=0.)
        problem,Q,param=self.problems[r];param.value=c[1:r+1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            problem.solve(solver='SCS',eps=5e-8,max_iters=30000,warm_start=True,verbose=False)
        status=problem.status;self.status[status]=self.status.get(status,0)+1
        if status not in ('optimal','optimal_inaccurate') or Q.value is None:raise RuntimeError('SDP status '+str(status))
        q=(Q.value+Q.value.conj().T)/2
        w,v=np.linalg.eigh(q);raw_min=float(w.min());wp=np.maximum(w,0)
        if wp.sum()<=0:raise RuntimeError('PSD projection has zero trace')
        repaired=(v*wp)@v.conj().T;repaired/=wp.sum()
        repair=float(np.linalg.norm(repaired-q,'fro'))
        self.min_eigenvalue=min(self.min_eigenvalue,raw_min);self.max_psd_repair=max(self.max_psd_repair,repair)
        extra=problem.solver_stats.extra_stats['info']
        residual=max(float(extra['res_pri']),float(extra['res_dual']),float(extra['gap']))
        self.max_residual=max(self.max_residual,residual)
        if repair>1e-5 or residual>2e-5:
            raise RuntimeError(f'SDP tolerance exceeded: {status} repair={repair}, residual={residual}')
        coeff=np.r_[1.+0j,[np.sum(np.diag(repaired,k=l)) for l in range(1,r+1)]]
        return coeff,dict(status=status,psd_repair=repair,min_eigenvalue=raw_min,residual=residual)


def grid_repair(c,r,ratio):
    if r==0:return np.array([1.+0j]),dict(D=0,delta=0.,stationarity=0.,grid_slack=1.)
    D=ratio*r;A=outer_matrix(r,D);target=np.r_[c[1:r+1].real,c[1:r+1].imag]
    z,_,_,stationarity=project_polytope(target,A)
    coeff=z[:r]+1j*z[r:]
    beta=(np.pi/(2*D+1))**2*np.sum(np.arange(1,r+1)**2*abs(coeff))
    delta=max(0.,beta-np.min(1+A@z))+1e-12
    z=z/(1+delta)
    return np.r_[1.+0j,z[:r]+1j*z[r:]],dict(D=D,delta=float(delta),stationarity=stationarity,
        grid_slack=float(np.min(1+A@z)))


def candidate(method,ct,r,ratio,nt,sdp):
    if method=='certified':return fit(ct,r,nt)
    if method=='grid_repair_cv':return grid_repair(ct,r,ratio)
    return sdp(ct,r)


def main():
    out=ROOT/'results'/'fair_baselines_20261104';out.mkdir(exist_ok=False)
    protocol=ROOT/'research'/'16_fair_baselines_protocol.md'
    meta=dict(seed=SEED,repetitions=REPEATS,folds=5,rgrid=RGRID,repair_ratios=RATIOS,
        eta=.1,protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Full SDP CV uses capped r<=12; candidate and grid repair share this cap.')
    (out/'started.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    rows=[];tuning=[];full_diagnostics=[];start=time.perf_counter();sdp=SDPSolver();n_fits=0
    try:
        for si,name in enumerate(NAMES):
            true=truth(name)
            for n in [100,500]:
                for rep in range(REPEATS):
                    rng=np.random.default_rng(np.random.SeedSequence([SEED,si,n,rep]))
                    x=draw(name,n,rng);folds=np.array_split(rng.permutation(n),5)
                    cv=[]
                    for ids in folds:
                        train=np.setdiff1d(np.arange(n),ids)
                        cv.append((moments(x[train],max(RGRID)),moments(x[ids],max(RGRID)),len(ids),len(train)))
                    full=moments(x,max(RGRID))
                    for method in METHODS:
                        choices=[(r,ratio) for r in RGRID for ratio in ([0] if r==0 or method!='grid_repair_cv' else RATIOS)]
                        scores=[]
                        for r,ratio in choices:
                            score=0.
                            for ct,valid,nv,nt in cv:
                                a,diag=candidate(method,ct,r,ratio,nt,sdp);n_fits+=1
                                score+=nv/n*l2score(a,valid)
                            scores.append(float(score))
                            tuning.append(dict(scenario=name,n=n,rep=rep,method=method,r=r,ratio=ratio,cv_score=score))
                        best=int(np.argmin(scores));r,ratio=choices[best]
                        a,diag=candidate(method,full,r,ratio,n,sdp);n_fits+=1
                        if method=='certified' and diag['l2_squared_bound']>diag['target']+1e-12:
                            raise RuntimeError('Certificate stopping bound exceeded')
                        padded=np.pad(a,(0,H+1-len(a)))
                        ise=float(np.sum(abs(padded[1:]-true[1:])**2)/np.pi)
                        rows.append(dict(scenario=name,n=n,rep=rep,method=method,r=r,ratio=ratio,
                            ise=ise,cv_score=scores[best],r_boundary=r==max(RGRID),
                            ratio_boundary=bool(method=='grid_repair_cv' and ratio==max(RATIOS))))
                        full_diagnostics.append(dict(scenario=name,n=n,rep=rep,method=method,
                            **{k:v for k,v in diag.items() if k!='history'}))
                pd.DataFrame(rows).to_csv(out/'trials.csv',index=False)
                pd.DataFrame(tuning).to_csv(out/'tuning.csv',index=False)
                pd.DataFrame(full_diagnostics).to_csv(out/'full_diagnostics.csv',index=False)
                print(f'{name} n={n} completed, {time.perf_counter()-start:.1f}s',flush=True)
    except Exception as exc:
        pd.DataFrame(rows).to_csv(out/'trials.csv',index=False)
        pd.DataFrame(tuning).to_csv(out/'tuning.csv',index=False)
        pd.DataFrame(full_diagnostics).to_csv(out/'full_diagnostics.csv',index=False)
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),scenario=name,n=n,rep=rep,method=method,r=r,ratio=ratio),indent=2),encoding='utf-8')
        raise
    df=pd.DataFrame(rows);summary=[]
    for (name,n),g in df.groupby(['scenario','n'],sort=False):
        ref=g[g.method=='certified'].set_index('rep').ise
        for method,z in g.groupby('method',sort=False):
            v=z.set_index('rep').ise;diff=ref-v
            summary.append(dict(scenario=name,n=n,method=method,mise=v.mean(),mcse=v.std(ddof=1)/np.sqrt(len(v)),
                certified_minus_method=diff.mean(),paired_mcse=diff.std(ddof=1)/np.sqrt(len(v)),
                r_boundary_fraction=z.r_boundary.mean(),ratio_boundary_fraction=z.ratio_boundary.mean()))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    meta.update(completed=True,sample_groups=6*2*REPEATS,final_fits=len(rows),candidate_scores=len(tuning),
        total_fits=n_fits,failures=0,sdp_calls=sdp.calls,sdp_analytic=sdp.analytic,
        sdp_feasible_shortcut=sdp.cert_shortcut,sdp_status=sdp.status,sdp_max_psd_repair=sdp.max_psd_repair,
        sdp_min_raw_eigenvalue=sdp.min_eigenvalue,sdp_max_solver_residual=sdp.max_residual,
        seconds=time.perf_counter()-start)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
