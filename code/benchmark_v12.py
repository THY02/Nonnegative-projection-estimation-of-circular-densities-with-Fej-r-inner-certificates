"""Fixed-r numerical timing comparison; raw samples and diagnostics are saved."""
import csv, json, os, platform, time
from pathlib import Path
import numpy as np
import scipy, cvxpy as cp, scs
from threadpoolctl import threadpool_limits
import certified_projection as cert
from circular import moments
from separated_degree import certificate_matrix

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'benchmark_v12'
OUT.mkdir(exist_ok=False)
orig=cert.minimize
timing={}
def timed(*args,**kwargs):
    start=time.perf_counter()
    try:return orig(*args,**kwargs)
    finally:timing['solver']+=time.perf_counter()-start
cert.minimize=timed
rows=[]; samples={}
with threadpool_limits(limits=1):
 for rep in range(3):
  rng=np.random.default_rng(20261030+rep)
  x=rng.vonmises(0,20,500) % (2*np.pi)
  samples[f'sample_{rep}']=x
  for r in (4,8,12,24):
   c=moments(x,r); target=np.r_[c[1:].real,c[1:].imag]
   timing['solver']=0.; start=time.perf_counter()
   try:
    a,d=cert.fit(c,r,500,eta=.1,max_ratio=256)
    t=time.perf_counter()-start
    zi=np.r_[a[1:].real,a[1:].imag]
    obj=.5*np.sum((zi-target)**2)
    rows.append(dict(rep=rep,r=r,method='certificate',status='ok',seconds=t,solver_seconds=timing['solver'],objective=obj,feasibility=max(0.,-float(np.min(1+certificate_matrix(r,d['D'])@zi))),bound=d['l2_squared_bound'],D=d['D'],iterations=sum(h['iterations'] for h in d['history'])))
    samples[f'cert_{rep}_{r}']=a
   except Exception as e: rows.append(dict(rep=rep,r=r,method='certificate',status=str(e),seconds=time.perf_counter()-start))
   start=time.perf_counter()
   try:
    Q=cp.Variable((r+1,r+1),hermitian=True)
    ac=cp.hstack([cp.sum(cp.diag(Q,k=k)) for k in range(1,r+1)])
    prob=cp.Problem(cp.Minimize(.5*cp.sum_squares(cp.abs(ac-c[1:]))),[Q >> 0,cp.trace(Q)==1])
    prob.solve(solver='SCS',eps=5e-8,max_iters=30000,warm_start=False)
    elapsed=time.perf_counter()-start
    if Q.value is None: raise RuntimeError(str(prob.status))
    q=(Q.value+Q.value.conj().T)/2
    vals,vec=np.linalg.eigh(q); repaired=(vec*np.maximum(vals,0))@vec.conj().T; repaired/=np.trace(repaired).real
    a=np.r_[1.+0j,[np.sum(np.diag(repaired,k=k)) for k in range(1,r+1)]]
    info=prob.solver_stats.extra_stats['info']
    rows.append(dict(rep=rep,r=r,method='sdp',status=prob.status,seconds=elapsed,solver_seconds=prob.solver_stats.solve_time,objective=.5*np.sum(abs(a[1:]-c[1:])**2),feasibility=max(0.,-float(vals.min())),residual=max(float(info[k]) for k in ('res_pri','res_dual','gap')),iterations=prob.solver_stats.num_iters))
    samples[f'sdp_{rep}_{r}']=a
   except Exception as e:rows.append(dict(rep=rep,r=r,method='sdp',status=str(e),seconds=time.perf_counter()-start))
   print(rep,r,flush=True)
with (OUT/'runs.csv').open('w',newline='',encoding='utf-8') as f:
 fields=['rep','r','method','status','seconds','solver_seconds','objective','feasibility','residual','bound','D','iterations']; w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
np.savez(OUT/'samples_and_coefficients.npz',**samples)
(OUT/'manifest.json').write_text(json.dumps(dict(seed=[20261030,20261031,20261032],n=500,kappa=20,r=[4,8,12,24],eta=.1,python=platform.python_version(),platform=platform.platform(),processor=platform.processor(),numpy=np.__version__,scipy=scipy.__version__,cvxpy=cp.__version__,scs=scs.__version__,threads=1,sdpsolver='SCS eps=5e-8 max_iters=30000 cold start',timing_note='SDP total includes CVXPY model construction and compilation; solve_time excludes them. Certificate solver time counts SLSQP calls, not NNLS or certificate evaluation.'),indent=2),encoding='utf-8')
