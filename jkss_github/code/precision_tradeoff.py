"""Precision/runtime sensitivity against independently tuned SDP; see protocol 25."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from circular import moments,l2score
from certified_experiment import NAMES,draw,truth,H
from certified_projection import fit
from fair_baselines import SDPSolver,RGRID

ROOT=Path(__file__).resolve().parents[1]
SEED=20261104
ETAS=[.02,.1,.5,2.]


def main():
    folder=ROOT/'results'/'precision_tradeoff';folder.mkdir(exist_ok=False)
    prev=pd.read_csv(ROOT/'results'/'fair_baselines_20261104'/'trials.csv')
    old=prev[prev.method.isin(['certified','sdp_cv'])].set_index(['scenario','n','rep','method'])
    meta=dict(stage='Post-hoc precision sensitivity on previously observed 480 samples',seed=SEED,
        eta_grid=ETAS,rgrid=RGRID,folds=5,per_cell=40,max_ratio=256,
        protocol_sha256=hashlib.sha256((ROOT/'research'/'25_precision_tradeoff_protocol.md').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_trial_sha256=hashlib.sha256((ROOT/'results'/'fair_baselines_20261104'/'trials.csv').read_bytes()).hexdigest())
    (folder/'started.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    rows=[];tuning=[];failures=[];times=[];clock=time.perf_counter();solver=SDPSolver()
    # Warm up SDP compilation outside timing. This does not establish fairness
    # for different optimizers or guarantee comparable numerical accuracy.
    for r in RGRID:
        if r>=2:
            c=np.r_[1.+0j,np.full(r,.8+0j)]
            try:solver(c,r)
            except Exception as exc:raise RuntimeError(f'benchmark SDP warm-up {r}: {exc}')
    for si,name in enumerate(NAMES):
        target=truth(name)
        for n in [100,500]:
            for rep in range(40):
                rng=np.random.default_rng(np.random.SeedSequence([SEED,si,n,rep]))
                x=draw(name,n,rng);folds=np.array_split(rng.permutation(n),5);cv=[]
                for val in folds:
                    train=np.setdiff1d(np.arange(n),val)
                    cv.append((moments(x[train],max(RGRID)),moments(x[val],max(RGRID)),len(val),len(train)))
                full=moments(x,max(RGRID));prior_sdp=old.loc[(name,n,rep,'sdp_cv')]
                prior_cert=old.loc[(name,n,rep,'certified')]
                for eta in ETAS:
                    method=f'eta_{eta:g}';scores=[];failed=False
                    for r in RGRID:
                        score=0.
                        for fold,(ct,v,nv,nt) in enumerate(cv):
                            try:a,d=fit(ct,r,nt,eta=eta,max_ratio=256)
                            except Exception as exc:
                                failures.append(dict(scenario=name,n=n,rep=rep,eta=eta,stage='cv',r=r,fold=fold,error=str(exc)))
                                failed=True;break
                            score+=nv/n*l2score(a,v)
                        if failed:break
                        scores.append(float(score));tuning.append(dict(scenario=name,n=n,rep=rep,eta=eta,r=r,cv_score=score))
                    if failed:
                        rows.append(dict(scenario=name,n=n,rep=rep,eta=eta,method=method,status='failed_cv'))
                        continue
                    r=RGRID[int(np.argmin(scores))]
                    try:a,d=fit(full,r,n,eta=eta,max_ratio=256)
                    except Exception as exc:
                        failures.append(dict(scenario=name,n=n,rep=rep,eta=eta,stage='refit',r=r,fold=-1,error=str(exc)))
                        rows.append(dict(scenario=name,n=n,rep=rep,eta=eta,method=method,status='failed_refit'))
                        continue
                    padded=np.pad(a,(0,len(target)-len(a)));ise=float(np.sum(abs(padded[1:]-target[1:])**2)/np.pi)
                    if eta==.1:
                        if r!=int(prior_cert.r) or abs(ise-prior_cert.ise)>1e-10:raise RuntimeError('eta=0.1 did not reproduce past result')
                    rows.append(dict(scenario=name,n=n,rep=rep,eta=eta,method=method,status='ok',r=r,D=d['D'],
                        ise=ise,l2_squared_bound=d['l2_squared_bound'],target_bound=d['target'],
                        minimum_weight=d['minimum_weight'],stationarity=d['max_kkt'],refinements=d['refinements'],
                        chosen_score=min(scores),r_boundary=(r==max(RGRID)),
                        paired_sdp_ise=float(prior_sdp.ise)))
                if rep<10:
                    r=int(prior_sdp.r)
                    start=time.perf_counter();ref,rd=solver(full,r);elapsed=time.perf_counter()-start
                    times.append(dict(scenario=name,n=n,rep=rep,r=r,method='sdp',status=rd['status'],
                        seconds=elapsed,sdp_psd_repair=rd['psd_repair'],sdp_residual=rd['residual']))
                    for eta in ETAS:
                        start=time.perf_counter()
                        try:
                            a,d=fit(full,r,n,eta=eta,max_ratio=256);elapsed=time.perf_counter()-start
                            actual=float(np.sum(abs(a[1:]-ref[1:])**2)/np.pi)
                            excess=actual-d['l2_squared_bound']
                            if excess>2e-7:raise RuntimeError(f'Numerical error bound exceeded: {excess}')
                            times.append(dict(scenario=name,n=n,rep=rep,r=r,method=f'eta_{eta:g}',eta=eta,status='ok',seconds=elapsed,
                                D=d['D'],actual_l2_squared=actual,l2_squared_bound=d['l2_squared_bound'],bound_excess=excess))
                        except Exception as exc:
                            times.append(dict(scenario=name,n=n,rep=rep,r=r,method=f'eta_{eta:g}',eta=eta,status='failed',
                                seconds=time.perf_counter()-start,error=str(exc)))
                            failures.append(dict(scenario=name,n=n,rep=rep,eta=eta,stage='benchmark',r=r,fold=-1,error=str(exc)))
            pd.DataFrame(rows).to_csv(folder/'trials.csv',index=False)
            pd.DataFrame(tuning).to_csv(folder/'tuning.csv',index=False)
            pd.DataFrame(failures,columns=['scenario','n','rep','eta','stage','r','fold','error']).to_csv(folder/'failures.csv',index=False)
            pd.DataFrame(times).to_csv(folder/'timings.csv',index=False)
            print(f'{name} n={n} finished, {time.perf_counter()-clock:.1f}s, failures={len(failures)}',flush=True)
    data=pd.DataFrame(rows);summary=[]
    for (name,n,eta),g in data.groupby(['scenario','n','eta'],sort=False):
        z=g[g.status=='ok'];v=z.ise;diff=z.ise-z.paired_sdp_ise;ratio=z[z.r>0].D/z[z.r>0].r
        summary.append(dict(scenario=name,n=n,eta=eta,cases=len(g),success=len(z),failure=len(g)-len(z),
            mise=v.mean(),mcse=v.std(ddof=1)/np.sqrt(len(v)),eta_minus_sdp=diff.mean(),
            paired_mcse=diff.std(ddof=1)/np.sqrt(len(v)),r_boundary_fraction=z.r_boundary.mean(),
            nonzero_ratio_median=ratio.median(),nonzero_ratio_max=ratio.max()))
    pd.DataFrame(summary).to_csv(folder/'summary.csv',index=False)
    ts=pd.DataFrame(times);ts.to_csv(folder/'timings.csv',index=False)
    meta.update(completed=True,samples=480,method_runs=len(rows),successful_runs=int((data.status=='ok').sum()),
        failed_runs=int((data.status!='ok').sum()),cv_candidate_rows=len(tuning),
        numerical_error_events=len(failures),benchmark_rows=len(times),seconds=time.perf_counter()-clock,
        warning='Mise may be conditional on successful solves; run times are environment-dependent diagnostics, not general speed claims.')
    (folder/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
