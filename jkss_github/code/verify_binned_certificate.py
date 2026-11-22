"""Cross-check weighted-bin certificate against quadrature and PSD projection."""
from pathlib import Path
import json,warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.integrate import quad
from threadpoolctl import threadpool_limits
from circular import density,bin_probabilities,TAU
from certified_binned import fit,bin_design

ROOT=Path(__file__).resolve().parents[1]


def sdp(y,r):
    H=len(y);B=bin_design(r,H);Q=cp.Variable((r+1,r+1),hermitian=True)
    coeff=cp.hstack([cp.sum(cp.diag(Q,k=l)) for l in range(1,r+1)])
    z=cp.hstack([cp.real(coeff),cp.imag(coeff)]);p=1/H+B@z
    prob=cp.Problem(cp.Minimize(.5*cp.sum_squares(p-y)),[Q>>0,cp.trace(Q)==1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        prob.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=300)
    if prob.status not in ['optimal','optimal_inaccurate']:raise RuntimeError(prob.status)
    return np.asarray(p.value).reshape(-1),dict(status=prob.status,minimum_eigenvalue=float(np.linalg.eigvalsh(Q.value).min()),
        trace_error=float(abs(np.trace(Q.value)-1)),objective=float(prob.value))


def main():
    d=pd.read_csv(ROOT/'data'/'processed'/'complete_days_hourly.csv',parse_dates=['dteday'])
    counts=d.pivot(index='dteday',columns='hr',values='cnt').sort_index()
    shares=counts.div(counts.sum(axis=1),axis=0)
    wd=d.groupby('dteday').workingday.first().reindex(shares.index)
    cases=[];max_integral_error=0.;max_gram_error=0.
    for month in ['2012-01-01','2012-04-01','2012-10-01']:
        start=pd.Timestamp(month);window=start-pd.Timedelta(days=180)
        for status in [0,1]:
            subset=shares[wd==status];refit=subset[(subset.index>=window)&(subset.index<start)]
            y=refit.mean().to_numpy()
            for r in [2,5,9]:
                q,a,info=fit(y,r,'adaptive');qref,info_ref=sdp(y,r)
                diff=float(np.sum((q-qref)**2));excess=diff-info['error_bound']
                if excess>2e-8:raise RuntimeError(f'Error certificate outside numerical tolerance: {excess}')
                assert info_ref['minimum_eigenvalue']>-1e-7
                expected=(2/24)*np.tile(np.sinc(np.arange(1,r+1)/24)**2,2)
                B=bin_design(r,24);gram=float(np.max(abs(B.T@B-np.diag(expected))))
                max_gram_error=max(max_gram_error,gram)
                exact=bin_probabilities(a,24)
                assert np.max(abs(q-exact))<1e-12
                for h in [0,7,12,23]:
                    integral=quad(lambda x:float(density(a,np.array([x]))[0]),TAU*h/24,TAU*(h+1)/24,epsabs=1e-12)[0]
                    max_integral_error=max(max_integral_error,abs(integral-q[h]))
                cases.append(dict(month=month,workingday=status,r=r,D=info['D'],
                    binned_sq_difference=diff,bound=info['error_bound'],bound_excess=excess,**info_ref))
                print(f'checked {month} status={status} r={r}',flush=True)
    assert max_integral_error<1e-12 and max_gram_error<1e-12
    out=dict(passed=True,SDP_cases=len(cases),status_counts=pd.Series([c['status'] for c in cases]).value_counts().to_dict(),
        maximum_bound_excess=max(c['bound_excess'] for c in cases),
        minimum_eigenvalue=min(c['minimum_eigenvalue'] for c in cases),
        max_bin_quadrature_difference=max_integral_error,max_gram_difference=max_gram_error,cases=cases,
        caveat='Floating-point numerical comparison with SDP; not interval arithmetic.')
    (ROOT/'results'/'binned_certificate_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
