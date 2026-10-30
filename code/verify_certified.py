"""Independent analytic and SDP checks of the proposed approximation bound."""
from pathlib import Path
import json, time
import numpy as np
import cvxpy as cp
from threadpoolctl import threadpool_limits
from circular import moments, density, TAU
from certified_projection import fit, outer_matrix

ROOT = Path(__file__).resolve().parents[1]


def sdp_projection(c, r):
    Q = cp.Variable((r+1, r+1), hermitian=True)
    a = cp.hstack([cp.sum(cp.diag(Q, k=l)) for l in range(1, r+1)])
    obj = .5*cp.sum_squares(cp.abs(a-c[1:r+1]))
    problem = cp.Problem(cp.Minimize(obj), [Q >> 0, cp.trace(Q) == 1])
    start = time.perf_counter()
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                  tol_feas=1e-10, max_iter=300)
    seconds = time.perf_counter()-start
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        raise RuntimeError(problem.status)
    aa = np.r_[1.+0j, np.asarray(a.value).reshape(-1)]
    eigmin = float(np.linalg.eigvalsh(Q.value).min())
    if eigmin < -1e-7:
        raise RuntimeError(f'SDP residual too large: {eigmin}')
    return aa, dict(status=problem.status, eigenvalue_min=eigmin,
                   trace_error=float(abs(np.trace(Q.value)-1)), objective=float(problem.value),
                   seconds=seconds)


def main():
    cases=[]
    for radius in [.2, .49, .5, .7, .99]:
        for phase in [0., .7, 3.1]:
            c=np.array([1, radius*np.exp(1j*phase)])
            a,d=fit(c,1,500)
            exact=c[1]*min(1.,.5/radius)
            error=float(abs(a[1]-exact)**2/np.pi)
            assert error<=d['l2_squared_bound']+1e-10
            assert d['minimum_weight']>=-1e-12
            cases.append(dict(kind='analytic_r1',r=1,radius=radius,phase=phase,
                actual_l2_squared=error,**{k:v for k,v in d.items() if k not in ['history','r']}))
    for r in [2,4,8,16]:
        for shape in ['uniform','smooth','narrow']:
            rng=np.random.default_rng(np.random.SeedSequence([20261030,r,['uniform','smooth','narrow'].index(shape)]))
            x=rng.uniform(0,TAU,100) if shape=='uniform' else rng.vonmises(.7,2 if shape=='smooth' else 20,100)
            c=moments(x,r)
            start=time.perf_counter();a,d=fit(c,r,100);seconds=time.perf_counter()-start
            aa,sd=sdp_projection(c,r)
            error=float(np.sum(abs(a[1:]-aa[1:])**2)/np.pi)
            assert error<=d['l2_squared_bound']+2e-7
            assert d['objective_lower']<=sd['objective']+1e-7
            assert sd['objective']<=d['objective_upper']+1e-7
            cases.append(dict(kind='sdp_comparison',shape=shape,actual_l2_squared=error,
                certified_seconds=seconds,sdp=sd,**{k:v for k,v in d.items() if k!='history'}))
            print(f'verified r={r} {shape}, D={d["D"]}',flush=True)
    # A degree-one polynomial positive on all five nodes but negative between them.
    c=np.array([1.,-.55*np.exp(-1j*np.pi/5)])
    z=np.r_[c[1:].real,c[1:].imag]
    node_min=float(np.min(1+outer_matrix(1,2)@z))
    true_min=float(density(c,np.array([np.pi/5]))[0])
    assert node_min>0 and true_min<0
    rotations=[]
    for r in [2,4,8,16]:
        rng=np.random.default_rng(127100+r)
        x=rng.vonmises(.7,5,100);c=moments(x,r)
        a,d=fit(c,r,100)
        for phase in [.42,1.7]:
            rotated_c=c*np.exp(-1j*np.arange(r+1)*phase)
            ar,dr=fit(rotated_c,r,100)
            rotated_a=a*np.exp(-1j*np.arange(r+1)*phase)
            err=float(np.sum(abs(ar[1:]-rotated_a[1:])**2)/np.pi)
            bound=(np.sqrt(d['l2_squared_bound'])+np.sqrt(dr['l2_squared_bound']))**2
            assert err<=bound+1e-10
            rotations.append(dict(r=r,phase=phase,l2_squared=err,bound=bound))
    meta=dict(passed=True,analytic_cases=15,sdp_cases=12,rotation_cases=rotations,cvxpy=cp.__version__,
        counterexample=dict(node_polynomial_min=node_min,continuous_density_min=true_min),
        maximum_bound_violation=max(x['actual_l2_squared']-x['l2_squared_bound'] for x in cases),
        cases=cases,limitation='Finite-precision cross-check, not a substitute for mathematical proof or interval arithmetic.')
    (ROOT/'results'/'certified_verification.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k!='cases'},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
