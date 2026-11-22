"""Check cases flagged optimal_inaccurate by Clarabel with another solver."""
from pathlib import Path
import json
import numpy as np
import cvxpy as cp
from threadpoolctl import threadpool_limits
from circular import moments,TAU
from certified_projection import fit

ROOT=Path(__file__).resolve().parents[1]


def main():
    original=json.loads((ROOT/'results'/'certified_verification.json').read_text())
    rows=[]
    for case in original['cases']:
        if case['kind']!='sdp_comparison' or case['sdp']['status']!='optimal_inaccurate':continue
        r,shape=case['r'],case['shape']
        rng=np.random.default_rng(np.random.SeedSequence([20261030,r,['uniform','smooth','narrow'].index(shape)]))
        x=rng.uniform(0,TAU,100) if shape=='uniform' else rng.vonmises(.7,2 if shape=='smooth' else 20,100)
        c=moments(x,r);ours,diag=fit(c,r,100)
        Q=cp.Variable((r+1,r+1),hermitian=True)
        a=cp.hstack([cp.sum(cp.diag(Q,k=l)) for l in range(1,r+1)])
        problem=cp.Problem(cp.Minimize(.5*cp.sum_squares(cp.abs(a-c[1:]))),[Q>>0,cp.trace(Q)==1])
        problem.solve(solver='SCS',eps=1e-8,max_iters=100000)
        assert problem.status=='optimal'
        eig=float(np.linalg.eigvalsh(Q.value).min())
        assert eig>=-2e-7
        objective_diff=abs(float(problem.value)-case['sdp']['objective'])
        error=float(np.sum(abs(ours[1:]-a.value)**2)/np.pi)
        assert error<=diag['l2_squared_bound']+2e-7 and objective_diff<2e-7
        rows.append(dict(r=r,shape=shape,status=problem.status,minimum_eigenvalue=eig,
            clarabel_objective_difference=objective_diff,l2_squared_difference=error,bound=diag['l2_squared_bound']))
    result=dict(passed=True,cases=rows,scope='Cross-solver finite-precision evidence; no exact arithmetic claim.')
    (ROOT/'results'/'sdp_followup.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
