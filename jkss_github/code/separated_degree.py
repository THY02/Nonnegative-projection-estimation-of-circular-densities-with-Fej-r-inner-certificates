"""Candidate: distinct statistical degree r and Fejer certificate degree D.

The statistical polynomial has degree r. Nonnegative inverse-Fejer weights
on 2D+1 roots certify its positivity everywhere. This is an inner polyhedral
approximation, not pointwise positivity enforced only on a sample grid.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize, nnls
from circular import TAU, multiplier


@lru_cache(maxsize=128)
def certificate_matrix(r,D):
    if D<r:raise ValueError('Certificate degree must be at least statistical degree.')
    phi=TAU*np.arange(2*D+1)/(2*D+1)
    e=np.exp(1j*np.outer(phi,np.arange(1,r+1)))
    t=multiplier(D)[1:r+1]
    return 2*np.c_[e.real/t,-e.imag/t]


def fit(c,r,D):
    A=certificate_matrix(r,D)
    target=np.r_[c[1:r+1].real,c[1:r+1].imag]
    if (1+A@target).min()>=0:
        z=target; iterations=0; stationarity=0.
    else:
        opt=minimize(lambda z:.5*np.sum((z-target)**2),np.zeros(2*r),jac=lambda z:z-target,
            constraints=[dict(type='ineq',fun=lambda z:1+A@z,jac=lambda z:A)],
            method='SLSQP',options={'ftol':1e-12,'maxiter':2000})
        if not opt.success:raise RuntimeError(opt.message)
        z=opt.x;iterations=opt.nit
        active=(1+A@z)<1e-7
        if np.any(active):
            _,stationarity=nnls(A[active].T,z-target,maxiter=10000)
        else:stationarity=float(np.linalg.norm(z-target))
    p=(1+A@z)/(2*D+1)
    if p.min() < -1e-10 or stationarity>2e-6:
        raise RuntimeError(f'Certificate/QP failed: {p.min()}, {stationarity}')
    a=np.r_[1.+0j,z[:r]+1j*z[r:]]
    return a,dict(minimum_weight=float(p.min()),stationarity=float(stationarity),iterations=iterations,
                  r=r,D=D,condition_bound=float(1/(1-r/(D+1))))
