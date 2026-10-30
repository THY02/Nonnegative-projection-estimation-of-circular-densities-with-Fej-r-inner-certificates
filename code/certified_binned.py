"""Globally nonnegative density with certified hourly-bin projection error.

The error certificate is in bin-probability squared norm, NOT continuous L2.
Floating point guards are numerical diagnostics, not interval arithmetic.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize,nnls
from circular import TAU
from separated_degree import certificate_matrix
from certified_projection import outer_matrix


@lru_cache(maxsize=128)
def bin_design(r,H):
    if r < 1 or 2*r+1>H:raise ValueError('Require r>=1 and 2r+1<=H.')
    l=np.arange(1,r+1);centers=TAU*(np.arange(H)+.5)/H
    e=np.exp(1j*np.outer(centers,l));s=np.sinc(l/H)
    return (2/H)*np.c_[e.real*s,-e.imag*s]


def polytope(c,w,A):
    """Weighted quadratic projection and a dual-feasible NNLS multiplier."""
    if np.min(1+A@c)>=1e-11:return c.copy(),np.zeros(A.shape[0]),0,0.
    op=minimize(lambda z:.5*np.sum(w*(z-c)**2),np.zeros_like(c),
        jac=lambda z:w*(z-c),constraints=[dict(type='ineq',fun=lambda z:1+A@z,jac=lambda z:A)],
        method='SLSQP',options={'ftol':1e-13,'maxiter':3000})
    if not op.success:raise RuntimeError(f'binned QP failed: {op.message}')
    z=op.x;slack=float(np.min(1+A@z));active=(1+A@z)<1e-7
    lam=np.zeros(A.shape[0])
    if np.any(active):lam[active],stationarity=nnls(A[active].T,w*(z-c),maxiter=20000)
    else:stationarity=float(np.linalg.norm(w*(z-c)))
    if slack < -1e-8 or stationarity>2e-6:raise RuntimeError(f'binned QP residual: {slack}, {stationarity}')
    if slack<1e-11:z*=((1-1e-11)/(1-slack))
    return z,lam,int(op.nit),float(stationarity)


def one_degree(y,r,D):
    H=len(y);B=bin_design(r,H);w=np.diag(B.T@B)
    if not np.allclose(B.T@B,np.diag(w),atol=1e-14):raise RuntimeError('Expected orthogonal bin design.')
    c=B.T@(y-1/H)/w
    AI=certificate_matrix(r,D);AO=outer_matrix(r,D)
    zi,_,ni,ki=polytope(c,w,AI);zo,lam,no,ko=polytope(c,w,AO)
    primal=.5*float(np.sum(w*(zi-c)**2));atl=AO.T@lam
    dual=-.5*float(np.sum(atl*atl/w))-float(lam@(1+AO@c))
    pad=1e-14*(1+abs(primal)+abs(dual)+float(np.sum(lam)))
    lower=max(0.,dual-pad);upper=primal+pad
    if lower>upper+1e-10:raise RuntimeError('Invalid weighted primal-dual ordering.')
    beta=(np.pi/(2*D+1))**2 * float(np.sum(np.arange(1,r+1)**2*abs(zo[:r]+1j*zo[r:])))
    delta=max(0.,beta-float(np.min(1+AO@zo)))+1e-12
    zr=zo/(1+delta);repaired=.5*float(np.sum(w*(zr-c)**2))+pad
    center=c+atl/w if dual>=0 else c.copy()
    radius=np.sqrt(max(0.,2*(repaired-lower)))
    direct=max(0.,2*(upper-lower))
    ball=(np.sqrt(np.sum(w*(zi-center)**2))+radius)**2
    bound=min(direct,ball)
    a=np.r_[1.+0j,zi[:r]+1j*zi[r:]];q=1/H+B@zi
    pmin=float(np.min(1+AI@zi)/(2*D+1))
    if q.min() < -1e-10 or abs(q.sum()-1)>1e-10 or pmin<-1e-11:
        raise RuntimeError('Binned density certificate failed.')
    return q,a,dict(r=r,D=D,min_weight=pmin,min_bin=float(q.min()),
        error_bound=float(bound),direct_bound=float(direct),ball_bound=float(ball),
        lower=lower,upper=upper,stationarity=max(ki,ko),iterations=ni+no,
        repair_delta=delta,dual_zero_fallback=bool(dual<0))


def fit(y,r,mode='adaptive',tol=1e-6,max_ratio=256):
    y=np.asarray(y,dtype=float);H=len(y)
    if np.min(y)<-1e-12 or abs(y.sum()-1)>1e-8:raise ValueError('Input must be a probability vector.')
    if r==0:return np.full(H,1/H),np.array([1.+0j]),dict(r=0,D=0,error_bound=0.,
        min_weight=1.,min_bin=1/H,stationarity=0.,iterations=0,refinements=0)
    D=2*r;refine=0
    while D<=max_ratio*r:
        q,a,d=one_degree(y,r,D)
        if mode=='fixed_2r' or d['error_bound']<=tol:
            return q,a,dict(d,refinements=refine)
        D*=2;refine+=1
    raise RuntimeError(f'bin error target not reached: r={r}, final={d}')
