"""Fejer inner / grid outer brackets for nonnegative Fourier projection.

The certificate bounds approximation to the *full* nonnegative polynomial
projection, not estimation error relative to the unknown density. Duality and
strong convexity are standard; the specific Fourier bracket is the candidate.
Floating point diagnostics are not interval-arithmetic certificates.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize, nnls
from separated_degree import certificate_matrix
from circular import TAU


@lru_cache(maxsize=32)
def outer_matrix(r, D):
    phi = TAU * np.arange(2 * D + 1) / (2 * D + 1)
    e = np.exp(1j * np.outer(phi, np.arange(1, r + 1)))
    return 2 * np.c_[e.real, -e.imag]


def project_polytope(target, A):
    """Return a feasible upper point and a nonnegative dual vector.

    For min .5||z-c||^2 s.t. 1+Az>=0, every lambda>=0 gives
    dual value -.5||A.T lambda||^2-lambda.T(1+A c).
    A computed primal objective alone is NEVER used as a lower bound.
    """
    if np.min(1 + A @ target) >= 1e-11:
        return target.copy(), np.zeros(A.shape[0]), 0, 0.
    opt = minimize(lambda z: .5 * np.sum((z - target)**2), np.zeros_like(target),
                   jac=lambda z: z - target,
                   constraints=[dict(type='ineq', fun=lambda z: 1 + A @ z,
                                     jac=lambda z: A)], method='SLSQP',
                   options={'ftol': 1e-12, 'maxiter': 2000})
    if not opt.success:
        raise RuntimeError(f'QP failed: {opt.message}')
    z = opt.x
    active = (1 + A @ z) < 1e-7
    lam = np.zeros(A.shape[0])
    if np.any(active):
        lam[active], residual = nnls(A[active].T, z - target, maxiter=20000)
    else:
        residual = np.linalg.norm(z - target)
    slack = float(np.min(1 + A @ z))
    if slack < -1e-8 or residual > 2e-6:
        raise RuntimeError(f'QP check failed: slack={slack}, KKT={residual}')
    # Homotopy toward the uniform density repairs rounding-scale infeasibility.
    if slack < 1e-11:
        z = z * ((1 - 1e-11) / (1 - slack))
    return z, lam, int(opt.nit), float(residual)


def bracket(c, r, D):
    if r < 1 or D < r or len(c) <= r:
        raise ValueError('Require 1 <= r <= D and c[0:r+1].')
    target = np.r_[c[1:r+1].real, c[1:r+1].imag]
    Ai, Ao = certificate_matrix(r, D), outer_matrix(r, D)
    zi, _, ni, ki = project_polytope(target, Ai)
    zo, lam, no, ko = project_polytope(target, Ao)
    primal = .5 * float(np.sum((zi - target)**2))
    atl = Ao.T @ lam
    dual_raw = -.5 * float(atl @ atl) - float(lam @ (1 + Ao @ target))
    # Small conservative arithmetic allowance, not a rigorous FP error analysis.
    pad = 1e-12 * (1 + primal + abs(dual_raw) + float(lam.sum()))
    lower = max(0., dual_raw - pad)
    upper = primal + pad
    if lower > upper + 1e-10:
        raise RuntimeError('Invalid primal-dual ordering.')
    direct_bound = max(0., 2 * (upper - lower) / np.pi)
    # A second, sharper bound. The grid outer solution has a certified
    # continuous lower bound from Taylor's theorem at its global minimum.
    # Repair it toward uniform, obtaining a full-cone feasible comparison.
    coeff_o = zo[:r] + 1j*zo[r:]
    beta = (np.pi/(2*D+1))**2 * float(np.sum(np.arange(1,r+1)**2*abs(coeff_o)))
    repair = max(0., beta - float(np.min(1 + Ao @ zo))) + 1e-12
    zr = zo / (1 + repair)
    repaired_upper = .5*float(np.sum((zr-target)**2)) + pad
    # A lower bound of zero corresponds to lambda=0, not to an arbitrary
    # multiplier with a negative dual objective. Keep its center consistent.
    dual_center = target + atl if dual_raw >= 0 else target.copy()
    # Every full-cone feasible z satisfies J(z)>=g(lambda)+||z-center||^2/2.
    radius = np.sqrt(max(0., 2*(repaired_upper-lower)))
    ball_bound = (np.linalg.norm(zi-dual_center)+radius)**2/np.pi
    bound = min(direct_bound, ball_bound)
    a = np.r_[1. + 0j, zi[:r] + 1j * zi[r:]]
    info = dict(r=int(r), D=int(D), objective_upper=upper, objective_lower=lower,
                l2_squared_bound=bound, direct_bound=direct_bound, ball_bound=ball_bound,
                dual_zero_fallback=bool(dual_raw < 0),
                repair_delta=repair, minimum_weight=float(np.min(1 + Ai @ zi)/(2*D+1)),
                max_kkt=max(ki, ko), iterations=ni+no,
                condition_bound=(D+1)/(D+1-r))
    return a, info


def fit(c, r, n, eta=.1, max_ratio=256):
    """Increase certificate degree until bound <= eta*r/(pi*n).

    r is fixed within each call; eta is fixed before simulation. Failure to
    reach the target is explicit. No silent substitution of an uncertified fit.
    """
    if n <= 0 or eta <= 0 or r < 0:
        raise ValueError('Invalid n, eta, or r.')
    if r == 0:
        return np.array([1.+0j]), dict(r=0, D=0, l2_squared_bound=0., target=0.,
            minimum_weight=1., max_kkt=0., iterations=0, condition_bound=1.,
            refinements=0, history=[])
    target = eta * r / (np.pi * n)
    D, history = 2*r, []
    while D <= max_ratio*r:
        a, info = bracket(c, r, D)
        history.append(info)
        if info['l2_squared_bound'] <= target:
            return a, dict(info, target=target, refinements=len(history)-1,
                           history=history)
        D *= 2
    raise RuntimeError(f'Certificate target not reached: r={r}, n={n}, last={history[-1]}')
