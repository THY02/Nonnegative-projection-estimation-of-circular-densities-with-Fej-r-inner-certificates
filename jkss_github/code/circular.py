"""Research prototype: circular Fourier coefficients use E[exp(-i*l*X)].

Fejer/DVP kernels, Euclidean simplex projection and accelerated projected
gradient are existing tools. This implementation does not establish novelty.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import gammaln, ive

TAU = 2 * np.pi


def simplex(v):
    u = np.sort(v)[::-1]
    s = np.cumsum(u) - 1
    ids = np.flatnonzero(u - s / np.arange(1, len(v) + 1) > 0)
    j = ids[-1]
    return np.maximum(v - s[j] / (j + 1), 0)


def moments(x, degree):
    return np.exp(-1j * np.outer(np.arange(degree + 1), x)).mean(axis=1)


def multiplier(d, family="fejer"):
    l = np.arange(d + 1)
    if family == "fejer":
        return 1 - l / (d + 1)
    if family == "dvp":
        return np.exp(2*gammaln(d+1) - gammaln(d-l+1) - gammaln(d+l+1))
    raise ValueError(family)


def complete(c):
    return np.r_[c, c[:0:-1].conj()]


def density(c, x):
    return (c[0].real + 2*np.real(c[1:] @ np.exp(1j*np.outer(np.arange(1, len(c)), x))))/TAU


def l2score(c, validation_moments):
    norm = (abs(c[0])**2 + 2*np.sum(abs(c[1:])**2))/TAU
    cross = (c[0].real + 2*np.real(np.vdot(validation_moments[1:len(c)], c[1:])))/TAU
    return float(norm - 2*cross)


def solve_simplex(grad, size, lipschitz=1., tol=1e-9, maxiter=20000):
    p = np.full(size, 1/size)
    y, alpha = p.copy(), 1.
    for iteration in range(1, maxiter+1):
        pn = simplex(y-grad(y)/lipschitz)
        an = (1+np.sqrt(1+4*alpha**2))/2
        y = pn+(alpha-1)/an*(pn-p)
        p, alpha = pn, an
        if iteration % 25 == 0:
            residual = np.linalg.norm(p-simplex(p-grad(p)/lipschitz))
            if residual <= tol:
                return p, iteration, float(residual)
    raise RuntimeError(f"Simplex solver did not converge: residual={residual}")


@dataclass
class Fit:
    coefficients: np.ndarray
    iterations: int = 0
    residual: float = 0.
    weights: np.ndarray | None = None


def fit_mixture(c, d, family="fejer", phase=0., tol=1e-8):
    t = complete(multiplier(d, family)).real
    target = complete(c[:d+1] * np.exp(1j*np.arange(d+1)*phase))
    # The objective divided by L has Hessian spectral radius <= 1.
    def grad(p):
        return np.fft.ifft(t*(t*np.fft.fft(p)-target)).real
    p, it, residual = solve_simplex(grad, 2*d+1, tol=tol)
    out = (t*np.fft.fft(p))[:d+1] * np.exp(-1j*np.arange(d+1)*phase)
    return Fit(out, it, residual, p)


def fit_continuous(c, method, parameter):
    if method == "fejer":
        d = int(parameter)
        return Fit(multiplier(d)*c[:d+1])
    if method == "raw_fourier":
        return Fit(c[:int(parameter)+1].copy())
    if method in ("fejer_mix", "dvp_mix"):
        return fit_mixture(c, int(parameter), "fejer" if method == "fejer_mix" else "dvp")
    if method == "von_mises":
        kappa = float(parameter)
        ratio = ive(np.arange(len(c)), kappa)/ive(0, kappa)
        return Fit(c*ratio)
    raise ValueError(method)


def bin_matrix(d, bins=24, family="fejer", phase=0.):
    """Exact integrals over [2*pi*h/H, 2*pi*(h+1)/H); no midpoint data fiction."""
    l = np.arange(1, d+1)
    centers = TAU*(np.arange(bins)+.5)/bins
    nodes = phase + TAU*np.arange(2*d+1)/(2*d+1)
    t = multiplier(d, family)[1:]
    difference = centers[:, None]-nodes[None, :]
    B = (1+2*np.sum((t*np.sinc(l/bins))[:, None, None] * np.cos(l[:, None, None]*difference), axis=0))/bins
    return B


def fit_binned(y, d, family="fejer", phase=0.):
    if 2*d+1 > len(y):
        raise ValueError("Require 2*d+1 <= number of observed bins for this identifiable model.")
    B = bin_matrix(d, len(y), family, phase)
    H = B.T@B
    z = B.T@y
    lipschitz = np.linalg.eigvalsh(H)[-1]
    p, it, res = solve_simplex(lambda p: H@p-z, len(z), lipschitz, tol=1e-8)
    coeff = multiplier(d, family)*np.fft.fft(p)[:d+1]*np.exp(-1j*np.arange(d+1)*phase)
    return B@p, Fit(coeff, it, res, p)


def bin_probabilities(c, bins=24):
    l = np.arange(1, len(c))
    centers = TAU*(np.arange(bins)+.5)/bins
    return (c[0].real+2*np.real((c[1:]*np.sinc(l/bins)) @ np.exp(1j*np.outer(l, centers))))/bins
