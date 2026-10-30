"""Paired pilot. All tuning uses observed samples only; truth is evaluation-only."""
from pathlib import Path
import argparse, json, time, platform, sys
import numpy as np
import pandas as pd
from scipy.special import ive
from threadpoolctl import threadpool_limits
from circular import moments, fit_continuous, l2score, density, TAU

SCENARIOS = {
    "smooth": [(1., .7, 2.)],
    "bimodal": [(.6, .5, 4.), (.4, 3.4, 7.)],
    "narrow": [(1., .73, 20.)],
    "boundary": [(.7, .03, 10.), (.3, 3.2, 3.)],
}
CANDIDATES = {
    "raw_fourier": [1, 2, 4, 8, 16, 32],
    "fejer": [1, 2, 4, 8, 16, 32],
    "fejer_mix": [1, 2, 4, 8, 16, 32],
    "dvp_mix": [1, 2, 4, 8, 16, 32],
    "von_mises": [0., .5, 1., 2., 4., 8., 16., 32., 64.],
}


def truth_coefficients(scenario, degree=128):
    l = np.arange(degree+1)
    return sum(w*ive(l, k)/ive(0, k)*np.exp(-1j*l*mu) for w, mu, k in SCENARIOS[scenario])


def sample(scenario, n, rng):
    components = SCENARIOS[scenario]
    which = rng.choice(len(components), n, p=[z[0] for z in components])
    x = np.empty(n)
    for j, (_, mu, kappa) in enumerate(components):
        x[which == j] = rng.vonmises(mu, kappa, np.sum(which == j))
    return x


def run(out, repetitions):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite recorded results: {out}')
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    seed = 20261026
    rows, tuning = [], []
    H = 128
    max_solver_residual = 0.
    for sindex, name in enumerate(SCENARIOS):
        true = truth_coefficients(name, H)
        for n in [100, 500]:
            for rep in range(repetitions):
                rng = np.random.default_rng(np.random.SeedSequence([seed, sindex, n, rep]))
                x = sample(name, n, rng)
                folds = np.array_split(rng.permutation(n), 3)
                train_val = []
                for validation in folds:
                    train = np.setdiff1d(np.arange(n), validation)
                    train_val.append((moments(x[train], H), moments(x[validation], H), len(validation)))
                full = moments(x, H)
                for method, candidates in CANDIDATES.items():
                    scores = []
                    for par in candidates:
                        score = 0.
                        for train, val, count in train_val:
                            fit = fit_continuous(train, method, par)
                            max_solver_residual = max(max_solver_residual, fit.residual)
                            score += count/n*l2score(fit.coefficients, val)
                        scores.append(score)
                        tuning.append(dict(scenario=name,n=n,rep=rep,method=method,parameter=par,cv_score=score))
                    chosen = int(np.argmin(scores))
                    fit = fit_continuous(full, method, candidates[chosen])
                    max_solver_residual = max(max_solver_residual, fit.residual)
                    c = np.pad(fit.coefficients, (0, H+1-len(fit.coefficients)))
                    ise = float((abs(c[0]-true[0])**2+2*np.sum(abs(c[1:]-true[1:])**2))/TAU)
                    rows.append(dict(scenario=name,n=n,rep=rep,method=method,parameter=candidates[chosen],
                        ise=ise,cv_score=scores[chosen],boundary_choice=chosen==len(candidates)-1,
                        min_density=float(density(fit.coefficients, TAU*np.arange(1024)/1024).min()),
                        solver_residual=fit.residual,iterations=fit.iterations))
            pd.DataFrame(rows).to_csv(out/'trials.csv', index=False)
            pd.DataFrame(tuning).to_csv(out/'tuning.csv', index=False)
            print(f"completed {name} n={n}, repetitions={repetitions}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out/'trials.csv', index=False)
    pd.DataFrame(tuning).to_csv(out/'tuning.csv', index=False)
    summary=[]
    for (name,n), grp in df.groupby(['scenario','n'],sort=False):
        reference = grp[grp.method=='fejer_mix'].set_index('rep').ise
        for method, z in grp.groupby('method',sort=False):
            v=z.set_index('rep').ise
            diff=reference-v
            summary.append(dict(scenario=name,n=n,method=method,repetitions=repetitions,
                mise=v.mean(),mcse=v.std(ddof=1)/np.sqrt(len(v)),mix_minus_method=diff.mean(),
                paired_mcse=diff.std(ddof=1)/np.sqrt(len(diff)),
                upper_boundary_fraction=z.boundary_choice.mean(),negative_fraction=(z.min_density < -1e-10).mean()))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    meta=dict(seed=seed,repetitions=repetitions,candidates=CANDIDATES,folds=3,
        truth_frequency_limit=H,truth_tail_note="VM tails above 128 are numerically negligible for kappa <= 20; verification supplied.",
        solver_tolerance=1e-8,max_solver_residual=max_solver_residual,elapsed_seconds=time.perf_counter()-start,
        python=sys.version,numpy=np.__version__,platform=platform.platform(),
        stage="pilot; no confirmatory significance claims; finite tuning grids")
    (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--repetitions',type=int,default=30)
    a=ap.parse_args()
    if a.repetitions < 2: raise ValueError('Need at least two repetitions for MCSE.')
    with threadpool_limits(limits=1): run(a.output,a.repetitions)
