"""Post-hoc same-sample study of penalized and unpenalized CV ensembles."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from circular import moments, l2score
from certified_experiment import NAMES, draw, truth, H
from certified_projection import fit
from fair_baselines import RGRID

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/cv_ensemble_sensitivity'
SEED = 20261104
K = 5
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    old = pd.read_csv(ROOT / 'results/fair_baselines_20261104/trials.csv')
    old = old[old.method == 'certified'].set_index(['scenario', 'n', 'rep'])
    old_cv = pd.read_csv(ROOT / 'results/fair_baselines_20261104/tuning.csv')
    old_cv = old_cv[old_cv.method == 'certified'].set_index(['scenario', 'n', 'rep', 'r'])
    meta = {'stage': 'Post-hoc sensitivity on previously observed 480 samples', 'seed': SEED,
        'K': K, 'eta': .1, 'rgrid': RGRID, 'scenarios': NAMES, 'per_cell': 40,
        'protocol_sha256': sha(ROOT / 'research/32_cv_ensemble_protocol.md'),
        'source_sha256': sha(Path(__file__)), 'engine_sha256': sha(ROOT / 'code/certified_projection.py'),
        'prior_trials_sha256': sha(ROOT / 'results/fair_baselines_20261104/trials.csv'),
        'prior_tuning_sha256': sha(ROOT / 'results/fair_baselines_20261104/tuning.csv')}
    (OUT / 'started.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    results=[]; candidates=[]; diagnostics=[]; coefficients=[]; keys=[]; started=time.perf_counter()
    try:
        for si,name in enumerate(NAMES):
            true = truth(name)
            for n in [100,500]:
                assert n % K == 0
                t = np.log(K * len(RGRID)**2 * n**2)
                for rep in range(40):
                    rng = np.random.default_rng(np.random.SeedSequence([SEED, si, n, rep]))
                    x = draw(name, n, rng)
                    folds = np.array_split(rng.permutation(n), K)
                    cv = []
                    for val in folds:
                        train = np.setdiff1d(np.arange(n), val)
                        cv.append((moments(x[train], max(RGRID)), moments(x[val], max(RGRID)), len(train)))
                    all_coeff = np.zeros((len(RGRID), K, max(RGRID)+1), dtype=complex)
                    score = np.zeros(len(RGRID))
                    for j,r in enumerate(RGRID):
                        for k,(train,valid,nt) in enumerate(cv):
                            a,diag = fit(train, r, nt, eta=.1, max_ratio=256)
                            all_coeff[j,k,:len(a)] = a
                            score[j] += l2score(a,valid) / K
                            diagnostics.append({'scenario':name,'n':n,'rep':rep,'r':r,'fold':k,
                                'D':diag['D'],'bound':diag['l2_squared_bound'],
                                'target':diag['target'],'minimum_weight':diag['minimum_weight']})
                        previous = float(old_cv.loc[(name,n,rep,r)].cv_score)
                        if abs(score[j]-previous)>1e-10:
                            raise RuntimeError(f'CV replay mismatch: {name} {n} {rep} {r}')
                    penalty = np.array([2*((r+1)/(2*np.pi))*t/(3*(n/K)) for r in RGRID])
                    ordinary = int(np.argmin(score)); penalized = int(np.argmin(score + penalty))
                    previous = old.loc[(name,n,rep)]
                    if RGRID[ordinary] != int(previous.r): raise RuntimeError('Historical r mismatch')
                    for j,r in enumerate(RGRID):
                        candidates.append({'scenario':name,'n':n,'rep':rep,'r':r,'score':score[j],
                            'penalty':penalty[j],'penalized_score':score[j]+penalty[j]})
                    for method,j in [('ensemble_cv',ordinary),('ensemble_penalized',penalized)]:
                        avg=all_coeff[j].mean(axis=0)
                        padded=np.pad(avg,(0,len(true)-len(avg)))
                        ise=float(np.sum(abs(padded[1:]-true[1:])**2)/np.pi)
                        results.append({'scenario':name,'n':n,'rep':rep,'method':method,'r':RGRID[j],
                            'ise':ise,'full_refit_ise':float(previous.ise),'full_refit_r':int(previous.r),
                            'boundary':RGRID[j]==max(RGRID),'uniform':RGRID[j]==0})
                    coefficients.append(all_coeff); keys.append([si,n,rep])
                print(f'completed {name} n={n} in {time.perf_counter()-started:.1f}s',flush=True)
    except Exception as exc:
        pd.DataFrame(results).to_csv(OUT/'partial_trials.csv',index=False)
        (OUT/'failure.json').write_text(json.dumps({'error':repr(exc),'scenario':name,'n':n,'rep':rep,'r':r,'fold':k},indent=2),encoding='utf-8')
        raise
    trials=pd.DataFrame(results); tuning=pd.DataFrame(candidates); diag=pd.DataFrame(diagnostics)
    trials.to_csv(OUT/'trials.csv',index=False); tuning.to_csv(OUT/'tuning.csv',index=False)
    diag.to_csv(OUT/'fold_diagnostics.csv',index=False)
    np.savez_compressed(OUT/'fold_coefficients.npz',coefficients=np.stack(coefficients),keys=np.array(keys))
    summary=[]
    for (name,n,method),g in trials.groupby(['scenario','n','method'],sort=True):
        delta=g.ise-g.full_refit_ise
        summary.append({'scenario':name,'n':int(n),'method':method,'cases':len(g),'mise':g.ise.mean(),
            'mcse':g.ise.std(ddof=1)/np.sqrt(len(g)),'minus_full_refit':delta.mean(),
            'paired_mcse':delta.std(ddof=1)/np.sqrt(len(g)),'r0_fraction':g.uniform.mean(),
            'rmax_fraction':g.boundary.mean()})
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    meta.update(completed=True,cases=len(keys),trials=len(trials),cv_candidates=len(tuning),
        fold_fits=len(diag),failures=0,seconds=time.perf_counter()-started,
        tail_note='MISE is Fourier sum through H=256; omitted nonnegative truth tail is common to compared methods, negligible except where checked explicitly.')
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1): main()
