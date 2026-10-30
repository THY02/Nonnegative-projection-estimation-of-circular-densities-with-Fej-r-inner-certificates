"""Post-hoc penalty sensitivity using saved CV candidates and fold coefficients."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from certified_experiment import NAMES,truth,H
from fair_baselines import RGRID

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/weakness_cv_penalty';RHO=[0.,.05,.1,.25,.5,1.]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    paths=['research/weakness_cv_penalty_protocol.md','code/weakness_cv_penalty.py',
        'results/cv_ensemble_sensitivity/tuning.csv','results/cv_ensemble_sensitivity/fold_coefficients.npz',
        'results/cv_ensemble_sensitivity/trials.csv']
    original=pd.read_csv(ROOT/paths[-1]).set_index(['scenario','n','rep','method'])
    tuning=pd.read_csv(ROOT/paths[2]).set_index(['scenario','n','rep','r'])
    data=np.load(ROOT/paths[3]);coeff=data['coefficients'];keys=data['keys']
    assert coeff.shape==(480,len(RGRID),5,max(RGRID)+1)
    rows=[]
    try:
        for ix,(si,n,rep) in enumerate(keys):
            si,n,rep=map(int,(si,n,rep));name=NAMES[si];target=truth(name)
            sub=tuning.loc[(name,n,rep)].reindex(RGRID)
            assert len(sub)==len(RGRID) and not sub[['score','penalty']].isna().any().any()
            score=sub.score.to_numpy();penalty=sub.penalty.to_numpy()
            for rho in RHO:
                j=int(np.argmin(score+rho*penalty));r=RGRID[j]
                a=coeff[ix,j].mean(axis=0);a=np.pad(a,(0,H+1-len(a)))
                ise=float(np.sum(abs(a[1:]-target[1:])**2)/np.pi)
                if rho in (0.,1.):
                    method='ensemble_cv' if rho==0 else 'ensemble_penalized'
                    prior=original.loc[(name,n,rep,method)]
                    assert r==int(prior.r) and abs(ise-prior.ise)<1e-12
                rows.append({'scenario':name,'n':n,'rep':rep,'rho':rho,'r':r,'ise':ise,
                    'validation_score':score[j],'penalty':penalty[j],
                    'chosen_criterion':score[j]+rho*penalty[j],'r0':r==0,'rmax':r==max(RGRID)})
    except Exception as exc:
        pd.DataFrame(rows).to_csv(OUT/'partial.csv',index=False)
        (OUT/'failure.json').write_text(json.dumps({'error':repr(exc),'scenario':name,'n':n,'rep':rep,'rho':rho},indent=2),encoding='utf-8')
        raise
    df=pd.DataFrame(rows);df.to_csv(OUT/'trials.csv',index=False)
    sums=[]
    for (name,n),g in df.groupby(['scenario','n'],sort=True):
        ref=g[g.rho==0].set_index('rep').ise
        for rho,z in g.groupby('rho',sort=True):
            values=z.set_index('rep').ise;difference=values-ref
            sums.append({'scenario':name,'n':n,'rho':rho,'cases':len(values),'mise':values.mean(),
                'mcse':values.std(ddof=1)/np.sqrt(len(values)),'minus_unpenalized':difference.mean(),
                'paired_mcse':difference.std(ddof=1)/np.sqrt(len(values)),
                'r0_fraction':z.r0.mean(),'rmax_fraction':z.rmax.mean()})
    pd.DataFrame(sums).to_csv(OUT/'summary.csv',index=False)
    meta={'completed':True,'source_samples':len(keys),'new_samples':0,'rho':RHO,'cases':len(rows),
        'failures':0,'inputs_sha256':{p:sha(ROOT/p) for p in paths},
        'interpretation':'Post-hoc sensitivity; no oracle guarantee for rho < 1 and no universal ranking.'}
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
