"""Independent sample experiment; protocol in research/08. No truth tuning."""
from pathlib import Path
import argparse, hashlib, json, time, sys
import numpy as np
import pandas as pd
from scipy.special import ive
from threadpoolctl import threadpool_limits
from circular import TAU, moments, l2score, fit_continuous, density
from simulation import SCENARIOS, sample, truth_coefficients
from separated_degree import fit as fixed_fit
from certified_projection import fit as certified_fit

ROOT=Path(__file__).resolve().parents[1]
DEGREES=[0,1,2,3,4,6,8,12,16,24,32]
FEJER=[0,1,2,3,4,6,8,12,16,24,32,48,64,96,128]
KAPPAS=[0.,.5,1.,2.,4.,8.,16.,32.,64.,128.,256.]
NAMES=list(SCENARIOS)+['uniform','zero_cardioid']
H=256


def grid_values(a, size=8192):
    freq=np.zeros(size,dtype=complex)
    freq[:len(a)]=a
    if len(a)>1:freq[-len(a)+1:]=a[:0:-1].conj()
    return np.fft.ifft(freq).real*size/TAU


def corrected(a):
    h=grid_values(a)
    u=np.sort(h)[::-1]
    s=(np.cumsum(u)-len(h)/TAU)/np.arange(1,len(h)+1)
    j=np.flatnonzero(u>s)[-1]
    threshold=float(s[j])
    values=np.maximum(h-threshold,0)
    return threshold,values


def draw(name,n,rng):
    if name=='uniform':return rng.uniform(0,TAU,n)
    if name=='zero_cardioid':
        accepted=[]
        while sum(len(a) for a in accepted)<n:
            y=rng.uniform(0,TAU,2*n)
            accepted.append(y[rng.uniform(0,2,2*n)<=1+np.cos(y-.7)])
        return np.concatenate(accepted)[:n]
    return sample(name,n,rng)


def truth(name):
    if name not in ['uniform','zero_cardioid']:return truth_coefficients(name,H)
    a=np.zeros(H+1,complex);a[0]=1
    if name=='zero_cardioid':a[1]=.5*np.exp(-.7j)
    return a


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=ROOT/'research'/'08_certified_projection_protocol.md'
    provenance=dict(seed=20261029,repetitions=50,folds=5,eta=.1,scenarios=NAMES,
        degrees=DEGREES,fejer_degrees=FEJER,kappas=KAPPAS,
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        engine_sha256=hashlib.sha256((ROOT/'code'/'certified_projection.py').read_bytes()).hexdigest(),
        date='2026-10-27',stage='Independent numerical validation; not proof of originality or acceptance.')
    (out/'started.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    rows=[];tuning=[];certificates=[];start=time.perf_counter()
    max_mass_error=0.;max_kkt=0.;min_weight=1.
    methods={'certified':DEGREES,'fixed_2r':DEGREES,'raw_fourier':DEGREES,
             'threshold_fourier':DEGREES,'fejer':FEJER,'von_mises':KAPPAS}
    try:
        for si,name in enumerate(NAMES):
            true=truth(name);truegrid=grid_values(true)
            for n in [100,500]:
                for rep in range(50):
                    rng=np.random.default_rng(np.random.SeedSequence([20261029,si,n,rep]))
                    x=draw(name,n,rng);folds=np.array_split(rng.permutation(n),5)
                    cv=[]
                    for val in folds:
                        train=np.setdiff1d(np.arange(n),val)
                        cv.append((moments(x[train],H),moments(x[val],H),x[val],len(train)))
                    full=moments(x,H)
                    for method,parameters in methods.items():
                        scores=[]
                        for par in parameters:
                            score=0.
                            for fold,(ct,cvv,xv,nt) in enumerate(cv):
                                if method=='threshold_fourier':
                                    a=ct[:int(par)+1];tau,h=corrected(a)
                                    contribution=TAU*np.mean(h*h)-2*np.maximum(density(a,xv)-tau,0).mean()
                                else:
                                    if method=='certified':
                                        a,diag=certified_fit(ct,int(par),nt)
                                        certificates.append(dict(scenario=name,n=n,rep=rep,fold=fold,
                                            **{k:v for k,v in diag.items() if k!='history'}))
                                        max_kkt=max(max_kkt,diag['max_kkt']);min_weight=min(min_weight,diag['minimum_weight'])
                                    elif method=='fixed_2r':
                                        a=np.array([1.+0j]) if par==0 else fixed_fit(ct,int(par),2*int(par))[0]
                                    else:a=fit_continuous(ct,method,par).coefficients
                                    contribution=l2score(a,cvv)
                                score+=len(xv)/n*contribution
                            scores.append(float(score))
                            tuning.append(dict(scenario=name,n=n,rep=rep,method=method,parameter=par,cv_score=score))
                        chosen=int(np.argmin(scores));par=parameters[chosen]
                        extra={}
                        if method=='threshold_fourier':
                            a=full[:int(par)+1];tau,h=corrected(a)
                            ise=float(TAU*np.mean((h-truegrid)**2))
                            mass_error=float(abs(TAU*np.mean(np.maximum(grid_values(a,16384)-tau,0))-1))
                            max_mass_error=max(max_mass_error,mass_error)
                            extra=dict(threshold=tau,refined_mass_error=mass_error)
                        else:
                            if method=='certified':
                                a,diag=certified_fit(full,int(par),n)
                                certificates.append(dict(scenario=name,n=n,rep=rep,fold=-1,
                                    **{k:v for k,v in diag.items() if k!='history'}))
                                extra={k:v for k,v in diag.items() if k!='history'}
                                max_kkt=max(max_kkt,diag['max_kkt']);min_weight=min(min_weight,diag['minimum_weight'])
                            elif method=='fixed_2r':
                                a=np.array([1.+0j]) if par==0 else fixed_fit(full,int(par),2*int(par))[0]
                            else:a=fit_continuous(full,method,par).coefficients
                            padded=np.pad(a,(0,H+1-len(a)))
                            ise=float(np.sum(abs(padded[1:]-true[1:])**2)/np.pi)
                        rows.append(dict(scenario=name,n=n,rep=rep,method=method,parameter=par,
                            ise=ise,cv_score=scores[chosen],upper_boundary=chosen==len(parameters)-1,**extra))
                pd.DataFrame(rows).to_csv(out/'trials.csv',index=False)
                pd.DataFrame(tuning).to_csv(out/'tuning.csv',index=False)
                pd.DataFrame(certificates).to_csv(out/'certificates.csv',index=False)
                print(f'completed {name} n={n}; {time.perf_counter()-start:.1f}s',flush=True)
    except Exception as exc:
        pd.DataFrame(rows).to_csv(out/'trials.csv',index=False)
        pd.DataFrame(tuning).to_csv(out/'tuning.csv',index=False)
        pd.DataFrame(certificates).to_csv(out/'certificates.csv',index=False)
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),scenario=name,n=n,rep=rep,
            method=method,parameter=par),indent=2),encoding='utf-8')
        raise
    df=pd.DataFrame(rows);summary=[]
    for (name,n),group in df.groupby(['scenario','n'],sort=False):
        ref=group[group.method=='certified'].set_index('rep').ise
        for method,z in group.groupby('method',sort=False):
            v=z.set_index('rep').ise;delta=ref-v
            summary.append(dict(scenario=name,n=n,method=method,mise=v.mean(),mcse=v.std(ddof=1)/np.sqrt(len(v)),
                certified_minus_method=delta.mean(),paired_mcse=delta.std(ddof=1)/np.sqrt(len(v)),
                upper_boundary_fraction=z.upper_boundary.mean()))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    tail=2*np.sum(ive(np.arange(H+1,1025),256)/ive(0,256))/TAU
    provenance.update(completed=True,final_fits=len(rows),cv_candidates=len(tuning),
        certificate_fits=len(certificates),failures=0,max_threshold_refined_mass_error=max_mass_error,
        max_kkt=max_kkt,minimum_weight=min_weight,vm_tail_density_bound=float(tail),
        seconds=time.perf_counter()-start,python=sys.version)
    (out/'manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    with threadpool_limits(limits=1):main(args.output)
