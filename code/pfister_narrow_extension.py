"""Post-hoc enlarged Pfister representation grid on previously used samples."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from circular import moments,l2score
from certified_experiment import NAMES,draw,truth
from fair_baselines import RGRID
from pfister_cv_comparison import estimate

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/pfister_narrow_extension';PREV=ROOT/'results/pfister_cv_comparison_v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False);prev=pd.read_csv(PREV/'trials.csv');prev=prev[prev.scenario=='narrow'].set_index(['n','rep'])
    pre_tuning=pd.read_csv(PREV/'tuning.csv'); pre_tuning=pre_tuning[pre_tuning.scenario=='narrow']
    meta={'stage':'Post-hoc expansion after observing narrow results','seed':20261104,'replications_per_cell':8,
        'rgrid':RGRID,'additional_ratios':[32,64],'full_ratios':[2,4,8,16,32,64],'folds':5,
        'protocol_sha256':sha(ROOT/'research/39_pfister_narrow_extension_protocol.md'),
        'source_sha256':sha(Path(__file__)),'engine_sha256':sha(ROOT/'code/pfister_cv_comparison.py'),
        'previous_trials_sha256':sha(PREV/'trials.csv'),'previous_tuning_sha256':sha(PREV/'tuning.csv')}
    (OUT/'started.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    records=[];candidates=[];diagnostics=[];failures=[]; start=time.perf_counter();name='narrow';true=truth(name)
    for n in [100,500]:
        for rep in range(8):
            rng=np.random.default_rng(np.random.SeedSequence([meta['seed'],NAMES.index(name),n,rep]))
            x=draw(name,n,rng);folds=np.array_split(rng.permutation(n),5);cv=[]
            for val in folds:
                train=np.setdiff1d(np.arange(n),val)
                cv.append((moments(x[train],max(RGRID)),moments(x[val],max(RGRID)),len(val)))
            old=pre_tuning[(pre_tuning.n==n)&(pre_tuning.rep==rep)].sort_values(['r','ratio'])
            assert len(old)==29
            scores=[(float(v.cv_score),int(v.r),int(v.ratio)) for v in old.itertuples(index=False)]
            failed=False
            for r in RGRID[1:]:
                for ratio in [32,64]:
                    score=0.
                    for k,(train,valid,nv) in enumerate(cv):
                        try:a,info=estimate(train,r,ratio)
                        except Exception as exc:
                            failures.append({'n':n,'rep':rep,'stage':'cv','r':r,'ratio':ratio,'fold':k,'error':repr(exc)});failed=True;break
                        score+=nv/n*l2score(a,valid);diagnostics.append({'n':n,'rep':rep,'r':r,'ratio':ratio,'fold':k,**info})
                    if failed:break
                    candidates.append({'n':n,'rep':rep,'r':r,'ratio':ratio,'cv_score':score});scores.append((score,r,ratio))
                if failed:break
            if failed:
                records.append({'n':n,'rep':rep,'status':'failed_cv'});continue
            score,r,ratio=min(scores,key=lambda z:z[0]); c=moments(x,max(RGRID))
            try:a,info=estimate(c,r,ratio)
            except Exception as exc:
                failures.append({'n':n,'rep':rep,'stage':'refit','r':r,'ratio':ratio,'fold':-1,'error':repr(exc)})
                records.append({'n':n,'rep':rep,'status':'failed_refit'});continue
            padded=np.pad(a,(0,len(true)-len(a)));ise=float(np.sum(abs(padded[1:]-true[1:])**2)/np.pi)
            oldrow=prev.loc[(n,rep)]; solver_status=info.pop('status')
            records.append({'n':n,'rep':rep,'status':'ok','solver_status':solver_status,'r':r,'ratio':ratio,'ise':ise,
                'cv_score':score,'old_pfister_ise':float(oldrow.ise),'old_pfister_ratio':int(oldrow.ratio),
                'old_pfister_r':int(oldrow.r),'fejer_ise':float(oldrow.certified_ise),
                'r_boundary':r==max(RGRID),'ratio_boundary':ratio==64,**info})
            print(f'finished n={n} rep={rep} time={time.perf_counter()-start:.1f}s',flush=True)
            pd.DataFrame(records).to_csv(OUT/'partial_trials.csv',index=False)
            pd.DataFrame(failures).to_csv(OUT/'partial_failures.csv',index=False)
    trials=pd.DataFrame(records);trials.to_csv(OUT/'trials.csv',index=False)
    pd.DataFrame(candidates).to_csv(OUT/'new_tuning.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'new_fold_diagnostics.csv',index=False)
    pd.DataFrame(failures,columns=['n','rep','stage','r','ratio','fold','error']).to_csv(OUT/'failures.csv',index=False)
    summ=[]
    for n,g in trials.groupby('n'):
        ok=g[g.status=='ok']; row={'n':int(n),'attempted':len(g),'success':len(ok),'failures':len(g)-len(ok)}
        if len(ok):
            row.update(mise=ok.ise.mean(),mcse=ok.ise.std(ddof=1)/np.sqrt(len(ok)),
                r_boundary=ok.r_boundary.mean(),ratio_boundary=ok.ratio_boundary.mean())
            for method in ['old_pfister','fejer']:
                diff=ok.ise-ok[method+'_ise'];row[method+'_delta']=diff.mean();row[method+'_paired_mcse']=diff.std(ddof=1)/np.sqrt(len(diff))
        summ.append(row)
    pd.DataFrame(summ).to_csv(OUT/'summary.csv',index=False)
    meta.update(completed=True,attempted=len(trials),success=int((trials.status=='ok').sum()),failures=len(failures),
        new_candidates=len(candidates),new_fold_fits=len(diagnostics),seconds=time.perf_counter()-start)
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
