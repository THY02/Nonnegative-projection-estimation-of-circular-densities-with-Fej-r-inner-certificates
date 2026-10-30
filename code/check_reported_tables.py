"""Check the English manuscript tables against saved summaries without rerunning fits."""
from pathlib import Path
import csv, re, statistics
R=Path(__file__).resolve().parents[1]
TEX=R/'manuscript'/'jns_manuscript.tex'
def csvrows(path):
    with (R/path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def table(n):
    s=TEX.read_text(encoding='utf-8').split(r'\label{tab:t'+str(n)+'}',1)[1]
    s=s.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
    return [[c.strip() for c in row.split('&')] for row in s.split(r'\\') if '&' in row]
def close(a,b,tol=5.1e-7):
    if abs(float(a)-float(b))>tol:raise AssertionError((a,b,tol))
name={'Smooth':'smooth','Bimodal':'bimodal','Narrow':'narrow','Boundary':'boundary','Boundary-crossing':'boundary','Uniform':'uniform','Cardioid':'zero_cardioid','Cardioid with zero':'zero_cardioid'}
def by(rows,*keys):return {tuple(x[k] for k in keys):x for x in rows}
base=by(csvrows('results/fair_baselines_20261104/summary.csv'),'scenario','n','method')
for row in table(1):
    label,n,*v=row
    for method,val in zip(['certified','grid_repair_cv','sdp_cv'],v):close(val,base[(name[label],n,method)]['mise'])
diagnostics=by(csvrows('results/certificate_bound_diagnostics/summary.csv'),'scenario','n')
for label,n,count,ratio,direct,ball,tie in table(2):
    x=diagnostics[(name[label],n)];assert int(count)==int(x['active'])
    for a,k in zip(re.findall(r'\d+(?:\.\d+)?',ratio),['ratio_median','ratio_q1','ratio_q3']):close(a,x[k])
    for a,k in zip([direct,ball,tie],['direct_fraction','ball_fraction','tie_fraction']):close(a.replace(r'\%',''),100*float(x[k]),.051)
target=by(csvrows('results/v8_targeted/summary.csv'),'scenario','n','batch','method')
for label,n,*v in table(3):
    for method,val in zip(['certified','grid_repair_cv','sdp_cv','fejer_kde'],v):close(val,target[(name[label],n,'pooled60',method)]['mise'])
pen=by(csvrows('results/v7_envelope_expansion/summary.csv'),'scenario','n','method')
for label,n,ordinary,fixed,local,diff in table(4):
    for method,val in zip(['ordinary','fixed','local'],[ordinary,fixed,local]):close(val,pen[(name[label],n,method)]['mise'])
    delta,se=re.findall(r'-?\d+\.\d+',diff);x=pen[(name[label],n,'local')];close(delta,x['fixed_delta']);close(se,x['fixed_paired_mcse'])
times=csvrows('results/benchmark_v12/runs.csv')
for r,D,tc,sc,ts,ss,gap,bound in table(5):
    a=[x for x in times if x['r']==r and x['method']=='certificate'];b=[x for x in times if x['r']==r and x['method']=='sdp']
    assert len(a)==len(b)==3 and all(x['status']=='ok' for x in a) and all(x['status']=='optimal' for x in b)
    for val,vec,tol in [(D,[float(x['D']) for x in a],0),(tc,[float(x['seconds']) for x in a],.000051),(sc,[float(x['solver_seconds']) for x in a],.000051),(ts,[float(x['seconds']) for x in b],.000051),(ss,[float(x['solver_seconds']) for x in b],.000051),(bound,[float(x['bound']) for x in a],.00000051)]:close(val,statistics.median(vec),tol)
    aa={x['rep']:float(x['objective']) for x in a};bb={x['rep']:float(x['objective']) for x in b};close(gap,statistics.median(aa[k]-bb[k] for k in aa),.00000051)
app=by(csvrows('results/binned_sdp_baseline/summary.csv'),'method')
methods=['certified_bins','sdp_bins','fixed_2r_bins','circular_smooth','cyclic_penalty','histogram','fejer_mix','dvp_mix']
for row,method in zip(table(6),methods):
    _,squared,log=row;x=app[(method,)];close(squared,x['mean_squared_share_error'],5.1e-10);close(log,x['mean_log_score'])
print('All six manuscript tables agree with saved summaries to displayed precision.')
print('This checks saved records, not independent reruns or exact arithmetic.')
