"""Audit recorded precision experiment against its source and previous results."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1]; p=R/'results'/'precision_tradeoff'
h=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
m=json.loads((p/'manifest.json').read_text()); a=pd.read_csv(p/'trials.csv'); c=pd.read_csv(p/'tuning.csv'); t=pd.read_csv(p/'timings.csv'); s=pd.read_csv(p/'summary.csv'); f=pd.read_csv(p/'failures.csv')
assert h(R/'research'/'25_precision_tradeoff_protocol.md')==m['protocol_sha256']
assert h(R/'code'/'precision_tradeoff.py')==m['source_sha256']
assert h(R/'results'/'fair_baselines_20261104'/'trials.csv')==m['original_trial_sha256']
assert len(a)==1920 and len(c)==15360 and len(t)==600 and len(s)==48 and len(f)==0
assert not a.duplicated(['scenario','n','rep','eta']).any()
assert not c.duplicated(['scenario','n','rep','eta','r']).any()
assert not t.duplicated(['scenario','n','rep','method']).any()
assert (a.status=='ok').all() and (t.status!='failed').all()
assert (a.l2_squared_bound<=a.target_bound+1e-12).all() and (a.minimum_weight>=-1e-10).all()
assert (a[a.r>0].D/a[a.r>0].r<=256).all()
key=['scenario','n','rep','eta']; chosen=c.loc[c.groupby(key,sort=False).cv_score.idxmin()]
joined=a.merge(chosen,on=key,suffixes=('','_tune'),validate='one_to_one')
assert (joined.r==joined.r_tune).all() and np.allclose(joined.chosen_score,joined.cv_score,atol=1e-12,rtol=0)
old=pd.read_csv(R/'results'/'fair_baselines_20261104'/'trials.csv')
oldcert=old[old.method=='certified'][['scenario','n','rep','r','ise']]
rep=a[a.eta==.1].merge(oldcert,on=['scenario','n','rep'],suffixes=('','_old'),validate='one_to_one')
assert len(rep)==480 and (rep.r==rep.r_old).all() and (abs(rep.ise-rep.ise_old)<1e-10).all()
oldsdp=old[old.method=='sdp_cv'][['scenario','n','rep','r','ise']]
paired=a.merge(oldsdp,on=['scenario','n','rep'],suffixes=('','_old'),validate='many_to_one')
assert (abs(paired.paired_sdp_ise-paired.ise_old)<1e-12).all()
test=t.merge(oldsdp,on=['scenario','n','rep'],suffixes=('','_old'),validate='many_to_one')
assert (test.r==test.r_old).all()
for row in s.itertuples(index=False):
    g=a[(a.scenario==row.scenario)&(a.n==row.n)&(a.eta==row.eta)]
    assert len(g)==40 and abs(g.ise.mean()-row.mise)<1e-12
    assert abs(g.ise.std(ddof=1)/np.sqrt(40)-row.mcse)<1e-12
    assert abs((g.ise-g.paired_sdp_ise).mean()-row.eta_minus_sdp)<1e-12
    assert abs((g.ise-g.paired_sdp_ise).std(ddof=1)/np.sqrt(40)-row.paired_mcse)<1e-12
out=dict(passed=True,certificate_rows=len(a),cv_candidate_rows=len(c),benchmark_rows=len(t),failures=len(f),
    original_eta_point1_max_absolute_ise_difference=float(abs(rep.ise-rep.ise_old).max()),
    original_eta_point1_r_mismatches=int((rep.r!=rep.r_old).sum()),
    max_recorded_bound_excess=float(t.bound_excess.max()),
    excess_over_1e_11=int((t.bound_excess>1e-11).sum()),
    over_prespecified_numerical_tolerance=int((t.bound_excess>2e-7).sum()),
    checks='hashes, unique keys, statuses, caps, finite-precision bound, selected CV minimum, prior eta=.1, paired SDP/r, 48 summary means and MCSE')
path=R/'results'/'precision_tradeoff_audit.json'
if path.exists():
    if json.loads(path.read_text(encoding='utf-8')) != out:
        raise RuntimeError('Existing audit result differs from newly calculated result')
else:
    path.write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
