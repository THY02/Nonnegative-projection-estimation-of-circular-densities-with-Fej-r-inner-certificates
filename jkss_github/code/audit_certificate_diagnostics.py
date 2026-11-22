"""Audit numerical summaries and immutable input hashes for certificate replay."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1];P=R/'results/certificate_bound_diagnostics'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    m=json.loads((P/'manifest.json').read_text(encoding='utf-8'))
    for file,key in [('research/42_certificate_diagnostics_protocol.md','protocol_sha256'),
      ('code/diagnose_certificate_bounds.py','source_sha256'),('code/certified_projection.py','engine_sha256'),
      ('results/precision_tradeoff/trials.csv','prior_sha256')]: assert sha(R/file)==m[key]
    cases=pd.read_csv(P/'cases.csv');summ=pd.read_csv(P/'summary.csv')
    assert m['completed'] and len(cases)==480 and len(summ)==12
    assert not cases.duplicated(['scenario','n','rep']).any()
    prior=pd.read_csv(R/'results/precision_tradeoff/trials.csv');prior=prior[prior.eta==.1]
    merged=cases.merge(prior,on=['scenario','n','rep'],validate='one_to_one',suffixes=('','_old'))
    assert len(merged)==480 and (merged.r==merged.r_old).all() and (merged.D==merged.D_old).all()
    assert np.max(abs(merged.ise-merged.ise_old))<1e-10
    assert np.max(abs(merged.bound-merged.l2_squared_bound))<1e-10
    assert (cases.bound<=cases.target+1e-14).all()
    for x in summ.itertuples(index=False):
        g=cases[(cases.scenario==x.scenario)&(cases.n==x.n)];active=g[g.r>0]
        assert len(g)==40 and len(active)==x.active and int((g.r==0).sum())==x.r0_count
        assert np.allclose([active.ratio.quantile(.25),active.ratio.median(),active.ratio.quantile(.75)],
          [x.ratio_q1,x.ratio_median,x.ratio_q3],rtol=0,atol=1e-12)
        tol=1e-12*np.maximum(abs(active.direct_bound),abs(active.ball_bound))
        tied=abs(active.direct_bound-active.ball_bound)<=tol
        dr=(active.direct_bound<active.ball_bound)&(~tied);ball=(active.ball_bound<active.direct_bound)&(~tied)
        assert (int(dr.sum()),int(ball.sum()),int(tied.sum()))==(x.direct_wins,x.ball_wins,x.ties)
        assert np.max(abs(active.bound-np.minimum(active.direct_bound,active.ball_bound)))<1e-14
    active=cases[cases.r>0];ball=active[active.selected=='ball'];stop=ball[ball.direct_bound>ball.target]
    audit={'passed':True,'cases':480,'active_cases':len(active),'r0_cases':int((cases.r==0).sum()),
      'direct_wins':int((active.selected=='direct').sum()),'ball_wins':len(ball),
      'ties':int((active.selected=='tie').sum()),'ball_stopped_while_direct_above_target':len(stop),
      'median_relative_reduction_when_ball_wins':float(np.median(1-ball.ball_bound/ball.direct_bound)),
      'max_ise_replay_difference':float(np.max(abs(merged.ise-merged.ise_old))),
      'max_bound_replay_difference':float(np.max(abs(merged.bound-merged.l2_squared_bound))),
      'scope':'post-hoc final fits; numerical diagnostics, not strict machine certification'}
    (R/'results/certificate_bound_diagnostics_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    labels={'smooth':'平滑','bimodal':'双峰','narrow':'尖峰','boundary':'跨边界','uniform':'均匀','zero_cardioid':'零点心形'}
    lines=['# 停止时表示次数和双误差界：同样本重放','',
      '日期：2026-10-27。复用原 480 个已观察样本并重放 η=0.1 的全样本最终拟合；r、D、ISE 与保存界重放差均为零。按预先写定的重放方案统计；后述“仅直接界不能停止”的计数及相对缩减为事后补充描述。','',
      '| 场景 | n | 非零 r / 全部 | D/r 中位数（Q1，Q3） | 直接界更小 | 球界更小 | 并列 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for x in summ.itertuples(index=False):
      lines.append(f'| {labels[x.scenario]} | {x.n} | {x.active}/{x.cases} | {x.ratio_median:g}（{x.ratio_q1:g}，{x.ratio_q3:g}） | {100*x.direct_fraction:.1f}% | {100*x.ball_fraction:.1f}% | {100*x.tie_fraction:.1f}% |')
    lines+=['',
      f'非零 r 共 {len(active)} 次：直接界较小 {audit["direct_wins"]} 次、球界较小 {len(ball)} 次、数值并列 {audit["ties"]} 次；另有 {audit["r0_cases"]} 次 r=0，不计算 D/r 和界比较。Q1、Q3采用线性插值，表内比例以每格非零 r 次数为分母；并列标准为两界差不超过较大界的 10⁻¹² 倍。',
      '',f'事后补充：{len(stop)} 次拟合在实际停止的 D 上，球界已小于等于阈值但直接界仍大于阈值；若只用直接界，这些拟合在该 D 不能停止。球界较小的 {len(ball)} 次中，其相对直接界的数值缩减中位数约 {100*audit["median_relative_reduction_when_ball_wins"]:.1f}%；此比例以球界较小的案例为条件。',
      '', '这些结果反映本次特定数据及实现，不能证明两界在所有问题都同时必要、得到该界一定节省运行时间、或浮点程序的严格可行性。逐次结果、代码与输入哈希见 `results/certificate_bound_diagnostics/`；独立汇总审计见 `results/certificate_bound_diagnostics_audit.json`。']
    (R/'research/43_certificate_diagnostics_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
