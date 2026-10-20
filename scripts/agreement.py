#!/usr/bin/env python3
"""Independent 50-case evaluator agreement for AIPI totals and six ACCS domains.
Linearly weighted Cohen kappa; percentile bootstrap resampling of complete cases.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


def linear_weighted_kappa(x, y, lo, hi):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask].astype(int), y[mask].astype(int)
    if len(x) < 2: return np.nan
    if np.any((x < lo) | (x > hi) | (y < lo) | (y > hi)):
        raise ValueError("Agreement score outside declared scale")
    classes = np.arange(lo, hi + 1)
    obs = np.array([[np.sum((x==i)&(y==j)) for j in classes] for i in classes], dtype=float)
    obs /= len(x)
    expected = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    weights = np.abs(classes[:,None]-classes[None,:]) / (hi-lo)
    denom = float((expected * weights).sum())
    return float(1 - (obs * weights).sum() / denom) if denom > 0 else np.nan


def percentile_case_bootstrap(frame, x, y, lo, hi, n_boot, seed, pooled=False):
    # Case-level resampling keeps all 3 repeated outputs within each case if pooled.
    grouped = [g[[x,y]].to_numpy(dtype=float) for _, g in frame.groupby('case_id', sort=False)]
    sizes = {len(g) for g in grouped}
    if len(sizes) != 1: raise ValueError('Expected equal numbers of observations per case')
    stacked = np.stack(grouped)
    rng = np.random.default_rng(seed)
    vals=[]
    for _ in range(n_boot):
        sample = stacked[rng.integers(0,len(stacked),len(stacked))].reshape(-1,2)
        vals.append(linear_weighted_kappa(sample[:,0], sample[:,1], lo, hi))
    if not np.isfinite(vals).any(): return (np.nan, np.nan)
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def analyze(input_csv, outdir, n_boot=10000, seed=20261020):
    df=pd.read_csv(input_csv)
    required=['case_id','repetition','consensus_total','third_total'] + [f'{f}_accs_{i}' for f in ('consensus','third') for i in range(1,7)]
    if any(c not in df for c in required): raise ValueError(f'Missing columns: {set(required)-set(df.columns)}')
    if df.duplicated(['case_id','repetition']).any(): raise ValueError('Duplicate case/repetition')
    allrows=[]
    for rep in (1,2,3):
        d=df.loc[df['repetition']==rep]
        if len(d) == 0: raise ValueError(f'No repetition {rep}')
        x,y = d['consensus_total'],d['third_total']
        ci=percentile_case_bootstrap(d,'consensus_total','third_total',0,20,n_boot,seed+rep)
        allrows.append({'analysis':'AIPI','repetition':rep,'domain':'Total score','n_cases':len(d),
                        'linear_weighted_kappa':linear_weighted_kappa(x,y,0,20),
                        'ci_low':ci[0],'ci_high':ci[1],
                        'exact_agreement_pct':float((x==y).mean()*100),
                        'within_one_point_pct':float(((x-y).abs()<=1).mean()*100)})
    ci=percentile_case_bootstrap(df,'consensus_total','third_total',0,20,n_boot,seed+100,pooled=True)
    x,y=df['consensus_total'],df['third_total']
    allrows.append({'analysis':'AIPI','repetition':'pooled','domain':'Total score','n_cases':df['case_id'].nunique(),
                    'linear_weighted_kappa':linear_weighted_kappa(x,y,0,20),'ci_low':ci[0],'ci_high':ci[1],
                    'exact_agreement_pct':float((x==y).mean()*100),
                    'within_one_point_pct':float(((x-y).abs()<=1).mean()*100)})
    d=df.loc[df['repetition']==1].copy()
    domains=['Previous history/actual context','Problem presented','Communication','Physical examination','Patient management','Prevention']
    for i, domain in enumerate(domains,1):
        xcol,ycol=f'consensus_accs_{i}',f'third_accs_{i}'
        x,y=d[xcol],d[ycol]
        ci=percentile_case_bootstrap(d,xcol,ycol,1,5,n_boot,seed+200+i)
        allrows.append({'analysis':'ACCS','repetition':'case-level','domain':domain,'n_cases':len(d),
                        'linear_weighted_kappa':linear_weighted_kappa(x,y,1,5),'ci_low':ci[0],'ci_high':ci[1],
                        'exact_agreement_pct':float((x==y).mean()*100),
                        'within_one_point_pct':float(((x-y).abs()<=1).mean()*100)})
    outdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(allrows).to_csv(outdir/'16_AIPI_ACCS_independent_agreement.csv',index=False)
    (outdir/'16_agreement_audit.json').write_text(json.dumps({'n_input_rows':len(df),'n_cases':int(df.case_id.nunique()),'bootstraps':n_boot,'seed':seed},indent=2))
    return pd.DataFrame(allrows)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--out',type=Path,default=Path('results'))
    p.add_argument('--bootstrap',type=int,default=10000)
    p.add_argument('--seed',type=int,default=20261020)
    a=p.parse_args()
    print(analyze(a.input,a.out,a.bootstrap,a.seed).to_string(index=False))
