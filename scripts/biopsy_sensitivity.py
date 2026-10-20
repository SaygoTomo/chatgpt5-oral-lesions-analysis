#!/usr/bin/env python3
"""Descriptive biopsy-recorded sensitivity analysis for correctness (I5==3).
Case-cluster bootstrap intervals for pooled repeated case responses.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm


def wilson(k,n,z=norm.ppf(0.975)):
    if not n: return (np.nan,np.nan)
    p=k/n; d=1+z*z/n
    a=(p+z*z/(2*n))/d
    b=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return float(a-b),float(a+b)


def analysis(df,n_boot=10000,seed=20261020):
    necessary={'case_id','repetition','I5','biopsy_recorded'}
    if not necessary.issubset(df):raise ValueError(f'Missing {necessary-set(df)}')
    if df.duplicated(['case_id','repetition']).any():raise ValueError('Case-repetition duplicates')
    df=df.copy()
    df['correct'] = pd.to_numeric(df['I5'], errors='coerce').eq(3).astype(float)
    df.loc[pd.to_numeric(df['I5'],errors='coerce').isna(),'correct']=np.nan
    rng=np.random.default_rng(seed)
    rows=[]
    for label,sub in [('All cases',df),('Biopsy recorded',df.loc[df.biopsy_recorded==1]),('No biopsy recorded',df.loc[df.biopsy_recorded==0])]:
        groups=[g['correct'].to_numpy() for _,g in sub.groupby('case_id',sort=True)]
        n_cases=len(groups)
        if not n_cases:continue
        for rep in [1,2,3]:
            dat=sub.loc[sub.repetition==rep,'correct'].dropna()
            k=int(dat.sum());n=int(len(dat))
            low,high=wilson(k,n)
            rows.append({'group':label,'repetition':rep,'n_cases':n_cases,'correct':k,'n_outputs':n,'accuracy':k/n,'ci_low':low,'ci_high':high,'interval_method':'Wilson'})
        valid=sub['correct'].dropna()
        obs=float(valid.mean())
        boots=[]
        for _ in range(n_boot):
            ids=rng.integers(0,len(groups),len(groups))
            arr=np.concatenate([groups[j] for j in ids])
            boots.append(float(np.nanmean(arr)))
        low,high=np.nanpercentile(boots,[2.5,97.5])
        rows.append({'group':label,'repetition':'pooled','n_cases':n_cases,'correct':int(valid.sum()),'n_outputs':int(len(valid)),
                     'accuracy':obs,'ci_low':float(low),'ci_high':float(high),'interval_method':'case-cluster bootstrap'})
    return pd.DataFrame(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--out',type=Path,default=Path('results'))
    p.add_argument('--bootstrap',type=int,default=10000)
    p.add_argument('--seed',type=int,default=20261020)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    tab=analysis(pd.read_csv(a.input),a.bootstrap,a.seed)
    tab.to_csv(a.out/'17_biopsy_sensitivity.csv',index=False)
    print(tab.to_string(index=False))
