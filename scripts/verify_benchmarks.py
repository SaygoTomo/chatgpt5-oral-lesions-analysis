#!/usr/bin/env python3
"""Compare reconstructed analysis outputs with public, rounded manuscript benchmarks.
This is a reproducibility smoke test, not a substitute for a full statistical review.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def check(out):
    failures=[]
    def expect(path,key,valuecol,expected,tol):
        d=pd.read_csv(out/path)
        rows=d.loc[key(d)]
        if len(rows)!=1: failures.append(f'{path}: expected one matching row, found {len(rows)}');return
        actual=float(rows.iloc[0][valuecol])
        if not np.isclose(actual,expected,atol=tol,rtol=0):failures.append(f'{path} {valuecol}: expected {expected}, found {actual}')
    for rep,alpha in [(1,.682),(2,.727),(3,.687)]:
        expect('04_cronbach_alpha_by_repetition.csv',lambda d,r=rep:d.repetition==r,'cronbach_alpha',alpha,.00051)
    for name,value in [('aipi_total',.539),('patient_feature',.449),('diagnosis',.476),('additional_exams',.469),('treatment',.493)]:
        expect('05_icc_main.csv',lambda d,n=name:d.outcome==n,'ICC_A1',value,.00051)
    for name,val in [('patient_feature',8.302),('diagnosis',1.062),('additional_exams',2.077),('treatment',4.084),('aipi_total',1.342)]:
        expect('06_friedman_kendallW.csv',lambda d,n=name:d.outcome==n,'friedman_chi2',val,.00051)
    for name,rho in [('patient_feature',.006),('diagnosis',-.192),('additional_exams',-.175),('treatment',-.299),('aipi_total',-.239)]:
        expect('08_challenge_vs_mean_performance.csv',lambda d,n=name:d.outcome==n,'spearman_rho',rho,.00051)
    # These two affected subgroups contain Case 119. Other groups are unchanged.
    for filename, group, expected in [
        ('15_clinical_subgroup_challenge_correlations.csv', 'Nodular lesions', -.344),
        ('15_pathologic_subgroup_challenge_correlations.csv', 'Non-neoplastic proliferative process', .321),
    ]:
        expect(filename,
               lambda d,g=group:(d.group==g)&(d.outcome=='treatment'),
               'spearman_rho', expected, .00051)
    if failures:
        for f in failures:print('FAIL',f)
        raise SystemExit(1)
    print('PASS: all 20 corrected rounded statistical point-estimate benchmarks matched.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results_directory',type=Path)
    args=p.parse_args()
    check(args.results_directory)
