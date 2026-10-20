# Provenance and numerical reproducibility

The programs in this repository were **reconstructed and validated after the analyses were
reported**. They are not the original scripts executed during the research. Do not claim
otherwise in the manuscript or public release.

The main statistical workflow was checked against the original 184-case workbook;
point estimates reproduced published Cronbach alpha, ICC, Friedman/Kendall W and
Holm-adjusted Spearman results. Agreement scripts reproduce weighted kappa point
estimates in the independent 50-case evaluation. Bootstrap 95% intervals are stochastic
and depend on random seed, implementation and resampling settings; they may differ
slightly from the manuscript's reported limits.

## Original source coding: AIPI Item 4

The private source workbook uses an older coding for Item 4:

| Original source code | Final AIPI score |
|---|---|
| 0 | 0 |
| 1 | 3 |
| 2 | 1 |
| 3 | 2 |

The analysis script requires an explicit flag (`--i4-coding raw-original` or
`--i4-coding final`) to prevent accidental double recoding.

## Corrected ACCS source-formula discrepancy (Case 119)

The source workbook's `Dados brutos!T121` originally had `=SUM(N122:S122)`,
which references the next case and displays **8**. The six component ratings for
Case 119 in `N121:S121` are `1, 1, 1, 1, 2, 1` and sum to **7**.
The corrected private workbook changes this single formula to `=SUM(N121:S121)`.

This updated repository **always derives ACCS challenge scores directly from the
six 1–5 ratings** when importing the source workbook. It flags any inconsistent
cached total rather than trusting it. The corrected analysis slightly changes
challenge correlations (main Table 4, supplementary S13 and S18), without changing
significance interpretations. All other analyses are unchanged.

The CSV converter adds the six columns `accs_domain_1` through `accs_domain_6`;
the analysis script independently recalculates their sum when those columns are present.
**Regenerate any private CSVs from the source workbook** before using the revised
benchmark checker; older CSVs based on the erroneous total are not valid for the
corrected manuscript.

## Inter-rater agreement

AIPI kappas here are computed from the *recorded total scores* on the main and
third-evaluator sheets, as reported in the supplementary materials. ACCS domain
kappas use the paired 1–5 ratings for 50 cases. The assessment of separate authors'
initial consensus discussions cannot be inferred from the consensus sheet.

## Interpretation of 'biopsy recorded'

The automated converter identifies any procedure text containing 'biopsy' and obtains
153 cases in this workbook. This is an indicator of biopsy **recorded in the source**,
not an independently verified pathology report. Confirm each histopathological
reference standard before stating that all 153 were biopsy-confirmed.

## No individual patient-level data are included here

Actual data and analysis outputs should be kept in a private directory and shared
only under the ethics approval and applicable privacy rules. Public GitHub files
contain only scripts, instructions, and a synthetic test dataset.
