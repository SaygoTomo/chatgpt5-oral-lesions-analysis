# ChatGPT-5 oral-lesion study — reproducible statistical analysis

Python code for the study **“ChatGPT-5 in oral lesion assessment: Clinical performance, reproducibility, and case complexity”** (184 clinical cases, three repeated LLM generations). The repository includes the main statistical analyses, AIPI/ACCS independent-evaluator agreement, and an exploratory biopsy-recorded sensitivity analysis.

**Provenance:** These are **reconstructed** scripts assembled and validated against the study workbook *after* the analyses were first performed. **They are not the original scripts used during the study.** Statistical point estimates were checked against the manuscript. See [REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md) for the corrected Case 119 ACCS formula and other qualifications.

**Privacy:** NO patient data, clinical photographs, model transcripts, clinical records or individual case-level scores are included. The `data/` and `results/` folders are excluded from Git. Do not upload the private original workbook or generated private CSVs to a public repository.

## 1. Install

Recommended environment: **Python 3.13.5**. The manuscript reports NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, and statsmodels 0.14.6. Install them with:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Convert the private source workbook

The original workbook consists of `Dados brutos` (184 cases) and `New Evaluator` (independent validation of 50 cases). This converter expects the 62-column layout inspected in October 2026 and will stop on incompatible column counts or case identifiers. **Run locally, never on public GitHub Actions.**

```bash
python scripts/prepare_input.py "data/Dados Projeto ChatGPT (BRUTO)(4).xlsx" \
  --out data/private_long.csv \
  --agreement-out data/private_agreement.csv
```

The prepared file has **552 rows** (184 cases × 3 repetitions) and columns `case_id`, `repetition`, `challenge_score`, `clinical_class`, `pathologic_class`, `biopsy_recorded`, `accs_domain_1` to `accs_domain_6`, `I1` to `I9`, plus optional patient characteristics. `private_agreement.csv` has 150 rows (50 cases × 3 repetitions).

ACCS challenge scores are **recalculated from the six domain ratings** for every case. In the uncorrected private source workbook, Case 119 has an incorrect sum formula and the converter warns but uses the correct component-derived total of 7. When using the companion corrected workbook, the discrepancy warning disappears.

The spreadsheet uses a **legacy ordering for Item 4**; the converter intentionally retains its original coding. The analysis command below explicitly fixes it. Using an already corrected dataset requires `--i4-coding final` instead.

## 3. Run the manuscript analyses

```bash
python scripts/analyze.py data/private_long.csv \
  --i4-coding raw-original \
  --out results \
  --bootstrap 5000 --seed 20260927
```

This produces CSV summaries for:

- AIPI item/domain/total descriptive statistics, Cronbach’s alpha and ICC(A,1)/ICC(A,3) with case-bootstrap 95% CIs
- Repeated-measure Friedman and Kendall’s W comparisons, post-hoc paired Wilcoxon with Holm correction, and inter-repetition Spearman correlations
- ACCS-derived case-challenge associations with each AIPI domain and the total, plus exploratory within-case variability
- Clinical and pathologic lesion-group comparisons using case-level mean scores and case-level success proportions, Kruskal–Wallis epsilon-squared and Holm-adjusted outcome families
- Exploratory Dunn pairwise comparisons, subgroup ICCs, Friedman analyses, and subgroup challenge correlations

All case-level analyses use one observation per case; three repeated generations are treated as dependent measurements. Code-derived variables follow the manuscript's published definitions. **Row-level results are not exported by default**; use `--export-row-level` only within private storage if genuinely necessary. Do not commit these results.

The script emits `01_analysis_audit.json` containing the source SHA-256 checksum, which can help document exactly which private input was used.

## 4. Independent evaluator agreement (supplementary S1 and S2)

```bash
python scripts/agreement.py data/private_agreement.csv \
  --out results --bootstrap 10000 --seed 20261020
```

Calculates linearly weighted Cohen’s κ, percentile bootstrap intervals, exact percentage agreement, and agreement within ±1 point for AIPI totals and six ACCS domains. The source workbook retains the *recorded* AIPI totals; that is the basis for agreement, consistent with the reported supplementary κ point estimates. The pooled AIPI CI resamples all three observations at the *case* level.

## 5. Diagnostic correctness and biopsy-recorded sensitivity

```bash
python scripts/biopsy_sensitivity.py data/private_long.csv \
  --out results --bootstrap 10000 --seed 20261020
```

Produces three-repetition proportions of correct primary diagnosis (`I5 = 3`) with Wilson 95% CIs, plus pooled rates with case-cluster bootstrap intervals for all cases, those with biopsy recorded, and those without. **Recorded biopsy is not synonymous with independent re-verification of each histological report.** This analysis is descriptive and should not be interpreted as causal.

## 6. Check numerical reproduction and run tests

```bash
python scripts/verify_benchmarks.py results
python -m pytest -q
```

The benchmark checker compares key statistics to *published rounded values*, not unpublished patient records or bootstrap percentile endpoints. Randomly generated test inputs are in `examples/synthetic_long.csv`.

To test the code without access to private records:

```bash
python scripts/analyze.py examples/synthetic_long.csv --i4-coding final \
  --out results_synthetic --bootstrap 30
python -m pytest -q
```

The synthetic dataset is for code demonstration and **does not reproduce study results**.

## 7. Data validation and transparency

- AIPI domains: patient features `I1+I2+I3`; diagnosis `I4+I5+I6`; additional examinations `I7+I8`; treatment `I9`; total=sum of all nine items.
- Scores outside each item's valid range are flagged and converted to missing, not zero. Derived scores require all their items.
- Individual-case data are never bundled. The private converter checks ACCS composite totals and uses the six domain scores to reconstruct them; the known Case 119 formula error is corrected.
- Reproduction of **reported bootstrap confidence limits** may require the original random seed and implementation, which were not recoverable from the manuscript alone.
- The `New Evaluator` worksheet scores support independent agreement between the original consensus scores and the third evaluator, not separate AIPI scoring by both specialists who originally formed the consensus.

## Suggested wording for a future manuscript revision

> The Python code used to reconstruct and independently verify the reported analyses is publicly available at [GITHUB REPOSITORY URL]. Patient-level data and clinical photographs are not publicly available because of confidentiality restrictions; anonymized analytic data may be made available from the corresponding author upon reasonable request and subject to ethics approval.

**Important:** Replace the placeholder only after the repository is published. Do not imply these scripts were the original contemporaneous analysis scripts.

## GitHub publication

1. Create a new public GitHub repository, e.g. `chatgpt5-oral-lesion-analysis`.
2. Unzip and upload its files/folders (or push using Git). Do **not** upload anything from the private `data/` directory.
3. Confirm GitHub Actions tests pass, then add the permanent GitHub URL to the paper's code availability statement.
4. Consider archiving a versioned release in Zenodo to obtain a DOI after your coauthors approve public distribution.

The MIT license covers code; it does **not** license or authorize public distribution of patient records.

## Correction history — 2026-10-20

The original `Dados brutos!T121` formula mistakenly referenced `N122:S122`.
The corrected total for Case 119 is **7** instead of **8**. This revision regenerates
the challenge correlations from component sums, updates verification benchmarks,
and adds a test for the six-item calculation. All other analyses and input coding
conventions remain as in the earlier reconstructed version.
