#!/usr/bin/env python3
"""
Reproducible statistical analysis for the 184-case ChatGPT-5 oral-lesion AIPI study.

Expected design
---------------
- One case evaluated in three repetitions (R1, R2, R3).
- AIPI items I1-I9.
- Case-level ACCS-derived challenge score.
- Clinical and pathologic lesion classifications.

Implemented analyses
--------------------
1. Data validation and I4 recoding.
2. Descriptive statistics.
3. Cronbach's alpha by repetition.
4. ICC(A,1) and ICC(A,3) with case-level percentile bootstrap 95% CIs.
5. Friedman tests + Kendall's W; significant tests followed by Wilcoxon signed-rank
   comparisons with Holm adjustment.
6. Pairwise Spearman correlations across repetitions with Holm adjustment.
7. Spearman associations between challenge score and case-level mean AIPI outcomes,
   with Holm adjustment across five outcomes; exploratory association with within-case SD.
8. Lesion-type analyses using one case-level mean/proportion per case:
   Kruskal-Wallis + epsilon-squared, Holm correction within endpoint families,
   and exploratory Dunn-Holm pairwise comparisons for raw omnibus p < 0.05.
9. Within-case variability by lesion classification.
10. Subgroup ICCs (n >= 10), subgroup Friedman analyses (>=5 complete triplets),
    and subgroup challenge correlations (n >= 10, Holm across five outcomes).

The script avoids a dependency on scikit-posthocs by implementing Dunn's test directly.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import norm, rankdata
import statsmodels
from statsmodels.stats.multitest import multipletests


# -----------------------------------------------------------------------------
# Constants / study definitions
# -----------------------------------------------------------------------------
ITEMS = [f"I{i}" for i in range(1, 10)]
ITEM_RANGES = {
    "I1": (0, 2),
    "I2": (0, 2),
    "I3": (0, 2),
    "I4": (0, 3),
    "I5": (0, 3),
    "I6": (0, 1),
    "I7": (0, 3),
    "I8": (0, 1),
    "I9": (0, 3),
}

DOMAIN_DEFS = {
    "patient_feature": ["I1", "I2", "I3"],          # 0-6
    "diagnosis": ["I4", "I5", "I6"],              # 0-7
    "additional_exams": ["I7", "I8"],              # 0-4
    "treatment": ["I9"],                            # 0-3
}

PERFORMANCE_OUTCOMES = [
    "patient_feature",
    "diagnosis",
    "additional_exams",
    "treatment",
    "aipi_total",
]

SCORE_ENDPOINTS = ITEMS + PERFORMANCE_OUTCOMES  # 9 items + 4 domains + total = 14

SUCCESS_DEFS = {
    "success_I4_ge2_plausible_differential": lambda d: d["I4"] >= 2,
    "success_I5_eq3_correct_primary": lambda d: d["I5"] == 3,
    "success_I5_ge2_correct_or_plausible_primary": lambda d: d["I5"] >= 2,
    "success_I6_eq1_diagnostic_management": lambda d: d["I6"] == 1,
    "success_I7_ge2_adequate_exams": lambda d: d["I7"] >= 2,
    "success_I7_eq3_all_exams_pertinent": lambda d: d["I7"] == 3,
    "success_I8_eq1_most_relevant_exam": lambda d: d["I8"] == 1,
    "success_I9_ge2_adequate_treatment": lambda d: d["I9"] >= 2,
    "success_I9_eq3_pertinent_treatment": lambda d: d["I9"] == 3,
}
SUCCESS_ENDPOINTS = list(SUCCESS_DEFS.keys())

# Raw/original I4 coding -> corrected coding used in the final analysis.
# 0 remains 0; old 3 -> new 2; old 2 -> new 1; old 1 -> new 3.
I4_RECODE = {0: 0, 1: 3, 2: 1, 3: 2}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def holm_adjust(p_values):
    """Holm-adjust p-values while preserving NaNs."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() > 0:
        out[mask] = multipletests(p[mask], method="holm")[1]
    return out


def safe_spearman(x, y):
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(tmp) < 3 or tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return np.nan, np.nan, len(tmp)
    r = stats.spearmanr(tmp["x"], tmp["y"])
    return float(r.statistic), float(r.pvalue), len(tmp)


def numeric_summary(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {k: np.nan for k in ["n", "mean", "sd", "median", "q1", "q3", "min", "max"]}
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)) if s.size > 1 else np.nan,
        "median": float(s.median()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def cronbach_alpha(df_items: pd.DataFrame):
    """Cronbach's alpha using complete observations for the supplied items."""
    x = df_items.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    n, k = x.shape
    if n < 2 or k < 2:
        return np.nan, n
    item_var = x.var(axis=0, ddof=1).sum()
    total = x.sum(axis=1)
    total_var = total.var(ddof=1)
    if not np.isfinite(total_var) or total_var <= 0:
        return np.nan, n
    alpha = (k / (k - 1)) * (1 - item_var / total_var)
    return float(alpha), n


def icc_a1_ak(matrix: np.ndarray):
    """
    Two-way random-effects, absolute-agreement ICC(A,1) and ICC(A,k).

    Rows = targets/cases; columns = repetitions/raters.
    Requires a balanced complete matrix.
    """
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError("ICC input must be a 2-D matrix.")
    n, k = x.shape
    if n < 2 or k < 2 or np.isnan(x).any():
        return np.nan, np.nan

    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)

    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_total = np.sum((x - grand) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom_a1 = ms_rows + (k - 1) * ms_error + (k * (ms_cols - ms_error) / n)
    denom_ak = ms_rows + ((ms_cols - ms_error) / n)

    icc_a1 = (ms_rows - ms_error) / denom_a1 if denom_a1 != 0 else np.nan
    icc_ak = (ms_rows - ms_error) / denom_ak if denom_ak != 0 else np.nan
    return float(icc_a1), float(icc_ak)


def bootstrap_icc(matrix: np.ndarray, n_boot=5000, seed=20260927):
    """Case-level percentile bootstrap CI for ICC(A,1) and ICC(A,k)."""
    x = np.asarray(matrix, dtype=float)
    n = x.shape[0]
    point_a1, point_ak = icc_a1_ak(x)
    if n < 2:
        return point_a1, np.nan, np.nan, point_ak, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boots_a1 = np.empty(n_boot, dtype=float)
    boots_ak = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots_a1[b], boots_ak[b] = icc_a1_ak(x[idx, :])

    a1_ci = np.nanpercentile(boots_a1, [2.5, 97.5])
    ak_ci = np.nanpercentile(boots_ak, [2.5, 97.5])
    return point_a1, a1_ci[0], a1_ci[1], point_ak, ak_ci[0], ak_ci[1]


def kendalls_w_from_friedman(chi2: float, n: int, k: int = 3):
    if n <= 0 or k <= 1:
        return np.nan
    return float(chi2 / (n * (k - 1)))


def epsilon_squared_kw(H: float, n: int, k: int):
    """Epsilon-squared for Kruskal-Wallis; negative estimates truncated to 0."""
    if n <= k or k < 2:
        return np.nan
    return float(max(0.0, (H - k + 1) / (n - k)))


def dunn_test(data: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    """Two-sided Dunn pairwise comparisons with Holm adjustment."""
    d = data[[value_col, group_col]].dropna().copy()
    if d.empty:
        return pd.DataFrame()

    d["rank"] = rankdata(d[value_col].to_numpy(), method="average")
    N = len(d)
    groups = list(d[group_col].dropna().unique())
    if len(groups) < 2:
        return pd.DataFrame()

    # Tie-corrected rank variance term used in Dunn's test.
    tie_counts = d[value_col].value_counts().to_numpy()
    tie_sum = np.sum(tie_counts**3 - tie_counts)
    var_term = (N * (N + 1) / 12.0)
    if N > 1:
        var_term -= tie_sum / (12.0 * (N - 1))

    rows = []
    for g1, g2 in combinations(groups, 2):
        a = d.loc[d[group_col] == g1, "rank"]
        b = d.loc[d[group_col] == g2, "rank"]
        n1, n2 = len(a), len(b)
        se = np.sqrt(var_term * (1.0 / n1 + 1.0 / n2))
        z = (a.mean() - b.mean()) / se if se > 0 else np.nan
        p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
        rows.append({
            "group_1": g1,
            "group_2": g2,
            "n_1": n1,
            "n_2": n2,
            "z": z,
            "p_raw": p,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm_adjust(out["p_raw"])
    return out


def normalize_repetition(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, np.integer)) and int(v) in (1, 2, 3):
        return int(v)
    if isinstance(v, (float, np.floating)) and np.isfinite(v) and int(v) in (1, 2, 3):
        return int(v)
    m = re.search(r"([123])", str(v))
    return int(m.group(1)) if m else np.nan


def first_nonnull(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Input must be CSV or Excel (.xlsx/.xlsm/.xls).")


def reshape_if_wide(df: pd.DataFrame, case_col: str, rep_col: str) -> pd.DataFrame:
    """
    Accept either long data (I1-I9 + repetition column) or common wide patterns:
    I1_R1 ... I9_R3, or R1_I1 ... R3_I9.
    """
    if rep_col in df.columns and all(c in df.columns for c in ITEMS):
        return df.copy()

    patterns = []
    for rep in (1, 2, 3):
        mapping_a = {item: f"{item}_R{rep}" for item in ITEMS}
        mapping_b = {item: f"R{rep}_{item}" for item in ITEMS}
        if all(col in df.columns for col in mapping_a.values()):
            patterns.append((rep, mapping_a))
        elif all(col in df.columns for col in mapping_b.values()):
            patterns.append((rep, mapping_b))
        else:
            patterns.append((rep, None))

    if not all(m is not None for _, m in patterns):
        raise ValueError(
            "Could not identify long-format I1-I9 + repetition data or wide-format "
            "I1_R1...I9_R3 / R1_I1...R3_I9 columns."
        )

    item_wide_cols = {col for _, mp in patterns for col in mp.values()}
    meta_cols = [c for c in df.columns if c not in item_wide_cols]
    if case_col not in meta_cols:
        raise ValueError(f"Case ID column '{case_col}' not found.")

    parts = []
    for rep, mp in patterns:
        p = df[meta_cols].copy()
        p[rep_col] = rep
        for item, source in mp.items():
            p[item] = df[source]
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def prepare_data(
    raw: pd.DataFrame,
    case_col: str,
    rep_col: str,
    i4_already_corrected: bool,
):
    df = reshape_if_wide(raw, case_col, rep_col)

    required = [case_col, rep_col] + ITEMS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[rep_col] = df[rep_col].map(normalize_repetition)
    if df[rep_col].isna().any():
        bad = df.loc[df[rep_col].isna(), [case_col, rep_col]]
        raise ValueError(f"Unrecognized repetition values found. Example:\n{bad.head()}")
    df[rep_col] = df[rep_col].astype(int)

    if df.duplicated([case_col, rep_col]).any():
        dups = df.loc[df.duplicated([case_col, rep_col], keep=False), [case_col, rep_col]]
        raise ValueError(
            "Duplicate case/repetition rows were found. Resolve them before analysis. "
            f"Examples:\n{dups.head(10)}"
        )

    for item in ITEMS:
        df[item] = pd.to_numeric(df[item], errors="coerce")

    # Apply the final I4 coding correction once, unless the supplied dataset is already corrected.
    if not i4_already_corrected:
        original = df["I4"].copy()
        df["I4"] = df["I4"].map(lambda x: I4_RECODE.get(int(x), np.nan) if pd.notna(x) and float(x).is_integer() else np.nan)
        recoded_n = int((original.notna() & df["I4"].notna()).sum())
    else:
        recoded_n = 0

    # Range validation. Invalid values become missing; derived affected scores therefore become missing.
    quality_rows = []
    for item, (lo, hi) in ITEM_RANGES.items():
        invalid = df[item].notna() & ~df[item].between(lo, hi, inclusive="both")
        for idx in df.index[invalid]:
            quality_rows.append({
                "case_id": df.at[idx, case_col],
                "repetition": df.at[idx, rep_col],
                "item": item,
                "invalid_value": df.at[idx, item],
                "action": "set to missing",
            })
        df.loc[invalid, item] = np.nan

    # Derived scores require all component items for that row.
    for domain, components in DOMAIN_DEFS.items():
        df[domain] = df[components].sum(axis=1, min_count=len(components))
    df["aipi_total"] = df[ITEMS].sum(axis=1, min_count=len(ITEMS))

    # Success indicators preserve missingness.
    for name, rule in SUCCESS_DEFS.items():
        # Identify the source item from the endpoint name.
        source = re.search(r"(I[4-9])", name).group(1)
        df[name] = np.where(df[source].notna(), rule(df).astype(float), np.nan)

    quality = pd.DataFrame(quality_rows)
    return df, quality, recoded_n


def make_case_metadata(df, case_col, rep_col, challenge_col, clinical_col, pathologic_col):
    candidate_meta = [challenge_col, clinical_col, pathologic_col]
    # Additional common case-level fields retained if present.
    candidate_meta += [
        "age", "sex", "race", "smoking", "alcohol", "comorbidity",
        "additional_examinations", "reference_diagnosis", "anatomic_location",
        "reference_management",
    ]
    candidate_meta = [c for c in candidate_meta if c in df.columns]

    meta = df.groupby(case_col, as_index=False)[candidate_meta].agg(first_nonnull) if candidate_meta else df[[case_col]].drop_duplicates()
    return meta


def make_case_aggregates(df, case_col):
    mean_cols = SCORE_ENDPOINTS + SUCCESS_ENDPOINTS
    agg = df.groupby(case_col, as_index=False)[mean_cols].mean(numeric_only=True)
    var_sd = (
        df.groupby(case_col)["aipi_total"]
        .agg(lambda s: s.dropna().std(ddof=1) if s.dropna().size >= 2 else np.nan)
        .rename("within_case_sd_total")
        .reset_index()
    )
    return agg.merge(var_sd, on=case_col, how="left")


# -----------------------------------------------------------------------------
# Analyses
# -----------------------------------------------------------------------------
def baseline_descriptives(case_meta: pd.DataFrame, challenge_col: str):
    numeric_candidates = [c for c in ["age", challenge_col] if c in case_meta.columns]
    categorical_candidates = [
        c for c in [
            "sex", "race", "smoking", "alcohol", "comorbidity",
            "additional_examinations", "reference_diagnosis", "anatomic_location",
            "reference_management",
        ] if c in case_meta.columns
    ]

    num_rows = []
    for c in numeric_candidates:
        row = {"variable": c, **numeric_summary(case_meta[c])}
        num_rows.append(row)

    cat_rows = []
    for c in categorical_candidates:
        s = case_meta[c]
        denom = int(s.notna().sum())
        for level, n in s.value_counts(dropna=True).items():
            cat_rows.append({
                "variable": c,
                "level": level,
                "n": int(n),
                "denominator_nonmissing": denom,
                "percent": 100 * n / denom if denom else np.nan,
            })
    return pd.DataFrame(num_rows), pd.DataFrame(cat_rows)


def repetition_descriptives(df, rep_col):
    rows = []
    for outcome in SCORE_ENDPOINTS:
        for rep in (1, 2, 3):
            row = {"outcome": outcome, "repetition": rep, **numeric_summary(df.loc[df[rep_col] == rep, outcome])}
            rows.append(row)
    return pd.DataFrame(rows)


def cronbach_by_repetition(df, rep_col):
    rows = []
    for rep in (1, 2, 3):
        alpha, n = cronbach_alpha(df.loc[df[rep_col] == rep, ITEMS])
        rows.append({"repetition": rep, "n_complete": n, "cronbach_alpha": alpha})
    return pd.DataFrame(rows)


def pivot_outcome(df, case_col, rep_col, outcome):
    w = df.pivot(index=case_col, columns=rep_col, values=outcome)
    for rep in (1, 2, 3):
        if rep not in w.columns:
            w[rep] = np.nan
    return w[[1, 2, 3]].sort_index()


def icc_main(df, case_col, rep_col, n_boot, seed):
    rows = []
    for j, outcome in enumerate(PERFORMANCE_OUTCOMES):
        w = pivot_outcome(df, case_col, rep_col, outcome).dropna()
        vals = bootstrap_icc(w.to_numpy(), n_boot=n_boot, seed=seed + j)
        rows.append({
            "outcome": outcome,
            "n_complete_triplets": len(w),
            "ICC_A1": vals[0], "ICC_A1_CI_low": vals[1], "ICC_A1_CI_high": vals[2],
            "ICC_A3": vals[3], "ICC_A3_CI_low": vals[4], "ICC_A3_CI_high": vals[5],
        })
    return pd.DataFrame(rows)


def friedman_and_posthoc(df, case_col, rep_col):
    omnibus = []
    posthoc_all = []
    for outcome in PERFORMANCE_OUTCOMES:
        w = pivot_outcome(df, case_col, rep_col, outcome).dropna()
        n = len(w)
        if n >= 2:
            fr = stats.friedmanchisquare(w[1], w[2], w[3])
            chi2, p = float(fr.statistic), float(fr.pvalue)
            W = kendalls_w_from_friedman(chi2, n, 3)
        else:
            chi2 = p = W = np.nan
        omnibus.append({
            "outcome": outcome,
            "n_complete_triplets": n,
            "friedman_chi2": chi2,
            "df": 2,
            "p_raw": p,
            "kendall_W": W,
        })

        if np.isfinite(p) and p < 0.05:
            ph = []
            for a, b in [(1, 2), (1, 3), (2, 3)]:
                try:
                    res = stats.wilcoxon(w[a], w[b], alternative="two-sided", method="auto")
                    stat, pp = float(res.statistic), float(res.pvalue)
                except ValueError:
                    stat, pp = 0.0, 1.0
                ph.append({
                    "outcome": outcome,
                    "comparison": f"R{a} vs R{b}",
                    "n": n,
                    "wilcoxon_W": stat,
                    "p_raw": pp,
                })
            ph = pd.DataFrame(ph)
            ph["p_holm"] = holm_adjust(ph["p_raw"])
            posthoc_all.append(ph)

    posthoc = pd.concat(posthoc_all, ignore_index=True) if posthoc_all else pd.DataFrame()
    return pd.DataFrame(omnibus), posthoc


def repetition_spearman(df, case_col, rep_col):
    rows = []
    for outcome in PERFORMANCE_OUTCOMES:
        w = pivot_outcome(df, case_col, rep_col, outcome)
        local = []
        for a, b in [(1, 2), (1, 3), (2, 3)]:
            rho, p, n = safe_spearman(w[a], w[b])
            local.append({
                "outcome": outcome,
                "comparison": f"R{a} vs R{b}",
                "n": n,
                "spearman_rho": rho,
                "p_raw": p,
            })
        local = pd.DataFrame(local)
        local["p_holm"] = holm_adjust(local["p_raw"])
        rows.append(local)
    return pd.concat(rows, ignore_index=True)


def challenge_analyses(case_agg, case_meta, case_col, challenge_col):
    if challenge_col not in case_meta.columns:
        return pd.DataFrame(), pd.DataFrame()
    d = case_agg.merge(case_meta[[case_col, challenge_col]], on=case_col, how="left")

    rows = []
    for outcome in PERFORMANCE_OUTCOMES:
        rho, p, n = safe_spearman(d[challenge_col], d[outcome])
        rows.append({"outcome": outcome, "n": n, "spearman_rho": rho, "p_raw": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_raw"])

    rho, p, n = safe_spearman(d[challenge_col], d["within_case_sd_total"])
    variability = pd.DataFrame([{
        "outcome": "within_case_sd_total",
        "n": n,
        "spearman_rho": rho,
        "p_raw_exploratory": p,
    }])
    return out, variability


def eligible_groups(data, group_col, min_n):
    counts = data[group_col].value_counts(dropna=True)
    return list(counts[counts >= min_n].index)


def kw_one_endpoint(data, value_col, group_col, groups):
    parts = []
    labels = []
    for g in groups:
        s = pd.to_numeric(data.loc[data[group_col] == g, value_col], errors="coerce").dropna()
        if len(s) > 0:
            parts.append(s.to_numpy())
            labels.append(g)
    if len(parts) < 2:
        return np.nan, np.nan, np.nan, sum(len(p) for p in parts), len(parts)
    H, p = stats.kruskal(*parts, nan_policy="omit")
    N = sum(len(p) for p in parts)
    k = len(parts)
    eps = epsilon_squared_kw(float(H), N, k)
    return float(H), float(p), eps, N, k


def lesion_type_analyses(case_data, group_col, min_n=5):
    if group_col not in case_data.columns:
        return (pd.DataFrame(),) * 6

    groups = eligible_groups(case_data, group_col, min_n)
    if len(groups) < 2:
        return (pd.DataFrame(),) * 6

    score_rows = []
    score_desc = []
    for endpoint in SCORE_ENDPOINTS:
        H, p, eps, N, k = kw_one_endpoint(case_data, endpoint, group_col, groups)
        score_rows.append({
            "endpoint": endpoint, "H": H, "df": k - 1 if k >= 2 else np.nan,
            "N": N, "k_groups": k, "p_raw": p, "epsilon_squared": eps,
        })
        for g in groups:
            s = case_data.loc[case_data[group_col] == g, endpoint]
            score_desc.append({"endpoint": endpoint, "group": g, **numeric_summary(s)})
    score_kw = pd.DataFrame(score_rows)
    score_kw["p_holm_family14"] = holm_adjust(score_kw["p_raw"])

    success_rows = []
    success_desc = []
    for endpoint in SUCCESS_ENDPOINTS:
        H, p, eps, N, k = kw_one_endpoint(case_data, endpoint, group_col, groups)
        success_rows.append({
            "endpoint": endpoint, "H": H, "df": k - 1 if k >= 2 else np.nan,
            "N": N, "k_groups": k, "p_raw": p, "epsilon_squared": eps,
        })
        for g in groups:
            s = case_data.loc[case_data[group_col] == g, endpoint]
            score = numeric_summary(s)
            success_desc.append({"endpoint": endpoint, "group": g, **score})
    success_kw = pd.DataFrame(success_rows)
    success_kw["p_holm_family9"] = holm_adjust(success_kw["p_raw"])

    # Dunn tests are exploratory whenever raw omnibus p < .05.
    dunn_scores = []
    for _, r in score_kw.loc[score_kw["p_raw"] < 0.05].iterrows():
        dd = case_data.loc[case_data[group_col].isin(groups), [group_col, r["endpoint"]]].copy()
        tmp = dunn_test(dd, r["endpoint"], group_col)
        if not tmp.empty:
            tmp.insert(0, "endpoint", r["endpoint"])
            dunn_scores.append(tmp)
    dunn_scores = pd.concat(dunn_scores, ignore_index=True) if dunn_scores else pd.DataFrame()

    dunn_success = []
    for _, r in success_kw.loc[success_kw["p_raw"] < 0.05].iterrows():
        dd = case_data.loc[case_data[group_col].isin(groups), [group_col, r["endpoint"]]].copy()
        tmp = dunn_test(dd, r["endpoint"], group_col)
        if not tmp.empty:
            tmp.insert(0, "endpoint", r["endpoint"])
            dunn_success.append(tmp)
    dunn_success = pd.concat(dunn_success, ignore_index=True) if dunn_success else pd.DataFrame()

    return (
        score_kw,
        pd.DataFrame(score_desc),
        success_kw,
        pd.DataFrame(success_desc),
        dunn_scores,
        dunn_success,
    )


def variability_by_group(case_data, group_col, min_n=5):
    if group_col not in case_data.columns:
        return pd.DataFrame()
    groups = eligible_groups(case_data, group_col, min_n)
    H, p, eps, N, k = kw_one_endpoint(case_data, "within_case_sd_total", group_col, groups)
    return pd.DataFrame([{
        "classification": group_col,
        "H": H,
        "df": k - 1 if k >= 2 else np.nan,
        "N": N,
        "k_groups": k,
        "p_raw": p,
        "epsilon_squared": eps,
    }])


def subgroup_icc(df, case_meta, case_col, rep_col, group_col, n_boot, seed, min_n=10):
    if group_col not in case_meta.columns:
        return pd.DataFrame()
    groups = eligible_groups(case_meta, group_col, min_n)
    rows = []
    for gi, g in enumerate(groups):
        ids = case_meta.loc[case_meta[group_col] == g, case_col]
        sub = df[df[case_col].isin(ids)]
        for oi, outcome in enumerate(PERFORMANCE_OUTCOMES):
            w = pivot_outcome(sub, case_col, rep_col, outcome).dropna()
            vals = bootstrap_icc(w.to_numpy(), n_boot=n_boot, seed=seed + gi * 100 + oi)
            rows.append({
                "classification": group_col,
                "group": g,
                "group_n_cases": int(len(ids)),
                "outcome": outcome,
                "n_complete_triplets": int(len(w)),
                "ICC_A1": vals[0], "ICC_A1_CI_low": vals[1], "ICC_A1_CI_high": vals[2],
                "ICC_A3": vals[3], "ICC_A3_CI_low": vals[4], "ICC_A3_CI_high": vals[5],
            })
    return pd.DataFrame(rows)


def subgroup_friedman(df, case_meta, case_col, rep_col, group_col, min_triplets=5):
    if group_col not in case_meta.columns:
        return pd.DataFrame()
    groups = list(case_meta[group_col].dropna().unique())
    all_rows = []
    for g in groups:
        ids = case_meta.loc[case_meta[group_col] == g, case_col]
        sub = df[df[case_col].isin(ids)]
        local = []
        for outcome in PERFORMANCE_OUTCOMES:
            w = pivot_outcome(sub, case_col, rep_col, outcome).dropna()
            n = len(w)
            if n >= min_triplets:
                fr = stats.friedmanchisquare(w[1], w[2], w[3])
                chi2, p = float(fr.statistic), float(fr.pvalue)
                W = kendalls_w_from_friedman(chi2, n, 3)
            else:
                chi2 = p = W = np.nan
            local.append({
                "classification": group_col,
                "group": g,
                "outcome": outcome,
                "n_complete_triplets": n,
                "friedman_chi2": chi2,
                "df": 2,
                "p_raw": p,
                "kendall_W": W,
            })
        local = pd.DataFrame(local)
        local["p_holm_within_group_5_outcomes"] = holm_adjust(local["p_raw"])
        all_rows.append(local)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def subgroup_challenge(case_data, group_col, challenge_col, min_n=10):
    if group_col not in case_data.columns or challenge_col not in case_data.columns:
        return pd.DataFrame()
    groups = eligible_groups(case_data, group_col, min_n)
    all_rows = []
    for g in groups:
        d = case_data[case_data[group_col] == g]
        local = []
        for outcome in PERFORMANCE_OUTCOMES:
            rho, p, n = safe_spearman(d[challenge_col], d[outcome])
            local.append({
                "classification": group_col,
                "group": g,
                "outcome": outcome,
                "n": n,
                "spearman_rho": rho,
                "p_raw": p,
            })
        local = pd.DataFrame(local)
        local["p_holm_within_group_5_outcomes"] = holm_adjust(local["p_raw"])
        all_rows.append(local)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def save_df(df, outdir: Path, filename: str):
    if df is None or df.empty:
        return
    df.to_csv(outdir / filename, index=False)


def save_versions(outdir: Path):
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "platform": platform.platform(),
    }
    (outdir / "00_software_versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Reproduce all AIPI manuscript analyses.")
    parser.add_argument("input", type=Path, help="Input CSV or Excel file.")
    parser.add_argument("--out", type=Path, default=Path("aipi_results"), help="Output directory.")
    parser.add_argument("--case-col", default="case_id", help="Case identifier column.")
    parser.add_argument("--rep-col", default="repetition", help="Repetition column (R1/R2/R3 or 1/2/3).")
    parser.add_argument("--challenge-col", default="challenge_score", help="ACCS-derived challenge score column.")
    parser.add_argument("--clinical-col", default="clinical_class", help="Clinical lesion classification column.")
    parser.add_argument("--pathologic-col", default="pathologic_class", help="Pathologic lesion classification column.")
    parser.add_argument(
        "--i4-coding", required=True, choices=["raw-original", "final"],
        help="MANDATORY: raw-original recodes 0->0, 1->3, 2->1, 3->2; final keeps I4 as-is.",
    )
    parser.add_argument("--export-row-level", action="store_true", help="Export sensitive case-level data locally (NEVER commit).")
    parser.add_argument("--bootstrap", type=int, default=5000, help="Number of case-level bootstrap resamples for ICC CIs.")
    parser.add_argument("--seed", type=int, default=20260927, help="Random seed for bootstrap reproducibility.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    save_versions(args.out)

    raw = load_table(args.input)
    component_cols = [f"accs_domain_{i}" for i in range(1, 7)]
    present = [c for c in component_cols if c in raw.columns]
    if present and len(present) != 6:
        raise ValueError("Incomplete ACCS component columns: all six are required")
    if len(present) == 6:
        components = raw[component_cols].apply(pd.to_numeric, errors="coerce")
        if components.isna().any().any() or not components.isin([1, 2, 3, 4, 5]).all().all():
            raise ValueError("ACCS component values must be present and integers 1–5")
        reconstructed = components.sum(axis=1)
        if args.challenge_col in raw:
            old = pd.to_numeric(raw[args.challenge_col], errors="coerce")
            n_wrong = int((old != reconstructed).sum())
            if n_wrong:
                print(f"WARNING: Replacing {n_wrong} inconsistent ACCS recorded totals with component-derived sums.")
        raw[args.challenge_col] = reconstructed
    else:
        print("NOTICE: No ACCS component columns supplied; provided challenge scores will be used. "
              "For source-spreadsheet verification, rerun prepare_input.py first.")
    df, quality, recoded_n = prepare_data(
        raw, args.case_col, args.rep_col, args.i4_coding == "final"
    )
    save_df(quality, args.out, "01_data_quality_invalid_scores.csv")

    case_meta = make_case_metadata(
        df, args.case_col, args.rep_col,
        args.challenge_col, args.clinical_col, args.pathologic_col,
    )
    case_agg = make_case_aggregates(df, args.case_col)
    case_data = case_agg.merge(case_meta, on=args.case_col, how="left")

    # Basic audit trail.
    import hashlib
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    audit = {
        "sha256_input": digest,
        "input_file": str(args.input),
        "n_rows_long": int(len(df)),
        "n_unique_cases": int(df[args.case_col].nunique()),
        "i4_recoding_applied": args.i4_coding == "raw-original",
        "n_nonmissing_i4_values_recoded": recoded_n,
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_seed": args.seed,
    }
    (args.out / "01_analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # Descriptive statistics.
    num_desc, cat_desc = baseline_descriptives(case_meta, args.challenge_col)
    save_df(num_desc, args.out, "02_case_numeric_descriptives.csv")
    save_df(cat_desc, args.out, "02_case_categorical_descriptives.csv")
    save_df(repetition_descriptives(df, args.rep_col), args.out, "03_repetition_score_descriptives.csv")

    # Internal consistency and reproducibility.
    save_df(cronbach_by_repetition(df, args.rep_col), args.out, "04_cronbach_alpha_by_repetition.csv")
    save_df(icc_main(df, args.case_col, args.rep_col, args.bootstrap, args.seed), args.out, "05_icc_main.csv")

    # Repeated-measures comparisons.
    friedman, wilcoxon = friedman_and_posthoc(df, args.case_col, args.rep_col)
    save_df(friedman, args.out, "06_friedman_kendallW.csv")
    save_df(wilcoxon, args.out, "06_wilcoxon_posthoc_holm.csv")
    save_df(repetition_spearman(df, args.case_col, args.rep_col), args.out, "07_spearman_repetition_stability.csv")

    # Challenge score.
    challenge, challenge_var = challenge_analyses(case_agg, case_meta, args.case_col, args.challenge_col)
    save_df(challenge, args.out, "08_challenge_vs_mean_performance.csv")
    save_df(challenge_var, args.out, "08_challenge_vs_within_case_variability.csv")

    # Lesion-type analyses.
    for label, group_col in [("clinical", args.clinical_col), ("pathologic", args.pathologic_col)]:
        if group_col not in case_data.columns:
            continue
        score_kw, score_desc, success_kw, success_desc, dunn_score, dunn_success = lesion_type_analyses(case_data, group_col, min_n=5)
        save_df(score_kw, args.out, f"09_{label}_KW_score_endpoints.csv")
        save_df(score_desc, args.out, f"09_{label}_score_group_descriptives.csv")
        save_df(success_kw, args.out, f"10_{label}_KW_success_endpoints.csv")
        save_df(success_desc, args.out, f"10_{label}_success_group_descriptives.csv")
        save_df(dunn_score, args.out, f"11_{label}_Dunn_score_endpoints.csv")
        save_df(dunn_success, args.out, f"11_{label}_Dunn_success_endpoints.csv")
        save_df(variability_by_group(case_data, group_col, min_n=5), args.out, f"12_{label}_within_case_variability_KW.csv")
        save_df(subgroup_icc(df, case_meta, args.case_col, args.rep_col, group_col, args.bootstrap, args.seed + (1000 if label == "clinical" else 2000)), args.out, f"13_{label}_subgroup_ICC.csv")
        save_df(subgroup_friedman(df, case_meta, args.case_col, args.rep_col, group_col), args.out, f"14_{label}_subgroup_Friedman.csv")
        save_df(subgroup_challenge(case_data, group_col, args.challenge_col), args.out, f"15_{label}_subgroup_challenge_correlations.csv")

    # Case-level records can contain sensitive information; do NOT export by default.
    if args.export_row_level:
        df.to_csv(args.out / "90_PRIVATE_cleaned_long_data.csv", index=False)
        case_data.to_csv(args.out / "91_PRIVATE_case_level_analysis.csv", index=False)

    print(f"Analysis complete. Results written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
