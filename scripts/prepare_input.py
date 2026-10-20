#!/usr/bin/env python3
"""Convert the original, locally held study workbook into standardized *private* CSVs.

No clinical case data, photographs, or clinical records are bundled with this project.
The converter is tailored to the original 62-column 'Dados brutos' layout and the
'New Evaluator' sheet used for 50-case validation. Inspect the sheet layout before
using this converter with a revised workbook.
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path
import re
import pandas as pd

ITEM_IDX = (
    (20, 21, 22, 24, 25, 26, 28, 29, 31),
    (34, 35, 36, 38, 39, 40, 42, 43, 45),
    (48, 49, 50, 52, 53, 54, 56, 57, 59),
)
TOTAL_IDX = (33, 47, 61)
ACCS_IDX = list(range(13, 19))
PATHOLOGY_SPELLING = {
    "Malignan leoplasms": "Malignant neoplasms",
    "Non-neplastic proloferative process": "Non-neoplastic proliferative process",
    "Autoimune disease": "Autoimmune disease",
    "Odontogenic Cysts": "Odontogenic cysts",
}


def derive_accs_total(ratings):
    """Construct ACCS score strictly from its six 1–5 component ratings."""
    if len(ratings) != 6:
        raise ValueError("ACCS requires exactly six component ratings")
    converted = []
    for index, rating in enumerate(ratings, start=1):
        if pd.isna(rating) or float(rating) not in (1, 2, 3, 4, 5):
            raise ValueError(f"Invalid ACCS rating for component {index}: {rating!r}")
        converted.append(int(float(rating)))
    return sum(converted)


def convert(workbook: Path, dest: Path, validation_dest: Path):
    main = pd.read_excel(workbook, sheet_name="Dados brutos", header=1)
    third = pd.read_excel(workbook, sheet_name="New Evaluator", header=1)
    if len(main) != 184 or len(third) != 50 or main.shape[1] < 62 or third.shape[1] < 62:
        raise ValueError("Unexpected workbook layout: expected 184 main cases, 50 validation cases, >=62 columns. Please inspect manually.")
    if main.iloc[:, 0].isna().any() or main.iloc[:, 0].duplicated().any():
        raise ValueError("Main sheet has blank or duplicated Case IDs.")
    if third.iloc[:, 0].tolist() != main.iloc[:50, 0].tolist():
        raise ValueError("Validation sheet case IDs do not match the main sheet's first 50 cases.")
    records, validation = [], []
    score_discrepancies = []
    for i, (_, row) in enumerate(main.iterrows()):
        case_id = str(row.iloc[0])
        ac = [row.iloc[j] for j in ACCS_IDX]
        challenge_sum = derive_accs_total(ac)
        if not pd.isna(row.iloc[19]) and float(row.iloc[19]) != float(challenge_sum):
            # The original source for Case 119 has a one-row-off Excel formula.
            # Use the sum of six component ratings, which is the study definition.
            score_discrepancies.append((case_id, row.iloc[19], challenge_sum))
        meta = {
            "case_id": case_id,
            "challenge_score": challenge_sum,
            **{f"accs_domain_{j}": score for j, score in enumerate(ac, 1)},
            "clinical_class": row.iloc[10],
            "pathologic_class": PATHOLOGY_SPELLING.get(row.iloc[11], row.iloc[11]),
            "biopsy_recorded": int(bool(re.search(r"biopsy", str(row.iloc[7]), flags=re.I))),
            "age": row.iloc[1], "sex": row.iloc[2], "race": row.iloc[3],
            "smoking": row.iloc[4], "alcohol": row.iloc[5], "comorbidity": row.iloc[6],
            "additional_examinations": row.iloc[7], "reference_diagnosis": row.iloc[8],
            "anatomic_location": row.iloc[9], "reference_management": row.iloc[12],
        }
        for rep, items in enumerate(ITEM_IDX, start=1):
            out = {**meta, "repetition": rep}
            out.update({f"I{k}": row.iloc[col] for k, col in enumerate(items, 1)})
            records.append(out)
        if i < len(third):
            other = third.iloc[i]
            for rep, idx in enumerate(TOTAL_IDX, start=1):
                v = {"case_id": case_id, "repetition": rep,
                     "consensus_total": row.iloc[idx], "third_total": other.iloc[idx],
                     "consensus_accs_total": challenge_sum,
                     "third_accs_total": derive_accs_total([other.iloc[k] for k in ACCS_IDX])}
                for j, col in enumerate(ACCS_IDX, 1):
                    v[f"consensus_accs_{j}"] = row.iloc[col]
                    v[f"third_accs_{j}"] = other.iloc[col]
                validation.append(v)
    dest.parent.mkdir(parents=True, exist_ok=True)
    validation_dest.parent.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(records)
    if d["biopsy_recorded"].groupby(d["case_id"]).first().sum() != 153:
        raise ValueError("Unexpected number of biopsy-recorded cases; inspect source coding before analysis.")
    d.to_csv(dest, index=False)
    pd.DataFrame(validation).to_csv(validation_dest, index=False)
    print(f"Wrote {len(d)} private case/repetition records to {dest}")
    for case, reported, derived in score_discrepancies:
        warnings.warn(f"ACCS formula discrepancy for {case}: recorded total {reported}, sum of six domains {derived}. Using component-derived {derived}, not the incorrect cached total.")
    print(f"Wrote {len(validation)} private agreement records to {validation_dest}")
    print("IMPORTANT: these CSVs contain sensitive clinical information. DO NOT commit to GitHub.")
    print("Source I4 codes are LEGACY; analyze with --i4-coding raw-original.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workbook", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/private_long.csv"))
    p.add_argument("--agreement-out", type=Path, default=Path("data/private_agreement.csv"))
    a = p.parse_args()
    convert(a.workbook, a.out, a.agreement_out)
