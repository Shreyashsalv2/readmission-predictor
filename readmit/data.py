"""Load and clean the Diabetes 130-US Hospitals dataset (UCI id 296, Strack et al. 2014).

Each row is one inpatient encounter of a diabetic patient (1999-2008, 130 US hospitals).
The target is readmission within 30 days (`readmitted == "<30"`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Discharge dispositions after which a readmission is impossible or not meaningful:
# 11 expired, 13 hospice/home, 14 hospice/medical facility, 19-21 expired (home / facility / unknown).
EXPIRED_OR_HOSPICE = {11, 13, 14, 19, 20, 21}

MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

NUMERIC = [
    "age_years", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]

BASE_CATEGORICAL = [
    "race", "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "medical_specialty", "diag_1_group", "diag_2_group", "diag_3_group",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
]

# Columns dropped on purpose, with the reason (reported in FINDINGS.md).
DROPPED = {
    "weight": "about 97% missing",
    "payer_code": "about 40% missing and an insurance code, not a clinical signal",
    "encounter_id": "identifier",
    "patient_nbr": "identifier (used only to keep one encounter per patient)",
}

TOP_SPECIALTIES = 10

# Coarse groups for the compact calculator model (ids from IDs_mapping.csv).
DISCHARGE_GROUPS = {1: "Home", 6: "Home with health service", 8: "Home with health service",
                    3: "Nursing or care facility", 4: "Nursing or care facility", 24: "Nursing or care facility",
                    23: "Nursing or care facility", 22: "Rehab or other hospital", 2: "Rehab or other hospital",
                    5: "Rehab or other hospital", 28: "Rehab or other hospital", 29: "Rehab or other hospital",
                    30: "Rehab or other hospital", 27: "Rehab or other hospital", 15: "Rehab or other hospital",
                    7: "Left against medical advice"}
ADMISSION_SOURCE_GROUPS = {7: "Emergency room", 1: "Referral", 2: "Referral", 3: "Referral",
                           4: "Transfer", 5: "Transfer", 6: "Transfer", 10: "Transfer", 18: "Transfer",
                           22: "Transfer", 25: "Transfer", 26: "Transfer"}


def icd9_group(code) -> str:
    """Map an ICD-9 code to the disease groups used by Strack et al. (2014)."""
    if code is None or (isinstance(code, float) and np.isnan(code)) or str(code).strip() == "":
        return "Missing"
    c = str(code).strip()
    if c[0] in "VE":                      # supplementary classification / external causes
        return "Other"
    try:
        x = float(c)
    except ValueError:
        return "Other"
    i = int(x)
    if i == 250:
        return "Diabetes"
    if 390 <= i <= 459 or i == 785:
        return "Circulatory"
    if 460 <= i <= 519 or i == 786:
        return "Respiratory"
    if 520 <= i <= 579 or i == 787:
        return "Digestive"
    if 800 <= i <= 999:
        return "Injury"
    if 710 <= i <= 739:
        return "Musculoskeletal"
    if 580 <= i <= 629 or i == 788:
        return "Genitourinary"
    if 140 <= i <= 239:
        return "Neoplasms"
    return "Other"


def age_midpoint(bracket: str) -> float:
    """'[70-80)' -> 75.0"""
    lo, hi = str(bracket).strip("[)").split("-")
    return (float(lo) + float(hi)) / 2


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV with '?' as missing; keep strings like 'None' as values."""
    return pd.read_csv(path, na_values=["?"], keep_default_na=False, low_memory=False)


@dataclass
class Cleaned:
    df: pd.DataFrame                      # features + 'target'
    numeric: list[str]
    categorical: list[str]
    steps: list[tuple[str, int]] = field(default_factory=list)   # (step, rows remaining)
    dropped_constant: list[str] = field(default_factory=list)


def _lab(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("").str.strip()
    return s.replace({"": "Not measured", "None": "Not measured"}).astype(str)


def clean(raw: pd.DataFrame) -> Cleaned:
    steps = [("raw encounters", len(raw))]
    df = raw.copy()
    df = df[df["gender"] != "Unknown/Invalid"]
    steps.append(("drop unknown gender", len(df)))
    df = df[~df["discharge_disposition_id"].astype(int).isin(EXPIRED_OR_HOSPICE)]
    steps.append(("drop expired / hospice discharges", len(df)))
    # One encounter per patient (the first), so no patient appears in both train and test
    # and repeat visitors don't dominate the sample.
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    steps.append(("keep first encounter per patient", len(df)))

    out = pd.DataFrame(index=df.index)
    out["age_years"] = df["age"].map(age_midpoint)
    for c in NUMERIC[1:]:
        out[c] = pd.to_numeric(df[c]).astype(float)
    out["race"] = df["race"].fillna("Missing").replace("", "Missing").astype(str)
    out["gender"] = df["gender"].astype(str)
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        out[c] = df[c].astype(int).astype(str)
    spec = df["medical_specialty"].fillna("Missing").replace("", "Missing").astype(str)
    top = spec[spec != "Missing"].value_counts().index[:TOP_SPECIALTIES]
    out["medical_specialty"] = spec.where(spec.isin(top) | (spec == "Missing"), "Other")
    for i in (1, 2, 3):
        out[f"diag_{i}_group"] = df[f"diag_{i}"].map(icd9_group)
    out["max_glu_serum"] = _lab(df["max_glu_serum"])
    out["A1Cresult"] = _lab(df["A1Cresult"])
    out["change"] = df["change"].astype(str)
    out["diabetesMed"] = df["diabetesMed"].astype(str)

    meds, constant = [], []
    for m in MEDICATIONS:
        if m not in df.columns:
            continue
        col = df[m].astype(str)
        if col.nunique() <= 1:
            constant.append(m)
            continue
        out[m] = col
        meds.append(m)

    # Compact-model groups
    out["discharge_group"] = df["discharge_disposition_id"].astype(int).map(DISCHARGE_GROUPS).fillna("Other or unknown")
    out["admission_source_group"] = df["admission_source_id"].astype(int).map(ADMISSION_SOURCE_GROUPS).fillna("Other or unknown")

    out["target"] = (df["readmitted"] == "<30").astype(int)
    out = out.reset_index(drop=True)
    return Cleaned(out, list(NUMERIC), BASE_CATEGORICAL + meds, steps, constant)


# Inputs of the compact model behind the in-browser calculator.
COMPACT_NUMERIC = ["age_years", "time_in_hospital", "num_medications", "number_diagnoses",
                   "number_inpatient", "number_emergency", "number_outpatient"]
COMPACT_CATEGORICAL = ["discharge_group", "admission_source_group", "diag_1_group", "A1Cresult",
                       "insulin", "diabetesMed"]
