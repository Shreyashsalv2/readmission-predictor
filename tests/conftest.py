"""A synthetic dataset with the real file's columns and quirks ('?' for missing, repeat patients,
expired discharges, unknown gender), so the tests run without downloading anything."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from readmit.data import MEDICATIONS


def synthetic_raw(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    inpatient = rng.poisson(0.6, n)
    logit = -2.4 + 1.0 * inpatient + rng.normal(0, 0.5, n)
    readmit = rng.random(n) < 1 / (1 + np.exp(-logit))
    later = (~readmit) & (rng.random(n) < 0.35)
    df = pd.DataFrame({
        "encounter_id": np.arange(n) * 7 + 1000,
        "patient_nbr": rng.integers(0, int(n * 0.75), n),            # some patients repeat
        "race": rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Other", "?"], n, p=[.7, .18, .04, .04, .04]),
        "gender": rng.choice(["Female", "Male", "Unknown/Invalid"], n, p=[.53, .465, .005]),
        "age": rng.choice([f"[{a}-{a + 10})" for a in range(0, 100, 10)], n),
        "weight": rng.choice(["?", "[75-100)"], n, p=[.97, .03]),
        "admission_type_id": rng.choice([1, 2, 3, 5, 6], n),
        "discharge_disposition_id": rng.choice([1, 3, 6, 11, 13, 22, 25], n, p=[.55, .14, .14, .02, .01, .08, .06]),
        "admission_source_id": rng.choice([1, 4, 7, 17], n, p=[.3, .05, .6, .05]),
        "time_in_hospital": rng.integers(1, 15, n),
        "payer_code": rng.choice(["?", "MC", "HM"], n),
        "medical_specialty": rng.choice(["?", "InternalMedicine", "Cardiology", "Surgery-General", "Nephrology"], n),
        "num_lab_procedures": rng.integers(1, 100, n),
        "num_procedures": rng.integers(0, 7, n),
        "num_medications": rng.integers(1, 60, n),
        "number_outpatient": rng.poisson(0.4, n),
        "number_emergency": rng.poisson(0.2, n),
        "number_inpatient": inpatient,
        "diag_1": rng.choice(["428", "250.83", "414", "786", "V57", "E888", "?", "715", "996"], n),
        "diag_2": rng.choice(["250", "427", "599", "?", "403"], n),
        "diag_3": rng.choice(["401", "250.01", "?", "V45", "272"], n),
        "number_diagnoses": rng.integers(1, 17, n),
        "max_glu_serum": rng.choice(["None", ">200", ">300", "Norm"], n, p=[.95, .02, .01, .02]),
        "A1Cresult": rng.choice(["None", ">7", ">8", "Norm"], n, p=[.83, .04, .08, .05]),
        "change": rng.choice(["No", "Ch"], n),
        "diabetesMed": rng.choice(["Yes", "No"], n, p=[.77, .23]),
        "readmitted": np.where(readmit, "<30", np.where(later, ">30", "NO")),
    })
    for m in MEDICATIONS:
        df[m] = "No"
    for m in ["metformin", "insulin", "glipizide"]:
        df[m] = rng.choice(["No", "Steady", "Up", "Down"], n, p=[.5, .3, .1, .1])
    return df


@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "diabetic_data.csv"
    synthetic_raw().to_csv(p, index=False)
    return p
