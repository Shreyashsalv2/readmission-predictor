import numpy as np
import pandas as pd
import pytest

from readmit import data as D


@pytest.mark.parametrize("code,group", [
    ("428", "Circulatory"), ("459.81", "Circulatory"), ("785", "Circulatory"), ("486", "Respiratory"),
    ("786", "Respiratory"), ("250", "Diabetes"), ("250.83", "Diabetes"), ("536", "Digestive"),
    ("996", "Injury"), ("715", "Musculoskeletal"), ("599", "Genitourinary"), ("788", "Genitourinary"),
    ("174", "Neoplasms"), ("V57", "Other"), ("E888", "Other"), ("401", "Circulatory"), ("272", "Other"),
    (np.nan, "Missing"), ("", "Missing"), (None, "Missing"),
])
def test_icd9_groups(code, group):
    assert D.icd9_group(code) == group


def test_age_midpoint():
    assert D.age_midpoint("[70-80)") == 75.0
    assert D.age_midpoint("[0-10)") == 5.0


def test_load_marks_question_marks_missing_but_keeps_none(raw_csv):
    raw = D.load_raw(raw_csv)
    assert raw["race"].isna().any()                         # '?' -> NaN
    assert (raw["A1Cresult"] == "None").any()               # 'None' stays a value (test not done)


def test_cleaning_rules(raw_csv):
    raw = D.load_raw(raw_csv)
    c = D.clean(raw)
    df = c.df
    kept = raw[(raw["gender"] != "Unknown/Invalid") & ~raw["discharge_disposition_id"].isin(D.EXPIRED_OR_HOSPICE)]
    assert len(df) == kept["patient_nbr"].nunique()         # exactly one row per patient
    assert [n for _, n in c.steps] == sorted([n for _, n in c.steps], reverse=True)
    assert "Unknown/Invalid" not in set(df["gender"])
    assert set(df["target"].unique()) <= {0, 1}
    assert df[c.numeric].notna().all().all()
    assert df[c.categorical].notna().all().all()
    assert "Not measured" in set(df["A1Cresult"]) and "None" not in set(df["A1Cresult"])
    assert set(df["race"]) >= {"Missing"}
    assert "examide" in c.dropped_constant and "examide" not in c.categorical
    for col in ("weight", "payer_code", "encounter_id", "patient_nbr", "readmitted"):
        assert col not in df.columns


def test_first_encounter_is_kept(raw_csv):
    raw = D.load_raw(raw_csv)
    raw = raw[(raw["gender"] != "Unknown/Invalid") & ~raw["discharge_disposition_id"].isin(D.EXPIRED_OR_HOSPICE)]
    pid = raw["patient_nbr"].value_counts().index[0]                 # a repeat patient
    rows = raw[raw["patient_nbr"] == pid].sort_values("encounter_id")
    first = rows.iloc[0]
    c = D.clean(D.load_raw(raw_csv))
    # the cleaned row for that patient has the first visit's length of stay and target
    expected = (first["time_in_hospital"], int(first["readmitted"] == "<30"))
    match = c.df[(c.df["time_in_hospital"] == first["time_in_hospital"]) &
                 (c.df["number_inpatient"] == first["number_inpatient"]) &
                 (c.df["num_medications"] == first["num_medications"])]
    assert len(match) >= 1 and expected[1] in set(match["target"])


def test_target_is_only_under_30_days():
    raw = pd.DataFrame({"readmitted": ["<30", ">30", "NO"]})
    assert ((raw["readmitted"] == "<30").astype(int)).tolist() == [1, 0, 0]


def test_specialties_are_capped(raw_csv):
    c = D.clean(D.load_raw(raw_csv))
    assert c.df["medical_specialty"].nunique() <= D.TOP_SPECIALTIES + 2


def test_id_mapping_parser(tmp_path):
    f = tmp_path / "IDS_mapping.csv"
    f.write_text("admission_type_id,description\r\n1,Emergency\r\n2,Urgent\r\n,\r\n"
                 "discharge_disposition_id,description\r\n1,Discharged to home\r\n22,\"Discharged/transferred to another rehab fac\"\r\n,\r\n"
                 "admission_source_id,description\r\n7, Emergency Room\r\n")
    assert D.find_id_mapping(tmp_path) == f
    m = D.load_id_maps(f)
    assert m["admission_type_id"] == {"1": "Emergency", "2": "Urgent"}
    assert m["discharge_disposition_id"]["22"] == "Discharged/transferred to another rehab fac"
    assert m["admission_source_id"]["7"] == "Emergency Room"
