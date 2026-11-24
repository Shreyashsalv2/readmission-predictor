"""Checks on the real UCI file; skipped unless scripts/get_data.sh has been run (CI runs it)."""
from pathlib import Path

import pytest

from readmit import data as D

CSV = Path(__file__).resolve().parents[1] / "data" / "diabetic_data.csv"
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="dataset not downloaded")


@pytest.fixture(scope="module")
def raw():
    return D.load_raw(CSV)


def test_shape_matches_uci(raw):
    assert raw.shape == (101766, 50)
    assert set(D.MEDICATIONS) <= set(raw.columns)
    assert set(raw["readmitted"].unique()) == {"<30", ">30", "NO"}


def test_cleaning_on_real_data(raw):
    c = D.clean(raw)
    assert len(c.df) == c.steps[-1][1]
    assert c.df["target"].between(0, 1).all()
    assert 0.05 < c.df["target"].mean() < 0.15
    assert c.df[c.numeric + c.categorical].notna().all().all()


def test_ids_mapping_groups(raw):
    """The discharge and admission-source groups use ids that exist in IDs_mapping.csv."""
    m = (CSV.parent / "IDs_mapping.csv").read_text()
    for i in list(D.DISCHARGE_GROUPS) + list(D.ADMISSION_SOURCE_GROUPS):
        assert f"\n{i}," in m
