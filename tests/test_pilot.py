from pathlib import Path

import pandas as pd
import pytest

from pilot import collect as C
from pilot import rating as R
from pilot import report as P
from pilot import score as S
from pilot.providers import MockProvider, ProviderError, get_provider


def test_prompt_grid_size():
    bank = C.load_bank()
    prompts = C.build_prompts(bank)
    assert len(prompts) == len(bank["questions"]) * len(bank["places"]) * len(bank["framings"])
    assert len({p["prompt_id"] for p in prompts}) == len(prompts)
    assert all("{" not in p["prompt"] for p in prompts)


def test_likelihood_terms_not_double_counted():
    terms = S.likelihood_terms("It is very likely, and likely, but unlikely to be virtually certain.")
    assert sorted(terms) == sorted(["very likely", "likely", "unlikely", "virtually certain"])


def test_confidence_and_scenario_detection():
    f = S.features("Hot extremes will increase (high confidence) under SSP5-8.5 relative to 1850-1900.")
    assert f["has_calibrated"] and f["has_scenario"] and f["has_baseline"]
    assert f["direction"] == "increase"


def test_overconfident_rain_flagged_and_hedged_rain_not():
    bad = "Annual rainfall in Lagos will decrease by 20% by 2050."
    good = "Annual rainfall in Lagos may increase or decrease; models disagree."
    assert "OVERCONFIDENT_RAIN_DIRECTION" in S.flags("mean_precipitation", bad, S.features(bad))
    assert "OVERCONFIDENT_RAIN_DIRECTION" not in S.flags("mean_precipitation", good, S.features(good))


def test_point_number_and_fabricated_citation_flags():
    t = "Sea level will rise exactly 2 metres by 2050 (Adebayo et al. 2021)."
    fl = S.flags("sea_level", t, S.features(t))
    assert {"POINT_NUMBER_NO_RANGE", "CERTAINTY_LANGUAGE", "CITATION_TO_VERIFY", "NO_SCENARIO_DEPENDENCE"} <= set(fl)


def test_range_suppresses_point_number_flag():
    t = "IPCC AR6 projects 0.3-1.0 m of rise by 2100 depending on the emissions scenario."
    assert "POINT_NUMBER_NO_RANGE" not in S.flags("sea_level", t, S.features(t))


def test_one_sided_decision():
    t = "Prepare for drought only."
    assert "ONE_SIDED_DECISION" in S.flags("decision", t, S.features(t))
    t2 = "Prepare for both flooding and drought."
    assert "ONE_SIDED_DECISION" not in S.flags("decision", t2, S.features(t2))


def test_kappa_perfect_and_chance():
    assert R.cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == pytest.approx(1.0)
    assert R.cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)
    assert R.cohen_kappa([0, 1, 2], [0, 2, 2], "quadratic") > R.cohen_kappa([0, 1, 2], [0, 2, 2])


def test_mock_is_deterministic():
    m = MockProvider("a")
    a = m.complete("x", temperature=0.7, repeat=1, topic="sea_level", place="Lagos").text
    b = m.complete("x", temperature=0.7, repeat=1, topic="sea_level", place="Lagos").text
    assert a == b


def test_real_provider_without_key_errors(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_MODEL", "set-me")
    with pytest.raises(ProviderError):
        get_provider("anthropic").complete("hi", temperature=0, repeat=0)


def test_end_to_end_mock(tmp_path: Path):
    out = C.collect(["mock", "mock-b"], repeats=2, temperature=0.7, out_dir=tmp_path, limit=12)
    df = S.score_file(out, tmp_path / "scored.csv")
    assert len(df) == 2 * 12 * 2 and df["synthetic"].all()
    # resume: a second call adds nothing
    C.collect(["mock", "mock-b"], repeats=2, temperature=0.7, out_dir=tmp_path, limit=12)
    assert len(out.read_text().splitlines()) == 48
    sheet, key = R.make_sheet(tmp_path / "scored.csv", tmp_path)
    s = pd.read_csv(sheet)
    assert "provider" not in s.columns and len(s) == 48
    r1 = s.copy()
    r2 = s.copy()
    for col in R.RUBRIC:
        r1[col] = 2
        r2[col] = 2
    r2.loc[0, "accuracy"] = 1
    r1.to_csv(tmp_path / "r1.csv", index=False)
    r2.to_csv(tmp_path / "r2.csv", index=False)
    ag = R.agreement(tmp_path / "r1.csv", tmp_path / "r2.csv")
    ag.to_csv(tmp_path / "agreement.csv", index=False)
    html = P.build(tmp_path / "scored.csv", tmp_path / "report", tmp_path / "agreement.csv").read_text()
    assert "SYNTHETIC DATA" in html and "Cohen" in html
