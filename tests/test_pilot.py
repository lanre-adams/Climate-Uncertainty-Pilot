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
    assert "Demo data: these numbers are made up" in html and "Cohen" in html


def test_every_flag_has_a_plain_label():
    import inspect
    from pilot import labels as Lb
    src = inspect.getsource(S.flags)
    import re as _re
    ids = set(_re.findall(r'"([A-Z_]{6,})"', src))
    assert ids and ids <= set(Lb.FLAGS)


def test_web_demo_run_end_to_end(tmp_path, monkeypatch):
    import time
    from pilot import web as Wb
    monkeypatch.setattr(Wb, "RUNS", tmp_path / "runs")
    app = Wb.create_app().test_client()
    assert app.get("/").status_code == 200
    assert "anthropic" in app.get("/api/status").get_json()["providers"]
    r = app.post("/api/run", json={"mode": "demo", "limit": 5, "repeats": 2, "run": "t1"})
    assert r.status_code == 200
    for _ in range(100):
        j = app.get("/api/job").get_json()
        if j["state"] in ("finished", "failed"):
            break
        time.sleep(0.1)
    assert j["state"] == "finished", j
    runs = app.get("/api/runs").get_json()
    assert runs[0]["name"] == "t1" and runs[0]["demo"] and runs[0]["answers"] == 20
    assert "This run sent 5 questions" in app.get("/runs/t1/report.html").get_data(as_text=True)
    page = app.get("/runs/t1/report.html").get_data(as_text=True)
    assert "Demo data: these numbers are made up" in page and "What this shows" in page
    assert app.get("/runs/t1/secret.txt").status_code == 404
    assert app.get("/runs/..%2F..%2Fetc/report.html").status_code == 404


def test_web_real_mode_requires_key_and_model(tmp_path, monkeypatch):
    from pilot import web as Wb
    monkeypatch.setattr(Wb, "RUNS", tmp_path / "runs")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(Wb, "load_dotenv", lambda *a, **k: None)
    app = Wb.create_app().test_client()
    r = app.post("/api/run", json={"mode": "real", "providers": {"openai": {"use": True}}})
    assert r.status_code == 400 and "key" in r.get_json()["error"]
    r = app.post("/api/run", json={"mode": "real", "providers": {}})
    assert r.status_code == 400


def test_report_describes_the_run_actually_done(tmp_path):
    """The 'About this report' text must reflect the settings used, not the full design."""
    d = tmp_path / "r"
    C.collect(["mock-a"], 1, 0.7, d, limit=7, spread=True)
    S.score_file(d / "responses.jsonl", d / "scored.csv")
    html = P.build(d / "scored.csv", d / "report").read_text()
    assert "This run sent 7 questions" in html
    assert "sent once, so this run cannot show whether answers change" in html
    assert "sample of 7 of the 84 possible questions" in html
    assert "Not measured in this run" in html
    assert "four ways" not in html
    rows = [l for l in (d / "responses.jsonl").read_text().splitlines() if l]
    assert len(rows) == 7 and len({__import__("json").loads(r)["topic"] for r in rows}) == 7


def test_report_full_run_wording(tmp_path):
    d = tmp_path / "r"
    C.collect(["mock-a"], 2, 0.7, d)
    S.score_file(d / "responses.jsonl", d / "scored.csv")
    html = P.build(d / "scored.csv", d / "report").read_text()
    assert "This run sent 84 questions" in html and "full set of questions" in html
    assert "sent 2 times" in html and "Not measured in this run" not in html
    assert "in 4 ways" in html and "Lagos, Uyo and Kano" in html


def test_timeout_is_recorded_and_run_continues(tmp_path, monkeypatch):
    """A slow local model must not stop the run: the answer is logged as failed and collection goes on."""
    import json as _json
    import requests
    from pilot import providers as Pv

    calls = []

    def slow(*a, **k):
        calls.append(k.get("timeout"))
        raise requests.exceptions.ReadTimeout("Read timed out.")

    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setattr(Pv.requests, "post", slow)
    d = tmp_path / "r"
    C.collect(["ollama"], 1, 0.7, d, limit=3, spread=True)
    rows = [_json.loads(l) for l in (d / "responses.jsonl").read_text().splitlines() if l]
    assert len(rows) == 3 and all("No answer within" in r["error"] for r in rows)
    assert calls and all(t == Pv.OLLAMA_TIMEOUT for t in calls)


def test_connection_refused_is_a_clear_message(monkeypatch):
    import requests
    from pilot import providers as Pv

    def refused(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(Pv.requests, "post", refused)
    try:
        Pv._post("http://localhost:11434/api/chat", {}, {})
    except Pv.ProviderError as e:
        assert "Could not connect to http://localhost:11434" in str(e)
    else:
        raise AssertionError("expected ProviderError")


def test_report_names_exact_model_and_date(tmp_path):
    d = tmp_path / "r"
    C.collect(["mock-a"], 1, 0.7, d, limit=3, spread=True)
    S.score_file(d / "responses.jsonl", d / "scored.csv")
    html = P.build(d / "scored.csv", d / "report").read_text()
    assert "Exact models used: Demo model A (pretend): synthetic-a" in html and "Collected " in html
