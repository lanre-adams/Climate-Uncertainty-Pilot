"""Automatic features and review flags.

These are TRANSPARENT PROXIES, not judgements of accuracy. They count what an answer
contains (IPCC calibrated language, hedges, scenarios, baselines, ranges, sources) and
raise flags that tell a human rater where to look. Accuracy is decided by human raters
using the rating sheet (see ratesheet.py), with agreement measured (agreement.py).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

PKG = Path(__file__).parent

# IPCC AR6 calibrated likelihood language (Mastrandrea et al. 2010 guidance note).
LIKELIHOOD = [
    "virtually certain", "extremely likely", "very likely", "about as likely as not",
    "exceptionally unlikely", "extremely unlikely", "very unlikely", "unlikely", "likely",
]
CONFIDENCE_RE = re.compile(r"\b(very high|high|medium|low|very low) confidence\b", re.I)
HEDGE_RE = re.compile(r"\b(may|might|could|possibly|uncertain|uncertainty|uncertainties|"
                      r"not clear|unclear|disagree|disagreement|depends on)\b", re.I)
CERTAIN_RE = re.compile(r"\b(definitely|certainly|exactly|guaranteed|fully certain|"
                        r"without doubt|will undoubtedly)\b", re.I)
SCENARIO_RE = re.compile(r"\b(SSP\d-\d\.\d|RCP\s?\d\.\d|emissions? (scenario|pathway)s?|"
                         r"warming levels?|\d(\.\d)?\s?°?C of (global )?warming)\b", re.I)
BASELINE_RE = re.compile(r"(1850\s?[-–]\s?1900|pre-?industrial|1995\s?[-–]\s?2014|"
                         r"1986\s?[-–]\s?2005|1981\s?[-–]\s?2010|relative to)", re.I)
RANGE_RE = re.compile(r"(\d+(\.\d+)?\s?[-–]\s?\d+(\.\d+)?\s?(%|m|cm|mm|°C|metres|meters)|"
                      r"between\s\d+(\.\d+)?\s?(%|m|cm|°C)?\sand\s\d)", re.I)
NUMBER_CLAIM_RE = re.compile(r"\b\d+(\.\d+)?\s?(%|percent|m\b|metres|meters|cm|°C)", re.I)
URL_RE = re.compile(r"https?://\S+")
AUTHOR_YEAR_RE = re.compile(r"\b[A-Z][a-z]+ et al\.?,?\s*\(?\d{4}\)?")
KNOWN_SOURCES = ["IPCC", "AR6", "CMIP6", "CORDEX", "NiMet", "Met Office", "WMO",
                 "Interactive Atlas", "World Bank", "Copernicus", "NASA", "NOAA"]
INCREASE_RE = re.compile(r"\b(increase|increases|increasing|more frequent|more intense|"
                         r"more common|wetter|rise|rising)\b", re.I)
DECREASE_RE = re.compile(r"\b(decrease|decreases|decreasing|less|fewer|drier|decline|declining)\b", re.I)
FLOOD_RE = re.compile(r"\bflood", re.I)
DROUGHT_RE = re.compile(r"\b(drought|drying|dry spells?)\b", re.I)


def likelihood_terms(text: str) -> list[str]:
    found, t = [], text.lower()
    for term in LIKELIHOOD:  # longest first so 'very likely' is not also counted as 'likely'
        pattern = r"\b" + re.escape(term) + r"\b"
        found += [term] * len(re.findall(pattern, t))
        t = re.sub(pattern, " ", t)
    return found


def direction(text: str) -> str:
    up, down = bool(INCREASE_RE.search(text)), bool(DECREASE_RE.search(text))
    return "mixed" if up and down else "increase" if up else "decrease" if down else "none"


def features(text: str) -> dict:
    lik = likelihood_terms(text)
    conf = [m.group(0).lower() for m in CONFIDENCE_RE.finditer(text)]
    return {
        "n_words": len(text.split()),
        "likelihood_terms": ";".join(lik),
        "confidence_terms": ";".join(conf),
        "has_calibrated": bool(lik or conf),
        "n_hedges": len(HEDGE_RE.findall(text)),
        "has_certainty_marker": bool(CERTAIN_RE.search(text)),
        "has_scenario": bool(SCENARIO_RE.search(text)),
        "has_baseline": bool(BASELINE_RE.search(text)),
        "has_range": bool(RANGE_RE.search(text)),
        "has_number": bool(NUMBER_CLAIM_RE.search(text)),
        "n_urls": len(URL_RE.findall(text)),
        "named_sources": ";".join(s for s in KNOWN_SOURCES if s.lower() in text.lower()),
        "author_year_citations": ";".join(AUTHOR_YEAR_RE.findall(text)),
        "direction": direction(text),
    }


def flags(topic: str, text: str, f: dict) -> list[str]:
    out = []
    hedged = f["n_hedges"] > 0 or f["has_calibrated"]
    if topic == "mean_precipitation" and f["direction"] in ("increase", "decrease") and not hedged:
        out.append("OVERCONFIDENT_RAIN_DIRECTION")  # AR6 gives no robust WAF mean-rain direction
    if topic in ("hot_extremes", "heavy_precipitation", "sea_level") and f["direction"] not in ("increase", "mixed"):
        out.append("MISSES_ROBUST_INCREASE")  # AR6: high confidence increase
    if topic in ("hot_extremes", "heavy_precipitation") and f["n_hedges"] > 0 and not f["has_calibrated"]:
        out.append("POSSIBLE_UNDERCONFIDENCE")
    if f["has_certainty_marker"]:
        out.append("CERTAINTY_LANGUAGE")
    if f["has_number"] and not f["has_range"]:
        out.append("POINT_NUMBER_NO_RANGE")
    if topic in ("sea_level", "general", "hot_extremes") and not f["has_scenario"]:
        out.append("NO_SCENARIO_DEPENDENCE")
    if topic == "decision":
        fl, dr = bool(FLOOD_RE.search(text)), bool(DROUGHT_RE.search(text))
        if fl != dr:
            out.append("ONE_SIDED_DECISION")
        if re.search(r"\b(flood\w*|drought)\s+(will not|won't|is not|isn't)\b|\b(flood\w*|drought) only\b", text, re.I):
            out.append("DISMISSES_A_HAZARD")
    if topic == "uncertainty" and not (f["has_scenario"] or re.search(r"model|variabilit", text, re.I)):
        out.append("NO_UNCERTAINTY_SOURCES")
    if f["author_year_citations"]:
        out.append("CITATION_TO_VERIFY")
    if not (f["named_sources"] or f["n_urls"] or f["author_year_citations"]):
        out.append("NO_SOURCE")
    return out


def uncertainty_proxy(f: dict) -> int:
    """0-5 count of uncertainty-communication ingredients. A proxy, not a quality score."""
    return int(f["has_calibrated"]) + int(f["n_hedges"] > 0) + int(f["has_scenario"]) \
        + int(f["has_baseline"]) + int(f["has_range"])


def score_file(responses: Path, out_csv: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in responses.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = []
    for r in rows:
        if r.get("error"):
            recs.append({**r, "flags": "ERROR"})
            continue
        f = features(r["response"])
        recs.append({**r, **f, "uncertainty_proxy": uncertainty_proxy(f),
                     "flags": ";".join(flags(r["topic"], r["response"], f))})
    df = pd.DataFrame(recs)
    df.to_csv(out_csv, index=False)
    return df


def consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Do repeated runs of the same prompt agree on direction? (share of modal answer)."""
    ok = df[df["error"].isna()] if "error" in df else df
    g = ok.groupby(["provider", "prompt_id", "topic", "framing"])["direction"]
    out = g.agg(lambda s: Counter(s).most_common(1)[0][1] / len(s)).rename("direction_agreement")
    return out.reset_index()


def load_reference() -> dict:
    return yaml.safe_load((PKG / "reference" / "ar6_africa.yaml").read_text(encoding="utf-8"))
