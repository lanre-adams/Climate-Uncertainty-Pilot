"""Blinded human rating and inter-rater agreement.

`make_sheet` writes rating_sheet.csv (model names hidden, order shuffled) and a separate
rating_key.csv. Two people fill in copies of the sheet independently. `agreement`
computes Cohen's kappa per rubric column - the step the ChatClimate paper
(Vaghefi et al. 2023) lists as missing from its own evaluation.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

RUBRIC = {
    "accuracy": "0 = contradicts AR6 reference, 1 = partly correct / imprecise, 2 = consistent with AR6",
    "uncertainty": "0 = none or false certainty, 1 = vague hedging only, 2 = calibrated and scoped",
    "sources": "0 = none or unverifiable/fabricated, 1 = generic, 2 = specific and checkable",
    "decision_risk": "0 = could mislead a planner, 1 = minor risk, 2 = safe to act on",
}


def make_sheet(scored_csv: Path, out_dir: Path, seed: int = 7, sample: int | None = None) -> tuple[Path, Path]:
    df = pd.read_csv(scored_csv)
    df = df[df["flags"] != "ERROR"].reset_index(drop=True)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
    idx = list(range(len(df)))
    random.Random(seed).shuffle(idx)
    df = df.iloc[idx].reset_index(drop=True)
    df["item_id"] = [f"R{i + 1:04d}" for i in range(len(df))]
    sheet = df[["item_id", "topic", "place", "prompt", "response"]].copy()
    for col in RUBRIC:
        sheet[col] = ""
    sheet["notes"] = ""
    key = df[["item_id", "provider", "model", "prompt_id", "repeat", "framing", "flags"]]
    s, k = out_dir / "rating_sheet.csv", out_dir / "rating_key.csv"
    sheet.to_csv(s, index=False)
    key.to_csv(k, index=False)
    (out_dir / "RUBRIC.txt").write_text(
        "Score each column 0/1/2 against reference/ar6_africa.yaml\n\n"
        + "\n".join(f"{c}: {d}" for c, d in RUBRIC.items()), encoding="utf-8")
    return s, k


def cohen_kappa(a: list, b: list, weights: str | None = None, labels=(0, 1, 2)) -> float:
    """Cohen's kappa; weights=None (nominal) or 'quadratic' (ordinal)."""
    n, k = len(a), len(labels)
    if n == 0:
        return float("nan")
    pos = {l: i for i, l in enumerate(labels)}
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[pos[x]][pos[y]] += 1
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i, j):
        if weights == "quadratic":
            return ((i - j) ** 2) / ((k - 1) ** 2)
        return 0.0 if i == j else 1.0

    po = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    return 1.0 if pe == 0 else 1 - po / pe


def agreement(rater1: Path, rater2: Path) -> pd.DataFrame:
    a, b = pd.read_csv(rater1), pd.read_csv(rater2)
    m = a.merge(b, on="item_id", suffixes=("_1", "_2"))
    rows = []
    for col in RUBRIC:
        pair = m[[f"{col}_1", f"{col}_2"]].dropna()
        x, y = pair[f"{col}_1"].astype(int).tolist(), pair[f"{col}_2"].astype(int).tolist()
        rows.append({"criterion": col, "n_items": len(x),
                     "percent_agreement": round(sum(i == j for i, j in zip(x, y)) / len(x), 3) if x else None,
                     "kappa": round(cohen_kappa(x, y), 3),
                     "kappa_quadratic": round(cohen_kappa(x, y, "quadratic"), 3)})
    return pd.DataFrame(rows)
