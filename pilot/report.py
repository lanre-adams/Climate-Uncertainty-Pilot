"""Summary tables, charts and a single-file HTML report."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .score import consistency  # noqa: E402

INK, MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"
PALETTE = ["#2563eb", "#d97706", "#059669", "#7c3aed", "#dc2626"]


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def _grouped_bar(table: pd.DataFrame, title: str, ylabel: str, ylim=None, horizontal=False) -> str:
    n = len(table.columns)
    width = 0.8 / max(n, 1)
    if horizontal:
        table = table.iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(table.index) + 1.2))
        for i, col in enumerate(table.columns):
            ys = [y + i * width - 0.4 + width / 2 for y in range(len(table.index))]
            ax.barh(ys, table[col].values, height=width * 0.92, label=str(col), color=PALETTE[i % len(PALETTE)])
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel(ylabel, color=MUTED, fontsize=9)
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        return _png(fig)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, col in enumerate(table.columns):
        xs = [x + i * width - 0.4 + width / 2 for x in range(len(table.index))]
        ax.bar(xs, table[col].values, width=width * 0.92, label=str(col), color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(range(len(table.index)))
    ax.set_xticklabels(table.index, rotation=0)
    if ylim:
        ax.set_ylim(*ylim)
    _style(ax, title, ylabel)
    ax.legend(frameon=False, fontsize=8, ncol=min(n, 4))
    return _png(fig)


def build(scored_csv: Path, out_dir: Path, agreement_csv: Path | None = None) -> Path:
    df = pd.read_csv(scored_csv)
    ok = df[df["flags"] != "ERROR"].copy()
    ok["flags"] = ok["flags"].fillna("")
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic = bool(ok.get("synthetic", pd.Series([False])).any())

    tables, charts = {}, []

    t = ok.groupby(["provider", "framing"])["has_calibrated"].mean().unstack("provider").round(2)
    tables["Share of answers using IPCC calibrated language, by framing"] = t
    charts.append(("Calibrated language by framing", _grouped_bar(
        t, "Share of answers using IPCC calibrated language", "share", (0, 1))))

    t2 = ok.groupby(["provider", "framing"])["uncertainty_proxy"].mean().unstack("provider").round(2)
    tables["Mean uncertainty-ingredient count (0-5 proxy), by framing"] = t2
    charts.append(("Uncertainty ingredients by framing", _grouped_bar(
        t2, "Mean uncertainty-ingredient count (proxy, 0-5)", "mean count", (0, 5))))

    exploded = ok.assign(flag=ok["flags"].str.split(";")).explode("flag")
    exploded = exploded[exploded["flag"].astype(bool)]
    t3 = (exploded.groupby(["flag", "provider"]).size() / ok.groupby("provider").size()) \
        .unstack("provider").fillna(0).round(2).sort_values(by=list(ok["provider"].unique())[0], ascending=False)
    tables["Review flags per answer (rate)"] = t3
    charts.append(("Review-flag rates", _grouped_bar(t3.head(10), "Review-flag rate per answer", "rate", horizontal=True)))

    cons = consistency(ok)
    t4 = cons.groupby(["provider", "topic"])["direction_agreement"].mean().unstack("provider").round(2)
    tables["Run-to-run consistency of stated direction (1 = all repeats agree)"] = t4

    src = ok.assign(src=ok["named_sources"].fillna("").str.split(";")).explode("src")
    src = src[src["src"].astype(bool)]
    t5 = src.groupby(["src", "provider"]).size().unstack("provider").fillna(0).astype(int)
    tables["Sources named (count of answers)"] = t5

    if agreement_csv and agreement_csv.exists():
        tables["Inter-rater agreement (Cohen's kappa)"] = pd.read_csv(agreement_csv).set_index("criterion")

    for name, tbl in tables.items():
        slug = "".join(c if c.isalnum() else "_" for c in name.lower())[:60]
        tbl.to_csv(out_dir / f"{slug}.csv")

    banner = ("<div class='warn'><b>SYNTHETIC DATA.</b> This report includes mock-provider output, "
              "generated to test the pipeline. It says nothing about any real model.</div>") if synthetic else ""
    meta = (f"{len(ok)} answers · providers: {', '.join(sorted(ok['provider'].unique()))} · "
            f"{ok['prompt_id'].nunique()} prompts · errors: {int((df['flags'] == 'ERROR').sum())}")
    parts = [f"<h2>{html.escape(n)}</h2>{tbl.to_html(classes='t', border=0)}" for n, tbl in tables.items()]
    imgs = [f"<figure><img alt='{html.escape(n)}' src='data:image/png;base64,{b}'/></figure>" for n, b in charts]
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Climate Uncertainty Pilot</title><style>
:root{{--ink:#1f2933;--muted:#6b7280;--line:#e5e7eb;--bg:#ffffff;--warn:#fef3c7}}
body{{font:15px/1.5 system-ui,sans-serif;color:var(--ink);background:var(--bg);max-width:960px;margin:0 auto;padding:16px}}
h1{{font-size:22px;margin:8px 0}} h2{{font-size:16px;margin:28px 0 8px}}
.meta{{color:var(--muted)}} .warn{{background:var(--warn);padding:10px 12px;border-radius:6px;margin:12px 0}}
.note{{color:var(--muted);font-size:13px}} img{{max-width:100%}} figure{{margin:16px 0}}
.t{{border-collapse:collapse;font-size:13px;overflow-x:auto;display:block}}
.t th,.t td{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}}
.t th:first-child,.t td:first-child{{text-align:left}}
</style></head><body>
<h1>LLM Climate Uncertainty Pilot — run report</h1>
<p class="meta">{html.escape(meta)}</p>{banner}
<p class="note">Automatic measures are transparent proxies (counts of IPCC calibrated terms, hedges,
scenarios, baselines, ranges and named sources). They locate answers for human review; they do not
judge accuracy. Accuracy comes from the blinded rating sheet, checked against
reference/ar6_africa.yaml.</p>
{''.join(imgs)}{''.join(parts)}
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(doc, encoding="utf-8")
    return path
