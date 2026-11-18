"""Summary tables, charts and a single-file HTML report, written in plain English.

Every chart has a plain title, a "what this shows" line and a "how to read it" line.
Technical IDs from the scorer are translated through pilot/labels.py.
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import labels as L  # noqa: E402
from .score import consistency  # noqa: E402

INK, MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#7c3aed", "#dc2626"]
FRAMING_ORDER = ["plain", "planner", "lay", "cite"]


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _axes(ax, xlabel=None, ylabel=None, horizontal=False):
    ax.grid(axis="x" if horizontal else "y", color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def _bars(table: pd.DataFrame, xlabel: str, xmax=None, horizontal=False, fmt="{:.0f}") -> str:
    """Grouped bars: rows = categories, columns = models. Values labelled on the bars."""
    n = len(table.columns)
    width = 0.8 / max(n, 1)
    if horizontal:
        table = table.iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(table.index) * max(1, n * 0.6) + 1.0))
        for i, col in enumerate(table.columns):
            ys = [y + (n - 1 - i) * width - 0.4 + width / 2 for y in range(len(table.index))]
            bars = ax.barh(ys, table[col].values, height=width * 0.9, label=str(col), color=PALETTE[i % len(PALETTE)])
            for b, v in zip(bars, table[col].values):
                ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(v), va="center", fontsize=7.5, color=INK)
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)
        if xmax:
            ax.set_xlim(0, xmax)
        _axes(ax, xlabel=xlabel, horizontal=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for i, col in enumerate(table.columns):
            xs = [x + i * width - 0.4 + width / 2 for x in range(len(table.index))]
            bars = ax.bar(xs, table[col].values, width=width * 0.9, label=str(col), color=PALETTE[i % len(PALETTE)])
            for b, v in zip(bars, table[col].values):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(table.index)))
        ax.set_xticklabels(table.index, fontsize=8.5)
        if xmax:
            ax.set_ylim(0, xmax)
        _axes(ax, ylabel=xlabel)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=min(n, 3), borderaxespad=0.3)
    return _png(fig)


def _rename(t: pd.DataFrame, index_fn=None) -> pd.DataFrame:
    t = t.copy()
    t.columns = [L.provider(c) for c in t.columns]
    if index_fn:
        t.index = [index_fn(i) for i in t.index]
    return t


def _section(title, what, how, chart=None, table=None, extra="") -> str:
    out = [f"<section><h2>{html.escape(title)}</h2>",
           f"<p class='what'><b>What this shows:</b> {html.escape(what)}</p>",
           f"<p class='how'><b>How to read it:</b> {html.escape(how)}</p>"]
    if chart:
        out.append(f"<figure><img alt='{html.escape(title)}' src='data:image/png;base64,{chart}'/></figure>")
    if table is not None:
        out.append("<details><summary>Show the numbers</summary>" + table.to_html(classes="t", border=0) + "</details>")
    out.append(extra + "</section>")
    return "".join(out)


def _examples(ok: pd.DataFrame, k: int = 3) -> str:
    """A few flagged answers, so a reader can see what the flags mean in practice."""
    shown, rows = set(), []
    for _, r in ok.iterrows():
        fl = [f for f in str(r["flags"]).split(";") if f and f not in shown and f != "NO_SOURCE"]
        if not fl:
            continue
        f = fl[0]
        shown.add(f)
        rows.append(f"<div class='ex'><div class='exhead'>{html.escape(L.flag(f))} · {html.escape(L.provider(r['provider']))} · "
                    f"{html.escape(L.topic(r['topic']))} · {html.escape(L.framing(r['framing']))}</div>"
                    f"<div class='exq'><b>Question:</b> {html.escape(str(r['prompt']))}</div>"
                    f"<div class='exa'><b>Answer:</b> {html.escape(str(r['response']))}</div>"
                    f"<div class='exwhy'>{html.escape(L.flag_help(f))}</div></div>")
        if len(rows) >= k:
            break
    return "".join(rows)


def _join(items: list[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _places(ok: pd.DataFrame) -> list[str]:
    order = {"lagos": 0, "uyo": 1, "kano": 2}
    ids = sorted(ok["place_id"].dropna().unique(), key=lambda x: order.get(x, 9)) if "place_id" in ok else []
    names = {r["place_id"]: str(r["place"]).split(",")[0] for _, r in ok.drop_duplicates("place_id").iterrows()} if len(ids) else {}
    return [names.get(i, i) for i in ids]


def _about(ok: pd.DataFrame, repeats: int, models: str) -> str:
    """Describe what this particular run actually covered."""
    if not len(ok):
        return "No answers were collected in this run."
    n_q = ok["prompt_id"].nunique()
    topics = [L.TOPIC_PHRASE[t] for t in L.TOPICS if t in set(ok["topic"])]
    frs = [f for f in FRAMING_ORDER if f in set(ok["framing"])]
    ways = {"plain": "as a plain question", "planner": "as a local planner",
            "lay": "asking for a simple explanation", "cite": "asking for sources"}
    n_all = len(L.TOPICS) * 3 * len(FRAMING_ORDER)
    scope = ("This is the full set of questions." if n_q >= n_all else
             f"This is a sample of {n_q} of the {n_all} possible questions, so the charts below are a quick look, "
             "not a full comparison.")
    ways_txt = (f"Questions were asked in {len(frs)} way{'s' if len(frs) > 1 else ''} "
                f"({_join([ways[f] for f in frs])}).")
    rep_txt = (f"Each question was sent {repeats} times, to see whether the answer changes when asked again."
               if repeats > 1 else
               "Each question was sent once, so this run cannot show whether answers change when asked again.")
    return html.escape(
        f"This run sent {n_q} question{'s' if n_q != 1 else ''} to {models}, about "
        f"{_join(topics)} in {_join(_places(ok))}. {ways_txt} {rep_txt} {scope} "
        "The answers were checked automatically for how they describe uncertainty and what sources they name. "
        "These checks only point to answers worth reading. Whether an answer is ") + "<i>correct</i>" + html.escape(
        " is judged by people, against the IPCC's assessment for Africa, using the blinded rating sheet.")


def _takeaways(cal_raw: pd.DataFrame, unc: pd.DataFrame, flags_t: pd.DataFrame, cons_t: pd.DataFrame,
               counts: pd.DataFrame | None = None) -> list[str]:
    """Plain, descriptive sentences generated from the tables. Descriptions, not conclusions."""
    out = []
    def n_of(m, f):
        if counts is None or f not in counts.index or m not in counts.columns:
            return " of answers"
        k = int(counts.loc[f, m])
        return f" of {k} answer{'s' if k != 1 else ''}"
    for m in cal_raw.columns:
        col = cal_raw[m].dropna()
        if col.empty:
            continue
        lo, hi = col.idxmin(), col.idxmax()
        if lo == hi:
            out.append(f"{L.provider(m)} used the scientists' standard confidence words in {col.max():.0f}%{n_of(m, hi)}.")
            continue
        out.append(f"{L.provider(m)} used the scientists' standard confidence words most when "
                   f"{L.FRAMING_PHRASE.get(hi, hi)} ({col.max():.0f}%{n_of(m, hi)}) and least when "
                   f"{L.FRAMING_PHRASE.get(lo, lo)} ({col.min():.0f}%{n_of(m, lo)}).")
    if len(flags_t):
        top = flags_t.mean(axis=1).idxmax()
        out.append(f"The most common warning sign was \"{top}\", in about {flags_t.loc[top].mean():.0f}% of answers.")
    if len(cons_t):
        worst = cons_t.mean(axis=1).idxmin()
        out.append(f"Answers changed most between repeats on \"{worst}\" "
                   f"(the same direction {cons_t.loc[worst].mean():.0f}% of the time when asked again).")
    return out


def build(scored_csv: Path, out_dir: Path, agreement_csv: Path | None = None) -> Path:
    df = pd.read_csv(scored_csv)
    ok = df[df["flags"] != "ERROR"].copy()
    ok["flags"] = ok["flags"].fillna("")
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic = bool(ok.get("synthetic", pd.Series([False])).astype(str).str.lower().eq("true").any())
    n_err = int((df["flags"] == "ERROR").sum())
    fr_order = [f for f in FRAMING_ORDER if f in set(ok["framing"])]

    sections, csvs = [], {}

    # 1. Calibrated language
    t = ok.groupby(["provider", "framing"])["has_calibrated"].mean().unstack("provider").reindex(fr_order)
    cal = _rename(t * 100, L.framing).round(0)
    csvs["1_standard_confidence_words_by_question_style"] = cal
    sections.append(_section(
        "1. Do the answers use scientists' standard confidence words?",
        "The share of answers (%) that use the IPCC's standard words for how sure scientists are, such as "
        "\"likely\", \"very likely\" or \"high confidence\", for each way the question was asked.",
        "Higher is better: these words tell a reader how much to trust a statement. Watch for drops when the "
        "question comes from a planner or asks for a simple explanation, because that is when the uncertainty matters most.",
        _bars(cal, "% of answers", 100), cal))

    # 2. Uncertainty score
    t2 = ok.groupby(["provider", "framing"])["uncertainty_proxy"].mean().unstack("provider").reindex(fr_order)
    unc = _rename(t2, L.framing).round(1)
    csvs["2_uncertainty_score_by_question_style"] = unc
    sections.append(_section(
        "2. How much of the uncertainty is explained? (score 0 to 5)",
        "An average score from 0 to 5. An answer gets one point each for: a standard confidence word; a cautious "
        "word such as \"may\" or \"uncertain\"; saying it depends on future emissions; saying what it is compared "
        "with (e.g. pre-industrial times); and giving a range (\"between X and Y\").",
        "Higher means the answer tells the reader more about how uncertain the projection is. It counts words, "
        "so it is a rough guide that points people to answers worth reading, not a mark for quality.",
        _bars(unc, "average score (0–5)", 5, fmt="{:.1f}"), unc))

    # 3. Warning signs
    exploded = ok.assign(flag=ok["flags"].str.split(";")).explode("flag")
    exploded = exploded[exploded["flag"].astype(bool)]
    if len(exploded):
        t3 = (exploded.groupby(["flag", "provider"]).size() / ok.groupby("provider").size()).unstack("provider").fillna(0)
        t3 = t3.loc[t3.mean(axis=1).sort_values(ascending=False).index]
        flags_t = _rename(t3 * 100, L.flag).round(0)
    else:
        flags_t = pd.DataFrame()
    csvs["3_warning_signs"] = flags_t
    legend = "".join(f"<li><b>{html.escape(n)}:</b> {html.escape(h)}</li>" for _, (n, h) in L.FLAGS.items())
    sections.append(_section(
        "3. Warning signs found in the answers",
        "The share of answers (%) with each warning sign. Warning signs are found automatically and mark answers "
        "a person should check.",
        "Lower is better. A warning sign is not proof the answer is wrong; the accuracy check is done by people "
        "using the blinded rating sheet.",
        _bars(flags_t.head(10), "% of answers", horizontal=True) if len(flags_t) else None, flags_t if len(flags_t) else None,
        f"<details><summary>What each warning sign means</summary><ul class='legend'>{legend}</ul></details>"
        f"<h3>Examples of answers with warning signs</h3>{_examples(ok)}"))

    # 4. Consistency
    cons = consistency(ok)
    t4 = cons.groupby(["provider", "topic"])["direction_agreement"].mean().unstack("provider")
    t4 = t4.reindex([k for k in L.TOPICS if k in t4.index])
    cons_t = _rename(t4 * 100, L.topic).round(0)
    repeats = int(ok.groupby(["provider", "prompt_id"]).size().max()) if len(ok) else 1
    if repeats > 1:
        csvs["4_same_answer_when_asked_again"] = cons_t
        sections.append(_section(
            "4. Does the model give the same answer when asked again?",
            f"Each question was sent {repeats} times. This shows how often (%) the repeats agreed on the direction "
            "of change (increase, decrease, or both).",
            "100% means the model always said the same thing. Lower means it changed its answer just from being asked "
            "again, which a planner would not see if they asked only once.",
            _bars(cons_t, "% of repeats that agree", 105, horizontal=True), cons_t))
    else:
        cons_t = pd.DataFrame()
        sections.append(_section(
            "4. Does the model give the same answer when asked again?",
            "Not measured in this run: each question was sent only once.",
            "To see whether answers change when the same question is asked again, run it again with "
            "\"Times to ask each question\" set to 2 or 3."))

    # 5. Sources
    src = ok.assign(src=ok["named_sources"].fillna("").str.split(";")).explode("src")
    src = src[src["src"].astype(bool)]
    if len(src):
        t5 = src.groupby(["src", "provider"]).size().unstack("provider").fillna(0).astype(int)
        t5 = t5.loc[t5.sum(axis=1).sort_values(ascending=False).index]
        src_t = _rename(t5, L.source)
    else:
        src_t = pd.DataFrame()
    csvs["5_sources_named"] = src_t
    sections.append(_section(
        "5. Which sources do the answers mention?",
        "How many answers mention each well-known source.",
        "Mentioning the IPCC is a good sign but does not prove the answer matches what the IPCC says. Any "
        "author-and-year citations are flagged separately so a person can check they exist.",
        _bars(src_t, "number of answers", horizontal=True) if len(src_t) else None, src_t if len(src_t) else None))

    if agreement_csv and agreement_csv.exists():
        ag = pd.read_csv(agreement_csv).set_index("criterion")
        csvs["6_rater_agreement"] = ag
        sections.append(_section(
            "6. Do the two human raters agree?",
            "Cohen's kappa for each rating criterion: how much the two raters agree beyond chance.",
            "1 = perfect agreement, 0 = no better than chance. Above about 0.6 is usually taken as good agreement.",
            None, ag))

    for name, tbl in csvs.items():
        if len(tbl):
            tbl.to_csv(out_dir / f"{name}.csv")

    counts = ok.groupby(["provider", "framing"]).size().unstack("provider") if len(ok) else None
    take = _takeaways((t * 100).round(0), unc, flags_t, cons_t, counts) if len(ok) else []
    smallest = int(counts.min().min()) if counts is not None and counts.size else 0
    if take and smallest < 10:
        take.insert(0, f"Small sample: some question styles have only {smallest} answer{'s' if smallest != 1 else ''} "
                       "per model, so these percentages can swing a lot. Treat them as a rough first look.")
    models = _join([L.provider(p) for p in sorted(ok["provider"].unique())]) if len(ok) else "no models"
    banner = ("<div class='warn'><b>Demo data: these numbers are made up.</b> This report uses pretend models "
              "built to test the tool. It says nothing about any real AI model. Use it only to show how the "
              "report works.</div>") if synthetic else ""
    err = (f"<div class='err'><b>{n_err} question(s) failed</b> (for example, a missing key or network "
           "problem) and are left out below.</div>") if n_err else ""
    summary = "".join(f"<li>{html.escape(s)}</li>" for s in take) or "<li>Not enough answers to summarise.</li>"
    about = _about(ok, repeats, models)
    places = ", ".join(_places(ok))
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Climate AI Answers Report</title><style>
:root{{--ink:#1f2933;--muted:#5b6370;--line:#e5e7eb;--bg:#ffffff;--warn:#fef3c7;--soft:#f5f7fa}}
body{{font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);background:var(--bg);max-width:900px;margin:0 auto;padding:16px}}
h1{{font-size:24px;margin:8px 0 4px}} h2{{font-size:18px;margin:0 0 6px}} h3{{font-size:15px;margin:18px 0 8px}}
section{{border-top:1px solid var(--line);padding:22px 0}}
.meta{{color:var(--muted);margin:0 0 10px}} .warn{{background:var(--warn);padding:10px 12px;border-radius:8px;margin:12px 0}}
.err{{background:#fde2e2;padding:10px 12px;border-radius:8px;margin:12px 0}}
.what,.how{{margin:4px 0}} .how{{color:var(--muted)}}
.box{{background:var(--soft);border-radius:8px;padding:12px 16px;margin:14px 0}}
img{{max-width:100%}} figure{{margin:14px 0}}
details{{margin:8px 0}} summary{{cursor:pointer;color:#2a78d6}}
.t{{border-collapse:collapse;font-size:13px;margin-top:8px;display:block;overflow-x:auto}}
.t th,.t td{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:right}}
.t th:first-child,.t td:first-child{{text-align:left}}
.legend li{{margin:4px 0}}
.ex{{border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0}}
.exhead{{font-weight:600;font-size:13px;color:#b45309}} .exq,.exa{{margin:4px 0;font-size:14px}}
.exwhy{{color:var(--muted);font-size:13px}}
</style></head><body>
<h1>What do AI models say about climate change in Nigeria?</h1>
<p class="meta">{len(ok)} answers from {html.escape(models)} · {ok['prompt_id'].nunique()} different questions ·
places: {html.escape(places)}</p>
{banner}{err}
<div class="box"><b>About this report.</b> {about}</div>
<div class="box"><b>In brief</b><ul>{summary}</ul></div>
{''.join(sections)}
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(doc, encoding="utf-8")
    return path
