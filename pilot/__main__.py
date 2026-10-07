"""Command line.

  python -m pilot all --providers mock,mock-b          # offline end-to-end test
  python -m pilot collect --providers anthropic,openai --repeats 3 --run my_run
  python -m pilot score --run my_run
  python -m pilot ratesheet --run my_run --sample 60
  python -m pilot agree --run my_run --r1 rater1.csv --r2 rater2.csv
  python -m pilot report --run my_run
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from . import collect as C
from . import rating as R
from . import report as P
from . import score as S

DATA = Path(__import__("os").environ.get("PILOT_DATA", "data"))


def run_dir(name: str | None) -> Path:
    return DATA / "runs" / (name or dt.datetime.now().strftime("run_%Y%m%d_%H%M"))


def main(argv=None):
    p = argparse.ArgumentParser(prog="pilot", description="LLM climate-uncertainty pilot")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--run", help="run folder name under data/runs/")
        return sp

    c = add("collect", "query the models")
    c.add_argument("--providers", default="mock")
    c.add_argument("--repeats", type=int, default=3)
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--limit", type=int, help="only the first N prompts (cheap smoke test)")
    add("score", "compute automatic features and review flags")
    rs = add("ratesheet", "write blinded rating sheet + key")
    rs.add_argument("--sample", type=int)
    ag = add("agree", "inter-rater agreement")
    ag.add_argument("--r1", required=True)
    ag.add_argument("--r2", required=True)
    add("report", "build report.html")
    a = add("all", "collect + score + ratesheet + report")
    a.add_argument("--providers", default="mock,mock-b")
    a.add_argument("--repeats", type=int, default=3)
    a.add_argument("--temperature", type=float, default=0.7)
    a.add_argument("--limit", type=int)
    a.add_argument("--sample", type=int)

    args = p.parse_args(argv)
    d = run_dir(args.run)
    d.mkdir(parents=True, exist_ok=True)

    if args.cmd in ("collect", "all"):
        print(f"Collecting into {d}")
        C.collect([s.strip() for s in args.providers.split(",") if s.strip()],
                  args.repeats, args.temperature, d, limit=args.limit)
    if args.cmd in ("score", "all"):
        df = S.score_file(d / "responses.jsonl", d / "scored.csv")
        print(f"Scored {len(df)} rows -> {d / 'scored.csv'}")
    if args.cmd in ("ratesheet", "all"):
        s, k = R.make_sheet(d / "scored.csv", d, sample=args.sample)
        print(f"Rating sheet -> {s}  (key kept separately: {k})")
    if args.cmd == "agree":
        t = R.agreement(Path(args.r1), Path(args.r2))
        t.to_csv(d / "agreement.csv", index=False)
        print(t.to_string(index=False))
    if args.cmd in ("report", "all"):
        path = P.build(d / "scored.csv", d / "report", d / "agreement.csv")
        print(f"Report -> {path}")


if __name__ == "__main__":
    main()
