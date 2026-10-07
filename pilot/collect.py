"""Collect answers: every question x place x framing x repeat, for each provider."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import yaml

from .providers import MockProvider, ProviderError, get_provider

PKG = Path(__file__).parent


def load_bank(path: Path | None = None) -> dict:
    return yaml.safe_load((path or PKG / "questions.yaml").read_text(encoding="utf-8"))


def build_prompts(bank: dict) -> list[dict]:
    items = []
    for q in bank["questions"]:
        for place in bank["places"]:
            question = q["text"].format(place=place["name"])
            for fr in bank["framings"]:
                items.append({
                    "prompt_id": f"{q['id']}|{place['id']}|{fr['id']}",
                    "question_id": q["id"], "topic": q["topic"],
                    "place_id": place["id"], "place": place["name"],
                    "framing": fr["id"],
                    "prompt": fr["template"].format(question=question),
                })
    return items


def collect(providers: list[str], repeats: int, temperature: float, out_dir: Path,
            bank_path: Path | None = None, limit: int | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts(load_bank(bank_path))
    if limit:
        prompts = prompts[:limit]
    out = out_dir / "responses.jsonl"
    done = set()
    if out.exists():  # resume an interrupted run instead of paying twice
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["provider"], r["prompt_id"], r["repeat"]))

    total = len(providers) * len(prompts) * repeats
    n = 0
    with out.open("a", encoding="utf-8") as fh:
        for pname in providers:
            provider = get_provider(pname)
            for item in prompts:
                for rep in range(repeats):
                    n += 1
                    if (provider.name, item["prompt_id"], rep) in done:
                        continue
                    try:
                        if isinstance(provider, MockProvider):
                            c = provider.complete(item["prompt"], temperature=temperature, repeat=rep,
                                                  topic=item["topic"], place=item["place"])
                        else:
                            c = provider.complete(item["prompt"], temperature=temperature, repeat=rep)
                        rec = {**item, "provider": provider.name, "model": c.model, "repeat": rep,
                               "temperature": temperature, "response": c.text,
                               "latency_s": round(c.latency_s, 2), "error": None,
                               "synthetic": provider.name.startswith("mock")}
                    except ProviderError as exc:
                        rec = {**item, "provider": provider.name, "model": None, "repeat": rep,
                               "temperature": temperature, "response": "", "latency_s": None,
                               "error": str(exc), "synthetic": False}
                    rec["collected_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    if n % 25 == 0 or n == total:
                        print(f"  {n}/{total} collected")
    return out
