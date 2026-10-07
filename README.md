# LLM Climate Uncertainty Pilot (v0.1.0)

A small, reproducible harness that asks large language models about projected climate
change for three Nigerian places (Lagos, Uyo, Kano), and records **what they say, how they
represent uncertainty, which sources they lean on, and whether repeated runs agree**.
Answers are checked against statements from the IPCC AR6 WGI Regional Fact Sheet for
Africa, first by transparent automatic proxies and then by blinded human raters whose
agreement is measured.

It was built as preparation for the UNRISK CDT project *AI in climate information and
communication: What's the benefit? What's the danger? What do people trust?*
(University of Leeds), whose suggested starting questions it operationalises.

 The study design choices, the question bank, the decision to
> benchmark against AR6 calibrated language, running the collection, rating the answers
> and interpreting the results are the author's responsibility. Say so whenever you
> show or describe this work.

---

## What it measures

| Question from the project advert | How the pilot measures it |
|---|---|
| What does AI generate about regional projections? | 7 questions × 3 places × 4 framings = 84 prompts, repeated (default 3×) per model |
| What sources does it prioritise? | Named sources, URLs and author–year citations; citations flagged for checking |
| How do answers characterise uncertainty? | IPCC likelihood and confidence terms, hedges, scenarios, baselines, ranges (0–5 proxy) |
| Is the information accurate? | Blinded human rating against `pilot/reference/ar6_africa.yaml`, with Cohen's kappa |
| How does output change with prompting and user context? | Four framings: plain, planner, lay, cite-your-sources |
| Is it stable? | Run-to-run agreement on the stated direction of change |

Review flags (e.g. `OVERCONFIDENT_RAIN_DIRECTION`, `POINT_NUMBER_NO_RANGE`,
`CITATION_TO_VERIFY`, `ONE_SIDED_DECISION`) point raters at answers worth checking. **They
are not accuracy judgements.**

## Run it with Docker

Needs Docker Desktop (Windows/macOS) or Docker Engine (Linux).

```bash
# 1. Offline test - no keys, no cost. Uses SYNTHETIC mock answers.
docker compose build
docker compose run --rm tests          # unit + end-to-end tests
docker compose run --rm pilot          # full pipeline on mock data -> data/runs/demo_mock/

# 2. Real models
cp .env.example .env                   # Windows: copy .env.example .env  - then fill in keys + model IDs
docker compose run --rm pilot collect --providers anthropic --repeats 1 --limit 5 --run smoke   # cheap check
docker compose run --rm pilot all --providers anthropic,openai,gemini --repeats 3 --run pilot1
```

Open `data/runs/<run>/report/report.html` in a browser.

**Free option:** install [Ollama](https://ollama.com), pull an open model, set
`OLLAMA_MODEL` in `.env`, and use `--providers ollama`.

**Cost:** a full run is 84 prompts × repeats per provider (252 calls at 3 repeats).
Always run `--limit 5` first. Collection resumes if interrupted, so nothing is paid twice.

## Run it without Docker

```bash
pip install -r requirements.txt
python -m pytest -q
python -m pilot all --providers mock,mock-b --run demo_mock
```

## Human rating (the part that turns proxies into findings)

```bash
docker compose run --rm pilot ratesheet --run pilot1 --sample 60
```

1. Give `rating_sheet.csv` to two raters (model names are hidden and order is shuffled).
2. Each scores `accuracy`, `uncertainty`, `sources`, `decision_risk` as 0/1/2 using `RUBRIC.txt`
   and the AR6 reference file. Raters work independently.
3. Save as `rater1.csv` and `rater2.csv` in the run folder, then:

```bash
docker compose run --rm pilot agree --run pilot1 --r1 data/runs/pilot1/rater1.csv --r2 data/runs/pilot1/rater2.csv
docker compose run --rm pilot report --run pilot1
```

`rating_key.csv` maps item IDs back to models. Keep it away from the raters until rating is done.

## Files

```
pilot/questions.yaml            question bank, places, framings (edit freely)
pilot/reference/ar6_africa.yaml expert reference statements with calibrated qualifiers
pilot/providers.py              Anthropic, OpenAI, Gemini, Ollama, and the synthetic mock
pilot/collect.py                runs the prompt grid, resumable, writes responses.jsonl
pilot/score.py                  automatic features, review flags, consistency
pilot/rating.py                 blinded rating sheet, Cohen's kappa (nominal + quadratic)
pilot/report.py                 tables, charts, single-file HTML report
tests/test_pilot.py             11 tests including an end-to-end mock run
```

## Honest limits

- **Mock output is synthetic.** `demo_mock` exists only to prove the pipeline runs. Reports
  that contain mock data carry a banner saying so.
- **The reference file is a starting point, not a validated benchmark.** It paraphrases the
  AR6 Africa fact sheet. Before scoring, confirm each statement and which AR6 region each
  place falls in, using the Interactive Atlas and the cited WGI chapters.
- **The regex proxies are crude.** They count words. Negation, sarcasm and paraphrase defeat
  them. That is why they only route answers to humans.
- **API answers are not app answers.** People use chat apps with system prompts, memory and
  web search that the API does not reproduce. Results describe API behaviour at one date.
- **Models change.** Record model IDs and dates (the harness does) and expect drift.
- **No human-subjects work yet.** The project's second half - how people interpret and trust
  these answers - needs ethical approval and a proper study design. This pilot only
  prepares the stimuli.

## Natural next steps

- Add an IPCC-grounded retrieval condition (as in ChatClimate; Vaghefi et al. 2023,
  *Communications Earth & Environment* 4:480) and compare it with the bare models.
- Use rated answers as stimuli in a small trust experiment: does calibrated language,
  or a citation, change how credible a planner finds an answer?


