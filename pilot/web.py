"""A small local web interface: choose demo or real models, run the pilot, read the report.

    python -m pilot web                 # then open http://127.0.0.1:8000
    docker compose up web               # same, inside Docker

It runs on your own computer only. API keys typed into the page are held in memory for
this session and are never written to disk.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading
import traceback
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

from . import collect as C
from . import labels as L
from . import rating as R
from . import report as P
from . import score as S

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("PILOT_DATA", ROOT / "data"))
RUNS = DATA / "runs"
SAFE = re.compile(r"^[A-Za-z0-9_\-]{1,60}$")
DOWNLOADS = {"report.html": "report/report.html", "answers.csv": "scored.csv",
             "rating_sheet.csv": "rating_sheet.csv", "RUBRIC.txt": "RUBRIC.txt",
             "answers.jsonl": "responses.jsonl"}

REAL = {
    "anthropic": {"key": "ANTHROPIC_API_KEY", "model": "ANTHROPIC_MODEL"},
    "openai": {"key": "OPENAI_API_KEY", "model": "OPENAI_MODEL"},
    "gemini": {"key": "GEMINI_API_KEY", "model": "GEMINI_MODEL"},
    "ollama": {"key": None, "model": "OLLAMA_MODEL", "host": "OLLAMA_HOST"},
}


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines from .env if present (without overriding real environment variables)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _has(var: str | None) -> bool:
    v = os.environ.get(var or "", "").strip()
    return bool(v) and not v.lower().startswith("set-me")


class Job:
    lock = threading.Lock()
    current: "Job | None" = None

    def __init__(self, run: str, providers: list[str], repeats: int, limit: int | None):
        self.run, self.providers, self.repeats, self.limit = run, providers, repeats, limit
        self.state, self.step, self.done, self.total, self.errors = "starting", "Preparing", 0, 0, 0
        self.message, self.last_error, self.started = "", "", dt.datetime.now().isoformat(timespec="seconds")

    def as_dict(self):
        return {k: getattr(self, k) for k in ("run", "providers", "repeats", "limit", "state", "step", "done",
                                              "total", "errors", "message", "last_error", "started")}

    def progress(self, n, total, rec):
        self.done, self.total = n, total
        if rec and rec.get("error"):
            self.errors += 1
            self.last_error = rec["error"][:300]

    def run_pipeline(self):
        d = RUNS / self.run
        try:
            d.mkdir(parents=True, exist_ok=True)
            self.state, self.step = "running", "Asking the models"
            C.collect(self.providers, self.repeats, 0.7, d, limit=self.limit, progress=self.progress, spread=True)
            self.step = "Checking the answers"
            df = S.score_file(d / "responses.jsonl", d / "scored.csv")
            if (df["flags"] == "ERROR").all():
                raise RuntimeError("Every question failed. " + (self.last_error or "Check your keys and model names."))
            self.step = "Making the rating sheet"
            R.make_sheet(d / "scored.csv", d)
            self.step = "Writing the report"
            P.build(d / "scored.csv", d / "report", d / "agreement.csv")
            self.state, self.step = "finished", "Done"
            self.message = f"{len(df) - int((df['flags'] == 'ERROR').sum())} answers collected"
        except Exception as exc:  # shown to the user in the page
            self.state, self.step = "failed", "Stopped"
            self.message = str(exc)[:500]
            traceback.print_exc()


def _runs() -> list[dict]:
    out = []
    if not RUNS.exists():
        return out
    for d in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not (d / "report" / "report.html").exists():
            continue
        meta = {"name": d.name, "answers": 0, "models": [], "demo": False,
                "when": dt.datetime.fromtimestamp((d / "report" / "report.html").stat().st_mtime).strftime("%d %b %Y, %H:%M")}
        f = d / "responses.jsonl"
        if f.exists():
            provs = set()
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                provs.add(r.get("provider"))
                meta["answers"] += 0 if r.get("error") else 1
                meta["demo"] |= bool(r.get("synthetic"))
            meta["models"] = [L.provider(p) for p in sorted(provs)]
        out.append(meta)
    return out


def create_app() -> Flask:
    load_dotenv()
    if not os.environ.get("OLLAMA_HOST"):  # outside Docker, Ollama is on this machine
        os.environ["OLLAMA_HOST"] = ("http://host.docker.internal:11434" if Path("/.dockerenv").exists()
                                     else "http://localhost:11434")
    app = Flask(__name__)
    n_prompts = len(C.build_prompts(C.load_bank()))

    @app.get("/")
    def index():
        return PAGE.replace("__N_PROMPTS__", str(n_prompts))

    @app.get("/api/status")
    def status():
        prov = {}
        for name, v in REAL.items():
            prov[name] = {"label": L.provider(name), "needs_key": bool(v["key"]), "has_key": _has(v["key"]) if v["key"] else True,
                          "model": os.environ.get(v["model"], ""),
                          "host": os.environ.get(v.get("host") or "", "") if v.get("host") else None}
        return jsonify({"providers": prov, "n_prompts": n_prompts,
                        "job": Job.current.as_dict() if Job.current else None})

    @app.post("/api/run")
    def run():
        body = request.get_json(force=True) or {}
        with Job.lock:
            if Job.current and Job.current.state in ("starting", "running"):
                return jsonify({"error": "A run is already in progress. Wait for it to finish."}), 409
            mode = body.get("mode", "demo")
            repeats = max(1, min(3, int(body.get("repeats", 1))))
            limit = body.get("limit")
            limit = None if limit in (None, "", "all", 0) else max(1, min(n_prompts, int(limit)))
            if mode == "demo":
                providers = ["mock-a", "mock-b"]
            else:
                providers = []
                for name, cfg in (body.get("providers") or {}).items():
                    if name not in REAL or not cfg.get("use"):
                        continue
                    spec = REAL[name]
                    if cfg.get("key"):
                        os.environ[spec["key"]] = cfg["key"].strip()  # memory only, never saved
                    if cfg.get("model"):
                        os.environ[spec["model"]] = cfg["model"].strip()
                    if spec.get("host") and cfg.get("host"):
                        os.environ[spec["host"]] = cfg["host"].strip()
                    if spec["key"] and not _has(spec["key"]):
                        return jsonify({"error": f"{L.provider(name)} needs an API key."}), 400
                    if not _has(spec["model"]):
                        return jsonify({"error": f"{L.provider(name)} needs a model name."}), 400
                    providers.append(name)
                if not providers:
                    return jsonify({"error": "Tick at least one real model, or choose the demo."}), 400
            name = (body.get("run") or "").strip() or (("demo" if mode == "demo" else "real") + dt.datetime.now().strftime("_%Y%m%d_%H%M%S"))
            if not SAFE.match(name):
                return jsonify({"error": "Run names can use letters, numbers, - and _ only."}), 400
            if (RUNS / name / "responses.jsonl").exists() and not body.get("resume"):
                return jsonify({"error": f"A run called '{name}' already exists. Choose another name."}), 400
            job = Job(name, providers, repeats, limit)
            Job.current = job
            threading.Thread(target=job.run_pipeline, daemon=True).start()
            return jsonify(job.as_dict())

    @app.get("/api/job")
    def job():
        return jsonify(Job.current.as_dict() if Job.current else None)

    @app.get("/api/runs")
    def runs():
        return jsonify(_runs())

    @app.get("/runs/<name>/<fname>")
    def files(name, fname):
        if not SAFE.match(name) or fname not in DOWNLOADS:
            abort(404)
        path = (RUNS / name / DOWNLOADS[fname]).resolve()
        if not path.is_file() or RUNS.resolve() not in path.parents:
            abort(404)
        return send_file(path, as_attachment=request.args.get("download") == "1", download_name=fname)

    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    print(f"Open http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port} in your browser. Press Ctrl+C to stop.")
    create_app().run(host=host, port=port, debug=False, threaded=True)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Climate AI Answers Pilot</title>
<style>
:root{--ink:#1f2933;--muted:#5b6370;--line:#e3e5e8;--bg:#f6f7f9;--card:#fff;--blue:#2a78d6;--amber:#fef3c7;--red:#fde2e2;--green:#e3f6ee}
*{box-sizing:border-box} body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);background:var(--bg)}
header{background:#fff;border-bottom:1px solid var(--line);padding:16px 20px}
header h1{margin:0;font-size:20px} header p{margin:4px 0 0;color:var(--muted)}
main{max-width:1100px;margin:0 auto;padding:16px;display:grid;grid-template-columns:380px 1fr;gap:16px}
@media(max-width:860px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
h2{font-size:16px;margin:0 0 10px} h3{font-size:14px;margin:14px 0 6px}
.mode{display:grid;gap:8px} .opt{border:1.5px solid var(--line);border-radius:8px;padding:10px;cursor:pointer;display:flex;gap:10px}
.opt:has(input:checked){border-color:var(--blue);background:#eef4fc} .opt small{display:block;color:var(--muted)}
.prov{border:1px solid var(--line);border-radius:8px;padding:10px;margin:8px 0}
.prov label.top{display:flex;align-items:center;gap:8px;font-weight:600}
.pill{font-size:11px;padding:2px 8px;border-radius:99px;margin-left:auto;font-weight:500}
.ok{background:var(--green);color:#11603f} .need{background:var(--amber);color:#8a5a00}
input[type=text],input[type=password],select{width:100%;padding:7px 9px;border:1px solid #cdd1d6;border-radius:6px;font:inherit;margin-top:4px}
.field{margin-top:8px;font-size:13px;color:var(--muted)}
button{background:var(--blue);color:#fff;border:0;border-radius:8px;padding:11px 16px;font:600 15px system-ui;cursor:pointer;width:100%}
button:disabled{opacity:.5;cursor:not-allowed}
.est{background:#f2f4f7;border-radius:6px;padding:8px 10px;font-size:13px;margin:10px 0}
.msg{border-radius:6px;padding:8px 10px;font-size:13px;margin-top:10px} .msg.err{background:var(--red)} .msg.info{background:var(--amber)}
.bar{height:10px;background:#e8eaee;border-radius:99px;overflow:hidden;margin:8px 0} .bar div{height:100%;background:var(--blue);width:0;transition:width .4s}
.runs a{display:block;padding:8px 10px;border:1px solid var(--line);border-radius:8px;margin:6px 0;color:inherit;text-decoration:none}
.runs a:hover,.runs a.sel{border-color:var(--blue)} .runs small{color:var(--muted);display:block}
.tag{font-size:11px;padding:1px 7px;border-radius:99px;background:var(--amber);margin-left:6px} .tag.real{background:var(--green)}
iframe{width:100%;height:78vh;border:1px solid var(--line);border-radius:10px;background:#fff}
.dl a{font-size:13px;margin-right:12px;color:var(--blue)} .hidden{display:none} .note{font-size:12px;color:var(--muted)}
</style></head><body>
<header><h1>What do AI models say about climate change in Nigeria?</h1>
<p>Ask AI models the same climate questions about Lagos, Uyo and Kano in four different ways, then see how they describe uncertainty.</p></header>
<main>
<div>
 <div class="card"><h2>1. Choose what to run</h2>
  <div class="mode">
   <label class="opt"><input type="radio" name="mode" value="demo" checked><span><b>Demo</b><small>Free, no internet or keys. Uses two pretend models to show how the tool works. The results are made up.</small></span></label>
   <label class="opt"><input type="radio" name="mode" value="real"><span><b>Real AI models</b><small>Sends the questions to real models. Needs an API key (costs a little) or a free local model (Ollama).</small></span></label>
  </div>
  <div id="real" class="hidden"><h3>Models to ask</h3><div id="provs"></div>
   <p class="note">Keys typed here stay in this program's memory on your computer and are never saved. You can also put them in a <code>.env</code> file.</p></div>
 </div>
 <div class="card"><h2>2. How big a run?</h2>
  <div class="field">Questions to send
   <select id="limit"><option value="7">Quick test: 7 questions, spread across topics</option><option value="28">Medium: 28 questions, spread across topics</option><option value="all" selected>Full: all __N_PROMPTS__ questions</option></select></div>
  <div class="field">Times to ask each question
   <select id="repeats"><option value="1">Once</option><option value="2">Twice</option><option value="3" selected>3 times (shows whether answers change)</option></select></div>
  <div class="field">Name for this run (optional)<input type="text" id="runname" placeholder="e.g. first_real_test"></div>
  <div class="est" id="est"></div>
  <button id="go">Run and make the report</button>
  <div id="msg"></div>
 </div>
 <div class="card hidden" id="progress"><h2>Progress</h2><div id="step"></div><div class="bar"><div id="fill"></div></div><div id="count" class="note"></div><div id="perr"></div></div>
 <div class="card"><h2>Past runs</h2><div class="runs" id="runs"><p class="note">None yet.</p></div></div>
</div>
<div>
 <div class="card" id="viewer"><h2 id="vtitle">Report</h2><div class="dl" id="dl"></div>
  <p class="note" id="vnote">Run the demo or pick a past run to see its report here.</p>
  <iframe id="frame" class="hidden" title="Report"></iframe></div>
</div>
</main>
<script>
const $=s=>document.querySelector(s); let STATUS=null, polling=null;
const N=__N_PROMPTS__;
function mode(){return document.querySelector('input[name=mode]:checked').value}
function providersFromForm(){const o={};document.querySelectorAll('.prov').forEach(p=>{const n=p.dataset.name;o[n]={use:p.querySelector('.use').checked,
 key:(p.querySelector('.key')||{}).value||'',model:p.querySelector('.model').value,host:(p.querySelector('.host')||{}).value||''}});return o}
function estimate(){const lim=$('#limit').value==='all'?N:Math.min(N,+$('#limit').value), rep=+$('#repeats').value;
 const m=mode()==='demo'?2:Object.values(providersFromForm()).filter(p=>p.use).length;
 const calls=lim*rep*m; $('#est').innerHTML=mode()==='demo'
  ? `${lim} questions × ${rep} time(s) × 2 pretend models = <b>${calls} answers</b>. Takes a few seconds, costs nothing.`
  : `${lim} questions × ${rep} time(s) × ${m} model(s) = <b>${calls} requests</b>. ${calls>60?'Try the quick test first. ':''}Paid APIs charge per request; Ollama is free.`}
async function loadStatus(){STATUS=await (await fetch('/api/status')).json(); const h=[];
 for(const [n,p] of Object.entries(STATUS.providers)){
  h.push(`<div class="prov" data-name="${n}"><label class="top"><input type="checkbox" class="use"> ${p.label}
   <span class="pill ${p.has_key?'ok':'need'}">${p.needs_key?(p.has_key?'Key found':'Needs a key'):'No key needed'}</span></label>
   ${p.needs_key?`<div class="field">API key${p.has_key?' (already set; leave blank to use it)':''}<input type="password" class="key" autocomplete="off"></div>`:''}
   <div class="field">Model name<input type="text" class="model" value="${p.model||''}" placeholder="the exact model ID from ${n==='ollama'?'ollama list':'your provider account'}"></div>
   ${n==='ollama'?`<div class="field">Ollama address<input type="text" class="host" value="${p.host||''}" placeholder="http://localhost:11434"></div><p class="note">Runs on your computer: free, but slow without a graphics card. The first answer can take a few minutes while the model loads; the app waits up to 10 minutes per answer. Tip: run <code>ollama run MODEL "hello"</code> first to load it.</p>`:''}</div>`)}
 $('#provs').innerHTML=h.join(''); document.querySelectorAll('#provs input').forEach(i=>i.addEventListener('input',estimate)); estimate();
 if(STATUS.job&&['starting','running'].includes(STATUS.job.state)) watch()}
document.querySelectorAll('input[name=mode]').forEach(r=>r.addEventListener('change',()=>{msg('');$('#real').classList.toggle('hidden',mode()!=='real');
 if(mode()==='real'&&$('#limit').value==='all') $('#limit').value='7'; if(mode()==='real') $('#repeats').value='1'; estimate()}));
['#limit','#repeats'].forEach(s=>$(s).addEventListener('change',estimate));
function msg(t,kind){$('#msg').innerHTML=t?`<div class="msg ${kind}">${t}</div>`:''}
$('#go').addEventListener('click',async()=>{msg('');const body={mode:mode(),limit:$('#limit').value,repeats:$('#repeats').value,run:$('#runname').value,providers:providersFromForm()};
 const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const j=await r.json();
 if(!r.ok){msg(j.error,'err');return} document.querySelectorAll('.key').forEach(k=>k.value=''); watch()});
function watch(){$('#progress').classList.remove('hidden');$('#go').disabled=true;clearInterval(polling);polling=setInterval(async()=>{
 const j=await (await fetch('/api/job')).json(); if(!j)return; $('#step').innerHTML=`<b>${j.step}</b> · run “${j.run}”`;
 const pct=j.total?Math.round(100*j.done/j.total):0; $('#fill').style.width=(j.state==='finished'?100:pct)+'%';
 $('#count').textContent=j.total?`${j.done} of ${j.total} answers`:'';
 $('#perr').innerHTML=j.errors?`<div class="msg info">${j.errors} request(s) failed so far. Last problem: ${esc(j.last_error)}</div>`:'';
 if(j.state==='finished'||j.state==='failed'){clearInterval(polling);$('#go').disabled=false;
  if(j.state==='failed') msg('The run stopped: '+esc(j.message),'err'); else {msg(j.message+'. The report is open on the right.','info'); show(j.run)} loadRuns()}},1000)}
function esc(s){return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function loadRuns(){const rs=await (await fetch('/api/runs')).json(); if(!rs.length)return;
 $('#runs').innerHTML=rs.map(r=>`<a href="#" data-run="${r.name}">${r.name}<span class="tag ${r.demo?'':'real'}">${r.demo?'demo':'real'}</span>
  <small>${r.when} · ${r.answers} answers · ${r.models.join(', ')}</small></a>`).join('');
 document.querySelectorAll('.runs a').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();show(a.dataset.run)}))}
function show(run){document.querySelectorAll('.runs a').forEach(a=>a.classList.toggle('sel',a.dataset.run===run));
 $('#vtitle').textContent='Report: '+run; $('#vnote').classList.add('hidden'); const f=$('#frame'); f.classList.remove('hidden'); f.src=`/runs/${run}/report.html`;
 $('#dl').innerHTML=`<a href="/runs/${run}/report.html" target="_blank">Open in new tab</a><a href="/runs/${run}/report.html?download=1">Download report</a>
  <a href="/runs/${run}/answers.csv?download=1">All answers (spreadsheet)</a><a href="/runs/${run}/rating_sheet.csv?download=1">Rating sheet</a><a href="/runs/${run}/RUBRIC.txt?download=1">Rating guide</a>`}
loadStatus(); loadRuns();
</script></body></html>"""
