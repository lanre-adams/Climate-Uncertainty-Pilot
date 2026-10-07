"""LLM providers.

Each provider takes a prompt and returns text. Real providers read their API key and
model name from environment variables, so nothing secret lives in the code.

MockProvider produces SYNTHETIC answers so the whole pipeline can be tested offline
(e.g. inside Docker with no keys). Mock output is not data about any real model and
must never be reported as a result.
"""
from __future__ import annotations

import hashlib
import os
import random
import time
from dataclasses import dataclass

import requests

TIMEOUT = 120


class ProviderError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    model: str
    latency_s: float


class Provider:
    name = "base"

    def complete(self, prompt: str, *, temperature: float, repeat: int) -> Completion:
        raise NotImplementedError


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or value.lower().startswith("set-me"):
        raise ProviderError(f"Environment variable {name} is not set (see .env.example).")
    return value


def _post(url: str, headers: dict, payload: dict) -> dict:
    for attempt in range(4):
        r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(2 ** attempt * 3)
            continue
        if r.status_code >= 400:
            raise ProviderError(f"{url} -> HTTP {r.status_code}: {r.text[:300]}")
        return r.json()
    raise ProviderError(f"{url} kept failing after retries")


class OpenAIProvider(Provider):
    name = "openai"

    def complete(self, prompt, *, temperature, repeat):
        model = _env("OPENAI_MODEL")
        t0 = time.time()
        data = _post(
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions",
            {"Authorization": f"Bearer {_env('OPENAI_API_KEY')}"},
            {"model": model, "temperature": temperature,
             "messages": [{"role": "user", "content": prompt}]},
        )
        return Completion(data["choices"][0]["message"]["content"], model, time.time() - t0)


class AnthropicProvider(Provider):
    name = "anthropic"

    def complete(self, prompt, *, temperature, repeat):
        model = _env("ANTHROPIC_MODEL")
        t0 = time.time()
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": _env("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"},
            {"model": model, "max_tokens": 1500, "temperature": temperature,
             "messages": [{"role": "user", "content": prompt}]},
        )
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return Completion(text, model, time.time() - t0)


class GeminiProvider(Provider):
    name = "gemini"

    def complete(self, prompt, *, temperature, repeat):
        model = _env("GEMINI_MODEL")
        t0 = time.time()
        data = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {"x-goog-api-key": _env("GEMINI_API_KEY")},
            {"contents": [{"parts": [{"text": prompt}]}],
             "generationConfig": {"temperature": temperature}},
        )
        try:
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"Unexpected Gemini response: {str(data)[:300]}") from exc
        return Completion(text, model, time.time() - t0)


class OllamaProvider(Provider):
    """Local open-weight models via Ollama (no API key, no cost)."""
    name = "ollama"

    def complete(self, prompt, *, temperature, repeat):
        model = _env("OLLAMA_MODEL")
        host = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
        t0 = time.time()
        data = _post(f"{host}/api/chat", {}, {
            "model": model, "stream": False, "options": {"temperature": temperature},
            "messages": [{"role": "user", "content": prompt}],
        })
        return Completion(data["message"]["content"], model, time.time() - t0)


# ---------------------------------------------------------------- mock ----

_MOCK = {
    "good_heat": "Heatwaves in {p} are very likely to become more frequent and more intense. "
                 "The IPCC AR6 assesses with high confidence that hot extremes across Africa will keep "
                 "increasing with further warming, relative to 1850-1900. How much depends on the "
                 "emissions scenario (for example SSP1-2.6 versus SSP5-8.5).",
    "bad_heat": "It is hard to say whether {p} will see more heatwaves; it may or may not happen.",
    "good_rain": "Projections for annual rainfall in {p} are uncertain. Models disagree on the "
                 "direction of change for this part of West Africa, and AR6 does not give a robust "
                 "projection of mean rainfall here, although heavy rainfall events are projected to "
                 "increase (high confidence).",
    "bad_rain": "Annual rainfall in {p} will decrease by 20% by 2050, leading to severe drought.",
    "good_heavy": "Yes. Heavy precipitation and pluvial flooding are projected to increase in West "
                  "Africa with additional warming (IPCC AR6, at least medium confidence).",
    "good_sea": "Relative sea level rise along the coast near {p} is likely to virtually certain to "
                "continue this century, increasing coastal flooding and erosion of sandy coasts "
                "(IPCC AR6 WGI). The amount depends on the emissions scenario; ranges of roughly "
                "0.3-1.0 m by 2100 are often cited globally.",
    "bad_sea": "Sea level near {p} will rise exactly 2 metres by 2050 according to the Nigerian "
               "Coastal Climate Report 2021 (Adebayo et al.).",
    "good_unc": "Confidence varies by variable. Temperature projections for {p} are high "
                "confidence; rainfall projections are low confidence because models disagree about "
                "the West African monsoon. The main sources of uncertainty are the emissions "
                "scenario, model differences and natural variability.",
    "bad_unc": "Scientists are fully certain about the climate future of {p}.",
    "good_dec": "Both. AR6 projects more heavy rainfall and pluvial flooding in West Africa, and "
                "also drying and drought at seasonal timescales, with more confidence at higher "
                "warming levels. Planning for both, with flexible options, is safer than betting on one.",
    "bad_dec": "Planners in {p} should prepare for drought only; flooding will not be an issue.",
    "general": "By the 2050s {p} is expected to be warmer, with more frequent hot extremes "
               "(high confidence). Changes in total rainfall are uncertain.",
}
_TOPIC_KEYS = {
    "hot_extremes": ("good_heat", "bad_heat"), "mean_precipitation": ("good_rain", "bad_rain"),
    "heavy_precipitation": ("good_heavy", "bad_rain"), "sea_level": ("good_sea", "bad_sea"),
    "uncertainty": ("good_unc", "bad_unc"), "decision": ("good_dec", "bad_dec"),
    "general": ("general", "bad_rain"),
}


class MockProvider(Provider):
    """Deterministic synthetic answers. About one in four is deliberately flawed so the
    scorer has something to catch. SYNTHETIC - never report as results."""
    name = "mock"

    def __init__(self, variant: str = "a"):
        self.variant = variant
        self.name = f"mock-{variant}"

    def complete(self, prompt, *, temperature, repeat, topic: str = "general", place: str = "the city"):
        seed = int(hashlib.sha256(f"{self.variant}|{prompt}|{repeat}".encode()).hexdigest(), 16)
        rng = random.Random(seed)
        bad_rate = 0.15 if self.variant == "a" else 0.35
        good, bad = _TOPIC_KEYS.get(topic, ("general", "bad_rain"))
        key = bad if rng.random() < bad_rate else good
        text = _MOCK[key].format(p=place)
        if "sources" in prompt.lower() and rng.random() < 0.7:
            text += " Sources: IPCC AR6 WGI Regional Fact Sheet - Africa; https://interactive-atlas.ipcc.ch/"
        return Completion(text, f"synthetic-{self.variant}", 0.0)


def get_provider(name: str) -> Provider:
    name = name.lower()
    if name.startswith("mock"):
        return MockProvider(name.split("-", 1)[1] if "-" in name else "a")
    table = {"openai": OpenAIProvider, "anthropic": AnthropicProvider,
             "gemini": GeminiProvider, "ollama": OllamaProvider}
    if name not in table:
        raise ProviderError(f"Unknown provider '{name}'. Choose from mock, mock-b, {', '.join(table)}.")
    return table[name]()
