"""Plain-English names and explanations used by the report and the web interface.

The analysis code keeps short technical IDs (e.g. NO_SCENARIO_DEPENDENCE); everything a
reader sees is translated here, so the wording can be changed in one place.
"""
from __future__ import annotations

PROVIDERS = {
    "mock-a": "Demo model A (pretend)",
    "mock-b": "Demo model B (pretend)",
    "mock": "Demo model A (pretend)",
    "anthropic": "Claude (Anthropic)",
    "openai": "ChatGPT (OpenAI)",
    "gemini": "Gemini (Google)",
    "ollama": "Local open model (Ollama)",
}

FRAMINGS = {
    "plain": "Plain question",
    "planner": "Asked as a local planner",
    "lay": "Asked to explain simply",
    "cite": "Asked to name sources",
}

FRAMING_PHRASE = {
    "plain": "the question was asked plainly",
    "planner": "it was asked as a local planner",
    "lay": "it was asked to explain simply",
    "cite": "it was asked to name sources",
}

TOPICS = {
    "general": "Overall change by the 2050s",
    "mean_precipitation": "Yearly rainfall",
    "heavy_precipitation": "Heavy rain and flash floods",
    "hot_extremes": "Heatwaves",
    "sea_level": "Sea-level rise",
    "uncertainty": "How sure scientists are",
    "decision": "Plan for flood or drought?",
}

# flag id -> (short plain name, one-line explanation)
FLAGS = {
    "NO_SOURCE": ("Gives no source",
                  "The answer does not say where its information comes from."),
    "NO_SCENARIO_DEPENDENCE": ("Ignores that it depends on future emissions",
                               "The answer does not mention that the outcome depends on how much the world emits."),
    "POINT_NUMBER_NO_RANGE": ("Gives an exact number with no range",
                              "A single figure is stated as if certain, with no 'between X and Y'."),
    "MISSES_ROBUST_INCREASE": ("Misses a well-established increase",
                               "Scientists are confident this will increase (heat, heavy rain or sea level), but the answer does not say so."),
    "CERTAINTY_LANGUAGE": ("Sounds too certain",
                           "Uses words like 'definitely' or 'exactly' about the future."),
    "OVERCONFIDENT_RAIN_DIRECTION": ("Too sure about rainfall",
                                     "Says yearly rainfall will go up or down with no caveat, although scientists do not agree for this region."),
    "NO_UNCERTAINTY_SOURCES": ("Doesn't explain why projections are uncertain",
                               "Asked about uncertainty, but does not name its causes (emissions, model differences, natural variation)."),
    "POSSIBLE_UNDERCONFIDENCE": ("May be too unsure",
                                 "Hedges about something scientists are confident of, without using their confidence terms."),
    "DISMISSES_A_HAZARD": ("Dismisses a real hazard",
                           "Tells planners that flooding or drought will not be a problem."),
    "ONE_SIDED_DECISION": ("One-sided advice",
                           "Mentions only flood or only drought when both are plausible."),
    "CITATION_TO_VERIFY": ("Cites a paper that needs checking",
                           "Names an author and year; this could be real or invented."),
}

SOURCES = {
    "IPCC": "IPCC (UN climate science panel)",
    "AR6": "IPCC Sixth Assessment Report",
    "CMIP6": "CMIP6 climate models",
    "CORDEX": "CORDEX regional models",
    "NiMet": "NiMet (Nigerian Meteorological Agency)",
    "Met Office": "UK Met Office",
    "WMO": "World Meteorological Organization",
    "Interactive Atlas": "IPCC Interactive Atlas",
    "World Bank": "World Bank",
    "Copernicus": "Copernicus (EU climate service)",
    "NASA": "NASA",
    "NOAA": "NOAA (US weather agency)",
}


TOPIC_PHRASE = {
    "general": "overall change by the 2050s",
    "mean_precipitation": "yearly rainfall",
    "heavy_precipitation": "heavy rain and flash floods",
    "hot_extremes": "heatwaves",
    "sea_level": "sea-level rise",
    "uncertainty": "how sure scientists are",
    "decision": "whether to plan for flood or drought",
}


def provider(name: str) -> str:
    return PROVIDERS.get(str(name), str(name))


def framing(name: str) -> str:
    return FRAMINGS.get(str(name), str(name))


def topic(name: str) -> str:
    return TOPICS.get(str(name), str(name))


def flag(name: str) -> str:
    return FLAGS.get(str(name), (str(name), ""))[0]


def flag_help(name: str) -> str:
    return FLAGS.get(str(name), ("", ""))[1]


def source(name: str) -> str:
    return SOURCES.get(str(name), str(name))
