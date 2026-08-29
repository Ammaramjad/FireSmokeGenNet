"""Prompt bank used as compact vision-language conditioning."""

PROMPT_TEMPLATES = [
    "gray wildfire smoke plume over a forested hillside, daytime, semi-transparent",
    "brownish smoke rising above a mountain valley, hazy atmosphere, distant plume",
    "thin white-gray smoke against a bright sky and forest canopy",
    "dense gray-blue wildfire smoke drifting across hilly terrain at dusk",
    "faint early-stage wildfire smoke, low contrast, translucent against vegetation",
    "column of gray smoke over a clear-weather mountain landscape",
    "layered wildfire smoke with soft boundaries above a forest ridge",
    "winter forest with a distant gray smoke plume, cool illumination",
]


def tokenize_prompt(text: str, dim: int = 64) -> list[float]:
    """Deterministic bag-of-characters embedding; no external tokenizer required."""
    vec = [0.0] * dim
    for i, ch in enumerate(text.lower()):
        vec[(ord(ch) + i) % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]
