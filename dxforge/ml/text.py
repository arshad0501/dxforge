"""Text normalisation and noise injection for diagnostic intents.

``normalize`` is the single cleaning function used at train AND inference time,
so there is no train/serve skew. ``add_noise`` is only used by the synthetic
data generator to emulate how engineers actually type (typos, abbreviations,
inconsistent casing, filler words, different hex notations).
"""
from __future__ import annotations

import random
import re

# Domain abbreviation lexicon (expanded during cleaning).
ABBREVIATIONS: dict[str, str] = {
    "diag": "diagnostic", "sess": "session", "ext": "extended",
    "prog": "programming", "pls": "please", "plz": "please",
    "info": "information", "sw": "software", "hw": "hardware",
    "rtn": "routine", "sec": "security", "comm": "communication",
    "comms": "communication", "temp": "temperature", "dtcs": "dtc",
    "num": "number", "ver": "version", "cal": "calibration",
    "msg": "message", "msgs": "messages", "u": "you", "ur": "your",
}
_REVERSE_ABBREV = {
    "diagnostic": ["diag"], "session": ["sess"], "extended": ["ext"],
    "programming": ["prog"], "please": ["pls", "plz"], "information": ["info"],
    "software": ["sw"], "routine": ["rtn"], "security": ["sec"],
    "communication": ["comm", "comms"], "temperature": ["temp"], "number": ["num"],
    "version": ["ver"], "calibration": ["cal"], "message": ["msg"], "you": ["u"],
}

# Hex identifier notations engineers use: 0xF190, $F190, F190h, bare F190.
_HEX_PATTERNS = [
    re.compile(r"\b0x([0-9a-f]{1,4})\b"),
    re.compile(r"\$([0-9a-f]{2,4})\b"),
    re.compile(r"\b([0-9a-f]{4})h\b"),
    re.compile(r"\b(?=[0-9a-f]*\d)([0-9a-f]{4})\b"),
]
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9$\s\-]")

HEX_TOKEN = "hexid"


def extract_hex_ids(text: str) -> list[int]:
    """Return every hex identifier in ``text`` (any supported notation), in order."""
    low = text.lower()
    found: list[tuple[int, int]] = []
    seen_spans: list[tuple[int, int]] = []
    for pat in _HEX_PATTERNS:
        for m in pat.finditer(low):
            span = m.span()
            if any(s <= span[0] < e for s, e in seen_spans):
                continue
            seen_spans.append(span)
            found.append((span[0], int(m.group(1), 16)))
    return [v for _, v in sorted(found)]


def normalize(text: object) -> str:
    """Canonical cleaning used for both training and inference.

    lower-case -> hex ids replaced by a single ``hexid`` token -> punctuation
    stripped -> abbreviation expansion -> whitespace collapsed.
    Non-string input (NaN / None) maps to the empty string.
    """
    if not isinstance(text, str):
        return ""
    low = text.lower().replace("-", " ").replace("_", " ")
    for pat in _HEX_PATTERNS:
        low = pat.sub(f" {HEX_TOKEN} ", low)
    low = _PUNCT.sub(" ", low).replace("$", " ")
    words = [ABBREVIATIONS.get(w, w) for w in low.split()]
    return _WS.sub(" ", " ".join(words)).strip()


# --------------------------------------------------------------------------- #
# Noise injection (data generation only)
# --------------------------------------------------------------------------- #
_PREFIXES = ["", "", "", "", "please ", "pls ", "can you ", "could you ", "hey, ",
             "i need to ", "quickly ", "now ", "ok "]
_SUFFIXES = ["", "", "", "", " please", " on the bcm", " for this ecu", " asap",
             " now", " on the bench", "?", " thanks", " via uds"]


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4 or not word.isalpha():
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.choice(["swap", "drop", "dup", "sub"])
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "dup":
        return word[:i] + word[i] + word[i:]
    near = "qwertyuiopasdfghjklzxcvbnm"
    return word[:i] + rng.choice(near) + word[i + 1:]


def add_noise(text: str, rng: random.Random, level: float = 1.0) -> tuple[str, dict]:
    """Apply realistic noise. Returns (noisy_text, flags) so EDA can measure it."""
    flags = {"typo": 0, "abbrev": 0, "filler": 0, "case": 0}
    words = text.split()
    out: list[str] = []
    for w in words:
        lw = w.lower()
        if lw in _REVERSE_ABBREV and rng.random() < 0.35 * level:
            w = rng.choice(_REVERSE_ABBREV[lw])
            flags["abbrev"] = 1
        elif rng.random() < 0.06 * level:
            w = _typo(w, rng)
            flags["typo"] = 1
        out.append(w)
    s = " ".join(out)
    if rng.random() < 0.45 * level:
        p, q = rng.choice(_PREFIXES), rng.choice(_SUFFIXES)
        if p or q:
            flags["filler"] = 1
        s = p + s + q
    r = rng.random()
    if r < 0.12 * level:
        s, flags["case"] = s.upper(), 1
    elif r < 0.30 * level:
        s, flags["case"] = s.capitalize(), 1
    return s, flags


def hex_surface(value: int, rng: random.Random) -> str:
    """Render an identifier the way different engineers would write it."""
    h = f"{value:04X}"
    return rng.choice([f"0x{h}", f"0x{h}", f"0x{h.lower()}", f"${h}", h, f"{h}h"])
