"""Hybrid resolution: Fate verdict + multi-track evidence + 24B decode.

Policy (valhalla-transformer hybrid, completed):
  - Fate emit            -> Mistral decodes from multi-track evidence.
  - Fate abstain / defer -> *weak emit*: if any track found candidate evidence,
                            Mistral must verify it under the evidence-only
                            constraint (it abstains itself when the evidence
                            does not answer). Only when no track finds
                            anything does the system stay silent.
  - Tracks: (1) Valhalla structure retrieval line, (2) Tile lexical track
    (keyword/block overlap over the session corpus, IDF-weighted), plus the
    native answer text. Fate remains the only gate that can force silence
    when there is no evidence at all.
"""

from __future__ import annotations

import math
import re

from valhalla_server import mistral_head

ABSTAIN_TEXT = "根据已摄入的资料无法确定。"
_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]*")
_STOP = {
    "what", "is", "the", "a", "an", "of", "in", "on", "for", "to", "and",
    "who", "when", "where", "which", "how", "many", "does", "did", "was",
    "were", "are", "it", "its", "project", "first",
}


def _tokens(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall((text or "").lower()):
        if tok in ("led", "leads", "leading"):
            tok = "lead"
        elif len(tok) > 4 and tok.endswith("s"):
            tok = tok[:-1]
        out.append(tok)
    return out


def tile_track(question: str, corpus: list[str], k: int = 3,
               exclude: set[int] | None = None) -> list[tuple[int, float]]:
    """Keyword/block track: IDF-weighted token overlap, best lines first."""
    if not corpus:
        return []
    q = [t for t in _tokens(question) if t not in _STOP]
    if not q:
        return []
    n = len(corpus)
    df: dict[str, int] = {}
    line_tokens: list[set[str]] = []
    for line in corpus:
        toks = set(_tokens(line))
        line_tokens.append(toks)
        for t in toks:
            df[t] = df.get(t, 0) + 1
    scored = []
    for i, toks in enumerate(line_tokens):
        if exclude and i in exclude:
            continue
        score = sum(math.log((n + 1) / (df.get(t, n) + 1)) + 0.2
                    for t in q if t in toks)
        if score > 0:
            scored.append((i, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def build_evidence(question: str, corpus: list[str], retrieval_idx: object,
                   native_answer: str, k: int = 3) -> list[str]:
    evidence: list[str] = []
    seen: set[str] = set()

    def add(line: str | None) -> None:
        if line and line not in seen:
            seen.add(line)
            evidence.append(line)

    idx = retrieval_idx if isinstance(retrieval_idx, int) else -1
    if 0 <= idx < len(corpus):
        add(corpus[idx])
    for i, _score in tile_track(question, corpus, k=k, exclude={idx}):
        add(corpus[i])
    add(native_answer)
    return evidence


def resolve(fog_kind: str, question: str, corpus: list[str],
            retrieval_idx: object, native_answer: str) -> tuple[str, str]:
    """Return (answer_text, mode). Modes: fog_abstain / mistral / mistral_weak /
    native_fallback."""
    evidence = build_evidence(question, corpus, retrieval_idx, native_answer)
    if fog_kind in ("abstain", "defer"):
        if not evidence:
            return ABSTAIN_TEXT, "fog_abstain"
        decoded = mistral_head.decode(question, evidence)
        if decoded is None:
            return ABSTAIN_TEXT, "fog_abstain"
        return decoded, "mistral_weak_emit"
    decoded = mistral_head.decode(question, evidence) if evidence else None
    if decoded:
        return decoded, "mistral"
    return native_answer, "native_fallback"
