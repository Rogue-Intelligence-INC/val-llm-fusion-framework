"""Mistral-24B decode head (valhalla-transformer hybrid adaptation).

Following the hybrid unified product policy (open -> LM decode, structure
guides): ValhallaBase owns the session, Fate routing and the FOG verdict;
the 24B LM only turns the retrieved evidence into a clean answer. It never
sees the whole corpus and it never overrides a Fate abstain.

Talks to a llama.cpp server (llama-server, OpenAI-compatible). If the head
is unreachable, callers fall back to the native Valhalla answer text.
"""

from __future__ import annotations

import json
import os
import urllib.request

URL = os.environ.get("VALHALLA_MISTRAL_URL", "http://127.0.0.1:18080/v1/chat/completions")
MODEL = os.environ.get("VALHALLA_MISTRAL_MODEL", "mistral-24b")
ENABLED = os.environ.get("VALHALLA_DECODE_HEAD", "mistral").lower() == "mistral"

SYSTEM = (
    "You are the decode head of a structure-memory system. Answer the question "
    "using ONLY the evidence lines provided. If the evidence does not contain "
    "the answer, say you don't know. One or two short sentences, no extra facts."
)


def available(timeout: float = 3.0) -> bool:
    if not ENABLED:
        return False
    try:
        health = URL.rsplit("/v1/", 1)[0] + "/health"
        with urllib.request.urlopen(health, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def variants(question: str, timeout: float = 120.0) -> list[str]:
    """Transformer-side question rewrites for Fate re-grasping: up to
    two simpler direct factual rewrites / sub-questions. The rewrites
    never answer anything; they only re-ask so Fate can grasp again
    in its own terms (paraphrase brittleness lives in phrasing)."""
    if not ENABLED:
        return []
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content":
             "You rewrite questions for a retrieval system. Output up to 2 "
             "simpler, direct factual rewrites or sub-questions of the "
             "user question, one per line, no numbering, no answers, "
             "keep all names and codenames exactly as written."},
            {"role": "user", "content": question},
        ],
        "temperature": 0.2,
        "max_tokens": 120,
    }
    try:
        req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        text = body["choices"][0]["message"]["content"].strip()
    except Exception:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip(" -0123456789.、")
        if line and line.lower() != question.strip().lower() and line not in out:
            out.append(line)
    return out[:2]


def decode(question: str, evidence: list[str], timeout: float = 300.0) -> str | None:
    """Structure-guided decode: question + Valhalla-retrieved evidence only."""
    if not ENABLED or not evidence:
        return None
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Evidence:\n" + "\n".join(evidence) +
                                        f"\n\nQuestion: {question}"},
        ],
        "temperature": 0.2,
        "max_tokens": 220,
    }
    try:
        req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        text = body["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception:
        return None
