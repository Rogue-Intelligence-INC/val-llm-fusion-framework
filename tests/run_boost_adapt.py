"""Simplified BOOST adaptation test (LLM-enhancement form).

Simplifies the incubation weight mechanisms into three operators on
top of the existing enhancement stack, same corpus/questions/head as
run_enhancement_ceiling (100,006 lines, ~1.5M tokens):

  spread    - query-time activation spreading (the Hebbian idea,
              simplified): round-1 retrieval lines yield bridge
              entities (capitalized spans); round-2 retrieval is
              driven by those entities + the question. This turns
              multihop from lexical-hit into graph-reachable.
  aggregate - deterministic count/list operator: rare-token predicate
              scan over the corpus, distinct Project names counted by
              code, never by LLM arithmetic; bundling is bypassed.
  (rest)    - unchanged: same Mistral decode head; boost_hybrid arm
              additionally feeds spread-gathered lines through the
              production engine session + hybrid.resolve, to test
              whether the Fate/evidence layer preserves the gain.

Baselines (hybrid 17/21, rag 14/21) are loaded from the stored
enhancement_ceiling_lines100006.json for side-by-side comparison.
"""
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_enhancement_ceiling import build_corpus, items, judge  # noqa: E402
from run_extreme_stress import CONFIG, fabrication, post  # noqa: E402
from run_extreme_targeted import ENGINE, slim  # noqa: E402
from valhalla_server import hybrid, mistral_head  # noqa: E402
from valhalla_server.hybrid import _STOP, _tokens  # noqa: E402

OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme")
N = 100006

corpus = build_corpus(N)
corpus_text = "\n".join(corpus)
print(f"corpus {len(corpus)} lines, ~{len(corpus_text)//4} est tokens", flush=True)

# ---- one-pass token index (same IDF math as hybrid.tile_track) ----
LINE_TOKENS = [set(_tokens(line)) for line in corpus]
DF: dict[str, int] = {}
for toks in LINE_TOKENS:
    for t in toks:
        DF[t] = DF.get(t, 0) + 1
NC = len(corpus)


def track(q_tokens, k):
    q = [t for t in dict.fromkeys(q_tokens) if t not in _STOP]
    scored = []
    for i, toks in enumerate(LINE_TOKENS):
        s = sum(math.log((NC + 1) / (DF.get(t, NC) + 1)) + 0.2
                for t in q if t in toks)
        if s > 0:
            scored.append((i, s))
    scored.sort(key=lambda x: -x[1])
    return [i for i, _s in scored[:k]]


_ENT = re.compile(r"(?:Dr\.\s*)?[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3}")


def entities(lines):
    out = set()
    for line in lines:
        for m in _ENT.findall(line):
            out.add(m.replace("Dr. ", "").strip())
    return out


def gather(question, rounds=2, k=8):
    """Activation spreading, simplified: retrieve -> bridge entities
    -> retrieve again with entities + question. Returns line texts."""
    q_tokens = _tokens(question)
    seen: list[int] = []
    seen_set: set[int] = set()
    frontier = q_tokens
    for _ in range(rounds):
        new = [i for i in track(frontier, k * 2) if i not in seen_set][:k]
        if not new:
            break
        for i in new:
            seen_set.add(i)
            seen.append(i)
        ents = entities([corpus[i] for i in new])
        ent_tokens = [t for e in ents for t in _tokens(e)
                      if t not in [x.lower() for x in _tokens(question)]]
        if not ent_tokens:
            break
        frontier = q_tokens + ent_tokens
    return [corpus[i] for i in seen]


_PROJ = re.compile(r"Project\s+([A-Z][a-z]+)")


def aggregate(question):
    """Deterministic count/list: rare-token predicate scan, distinct
    project names counted by code. Returns answer text or None."""
    q = question.lower()
    if not (q.startswith("how many") or q.startswith("list all")):
        return None
    rare = [t for t in dict.fromkeys(_tokens(question))
            if t not in _STOP and DF.get(t, NC) <= max(50, NC // 2000)]
    if len(rare) < 2:
        return None
    names = []
    for i, toks in enumerate(LINE_TOKENS):
        if all(t in toks for t in rare):
            m = _PROJ.search(corpus[i])
            if m and m.group(1) not in names:
                names.append(m.group(1))
    if not names:
        return None
    if q.startswith("how many"):
        return f"{len(names)} projects ({', '.join(names)}) match: {question}"
    return ", ".join(f"Project {n}" for n in names)


def run_boost(its):
    rows = []
    for it in its:
        t1 = time.time()
        op = aggregate(it["text"])
        if op is not None:
            answer, mode = op, "boost_operator"
        else:
            ev = gather(it["text"])
            answer = mistral_head.decode(it["text"], ev[:10]) or ""
            mode = "boost_spread"
        sc = judge(it, answer)
        rows.append({"id": it["id"], "kind": it["kind"],
                     "answer": f"[{mode}] {answer}"[:300],
                     "sec": round(time.time() - t1, 1), **sc})
        print(f"  [boost] {it['id']:16s} pass={sc['pass']} ({mode}) {rows[-1]['sec']}s", flush=True)
    return rows


def run_boost_hybrid(its):
    sess = None
    rows = []
    for it in its:
        t1 = time.time()
        lines = gather(it["text"])[:12]
        if sess is None:
            payload = {"questions": [{"id": it["id"], "text": it["text"], "score_type": "open"}],
                       "config": CONFIG, "context": lines}
        else:
            payload = {"questions": [{"id": it["id"], "text": it["text"], "score_type": "open"}],
                       "config": CONFIG, "context": [], "session": slim(sess),
                       "append_lines": lines}
        body = post(ENGINE, payload)
        sess = body.get("session")
        row = (body.get("rows") or [{}])[0]
        fog = (row.get("fog_decision") or {}).get("decision", "")
        sess_corpus = list((sess or {}).get("corpus") or [])
        def _regrasp(variant, _c=list(sess_corpus)):
            b = post(ENGINE, {"questions": [{"id": "rg", "text": variant, "score_type": "open"}],
                              "config": CONFIG, "context": _c})
            return (b.get("rows") or [{}])[0]

        answer, mode = hybrid.resolve(fog, it["text"], sess_corpus,
                                      row.get("retrieval_corpus_idx"),
                                      str(row.get("answer", "")),
                                      fog_decision=row.get("fog_decision"),
                                      fog_evidence=row.get("fog_evidence"),
                                      regrasp=_regrasp)
        sc = judge(it, answer)
        rows.append({"id": it["id"], "kind": it["kind"],
                     "answer": f"[{mode}] {answer}"[:300], "fog": fog,
                     "sec": round(time.time() - t1, 1), **sc})
        print(f"  [boost_hybrid] {it['id']:16s} pass={sc['pass']} fog={str(fog):8s} {rows[-1]['sec']}s", flush=True)
    return rows


_INTENT = re.compile(r"how many projects|list all|which projects", re.I)


def run_boost_v2(its):
    """v2: same operators, but inside the discipline that made hybrid
    safe -- injection quarantine on spread evidence, intent-gated
    operator (enumeration over a class only, never attribute
    questions), empty decode falls back to an explicit abstention."""
    rows = []
    for it in its:
        t1 = time.time()
        op = aggregate(it["text"]) if _INTENT.search(it["text"]) else None
        if op is not None:
            answer, mode = op, "boost2_operator"
        else:
            ev = [l for l in gather(it["text"])
                  if not hybrid._INJECTION.search(l)]
            answer = mistral_head.decode(it["text"], ev[:10]) or "I don't know."
            mode = "boost2_spread"
        sc = judge(it, answer)
        rows.append({"id": it["id"], "kind": it["kind"],
                     "answer": f"[{mode}] {answer}"[:300],
                     "sec": round(time.time() - t1, 1), **sc})
        print(f"  [boost2] {it['id']:16s} pass={sc['pass']} ({mode}) {rows[-1]['sec']}s", flush=True)
    return rows


def gather2(question, rounds=2, k=8):
    """v3 spreading: only STRONG edges carry activation -- bridge
    entities are kept only if they contain a rare token (high IDF),
    top-3 by rarity; identical evidence lines are deduped. Mirrors
    the incubation rule that grasp follows field strength, not mere
    co-occurrence (a shared surname in filler must not flood hop 2)."""
    q_tokens = _tokens(question)
    seen: list[int] = []
    seen_set: set[int] = set()
    frontier = q_tokens
    for _ in range(rounds):
        new = [i for i in track(frontier, k * 2) if i not in seen_set][:k]
        if not new:
            break
        for i in new:
            seen_set.add(i)
            seen.append(i)
        ents = entities([corpus[i] for i in new])
        q_low = {t.lower() for t in q_tokens}
        strong = []
        for e in ents:
            toks = [t for t in _tokens(e) if t.lower() not in q_low]
            rare = [t for t in toks if DF.get(t, NC) <= NC // 100]
            if rare:
                strong.append((sum(math.log((NC + 1) / (DF[t] + 1)) for t in rare), e, toks))
        if not strong:
            break
        strong.sort(reverse=True)
        frontier = q_tokens + [t for _s, _e, toks in strong[:3] for t in toks]
    out, done = [], set()
    for i in seen:
        if corpus[i] not in done:
            done.add(corpus[i])
            out.append(corpus[i])
    return out


def run_boost_v3(its):
    rows = []
    for it in its:
        t1 = time.time()
        op = aggregate(it["text"]) if _INTENT.search(it["text"]) else None
        if op is not None:
            answer, mode = op, "boost3_operator"
        else:
            ev = [l for l in gather2(it["text"])
                  if not hybrid._INJECTION.search(l)]
            answer = mistral_head.decode(it["text"], ev[:10]) or "I don't know."
            mode = "boost3_spread"
        sc = judge(it, answer)
        rows.append({"id": it["id"], "kind": it["kind"],
                     "answer": f"[{mode}] {answer}"[:300],
                     "sec": round(time.time() - t1, 1), **sc})
        print(f"  [boost3] {it['id']:16s} pass={sc['pass']} ({mode}) {rows[-1]['sec']}s", flush=True)
    return rows


def _stem(t):
    t = t.lower()
    for suf in ("'s", "ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) > len(suf) + 3:
            return t[: -len(suf)]
    return t


def gather3(question, k=8):
    """v4: entity-anchored spreading. Hop 2 candidates are the lines
    containing the exact bridge-entity phrase (entity linking, as in
    HippoRAG), ranked by stemmed question overlap plus a bonus for
    introducing new entities. Needed because at 100k scale a bridge
    name's single tokens are common (name collisions in filler)."""
    q_tokens = _tokens(question)
    first = track(q_tokens, k * 2)[:k]
    seen, seen_set = list(first), set(first)
    ents = entities([corpus[i] for i in first])
    anchored = []
    for e in ents:
        hits = [i for i, line in enumerate(corpus) if e in line]
        if 1 < len(hits) <= 300:
            anchored.append((len(hits), e))
    anchored.sort()
    q_stems = {_stem(t) for t in q_tokens}
    round1_text = " ".join(corpus[i] for i in first)
    for _cnt, e in anchored[:3]:
        cand = []
        for i, line in enumerate(corpus):
            if e not in line or i in seen_set:
                continue
            overlap = len({_stem(t) for t in _tokens(line)} & q_stems)
            new_ent = sum(1 for m in _ENT.findall(line) if m not in round1_text)
            cand.append((overlap + 0.5 * new_ent, i))
        cand.sort(reverse=True)
        for _s, i in cand[:3]:
            if i not in seen_set:
                seen_set.add(i)
                seen.append(i)
    out, done = [], set()
    for i in seen:
        if corpus[i] not in done:
            done.add(corpus[i])
            out.append(corpus[i])
    return out


def run_boost_v4(its):
    rows = []
    for it in its:
        t1 = time.time()
        op = aggregate(it["text"]) if _INTENT.search(it["text"]) else None
        if op is not None:
            answer, mode = op, "boost4_operator"
        else:
            ev = [l for l in gather3(it["text"])
                  if not hybrid._INJECTION.search(l)]
            answer = mistral_head.decode(it["text"], ev[:10]) or "I don't know."
            mode = "boost4_spread"
        sc = judge(it, answer)
        rows.append({"id": it["id"], "kind": it["kind"],
                     "answer": f"[{mode}] {answer}"[:300],
                     "sec": round(time.time() - t1, 1), **sc})
        print(f"  [boost4] {it['id']:16s} pass={sc['pass']} ({mode}) {rows[-1]['sec']}s", flush=True)
    return rows


def main():
    its = items()
    res = {"size": N, "form": "boost_adapt (spread+aggregate operators)", "arms": {}}
    res["arms"]["boost"] = run_boost(its)
    res["arms"]["boost_hybrid"] = run_boost_hybrid(its)
    res["arms"]["boost_v2"] = run_boost_v2(its)
    res["arms"]["boost_v3"] = run_boost_v3(its)
    res["arms"]["boost_v4"] = run_boost_v4(its)
    for rows in res["arms"].values():
        for r in rows:
            r["fabricated_tokens"] = fabrication(r["answer"], corpus_text)
    dest = OUT / "boost_adapt_lines100006.json"
    dest.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    base = json.loads((OUT / "enhancement_ceiling_lines100006.json").read_text(encoding="utf-8"))
    allarms = dict(base["arms"])
    allarms.update(res["arms"])
    kinds = sorted({r["kind"] for rows in allarms.values() for r in rows})
    print(f"\n{'kind':16s}" + "".join(f"{a:>14s}" for a in allarms))
    for kind in kinds:
        cells = ""
        for arm, rows in allarms.items():
            sc = [r for r in rows if r["kind"] == kind and r["pass"] is not None]
            cells += f"{sum(1 for r in sc if r['pass'])}/{len(sc):<12d}" if sc else f"{'-':>14s}"
        print(f"{kind:16s}{cells}")
    for arm, rows in allarms.items():
        sc = [r for r in rows if r["pass"] is not None]
        print(f"  == {arm}: {sum(1 for r in sc if r['pass'])}/{len(sc)} pass, "
              f"fabricated={sum(len(r.get('fabricated_tokens', [])) for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
