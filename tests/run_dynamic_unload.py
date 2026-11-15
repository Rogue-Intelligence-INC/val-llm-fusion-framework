"""Dynamic unload A/B: fragment-only persistence + query fragment merge
(engine's dynamic-unload stack) ON vs OFF, same persistent dialogue.

Per arm: spawn a private valhalla_base serve with env toggles, run the
first 24 turns of the dialogue-drift schedule (10,006-line corpus,
targeted append), record per-turn wire MB / seconds / fog / pass /
fabrication, and whether/where the session hits the 413 wall.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, fabrication, gen_corpus, score  # noqa: E402
from run_extreme_targeted import targeted  # noqa: E402
from run_dialogue_drift import build_schedule, distractors  # noqa: E402
from valhalla_server import hybrid, internal  # noqa: E402

EXE = r"C:\Ai\Valhalla-main\target\release\valhalla_base.exe"
LM = r"C:\Ai\rogue-products\valhalla-server\data\builds\unified-core-v1"
OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme\dynamic_unload_ab.json")
TURNS = 24


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def spawn(port, extra_env):
    env = {**os.environ,
           "VALHALLA_LM_MODEL": LM,
           "VALHALLA_FATE_LINK_MAX_GROUP": "8",
           **extra_env}
    proc = subprocess.Popen([EXE, "serve", "--port", str(port)],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(60):
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"engine on {port} never became healthy")


def run_arm(name, port, extra_env, sched, corpus, corpus_text):
    proc = spawn(port, extra_env)
    engine = f"http://127.0.0.1:{port}/v1/infer"
    rows, sess, wall = [], None, None
    try:
        for turn, q in enumerate(sched):
            lines = targeted(q[1], corpus)
            if sess is None:
                payload = {"questions": [{"id": q[0], "text": q[1], "score_type": "open"}],
                           "config": CONFIG, "context": lines}
            else:
                payload = {"questions": [{"id": q[0], "text": q[1], "score_type": "open"}],
                           "config": CONFIG, "context": [],
                           "session": internal.slim_session(sess), "append_lines": lines}
            wire = len(json.dumps(payload)) / 1e6
            t0 = time.time()
            try:
                body = post(engine, payload)
            except urllib.error.HTTPError as e:
                wall = {"turn": turn, "wire_mb": round(wire, 2), "http": e.code}
                break
            sess = body.get("session")
            row = (body.get("rows") or [{}])[0]
            fog = (row.get("fog_decision") or {}).get("decision", "")
            sess_corpus = list((sess or {}).get("corpus") or [])
            answer, mode = hybrid.resolve_fate_answer(
                fog, str(row.get("answer", "")), sess_corpus,
                row.get("retrieval_corpus_idx"))
            sc = score(q, answer)
            rows.append({"turn": turn, "id": q[0], "kind": q[4], "fog": fog,
                         "pass": sc["pass"], "wire_mb": round(wire, 2),
                         "sec": round(time.time() - t0, 1),
                         "fabricated": fabrication(answer, corpus_text),
                         "frags": len(((sess or {}).get("snapshot") or {}).get("fragments") or [])})
            print(f"[{name}] t{turn:<3} {q[0]:<12} wire={wire:.2f}MB pass={sc['pass']} "
                  f"fog={fog} {rows[-1]['sec']}s", flush=True)
    finally:
        proc.terminate()
    core = [r for r in rows if r["id"] in ("n_start", "n_mid", "n_end")]
    return {"arm": name, "turns_done": len(rows), "wall": wall,
            "max_wire_mb": max((r["wire_mb"] for r in rows), default=0),
            "core_pass": f"{sum(1 for r in core if r['pass'])}/{len(core)}",
            "fabricated_total": sum(len(r["fabricated"]) for r in rows),
            "mean_sec": round(sum(r["sec"] for r in rows) / max(1, len(rows)), 2),
            "rows": rows}


def main():
    corpus, _ = gen_corpus(10006)
    corpus_text = "\n".join(corpus)
    sched = build_schedule(distractors(corpus))[:TURNS]
    arms = [
        run_arm("unload_on", 8789, {}, sched, corpus, corpus_text),
        run_arm("unload_off", 8790,
                {"VALHALLA_DYNAMIC_CORPUS_FRAGMENTS": "0",
                 "VALHALLA_FRAGMENT_ONLY": "0",
                 "VALHALLA_QUERY_FRAGMENT_MERGE": "0"},
                sched, corpus, corpus_text),
    ]
    OUT.write_text(json.dumps(arms, ensure_ascii=False, indent=2), encoding="utf-8")
    for a in arms:
        print(json.dumps({k: a[k] for k in ("arm", "turns_done", "wall", "max_wire_mb",
                                            "core_pass", "fabricated_total", "mean_sec")},
                         ensure_ascii=False), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
