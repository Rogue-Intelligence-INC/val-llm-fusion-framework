"""Smoke: wire dedupe (drop duplicate top-level fragments) must not
break the three HTTP session paths -- QA, session_extend_only, and
checkpoint save -> reload (restore). Uses gen_corpus(160) needles."""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, gen_corpus, post  # noqa: E402
from valhalla_server.internal import slim_session  # noqa: E402

ENGINE = "http://127.0.0.1:8787/v1/infer"
OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme")


def mb(x):
    return round(len(json.dumps(x)) / 1e6, 2)


def ask(sess, qtext, qid="q"):
    payload = {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
               "config": CONFIG, "context": []}
    if sess:
        payload["session"] = sess
    body = post(ENGINE, payload)
    row = (body.get("rows") or [{}])[0]
    return (body.get("session"), str(row.get("answer", "")),
            (row.get("fog_decision") or {}), row.get("retrieval_corpus_idx"))


def main():
    corpus, _ = gen_corpus(120)
    ok = []

    body = post(ENGINE, {"questions": [{"id": "w", "text": "What is Project Aurora's internal codename?", "score_type": "open"}],
                         "config": CONFIG, "context": corpus[:30]})
    sess = body.get("session")
    print(f"bootstrap: lines={len(sess.get('corpus') or [])} full={mb(sess)}MB slim={mb(slim_session(sess))}MB", flush=True)
    # Stay inside the measured wire ceiling (~117 slimmed lines): send
    # lines[:90] -> 88 unique lines, slim 10.7MB. Needles at 5%/50% are
    # inside this range; the 95% needle is beyond the HTTP ceiling and
    # is covered by the targeted-pattern suite instead.
    for i in range(30, 90, 30):
        # Slim before every round-trip, as production keep_session does;
        # carrying the full copy dies with 413 around ~100 lines.
        try:
            body = post(ENGINE, {"questions": [], "session_extend_only": True,
                                 "session": slim_session(sess),
                                 "append_lines": corpus[i:i + 30],
                                 "config": CONFIG})
        except Exception as exc:  # noqa: BLE001
            print(f"extend at offset {i} FAILED slim={mb(slim_session(sess))}MB: {str(exc)[:80]}", flush=True)
            raise
        sess = body.get("session")
        print(f"extend @{i}: lines={len(sess.get('corpus') or [])} full={mb(sess)}MB slim={mb(slim_session(sess))}MB", flush=True)
    print(f"control session: lines={len(sess.get('corpus') or [])} wire={mb(sess)}MB "
          f"fragments={len(sess.get('fragments') or [])}")
    # NOTE: at ~150 ingested lines the full session is already >80MB and
    # any further full-session request 413s (measured separately); the
    # paths below therefore run on the slimmed copy, which is exactly
    # what production now stores/sends.

    slim = slim_session(sess)
    print(f"slimmed: wire={mb(slim)}MB fragments={len(slim.get('fragments') or [])} "
          f"snapshot_fragments={len((slim.get('snapshot') or {}).get('fragments') or [])}")
    ok.append(("wire shrunk >=40%", mb(slim) <= mb(sess) * 0.6))

    # Path 1: QA on slimmed session
    s2, b_start, _, _ = ask(slim, "What is Project Aurora's internal codename?", "n_start")
    s2, b_mid, _, midx = ask(s2, "Who leads the Borealis satellite program?", "n_mid")
    print(f"[slim QA] start={b_start[:70]!r}\n[slim QA] mid={b_mid[:70]!r} idx={midx}")
    ok.append(("slim QA n_start", "AUR-7" in b_start))
    mid_line = (s2.get("corpus") or [])[midx] if isinstance(midx, int) and 0 <= midx < len(s2.get("corpus") or []) else ""
    ok.append(("slim QA n_mid", "Halvorsen" in b_mid or "Halvorsen" in mid_line))

    # Path 2: extend on slimmed session
    body = post(ENGINE, {"questions": [], "session_extend_only": True,
                         "session": slim_session(s2),
                         "append_lines": ["Project Zephyr's internal codename is ZEP-9."],
                         "config": CONFIG})
    s3 = body.get("session")
    grew = "Project Zephyr's internal codename is ZEP-9." in (s3.get("corpus") or [])
    print(f"[slim extend] corpus grew with Zephyr line: {grew} lines={len(s3.get('corpus') or [])}")
    ok.append(("slim extend", grew))
    s3, b_zep, fogz, zidx = ask(s3, "What is Project Zephyr's internal codename?", "zep")
    print(f"[slim extend QA] {b_zep[:70]!r} fog={fogz.get('decision')} idx={zidx}")
    zep_line = (s3.get("corpus") or []).index("Project Zephyr's internal codename is ZEP-9.")
    ok.append(("slim extend QA", "ZEP-9" in b_zep or zidx == zep_line))

    # Path 3: checkpoint -> reload (restore) on slimmed session
    ckpt = OUT / "smoke_slim_session.json"
    ckpt.write_text(json.dumps(slim_session(s3), ensure_ascii=False), encoding="utf-8")
    restored = json.loads(ckpt.read_text(encoding="utf-8"))
    _s4, c_start, _, _ = ask(restored, "What is Project Aurora's internal codename?", "n_start")
    print(f"[restore QA] {c_start[:70]!r}")
    ok.append(("restore QA n_start", "AUR-7" in c_start))

    print("\n".join(f"  {'PASS' if v else 'FAIL'} {k}" for k, v in ok))
    print("SMOKE", "PASS" if all(v for _, v in ok) else "FAIL")


if __name__ == "__main__":
    main()
