"""One-click reproduction of Valhalla's consolidated status.

Brings up whatever is missing (llama decode head on 18080, gateway on
8780, internal engine on 8787), runs the measured suites, parses the
result artifacts against the recorded baselines, and writes a dated
status report (docs/CURRENT_STATUS_<date>.md + tests/results_repro/).

Usage:
  python tests/reproduce_all.py            # full battery (~15 min)
  python tests/reproduce_all.py --quick    # probes+smoke+806 tier only

Exit code 0 iff every gated suite meets its baseline (LLM-in-the-loop
suites get a tolerance of 1 question; fabrication must stay 0).
"""
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV_PY = Path(r"C:\Ai\Valhalla-main\.venv-llm\Scripts\python.exe")
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable
RES = REPO / "tests" / "results_extreme"
OUTDIR = REPO / "tests" / "results_repro"
OUTDIR.mkdir(exist_ok=True)
HEAD_GGUF_KEY = "mistral24-valhalla-s0.08"


def health(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def wait_health(url, seconds=120):
    t0 = time.time()
    while time.time() - t0 < seconds:
        h = health(url)
        if h:
            return h
        time.sleep(3)
    return None


def start_services():
    notes = []
    if not health("http://127.0.0.1:18080/health"):
        reg = json.loads((REPO / "data" / "decode_heads.json").read_text(encoding="utf-8"))
        gguf = reg[HEAD_GGUF_KEY]["gguf"]
        sys.path.insert(0, str(REPO))
        from valhalla_server import head_server
        ok = head_server.start(gguf, alias="mistral-24b", ctx=16384)
        notes.append(f"decode head start({HEAD_GGUF_KEY}): {ok}")
    else:
        notes.append("decode head: already healthy")
    if not health("http://127.0.0.1:8780/health"):
        subprocess.Popen([PY, "-m", "valhalla_server", "serve"], cwd=str(REPO),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        notes.append(f"gateway start: {bool(wait_health('http://127.0.0.1:8780/health', 90))}")
    else:
        notes.append("gateway: already healthy")
    if not health("http://127.0.0.1:8787/health"):
        # Boot under a throwaway model id: /v1/infer defaults to
        # unified-core-v1, and booting there would pollute the very
        # session the multimodal suite measures (found by this repro).
        payload = {"model_id": "repro-boot",
                   "questions": [{"id": "boot", "text": "What is Project Aurora's internal codename?",
                                  "score_type": "open"}],
                   "context": ["Project Aurora's internal codename is AUR-7."],
                   "config": {"decode": "hybrid", "max_new": 16}}
        req = urllib.request.Request("http://127.0.0.1:8780/v1/infer",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                r.read()
        except Exception as exc:  # noqa: BLE001
            notes.append(f"engine boot via gateway failed: {exc}")
        notes.append(f"engine start: {bool(wait_health('http://127.0.0.1:8787/health', 60))}")
    else:
        notes.append("engine: already healthy")
    return notes


def reset_stack():
    """Fresh gateway+engine so gateway-session suites (multimodal,
    probes) start from a clean unified-core-v1 session, exactly as in
    the standalone campaigns. Engine-direct suites carry explicit
    sessions and do not need this."""
    ps = ("Get-NetTCPConnection -LocalPort 8780,8787 -State Listen "
          "-ErrorAction SilentlyContinue | ForEach-Object "
          "{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   capture_output=True, timeout=60)
    time.sleep(2)
    subprocess.Popen([PY, "-m", "valhalla_server", "serve"], cwd=str(REPO),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok_gw = bool(wait_health("http://127.0.0.1:8780/health", 90))
    payload = {"model_id": "repro-boot",
               "questions": [{"id": "boot", "text": "What is Project Aurora's internal codename?",
                              "score_type": "open"}],
               "context": ["Project Aurora's internal codename is AUR-7."],
               "config": {"decode": "hybrid", "max_new": 16}}
    req = urllib.request.Request("http://127.0.0.1:8780/v1/infer",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
    except Exception:
        pass
    ok_eng = bool(wait_health("http://127.0.0.1:8787/health", 60))
    print(f"  reset_stack: gateway={ok_gw} engine={ok_eng}", flush=True)
    return ok_gw and ok_eng


def run_step(name, argv, timeout=1800):
    print(f"\n>>> {name}: {' '.join(argv[1:])}", flush=True)
    t1 = time.time()
    p = subprocess.run([PY] + argv, cwd=str(REPO), capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout)
    tail = (p.stdout or "")[-1500:]
    print(tail[-600:], flush=True)
    if p.returncode != 0:
        print(f"!!! {name} exit={p.returncode}\n{(p.stderr or '')[-800:]}", flush=True)
    return {"name": name, "exit": p.returncode, "seconds": round(time.time() - t1, 1),
            "tail": tail}

def load(name):
    try:
        return json.loads((RES / name).read_text(encoding="utf-8"))
    except Exception:
        return {}


def tally(rows):
    scored = [r for r in rows if r.get("pass") is not None]
    return (sum(1 for r in scored if r["pass"]), len(scored),
            sum(len(r.get("fabricated_tokens", [])) for r in rows))


def judge_suite(label, got, scored, fab, expect, tol=0):
    ok = scored > 0 and got >= expect - tol and fab == 0
    return {"suite": label, "score": f"{got}/{scored}", "expected": f">={expect - tol}/{scored or '?'}",
            "fabricated": fab, "verdict": "PASS" if ok else "FAIL"}


def collect(quick, steps):
    verdicts = []
    by_name = {s["name"]: s for s in steps}
    if "smoke" in by_name:
        ok = by_name["smoke"]["exit"] == 0 and "SMOKE PASS" in by_name["smoke"]["tail"]
        verdicts.append({"suite": "wire smoke (QA/extend/restore)", "score": "-",
                         "expected": "SMOKE PASS", "fabricated": 0,
                         "verdict": "PASS" if ok else "FAIL"})
    for s in by_name.values():
        if s["name"].startswith("probe"):
            verdicts.append({"suite": s["name"], "score": "-", "expected": "exit 0",
                             "fabricated": 0, "verdict": "PASS" if s["exit"] == 0 else "FAIL"})
    mm = load("multimodal_fixed.json").get("arms", {})
    if mm:
        g, n, f = tally(mm.get("valhalla", []))
        verdicts.append(judge_suite("multimodal (valhalla arm)", g, n, f, 9))
    sizes = [806] if quick else [806, 3006, 10006]
    for n in sizes:
        d = load(f"extreme_targeted_lines{n}.json")
        rows = (d.get("arms") or {}).get("hybrid", [])
        if rows:
            g, sc, f = tally(rows)
            verdicts.append(judge_suite(f"extreme targeted {n} (hybrid)", g, sc, f, 12))
    if not quick:
        d = load("enhancement_ceiling_lines100006.json")
        rows = (d.get("arms") or {}).get("hybrid", [])
        if rows:
            g, sc, f = tally(rows)
            verdicts.append(judge_suite("enhancement ceiling 1.5M tok (hybrid)", g, sc, f, 17, tol=1))
        d = load("boost_adapt_lines100006.json")
        rows = (d.get("arms") or {}).get("boost_v2", [])
        if rows:
            g, sc, f = tally(rows)
            verdicts.append(judge_suite("BOOST adapt v2 (spread+operator+discipline)", g, sc, f, 20, tol=1))
    return verdicts


def main():
    quick = "--quick" in sys.argv
    print("== Valhalla one-click reproduction ==", flush=True)
    notes = start_services()
    for n in notes:
        print("  svc:", n, flush=True)
    eng = health("http://127.0.0.1:8787/health") or {}

    steps = []
    if not quick:
        reset_stack()
        steps.append(run_step("multimodal", ["tests/run_multimodal_fixed.py"], 1800))
    steps.append(run_step("probe_fate_gateway", ["tests/probe_fate_gateway.py"], 600))
    steps.append(run_step("probe_corpus_switch", ["tests/probe_fate_corpus_switch.py"], 600))
    steps.append(run_step("smoke", ["tests/smoke_wire_dedupe.py"], 600))
    sizes = "806" if quick else "806,3006,10006"
    steps.append(run_step("extreme_targeted", ["tests/run_extreme_targeted.py", sizes, "native,hybrid,mistral"], 1800))
    if not quick:
        steps.append(run_step("enhancement_ceiling", ["tests/run_enhancement_ceiling.py", "100006", "hybrid,rag"], 1800))
        steps.append(run_step("boost_adapt", ["tests/run_boost_adapt.py"], 1800))

    verdicts = collect(quick, steps)
    all_pass = all(v["verdict"] == "PASS" for v in verdicts) and bool(verdicts)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"when": stamp, "quick": quick, "engine": eng, "services": notes,
              "steps": [{k: s[k] for k in ("name", "exit", "seconds")} for s in steps],
              "verdicts": verdicts, "overall": "PASS" if all_pass else "FAIL"}
    (OUTDIR / f"repro_{stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    date = datetime.now().strftime("%Y%m%d")
    lines = [f"# Valhalla 现状巩固报告（{date}）", "",
             f"一键复现：`powershell -File reproduce.ps1`（full）/ `--quick` 快速档。",
             f"引擎：{eng.get('service')} {eng.get('version')} · 协议 {eng.get('protocol')} · 解码头 {HEAD_GGUF_KEY}（16K ctx）",
             f"总判定：**{report['overall']}**", "",
             "| 套件 | 实测 | 基线 | 编造 | 判定 |", "|---|---|---|---|---|"]
    for v in verdicts:
        lines.append(f"| {v['suite']} | {v['score']} | {v['expected']} | {v['fabricated']} | {v['verdict']} |")
    lines += ["",
              "倍率口径（工件派生）：工作语料 14,945,110 token = 原生窗口 912×（千倍级）、全上下文死亡点 331×、每题触达语料 ~1/83,000、会话线缆 3.42MB 恒定。",
              "",
              "已知边界（不因复现通过而改变）：LoCoMo-15 0/15 不赢；多跳残留 hop_employer（实体消歧，失败模式为弃权）；矛盾事实静默选后写；proactive 发声未实现；特征抓取为同文件再识别。",
              "",
              "服务编排记录：" + "；".join(notes)]
    doc = REPO / "docs" / f"CURRENT_STATUS_{date}.md"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n== OVERALL {report['overall']} · report {doc}", flush=True)
    for v in verdicts:
        print(f"  {v['verdict']}  {v['suite']}: {v['score']} (fab={v['fabricated']})", flush=True)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
