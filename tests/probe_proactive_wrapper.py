"""Proactive via the repo's own in-process wrapper flow (their official test
path): fate_warmup_corpus -> set_context -> warmup question pairs -> idle."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"C:\Ai\Valhalla-main\tools\valhalla_inference")

from run_unique_points import FACTS_A, FACTS_B, UPDATES  # noqa: E402


def main():
    import valhalla_base as vb

    exe = Path(r"C:\Ai\Valhalla-main\target\release\valhalla_base.exe")
    print("bin exists:", exe.exists(), flush=True)
    if exe.exists():
        vb.BIN = exe
    base = vb.ValhallaBase(
        lm_model=r"C:\Ai\models\Qwen2.5-0.5B-Instruct",
        decode="hybrid",
        follow_up_decode="native_follow_up_aware",
        fate_ingress_routing="quad_cycle",
        fate_qa_feedback=True,
        fate_warmup_enabled=True,
        proactive_enabled=True,
        proactive_min_fate_signals=64,
        proactive_min_turns=2,
        incubation_cycles=2,
    )
    corpus = FACTS_A + FACTS_B + UPDATES
    base.fate_warmup_corpus(corpus, n_lines=4)
    base.set_context(corpus)
    pairs = [
        ("Who leads Project Aurora?", "Does the context say anything about Mara Voss?"),
        ("Who leads Project Borealis?", "Does the context say anything about Ilya Petrov?"),
        ("What is Project Aurora's internal codename?", "Does the context say anything about AUR-7?"),
    ]
    for i, (q, f) in enumerate(pairs):
        base.ask(q, qid=f"w{i}a")
        base.ask(f, qid=f"w{i}b")
        print(f"warm pair {i} done", flush=True)
    doc = base.proactive_idle()
    pro = doc.get("proactive") or {}
    print("PROACTIVE:", json.dumps(pro, ensure_ascii=False)[:1500], flush=True)
    print("ROWS:", json.dumps(doc.get("rows"), ensure_ascii=False)[:1000], flush=True)


if __name__ == "__main__":
    main()
