# Val-LLM Fusion Framework

[中文](README.md) | **English** | [日本語](README.ja.md) | [Français](README.fr.md)

`valhalla-server` — a framework that decomposes any LLM by layer depth,
adapts it through the Fate system, freely fuses text/audio/image
modalities, bakes the result back into weights, and serves it as
structured sessions.

## Positioning

Valhalla is a **Fate-native long-horizon memory substrate**. It does not
retrieve and it does not generate. Experience is passively converged
into blocks (bundles); grasp is by field strength
(energy · √complexity); Fate's own cluster equation judges
"answer / silence"; learning happens *after* judgment, over usage
duration. What it sells is zero fabrication, silence discipline, and
behavioral stationarity — not answer accuracy. It is **not** a chatbot
memory API, **not** a RAG replacement, and it does not compete on the
LoCoMo leaderboard (same-question comparisons and boundaries:
`docs/VALHALLA_COMPETITIVE_COMPARISON_20261113.md`).

## How it works

1. **Decomposition** — an LM is split by layer depth into four quadrants
   (Q0→Hub, Q1→Tile, Q2→Stem, Q3→Hub) and adapted through Fate.
2. **Fusion** — text/audio/image ports of one sample pulse interleaved
   in a single Triad session; no isolation, no preset routing.
3. **Memory** — signals enter through Fate passive intake
   (`receive_passive`); convergence closes tiles/stem cells; material
   bundles into blocks when `(x−1)²` fits the block's own equation.
4. **Judgment** — the fog decision (emit / abstain / defer) is made by
   the engine's live Fate equation. The gateway must execute it; the
   transformer decode head may only phrase evidence Fate released and
   can never override an abstention.
5. **Baking** — structure is written back into a base model to produce
   a new checkpoint (the original is untouched), with model card and
   hashed registry.

## Measured evidence (fresh local reruns, 2026-11-15)

- Extreme targeted 806 questions: **12/12**, fabricated tokens **0**
  (identical on the previous and the newly synced engine).
- Multimodal gateway, Fate-native arm: **6/9**, fabricated **0**.
- Local 312-turn limits: core needles **89/90**, fabrication **0/312**,
  distractor questions answered 3/120 (silence is the default).
- Window invariance: core answers identical across window arms,
  fabrication 0; a 356-line arm now ingests in ~1.2 s
  (stepwise optimization: 462 s → 3.8 s on the heaviest arm).
- Scale: a 14.95M-token corpus showed scale-invariant behavior in the
  recorded stress series (see `docs/COMPREHENSIVE_SUMMARY_20261114.md`).

Reproduce: `.\reproduce.ps1 --quick` (full battery ~15 min), or see
`engine/README.md` for the local-engine battery.

## Repository layout

- `valhalla_server/` — the gateway (Fate-first adjudication, decode-head
  registry, modelops pipeline, VALENC1 engine provisioning).
- `engine/` — synced Rust sources of the ValhallaBase/Fate engine
  (source of truth: the Valhalla workspace; sync manifest inside).
- `docs/` — measured summaries, competitive comparison, status reports.
- `tests/` — probes and extreme-limit suites; `tests/reproduce_all.py`
  is the one-click reproduction driver.
- `runtime/` — engines at rest, encrypted (VALENC1); `data/` — decode
  head registry and model vault.

## Boundaries (stated plainly)

- Same-question benchmark comparisons against RAG/Mem0-class systems
  are documented, including where Valhalla loses; we do not claim
  LoCoMo wins.
- Proactive emission is not claimed.
- Today's known failure: instruction-form poison under false-premise
  questions can still be emitted (measured; under analysis, not
  patched over).

Contact: licensing@rogue-intelligence.com
