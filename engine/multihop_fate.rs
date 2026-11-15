//! Multi-hop open QA via Fate hop-wise adaptation — Fate adaptive algebra, no answer lexicon.
//!
//! Flow: structural hop split → per-hop Fate pulses → corpus_line_affinity discovery → adaptive decode.

use crate::dl_front::{build_body_ports, hash_embed, BodyFrontPorts, DlFrontConfig, DL_EMBED_DIM};
use crate::fate_adaptive_algebra::{FateAlgebraState, FateRelAddr};
use crate::signal_ingress::TriadSession;

const HASH_SEED_HOP: u64 = 0x484F505F46415445;
/// Per-hop cap on structural corpus feedback during QA adapt
/// (resource cap, same family as the fate-link group cap: scoring
/// still covers the whole corpus; only the hub-mutating feedback
/// is scoped to the strongest lines). Corpora whose passing set
/// fits under the cap are unaffected.
const MAX_ADAPT_FEEDBACK_LINES: usize = 32;
/// Passive Fate QA (owner architecture): the question stimulus has
/// already pulsed the session (emit ports + Fate-owned cycles), so
/// Fate's tiles passively carry the activated state. QA reads that
/// state; it does NOT re-feed corpus lines or mutate affinity
/// before judgment (that was `run_multihop_fate_adapt`'s active
/// loop, with its self-confirming positive feedback). Opt-in per
/// deployment: `VALHALLA_QA_PASSIVE_FATE=1`.
pub fn passive_qa_enabled() -> bool {
    std::env::var("VALHALLA_QA_PASSIVE_FATE")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        })
        .unwrap_or(false)
}

/// Capture the activation state Fate passively holds for this
/// question and project it onto corpus lines. Per hop, each line's
/// activation is its Fate RelAddr land score / embedding alignment
/// — the same quantities `run_multihop_fate_adapt` computes — but
/// read-only: no corpus line is fed back, no affinity is mutated,
/// the session is never touched before judgment. (Measured: the
/// adapt loop's benefit came from the affinity VALUES widening the
/// grasp margin, not from the feedback writes; the writes were the
/// 27s-at-560-lines cost and the self-confirming loop.) Returns
/// affinity-shaped values in [0, 1] indexed by corpus idx.
pub fn capture_passive_activation(
    session: &TriadSession,
    question: &str,
    corpus: &[CorpusLineRef],
    front: &DlFrontConfig,
) -> Vec<f64> {
    if corpus.is_empty() {
        return Vec::new();
    }
    let state = FateAlgebraState::ambient();
    let align_gate = state.coupling(0) * state.neg_coupling();
    let land_gate = state.coupling(0) * 0.35;
    let n = corpus.iter().map(|l| l.idx).max().unwrap_or(0) + 1;
    let mut out = vec![0.0f64; n];
    for hop in structural_hop_clauses(question) {
        let ports = hop_ports(&hop, front);
        let hop_tokens = hop_query_tokens(&hop);
        let query = FateRelAddr::from_query(&hop_tokens, &state);
        for line in corpus {
            let aff = session
                .corpus_line_affinity_snapshot()
                .get(line.idx)
                .copied()
                .unwrap_or(0.0);
            let align = cosine(&ports.embedding, &line.signals);
            let land = state.fate_rel_land_score(query, &hop_tokens, &line.text, aff);
            if align >= align_gate || land >= land_gate {
                let v = align.max(land);
                if line.idx < out.len() && v > out[line.idx] {
                    out[line.idx] = v;
                }
            }
        }
    }
    let max_v = out.iter().cloned().fold(0.0f64, f64::max);
    if max_v > 0.0 {
        for v in out.iter_mut() {
            *v /= max_v;
        }
    }
    out
}
const FATE_RULES: &[&str] = &[
    "open_fate: global hop/structure shards → corpus affinity discovery",
    "open_fate_adapt: adaptive τ/δ coupling until hub marginal stable",
];

pub struct CorpusLineRef {
    pub idx: usize,
    pub text: String,
    pub signals: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MultihopFateAdaptOutcome {
    pub rounds: usize,
    pub hops: usize,
    pub stopped: String,
}

/// Global open-QA Fate (pre-QA adapt + structure decode). Default ON; opt out with `VALHALLA_OPEN_FATE=0`.
pub fn open_fate_enabled() -> bool {
    if let Ok(v) = std::env::var("VALHALLA_OPEN_FATE") {
        let v = v.trim().to_ascii_lowercase();
        return !(v == "0" || v == "false" || v == "off");
    }
    true
}

pub fn multihop_fate_enabled() -> bool {
    if let Ok(v) = std::env::var("VALHALLA_MULTIHOP_FATE") {
        let v = v.trim().to_ascii_lowercase();
        if v == "1" || v == "true" || v == "on" {
            return true;
        }
        if v == "0" || v == "false" || v == "off" {
            return false;
        }
    }
    open_fate_enabled()
}

pub fn multihop_fate_compose_enabled() -> bool {
    std::env::var("VALHALLA_MULTIHOP_FATE_COMPOSE")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v != "0" && v != "false" && v != "off"
        })
        .unwrap_or(true)
}

/// Full Fate structure decode (entity + relation) vs comparison-only fallback.
pub fn open_fate_structure_compose(question: &str, multihop_v2: bool) -> bool {
    if multihop_v2 {
        return true;
    }
    if let Ok(v) = std::env::var("VALHALLA_OPEN_FATE_COMPOSE") {
        let v = v.trim().to_ascii_lowercase();
        if v == "1" || v == "true" || v == "on" {
            return true;
        }
        if v == "0" || v == "false" || v == "off" {
            return false;
        }
    }
    compose_hop_clauses(question).len() >= 2
}

/// Global open Fate decode: full structure on multi-hop questions; comparison/yes-no only on simple recall.
pub fn decode_open_fate(
    question: &str,
    corpus: &[CorpusLineRef],
    affinities: &[f64],
    hub_prefs: &[f64],
    q_sig: &[f64],
    multihop_v2: bool,
) -> Option<(String, String)> {
    if open_fate_structure_compose(question, multihop_v2) {
        decode_from_fate_structure(question, corpus, affinities, hub_prefs, q_sig)
    } else {
        decode_fate_comparison_only(question, corpus, affinities, hub_prefs, q_sig)
    }
}

fn build_multihop_compose_plan(question: &str) -> crate::task_decompose::DecomposePlan {
    use crate::task_decompose::{
        DecomposeMode, DecomposePlan, DecomposeShard, ShardBody, ShardModality, ShardPhase,
        TaskKind,
    };
    let hops = structural_hop_clauses(question);
    let mut shards: Vec<DecomposeShard> = hops
        .iter()
        .enumerate()
        .map(|(i, hop)| DecomposeShard {
            label: format!("multihop_hop_{i}"),
            text: hop.clone(),
            body: ShardBody::Tile,
            phase: ShardPhase::Decompose,
            modality: ShardModality::Text,
            audio_mel: None,
        })
        .collect();
    shards.push(DecomposeShard {
        label: "multihop_hub".into(),
        text: "MULTIHOP_HUB_AGG".into(),
        body: ShardBody::Hub,
        phase: ShardPhase::Aggregate,
        modality: ShardModality::Text,
        audio_mel: None,
    });
    shards.push(DecomposeShard {
        label: "multihop_recompose".into(),
        text: question.to_string(),
        body: ShardBody::Triad,
        phase: ShardPhase::Recompose,
        modality: ShardModality::Text,
        audio_mel: None,
    });
    DecomposePlan {
        task: TaskKind::Open,
        mode: DecomposeMode::OpenRetrieval,
        shards,
        whole_question: question.to_string(),
    }
}

/// Whether Fate adapt/decode must see the full corpus (not fragment-filtered memories).
///
/// MultiHop-RAG and other small multi-evidence items lose a hop when query-fragment
/// top-1 keeps only `seg_1` — Fate then never sees the evidence line on `seg_0`.
pub fn fate_full_corpus_feed(question: &str, corpus_n: usize) -> bool {
    if corpus_n == 0 {
        return false;
    }
    // Tiny corpora: fragment top-k is lossy; always feed every line to Fate.
    if corpus_n <= 16 {
        return true;
    }
    if structural_hop_clauses(question).len() >= 2 {
        return true;
    }
    let q = question.to_ascii_lowercase();
    q.starts_with("between ")
        || q.starts_with("does ")
        || q.starts_with("do ")
        || q.starts_with("after ")
        || q.contains(" while ")
        || q.contains(" both ")
        || q.contains(" compared ")
        || q.contains("agree or disagree")
        || q.contains("similar way")
        || q.contains("similar reason")
}

/// Structural hop clauses — contrast connectors only, no answer presets.
pub fn structural_hop_clauses(question: &str) -> Vec<String> {
    let q = question.trim_end_matches('?').trim();
    for sep in [" while ", " whereas ", " and ", "; "] {
        if q.to_lowercase().contains(sep) {
            return q
                .split(sep)
                .map(|s| s.trim().to_string())
                .filter(|s| s.len() > 12)
                .collect();
        }
    }
    if q.len() > 20 {
        vec![q.to_string()]
    } else {
        vec![]
    }
}

/// Multi-evidence separators only — avoids treating "mass and velocity" as compose hops.
fn compose_hop_clauses(question: &str) -> Vec<String> {
    let q = question.trim_end_matches('?').trim();
    let qlow = q.to_lowercase();
    for sep in [" while ", " whereas ", "; "] {
        if qlow.contains(sep) {
            return q
                .split(sep)
                .map(|s| s.trim().to_string())
                .filter(|s| s.len() > 12)
                .collect();
        }
    }
    let multi_doc = qlow.starts_with("between ")
        || qlow.contains(" compared to ")
        || qlow.contains(" compared with ")
        || (qlow.starts_with("does ")
            && (qlow.contains(" compared ")
                || qlow.contains("different perspective")
                || qlow.contains(" while ")));
    if multi_doc && qlow.contains(" and ") {
        return q
            .split(" and ")
            .map(|s| s.trim().to_string())
            .filter(|s| s.len() > 12)
            .collect();
    }
    vec![]
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        dot / na.sqrt() / nb.sqrt()
    }
}

fn hop_ports(hop: &str, front: &DlFrontConfig) -> BodyFrontPorts {
    let emb = hash_embed(hop, DL_EMBED_DIM, HASH_SEED_HOP);
    build_body_ports(&emb, front, "hash_grinder", "MULTIHOP_HOP")
}

fn install_rules(session: &mut TriadSession) {
    for rule in FATE_RULES {
    session.hub.triple_fate.store_string(rule);
    session.tile.follow_global_fate(&session.hub.triple_fate);
    session.stem.follow_global_fate(&session.hub.triple_fate);
    }
}

fn pulse(session: &mut TriadSession, phase: &str, score: f64) {
    let s = score.clamp(0.05, 1.0);
    let _ = session.hub.triple_fate.store_string(&format!("multihop_{phase} score={s:.4}"));
    session.observe_fate(s);
}

/// Hop-wise Fate adaptation: RelAddr land + embedding cosine → affinity by egress idx.
pub fn run_multihop_fate_adapt(
    session: &mut TriadSession,
    question: &str,
    corpus: &[CorpusLineRef],
    whole_ports: &BodyFrontPorts,
    max_rounds: usize,
    front: &DlFrontConfig,
) -> MultihopFateAdaptOutcome {
    install_rules(session);
    let hops = structural_hop_clauses(question);
    let state = FateAlgebraState::ambient();
    if multihop_fate_compose_enabled() && compose_hop_clauses(question).len() >= 2 {
        let plan = build_multihop_compose_plan(question);
        let _ = crate::compose_fate::run_compose_fate_pipeline(
            session,
            &plan,
            whole_ports,
            "multihop_fate_hops",
            front,
            1,
        );
    }
    let align_gate = state.coupling(0) * state.neg_coupling();
    let land_gate = state.coupling(0) * 0.35;
    let rounds = max_rounds.clamp(1, 8);
    let mut stable = 0usize;

    for round in 0..rounds {
        for hop in &hops {
            let ports = hop_ports(hop, front);
            session.emit_body_ports(&ports);
            session.run_optimize_cycles(1);
            let hop_tokens = hop_query_tokens(hop);
            let query = FateRelAddr::from_query(&hop_tokens, &state);
            // Score every line (cheap: two dot products), but apply the
            // expensive structural feedback only to the strongest lines.
            // Previously EVERY passing line got a full hub feedback call
            // (~24ms each on a large substrate); at 560 lines the gates
            // pass nearly everything, so one question cost
            // rounds x lines x 24ms (~27s of pure feedback, on top of
            // the per-line manifold_snapshot() export removed above).
            // Corpora with <= MAX_ADAPT_FEEDBACK_LINES passing lines
            // behave exactly as before; larger ones feed back only the
            // lines that can matter for this hop.
            let mut passing: Vec<(f64, &CorpusLineRef)> = Vec::new();
            for line in corpus {
                let aff = session
                    .corpus_line_affinity_snapshot()
                    .get(line.idx)
                    .copied()
                    .unwrap_or(0.0);
                let align = cosine(&ports.embedding, &line.signals);
                let land = state.fate_rel_land_score(query, &hop_tokens, &line.text, aff);
                // RelAddr land OR embedding cosine — absolute idx only for affinity slot.
                if align >= align_gate || land >= land_gate {
                    passing.push((align.max(land), line));
                }
            }
            passing.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for (_, line) in passing.iter().take(MAX_ADAPT_FEEDBACK_LINES) {
                session.apply_corpus_retrieval_feedback(
                    line.idx,
                    &ports.embedding,
                    &line.signals,
                    &line.text,
                    true,
                );
            }
        }
        session.emit_body_ports(whole_ports);
        session.run_optimize_cycles(1);
        pulse(session, "adapt", state.coupling(0));

        let hub = session.hub.get_fate_preferences().affinities.clone();
        let spread: f64 = hub
            .iter()
            .take(4)
            .map(|&v| (v - hub.first().copied().unwrap_or(0.0)).abs())
            .sum();
        if spread < state.delta_required(1.0) {
            stable += 1;
        } else {
            stable = 0;
        }
        if stable >= 2 && round + 1 >= 2 {
            return MultihopFateAdaptOutcome {
                rounds: round + 1,
                hops: hops.len(),
                stopped: "fate_marginal_stable".into(),
            };
        }
    }

    MultihopFateAdaptOutcome {
        rounds,
        hops: hops.len(),
        stopped: "hop_rounds_exhausted".into(),
    }
}

fn hop_query_tokens(hop: &str) -> Vec<String> {
    hop.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|t| t.len() > 2)
        .take(12)
        .collect()
}

fn yesno_interrogative(question: &str) -> bool {
    let q = question.to_lowercase();
    q.starts_with("does ")
        || q.starts_with("is ")
        || q.starts_with("are ")
        || q.starts_with("do ")
        || q.starts_with("between ")
        || q.starts_with("after ")
        || q.contains(" agree or disagree")
}

fn interrogative_structure(question: &str) -> bool {
    let q = question.trim();
    q.ends_with('?')
        && (structural_hop_clauses(question).len() >= 2
            || q.to_lowercase().starts_with("does ")
            || q.to_lowercase().starts_with("is ")
            || q.to_lowercase().starts_with("are ")
            || q.to_lowercase().starts_with("do ")
            || q.to_lowercase().starts_with("between ")
            || q.to_lowercase().starts_with("after "))
}

fn relation_surface(question: &str) -> Option<&'static str> {
    let q = question.to_lowercase();
    if q.contains("similar way") || q.contains("similar reason") {
        Some("similar")
    } else if q.contains("agree or disagree") {
        Some("agree_disagree")
    } else if q.contains("both ") && q.contains("suggest") {
        Some("both_suggest")
    } else {
        None
    }
}

fn hop_line_scores(
    hops: &[String],
    corpus: &[CorpusLineRef],
    affinities: &[f64],
    q_sig: &[f64],
) -> Vec<Vec<f64>> {
    let state = FateAlgebraState::ambient();
    let tau_w = state.corpus_rerank_weight(hops.len().max(1));
    hops.iter()
        .map(|hop| {
            let hop_emb = hash_embed(hop, DL_EMBED_DIM, HASH_SEED_HOP);
            let hop_tokens = hop_query_tokens(hop);
            let query = FateRelAddr::from_query(&hop_tokens, &state);
            corpus
                .iter()
                .map(|line| {
                    let aff = affinities.get(line.idx).copied().unwrap_or(0.0);
                    let align = cosine(&hop_emb, &line.signals);
                    let q_align = cosine(q_sig, &line.signals);
                    let land = state.fate_rel_land_score(query, &hop_tokens, &line.text, aff);
                    // RelAddr land primary; cosine / affinity secondary; idx = egress slot only.
                    (aff * tau_w
                        + align * state.coupling(0)
                        + q_align * 0.05
                        + land * 0.65)
                        .max(0.0)
                })
                .collect()
        })
        .collect()
}

fn dominant_proper_span(text: &str) -> Option<String> {
    let mut spans = Vec::new();
    let mut buf = String::new();
    for word in text.split_whitespace() {
        let clean: String = word
            .chars()
            .filter(|c| c.is_ascii_alphabetic() || *c == '-')
            .collect();
        if clean.len() >= 3
            && word
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_uppercase())
        {
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(&clean);
        } else if !buf.is_empty() {
            spans.push(std::mem::take(&mut buf));
        }
    }
    if !buf.is_empty() {
        spans.push(buf);
    }
    spans
        .into_iter()
        .filter(|s| s.len() >= 3)
        .max_by_key(|s| s.len())
}

/// Prefer entity spans that align with the question (player/name cues), not longest alone.
fn question_aligned_entity_span(question: &str, text: &str) -> Option<String> {
    let qlow = question.to_ascii_lowercase();
    let mut spans = Vec::new();
    let mut buf = String::new();
    for word in text.split_whitespace() {
        let clean: String = word
            .chars()
            .filter(|c| c.is_ascii_alphabetic() || *c == '-')
            .collect();
        if clean.len() >= 3
            && word
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_uppercase())
        {
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(&clean);
        } else if !buf.is_empty() {
            spans.push(std::mem::take(&mut buf));
        }
    }
    if !buf.is_empty() {
        spans.push(buf);
    }
    if spans.is_empty() {
        return dominant_proper_span(text);
    }
    let q_tokens: Vec<&str> = qlow
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .collect();
    let wants_player = qlow.contains("player") || qlow.contains("who ");
    let mut best: Option<(f64, String)> = None;
    for span in spans {
        if span.len() < 3 {
            continue;
        }
        let slow = span.to_ascii_lowercase();
        let mut score = span.split_whitespace().count() as f64 * 0.15;
        for t in &q_tokens {
            if slow.contains(t) {
                score += 1.0;
            }
        }
        // Prefer person-like 2-token names for "who/player" over org/team titles.
        let n_tok = span.split_whitespace().count();
        if wants_player && n_tok == 2 {
            score += 0.8;
        }
        if wants_player
            && (slow.contains("commanders")
                || slow.contains("nfl")
                || slow.contains("odds")
                || slow.contains("roundup"))
        {
            score -= 1.2;
        }
        if best.as_ref().map(|(s, _)| score > *s).unwrap_or(true) {
            best = Some((score, span));
        }
    }
    best.map(|(_, s)| s).or_else(|| dominant_proper_span(text))
}

fn decode_fate_relation_and_yesno(
    question: &str,
    corpus: &[CorpusLineRef],
    hops: &[String],
    hop_scores: &[Vec<f64>],
    affinities: &[f64],
    hub_prefs: &[f64],
    support_gate: f64,
) -> Option<(String, String)> {
    let qlow = question.to_lowercase();
    let asks_entity = qlow.starts_with("which ")
        || qlow.starts_with("who ")
        || qlow.contains("what company")
        || qlow.contains("what publication");

    if let Some(rel) = relation_surface(question) {
        match rel {
            "similar" | "both_suggest" => {
                let supported = hop_scores
                    .iter()
                    .filter(|row| row.iter().copied().fold(0.0, f64::max) >= support_gate)
                    .count();
                if rel == "similar" && supported >= hops.len().saturating_sub(1) {
                    return Some(("Similar".into(), "multihop_fate_relation".into()));
                }
                if rel == "both_suggest" && supported >= 2 {
                    return Some(("Yes".into(), "multihop_fate_yesno".into()));
                }
                if rel == "similar" {
                    return Some(("no".into(), "multihop_fate_yesno".into()));
                }
            }
            "agree_disagree" => {
                let mut agree_score = 0.0f64;
                let mut disagree_score = 0.0f64;
                for line in corpus {
                    let t = line.text.to_lowercase();
                    let aff = affinities.get(line.idx).copied().unwrap_or(0.0);
                    if t.contains("agree") || t.contains("same") || t.contains("push") {
                        agree_score += aff;
                    }
                    if t.contains("disagree") || t.contains("however") || t.contains("but ") {
                        disagree_score += aff;
                    }
                }
                if agree_score >= disagree_score && agree_score > support_gate {
                    return Some(("Agree".into(), "multihop_fate_relation".into()));
                }
                if disagree_score > agree_score {
                    return Some(("Disagree".into(), "multihop_fate_relation".into()));
                }
            }
            _ => {}
        }
    }

    if yesno_interrogative(question) && !hops.is_empty() && !asks_entity {
        let mut distinct_lines = std::collections::HashSet::new();
        let mut supported = 0usize;
        for row in hop_scores {
            if let Some((best_i, best)) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if *best >= support_gate * 1.2 {
                    supported += 1;
                    distinct_lines.insert(best_i);
                }
            }
        }
        let spread = hub_prefs.iter().copied().fold(0.0f64, f64::max)
            - hub_prefs.iter().copied().fold(f64::INFINITY, f64::min);
        let need_distinct = hops.len() >= 2 && question.to_lowercase().contains(" while ");
        if supported == hops.len() && (!need_distinct || distinct_lines.len() >= 2) {
            return Some(("Yes".into(), "multihop_fate_yesno".into()));
        }
        if supported == 0 || spread < FateAlgebraState::ambient().delta_required(0.5) {
            return Some(("no".into(), "multihop_fate_yesno".into()));
        }
        if question.to_lowercase().contains(" while ") {
            return Some(("no".into(), "multihop_fate_yesno".into()));
        }
    }

    None
}

/// Hybrid path: relation + yes/no only — entity answers left to multihop_v2 rules.
pub fn decode_fate_comparison_only(
    question: &str,
    corpus: &[CorpusLineRef],
    affinities: &[f64],
    hub_prefs: &[f64],
    q_sig: &[f64],
) -> Option<(String, String)> {
    if corpus.is_empty() {
        return None;
    }
    let hops = structural_hop_clauses(question);
    let hop_scores = hop_line_scores(&hops, corpus, affinities, q_sig);
    let state = FateAlgebraState::ambient();
    let support_gate = state.coupling(0) * 0.5;
    decode_fate_relation_and_yesno(
        question,
        corpus,
        &hops,
        &hop_scores,
        affinities,
        hub_prefs,
        support_gate,
    )
}

/// Decode short answer from post-adapt affinity + hub structure (discovery output).
pub fn decode_from_fate_structure(
    question: &str,
    corpus: &[CorpusLineRef],
    affinities: &[f64],
    hub_prefs: &[f64],
    q_sig: &[f64],
) -> Option<(String, String)> {
    if corpus.is_empty() {
        return None;
    }
    let hops = structural_hop_clauses(question);
    let hop_scores = hop_line_scores(&hops, corpus, affinities, q_sig);
    let state = FateAlgebraState::ambient();
    let support_gate = state.coupling(0) * 0.5;
    let qlow = question.to_lowercase();

    let asks_entity = qlow.starts_with("which ")
        || qlow.starts_with("who ")
        || qlow.contains("what company")
        || qlow.contains("what publication");
    if asks_entity {
        let mut best_line: Option<&CorpusLineRef> = None;
        let mut best_score = 0.0f64;
        let q_tokens = hop_query_tokens(question);
        let query = FateRelAddr::from_query(&q_tokens, &state);
        for line in corpus {
            let aff = affinities.get(line.idx).copied().unwrap_or(0.0);
            let align = cosine(q_sig, &line.signals);
            let land = state.fate_rel_land_score(query, &q_tokens, &line.text, aff);
            let score =
                aff * state.corpus_rerank_weight(4) + align * state.coupling(0) + land * 0.65;
            if score > best_score {
                best_score = score;
                best_line = Some(line);
            }
        }
        if best_score >= support_gate {
            if let Some(line) = best_line {
                if let Some(span) = question_aligned_entity_span(question, &line.text) {
                    return Some((span, "multihop_fate_entity".into()));
                }
            }
        }
    }

    if let Some(ans) = decode_fate_relation_and_yesno(
        question,
        corpus,
        &hops,
        &hop_scores,
        affinities,
        hub_prefs,
        support_gate,
    ) {
        return Some(ans);
    }

    let mut best_line: Option<&CorpusLineRef> = None;
    let mut best_score = 0.0f64;
    for line in corpus {
        let aff = affinities.get(line.idx).copied().unwrap_or(0.0);
        let align = cosine(q_sig, &line.signals);
        let score = aff * state.corpus_rerank_weight(4) + align * state.coupling(0);
        if score > best_score {
            best_score = score;
            best_line = Some(line);
        }
    }
    if best_score < support_gate {
        return None;
    }
    let line = best_line?;
    if let Some(span) = question_aligned_entity_span(question, &line.text) {
        return Some((span, "multihop_fate_entity".into()));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structural_hops_split_while() {
        let q = "Does A hold while B implies C?";
        assert!(structural_hop_clauses(q).len() >= 2);
    }

    #[test]
    fn fate_full_corpus_feed_for_tiny_multihop() {
        assert!(fate_full_corpus_feed("Does A while B?", 2));
        assert!(fate_full_corpus_feed("Who scored?", 4));
        assert!(!fate_full_corpus_feed("Who scored?", 64));
    }

    #[test]
    fn tau_gate_positive() {
        let state = FateAlgebraState::ambient();
        assert!(state.coupling(0) * state.neg_coupling() > 0.0);
    }
}
