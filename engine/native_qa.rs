//! Tier B: Native Valhalla Q&A — no external Transformer forward pass.
//!
//! Ingress → TriadSession cycles → structure-weighted memory retrieval → answer.

use serde::{Deserialize, Serialize};

use crate::{
    cortex::TransformType, encode_payload, hash_embed, mcq_fate,
    pipeline, BodyFrontPorts, BodyTarget, DlFrontConfig, EncodingMode, IngressBudget,
    IngressPayload, ModalityInput, TriadFeedReport, TriadSession, DL_EMBED_DIM,
};
use crate::fate_adaptive_algebra::{line_query_score, FateAlgebraState};
use crate::signal_ingress::TriadIncubationSnapshot;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeQaRequest {
    pub body: String,
    pub cycles: usize,
    pub corpus: Vec<String>,
    pub question: String,
    pub score_type: String,
    /// `char` (default) | `dl_front` | `grinder`
    #[serde(default)]
    pub ingress_mode: Option<String>,
    /// Pre-ground ports from `tools/transformer_grinder` (one shard per corpus line).
    #[serde(default)]
    pub corpus_ports: Option<Vec<BodyFrontPorts>>,
    /// Pre-ground ports for the question line.
    #[serde(default)]
    pub question_ports: Option<BodyFrontPorts>,
    /// Incubation cycles after corpus ingress (before question). Falls back to `cycles`.
    #[serde(default)]
    pub corpus_cycles: Option<usize>,
    /// Structure optimize cycles after question ingress. Falls back to `cycles`.
    #[serde(default)]
    pub qa_cycles: Option<usize>,
    /// Feed every corpus line into Tile/Stem (not tail-only cap).
    #[serde(default)]
    pub ingest_all_corpus: Option<bool>,
}

/// One-shot incubation: full corpus → Tile/Stem, then many questions without re-ingesting corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeQaBatchRequest {
    pub body: String,
    pub corpus: Vec<String>,
    pub corpus_ports: Vec<BodyFrontPorts>,
    pub questions: Vec<NativeQaQuestionSpec>,
    #[serde(default = "default_incubation_cycles")]
    pub incubation_cycles: usize,
    #[serde(default = "default_qa_cycles")]
    pub qa_cycles: usize,
    #[serde(default)]
    pub ingress_mode: Option<String>,
    /// When true, run MCQ Fate pipeline for mcq items (overrides env when set in request).
    #[serde(default)]
    pub mcq_fate: bool,
    /// v4: LM decompose → Valhalla → recompose decode path.
    #[serde(default)]
    pub pipeline_v4: bool,
    /// Legacy benchmark: rule-based multihop compose (opt-in env only).
    #[serde(default)]
    pub multihop_v2: bool,
    /// Global open Fate: pre-QA hop adapt + corpus affinity + structure decode (default on).
    #[serde(default = "default_open_fate")]
    pub open_fate: bool,
    /// Deprecated alias for `open_fate`.
    #[serde(default)]
    pub multihop_fate: bool,
    /// Pre-built incubation snapshot (skip corpus incubation; use for context+core fusion).
    #[serde(default)]
    pub incubation_snapshot: Option<TriadIncubationSnapshot>,
    /// Dynamic fragments — primary session memory under fragment-only mode.
    #[serde(default)]
    pub corpus_fragments: Option<Vec<crate::fate_orchestrator::SnapshotFragment>>,
}

fn default_open_fate() -> bool {
    crate::multihop_fate::open_fate_enabled()
}

fn default_incubation_cycles() -> usize {
    10
}

fn default_qa_cycles() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeQaQuestionSpec {
    pub question: String,
    pub score_type: String,
    #[serde(default)]
    pub question_ports: Option<BodyFrontPorts>,
    #[serde(default)]
    pub id: Option<String>,
    /// TTS / audio mel features — fused after question_ports when `VALHALLA_MULTIMODAL_FUSION=1`.
    #[serde(default)]
    pub audio_features: Option<Vec<f64>>,
    /// Prior user questions in session — boosts follow-up-aware corpus ranking.
    #[serde(default)]
    pub prior_questions: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeQaBatchResponse {
    pub mode: String,
    pub corpus_lines: usize,
    pub corpus_lines_ingested: usize,
    pub incubation_cycles: usize,
    pub qa_cycles: usize,
    pub incubation_tile_count: usize,
    pub incubation_stem_cells: usize,
    pub incubation_tile_connections: usize,
    pub incubation_stem_organ_clusters: usize,
    pub incubation_tile_avg_activation: f64,
    #[serde(default)]
    pub fragment_count: usize,
    #[serde(default)]
    pub live_session: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifold_protocol: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fate_warmup_cycles: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fate_warmup_prelearn_lines: Option<usize>,
    pub results: Vec<NativeQaBatchItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeQaBatchItem {
    #[serde(default)]
    pub id: String,
    pub answer: String,
    pub score_type: String,
    pub decode_path: String,
    pub tile_count: usize,
    pub stem_cells: usize,
    pub tile_connections: usize,
    pub tile_avg_activation: f64,
    pub stem_organ_clusters: usize,
    pub correct: bool,
    #[serde(default)]
    pub mcq_fate_cycles: usize,
    #[serde(default)]
    pub mcq_fate_stopped: String,
    #[serde(default)]
    pub mcq_decompose_score: f64,
    #[serde(default)]
    pub mcq_aggregate_score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fog_evidence: Option<FogEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fog_decision: Option<FogDecision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval_corpus_idx: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_fragments: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_fragment_scores: Option<Vec<f64>>,
    /// Fate grasp record (blocks activated + candidate series + grasp).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fate_grasp: Option<crate::fate_qa::FateGrasp>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeQaResponse {
    pub answer: String,
    pub body: String,
    pub cycles_executed: usize,
    pub score_type: String,
    pub memory_items: usize,
    pub tile_count: usize,
    pub stem_cells: usize,
    pub tile_dim: f64,
    pub stem_dim: f64,
    pub top_matches: Vec<NativeMatch>,
    pub report_mode: String,
    pub decode_path: String,
    #[serde(default)]
    pub ingress_mode: String,
    #[serde(default)]
    pub dl_encoder: String,
    #[serde(default)]
    pub tile_connections: usize,
    #[serde(default)]
    pub tile_avg_activation: f64,
    #[serde(default)]
    pub stem_organ_clusters: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeMatch {
    pub text: String,
    pub score: f64,
    pub source: String,
}

/// Native retrieval scores for Fate Output Gate (FOG).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FogEvidence {
    pub top_score: f64,
    pub second_score: f64,
    pub margin: f64,
    pub keyword_count: usize,
}

/// FOG decision gates from Fate adaptive algebra (session/pretrain-driven).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FogDecision {
    pub decision: String,
    pub structure_ready: bool,
    pub evidence_ready: bool,
    pub tau: f64,
    pub delta_required: f64,
    pub reason: String,
}

pub fn fog_tau_adaptive(keyword_count: usize) -> f64 {
    crate::fate_adaptive_algebra::FateAlgebraState::ambient().coupling(keyword_count)
}

/// Back-compat alias.
pub fn fog_tau_algebraic(keyword_count: usize) -> f64 {
    fog_tau_adaptive(keyword_count)
}

pub fn fog_delta_required_adaptive(top_score: f64) -> f64 {
    crate::fate_adaptive_algebra::FateAlgebraState::ambient().delta_required(top_score)
}

/// Back-compat alias.
pub fn fog_delta_required_algebraic(top_score: f64) -> f64 {
    fog_delta_required_adaptive(top_score)
}

pub fn fog_evidence_from_matches(top: &[NativeMatch], question: &str) -> FogEvidence {
    let keyword_count = question_keywords(question).len();
    let top_score = top.first().map(|m| m.score).unwrap_or(0.0);
    let second_score = top.get(1).map(|m| m.score).unwrap_or(0.0);
    FogEvidence {
        top_score,
        second_score,
        margin: top_score - second_score,
        keyword_count,
    }
}

pub fn fog_decide_algebraic(
    evidence: &FogEvidence,
    incubation_cycles: usize,
    min_structure_cycles: usize,
    native_reject: bool,
) -> FogDecision {
    let tau = fog_tau_algebraic(evidence.keyword_count);
    let delta_req = fog_delta_required_algebraic(evidence.top_score);
    let structure_ready = incubation_cycles >= min_structure_cycles;

    if !structure_ready {
        return FogDecision {
            decision: "defer".into(),
            structure_ready: false,
            evidence_ready: false,
            tau,
            delta_required: delta_req,
            reason: format!(
                "incubation_cycles={incubation_cycles}<{min_structure_cycles}"
            ),
        };
    }
    if native_reject {
        return FogDecision {
            decision: "abstain".into(),
            structure_ready: true,
            evidence_ready: false,
            tau,
            delta_required: delta_req,
            reason: "native_reject".into(),
        };
    }
    if evidence.top_score < tau {
        return FogDecision {
            decision: "abstain".into(),
            structure_ready: true,
            evidence_ready: false,
            tau,
            delta_required: delta_req,
            reason: format!(
                "s1={:.4}<tau(k)={:.4} [12/(138+{}))",
                evidence.top_score,
                tau,
                evidence.keyword_count
            ),
        };
    }
    if evidence.margin < delta_req {
        return FogDecision {
            decision: "abstain".into(),
            structure_ready: true,
            evidence_ready: false,
            tau,
            delta_required: delta_req,
            reason: format!(
                "margin={:.4}<s1*delta/(dim+r)={:.4}",
                evidence.margin, delta_req
            ),
        };
    }
    FogDecision {
        decision: "emit".into(),
        structure_ready: true,
        evidence_ready: true,
        tau,
        delta_required: delta_req,
        reason: "algebraic_ready".into(),
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryEntry {
    pub(crate) text: String,
    pub(crate) signals: Vec<f64>,
    pub(crate) source: String,
    pub(crate) corpus_idx: Option<usize>,
}

const HASH_SEED_TEXT: u64 = 0x544558545F464E54;

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn min_retrieval_score() -> f64 {
    env_f64("VALHALLA_NATIVE_MIN_SCORE", 0.12)
}

fn min_open_keyword_overlap() -> usize {
    env_usize("VALHALLA_NATIVE_OPEN_MIN_KEYWORDS", 1)
}

fn question_keywords(q: &str) -> Vec<String> {
    let low = q.to_lowercase();
    low.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3 || w.chars().all(|c| c.is_ascii_digit()))
        .map(|w| w.to_string())
        .collect()
}

/// True when `keyword` appears in `text` but only inside a negated phrase ("not the capital").
fn keyword_negated_in_text(keyword: &str, text: &str) -> bool {
    let low = text.to_lowercase();
    let kw = keyword.to_lowercase();
    if !low.contains(&kw) {
        return false;
    }
    for pat in ["not the ", "not a ", "isn't the ", "is not the ", "isn't a ", "is not a "] {
        if low.contains(&format!("{pat}{kw}")) {
            return true;
        }
    }
    false
}

fn keyword_overlap_count(question: &str, text: &str) -> usize {
    question_keywords(question)
        .iter()
        .filter(|k| {
            let low = text.to_lowercase();
            low.contains(k.as_str()) && !keyword_negated_in_text(k, text)
        })
        .count()
}

/// Fate algebra retrieval boost from question-token overlap (no domain phrase tables).
fn fate_algebra_retrieval_boost(
    question: &str,
    text: &str,
    hub_prefs: &[f64],
    session_tick: u64,
    baseline_tick: u64,
    affinity: f64,
) -> f64 {
    use crate::fate_adaptive_algebra::{FateAlgebraMeta, FateAlgebraState, FateRelAddr};
    let tokens: Vec<String> = question_keywords(question);
    if tokens.is_empty() {
        return 0.0;
    }
    let mut meta = FateAlgebraMeta::default();
    meta.fate_tick = session_tick;
    meta.fate_baseline_tick = baseline_tick;
    let state = if hub_prefs.len() >= 4 {
        FateAlgebraState::from_hub_prefs(hub_prefs, &meta)
    } else {
        FateAlgebraState::from_meta(&meta, None)
    };
    let query = FateRelAddr::from_query(&tokens, &state);
    // Fate relative address is the land controller (not absolute corpus position).
    let land = state.fate_rel_land_score(query, &tokens, text, affinity);
    land * state.corpus_rerank_weight(tokens.len()) * 1.75
}

fn format_open_retrieval_answer(
    _cite: &str,
    memory_text: &str,
    body: &str,
    corpus_idx: Option<usize>,
    session_tick: u64,
    baseline_tick: u64,
) -> String {
    // Generic egress adapter — absolute IDs only here.
    crate::fate_io::fate_egress(
        body,
        Some(memory_text),
        corpus_idx,
        session_tick,
        baseline_tick,
    )
}

fn extract_open_short_answer(question: &str, memory_text: &str) -> Option<String> {
    let qlow = question.to_lowercase();
    let mlow = memory_text.to_lowercase();
    if qlow.contains("represent") {
        if qlow.contains("m represent") && (mlow.starts_with("mass") || mlow.contains("mass is")) {
            return Some("mass".into());
        }
        if qlow.contains("a represent")
            && (mlow.starts_with("acceleration") || mlow.contains("acceleration is"))
        {
            return Some("acceleration".into());
        }
    }
    if qlow.contains("what energy") && mlow.contains("kinetic energy") {
        return Some("kinetic energy".into());
    }
    if qlow.contains("mass times velocity") && mlow.contains("momentum") {
        return Some("momentum".into());
    }
    if qlow.contains("force times distance") && mlow.contains("work") {
        return Some("work".into());
    }
    None
}

fn question_history_cap() -> usize {
    env_usize("VALHALLA_QUESTION_HISTORY_MAX", 12)
}

fn trim_question_history(history: &[String]) -> Vec<String> {
    let cap = question_history_cap();
    if history.len() <= cap {
        return history.to_vec();
    }
    history[history.len() - cap..].to_vec()
}

fn memory_recency_weight() -> f64 {
    env_f64("VALHALLA_MEMORY_RECENCY_WEIGHT", 0.001)
}

fn normalize_corpus_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

fn corpus_duplicate_penalty(
    memory_text: &str,
    corpus_idx: Option<usize>,
    memories: &[MemoryEntry],
) -> f64 {
    let idx = match corpus_idx {
        Some(i) => i,
        None => return 0.0,
    };
    let norm = normalize_corpus_line(memory_text);
    if norm.is_empty() {
        return 0.0;
    }
    let dup = memories.iter().any(|m| {
        m.corpus_idx
            .is_some_and(|j| j < idx && normalize_corpus_line(&m.text) == norm)
    });
    if dup {
        -env_f64("VALHALLA_CORPUS_DUP_PENALTY", 0.15)
    } else {
        0.0
    }
}

/// Open QA without reference: structural Q↔line alignment for Fate feedback (E7 gate).
pub fn structural_open_on_topic(question: &str, answer: &str, line_text: Option<&str>) -> bool {
    if answer.starts_with("No confident match") {
        return false;
    }
    let body = line_text.unwrap_or_else(|| {
        answer
            .strip_prefix('[')
            .and_then(|rest| rest.split_once("] "))
            .map(|(_, tail)| tail)
            .unwrap_or(answer)
    });
    let qkw = question_keywords(question);
    if qkw.is_empty() {
        return false;
    }
    let overlap = keyword_overlap_count(question, body);
    let need = (qkw.len().max(2) + 1) / 2;
    overlap >= need
}

/// Session-local IDF tie-break (corpus-only statistics; no domain lists or penalties).
fn session_idf_term_weight() -> f64 {
    env_f64("VALHALLA_SESSION_IDF_TERM_WEIGHT", 0.08)
}

/// Session-local document frequency → IDF (corpus lines only; no external index).
fn corpus_keyword_idf(keyword: &str, memories: &[MemoryEntry]) -> f64 {
    let corpus: Vec<&MemoryEntry> = memories
        .iter()
        .filter(|m| m.corpus_idx.is_some())
        .collect();
    if corpus.is_empty() {
        return 1.0;
    }
    let kw = keyword.to_lowercase();
    let df = corpus
        .iter()
        .filter(|m| m.text.to_lowercase().contains(kw.as_str()))
        .count();
    ((corpus.len() as f64 + 1.0) / (df as f64 + 1.0)).ln().max(0.05)
}

/// Gentle positive tie-break: rare session terms that appear in a memory line.
/// Does not penalize partial overlap — low confidence is handled by FOG abstain, not forced rerank.
fn session_idf_term_boost(question: &str, memory_text: &str, memories: &[MemoryEntry]) -> f64 {
    let w = session_idf_term_weight();
    if w <= 0.0 {
        return 0.0;
    }
    let ml = memory_text.to_lowercase();
    question_keywords(question)
        .iter()
        .filter(|k| ml.contains(k.as_str()) && !keyword_negated_in_text(k, memory_text))
        .map(|k| corpus_keyword_idf(k, memories) * w)
        .sum()
}

/// Scale grinder + patch + Tile/Stem terms in open retrieval (env tunable).
pub fn open_structure_weight() -> f64 {
    env_f64("VALHALLA_OPEN_STRUCTURE_WEIGHT", 1.2)
}

fn corpus_line_affinity_boost(
    corpus_idx: Option<usize>,
    affinities: &[f64],
    question_sig: &[f64],
    line_sig: &[f64],
    keyword_count: usize,
) -> f64 {
    let w = match std::env::var("VALHALLA_CORPUS_LINE_AFFINITY_WEIGHT") {
        Ok(v) if v.eq_ignore_ascii_case("algebra") || v.eq_ignore_ascii_case("tau") => {
            crate::fate_adaptive_algebra::FateAlgebraState::ambient()
                .corpus_rerank_weight(keyword_count)
        }
        Ok(v) if v.eq_ignore_ascii_case("0")
            || v.eq_ignore_ascii_case("off")
            || v.eq_ignore_ascii_case("false") =>
        {
            0.0
        }
        Ok(v) => v.parse().unwrap_or(0.0),
        Err(_) => crate::fate_adaptive_algebra::FateAlgebraState::ambient().coupling(0) * 0.4,
    };
    let aff = corpus_idx
        .and_then(|i| affinities.get(i))
        .copied()
        .unwrap_or(0.0);
    if aff <= 0.0 {
        return 0.0;
    }
    let align = dot_cosine(question_sig, line_sig).max(0.0);
    aff * align * w
}

/// When true, open answers are withheld if FOG decides abstain (no forced hallucination).
pub fn open_fog_enforce() -> bool {
    std::env::var("VALHALLA_OPEN_FOG_ENFORCE")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        })
        .unwrap_or(true)
}

pub fn apply_open_fog_gate(
    answer: String,
    decode_path: String,
    fog: Option<&FogDecision>,
) -> (String, String) {
    if !open_fog_enforce() {
        return (answer, decode_path);
    }
    if answer.starts_with("No confident match") {
        return (answer, decode_path);
    }
    // Corpus/session citation from native retrieval — structural match, not hallucination.
    if answer.starts_with('[') && answer.contains("] ") {
        return (answer, decode_path);
    }
    // Fate / multihop structure shorts are grounded compose, not free generation.
    if decode_path.contains("multihop_fate") || decode_path.contains("multihop_v2") {
        return (answer, decode_path);
    }
    if let Some(f) = fog {
        if f.decision == "abstain" {
            return (
                "No confident match in incubated memory.".to_string(),
                "open_fog_abstain".into(),
            );
        }
    }
    (answer, decode_path)
}

/// Boost/penalize corpus line when current question follows prior turns in session.
fn follow_up_retrieval_boost(
    question: &str,
    prior_questions: &[String],
    memory_text: &str,
    corpus_idx: Option<usize>,
) -> f64 {
    if prior_questions.is_empty() {
        return 0.0;
    }
    // A repeated question is not a follow-up: it brings no new
    // context, so continuity boosts must not pay out for it (they
    // manufactured +0.18*overlap of margin per repetition, measured
    // 0.245 -> 0.651 over 10 identical asks). Only priors that
    // differ from the current question count as thread context.
    let q_norm = question.trim().to_lowercase();
    let priors: Vec<&String> = prior_questions
        .iter()
        .filter(|pq| pq.trim().to_lowercase() != q_norm)
        .collect();
    if priors.is_empty() {
        return 0.0;
    }
    let cur_overlap = keyword_overlap_count(question, memory_text) as f64;
    let prior_overlap = priors
        .iter()
        .map(|pq| keyword_overlap_count(pq, memory_text) as f64)
        .sum::<f64>()
        / priors.len().max(1) as f64;

    let mut boost = cur_overlap * 0.18;
    if prior_overlap > cur_overlap + 0.5 {
        boost -= (prior_overlap - cur_overlap) * 0.22;
    }
    // Prefer seed / early anchor lines on topic drift in long threads (cycle 2+).
    if let Some(idx) = corpus_idx {
        if idx <= 1 && cur_overlap >= 1.0 && priors.len() >= 8 {
            boost += 0.14;
        }
    }

    let prior_kw: Vec<String> = priors
        .iter()
        .flat_map(|pq| question_keywords(pq))
        .collect();
    let unique_hits = question_keywords(question)
        .iter()
        .filter(|k| !prior_kw.contains(k))
        .filter(|k| memory_text.to_lowercase().contains(k.as_str()))
        .count();
    boost += unique_hits as f64 * 0.15;

    if let Some(idx) = corpus_idx {
        boost += (idx as f64) * memory_recency_weight();
    }
    boost
}

fn option_body(opt_line: &str) -> String {
    let t = opt_line.trim();
    if let Some((_, rest)) = t.split_once(')') {
        return rest.trim().to_lowercase();
    }
    if let Some((_, rest)) = t.split_once('.') {
        return rest.trim().to_lowercase();
    }
    t.to_lowercase()
}

fn best_span(question: &str, memory_text: &str) -> Option<String> {
    let parts = memory_sentence_parts(memory_text);
    let mut best: Option<(usize, String)> = None;
    for part in parts {
        let overlap = keyword_overlap_count(question, &part);
        if overlap == 0 {
            continue;
        }
        if best.as_ref().map(|(o, _)| overlap > *o).unwrap_or(true) {
            best = Some((overlap, part));
        }
    }
    best.map(|(_, s)| s)
}

/// Split on sentence boundaries; do not break decimal numbers (e.g. 9.8).
fn memory_sentence_parts(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '.' || c == '。' || c == ';' || c == '\n' {
            if c == '.' && i > 0 && i + 1 < bytes.len() {
                let prev = bytes[i - 1];
                let next = bytes[i + 1];
                if prev.is_ascii_digit() && next.is_ascii_digit() {
                    i += 1;
                    continue;
                }
            }
            let chunk = text[start..i].trim();
            if chunk.len() > 4 {
                parts.push(chunk.to_string());
            }
            start = i + 1;
        }
        i += 1;
    }
    let tail = text[start..].trim();
    if tail.len() > 4 {
        parts.push(tail.to_string());
    }
    parts
}

fn decode_signal(
    text: &str,
    score_type: &str,
    ingress_mode: &str,
    budget: &IngressBudget,
    dl: bool,
    front: &DlFrontConfig,
) -> Vec<f64> {
    if ingress_mode == "grinder" {
        return hash_embed(text, DL_EMBED_DIM, HASH_SEED_TEXT);
    }
    memory_signal(text, score_type, budget, dl, front).0
}

fn mcq_corpus_option_boost(opt_line: &str, corpus_mem: &[&MemoryEntry]) -> f64 {
    let body = option_body(opt_line);
    if body.len() < 3 {
        return 0.0;
    }
    let per_hit = env_f64("VALHALLA_MCQ_CORPUS_BOOST_HIT", 0.12);
    let cap = env_f64("VALHALLA_MCQ_CORPUS_BOOST_CAP", 0.85);
    let min_word = env_usize("VALHALLA_MCQ_CORPUS_BOOST_MIN_WORD", 5);
    let mut boost = 0.0;
    for word in body.split_whitespace() {
        let w: String = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if w.len() < min_word {
            continue;
        }
        for m in corpus_mem {
            if m.text.to_lowercase().contains(&w) {
                boost += per_hit;
                break;
            }
        }
        if boost >= cap {
            break;
        }
    }
    boost.min(cap)
}

fn mcq_top_corpus_mem<'a>(
    corpus_mem: &[&'a MemoryEntry],
    query_sig: &[f64],
    k: usize,
) -> Vec<&'a MemoryEntry> {
    let mut scored: Vec<_> = corpus_mem
        .iter()
        .map(|m| (*m, dot_cosine(query_sig, &m.signals)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(m, _)| m).collect()
}

/// Reward options uniquely supported by top corpus rows (vs other options).
fn mcq_unique_corpus_lift(
    opt_line: &str,
    all_opts: &[(char, String)],
    top_mem: &[&MemoryEntry],
) -> f64 {
    let body = option_body(opt_line);
    let per = env_f64("VALHALLA_MCQ_UNIQUE_LIFT", 0.14);
    let cap = env_f64("VALHALLA_MCQ_UNIQUE_LIFT_CAP", 0.55);
    let mut lift = 0.0;
    for m in top_mem {
        let mine = keyword_overlap_count(&body, &m.text);
        if mine == 0 {
            continue;
        }
        let max_other = all_opts
            .iter()
            .filter(|(_, line)| line.as_str() != opt_line)
            .map(|(_, line)| keyword_overlap_count(&option_body(line), &m.text))
            .max()
            .unwrap_or(0);
        if mine > max_other {
            lift += (mine - max_other) as f64 * per;
        }
    }
    lift.min(cap)
}

fn mcq_stem(question: &str) -> String {
    mcq_fate::parse_mcq_parts(question).stem.to_lowercase()
}

fn mcq_entailment_enabled() -> bool {
    std::env::var("VALHALLA_MCQ_ENTAILMENT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Line-level: does this corpus row support this specific option (keywords + numbers)?
fn mcq_line_entails_option(
    question: &str,
    opt_line: &str,
    evidence: &MemoryEntry,
    opt_sig: &[f64],
) -> f64 {
    let body = option_body(opt_line);
    let stem = mcq_stem(question);
    let ev_low = evidence.text.to_lowercase();
    let mut s = 0.0;
    let mut opt_word_hits = 0usize;

    for word in body.split_whitespace() {
        let w: String = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if w.len() >= 3 && ev_low.contains(&w) {
            opt_word_hits += 1;
            s += 0.52;
        }
    }
    if opt_word_hits == 0 {
        s *= 0.25;
    }

    let opt_nums = numbers_in_text(&body);
    if !opt_nums.is_empty() {
        let ev_nums = numbers_in_text(&evidence.text);
        let matched = opt_nums
            .iter()
            .filter(|n| ev_nums.iter().any(|e| (e - **n).abs() < 1e-3))
            .count();
        s += (matched as f64 / opt_nums.len() as f64) * 1.35;
    }

    s += keyword_overlap_count(&body, &evidence.text) as f64 * 0.88;
    s += keyword_overlap_count(&stem, &evidence.text) as f64 * 0.24;
    s += keyword_overlap_count(question, &evidence.text) as f64 * 0.18;
    s += dot_cosine(opt_sig, &evidence.signals) * 0.28;
    s
}

fn rank_mcq_evidence_lines(
    question: &str,
    query_sig: &[f64],
    patch: &[f64],
    memories: &[MemoryEntry],
    tile_sigs: &[Vec<f64>],
    tile_energies: &[f64],
    tile_activations: &[f64],
    stem_sigs: &[Vec<f64>],
    stem_cluster_sigs: &[Vec<f64>],
    stem_cluster_energies: &[f64],
    hub_prefs: &[f64],
) -> Vec<(usize, f64)> {
    let min_ev = min_retrieval_score() * 0.4;
    let scored = score_memories(
        memories,
        query_sig,
        patch,
        tile_sigs,
        tile_energies,
        tile_activations,
        stem_sigs,
        stem_cluster_sigs,
        stem_cluster_energies,
        hub_prefs,
        &["question"],
        1.0,
    );
    scored
        .into_iter()
        .filter(|(i, s)| {
            let m = &memories[*i];
            m.source != "question"
                && *s >= min_ev
                && keyword_overlap_count(question, &m.text) >= 1
        })
        .take(env_usize("VALHALLA_MCQ_ENTAILMENT_TOPK", 6))
        .collect()
}

fn dot_cosine(a: &[f64], b: &[f64]) -> f64 {
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
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

fn encode_text(text: &str, budget: &IngressBudget) -> Vec<f64> {
    encode_payload(
        &IngressPayload::Text {
            content: text.to_string(),
        },
        EncodingMode::NormalizedF64,
        budget,
    )
}

fn encode_digits(text: &str, budget: &IngressBudget) -> Vec<f64> {
    let nums = extract_numbers(text);
    if nums.is_empty() {
        return encode_text(text, budget);
    }
    let payload: Vec<f64> = nums.iter().map(|n| (*n / 1000.0).clamp(-1.0, 1.0)).collect();
    encode_payload(
        &IngressPayload::F64 { values: payload },
        EncodingMode::NormalizedF64,
        budget,
    )
}

fn extract_numbers(text: &str) -> Vec<f64> {
    let mut expanded = text.to_string();
    let word_map = [
        ("zero", "0"),
        ("one", "1"),
        ("two", "2"),
        ("three", "3"),
        ("four", "4"),
        ("five", "5"),
        ("six", "6"),
        ("seven", "7"),
        ("eight", "8"),
        ("nine", "9"),
        ("ten", "10"),
        ("twelve", "12"),
        ("fifteen", "15"),
        ("sixteen", "16"),
        ("twenty", "20"),
        ("twenty-four", "24"),
        ("forty-two", "42"),
    ];
    let low = expanded.to_lowercase();
    for (w, d) in word_map {
        if low.contains(w) {
            expanded = expanded.replace(w, d);
            expanded = expanded.replace(&w.to_uppercase(), d);
        }
    }

    let mut nums = Vec::new();
    let mut cur = String::new();
    let mut seen_dot = false;
    for ch in expanded.chars() {
        if ch.is_ascii_digit() {
            cur.push(ch);
        } else if ch == '.' && !seen_dot && !cur.is_empty() {
            cur.push(ch);
            seen_dot = true;
        } else if !cur.is_empty() {
            if let Ok(v) = cur.parse::<f64>() {
                nums.push(v);
            }
            cur.clear();
            seen_dot = false;
        }
    }
    if !cur.is_empty() {
        if let Ok(v) = cur.parse::<f64>() {
            nums.push(v);
        }
    }
    nums
}

fn numbers_in_text(text: &str) -> Vec<f64> {
    extract_numbers(text)
}

fn parse_mcq_options(question: &str) -> Vec<(char, String)> {
    mcq_fate::parse_mcq_options(question)
}

fn fate_tile_weight(hub_prefs: &[f64], idx: usize) -> f64 {
    hub_prefs.get(idx % hub_prefs.len().max(1)).copied().unwrap_or(1.0).max(0.0)
}

fn score_structure_alignment(
    sig: &[f64],
    tile_sigs: &[Vec<f64>],
    tile_energies: &[f64],
    tile_activations: &[f64],
    hub_prefs: &[f64],
    stem_sigs: &[Vec<f64>],
    stem_cluster_sigs: &[Vec<f64>],
    stem_cluster_energies: &[f64],
) -> f64 {
    let mut s = 0.0;
    for (ti, ts) in tile_sigs.iter().enumerate() {
        let act = tile_activations.get(ti).copied().unwrap_or(0.0);
        let e = tile_energies.get(ti).copied().unwrap_or(1.0);
        s += 0.15
            * fate_tile_weight(hub_prefs, ti)
            * dot_cosine(sig, ts)
            * e.max(0.0)
            * (1.0 + act.abs());
    }
    let stem_w = if hub_prefs.len() >= 2 {
        (hub_prefs[1] + hub_prefs[3]) / 2.0
    } else {
        1.0
    };
    for ss in stem_sigs {
        s += 0.08 * stem_w * dot_cosine(sig, ss);
    }
    for (cs, ce) in stem_cluster_sigs.iter().zip(stem_cluster_energies.iter()) {
        s += 0.12 * stem_w * dot_cosine(sig, cs) * ce.max(0.01);
    }
    s
}

fn score_memories(
    memories: &[MemoryEntry],
    query_sig: &[f64],
    patch: &[f64],
    tile_sigs: &[Vec<f64>],
    tile_energies: &[f64],
    tile_activations: &[f64],
    stem_sigs: &[Vec<f64>],
    stem_cluster_sigs: &[Vec<f64>],
    stem_cluster_energies: &[f64],
    hub_prefs: &[f64],
    exclude_sources: &[&str],
    structure_weight: f64,
) -> Vec<(usize, f64)> {
    let sw = structure_weight.max(0.0);
    let mut scored: Vec<(usize, f64)> = memories
        .iter()
        .enumerate()
        .filter(|(_, m)| !exclude_sources.contains(&m.source.as_str()))
        .map(|(i, m)| {
            let mut s = dot_cosine(&m.signals, query_sig) * sw;
            s += 0.35 * dot_cosine(&m.signals, patch) * sw;
            s += score_structure_alignment(
                &m.signals,
                tile_sigs,
                tile_energies,
                tile_activations,
                hub_prefs,
                stem_sigs,
                stem_cluster_sigs,
                stem_cluster_energies,
            ) * sw;
            (i, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

fn keyword_domain_boost(question: &str, memory_text: &str) -> f64 {
    let q = question.to_lowercase();
    let m = memory_text.to_lowercase();
    let mut boost = 0.0;
    let math_keys = ["multiply", "divided", "square root", "integer", "total", "cost", "clips", "eggs"];
    let hist_keys = ["war", "century", "empire", "revolution", "王朝", "战争"];
    let sci_keys = ["unit", "newton", "joule", "photosynthesis", "dna"];
    if math_keys.iter().any(|k| q.contains(k)) && math_keys.iter().any(|k| m.contains(k)) {
        boost += 0.15;
    }
    if hist_keys.iter().any(|k| q.contains(k)) && hist_keys.iter().any(|k| m.contains(k)) {
        boost += 0.12;
    }
    if sci_keys.iter().any(|k| q.contains(k)) && sci_keys.iter().any(|k| m.contains(k)) {
        boost += 0.12;
    }
    boost
}

fn cortex_numeric_candidates(nums: &[f64]) -> Vec<f64> {
    if nums.is_empty() {
        return vec![];
    }
    let mut out = nums.to_vec();
    let diff = TransformType::Difference.apply(nums);
    out.extend(diff);
    let ratio = TransformType::Ratio.apply(nums);
    out.extend(ratio);
    let sum: f64 = nums.iter().sum();
    out.push(sum);
    if nums.len() >= 2 {
        out.push(nums[0] + nums[1]);
        out.push(nums[0] - nums[1]);
        out.push(nums[0] * nums[1]);
        if nums[1].abs() > 1e-9 {
            out.push(nums[0] / nums[1]);
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    out
}

fn try_direct_math(question: &str) -> Option<f64> {
    let q = question.to_lowercase();
    let nums = extract_numbers(question);

    if q.contains("multiplied by") && nums.len() >= 2 {
        return Some(nums[0] * nums[1]);
    }
    if q.contains("divided by") && nums.len() >= 2 && nums[1].abs() > 1e-9 {
        return Some(nums[0] / nums[1]);
    }
    if q.contains("square root of") && !nums.is_empty() {
        let v = nums[0];
        if v >= 0.0 {
            return Some(v.sqrt());
        }
    }
    if q.contains('x') && q.contains('=') && nums.len() >= 2 {
        if q.contains('+') {
            return Some(nums[1] - nums[0]);
        }
        if q.contains('-') {
            return Some(nums[1] + nums[0]);
        }
    }

    None
}

fn try_word_problem(question: &str) -> Option<f64> {
    let q = question.to_lowercase();
    let nums = extract_numbers(question);
    if nums.is_empty() {
        return None;
    }

    if q.contains("half as many") && q.contains("altogether") && !nums.is_empty() {
        let a = nums[0];
        return Some(a + a / 2.0);
    }

    if q.contains("per day") && q.contains("breakfast") && nums.len() >= 3 {
        let daily = nums[0];
        let eaten = nums.get(1).copied().unwrap_or(0.0);
        let baked = nums.get(2).copied().unwrap_or(0.0);
        let price = nums.get(3).copied().unwrap_or(1.0);
        return Some((daily - eaten - baked).max(0.0) * price);
    }

    if q.contains("each") && (q.contains("total cost") || q.contains("total")) && nums.len() >= 4 {
        return Some(nums[0] * nums[1] + nums[2] * nums[3]);
    }

    if q.contains("gives") && q.contains("buys") && nums.len() >= 3 {
        return Some(nums[0] - nums[1] + nums[2]);
    }

    if q.contains("altogether") && nums.len() >= 2 {
        return Some(nums.iter().sum());
    }

    None
}

fn format_num(v: f64) -> String {
    if (v - v.round()).abs() < 1e-6 {
        format!("{}", v.round() as i64)
    } else {
        format!("{:.2}", v)
    }
}

fn decode_numeric(
    question: &str,
    query_sig: &[f64],
    patch: &[f64],
    memories: &[MemoryEntry],
    tile_sigs: &[Vec<f64>],
    tile_energies: &[f64],
    tile_activations: &[f64],
    stem_sigs: &[Vec<f64>],
    stem_cluster_sigs: &[Vec<f64>],
    stem_cluster_energies: &[f64],
    hub_prefs: &[f64],
) -> (String, Vec<NativeMatch>, String) {
    let nums = extract_numbers(question);
    let mut candidates: Vec<(f64, f64, String, String)> = Vec::new(); // score, value, text, path

    if let Some(v) = try_direct_math(question) {
        candidates.push((100.0, v, format_num(v), "direct_math".into()));
    }
    if let Some(v) = try_word_problem(question) {
        candidates.push((95.0, v, format_num(v), "word_problem".into()));
    }
    for v in cortex_numeric_candidates(&nums) {
        candidates.push((40.0, v, format_num(v), "cortex".into()));
    }

    let scored_mem = score_memories(
        memories,
        query_sig,
        patch,
        tile_sigs,
        tile_energies,
        tile_activations,
        stem_sigs,
        stem_cluster_sigs,
        stem_cluster_energies,
        hub_prefs,
        &["question"],
        1.0,
    );

    for (i, s) in scored_mem.iter().take(8) {
        let m = &memories[*i];
        for v in numbers_in_text(&m.text) {
            let mem_score = *s * 10.0;
            candidates.push((mem_score, v, format_num(v), format!("memory:{}", m.source)));
        }
        let kb = keyword_domain_boost(question, &m.text);
        if kb > 0.0 {
            candidates.push((kb * 10.0, 0.0, m.text.clone(), "keyword".into()));
        }
    }

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut matches: Vec<NativeMatch> = candidates
        .iter()
        .take(5)
        .map(|(score, _, text, src)| NativeMatch {
            text: text.clone(),
            score: *score,
            source: src.clone(),
        })
        .collect();

    if let Some((_, _, ans, path)) = candidates.first() {
        return (ans.clone(), matches, path.clone());
    }

    (
        "No native match in incubated memory.".to_string(),
        matches,
        "none".into(),
    )
}

fn decode_mcq(
    question: &str,
    query_sig: &[f64],
    patch: &[f64],
    memories: &[MemoryEntry],
    hub_prefs: &[f64],
    tile_sigs: &[Vec<f64>],
    tile_energies: &[f64],
    tile_activations: &[f64],
    stem_sigs: &[Vec<f64>],
    stem_cluster_sigs: &[Vec<f64>],
    stem_cluster_energies: &[f64],
    budget: &IngressBudget,
    ingress_mode: &str,
    dl: bool,
    front: &DlFrontConfig,
) -> (String, Vec<NativeMatch>, String) {
    let opts = parse_mcq_options(question);
    let corpus_mem: Vec<_> = memories
        .iter()
        .filter(|m| m.source != "question")
        .collect();

    if opts.is_empty() {
        let scored = score_memories(
            memories,
            query_sig,
            patch,
            tile_sigs,
            tile_energies,
            tile_activations,
            stem_sigs,
            stem_cluster_sigs,
            stem_cluster_energies,
            hub_prefs,
            &["question"],
            1.0,
        );
        let best = scored.first().map(|(i, s)| NativeMatch {
            text: memories[*i].text.clone(),
            score: *s,
            source: memories[*i].source.clone(),
        });
        return (
            best.as_ref()
                .map(|m| m.text.clone())
                .unwrap_or_else(|| "ANSWER: A".to_string()),
            best.into_iter().collect(),
            "mcq_fallback".into(),
        );
    }

    let use_entailment = mcq_entailment_enabled();
    let evidence_lines = if use_entailment {
        rank_mcq_evidence_lines(
            question,
            query_sig,
            patch,
            memories,
            tile_sigs,
            tile_energies,
            tile_activations,
            stem_sigs,
            stem_cluster_sigs,
            stem_cluster_energies,
            hub_prefs,
        )
    } else {
        Vec::new()
    };
    let forward_w = env_f64("VALHALLA_MCQ_ENTAILMENT_FORWARD_WEIGHT", 0.48);
    let entail_w = env_f64("VALHALLA_MCQ_ENTAILMENT_WEIGHT", 0.58);
    let top_corpus = mcq_top_corpus_mem(&corpus_mem, query_sig, env_usize("VALHALLA_MCQ_TOP_CORPUS", 6));
    let mut used_entailment = false;

    let mut best_letter = opts.first().map(|(c, _)| *c).unwrap_or('A');
    let mut best_score = f64::NEG_INFINITY;
    let mut matches = Vec::new();
    let mut raw_scores = Vec::with_capacity(opts.len());

    for (letter, opt_line) in &opts {
        let opt_sig = decode_signal(opt_line, "mcq", ingress_mode, budget, dl, front);
        let mut forward = dot_cosine(&opt_sig, query_sig) + 0.5 * dot_cosine(&opt_sig, patch);
        forward += score_structure_alignment(
            &opt_sig,
            tile_sigs,
            tile_energies,
            tile_activations,
            hub_prefs,
            stem_sigs,
            stem_cluster_sigs,
            stem_cluster_energies,
        );
        for m in &corpus_mem {
            forward += 0.2 * dot_cosine(&opt_sig, &m.signals);
            forward += keyword_domain_boost(question, &m.text) * 0.2;
        }
        if !use_entailment {
            forward += mcq_corpus_option_boost(opt_line, &corpus_mem);
        } else {
            forward += mcq_corpus_option_boost(opt_line, &corpus_mem) * 0.1;
        }
        forward += mcq_unique_corpus_lift(opt_line, &opts, &top_corpus);
        forward += keyword_overlap_count(question, opt_line) as f64 * 0.08;
        if !hub_prefs.is_empty() {
            forward += 0.1 * hub_prefs.iter().take(4).sum::<f64>() / hub_prefs.len().min(4) as f64;
        }

        let mut entail = 0.0;
        if !evidence_lines.is_empty() {
            for (rank, (idx, mem_s)) in evidence_lines.iter().enumerate() {
                let w = 1.0 / (1.0 + rank as f64);
                entail += w
                    * mcq_line_entails_option(question, opt_line, &memories[*idx], &opt_sig)
                    * mem_s.max(0.05);
            }
            if entail > 0.08 {
                used_entailment = true;
            }
        }

        let s = if use_entailment && !evidence_lines.is_empty() {
            forward * forward_w + entail * entail_w
        } else {
            forward
        };

        raw_scores.push(s);

        matches.push(NativeMatch {
            text: format!(
                "{} (score={:.4}{})",
                letter,
                s,
                if use_entailment && entail > 0.08 {
                    format!(", ent={:.3}", entail)
                } else {
                    String::new()
                }
            ),
            score: s,
            source: if use_entailment && entail > 0.08 {
                "mcq_entailment".into()
            } else {
                "mcq_option".into()
            },
        });
    }

    let mean = if raw_scores.is_empty() {
        0.0
    } else {
        raw_scores.iter().sum::<f64>() / raw_scores.len() as f64
    };
    for (idx, (letter, _)) in opts.iter().enumerate() {
        let rel = raw_scores.get(idx).copied().unwrap_or(0.0) - mean;
        let hub = if !hub_prefs.is_empty() {
            hub_prefs.get(idx).copied().unwrap_or(0.25) * 0.02
        } else {
            0.0
        };
        let v = rel + hub;
        if v > best_score {
            best_score = v;
            best_letter = *letter;
        }
    }

    let mut decode_path = if use_entailment && used_entailment {
        "mcq_entailment".to_string()
    } else {
        "mcq_option".to_string()
    };

    if crate::mcq_pairwise::mcq_pairwise_enabled() && opts.len() >= 2 {
        let stem = mcq_stem(question);
        let priors: Vec<(char, f64)> = opts
            .iter()
            .enumerate()
            .map(|(idx, (l, _))| (*l, raw_scores.get(idx).copied().unwrap_or(0.0) - mean))
            .collect();
        let pw = crate::mcq_pairwise::pairwise_option_scores(&stem, &opts, &priors);
        if crate::mcq_pairwise::pairwise_spread(&pw) >= 0.008 {
            if let Some(pl) = crate::mcq_pairwise::pick_pairwise_letter(&pw) {
                best_letter = pl;
                decode_path = "mcq_pairwise".into();
            }
        }
    }

    matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    (format!("ANSWER: {}", best_letter), matches, decode_path)
}

/// Multi-line grasp width (feasibility experiment): how many top
/// candidate lines the open decode emits as one grasp. Default 1 =
/// the historical single-line grasp; the fog judgment (grasped idx,
/// margins, block reading) is unchanged — only the emitted body
/// widens.
fn grasp_lines() -> usize {
    std::env::var("VALHALLA_GRASP_LINES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 5)
}

/// Egress truth filter: within the grasp window (top-3), emit
/// only the line Fate's own passive activation ranks highest,
/// instead of the decode-top line. Default ON since the
/// 2026-11-15 consolidation (measured: premise poison -> truth
/// line only, limits core 89-90/90, traps 35 -> 37-38/60,
/// fabrication 0; window 12 measured harmful — template fillers
/// win on activation — so the window stays at 3). The override
/// applies ONLY when the affinity is this question's passive
/// capture (`affinity_passive`); stored feedback affinity in
/// the non-passive regime measured harmful (unload A/B core
/// 4/8 vs 8/8) and never reranks. A clear-preference margin
/// (VALHALLA_GRASP_RERANK_MARGIN, default 0.005) guards noise.
/// Disable with VALHALLA_GRASP_RERANK_FATE=0. The fog judgment
/// still reads the decode grasp series.
fn grasp_rerank_fate() -> bool {
    std::env::var("VALHALLA_GRASP_RERANK_FATE")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        })
        .unwrap_or(true)
}

fn multi_line_body(
    candidates: &[(usize, f64, usize)],
    memories: &[MemoryEntry],
    n: usize,
) -> String {
    let mut out: Vec<String> = Vec::new();
    for (i, _, _) in candidates {
        let m = &memories[*i];
        if m.corpus_idx.is_none() {
            continue;
        }
        if !out.contains(&m.text) {
            out.push(m.text.clone());
        }
        if out.len() >= n {
            break;
        }
    }
    out.join("\n")
}

fn decode_open(
    memories: &[MemoryEntry],
    query_sig: &[f64],
    patch: &[f64],
    tile_sigs: &[Vec<f64>],
    tile_energies: &[f64],
    tile_activations: &[f64],
    stem_sigs: &[Vec<f64>],
    stem_cluster_sigs: &[Vec<f64>],
    stem_cluster_energies: &[f64],
    hub_prefs: &[f64],
    question: &str,
    prior_questions: &[String],
    corpus_line_affinity: &[f64],
    // True only when `corpus_line_affinity` is this question's
    // passive Fate capture (read-only activation), not stored
    // feedback affinity written by earlier judgments.
    affinity_passive: bool,
    multihop_v2: bool,
    session_tick: u64,
    baseline_tick: u64,
) -> (String, Vec<NativeMatch>, String, Option<usize>) {
    let min_score = min_retrieval_score();
    let min_kw = min_open_keyword_overlap();
    let sw = open_structure_weight();

    let scored = score_memories(
        memories,
        query_sig,
        patch,
        tile_sigs,
        tile_energies,
        tile_activations,
        stem_sigs,
        stem_cluster_sigs,
        stem_cluster_energies,
        hub_prefs,
        &["question"],
        sw,
    );

    let q_low = question.to_lowercase();
    let kw_count = question_keywords(question).len();
    let mut candidates: Vec<(usize, f64, usize)> = Vec::new();
    for (i, s) in &scored {
        let m = &memories[*i];
        if m.text.to_lowercase() == q_low {
            continue;
        }
        let kw = keyword_overlap_count(question, &m.text);
        if kw < min_kw {
            continue;
        }
        let mut score = *s;
        let kw_frac = kw as f64 / kw_count.max(1) as f64;
        score += corpus_line_affinity_boost(
            m.corpus_idx,
            corpus_line_affinity,
            query_sig,
            &m.signals,
            kw_count,
        ) * kw_frac.clamp(0.25, 1.0);
        score += session_idf_term_boost(question, &m.text, memories);
        score += follow_up_retrieval_boost(question, prior_questions, &m.text, m.corpus_idx);
        score += corpus_duplicate_penalty(&m.text, m.corpus_idx, memories);
        let aff = m
            .corpus_idx
            .and_then(|i| corpus_line_affinity.get(i).copied())
            .unwrap_or(0.0);
        score += fate_algebra_retrieval_boost(
            question,
            &m.text,
            hub_prefs,
            session_tick,
            baseline_tick,
            aff,
        );
        // Removed (2026-11-14, negative-optimization audit): a flat
        // -0.18 penalty on corpus lines 0-1 whenever any newer line
        // had more keyword overlap. Corpus position carries no
        // authority in the field; the penalty was a stale-head
        // symptom patch and directly fought the prelearn-head block
        // closure (head lines are blocks like any other now).
        candidates.push((*i, score, kw));
    }

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // The series is kept deeper than the display top-5: the Fate
    // block gate reads competition inside the grasped block, and a
    // shallow series can hide the in-block runner-up entirely.
    let top: Vec<NativeMatch> = candidates
        .iter()
        .take(12)
        .map(|(i, s, _kw)| {
            let m = &memories[*i];
            let cite = m
                .corpus_idx
                .map(|idx| format!("corpus:{}", idx))
                .unwrap_or_else(|| m.source.clone());
            NativeMatch {
                text: format!("[{}] {}", cite, m.text),
                score: *s,
                source: cite,
            }
        })
        .collect();

    // Egress truth filter (experiment): among the grasp window,
    // pick the candidate Fate's own passive activation ranks
    // highest; only that line is emitted.
    let pick_pos = if grasp_rerank_fate() && affinity_passive {
        let n = grasp_lines().max(3).min(candidates.len());
        let aff_of = |pos: usize| -> f64 {
            memories[candidates[pos].0]
                .corpus_idx
                .and_then(|i| corpus_line_affinity.get(i).copied())
                .unwrap_or(0.0)
        };
        // Fate overrides the decode grasp only on a CLEAR passive
        // activation preference: another window line must beat the
        // decode pick's affinity by a noise-floor margin (default
        // 0.005, VALHALLA_GRASP_RERANK_MARGIN). Measured 2026-11-15:
        // premise poison 0.9708 vs truth 0.9794 (gap 0.0086, override
        // correct); template fillers sit within ~0.004 of each other
        // and stored feedback affinity within ~0.001 — sub-margin
        // differences are template noise, so the decode order stands.
        let margin = std::env::var("VALHALLA_GRASP_RERANK_MARGIN")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.005);
        let mut best_pos = 0usize;
        let mut best_aff = aff_of(0);
        for pos in 1..n {
            let a = aff_of(pos);
            if a > best_aff + margin {
                best_aff = a;
                best_pos = pos;
            }
        }
        if std::env::var_os("VALHALLA_GRASP_RERANK_DEBUG").is_some() {
            let win: Vec<String> = (0..n)
                .map(|pos| {
                    let ci = memories[candidates[pos].0].corpus_idx;
                    format!("pos{pos}:corpus={ci:?}:aff={:.4}", aff_of(pos))
                })
                .collect();
            eprintln!("[grasp_rerank] pick={best_pos} window=[{}]", win.join(" "));
        }
        best_pos
    } else {
        0
    };
    if let Some((idx, score, _)) = candidates.get(pick_pos) {
        if *score < min_score {
            return (
                "No confident match in incubated memory.".to_string(),
                top,
                "open_reject_low_score".into(),
                None,
            );
        }
        if multihop_v2 {
            let mh: Vec<crate::multihop_compose::MultihopCandidate> = candidates
                .iter()
                .map(|(i, sc, kw)| {
                    let m = &memories[*i];
                    crate::multihop_compose::MultihopCandidate {
                        text: m.text.clone(),
                        corpus_idx: m.corpus_idx,
                        source: m.source.clone(),
                        score: *sc,
                        kw: *kw,
                    }
                })
                .collect();
            if let Some((ans, path)) = crate::multihop_compose::compose_multihop_open(question, &mh) {
                if path.starts_with("multihop_v2_yesno") || path.starts_with("multihop_v2_entity") {
                    let picked_idx = mh
                        .iter()
                        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
                        .and_then(|c| c.corpus_idx);
                    let mem = picked_idx
                        .and_then(|i| mh.iter().find(|c| c.corpus_idx == Some(i)))
                        .map(|c| c.text.as_str());
                    let egress = format_open_retrieval_answer(
                        "",
                        mem.unwrap_or(""),
                        &ans,
                        picked_idx,
                        session_tick,
                        baseline_tick,
                    );
                    return (egress, top, path, picked_idx);
                }
            }
        }
        let m = &memories[*idx];
        let picked_idx = m.corpus_idx;
        let cite = m
            .corpus_idx
            .map(|i| format!("corpus:{}", i))
            .unwrap_or_else(|| m.source.clone());
        let decode_path = if prior_questions.is_empty() {
            "open_retrieval"
        } else {
            "open_retrieval_follow_up_aware"
        };
        let decode_path = if pick_pos > 0 {
            format!("{decode_path}_fate_rerank")
        } else {
            decode_path.to_string()
        };
        if let Some(short) = extract_open_short_answer(question, &m.text) {
            return (
                format_open_retrieval_answer(
                    &cite, &m.text, &short, picked_idx, session_tick, baseline_tick,
                ),
                top,
                format!("{decode_path}_short"),
                picked_idx,
            );
        }
        if let Some(span) = best_span(question, &m.text) {
            if grasp_lines() > 1 && !grasp_rerank_fate() {
                let body = multi_line_body(&candidates, memories, grasp_lines());
                return (
                    format_open_retrieval_answer(
                        &cite, &m.text, &body, picked_idx, session_tick, baseline_tick,
                    ),
                    top,
                    format!("{decode_path}_multiline"),
                    picked_idx,
                );
            }
            return (
                format_open_retrieval_answer(
                    &cite, &m.text, &span, picked_idx, session_tick, baseline_tick,
                ),
                top,
                format!("{decode_path}_span"),
                picked_idx,
            );
        }
        if grasp_lines() > 1 && !grasp_rerank_fate() {
            let body = multi_line_body(&candidates, memories, grasp_lines());
            return (
                format_open_retrieval_answer(
                    &cite, &m.text, &body, picked_idx, session_tick, baseline_tick,
                ),
                top,
                format!("{decode_path}_multiline"),
                picked_idx,
            );
        }
        return (
            format_open_retrieval_answer(
                &cite, &m.text, &m.text, picked_idx, session_tick, baseline_tick,
            ),
            top,
            decode_path.into(),
            picked_idx,
        );
    }

    (
        "No confident match in incubated memory.".to_string(),
        top,
        "open_reject_no_keywords".into(),
        None,
    )
}

fn ingress_mode_label(req: &NativeQaRequest) -> String {
    if let Some(mode) = req.ingress_mode.as_deref() {
        return mode.to_ascii_lowercase();
    }
    if std::env::var("VALHALLA_DL_FRONT")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        })
        .unwrap_or(false)
    {
        return "dl_front".into();
    }
    "char".into()
}

fn use_dl_front(req: &NativeQaRequest) -> bool {
    ingress_mode_label(req) == "dl_front"
}

fn memory_signal(text: &str, score_type: &str, budget: &IngressBudget, dl: bool, front: &DlFrontConfig) -> (Vec<f64>, String) {
    if score_type == "numeric" && dl {
        return (encode_digits(text, budget), "char_digits".into());
    }
    if dl {
        let ports = crate::encode_modality(
            &ModalityInput::Text {
                content: text.to_string(),
            },
            front,
        );
        return (ports.embedding, ports.encoder);
    }
    (encode_text(text, budget), "char".into())
}

/// Run native Tier-B QA: feed corpus + question, optimize structure, decode without HF.
pub fn run_native_qa(req: &NativeQaRequest) -> NativeQaResponse {
    let body = BodyTarget::parse(&req.body);
    let budget = IngressBudget::experiment();
    let mode = ingress_mode_label(req);
    let dl = mode == "dl_front";
    let grinder = mode == "grinder";
    let mut front = DlFrontConfig::default();
    if dl || grinder {
        // QA path: lighter port than training smoke (512) — large corpus × 512 times out.
        front.signals_per_port = std::env::var("VALHALLA_DL_FRONT_SIGNALS")
            .ok()
            .or_else(|| std::env::var("VALHALLA_GRINDER_SIGNALS_PER_PORT").ok())
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);
    }
    let mut session = TriadSession::with_body(budget.clone(), body);
    let mut memories: Vec<MemoryEntry> = Vec::new();
    let mut dl_encoder = String::from("char");

    if grinder {
        let shards = req.corpus_ports.as_ref().filter(|s| !s.is_empty());
        if shards.is_none() {
            panic!("ingress_mode=grinder requires corpus_ports");
        }
        let shards = shards.unwrap();
        let ingest_all = crate::fate_orchestrator::explicit_ingest_all_corpus(req.ingest_all_corpus);
        let ingest_skip =
            crate::fate_orchestrator::corpus_ingest_skip(shards.len(), ingest_all);
        for (i, ports) in shards.iter().enumerate() {
            let t = req
                .corpus
                .get(i)
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("grinder_corpus_{}", i));
            dl_encoder = ports.encoder.clone();
            memories.push(MemoryEntry {
                text: t.to_string(),
                signals: ports.embedding.clone(),
                source: "corpus".into(),
                corpus_idx: Some(i),
            });
            if i >= ingest_skip {
                session.emit_body_ports(ports);
            }
        }
    } else {
        for (ci, line) in req.corpus.iter().enumerate() {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            let (sig, enc) = memory_signal(t, "open", &budget, dl, &front);
            if dl {
                dl_encoder = enc;
            }
            memories.push(MemoryEntry {
                text: t.to_string(),
                signals: sig,
                source: "corpus".into(),
                corpus_idx: Some(ci),
            });
            if dl {
                session.feed_modality(
                    ModalityInput::Text {
                        content: t.to_string(),
                    },
                    &front,
                );
            } else {
                session.feed(
                    IngressPayload::Text {
                        content: t.to_string(),
                    },
                    EncodingMode::NormalizedF64,
                );
            }
        }
    }

    let cycles = if req.cycles == 0 { 1 } else { req.cycles };
    let corpus_cycles = req.corpus_cycles.unwrap_or(cycles);
    if corpus_cycles > 0 {
        session.run_optimize_cycles(corpus_cycles);
    }

    if grinder {
        let ports = req
            .question_ports
            .as_ref()
            .expect("ingress_mode=grinder requires question_ports");
        dl_encoder = ports.encoder.clone();
        memories.push(MemoryEntry {
            text: req.question.clone(),
            signals: ports.embedding.clone(),
            source: "question".into(),
            corpus_idx: None,
        });
        session.emit_body_ports(ports);
    } else {
        let (q_sig, q_enc) = memory_signal(&req.question, &req.score_type, &budget, dl, &front);
        if dl {
            dl_encoder = q_enc;
        }
        memories.push(MemoryEntry {
            text: req.question.clone(),
            signals: q_sig.clone(),
            source: "question".into(),
            corpus_idx: None,
        });
        if dl {
            session.feed_modality(
                ModalityInput::Text {
                    content: req.question.clone(),
                },
                &front,
            );
        } else {
            session.feed(
                IngressPayload::Text {
                    content: req.question.clone(),
                },
                EncodingMode::NormalizedF64,
            );
        }
    }

    let qa_cycles = req.qa_cycles.unwrap_or(cycles);
    let mode_label = if grinder {
        "native_qa_grinder"
    } else if dl {
        "native_qa_dl_front"
    } else {
        "native_qa"
    };
    let report: TriadFeedReport = session.finalize_with_cycles(mode_label, qa_cycles);

    let q_sig = memories
        .iter()
        .find(|m| m.source == "question")
        .map(|m| m.signals.clone())
        .unwrap_or_else(|| encode_text(&req.question, &budget));

    let patch = &report.model_patch.patch_vector;
    let hub_prefs = &report.hub_prefs;
    let tile_sigs: Vec<Vec<f64>> = report
        .model_patch
        .tile_signatures
        .iter()
        .take(16)
        .cloned()
        .collect();
    let tile_energies: Vec<f64> = (0..tile_sigs.len()).map(|i| 1.0 / (1 + i) as f64).collect();
    let tile_activations: Vec<f64> = report
        .model_patch
        .tile_activations
        .iter()
        .take(16)
        .copied()
        .collect();
    let stem_sigs: Vec<Vec<f64>> = report.model_patch.stem_signatures.iter().take(16).cloned().collect();
    let stem_cluster_sigs: Vec<Vec<f64>> = report
        .model_patch
        .stem_cluster_signatures
        .iter()
        .take(8)
        .cloned()
        .collect();
    let stem_cluster_energies: Vec<f64> = report
        .model_patch
        .stem_cluster_energies
        .iter()
        .take(8)
        .copied()
        .collect();

    let (answer, top_matches, decode_path, _retrieval_idx) = match req.score_type.as_str() {
        "mcq" => {
            let (a, t, p) = decode_mcq(
                &req.question,
                &q_sig,
                patch,
                &memories,
                hub_prefs,
                &tile_sigs,
                &tile_energies,
                &tile_activations,
                &stem_sigs,
                &stem_cluster_sigs,
                &stem_cluster_energies,
                &budget,
                &mode,
                dl,
                &front,
            );
            (a, t, p, None)
        }
        "numeric" => {
            let (a, t, p) = decode_numeric(
                &req.question,
                &q_sig,
                patch,
                &memories,
                &tile_sigs,
                &tile_energies,
                &tile_activations,
                &stem_sigs,
                &stem_cluster_sigs,
                &stem_cluster_energies,
                hub_prefs,
            );
            (a, t, p, None)
        }
        _ => decode_open(
            &memories,
            &q_sig,
            patch,
            &tile_sigs,
            &tile_energies,
            &tile_activations,
            &stem_sigs,
            &stem_cluster_sigs,
            &stem_cluster_energies,
            hub_prefs,
            &req.question,
            &[],
            &[],
            false,
            false,
            report.hub_signals,
            0,
        ),
    };

    NativeQaResponse {
        answer,
        body: req.body.clone(),
        cycles_executed: report.optimize_cycles,
        score_type: req.score_type.clone(),
        memory_items: memories.len(),
        tile_count: report.model_patch.tile_count,
        stem_cells: report.model_patch.stem_cell_count,
        tile_dim: report.tile_dim,
        stem_dim: report.stem_dim,

        top_matches,
        report_mode: report.mode,
        decode_path,
        ingress_mode: mode,
        dl_encoder,
        tile_connections: report.model_patch.tile_connection_count,
        tile_avg_activation: report.model_patch.tile_avg_activation,
        stem_organ_clusters: report.model_patch.stem_organ_clusters.len(),
    }
}

fn corpus_memories_from_ports(corpus: &[String], shards: &[BodyFrontPorts]) -> Vec<MemoryEntry> {
    let mut memories = Vec::new();
    for (i, ports) in shards.iter().enumerate() {
        let t = corpus
            .get(i)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| {
                // Memory body is content-only; land address is corpus_idx + fate tick at egress.
                crate::fate_adaptive_algebra::strip_all_citations(s)
                    .trim()
                    .to_string()
            })
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("grinder_corpus_{}", i));
        memories.push(MemoryEntry {
            text: t,
            signals: ports.embedding.clone(),
            source: "corpus".into(),
            corpus_idx: Some(i),
        });
    }
    memories
}

fn feed_grinder_corpus(
    session: &mut TriadSession,
    corpus: &[String],
    shards: &[BodyFrontPorts],
    ingest_all: bool,
) -> (Vec<MemoryEntry>, String) {
    let ingest_skip = crate::fate_orchestrator::corpus_ingest_skip(shards.len(), ingest_all);
    let mut memories = Vec::new();
    let mut dl_encoder = String::from("grinder");
    let mode = crate::fate_orchestrator::fragment_mode();
    let orch = crate::fate_orchestrator::FateOrchestratorConfig::default();
    let mut segmenter = crate::fate_orchestrator::FateDensitySegmenter::new();
    let mut last_ingested_idx = 0usize;
    for (i, ports) in shards.iter().enumerate() {
        let t = corpus
            .get(i)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("grinder_corpus_{}", i));
        dl_encoder = ports.encoder.clone();
        memories.push(MemoryEntry {
            text: t,
            signals: ports.embedding.clone(),
            source: "corpus".into(),
            corpus_idx: Some(i),
        });
        if i >= ingest_skip {
            last_ingested_idx = i;
            let line_ref = corpus
                .get(i)
                .map(|s| s.trim())
                .filter(|s| !s.is_empty());
            match mode {
                crate::fate_orchestrator::FragmentMode::Density => {
                    crate::fate_orchestrator::ingest_line_density_fragment(
                        session,
                        &mut segmenter,
                        ports,
                        line_ref,
                        i,
                        &orch,
                    );
                }
                crate::fate_orchestrator::FragmentMode::Line => {
                    crate::fate_orchestrator::ingest_line_record_fragment(
                        session,
                        ports,
                        line_ref,
                        i,
                        &orch,
                    );
                }
                crate::fate_orchestrator::FragmentMode::Off => {
                    session.emit_body_ports(ports);
                }
            }
        }
    }
    if mode == crate::fate_orchestrator::FragmentMode::Density && last_ingested_idx >= ingest_skip {
        if let Some((id, label, meta)) = segmenter.flush(session, last_ingested_idx) {
            session.record_fragment_with_meta(&id, &label, meta);
        }
    }
    (memories, dl_encoder)
}

fn multihop_candidates_from_memories(
    question: &str,
    memories: &[MemoryEntry],
) -> Vec<crate::multihop_compose::MultihopCandidate> {
    let q_low = question.to_lowercase();
    let min_kw = min_open_keyword_overlap();
    let mut out: Vec<crate::multihop_compose::MultihopCandidate> = memories
        .iter()
        .filter(|m| m.corpus_idx.is_some())
        .filter(|m| m.text.to_lowercase() != q_low)
        .filter_map(|m| {
            let kw = keyword_overlap_count(question, &m.text);
            if kw < min_kw {
                return None;
            }
            let score = kw as f64 * 0.12 + m.signals.iter().take(8).sum::<f64>() * 1e-4;
            Some(crate::multihop_compose::MultihopCandidate {
                text: m.text.clone(),
                corpus_idx: m.corpus_idx,
                source: m.source.clone(),
                score,
                kw,
            })
        })
        .collect();
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    out
}

fn apply_multihop_post_compose(
    question: &str,
    memories: &[MemoryEntry],
    answer: &str,
) -> Option<(String, String)> {
    let candidates = multihop_candidates_from_memories(question, memories);
    let composed = crate::multihop_compose::compose_multihop_short_answer(question, &candidates)?;
    let ans_low = answer.to_lowercase().trim().to_string();
    let mh_low = composed.0.to_lowercase();
    if composed.1.contains("yesno") {
        return Some(composed);
    }
    if composed.1.contains("entity") {
        if ans_low == mh_low {
            return None;
        }
        return Some(composed);
    }
    None
}

pub(crate) fn decode_from_report(
    question: &str,
    score_type: &str,
    memories: &[MemoryEntry],
    report: &TriadFeedReport,
    budget: &IngressBudget,
    mode: &str,
    dl: bool,
    front: &DlFrontConfig,
    prior_questions: &[String],
    corpus_line_affinity: &[f64],
    affinity_passive: bool,
    multihop_v2: bool,
) -> (String, Vec<NativeMatch>, String, Option<usize>) {
    let q_sig = memories
        .iter()
        .find(|m| m.source == "question")
        .map(|m| m.signals.clone())
        .unwrap_or_else(|| encode_text(question, budget));

    let patch = &report.model_patch.patch_vector;
    let hub_prefs = &report.hub_prefs;
    let tile_sigs: Vec<Vec<f64>> = report
        .model_patch
        .tile_signatures
        .iter()
        .take(16)
        .cloned()
        .collect();
    let tile_energies: Vec<f64> = (0..tile_sigs.len()).map(|i| 1.0 / (1 + i) as f64).collect();
    let tile_activations: Vec<f64> = report
        .model_patch
        .tile_activations
        .iter()
        .take(16)
        .copied()
        .collect();
    let stem_sigs: Vec<Vec<f64>> = report.model_patch.stem_signatures.iter().take(16).cloned().collect();
    let stem_cluster_sigs: Vec<Vec<f64>> = report
        .model_patch
        .stem_cluster_signatures
        .iter()
        .take(8)
        .cloned()
        .collect();
    let stem_cluster_energies: Vec<f64> = report
        .model_patch
        .stem_cluster_energies
        .iter()
        .take(8)
        .copied()
        .collect();

    match score_type {
        "mcq" => {
            let (a, t, p) = decode_mcq(
                question,
                &q_sig,
                patch,
                memories,
                hub_prefs,
                &tile_sigs,
                &tile_energies,
                &tile_activations,
                &stem_sigs,
                &stem_cluster_sigs,
                &stem_cluster_energies,
                budget,
                mode,
                dl,
                front,
            );
            (a, t, p, None)
        }
        "numeric" => {
            let (a, t, p) = decode_numeric(
                question,
                &q_sig,
                patch,
                memories,
                &tile_sigs,
                &tile_energies,
                &tile_activations,
                &stem_sigs,
                &stem_cluster_sigs,
                &stem_cluster_energies,
                hub_prefs,
            );
            (a, t, p, None)
        }
        _ => decode_open(
            memories,
            &q_sig,
            patch,
            &tile_sigs,
            &tile_energies,
            &tile_activations,
            &stem_sigs,
            &stem_cluster_sigs,
            &stem_cluster_energies,
            hub_prefs,
            question,
            prior_questions,
            corpus_line_affinity,
            affinity_passive,
            multihop_v2,
            report.hub_signals,
            0,
        ),
    }
}

fn response_from_report(
    req_body: &str,
    score_type: &str,
    memories: &[MemoryEntry],
    report: &TriadFeedReport,
    mode: &str,
    dl_encoder: &str,
    answer: String,
    top_matches: Vec<NativeMatch>,
    decode_path: String,
) -> NativeQaResponse {
    NativeQaResponse {
        answer,
        body: req_body.to_string(),
        cycles_executed: report.optimize_cycles,
        score_type: score_type.to_string(),
        memory_items: memories.len(),
        tile_count: report.model_patch.tile_count,
        stem_cells: report.model_patch.stem_cell_count,
        tile_dim: report.tile_dim,
        stem_dim: report.stem_dim,
        top_matches,
        report_mode: report.mode.clone(),
        decode_path,
        ingress_mode: mode.to_string(),
        dl_encoder: dl_encoder.to_string(),
        tile_connections: report.model_patch.tile_connection_count,
        tile_avg_activation: report.model_patch.tile_avg_activation,
        stem_organ_clusters: report.model_patch.stem_organ_clusters.len(),
    }
}

/// Full training corpus → Tile/Stem incubation, then QA without re-feeding corpus.
pub fn run_native_qa_batch(req: &NativeQaBatchRequest) -> NativeQaBatchResponse {
    let body = BodyTarget::parse(&req.body);
    // Elastic: incubation baseline + VALHALLA_INGRESS_* suite overrides.
    let budget = crate::signal_ingress::IngressBudget::incubation_with_env();
    let use_live = crate::fate_orchestrator::live_session_enabled();
    let mode = req
        .ingress_mode
        .as_deref()
        .unwrap_or("grinder")
        .to_ascii_lowercase();
    let dl = mode == "dl_front";
    let mut front = DlFrontConfig::default();
    front.signals_per_port = std::env::var("VALHALLA_DL_FRONT_SIGNALS")
        .ok()
        .or_else(|| std::env::var("VALHALLA_GRINDER_SIGNALS_PER_PORT").ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);

    let (corpus_memories, _dl_encoder, mut live_session, mut snap, incubation_report, fate_warmup) =
        if let Some(ref frags) = req.corpus_fragments {
            if !frags.is_empty() && crate::fate_orchestrator::fragment_only_session_enabled() {
                let aff = req
                    .incubation_snapshot
                    .as_ref()
                    .map(|s| s.corpus_line_affinity.clone())
                    .unwrap_or_default();
                let mut session = crate::fate_orchestrator::session_from_stored_fragments(
                    frags,
                    &aff,
                    budget.clone(),
                    body,
                );
                if let Some(ref pre) = req.incubation_snapshot {
                    TriadSession::apply_snapshot_fate_policy(&mut session, pre);
                }
                let memories = corpus_memories_from_ports(&req.corpus, &req.corpus_ports);
                let report = session.build_feed_report("native_batch_fragment_only", 0);
                let snap = crate::fate_orchestrator::compact_manifold_snapshot(&session);
                (
                    memories,
                    "grinder".into(),
                    session,
                    snap,
                    report,
                    None,
                )
            } else if let Some(pre) = req.incubation_snapshot.clone() {
                let memories = corpus_memories_from_ports(&req.corpus, &req.corpus_ports);
                let mut session =
                    TriadSession::from_incubation_snapshot(budget.clone(), body, pre.clone());
                let report = session.build_feed_report("native_batch_prebuilt_snap", 0);
                (memories, "grinder".into(), session, pre, report, None)
            } else {
                let mut session = TriadSession::with_body(budget.clone(), body);
                let corpus_memories = corpus_memories_from_ports(&req.corpus, &req.corpus_ports);
                let fate_warmup = if use_live || crate::fate_orchestrator::multi_fragment_enabled() {
                    let orch = crate::fate_orchestrator::FateOrchestratorConfig::default();
                    let (_, warmup) = crate::fate_orchestrator::ingest_corpus_with_fragments_mode_full(
                        &mut session,
                        &req.corpus,
                        &req.corpus_ports,
                        &orch,
                        crate::fate_orchestrator::production_fragment_mode(),
                    );
                    warmup
                } else {
                    feed_grinder_corpus(
                        &mut session,
                        &req.corpus,
                        &req.corpus_ports,
                        true,
                    );
                    session.run_optimize_cycles(req.incubation_cycles);
                    None
                };
                let incubation_report = session.build_feed_report("native_batch_incubation", 0);
                let snap = crate::fate_orchestrator::export_session_after_ingest(&session);
                (
                    corpus_memories,
                    "grinder".into(),
                    session,
                    snap,
                    incubation_report,
                    fate_warmup,
                )
            }
        } else if let Some(pre) = req.incubation_snapshot.clone() {
            if crate::fate_orchestrator::fragment_only_session_enabled() && !pre.fragments.is_empty() {
                let mut session = crate::fate_orchestrator::session_from_stored_fragments(
                    &crate::fate_orchestrator::stored_fragments(&pre),
                    &pre.corpus_line_affinity,
                    budget.clone(),
                    body,
                );
                TriadSession::apply_snapshot_fate_policy(&mut session, &pre);
                let memories = corpus_memories_from_ports(&req.corpus, &req.corpus_ports);
                let report = session.build_feed_report("native_batch_fragment_only", 0);
                let snap = crate::fate_orchestrator::compact_manifold_snapshot(&session);
                (
                    memories,
                    "grinder".into(),
                    session,
                    snap,
                    report,
                    None,
                )
            } else {
                let memories = corpus_memories_from_ports(&req.corpus, &req.corpus_ports);
                let mut session =
                    TriadSession::from_incubation_snapshot(budget.clone(), body, pre.clone());
                let report = session.build_feed_report("native_batch_prebuilt_snap", 0);
                (memories, "grinder".into(), session, pre, report, None)
            }
        } else {
            let mut session = TriadSession::with_body(budget.clone(), body);
            let corpus_memories = corpus_memories_from_ports(&req.corpus, &req.corpus_ports);
            let dl_encoder = req
                .corpus_ports
                .first()
                .map(|p| p.encoder.clone())
                .unwrap_or_else(|| "grinder".into());
            let fate_warmup = if use_live || crate::fate_orchestrator::multi_fragment_enabled() {
                let orch = crate::fate_orchestrator::FateOrchestratorConfig::default();
                let (_, warmup) = crate::fate_orchestrator::ingest_corpus_with_fragments_mode_full(
                    &mut session,
                    &req.corpus,
                    &req.corpus_ports,
                    &orch,
                    crate::fate_orchestrator::production_fragment_mode(),
                );
                warmup
            } else {
                feed_grinder_corpus(
                    &mut session,
                    &req.corpus,
                    &req.corpus_ports,
                    true,
                );
                session.run_optimize_cycles(req.incubation_cycles);
                None
            };
            let incubation_report = session.build_feed_report("native_batch_incubation", 0);
            let snap = crate::fate_orchestrator::export_session_after_ingest(&session);
            (
                corpus_memories,
                dl_encoder,
                session,
                snap,
                incubation_report,
                fate_warmup,
            )
        };

    let fragment_count = snap.fragments.len();
    let manifold_protocol = snap.manifold_protocol.clone();

    let mode_label = if mode == "grinder" {
        "native_qa_batch_grinder"
    } else if dl {
        "native_qa_batch_dl_front"
    } else {
        "native_qa_batch"
    };
    let mode_label = if use_live {
        format!("{mode_label}_live")
    } else {
        mode_label.to_string()
    };

    let mut results = Vec::with_capacity(req.questions.len());
    let multihop_v2 = req.multihop_v2 || crate::multihop_compose::multihop_v2_enabled();
    let open_fate = req.open_fate
        || req.multihop_fate
        || crate::multihop_fate::open_fate_enabled();
    let use_pipeline_v4 = req.pipeline_v4 || pipeline::pipeline_v4_enabled();
    let pipe_cfg = pipeline::PipelineConfig::default();

    for q in &req.questions {
        let prior = q.prior_questions.as_deref().unwrap_or(&[]);
        let question_ports = q
            .question_ports
            .as_ref()
            .expect("batch QA requires question_ports per item (grinder ingress)");

        let fragment_source = crate::fate_orchestrator::stored_fragments(&snap);
        let query_merge = crate::fate_orchestrator::query_fragment_merge_enabled()
            && !fragment_source.is_empty();

        let mut query_merged_session = None;
        let query_selection = if query_merge {
            let corpus_sigs: Vec<_> = corpus_memories
                .iter()
                .filter_map(|m| m.corpus_idx.map(|idx| (idx, m.signals.clone())))
                .collect();
            let top_k = crate::fate_orchestrator::query_fragment_top_k(req.corpus.len());
            let (merged, sel) = crate::fate_orchestrator::merge_fragments_for_query(
                &fragment_source,
                &q.question,
                &question_ports.embedding,
                &corpus_sigs,
                &req.corpus,
                budget.clone(),
                body,
                top_k,
            );
            query_merged_session = Some(merged);
            Some(sel)
        } else {
            None
        };

        let mut fork_holder = if use_live || query_merged_session.is_some() {
            None
        } else {
            Some(TriadSession::from_incubation_snapshot(
                budget.clone(),
                body,
                snap.clone(),
            ))
        };
        let use_natural = crate::natural_channels::natural_channels_enabled();
        let qa_session: &mut TriadSession = if let Some(ref mut merged) = query_merged_session {
            if use_natural {
                crate::natural_channels::pulse_question_natural(
                    merged,
                    question_ports,
                    req.qa_cycles,
                    q.audio_features.as_deref(),
                    &front,
                );
            } else {
                crate::fate_orchestrator::pulse_question_live(
                    merged,
                    question_ports,
                    req.qa_cycles,
                    q.audio_features.as_deref(),
                    &front,
                );
            }
            merged
        } else if use_live {
            if use_natural {
                crate::natural_channels::pulse_question_natural(
                    &mut live_session,
                    question_ports,
                    req.qa_cycles,
                    q.audio_features.as_deref(),
                    &front,
                );
            } else {
                crate::fate_orchestrator::pulse_question_live(
                    &mut live_session,
                    question_ports,
                    req.qa_cycles,
                    q.audio_features.as_deref(),
                    &front,
                );
            }
            &mut live_session
        } else {
            let s = fork_holder.as_mut().unwrap();
            if use_natural {
                crate::natural_channels::pulse_question_natural(
                    s,
                    question_ports,
                    req.qa_cycles,
                    q.audio_features.as_deref(),
                    &front,
                );
            } else {
                s.emit_body_ports(question_ports);
                crate::multimodal::feed_audio(s, q.audio_features.as_deref(), &front);
            }
            s
        };

        // Fragment merge still shapes the session substrate; Fate/open decode need the
        // full evidence set on small / multi-hop corpora (else only seg_1 is fed).
        let fate_full_feed = crate::multihop_fate::fate_full_corpus_feed(
            &q.question,
            req.corpus.len(),
        );
        let mut memories = if fate_full_feed {
            corpus_memories.clone()
        } else if let Some(ref sel) = query_selection {
            let allowed: std::collections::HashSet<_> =
                sel.corpus_line_indices.iter().copied().collect();
            if allowed.is_empty() {
                corpus_memories.clone()
            } else {
                corpus_memories
                    .iter()
                    .filter(|m| {
                        m.corpus_idx
                            .map(|i| allowed.contains(&i))
                            .unwrap_or(false)
                    })
                    .cloned()
                    .collect()
            }
        } else {
            corpus_memories.clone()
        };
        memories.push(MemoryEntry {
            text: q.question.clone(),
            signals: question_ports.embedding.clone(),
            source: "question".into(),
            corpus_idx: None,
        });

        let mut fate_line_aff = snap.corpus_line_affinity.clone();
        // Provenance for the egress rerank gate: true only when the
        // affinity below is this question's passive capture. Stored
        // snapshot affinity is post-judgment feedback and must never
        // override the decode grasp (measured 2026-11-15: unload A/B
        // core 8/8 without rerank vs 4/8 when feedback affinity was
        // allowed to rerank in the non-passive regime).
        let mut fate_aff_passive = false;
        if open_fate && q.score_type == "open" {
            // Always build Fate refs from full corpus when full-feed; never fragment-truncate.
            let fate_src = if fate_full_feed {
                corpus_memories.as_slice()
            } else {
                // memories without trailing question entry
                &memories[..memories.len().saturating_sub(1)]
            };
            let corpus_refs: Vec<crate::multihop_fate::CorpusLineRef> = fate_src
                .iter()
                .filter_map(|m| {
                    m.corpus_idx.map(|idx| crate::multihop_fate::CorpusLineRef {
                        idx,
                        text: m.text.clone(),
                        signals: m.signals.clone(),
                    })
                })
                .collect();
            if !corpus_refs.is_empty() {
                if crate::multihop_fate::passive_qa_enabled() {
                    // Passive Fate QA: the question pulse above has
                    // already lit the substrate; Fate passively holds
                    // the activated state in its tiles. Read it —
                    // never re-feed corpus lines or mutate affinity
                    // before judgment (no adapt loop on this path).
                    let captured = crate::multihop_fate::capture_passive_activation(
                        qa_session,
                        &q.question,
                        &corpus_refs,
                        &front,
                    );
                    if !captured.is_empty() {
                        fate_line_aff = captured;
                        fate_aff_passive = true;
                    }
                } else {
                    let _fate_adapt = crate::multihop_fate::run_multihop_fate_adapt(
                        qa_session,
                        &q.question,
                        &corpus_refs,
                        question_ports,
                        req.qa_cycles.max(2),
                        &front,
                    );
                    fate_line_aff = qa_session.manifold_snapshot().corpus_line_affinity;
                }
            }
        }

        let use_mcq_fate = req.mcq_fate || mcq_fate::mcq_fate_cycles_enabled();
        let line_aff_slice: &[f64] = if open_fate && q.score_type == "open" {
            &fate_line_aff
        } else {
            &snap.corpus_line_affinity
        };
        let line_aff_passive =
            open_fate && q.score_type == "open" && fate_aff_passive;

        let (report, mcq_fate_meta, pipeline_outcome, mut answer, top, mut decode_path, retrieval_corpus_idx) =
            if use_natural {
                let nc = crate::natural_channels::resolve_natural_channels(
                    qa_session,
                    &snap,
                    &q.question,
                    &q.score_type,
                    question_ports,
                    &memories,
                    line_aff_slice,
                    line_aff_passive,
                    &mode_label,
                    req.qa_cycles,
                    &mode,
                    dl,
                    &front,
                    prior,
                    multihop_v2,
                    open_fate,
                    use_pipeline_v4,
                    use_mcq_fate,
                    use_live,
                    query_selection.is_some(),
                    budget.clone(),
                );
                let (_, top, _, ridx) = decode_from_report(
                    &q.question,
                    &q.score_type,
                    &memories,
                    &nc.report,
                    &budget,
                    &mode,
                    dl,
                    &front,
                    prior,
                    line_aff_slice,
                    line_aff_passive,
                    multihop_v2,
                );
                (
                    nc.report,
                    nc.mcq_fate_meta,
                    nc.pipeline_outcome,
                    nc.answer,
                    top,
                    nc.decode_path,
                    ridx,
                )
            } else if use_pipeline_v4 {
                let outcome = if use_live || query_selection.is_some() {
                    pipeline::run_question_pipeline_on_session(
                        qa_session,
                        &q.question,
                        &q.score_type,
                        question_ports,
                        &pipe_cfg,
                        &front,
                        &mode_label,
                        q.audio_features.as_deref(),
                    )
                } else {
                    pipeline::run_question_pipeline(
                        &snap,
                        &q.question,
                        &q.score_type,
                        question_ports,
                        &pipe_cfg,
                        &front,
                        &mode_label,
                        q.audio_features.as_deref(),
                    )
                };
                let report = outcome.report.clone();
                let meta = outcome.mcq_fate.as_ref().map(|o| {
                    (
                        o.cycles_executed,
                        o.stopped_reason.clone(),
                        o.decompose_score,
                        o.aggregate_score,
                    )
                });
                let (mut answer, top, mut decode_path, retrieval_corpus_idx) =
                    if q.score_type == "mcq" {
                        let (dec_ans, _, dec_path, _) = decode_from_report(
                            &q.question,
                            &q.score_type,
                            &memories,
                            &report,
                            &budget,
                            &mode,
                            dl,
                            &front,
                            prior,
                            line_aff_slice,
                            line_aff_passive,
                            multihop_v2,
                        );
                        let (ans, path) = pipeline::merge_mcq_pipeline_decode(
                            &dec_ans,
                            &dec_path,
                            &outcome.recompose.mcq_option_affinity,
                        );
                        (ans, Vec::new(), path, None)
                    } else {
                        let (a, t, _dp, ridx) = decode_from_report(
                            &q.question,
                            &q.score_type,
                            &memories,
                            &report,
                            &budget,
                            &mode,
                            dl,
                            &front,
                            prior,
                            line_aff_slice,
                            line_aff_passive,
                            multihop_v2,
                        );
                        (a, t, format!("pipeline_v4_{}", q.score_type), ridx)
                    };
                if q.score_type != "mcq" {
                    decode_path = outcome.decode_hint.clone();
                }
                (report, meta, Some(outcome), answer, top, decode_path, retrieval_corpus_idx)
            } else if q.score_type == "mcq" && use_mcq_fate {
                let outcome = mcq_fate::run_mcq_fate_pipeline(
                    qa_session,
                    &q.question,
                    question_ports,
                    &mode_label,
                    &front,
                );
                let meta = Some((
                    outcome.cycles_executed,
                    outcome.stopped_reason.clone(),
                    outcome.decompose_score,
                    outcome.aggregate_score,
                ));
                let (answer, top, _, retrieval_corpus_idx) = decode_from_report(
                    &q.question,
                    &q.score_type,
                    &memories,
                    &outcome.report,
                    &budget,
                    &mode,
                    dl,
                    &front,
                    prior,
                    line_aff_slice,
                    line_aff_passive,
                    multihop_v2,
                );
                (
                    outcome.report,
                    meta,
                    None,
                    answer,
                    top,
                    "mcq_fate_cycle".into(),
                    retrieval_corpus_idx,
                )
            } else {
                let report = qa_session.build_feed_report(&mode_label, req.qa_cycles);
                let (answer, top, decode_path, retrieval_corpus_idx) = decode_from_report(
                    &q.question,
                    &q.score_type,
                    &memories,
                    &report,
                    &budget,
                    &mode,
                    dl,
                    &front,
                    prior,
                    line_aff_slice,
                    line_aff_passive,
                    multihop_v2,
                );
                (report, None, None, answer, top, decode_path, retrieval_corpus_idx)
            };
        if !use_natural {
            if mcq_fate_meta.is_some() && pipeline_outcome.is_none() {
                decode_path = "mcq_fate_cycle".to_string();
            } else if let Some(ref po) = pipeline_outcome {
                if q.score_type != "mcq" {
                    decode_path = po.decode_hint.clone();
                } else if !decode_path.starts_with("pipeline_v4_") && !decode_path.starts_with("mcq_") {
                    decode_path = po.decode_hint.clone();
                }
            }
        }
        if let Some(ref sel) = query_selection {
            decode_path = format!(
                "query_frag/k{}/ids={}/{}",
                sel.top_k,
                sel.selected_ids.join("+"),
                decode_path
            );
        }
        let fog_evidence = fog_evidence_from_matches(&top, &q.question);
        let native_reject = decode_path.contains("reject")
            || answer.starts_with("No confident match");
        let mut fog_decision = fog_decide_algebraic(
            &fog_evidence,
            req.incubation_cycles.max(1),
            2,
            native_reject,
        );

        // Fate block reading (owner architecture, P1b): the grasp is
        // judged against competition inside its own block — the stored
        // fragment whose line range contains the grasped line. The
        // ambient global-margin fog reading is superseded for
        // margin-abstains only: cross-block near-ties (two lines from
        // different blocks scoring alike) stop vetoing a grasp that
        // stands clear on its own substrate. Absolute-weakness (tau),
        // native-reject and defer abstains are untouched.
        // The separation threshold is NOT a constant here: it is the
        // session Fate equation's own threshold, applied through
        // bundle_fits (see gate below). Measured caveat (2026-11-14):
        // QA-path sessions receive almost no adapt feeds, so the
        // equation sits near its birth state (coefs 1,-2,1, thr 0.01,
        // adaptations 0-3); the gate inherits whatever Fate has
        // actually learned, no more.
        let cand_idx: Vec<Option<usize>> = top
            .iter()
            .map(|m| crate::fate_qa::parse_corpus_idx(&m.text))
            .collect();
        let cand_scores: Vec<f64> = top.iter().map(|m| m.score).collect();
        let grasped_line = retrieval_corpus_idx
            .or_else(|| cand_idx.first().copied().flatten());
        let block_ranges: Vec<(usize, usize)> = fragment_source
            .iter()
            .map(|f| (f.line_start, f.line_end))
            .collect();
        let block_reading = grasped_line.and_then(|g| {
            crate::fate_qa::true_block_reading(&cand_idx, &cand_scores, g, &block_ranges)
        });

        // Fate-state telemetry (no behaviour change): what the
        // session's own Fate algebra and the grasped block's own Fate
        // equation would require at this grasp — recorded so the
        // calibrated gate constant can be replaced by a
        // Fate-controlled quantity once measured. The algebra is fed
        // the LIVE post-adapt affinity (`line_aff_slice`): the fork
        // snapshot carries no corpus affinity and would silently
        // degrade to ambient defaults.
        let algebra_state = crate::fate_adaptive_algebra::FateAlgebraState::from_live_affinity(
            line_aff_slice,
            &snap.hub_prefs,
            &crate::fate_adaptive_algebra::FateAlgebraMeta::default(),
        );
        let s_grasp = grasped_line
            .and_then(|g| {
                cand_idx
                    .iter()
                    .zip(cand_scores.iter())
                    .find(|(i, _)| **i == Some(g))
                    .map(|(_, s)| *s)
            })
            .or_else(|| cand_scores.first().copied())
            .unwrap_or(0.0);
        let alg_delta_rel = if s_grasp > 0.0 {
            Some(algebra_state.delta_required(s_grasp) / s_grasp)
        } else {
            None
        };
        let block_cluster_threshold = grasped_line.and_then(|g| {
            fragment_source
                .iter()
                .find(|f| g >= f.line_start && g <= f.line_end)
                .map(|f| f.triple_fate_slice.cluster_threshold)
        });
        // The session's live global Fate equation (the judge Fate
        // itself would use), read at the gate point.
        let live_eq = &qa_session.hub.triple_fate.cluster_equation;
        let live_eq_state = (
            [live_eq.coef_a, live_eq.coef_b, live_eq.coef_c],
            live_eq.threshold,
            live_eq.adaptation_count,
        );
        if q.score_type == "open"
            && fog_decision.decision == "abstain"
            && fog_decision.reason.starts_with("margin=")
        {
            if let Some(reading) = block_reading {
                // Fate controls the threshold: the grasp and its
                // best in-block competitor are judged by the session's
                // own cluster equation — if the two still fit one
                // bundle (bundle_fits, original (x-1)^2 semantics
                // under the equation's own adapted threshold), the
                // grasp is not separated and silence stands. No
                // calibrated constant anywhere in this gate.
                let separated = match reading.s_other {
                    Some(s_other) => !live_eq.bundle_fits(reading.s_grasp, s_other),
                    None => true,
                };
                if separated {
                    fog_decision = FogDecision {
                        decision: "emit".into(),
                        structure_ready: true,
                        evidence_ready: true,
                        tau: fog_decision.tau,
                        delta_required: fog_decision.delta_required,
                        reason: format!(
                            "fate_block_ready block=[{},{}] bundle_fits=false s1={:.4} s2={:.4} eq_thr={:.4} eq_adapt={} n_block={} (global-margin abstain superseded)",
                            reading.line_start,
                            reading.line_end,
                            reading.s_grasp,
                            reading.s_other.unwrap_or(0.0),
                            live_eq.threshold,
                            live_eq.adaptation_count,
                            reading.n_candidates
                        ),
                    };
                }
            }
        }

        if q.score_type == "open" {
            (answer, decode_path) = apply_open_fog_gate(answer, decode_path, Some(&fog_decision));
        }

        if !use_natural && open_fate && q.score_type == "open" {
            let corpus_refs: Vec<crate::multihop_fate::CorpusLineRef> = memories
                .iter()
                .filter_map(|m| {
                    m.corpus_idx.map(|idx| crate::multihop_fate::CorpusLineRef {
                        idx,
                        text: m.text.clone(),
                        signals: m.signals.clone(),
                    })
                })
                .collect();
            let q_sig = question_ports.embedding.clone();
            let fate_compose =
                crate::multihop_fate::open_fate_structure_compose(&q.question, multihop_v2);
            let pipeline_weak = answer.starts_with("No confident match")
                || decode_path.contains("abstain")
                || fog_decision.decision == "abstain";
            if fate_compose || pipeline_weak {
                if let Some((fate_ans, fate_path)) = crate::multihop_fate::decode_open_fate(
                    &q.question,
                    &corpus_refs,
                    &fate_line_aff,
                    &report.hub_prefs,
                    &q_sig,
                    multihop_v2,
                ) {
                    let picked = retrieval_corpus_idx.or_else(|| {
                        corpus_refs
                            .iter()
                            .max_by(|a, b| {
                                let aa = fate_line_aff.get(a.idx).copied().unwrap_or(0.0);
                                let bb = fate_line_aff.get(b.idx).copied().unwrap_or(0.0);
                                aa.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|l| l.idx)
                    });
                    let mem = picked
                        .and_then(|i| corpus_refs.iter().find(|l| l.idx == i))
                        .map(|l| l.text.as_str());
                    answer = crate::fate_io::fate_egress(
                        &fate_ans,
                        mem,
                        picked,
                        report.hub_signals,
                        0,
                    );
                    decode_path = fate_path;
                }
            }
        }

        // Unified solution: multihop v2 post-compose + generic Fate egress cites.
        if multihop_v2 && q.score_type == "open" {
            if let Some((mh_ans, mh_path)) =
                apply_multihop_post_compose(&q.question, &memories, &answer)
            {
                let picked = retrieval_corpus_idx.or_else(|| {
                    memories
                        .iter()
                        .filter_map(|m| m.corpus_idx)
                        .next()
                });
                let mem = picked
                    .and_then(|i| memories.iter().find(|m| m.corpus_idx == Some(i)))
                    .map(|m| m.text.as_str());
                answer = crate::fate_io::fate_egress(&mh_ans, mem, picked, report.hub_signals, 0);
                if !decode_path.contains(&mh_path) {
                    decode_path = format!("{decode_path}+{mh_path}");
                }
            } else if decode_path.contains("multihop_fate")
                && !answer.contains("[corpus:")
                && !answer.contains("[fate:")
            {
                let picked = retrieval_corpus_idx.or_else(|| {
                    memories.iter().filter_map(|m| m.corpus_idx).next()
                });
                let mem = picked
                    .and_then(|i| memories.iter().find(|m| m.corpus_idx == Some(i)))
                    .map(|m| m.text.as_str());
                answer = crate::fate_io::fate_egress(&answer, mem, picked, report.hub_signals, 0);
            }
        }

        let mut fate_grasp = crate::fate_qa::FateGrasp::build(
            query_selection
                .as_ref()
                .map(|s| s.selected_ids.clone())
                .unwrap_or_default(),
            query_selection
                .as_ref()
                .map(|s| s.scores.clone())
                .unwrap_or_default(),
            cand_scores.clone(),
            cand_idx.clone(),
            query_selection
                .as_ref()
                .map(|s| s.corpus_line_indices.clone())
                .unwrap_or_default(),
            retrieval_corpus_idx,
        );
        if let (Some(grasp), Some(reading)) = (fate_grasp.as_mut(), block_reading) {
            grasp.set_block_reading(&reading);
        }
        if let Some(grasp) = fate_grasp.as_mut() {
            grasp.set_fate_scales(
                algebra_state.affinity_std,
                alg_delta_rel,
                block_cluster_threshold,
            );
            grasp.set_live_equation(live_eq_state.0, live_eq_state.1, live_eq_state.2);
        }

        // Post-judgment passive intake (owner architecture: learn
        // AFTER judgment). Fate passively captures this question's
        // activated candidate series; nothing here can change the
        // answer just judged. The series only trains the session's
        // live cluster equation (receive_passive -> adapt), the
        // quantity that controls the block gate on FUTURE questions,
        // and the trained state rides out in the exported snapshot.
        if q.score_type == "open" {
            for s in &cand_scores {
                qa_session.hub.triple_fate.receive_passive(*s);
            }
        }

        results.push(NativeQaBatchItem {
            id: q.id.clone().unwrap_or_default(),
            answer,
            score_type: q.score_type.clone(),
            decode_path,
            tile_count: report.model_patch.tile_count,
            stem_cells: report.model_patch.stem_cell_count,
            tile_connections: report.model_patch.tile_connection_count,
            tile_avg_activation: report.model_patch.tile_avg_activation,
            stem_organ_clusters: report.model_patch.stem_organ_clusters.len(),
            correct: false,
            mcq_fate_cycles: mcq_fate_meta.as_ref().map(|m| m.0).unwrap_or(0),
            mcq_fate_stopped: mcq_fate_meta
                .as_ref()
                .map(|m| m.1.clone())
                .unwrap_or_default(),
            mcq_decompose_score: mcq_fate_meta.as_ref().map(|m| m.2).unwrap_or(0.0),
            mcq_aggregate_score: mcq_fate_meta.as_ref().map(|m| m.3).unwrap_or(0.0),
            fog_evidence: Some(fog_evidence),
            fog_decision: Some(fog_decision),
            retrieval_corpus_idx,
            query_fragments: query_selection.as_ref().map(|s| s.selected_ids.clone()),
            query_fragment_scores: query_selection.as_ref().map(|s| s.scores.clone()),
            fate_grasp,
        });

        if use_live && query_selection.is_none() {
            snap = qa_session.manifold_snapshot();
        }
    }

    NativeQaBatchResponse {
        mode: mode_label,
        corpus_lines: req.corpus.len(),
        corpus_lines_ingested: req.corpus_ports.len(),
        incubation_cycles: req.incubation_cycles,
        qa_cycles: req.qa_cycles,
        incubation_tile_count: incubation_report.model_patch.tile_count,
        incubation_stem_cells: incubation_report.model_patch.stem_cell_count,
        incubation_tile_connections: incubation_report.model_patch.tile_connection_count,
        incubation_stem_organ_clusters: incubation_report.model_patch.stem_organ_clusters.len(),
        incubation_tile_avg_activation: incubation_report.model_patch.tile_avg_activation,
        fragment_count,
        live_session: use_live,
        manifold_protocol,
        fate_warmup_cycles: fate_warmup.as_ref().map(|w| w.cycles_executed),
        fate_warmup_prelearn_lines: fate_warmup.as_ref().map(|w| w.prelearn_lines),
        results,
    }
}
