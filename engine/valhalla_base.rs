//! Valhalla-as-base inference — context-flexible structure substrate + optional LM decode.
//!
//! v1 extensions: hybrid routing, persistent session, multimodal context ingress.

use serde::{Deserialize, Serialize};

use crate::dl_front::{encode_modality, BodyFrontPorts, DlFrontConfig, ModalityInput};
use crate::fair_score::{extract_mcq_letter, format_prompt_text, score_answer, FairPrompt, ScoreResult};
use crate::fate_orchestrator::{self, FateOrchestratorConfig};
use crate::free_multimodal_fate::{fuse_universal_core_on_snapshot, UniversalCoreMetrics};
use crate::grinder::{default_port_cap, grind_text_hash};
use crate::native_qa::{
    fog_tau_algebraic, run_native_qa, run_native_qa_batch, structural_open_on_topic, FogDecision,
    FogEvidence, NativeQaBatchRequest, NativeQaQuestionSpec, NativeQaRequest,
};
use crate::patch_lm::{PatchLmConfig, DEFAULT_MAX_NEW, DEFAULT_STRENGTH};
use crate::signal_ingress::{BodyTarget, IngressBudget, TriadIncubationSnapshot, TriadSession};
use crate::{FateWeightMode, IngressRoutingMode};

pub const VALHALLA_BASE_PROTOCOL: &str = "valhalla-base-v1";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ValhallaBaseDecode {
    Native,
    #[default]
    LmPatch,
    /// open → native retrieval; mcq/numeric → LM logprob/generate.
    Hybrid,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ValhallaMcqDecode {
    #[default]
    LmPatch,
    /// MCQ via native pipeline + mcq_fate (no LM logprob).
    StructureFate,
    /// Run native mcq_option + lm_patch logprob; deploy pick by letter agreement + margin.
    MaxPick,
}

/// How to decode open questions when session has prior turns.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ValhallaFollowUpDecode {
    /// Same routing as first question (native under Hybrid).
    #[default]
    Inherit,
    /// Follow-ups use LM patch on incubated snapshot (structure→LM).
    LmPatchOnFollowUp,
    /// Stay native but rank corpus with prior-question-aware retrieval.
    NativeFollowUpAware,
}

/// Fate quality assignment during ingress (high constraint → relaxed).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ValhallaFateWeightMode {
    #[default]
    Fixed,
    Evaluated,
}

/// Hub ingress routing while Fate routing is locked.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ValhallaFateIngressRouting {
    #[default]
    /// Within each corpus-line loop, rotate signals across all 4 quads (production default).
    QuadCycle,
    RoundRobin,
    /// Route corpus lines by semantic keywords → asymmetric quad load.
    Semantic,
    /// 50% quad-cycle + 50% semantic primary quad per corpus line.
    HybridQuadSemantic,
}

/// Open-answer evidence filter across non-isolated multimodal session lines.
///
/// Default **mixed**: all modality-tagged corpus lines can land (product claim).
/// Optional single-channel modes exist for ablation / user preference.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ValhallaAnswerMix {
    #[default]
    Mixed,
    /// Pure text items only (no audio/image tags on the line).
    Text,
    /// Lines that carried audio_features (often caption + audio).
    Audio,
    /// Lines that carried image_features.
    Image,
}

pub const MOD_TEXT: u8 = 1;
pub const MOD_AUDIO: u8 = 2;
pub const MOD_IMAGE: u8 = 4;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValhallaContextItem {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub audio_features: Option<Vec<f64>>,
    #[serde(default)]
    pub image_features: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValhallaBaseSessionState {
    pub corpus: Vec<String>,
    pub fingerprint: String,
    pub lines_ingested: usize,
    /// Dynamic density fragments — primary session memory when `fragment_only` is on.
    #[serde(default)]
    pub fragments: Vec<crate::fate_orchestrator::SnapshotFragment>,
    /// Legacy carrier; under fragment-only holds metadata + fragments only (no tile/stem blob).
    pub snapshot: TriadIncubationSnapshot,
    /// Fate adaptive algebra counters + rollback stack for long-run sessions.
    #[serde(default)]
    pub fate_algebra_meta: crate::fate_adaptive_algebra::FateAlgebraMeta,
    #[serde(default, skip_serializing)]
    pub fate_rollback_stack: Vec<crate::signal_ingress::TriadIncubationSnapshot>,
    /// Prior user questions in this session (for follow-up routing / retrieval).
    #[serde(default)]
    pub question_history: Vec<String>,
    /// Fate tick after context ingest — proactive gate counts signals since this baseline.
    #[serde(default)]
    pub fate_baseline_tick: u64,
    /// Interface→container address book (absolute IDs only at edges; Fate schedules inside).
    #[serde(default)]
    pub address_book: Option<crate::valhalla_address::AddressBook>,
    /// Parallel to `corpus`: modality bits (TEXT|AUDIO|IMAGE). Empty ⇒ treat as all-mixed.
    #[serde(default)]
    pub corpus_modality_mask: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValhallaBaseQuestion {
    pub id: String,
    pub text: String,
    #[serde(default = "default_open")]
    pub score_type: String,
    #[serde(default)]
    pub reference_answer: String,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub audio_features: Option<Vec<f64>>,
    #[serde(default)]
    pub image_features: Option<Vec<f64>>,
}

fn default_open() -> String {
    "open".into()
}

impl From<ValhallaBaseQuestion> for FairPrompt {
    fn from(q: ValhallaBaseQuestion) -> Self {
        FairPrompt {
            id: q.id,
            domain: String::new(),
            text: q.text,
            reference_answer: q.reference_answer,
            keywords: q.keywords,
            score_type: q.score_type,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValhallaBaseConfig {
    #[serde(default = "default_lm_model")]
    pub lm_model: String,
    #[serde(default = "default_triad")]
    pub patch_body: String,
    #[serde(default = "default_strength")]
    pub strength: f64,
    #[serde(default = "default_max_new")]
    pub max_new: usize,
    #[serde(default = "default_cycles")]
    pub incubation_cycles: usize,
    #[serde(default)]
    pub use_cpu: bool,
    #[serde(default)]
    pub decode: ValhallaBaseDecode,
    /// Fuse task text/TTS/image into context snapshot before decode (structure→LM).
    #[serde(default)]
    pub core_fusion: bool,
    /// Answer-time modality mix (default mixed — non-isolated Triad answering).
    #[serde(default)]
    pub answer_mix: ValhallaAnswerMix,
    /// When decode=Hybrid, how to answer MCQ (default: LM logprob).
    #[serde(default)]
    pub mcq_decode: ValhallaMcqDecode,
    /// Follow-up turn decode policy for open questions.
    #[serde(default)]
    pub follow_up_decode: ValhallaFollowUpDecode,
    /// Fate ingress weight mode (fixed = high constraint).
    #[serde(default)]
    pub fate_weight_mode: ValhallaFateWeightMode,
    /// When true, Fate preference routing is enabled; false = round-robin explore.
    #[serde(default)]
    pub fate_routing_free: bool,
    /// Ingress routing while `fate_routing_free` is false.
    #[serde(default)]
    pub fate_ingress_routing: ValhallaFateIngressRouting,
    /// When true, push QA correct/incorrect back into Fate after each answer.
    #[serde(default = "default_true")]
    pub fate_qa_feedback: bool,
    /// Enable proactive idle path (self-probe + FOG-gated nudge after Fate signal warmup).
    #[serde(default = "default_true")]
    pub proactive_enabled: bool,
    /// Minimum Fate signal triggers since context baseline before proactive idle may fire.
    #[serde(default = "default_proactive_min_fate_signals")]
    pub proactive_min_fate_signals: usize,
    /// Deprecated alias — mapped to fate signals if set and min_fate_signals unset in env.
    #[serde(default = "default_proactive_min_turns_legacy")]
    pub proactive_min_turns: usize,
    /// Minimum hub_prefs spread — default from Fate adaptive coupling scale.
    #[serde(default = "default_proactive_min_spread")]
    pub proactive_min_spread: f64,
    /// Legacy benchmark: rule-based multihop compose (opt-in `VALHALLA_MULTIHOP_V2`).
    #[serde(default)]
    pub multihop_v2: bool,
    /// Global open Fate — pre-QA adapt + structure decode (default on).
    #[serde(default = "default_open_fate_cfg")]
    pub open_fate: bool,
    /// When true, Fate orchestrator runs adapt cycles (default manifold runtime).
    #[serde(default = "default_true")]
    pub fate_orchestrate: bool,
    /// Prelearn + Fate warmup cycles before fragment ingest (production default on).
    #[serde(default = "default_true")]
    pub fate_warmup_enabled: bool,
    /// Fate warmup cycle budget (maps to `FateOrchestratorConfig::warmup_cycles`).
    #[serde(default = "default_fate_warmup_cycles")]
    pub fate_warmup_cycles: usize,
    /// Prelearn line count before warmup; 0 = auto (corpus/3 clamped).
    #[serde(default)]
    pub fate_warmup_prelearn_lines: usize,
    /// Deprecated alias for `open_fate`.
    #[serde(default)]
    pub multihop_fate: bool,
}
fn default_proactive_min_fate_signals() -> usize {
    std::env::var("VALHALLA_PROACTIVE_MIN_FATE_SIGNALS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
}

fn default_proactive_min_turns_legacy() -> usize {
    0
}

fn default_proactive_min_spread() -> f64 {
    // adaptive coupling × delta scale (session-driven, not fixed constants)
    fog_tau_algebraic(0) * crate::native_qa::fog_delta_required_algebraic(1.0)
}

fn default_true() -> bool {
    true
}

fn default_triad() -> String {
    "triad".into()
}
fn default_lm_model() -> String {
    std::env::var("VALHALLA_LM_MODEL")
        .unwrap_or_else(|_| "Qwen/Qwen2.5-0.5B-Instruct".into())
}

fn default_use_cpu() -> bool {
    std::env::var("VALHALLA_USE_CPU")
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        })
        .unwrap_or(false)
}
fn default_strength() -> f64 {
    DEFAULT_STRENGTH
}
fn default_max_new() -> usize {
    DEFAULT_MAX_NEW
}
fn default_cycles() -> usize {
    2
}

fn default_fate_warmup_cycles() -> usize {
    std::env::var("VALHALLA_FATE_WARMUP_CYCLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
        .clamp(1, 64)
}

fn question_history_cap() -> usize {
    std::env::var("VALHALLA_QUESTION_HISTORY_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12)
}

fn cap_question_history(history: Vec<String>) -> Vec<String> {
    let cap = question_history_cap();
    if history.len() <= cap {
        return history;
    }
    history[history.len() - cap..].to_vec()
}

fn default_open_fate_cfg() -> bool {
    crate::multihop_fate::open_fate_enabled()
}

impl Default for ValhallaBaseConfig {
    fn default() -> Self {
        Self {
            lm_model: default_lm_model(),
            patch_body: default_triad(),
            strength: default_strength(),
            max_new: default_max_new(),
            incubation_cycles: default_cycles(),
            use_cpu: default_use_cpu(),
            decode: ValhallaBaseDecode::Hybrid,
            core_fusion: false,
            answer_mix: ValhallaAnswerMix::Mixed,
            mcq_decode: ValhallaMcqDecode::LmPatch,
            follow_up_decode: ValhallaFollowUpDecode::NativeFollowUpAware,
            fate_weight_mode: ValhallaFateWeightMode::Fixed,
            fate_routing_free: false,
            fate_ingress_routing: ValhallaFateIngressRouting::QuadCycle,
            fate_qa_feedback: true,
            proactive_enabled: true,
            proactive_min_fate_signals: default_proactive_min_fate_signals(),
            proactive_min_turns: 0,
            proactive_min_spread: default_proactive_min_spread(),
            multihop_v2: crate::multihop_compose::multihop_v2_enabled(),
            open_fate: default_open_fate_cfg(),
            fate_orchestrate: true,
            fate_warmup_enabled: true,
            fate_warmup_cycles: default_fate_warmup_cycles(),
            fate_warmup_prelearn_lines: 0,
            multihop_fate: default_open_fate_cfg(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValhallaBaseRequest {
    /// Legacy text-only context lines.
    #[serde(default)]
    pub context: Vec<String>,
    /// Multimodal context bundle (text / audio_features / image_features per item).
    #[serde(default)]
    pub context_items: Vec<ValhallaContextItem>,
    /// Incremental append — merged with session or full context (no full re-send).
    #[serde(default)]
    pub append_lines: Vec<String>,
    #[serde(default)]
    pub append_items: Vec<ValhallaContextItem>,
    #[serde(default)]
    pub session: Option<ValhallaBaseSessionState>,
    pub questions: Vec<ValhallaBaseQuestion>,
    /// When true with empty `questions`, run proactive idle (self-probe + FOG nudge).
    #[serde(default)]
    pub proactive_idle: bool,
    /// When true with context set, run Fate prelearn+warmup ingest only (no QA decode).
    #[serde(default)]
    pub fate_warmup_only: bool,
    /// When true with session + append payload, extend session snapshot only (no QA).
    #[serde(default)]
    pub session_extend_only: bool,
    /// When true with session, Fate-link dynamic fragments + adapt (long-chain finalize).
    #[serde(default)]
    pub fate_link_only: bool,
    /// When true with session, Fate-split + callback steps (refine fragment grasp).
    #[serde(default)]
    pub fate_split_callback_only: bool,
    /// When true with session, extra Fate warmup+link+adapt (multi-pass pretrain).
    #[serde(default)]
    pub fate_pretrain_adapt_only: bool,
    #[serde(default)]
    pub config: ValhallaBaseConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProactiveOutcome {
    pub triggered: bool,
    pub trigger_reason: String,
    pub probe_id: String,
    pub probe_question: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nudge: Option<String>,
    pub fog_decision: String,
    pub emitted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fog_evidence: Option<FogEvidence>,
    #[serde(default)]
    pub fate_signals_since_baseline: u64,
    #[serde(default)]
    pub fate_baseline_tick: u64,
    #[serde(default)]
    pub proactive_min_fate_signals: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValhallaBaseRow {
    pub id: String,
    pub score_type: String,
    pub answer: String,
    pub decode_path: String,
    pub correct: bool,
    pub score: ScoreResult,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub core_fusion: Option<CoreFusionRow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fog_evidence: Option<FogEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fog_decision: Option<FogDecision>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcq_native_aggregate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcq_patch_confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval_corpus_idx: Option<usize>,
    /// Fate grasp record (blocks activated + candidate series + grasp).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fate_grasp: Option<crate::fate_qa::FateGrasp>,
    /// Echo of config answer_mix for this row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_mix: Option<String>,
    /// Corpus lines eligible under answer_mix (after gate).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_mix_eligible: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreFusionRow {
    pub universal_score: f64,
    pub universal_formed: bool,
    pub shared_patch: f64,
    pub decode_snap: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FateTelemetry {
    pub hub_prefs: Vec<f64>,
    pub hub_prefs_spread: f64,
    pub hub_routed: [u64; 4],
    pub fate_weight_mode: String,
    pub fate_routing_free: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fate_ingress_routing: Option<String>,
    pub incubation_cycles: usize,
    /// Fate adaptive algebra identifiers for Fate/FOG gates
    #[serde(default = "default_algebra_formula")]
    pub algebra_formula: String,
    #[serde(default)]
    pub corpus_line_affinity_nonzero: usize,
    #[serde(default)]
    pub corpus_line_affinity_max: f64,
}

fn default_algebra_formula() -> String {
    "fate_adaptive_v1".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FateWarmupSummary {
    pub cycles_executed: usize,
    pub prelearn_lines: usize,
    pub aggregate_score: f64,
    pub stopped_reason: String,
}

impl From<crate::fate_orchestrator::FateWarmupOutcome> for FateWarmupSummary {
    fn from(w: crate::fate_orchestrator::FateWarmupOutcome) -> Self {
        Self {
            cycles_executed: w.cycles_executed,
            prelearn_lines: w.prelearn_lines,
            aggregate_score: w.aggregate_score,
            stopped_reason: w.stopped_reason,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValhallaBaseResponse {
    pub protocol: String,
    pub context_lines: usize,
    pub context_fingerprint: String,
    pub decode: ValhallaBaseDecode,
    pub session: ValhallaBaseSessionState,
    pub features: Vec<String>,
    pub rows: Vec<ValhallaBaseRow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fate_telemetry: Option<FateTelemetry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fate_warmup: Option<FateWarmupSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proactive: Option<ProactiveOutcome>,
}

pub fn context_fingerprint(context: &[String]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    context.len().hash(&mut h);
    for line in context {
        line.trim().hash(&mut h);
    }
    format!("{:016x}", h.finish())
}

pub fn item_to_corpus_line(item: &ValhallaContextItem) -> Option<String> {
    if let Some(t) = item.text.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
        return Some(t.to_string());
    }
    if item.audio_features.as_ref().is_some_and(|v| !v.is_empty()) {
        return Some("[audio_context]".into());
    }
    if item.image_features.as_ref().is_some_and(|v| !v.is_empty()) {
        return Some("[image_context]".into());
    }
    None
}

pub fn item_modality_mask(item: &ValhallaContextItem) -> u8 {
    let mut m = 0u8;
    if item
        .text
        .as_ref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
    {
        m |= MOD_TEXT;
    }
    if item.audio_features.as_ref().is_some_and(|v| !v.is_empty()) {
        m |= MOD_AUDIO;
    }
    if item.image_features.as_ref().is_some_and(|v| !v.is_empty()) {
        m |= MOD_IMAGE;
    }
    m
}

pub fn answer_mix_label(mode: ValhallaAnswerMix) -> &'static str {
    match mode {
        ValhallaAnswerMix::Mixed => "mixed",
        ValhallaAnswerMix::Text => "text",
        ValhallaAnswerMix::Audio => "audio",
        ValhallaAnswerMix::Image => "image",
    }
}

fn answer_mix_allows(mask: u8, mode: ValhallaAnswerMix) -> bool {
    match mode {
        ValhallaAnswerMix::Mixed => true,
        // Pure text channel preference: no audio/image tags on the line.
        ValhallaAnswerMix::Text => {
            mask & MOD_TEXT != 0 && (mask & (MOD_AUDIO | MOD_IMAGE)) == 0
        }
        ValhallaAnswerMix::Audio => mask & MOD_AUDIO != 0,
        ValhallaAnswerMix::Image => mask & MOD_IMAGE != 0,
    }
}

/// Build per-line modality masks aligned with `lines`.
fn build_corpus_modality_masks(
    lines: &[String],
    items: &[ValhallaContextItem],
    prev: Option<&ValhallaBaseSessionState>,
) -> Vec<u8> {
    if let Some(sess) = prev {
        if sess.corpus_modality_mask.len() == lines.len()
            && sess.corpus.len() == lines.len()
            && sess
                .corpus
                .iter()
                .zip(lines.iter())
                .all(|(a, b)| normalize_line_key(a) == normalize_line_key(b))
        {
            return sess.corpus_modality_mask.clone();
        }
    }
    let mut masks = vec![MOD_TEXT; lines.len()];
    for item in items {
        let Some(line) = item_to_corpus_line(item) else {
            continue;
        };
        let key = normalize_line_key(&line);
        let mask = item_modality_mask(item);
        if mask == 0 {
            continue;
        }
        for (i, existing) in lines.iter().enumerate() {
            if normalize_line_key(existing) == key {
                masks[i] = mask;
                break;
            }
        }
    }
    masks
}

/// Zero affinity on lines outside the selected answer_mix (non-isolated session remains intact).
fn gate_affinity_by_answer_mix(
    snap: &mut TriadIncubationSnapshot,
    masks: &[u8],
    mode: ValhallaAnswerMix,
) -> usize {
    if mode == ValhallaAnswerMix::Mixed || masks.is_empty() {
        return masks.len().max(snap.corpus_line_affinity.len());
    }
    if snap.corpus_line_affinity.len() < masks.len() {
        snap.corpus_line_affinity
            .resize(masks.len(), 0.0);
    }
    let mut eligible = 0usize;
    for (i, &mask) in masks.iter().enumerate() {
        if answer_mix_allows(mask, mode) {
            eligible += 1;
        } else if let Some(v) = snap.corpus_line_affinity.get_mut(i) {
            *v = 0.0;
        }
    }
    eligible
}

fn corpus_from_items(items: &[ValhallaContextItem]) -> Vec<String> {
    items.iter().filter_map(item_to_corpus_line).collect()
}

fn merge_context(
    req: &ValhallaBaseRequest,
) -> (Vec<String>, Vec<ValhallaContextItem>, bool, Vec<u8>) {
    let mut lines: Vec<String> = Vec::new();
    let mut masks: Vec<u8> = Vec::new();
    let mut items = req.context_items.clone();

    let push_line = |lines: &mut Vec<String>, masks: &mut Vec<u8>, line: String, mask: u8| {
        if line_already_present(lines, &line) {
            // Orphan duplicate: keep existing mask (stronger bits win).
            if let Some(i) = lines
                .iter()
                .position(|l| normalize_line_key(l) == normalize_line_key(&line))
            {
                masks[i] |= mask;
            }
            return;
        }
        lines.push(line);
        masks.push(mask);
    };

    for s in &req.context {
        let t = s.trim();
        if t.is_empty() {
            continue;
        }
        push_line(&mut lines, &mut masks, t.to_string(), MOD_TEXT);
    }

    if let Some(sess) = &req.session {
        if req.context.is_empty() && req.context_items.is_empty() {
            lines = sess.corpus.clone();
            masks = if sess.corpus_modality_mask.len() == lines.len() {
                sess.corpus_modality_mask.clone()
            } else {
                vec![MOD_TEXT | MOD_AUDIO | MOD_IMAGE; lines.len()]
            };
        }
    }

    let mut new_append_lines: Vec<String> = Vec::new();
    for l in &req.append_lines {
        let t = l.trim();
        if t.is_empty() {
            continue;
        }
        if line_already_present(&lines, t) {
            continue;
        }
        new_append_lines.push(t.to_string());
        push_line(&mut lines, &mut masks, t.to_string(), MOD_TEXT);
        items.push(ValhallaContextItem {
            text: Some(t.to_string()),
            ..Default::default()
        });
    }
    for item in &req.append_items {
        if let Some(t) = item.text.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            if line_already_present(&lines, t) {
                continue;
            }
        } else if item.audio_features.as_ref().is_none_or(|v| v.is_empty())
            && item.image_features.as_ref().is_none_or(|v| v.is_empty())
        {
            continue;
        }
        items.push(item.clone());
        if let Some(l) = item_to_corpus_line(item) {
            push_line(
                &mut lines,
                &mut masks,
                l,
                item_modality_mask(item).max(MOD_TEXT),
            );
        }
    }

    for item in &req.context_items {
        if let Some(l) = item_to_corpus_line(item) {
            push_line(
                &mut lines,
                &mut masks,
                l,
                item_modality_mask(item).max(MOD_TEXT),
            );
        }
    }

    let incremental = req.session.is_some()
        && (!new_append_lines.is_empty()
            || req.append_items.iter().any(|it| {
                it.text
                    .as_ref()
                    .map(|s| {
                        let t = s.trim();
                        !t.is_empty()
                            && !line_already_present(
                                req.session
                                    .as_ref()
                                    .map(|s| s.corpus.as_slice())
                                    .unwrap_or(&[]),
                                t,
                            )
                    })
                    .unwrap_or(false)
                    || it.audio_features.as_ref().is_some_and(|v| !v.is_empty())
                    || it.image_features.as_ref().is_some_and(|v| !v.is_empty())
            }))
        && req.context.is_empty()
        && req.context_items.is_empty();

    let _ = new_append_lines;
    if masks.len() != lines.len() {
        masks.resize(lines.len(), MOD_TEXT);
    }
    (lines, items, incremental, masks)
}

/// Redact corpus lines outside answer_mix so keyword land cannot leak gated modalities.
fn corpus_view_for_answer_mix(lines: &[String], masks: &[u8], mode: ValhallaAnswerMix) -> Vec<String> {
    if mode == ValhallaAnswerMix::Mixed || masks.is_empty() {
        return lines.to_vec();
    }
    lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let mask = masks.get(i).copied().unwrap_or(MOD_TEXT);
            if answer_mix_allows(mask, mode) {
                line.clone()
            } else {
                String::new()
            }
        })
        .collect()
}

fn feed_context_item(session: &mut TriadSession, item: &ValhallaContextItem, front: &DlFrontConfig) {
    let mut idx = session.fragment_tracks.len();
    let orch = crate::fate_orchestrator::FateOrchestratorConfig::default();
    if let Some(text) = item.text.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
        if crate::fate_orchestrator::multi_fragment_enabled() {
            crate::fate_orchestrator::ingest_modality_record_fragment(
                session,
                &ModalityInput::Text {
                    content: text.to_string(),
                },
                front,
                idx,
                &orch,
            );
        } else {
            session.feed_modality(
                ModalityInput::Text {
                    content: text.to_string(),
                },
                front,
            );
        }
        idx += 1;
    }
    if let Some(audio) = item.audio_features.as_ref().filter(|v| !v.is_empty()) {
        if crate::fate_orchestrator::multi_fragment_enabled() {
            crate::fate_orchestrator::ingest_modality_record_fragment(
                session,
                &ModalityInput::AudioFeatures {
                    values: audio.clone(),
                },
                front,
                idx,
                &orch,
            );
        } else {
            session.feed_modality(
                ModalityInput::AudioFeatures {
                    values: audio.clone(),
                },
                front,
            );
        }
        idx += 1;
    }
    if let Some(img) = item.image_features.as_ref().filter(|v| !v.is_empty()) {
        if crate::fate_orchestrator::multi_fragment_enabled() {
            crate::fate_orchestrator::ingest_modality_record_fragment(
                session,
                &ModalityInput::ImageFeatures {
                    values: img.clone(),
                },
                front,
                idx,
                &orch,
            );
        } else {
            session.feed_modality(
                ModalityInput::ImageFeatures {
                    values: img.clone(),
                },
                front,
            );
        }
    }
}

fn apply_fate_config(session: &mut TriadSession, cfg: &ValhallaBaseConfig) {
    let mode = match cfg.fate_weight_mode {
        ValhallaFateWeightMode::Fixed => FateWeightMode::Fixed,
        ValhallaFateWeightMode::Evaluated => FateWeightMode::Evaluated,
    };
    let ingress = match cfg.fate_ingress_routing {
        ValhallaFateIngressRouting::RoundRobin => IngressRoutingMode::RoundRobin,
        ValhallaFateIngressRouting::Semantic => IngressRoutingMode::Semantic,
        ValhallaFateIngressRouting::QuadCycle => IngressRoutingMode::QuadCycle,
        ValhallaFateIngressRouting::HybridQuadSemantic => IngressRoutingMode::HybridQuadSemantic,
    };
    session.hub.set_fate_policy(mode, cfg.fate_routing_free, ingress);
}

fn hub_prefs_spread(prefs: &[f64]) -> f64 {
    if prefs.is_empty() {
        return 0.0;
    }
    let max = prefs.iter().copied().fold(0.0f64, f64::max);
    let min = prefs.iter().copied().fold(f64::INFINITY, f64::min);
    if min.is_finite() {
        max - min
    } else {
        0.0
    }
}

fn fate_telemetry_from_snap(snap: &TriadIncubationSnapshot, cfg: &ValhallaBaseConfig) -> FateTelemetry {
    let aff = &snap.corpus_line_affinity;
    FateTelemetry {
        hub_prefs: snap.hub_prefs.clone(),
        hub_prefs_spread: hub_prefs_spread(&snap.hub_prefs),
        hub_routed: snap.hub_routed,
        fate_weight_mode: match cfg.fate_weight_mode {
            ValhallaFateWeightMode::Fixed => "fixed".into(),
            ValhallaFateWeightMode::Evaluated => "evaluated".into(),
        },
        fate_routing_free: cfg.fate_routing_free,
        fate_ingress_routing: Some(match cfg.fate_ingress_routing {
            ValhallaFateIngressRouting::RoundRobin => "round_robin".into(),
            ValhallaFateIngressRouting::Semantic => "semantic".into(),
            ValhallaFateIngressRouting::QuadCycle => "quad_cycle".into(),
            ValhallaFateIngressRouting::HybridQuadSemantic => "hybrid_quad_semantic".into(),
        }),
        incubation_cycles: cfg.incubation_cycles.max(1),
        algebra_formula: default_algebra_formula(),
        corpus_line_affinity_nonzero: aff.iter().filter(|&&v| v > 0.0).count(),
        corpus_line_affinity_max: aff.iter().copied().fold(0.0, f64::max),
    }
}

fn run_incubation_adapt(session: &mut TriadSession, cycles: usize, cfg: &ValhallaBaseConfig) -> TriadIncubationSnapshot {
    if cfg.fate_orchestrate {
        let orch = FateOrchestratorConfig::default();
        let (_, snap) = fate_orchestrator::ingress_and_adapt(session, cycles, &orch);
        snap
    } else {
        session.run_optimize_cycles(cycles.max(1));
        session.incubation_snapshot()
    }
}

fn fate_orchestrator_config(cfg: &ValhallaBaseConfig) -> crate::fate_orchestrator::FateOrchestratorConfig {
    let mut orch = crate::fate_orchestrator::FateOrchestratorConfig::default();
    orch.warmup_enabled = cfg.fate_warmup_enabled;
    orch.warmup_cycles = cfg.fate_warmup_cycles.clamp(1, 64);
    if cfg.fate_warmup_prelearn_lines > 0 {
        orch.warmup_prelearn_lines = cfg.fate_warmup_prelearn_lines;
    }
    orch
}

fn feed_lines_with_fragments(
    session: &mut TriadSession,
    lines: &[String],
    cfg: &ValhallaBaseConfig,
) -> (TriadIncubationSnapshot, Option<crate::fate_orchestrator::FateWarmupOutcome>) {
    let cap = default_port_cap();
    if cfg.fate_orchestrate {
        let ports: Vec<BodyFrontPorts> = lines
            .iter()
            .map(|l| grind_text_hash(l, cap))
            .collect();
        let orch = fate_orchestrator_config(cfg);
        let mode = crate::fate_orchestrator::production_fragment_mode();
        let (_, warmup) = crate::fate_orchestrator::ingest_corpus_with_fragments_mode_full(
            session, lines, &ports, &orch, mode,
        );
        (crate::fate_orchestrator::export_session_after_ingest(session), warmup)
    } else {
        for line in lines {
            let ports = grind_text_hash(line, cap);
            session.emit_body_ports_with_line(&ports, Some(line));
        }
        (
            run_incubation_adapt(session, cfg.incubation_cycles.max(1), cfg),
            None,
        )
    }
}

fn build_snapshot_from_items(
    body: &str,
    lines: &[String],
    items: &[ValhallaContextItem],
    cycles: usize,
    cfg: &ValhallaBaseConfig,
) -> (TriadIncubationSnapshot, Option<crate::fate_orchestrator::FateWarmupOutcome>) {
    let has_mm = items.iter().any(|i| {
        i.audio_features.as_ref().is_some_and(|v| !v.is_empty())
            || i.image_features.as_ref().is_some_and(|v| !v.is_empty())
    });
    if !has_mm {
        let target = BodyTarget::parse(body);
        let budget = IngressBudget::experiment();
        let mut session = TriadSession::with_body(budget, target);
        apply_fate_config(&mut session, cfg);
        return feed_lines_with_fragments(&mut session, lines, cfg);
    }
    let target = BodyTarget::parse(body);
    let budget = IngressBudget::experiment();
    let mut session = TriadSession::with_body(budget, target);
    apply_fate_config(&mut session, cfg);
    let front = DlFrontConfig::default();
    let cap = default_port_cap();
    for line in lines {
        let ports = grind_text_hash(line, cap);
        session.emit_body_ports_with_line(&ports, Some(line));
    }
    for item in items {
        feed_context_item(&mut session, item, &front);
    }
    (run_incubation_adapt(&mut session, cycles, cfg), None)
}

fn extend_snapshot(
    body: &str,
    prev: TriadIncubationSnapshot,
    new_lines: &[String],
    new_items: &[ValhallaContextItem],
    cycles: usize,
    cfg: &ValhallaBaseConfig,
    corpus_base_idx: usize,
) -> TriadIncubationSnapshot {
    let target = BodyTarget::parse(body);
    let budget = IngressBudget::experiment();
    let mut session = if cfg.fate_orchestrate && crate::fate_orchestrator::fragment_only_session_enabled() {
        let frags = crate::fate_orchestrator::stored_fragments(&prev);
        let mut s = crate::fate_orchestrator::session_from_stored_fragments(
            &frags,
            &prev.corpus_line_affinity,
            budget.clone(),
            target,
        );
        apply_fate_config(&mut s, cfg);
        s
    } else {
        let mut s = TriadSession::from_incubation_snapshot(budget.clone(), target, prev);
        apply_fate_config(&mut s, cfg);
        s
    };
    let front = DlFrontConfig::default();
    let cap = default_port_cap();

    if cfg.fate_orchestrate && !new_lines.is_empty() {
        let ports: Vec<BodyFrontPorts> = new_lines
            .iter()
            .map(|l| grind_text_hash(l, cap))
            .collect();
        let orch = fate_orchestrator_config(cfg);
        let mode = crate::fate_orchestrator::production_fragment_mode();
        let _ = crate::fate_orchestrator::append_corpus_with_fragments_mode(
            &mut session,
            new_lines,
            &ports,
            &orch,
            corpus_base_idx,
            mode,
        );
        if crate::fate_orchestrator::fate_link_fragments_enabled()
            && crate::fate_orchestrator::fate_link_each_extend_enabled()
        {
            let _ = crate::fate_orchestrator::fate_link_fragments(&mut session, &orch);
        }
    } else {
        for line in new_lines {
            let ports = grind_text_hash(line, cap);
            session.emit_body_ports_with_line(&ports, Some(line));
        }
        let _ = run_incubation_adapt(&mut session, cycles, cfg);
    }

    for item in new_items {
        feed_context_item(&mut session, item, &front);
    }
    if new_lines.is_empty() && !new_items.is_empty() && !cfg.fate_orchestrate {
        let _ = run_incubation_adapt(&mut session, cycles, cfg);
    }
    crate::fate_orchestrator::export_session_after_ingest(&session)
}

fn normalize_line_key(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn line_already_present(lines: &[String], candidate: &str) -> bool {
    let key = normalize_line_key(candidate);
    lines.iter().any(|l| normalize_line_key(l) == key)
}

/// Map learned affinity from prior session corpus onto fresh targeted lines.
fn reanchor_snapshot_affinity(
    prev: &ValhallaBaseSessionState,
    new_lines: &[String],
    mut snap: TriadIncubationSnapshot,
) -> TriadIncubationSnapshot {
    let old_aff = &prev.snapshot.corpus_line_affinity;
    if old_aff.is_empty() || new_lines.is_empty() {
        return snap;
    }
    let mut new_aff = vec![0.0; new_lines.len()];
    for (ni, line) in new_lines.iter().enumerate() {
        let key = normalize_line_key(line);
        for (oi, old_line) in prev.corpus.iter().enumerate() {
            if normalize_line_key(old_line) == key {
                if let Some(&v) = old_aff.get(oi) {
                    new_aff[ni] = v;
                }
                break;
            }
        }
    }
    snap.corpus_line_affinity = new_aff;
    snap
}

fn resolve_snapshot(
    req: &ValhallaBaseRequest,
    lines: &[String],
    items: &[ValhallaContextItem],
    incremental: bool,
) -> (TriadIncubationSnapshot, Option<crate::fate_orchestrator::FateWarmupOutcome>) {
    let cycles = req.config.incubation_cycles.max(1);
    let body = &req.config.patch_body;
    if let Some(sess) = &req.session {
        if req.context.is_empty() && req.context_items.is_empty() {
            let has_new_append = req.append_lines.iter().any(|l| {
                let t = l.trim();
                !t.is_empty() && !line_already_present(&sess.corpus, t)
            }) || req.append_items.iter().any(|it| {
                it.text
                    .as_ref()
                    .map(|s| {
                        let t = s.trim();
                        !t.is_empty() && !line_already_present(&sess.corpus, t)
                    })
                    .unwrap_or(false)
                    || it.audio_features.as_ref().is_some_and(|v| !v.is_empty())
                    || it.image_features.as_ref().is_some_and(|v| !v.is_empty())
            });
            // Duplicate retrieve→append on an already-ingested long-chain: reuse snapshot.
            if !has_new_append {
                return (sess.snapshot.clone(), None);
            }
        }
    }
    if incremental {
        if let Some(sess) = &req.session {
            let fresh_lines: Vec<String> = req
                .append_lines
                .iter()
                .map(|s| s.trim().to_string())
                .filter(|t| !t.is_empty() && !line_already_present(&sess.corpus, t))
                .collect();
            let fresh_items: Vec<ValhallaContextItem> = req
                .append_items
                .iter()
                .filter(|it| {
                    it.text
                        .as_ref()
                        .map(|s| {
                            let t = s.trim();
                            !t.is_empty() && !line_already_present(&sess.corpus, t)
                        })
                        .unwrap_or(false)
                        || it.audio_features.as_ref().is_some_and(|v| !v.is_empty())
                        || it.image_features.as_ref().is_some_and(|v| !v.is_empty())
                })
                .cloned()
                .collect();
            if fresh_lines.is_empty() && fresh_items.is_empty() {
                return (sess.snapshot.clone(), None);
            }
            return (
                extend_snapshot(
                    body,
                    sess.snapshot.clone(),
                    &fresh_lines,
                    &fresh_items,
                    cycles,
                    &req.config,
                    sess.corpus.len(),
                ),
                None,
            );
        }
    }
    if let Some(sess) = &req.session {
        if req.context.is_empty()
            && req.context_items.is_empty()
            && req.append_lines.is_empty()
            && req.append_items.is_empty()
        {
            return (sess.snapshot.clone(), None);
        }
    }
    let (snap, warmup) = build_snapshot_from_items(body, lines, items, cycles, &req.config);
    if let Some(sess) = &req.session {
        if !incremental
            && (!req.context.is_empty() || !req.context_items.is_empty())
            && req.config.fate_qa_feedback
        {
            return (reanchor_snapshot_affinity(sess, lines, snap), warmup);
        }
    }
    (snap, warmup)
}

fn effective_decode(cfg: &ValhallaBaseConfig, score_type: &str) -> ValhallaBaseDecode {
    match cfg.decode {
        ValhallaBaseDecode::Hybrid => match score_type {
            "mcq" | "numeric" if cfg.mcq_decode == ValhallaMcqDecode::StructureFate => {
                ValhallaBaseDecode::Native
            }
            "mcq" | "numeric" => ValhallaBaseDecode::LmPatch,
            _ => ValhallaBaseDecode::Native,
        },
        other => other,
    }
}

fn effective_decode_for_question(
    cfg: &ValhallaBaseConfig,
    score_type: &str,
    is_follow_up: bool,
) -> ValhallaBaseDecode {
    let base = effective_decode(cfg, score_type);
    if score_type != "open" || !is_follow_up {
        return base;
    }
    match cfg.follow_up_decode {
        ValhallaFollowUpDecode::Inherit => base,
        ValhallaFollowUpDecode::LmPatchOnFollowUp => ValhallaBaseDecode::LmPatch,
        ValhallaFollowUpDecode::NativeFollowUpAware => ValhallaBaseDecode::Native,
    }
}

fn task_ports(q: &ValhallaBaseQuestion, p: &FairPrompt, front: &DlFrontConfig) -> (BodyFrontPorts, BodyFrontPorts, BodyFrontPorts) {
    let llm = encode_modality(
        &ModalityInput::Text {
            content: format_prompt_text(p),
        },
        front,
    );
    let tts = q
        .audio_features
        .as_ref()
        .filter(|v| !v.is_empty())
        .map(|v| {
            encode_modality(
                &ModalityInput::AudioFeatures {
                    values: v.clone(),
                },
                front,
            )
        })
        .unwrap_or_else(|| llm.clone());
    let img = q
        .image_features
        .as_ref()
        .filter(|v| !v.is_empty())
        .map(|v| {
            encode_modality(
                &ModalityInput::ImageFeatures {
                    values: v.clone(),
                },
                front,
            )
        })
        .unwrap_or_else(|| llm.clone());
    (llm, tts, img)
}

fn maybe_fuse_core(
    cfg: &ValhallaBaseConfig,
    context_snap: TriadIncubationSnapshot,
    q: &ValhallaBaseQuestion,
    p: &FairPrompt,
) -> (TriadIncubationSnapshot, Option<CoreFusionRow>) {
    if !cfg.core_fusion {
        return (context_snap, None);
    }
    let has_mm = q.audio_features.as_ref().is_some_and(|v| !v.is_empty())
        || q.image_features.as_ref().is_some_and(|v| !v.is_empty());
    if !has_mm {
        return (context_snap, None);
    }
    let front = DlFrontConfig::default();
    let body = BodyTarget::parse(&cfg.patch_body);
    let (llm, tts, img) = task_ports(q, p, &front);
    let (fused, metrics) = fuse_universal_core_on_snapshot(context_snap, body, &llm, &tts, &img);
    let row = core_fusion_row(&metrics, "context+task_non_isolated");
    (fused, Some(row))
}

fn core_fusion_row(metrics: &UniversalCoreMetrics, snap_label: &str) -> CoreFusionRow {
    CoreFusionRow {
        universal_score: metrics.universal_core_score,
        universal_formed: metrics.universal_core_formed,
        shared_patch: metrics.shared_patch_pairwise_avg,
        decode_snap: snap_label.to_string(),
    }
}

fn open_prompt_has_scorable_reference(p: &FairPrompt) -> bool {
    !p.reference_answer.trim().is_empty()
        || p.keywords.iter().any(|k| !k.trim().is_empty())
}

fn fate_feedback_on_topic(p: &FairPrompt, row: &ValhallaBaseRow, lines: &[String]) -> bool {
    if p.score_type != "open" {
        return row.correct;
    }
    if open_prompt_has_scorable_reference(p) {
        return row.correct;
    }
    let line_text = row
        .retrieval_corpus_idx
        .and_then(|i| lines.get(i).map(|s| s.as_str()));
    structural_open_on_topic(&p.text, &row.answer, line_text)
}

fn apply_qa_feedback_to_snapshot(
    snap: TriadIncubationSnapshot,
    cfg: &ValhallaBaseConfig,
    question: &str,
    on_topic: bool,
    question_ports: &BodyFrontPorts,
    retrieved_corpus_idx: Option<usize>,
    question_sig: Option<Vec<f64>>,
    line_sig: Option<Vec<f64>>,
    line_text: Option<&str>,
) -> TriadIncubationSnapshot {
    let budget = IngressBudget::experiment();
    let target = BodyTarget::parse(&cfg.patch_body);
    let frags = crate::fate_orchestrator::stored_fragments(&snap);
    let mut session = if crate::fate_orchestrator::fragment_only_session_enabled() && !frags.is_empty() {
        crate::fate_orchestrator::session_from_stored_fragments(
            &frags,
            &snap.corpus_line_affinity,
            budget,
            target,
        )
    } else {
        TriadSession::from_incubation_snapshot(budget, target, snap)
    };
    apply_fate_config(&mut session, cfg);
    session.emit_body_ports(question_ports);
    if cfg.fate_qa_feedback {
        session.apply_qa_feedback(question, on_topic);
        if let (Some(idx), Some(qs), Some(ls), Some(lt)) =
            (retrieved_corpus_idx, question_sig, line_sig, line_text)
        {
            session.apply_corpus_retrieval_feedback(idx, &qs, &ls, lt, on_topic);
        }
    }
    crate::fate_orchestrator::export_session_after_ingest(&session)
}

fn session_state_from_snap(
    corpus: &[String],
    snap: TriadIncubationSnapshot,
    question_history: Vec<String>,
    fate_baseline_tick: u64,
) -> ValhallaBaseSessionState {
    session_state_from_snap_with_book(
        corpus,
        snap,
        question_history,
        fate_baseline_tick,
        None,
        None,
        Vec::new(),
    )
}

fn session_state_from_snap_with_book(
    corpus: &[String],
    snap: TriadIncubationSnapshot,
    question_history: Vec<String>,
    fate_baseline_tick: u64,
    address_book: Option<crate::valhalla_address::AddressBook>,
    algebra_meta: Option<crate::fate_adaptive_algebra::FateAlgebraMeta>,
    corpus_modality_mask: Vec<u8>,
) -> ValhallaBaseSessionState {
    ValhallaBaseSessionState {
        corpus: corpus.to_vec(),
        fingerprint: context_fingerprint(corpus),
        lines_ingested: corpus.len(),
        fragments: crate::fate_orchestrator::stored_fragments(&snap),
        snapshot: snap,
        // Preserve RelAddr / pretrain algebra — do not Default-wipe each round-trip.
        fate_algebra_meta: algebra_meta.unwrap_or_default(),
        fate_rollback_stack: Vec::new(),
        question_history,
        fate_baseline_tick,
        address_book,
        corpus_modality_mask,
    }
}

fn effective_proactive_min_fate_signals(cfg: &ValhallaBaseConfig) -> usize {
    if cfg.proactive_min_turns > 0 {
        cfg.proactive_min_turns.saturating_mul(32)
    } else {
        cfg.proactive_min_fate_signals
    }
}

fn fate_signals_since_baseline(snap: &TriadIncubationSnapshot, baseline: u64) -> u64 {
    snap.fate_signal_tick.saturating_sub(baseline)
}

fn grind_ports_for_line(line: &str) -> BodyFrontPorts {
    let cap = default_port_cap();
    grind_text_hash(line, cap)
}

fn run_native_one(
    corpus: &[String],
    p: &FairPrompt,
    q: &ValhallaBaseQuestion,
    cfg: &ValhallaBaseConfig,
    context_snap: TriadIncubationSnapshot,
    prior_questions: &[String],
) -> Result<ValhallaBaseRow, String> {
    let (fused_snap, core_row) = maybe_fuse_core(cfg, context_snap, q, p);
    let corpus_ports: Vec<BodyFrontPorts> =
        corpus.iter().map(|l| grind_ports_for_line(l)).collect();
    let q_ports = grind_ports_for_line(&format_prompt_text(p));
    let fragment_only = crate::fate_orchestrator::fragment_only_session_enabled();
    let stored = crate::fate_orchestrator::stored_fragments(&fused_snap);
    let batch = NativeQaBatchRequest {
        body: cfg.patch_body.clone(),
        corpus: corpus.to_vec(),
        corpus_ports,
        questions: vec![NativeQaQuestionSpec {
            id: Some(p.id.clone()),
            question: format_prompt_text(p),
            score_type: p.score_type.clone(),
            question_ports: Some(q_ports),
            audio_features: q.audio_features.clone(),
            prior_questions: if prior_questions.is_empty() {
                None
            } else {
                Some(prior_questions.to_vec())
            },
        }],
        incubation_cycles: cfg.incubation_cycles.max(1),
        qa_cycles: 1,
        ingress_mode: Some("grinder".into()),
        mcq_fate: p.score_type == "mcq" && cfg.mcq_decode == ValhallaMcqDecode::StructureFate,
        pipeline_v4: true,
        multihop_v2: cfg.multihop_v2,
        open_fate: cfg.open_fate || cfg.multihop_fate,
        multihop_fate: cfg.open_fate || cfg.multihop_fate,
        incubation_snapshot: if fragment_only && !stored.is_empty() {
            None
        } else {
            Some(fused_snap)
        },
        corpus_fragments: if fragment_only && !stored.is_empty() {
            Some(stored)
        } else {
            None
        },
    };
    let resp = run_native_qa_batch(&batch);
    let row = resp.results.into_iter().next().ok_or("native batch empty")?;
    let sc = score_answer(&row.answer, p);
    Ok(ValhallaBaseRow {
        id: p.id.clone(),
        score_type: p.score_type.clone(),
        answer: row.answer,
        decode_path: format!("valhalla_base/native/{}", row.decode_path),
        correct: sc.correct,
        score: sc,
        core_fusion: core_row,
        fog_evidence: row.fog_evidence,
        fog_decision: row.fog_decision,
        mcq_native_aggregate: if p.score_type == "mcq" {
            Some(row.mcq_aggregate_score)
        } else {
            None
        },
        mcq_patch_confidence: None,
        retrieval_corpus_idx: row.retrieval_corpus_idx,
        fate_grasp: row.fate_grasp,
        answer_mix: None,
        answer_mix_eligible: None,
    })
}

fn run_lm_one(
    corpus: &[String],
    p: &FairPrompt,
    q: &ValhallaBaseQuestion,
    cfg: &ValhallaBaseConfig,
    context_snap: TriadIncubationSnapshot,
) -> Result<ValhallaBaseRow, String> {
    let (fused_snap, core_row) = maybe_fuse_core(cfg, context_snap, q, p);
    let snap_for_patch = if crate::fate_orchestrator::fragment_only_session_enabled() {
        let frags = crate::fate_orchestrator::stored_fragments(&fused_snap);
        if frags.is_empty() {
            fused_snap
        } else {
            let qtext = format_prompt_text(p);
            let q_ports = grind_ports_for_line(&qtext);
            let corpus_sigs: Vec<(usize, Vec<f64>)> = corpus
                .iter()
                .enumerate()
                .map(|(i, l)| (i, grind_ports_for_line(l).embedding))
                .collect();
            let body = BodyTarget::parse(&cfg.patch_body);
            let budget = IngressBudget::experiment();
            let top_k = crate::fate_orchestrator::query_fragment_top_k(corpus.len());
            let (merged, _) = crate::fate_orchestrator::merge_fragments_for_query(
                &frags,
                &qtext,
                &q_ports.embedding,
                &corpus_sigs,
                corpus,
                budget,
                body,
                top_k,
            );
            crate::fate_orchestrator::compact_manifold_snapshot(&merged)
        }
    } else {
        fused_snap
    };
    let patch_cfg = PatchLmConfig {
        model_dir: crate::patch_lm::resolve_model_dir(&cfg.lm_model),
        patch_body: cfg.patch_body.clone(),
        strength: cfg.strength,
        max_new: cfg.max_new,
        export_cycles: cfg.incubation_cycles.max(1),
        use_cpu: cfg.use_cpu,
        pipeline_v4: true,
        mcq_decode: crate::patch_lm::McqDecodeMode::LogprobBatch,
        runtime_patch: Some(true),
    };
    let mut audio_by_id = std::collections::HashMap::new();
    if let Some(v) = &q.audio_features {
        if !v.is_empty() {
            audio_by_id.insert(p.id.clone(), v.clone());
        }
    }
    let patch_rows = crate::patch_lm::run_patch_track_v4_with_snap(
        &snap_for_patch,
        std::slice::from_ref(p),
        &patch_cfg,
        if audio_by_id.is_empty() {
            None
        } else {
            Some(&audio_by_id)
        },
    )?;
    let pr = &patch_rows[0];
    let answer = pr
        .assistant
        .clone()
        .or_else(|| pr.native_answer.clone())
        .unwrap_or_default();
    let sc = score_answer(&answer, p);
    let decode_path = pr
        .decode_path
        .clone()
        .unwrap_or_else(|| "lm_patch".into());
    Ok(ValhallaBaseRow {
        id: p.id.clone(),
        score_type: p.score_type.clone(),
        answer,
        decode_path: format!("valhalla_base/lm_patch/{decode_path}"),
        correct: sc.correct,
        score: sc,
        core_fusion: core_row,
        fog_evidence: None,
        fog_decision: None,
        mcq_native_aggregate: None,
        mcq_patch_confidence: if p.score_type == "mcq" {
            pr.mcq_patch_confidence
        } else {
            None
        },
        retrieval_corpus_idx: None,
        fate_grasp: None,
        answer_mix: None,
        answer_mix_eligible: None,
    })
}

/// True when native open retrieval abstained (FOG / low score / no keyword overlap).
fn native_open_answer_rejected(answer: &str, decode_path: &str) -> bool {
    answer.starts_with("No confident match")
        || decode_path.contains("open_reject")
        || decode_path.contains("open_fog_abstain")
}

fn hybrid_open_lm_fallback_enabled() -> bool {
    std::env::var("VALHALLA_HYBRID_OPEN_LM_FALLBACK")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        })
        // Default off under low-resource budget (avoids CUDA reload hang).
        .unwrap_or_else(|_| {
            std::env::var("VALHALLA_INGRESS_RESOURCE_PERCENT")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .map(|p| p > 60)
                .unwrap_or(false)
        })
}

/// Hybrid open path: native retrieval first; LM decode when native abstains.
fn run_open_decode_row(
    corpus: &[String],
    p: &FairPrompt,
    q: &ValhallaBaseQuestion,
    cfg: &ValhallaBaseConfig,
    context_snap: TriadIncubationSnapshot,
    prior_questions: &[String],
) -> Result<ValhallaBaseRow, String> {
    let native = run_native_one(
        corpus,
        p,
        q,
        cfg,
        context_snap.clone(),
        prior_questions,
    )?;
    if cfg.decode != ValhallaBaseDecode::Hybrid
        || p.score_type != "open"
        || !hybrid_open_lm_fallback_enabled()
        || !native_open_answer_rejected(&native.answer, &native.decode_path)
    {
        return Ok(native);
    }
    let mut patch = run_lm_one(corpus, p, q, cfg, context_snap)?;
    let native_tag = native
        .decode_path
        .strip_prefix("valhalla_base/native/")
        .unwrap_or(&native.decode_path);
    let lm_tag = patch
        .decode_path
        .strip_prefix("valhalla_base/lm_patch/")
        .unwrap_or(&patch.decode_path);
    patch.decode_path = format!("valhalla_base/hybrid/open_lm_fallback/{native_tag}/{lm_tag}");
    Ok(patch)
}

/// Deploy MCQ pick: native structure vs lm_patch logprob (no oracle labels).
fn pick_mcq_max_deploy(native: ValhallaBaseRow, patch: ValhallaBaseRow) -> ValhallaBaseRow {
    let n_l = extract_mcq_letter(&native.answer);
    let p_l = extract_mcq_letter(&patch.answer);
    let native_agg = native.mcq_native_aggregate.unwrap_or(0.0);
    let patch_conf = patch.mcq_patch_confidence.unwrap_or(0.0);
    let native_margin = native
        .fog_evidence
        .as_ref()
        .map(|e| e.margin)
        .unwrap_or(0.0);

    let use_native = match (n_l, p_l) {
        (Some(_), None) => true,
        (None, Some(_)) | (None, None) => false,
        (Some(a), Some(b)) if a == b => false,
        (Some(_), Some(_)) => native_agg >= 0.48 && patch_conf <= 0.45 && native_margin >= 0.08,
    };

    if use_native {
        ValhallaBaseRow {
            decode_path: format!(
                "valhalla_base/mcq_max_pick/native/{}",
                native
                    .decode_path
                    .strip_prefix("valhalla_base/native/")
                    .unwrap_or(&native.decode_path)
            ),
            mcq_native_aggregate: native.mcq_native_aggregate,
            mcq_patch_confidence: patch.mcq_patch_confidence,
            ..native
        }
    } else {
        ValhallaBaseRow {
            decode_path: format!(
                "valhalla_base/mcq_max_pick/lm_patch/{}",
                patch
                    .decode_path
                    .strip_prefix("valhalla_base/lm_patch/")
                    .unwrap_or(&patch.decode_path)
            ),
            mcq_native_aggregate: native.mcq_native_aggregate,
            mcq_patch_confidence: patch.mcq_patch_confidence,
            ..patch
        }
    }
}

fn run_mcq_max_pick(
    corpus: &[String],
    p: &FairPrompt,
    q: &ValhallaBaseQuestion,
    cfg: &ValhallaBaseConfig,
    context_snap: TriadIncubationSnapshot,
    prior: &[String],
) -> Result<ValhallaBaseRow, String> {
    let native = run_native_one(corpus, p, q, cfg, context_snap.clone(), prior)?;
    let patch = run_lm_one(corpus, p, q, cfg, context_snap)?;
    Ok(pick_mcq_max_deploy(native, patch))
}

fn run_legacy_native_track(
    context: &[String],
    questions: &[FairPrompt],
    cfg: &ValhallaBaseConfig,
) -> Result<Vec<ValhallaBaseRow>, String> {
    let corpus = context.to_vec();
    let mut rows = Vec::with_capacity(questions.len());
    for p in questions {
        let req = NativeQaRequest {
            body: cfg.patch_body.clone(),
            cycles: cfg.incubation_cycles.max(1),
            corpus: corpus.clone(),
            question: format_prompt_text(p),
            score_type: p.score_type.clone(),
            ingress_mode: None,
            corpus_ports: None,
            question_ports: None,
            corpus_cycles: Some(cfg.incubation_cycles.max(1)),
            qa_cycles: Some(1),
            ingest_all_corpus: Some(true),
        };
        let resp = run_native_qa(&req);
        let sc = score_answer(&resp.answer, p);
        rows.push(ValhallaBaseRow {
            id: p.id.clone(),
            score_type: p.score_type.clone(),
            answer: resp.answer,
            decode_path: format!("valhalla_base/native/{}", resp.decode_path),
            correct: sc.correct,
            score: sc,
            core_fusion: None,
            fog_evidence: None,
            fog_decision: None,
            mcq_native_aggregate: None,
            mcq_patch_confidence: None,
            retrieval_corpus_idx: None,
            fate_grasp: None,
            answer_mix: None,
            answer_mix_eligible: None,
        });
    }
    Ok(rows)
}

fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
        || ('\u{3040}'..='\u{309f}').contains(&c)
        || ('\u{30a0}'..='\u{30ff}').contains(&c)
        || ('\u{ac00}'..='\u{d7a3}').contains(&c)
}

const PROACTIVE_STOP: &[&str] = &[
    "the", "what", "does", "context", "about", "which", "that", "this", "from", "with",
    "have", "been", "were", "when", "where", "would", "could", "should", "does", "say",
    "anything", "mention", "stated", "based", "conversation", "related", "our",
];

fn text_tokens(text: &str) -> std::collections::HashSet<String> {
    use std::collections::HashSet;
    let mut out = HashSet::new();
    for w in text.split_whitespace() {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        if w.len() > 2 && !PROACTIVE_STOP.contains(&w.as_str()) {
            out.insert(w);
        }
    }
    let mut digits = String::new();
    for c in text.chars() {
        if c.is_ascii_digit() {
            digits.push(c);
        } else if !digits.is_empty() {
            out.insert(digits.clone());
            digits.clear();
        }
    }
    if !digits.is_empty() {
        out.insert(digits);
    }
    let cjk: Vec<char> = text.chars().filter(|c| is_cjk(*c)).collect();
    // CJK bigrams only (avoid single-char false positives on mismatch corpora)
    for w in cjk.windows(2) {
        out.insert(w.iter().collect());
    }
    // Single CJK only when rare/specific (e.g. 秦)
    if cjk.len() == 1 {
        out.insert(cjk[0].to_string());
    }
    out
}

/// Acronym in prior (e.g. GDP) ↔ consecutive word initials in corpus line.
fn proactive_acronym_matches(line: &str, prior: &[String]) -> usize {
    let words: Vec<String> = line
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphabetic())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect();
    let mut hits = 0usize;
    for p in prior {
        for raw in p.split_whitespace() {
            let w = raw.trim_matches(|c: char| !c.is_alphabetic());
            if w.len() < 2
                || w.len() > 6
                || !w.chars().all(|c| c.is_ascii_uppercase())
            {
                continue;
            }
            let initials: Vec<char> = w.chars().map(|c| c.to_ascii_lowercase()).collect();
            if words.len() < initials.len() {
                continue;
            }
            let matched = (0..=words.len() - initials.len()).any(|start| {
                initials
                    .iter()
                    .enumerate()
                    .all(|(i, c)| words[start + i].starts_with(*c))
            });
            if matched {
                hits += 1;
            }
        }
    }
    hits
}

fn proactive_overlap_count(line: &str, prior: &[String]) -> usize {
    let line_set = text_tokens(line);
    let mut best = 0usize;
    for p in prior {
        let prior_set = text_tokens(p);
        let mut overlap = prior_set.intersection(&line_set).count();
        overlap += prior_set
            .iter()
            .filter(|t| t.chars().any(is_cjk) && line.contains(t.as_str()))
            .count();
        best = best.max(overlap);
    }
    best + proactive_acronym_matches(line, prior)
}

fn dominant_script(prior: &[String]) -> &'static str {
    let mut cjk = 0usize;
    let mut latin = 0usize;
    for p in prior {
        for c in p.chars() {
            if is_cjk(c) {
                cjk += 1;
            } else if c.is_ascii_alphabetic() {
                latin += 1;
            }
        }
    }
    if cjk > latin { "cjk" } else { "latin" }
}

fn topic_snippet(line: &str) -> String {
    let cjk: String = line.chars().filter(|c| is_cjk(*c)).take(12).collect();
    if cjk.chars().count() >= 2 {
        return cjk;
    }
    line.split_whitespace().take(8).collect::<Vec<_>>().join(" ")
}

fn format_probe_question(topic: &str, prior: &[String]) -> String {
    if dominant_script(prior) == "cjk" {
        format!("根据上下文，关于「{topic}」说了什么？")
    } else {
        format!("Based on the context, what is stated about: {topic}?")
    }
}

fn build_proactive_probe(corpus: &[String], prior: &[String]) -> Option<(String, String, String)> {
    if corpus.is_empty() || prior.is_empty() {
        return None;
    }
    // Topic-continuous: pick corpus line with highest overlap with prior user turns.
    let (idx, overlap) = corpus
        .iter()
        .enumerate()
        .map(|(i, line)| (i, proactive_overlap_count(line, prior)))
        .max_by_key(|(_, count)| *count)?;
    if overlap == 0 {
        return None;
    }
    // Require meaningful overlap — single weak token insufficient on distractor corpora.
    if overlap < 2 {
        let picked = &corpus[idx];
        let line_set = text_tokens(picked);
        let strong = prior.iter().any(|p| {
            text_tokens(p).intersection(&line_set).any(|t| {
                t.len() >= 4
                    || t.chars().any(is_cjk)
                    || (t.chars().all(|c| c.is_ascii_digit()) && !t.is_empty())
            })
        });
        if !strong && proactive_acronym_matches(picked, prior) == 0 {
            return None;
        }
    }
    let line = &corpus[idx];
    let topic = topic_snippet(line);
    let id = format!("proactive_{}", prior.len());
    let text = format_probe_question(&topic, prior);
    Some((id, text, line.clone()))
}

fn proactive_trigger_gate(
    snap: &TriadIncubationSnapshot,
    fate_baseline: u64,
    cfg: &ValhallaBaseConfig,
) -> (bool, String, u64, usize) {
    let since = fate_signals_since_baseline(snap, fate_baseline);
    let min = effective_proactive_min_fate_signals(cfg);
    if !cfg.proactive_enabled {
        return (false, "proactive_disabled".into(), since, min);
    }
    if since < min as u64 {
        return (
            false,
            format!("fate_signals={since}<min_fate_signals={min}"),
            since,
            min,
        );
    }
    let spread = hub_prefs_spread(&snap.hub_prefs);
    if spread < cfg.proactive_min_spread {
        return (
            false,
            format!(
                "spread={spread:.4}<{min:.4}",
                spread = spread,
                min = cfg.proactive_min_spread
            ),
            since,
            min,
        );
    }
    (true, "idle_ready".into(), since, min)
}

fn proactive_outcome_blocked(
    reason: String,
    since: u64,
    min: usize,
    fate_baseline: u64,
) -> ProactiveOutcome {
    ProactiveOutcome {
        triggered: false,
        trigger_reason: reason,
        probe_id: String::new(),
        probe_question: String::new(),
        nudge: None,
        fog_decision: "defer".into(),
        emitted: false,
        fog_evidence: None,
        fate_signals_since_baseline: since,
        fate_baseline_tick: fate_baseline,
        proactive_min_fate_signals: min,
    }
}

fn run_proactive_idle(
    _req: &ValhallaBaseRequest,
    lines: &[String],
    context_snap: TriadIncubationSnapshot,
    prior: &[String],
    fate_baseline: u64,
    cfg: &ValhallaBaseConfig,
) -> Result<(ProactiveOutcome, Vec<ValhallaBaseRow>), String> {
    let (ready, reason, since, min) =
        proactive_trigger_gate(&context_snap, fate_baseline, cfg);
    if !ready {
        return Ok((proactive_outcome_blocked(reason, since, min, fate_baseline), vec![]));
    }

    let Some((probe_id, probe_text, source_line)) = build_proactive_probe(lines, prior) else {
        return Ok((
            proactive_outcome_blocked("no_topic_overlap".into(), since, min, fate_baseline),
            vec![],
        ));
    };

    let q = ValhallaBaseQuestion {
        id: probe_id.clone(),
        text: probe_text.clone(),
        score_type: "open".into(),
        reference_answer: source_line,
        keywords: vec![],
        audio_features: None,
        image_features: None,
    };
    let p: FairPrompt = q.clone().into();
    let row = run_native_one(lines, &p, &q, cfg, context_snap, prior)?;
    let fog_decision = row
        .fog_decision
        .as_ref()
        .map(|d| d.decision.clone())
        .unwrap_or_else(|| "emit".into());
    let emitted = fog_decision == "emit";
    let nudge = if emitted {
        Some(format!(
            "Related to our conversation: {}",
            row.answer.trim()
        ))
    } else {
        None
    };

    Ok((
        ProactiveOutcome {
            triggered: true,
            trigger_reason: reason,
            probe_id,
            probe_question: probe_text,
            nudge,
            fog_decision,
            emitted,
            fog_evidence: row.fog_evidence.clone(),
            fate_signals_since_baseline: since,
            fate_baseline_tick: fate_baseline,
            proactive_min_fate_signals: min,
        },
        vec![row],
    ))
}

/// Run Valhalla-as-base with hybrid / session / multimodal context.
pub fn run_valhalla_base(req: &ValhallaBaseRequest) -> Result<ValhallaBaseResponse, String> {
    let (raw_lines, items, incremental, merge_masks) = merge_context(req);
    if raw_lines.is_empty() && items.is_empty() && !req.fate_link_only && !req.fate_pretrain_adapt_only && !req.fate_split_callback_only {
        return Err("valhalla_base: context must contain at least one item or line".into());
    }
    if req.questions.is_empty()
        && !req.proactive_idle
        && !req.fate_warmup_only
        && !req.session_extend_only
        && !req.fate_link_only
        && !req.fate_split_callback_only
        && !req.fate_pretrain_adapt_only
    {
        return Err(
            "valhalla_base: questions must be non-empty (or set proactive_idle / fate_warmup_only / session_extend_only / fate_link_only / fate_split_callback_only / fate_pretrain_adapt_only)"
                .into(),
        );
    }

    // Interface absolute IDs → container address modes (Fate schedules inside).
    let algebra0 = req
        .session
        .as_ref()
        .map(crate::fate_adaptive_algebra::FateAlgebraState::from_session)
        .unwrap_or_else(crate::fate_adaptive_algebra::FateAlgebraState::ambient);
    let address_book = match &req.session {
        Some(s)
            if s.address_book
                .as_ref()
                .is_some_and(|b| b.len() == raw_lines.len() && !incremental) =>
        {
            s.address_book.clone().unwrap()
        }
        Some(s) if incremental && s.address_book.as_ref().is_some_and(|b| b.len() <= raw_lines.len()) => {
            let prev = s.address_book.as_ref().unwrap();
            let new_raw: Vec<String> = raw_lines.iter().skip(prev.len()).cloned().collect();
            if new_raw.is_empty() {
                prev.clone()
            } else {
                crate::valhalla_address::extend_address_book(prev, &new_raw, &algebra0)
            }
        }
        _ => crate::valhalla_address::convert_interface_corpus(&raw_lines, &algebra0),
    };
    // Containers ingest scrubbed bodies; absolute IDs live only in the address book.
    let lines = if address_book.len() == raw_lines.len() && !address_book.is_empty() {
        address_book.scrubbed_corpus()
    } else {
        raw_lines.clone()
    };
    let corpus_modality_mask = if merge_masks.len() == raw_lines.len() {
        merge_masks
    } else {
        build_corpus_modality_masks(&raw_lines, &items, req.session.as_ref())
    };

    if req.fate_pretrain_adapt_only {
        let prev = req
            .session
            .as_ref()
            .ok_or("valhalla_base: fate_pretrain_adapt_only requires session")?;
        let orch = fate_orchestrator_config(&req.config);
        let target = BodyTarget::parse(&req.config.patch_body);
        let explore = std::env::var("VALHALLA_FATE_PRETRAIN_EXPLORE")
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !(v == "0" || v == "false" || v == "off")
            })
            .unwrap_or(true);
        let (linked_snap, warmup, link_out) = crate::fate_orchestrator::fate_pretrain_adapt_snapshot(
            &prev.snapshot,
            &orch,
            target,
            explore,
        );
        let session = session_state_from_snap_with_book(
            &prev.corpus,
            linked_snap.clone(),
            prev.question_history.clone(),
            prev.fate_baseline_tick,
            prev.address_book.clone(),
            Some(prev.fate_algebra_meta.clone()),
            prev.corpus_modality_mask.clone(),
        );
        return Ok(ValhallaBaseResponse {
            protocol: VALHALLA_BASE_PROTOCOL.to_string(),
            context_lines: prev.corpus.len(),
            context_fingerprint: session.fingerprint.clone(),
            decode: req.config.decode,
            session,
            features: vec![
                "fate_pretrain_adapt_only".into(),
                "fate_fragment_link".into(),
                format!("warmup_cycles:{}", warmup.cycles_executed),
                format!("links_merged:{}", link_out.links_merged),
            ],
            rows: vec![],
            fate_telemetry: Some(fate_telemetry_from_snap(&linked_snap, &req.config)),
            fate_warmup: Some(FateWarmupSummary::from(warmup)),
            proactive: None,
        });
    }

    if req.fate_link_only {
        let prev = req
            .session
            .as_ref()
            .ok_or("valhalla_base: fate_link_only requires session")?;
        let orch = fate_orchestrator_config(&req.config);
        let target = BodyTarget::parse(&req.config.patch_body);
        let (linked_snap, link_out) =
            crate::fate_orchestrator::fate_link_snapshot(&prev.snapshot, &orch, target);
        let session = session_state_from_snap_with_book(
            &prev.corpus,
            linked_snap.clone(),
            prev.question_history.clone(),
            prev.fate_baseline_tick,
            prev.address_book.clone().or_else(|| Some(address_book.clone())),
            Some(prev.fate_algebra_meta.clone()),
            prev.corpus_modality_mask.clone(),
        );
        return Ok(ValhallaBaseResponse {
            protocol: VALHALLA_BASE_PROTOCOL.to_string(),
            context_lines: prev.corpus.len(),
            context_fingerprint: session.fingerprint.clone(),
            decode: req.config.decode,
            session,
            features: vec![
                "fate_link_only".into(),
                "fate_fragment_link".into(),
                format!("links_merged:{}", link_out.links_merged),
            ],
            rows: vec![],
            fate_telemetry: Some(fate_telemetry_from_snap(&linked_snap, &req.config)),
            fate_warmup: None,
            proactive: None,
        });
    }

    if req.fate_split_callback_only {
        let prev = req
            .session
            .as_ref()
            .ok_or("valhalla_base: fate_split_callback_only requires session")?;
        let orch = fate_orchestrator_config(&req.config);
        let target = BodyTarget::parse(&req.config.patch_body);
        let q = req.questions.first().map(|q| q.text.as_str());
        let (split_snap, split_out) = crate::fate_orchestrator::fate_split_snapshot(
            &prev.snapshot,
            &orch,
            target,
            &prev.corpus,
            q,
        );
        let session = session_state_from_snap_with_book(
            &prev.corpus,
            split_snap.clone(),
            prev.question_history.clone(),
            prev.fate_baseline_tick,
            prev.address_book.clone().or_else(|| Some(address_book.clone())),
            Some(prev.fate_algebra_meta.clone()),
            prev.corpus_modality_mask.clone(),
        );
        return Ok(ValhallaBaseResponse {
            protocol: VALHALLA_BASE_PROTOCOL.to_string(),
            context_lines: prev.corpus.len(),
            context_fingerprint: session.fingerprint.clone(),
            decode: req.config.decode,
            session,
            features: vec![
                "fate_split_callback_only".into(),
                "fate_fragment_split".into(),
                format!("splits_done:{}", split_out.splits_done),
                format!("callback_steps:{}", split_out.callback_steps),
                format!("rolled_back:{}", split_out.rolled_back),
            ],
            rows: vec![],
            fate_telemetry: Some(fate_telemetry_from_snap(&split_snap, &req.config)),
            fate_warmup: None,
            proactive: None,
        });
    }

    let (mut context_snap, fate_warmup) = resolve_snapshot(req, &lines, &items, incremental);
    let cfg = &req.config;

    if req.session_extend_only {
        if !incremental {
            return Err(
                "valhalla_base: session_extend_only requires session + append_lines/items (no full context)"
                    .into(),
            );
        }
        let session = session_state_from_snap_with_book(
            &lines,
            context_snap.clone(),
            req.session
                .as_ref()
                .map(|s| s.question_history.clone())
                .unwrap_or_default(),
            req.session
                .as_ref()
                .map(|s| s.fate_baseline_tick)
                .unwrap_or(context_snap.fate_signal_tick),
            Some(address_book.clone()),
            req.session.as_ref().map(|s| s.fate_algebra_meta.clone()),
            corpus_modality_mask.clone(),
        );
        return Ok(ValhallaBaseResponse {
            protocol: VALHALLA_BASE_PROTOCOL.to_string(),
            context_lines: lines.len(),
            context_fingerprint: session.fingerprint.clone(),
            decode: cfg.decode,
            session,
            features: vec!["context_flex".into(), "session_extend_only".into(), "session_append".into()],
            rows: vec![],
            fate_telemetry: Some(fate_telemetry_from_snap(&context_snap, cfg)),
            fate_warmup: fate_warmup.map(FateWarmupSummary::from),
            proactive: None,
        });
    }

    if req.fate_warmup_only {
        let session = session_state_from_snap_with_book(
            &lines,
            context_snap.clone(),
            req.session
                .as_ref()
                .map(|s| s.question_history.clone())
                .unwrap_or_default(),
            context_snap.fate_signal_tick,
            Some(address_book.clone()),
            req.session.as_ref().map(|s| s.fate_algebra_meta.clone()),
            corpus_modality_mask.clone(),
        );
        let mut features = vec!["context_flex".into(), "fate_warmup_only".into()];
        if fate_warmup.is_some() {
            features.push("fate_warmup".into());
        }
        return Ok(ValhallaBaseResponse {
            protocol: VALHALLA_BASE_PROTOCOL.to_string(),
            context_lines: lines.len(),
            context_fingerprint: session.fingerprint.clone(),
            decode: cfg.decode,
            session,
            features,
            rows: vec![],
            fate_telemetry: Some(fate_telemetry_from_snap(&context_snap, cfg)),
            fate_warmup: fate_warmup.map(FateWarmupSummary::from),
            proactive: None,
        });
    }

    let continuing_session_only = req.session.is_some()
        && req.context.is_empty()
        && req.context_items.is_empty()
        && req.append_lines.is_empty()
        && req.append_items.is_empty();
    let mut fate_baseline = req
        .session
        .as_ref()
        .map(|s| s.fate_baseline_tick)
        .unwrap_or(0);
    if !continuing_session_only || fate_baseline == 0 {
        fate_baseline = context_snap.fate_signal_tick;
    }
    let mut features = vec!["context_flex".into(), "address_container_v1".into()];
    if cfg.decode == ValhallaBaseDecode::Hybrid {
        features.push("hybrid_routing".into());
    }
    if cfg.mcq_decode == ValhallaMcqDecode::StructureFate {
        features.push("mcq_structure_fate".into());
    }
    if cfg.mcq_decode == ValhallaMcqDecode::MaxPick {
        features.push("mcq_max_pick".into());
    }
    if cfg.core_fusion {
        features.push("core_fusion".into());
    }
    features.push(format!(
        "answer_mix_{}",
        answer_mix_label(cfg.answer_mix)
    ));
    if cfg.answer_mix == ValhallaAnswerMix::Mixed {
        features.push("answer_mix_default_mixed".into());
    }
    if incremental {
        features.push("session_append".into());
    }
    match cfg.follow_up_decode {
        ValhallaFollowUpDecode::LmPatchOnFollowUp => features.push("follow_up_lm_patch".into()),
        ValhallaFollowUpDecode::NativeFollowUpAware => {
            features.push("follow_up_aware_retrieval".into())
        }
        ValhallaFollowUpDecode::Inherit => {}
    }
    match cfg.fate_weight_mode {
        ValhallaFateWeightMode::Evaluated => features.push("fate_weight_evaluated".into()),
        ValhallaFateWeightMode::Fixed => features.push("fate_weight_fixed".into()),
    }
    if cfg.fate_routing_free {
        features.push("fate_routing_free".into());
    }
    if cfg.fate_ingress_routing == ValhallaFateIngressRouting::Semantic {
        features.push("fate_ingress_semantic".into());
    }
    if cfg.fate_ingress_routing == ValhallaFateIngressRouting::QuadCycle {
        features.push("fate_ingress_quad_cycle".into());
    }
    if cfg.fate_ingress_routing == ValhallaFateIngressRouting::HybridQuadSemantic {
        features.push("fate_ingress_hybrid_quad_semantic".into());
    }
    if cfg.fate_qa_feedback {
        features.push("fate_qa_feedback".into());
    }
    if cfg.fate_warmup_enabled {
        features.push("fate_warmup".into());
    }
    if crate::fate_orchestrator::fragment_only_session_enabled() {
        features.push("fragment_only".into());
        features.push("dynamic_fragments".into());
    }
    if crate::fate_orchestrator::fate_link_fragments_enabled() {
        features.push("fate_fragment_link".into());
    }
    if crate::natural_channels::natural_channels_enabled() {
        features.push("natural_channels".into());
    }
    if cfg.proactive_enabled {
        features.push("proactive_idle".into());
    }
    if req.proactive_idle {
        features.push("proactive_idle_request".into());
    }
    if items.iter().any(|i| {
        i.audio_features.as_ref().is_some_and(|v| !v.is_empty())
            || i.image_features.as_ref().is_some_and(|v| !v.is_empty())
    }) {
        features.push("multimodal_context".into());
    }

    let mut prior: Vec<String> = cap_question_history(
        req.session
            .as_ref()
            .map(|s| s.question_history.clone())
            .unwrap_or_default(),
    );

    let mut proactive_outcome: Option<ProactiveOutcome> = None;
    let mut rows = Vec::new();

    if req.proactive_idle && req.questions.is_empty() {
        let (outcome, probe_rows) = run_proactive_idle(
            req,
            &lines,
            context_snap.clone(),
            &prior,
            fate_baseline,
            cfg,
        )?;
        proactive_outcome = Some(outcome);
        rows = probe_rows;
    } else {
        rows = Vec::with_capacity(req.questions.len());
        for q in &req.questions {
            // Question → container probes; Fate schedules Hub∩Tile∩Stem land into affinity.
            {
                let state = crate::fate_adaptive_algebra::FateAlgebraState::from_meta(
                    &crate::fate_adaptive_algebra::FateAlgebraMeta::default(),
                    Some(&context_snap),
                );
                let probe = crate::valhalla_address::convert_interface_question(q, &state);
                let frags = crate::fate_orchestrator::stored_fragments(&context_snap);
                let mut schedule = crate::valhalla_address::fate_schedule_affinity(
                    &address_book,
                    &probe,
                    &frags,
                    &state,
                );
                // Open QA: crush sticky schedule peaks that ignore query content.
                if q.score_type.eq_ignore_ascii_case("open") {
                    let tokens = crate::fate_adaptive_algebra::query_tokens_from_q(q);
                    let gated_lines = corpus_view_for_answer_mix(
                        &lines,
                        &corpus_modality_mask,
                        cfg.answer_mix,
                    );
                    schedule = crate::valhalla_address::content_gate_schedule(
                        &schedule, &gated_lines, &tokens,
                    );
                }
                crate::valhalla_address::apply_fate_schedule_to_affinity(
                    &mut context_snap.corpus_line_affinity,
                    &schedule,
                );
            }
            let mix_eligible = gate_affinity_by_answer_mix(
                &mut context_snap,
                &corpus_modality_mask,
                cfg.answer_mix,
            );
            let answer_lines =
                corpus_view_for_answer_mix(&lines, &corpus_modality_mask, cfg.answer_mix);
            let p: FairPrompt = q.clone().into();
            let is_follow_up = !prior.is_empty();
            let mut row = if p.score_type == "mcq" && cfg.mcq_decode == ValhallaMcqDecode::MaxPick {
                run_mcq_max_pick(&answer_lines, &p, q, cfg, context_snap.clone(), &prior)?
            } else {
                let mode = effective_decode_for_question(cfg, &p.score_type, is_follow_up);
                match mode {
                    ValhallaBaseDecode::Native => {
                        if cfg.decode == ValhallaBaseDecode::Hybrid && p.score_type == "open" {
                            run_open_decode_row(
                                &answer_lines,
                                &p,
                                q,
                                cfg,
                                context_snap.clone(),
                                &prior,
                            )?
                        } else {
                            run_native_one(
                                &answer_lines,
                                &p,
                                q,
                                cfg,
                                context_snap.clone(),
                                &prior,
                            )?
                        }
                    }
                    ValhallaBaseDecode::LmPatch => {
                        run_lm_one(&answer_lines, &p, q, cfg, context_snap.clone())?
                    }
                    ValhallaBaseDecode::Hybrid => unreachable!(),
                }
            };
            row.answer_mix = Some(answer_mix_label(cfg.answer_mix).into());
            row.answer_mix_eligible = Some(mix_eligible);
            prior.push(q.text.clone());
            rows.push(row.clone());
            let q_ports = grind_ports_for_line(&format_prompt_text(&p));
            context_snap = apply_qa_feedback_to_snapshot(
                context_snap,
                cfg,
                &q.text,
                fate_feedback_on_topic(&p, &row, &lines),
                &q_ports,
                row.retrieval_corpus_idx,
                Some(q_ports.embedding.clone()),
                row.retrieval_corpus_idx.and_then(|i| {
                    lines.get(i).map(|l| grind_ports_for_line(l).embedding)
                }),
                row.retrieval_corpus_idx
                    .and_then(|i| lines.get(i).map(|l| l.as_str())),
            );
        }
    }

    let snap = context_snap;
    let fate_tel = fate_telemetry_from_snap(&snap, cfg);
    let session = session_state_from_snap_with_book(
        &lines,
        snap,
        cap_question_history(prior),
        fate_baseline,
        Some(address_book),
        req.session.as_ref().map(|s| s.fate_algebra_meta.clone()),
        corpus_modality_mask,
    );

    Ok(ValhallaBaseResponse {
        protocol: VALHALLA_BASE_PROTOCOL.to_string(),
        context_lines: lines.len(),
        context_fingerprint: session.fingerprint.clone(),
        decode: cfg.decode,
        session,
        features,
        rows,
        fate_telemetry: Some(fate_tel),
        fate_warmup: fate_warmup.map(FateWarmupSummary::from),
        proactive: proactive_outcome,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_changes_with_context() {
        let a = context_fingerprint(&["alpha".into()]);
        let b = context_fingerprint(&["beta".into()]);
        assert_ne!(a, b);
    }

    #[test]
    fn hybrid_routes_mcq_to_lm_by_default() {
        let cfg = ValhallaBaseConfig {
            decode: ValhallaBaseDecode::Hybrid,
            mcq_decode: ValhallaMcqDecode::LmPatch,
            ..Default::default()
        };
        assert_eq!(
            effective_decode(&cfg, "mcq"),
            ValhallaBaseDecode::LmPatch
        );
        assert_eq!(
            effective_decode(&cfg, "open"),
            ValhallaBaseDecode::Native
        );
    }

    #[test]
    fn native_open_reject_detection() {
        assert!(native_open_answer_rejected(
            "No confident match in incubated memory.",
            "open_reject_low_score"
        ));
        assert!(native_open_answer_rejected(
            "Some text",
            "valhalla_base/native/open_fog_abstain"
        ));
        assert!(!native_open_answer_rejected(
            "[corpus:0] answer",
            "open_retrieval_span"
        ));
    }

    #[test]
    fn proactive_acronym_gdp_matches_gross_domestic_product() {
        let prior = vec![
            "What does GDP stand for?".into(),
            "Does the context say anything about gross?".into(),
        ];
        let line = "Gross Domestic Product sums consumption investment government and net exports.";
        assert!(proactive_acronym_matches(line, &prior) >= 1);
        assert!(proactive_overlap_count(line, &prior) >= 1);
        assert!(build_proactive_probe(&[line.into()], &prior).is_some());
    }

    #[test]
    fn append_lines_dedup_by_normalized_text() {
        let req = ValhallaBaseRequest {
            context: vec!["Alpha line".into()],
            append_lines: vec!["Alpha   line".into(), "Beta".into()],
            ..Default::default()
        };
        let (lines, _, _, _) = merge_context(&req);
        assert_eq!(lines.len(), 2);
        assert!(lines.iter().any(|l| l.starts_with("Beta")));
    }

    #[test]
    fn reanchor_carries_line_affinity_by_text() {
        let prev = ValhallaBaseSessionState {
            corpus: vec!["Paris is capital.".into(), "Other.".into()],
            fingerprint: "x".into(),
            lines_ingested: 2,
            fragments: vec![],
            snapshot: TriadIncubationSnapshot {
                corpus_line_affinity: vec![0.0, 0.42],
                completed_tiles: vec![],
                stem_cells: vec![],
                tile_total_operations: 0,
                hub_routed: [0; 4],
                ingress_encoder: "char".into(),
                hub_prefs: vec![],
                fate_weight_mode: "fixed".into(),
                fate_routing_free: false,
                fate_ingress_routing: "quad_cycle".into(),
                fate_signal_tick: 0,
                manifold_protocol: None,
                patch_vector: vec![],
                fate_convergence: None,
                hub_triple_fate: None,
                fragments: vec![],
            },
            fate_algebra_meta: Default::default(),
            fate_rollback_stack: Vec::new(),
            question_history: vec![],
            fate_baseline_tick: 0,
            address_book: None,
            corpus_modality_mask: vec![],
        };
        let snap = TriadIncubationSnapshot {
            completed_tiles: vec![],
            stem_cells: vec![],
            tile_total_operations: 0,
            hub_routed: [0; 4],
            ingress_encoder: "char".into(),
            hub_prefs: vec![],
            fate_weight_mode: "fixed".into(),
            fate_routing_free: false,
            fate_ingress_routing: "quad_cycle".into(),
            corpus_line_affinity: vec![],
            fate_signal_tick: 0,
            manifold_protocol: None,
            patch_vector: vec![],
            fate_convergence: None,
            hub_triple_fate: None,
            fragments: vec![],
        };
        let out = reanchor_snapshot_affinity(&prev, &["Other.".into()], snap);
        assert!((out.corpus_line_affinity[0] - 0.42).abs() < 1e-9);
    }

    #[test]
    fn answer_mix_gates_modality_bits() {
        let text = MOD_TEXT;
        let audio = MOD_TEXT | MOD_AUDIO;
        let image = MOD_TEXT | MOD_IMAGE;
        assert!(answer_mix_allows(text, ValhallaAnswerMix::Mixed));
        assert!(answer_mix_allows(audio, ValhallaAnswerMix::Mixed));
        assert!(answer_mix_allows(text, ValhallaAnswerMix::Text));
        assert!(!answer_mix_allows(audio, ValhallaAnswerMix::Text));
        assert!(answer_mix_allows(audio, ValhallaAnswerMix::Audio));
        assert!(!answer_mix_allows(text, ValhallaAnswerMix::Audio));
        assert!(answer_mix_allows(image, ValhallaAnswerMix::Image));
        assert!(!answer_mix_allows(audio, ValhallaAnswerMix::Image));
    }
}
