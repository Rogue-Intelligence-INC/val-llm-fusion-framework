//! Fate-orchestrated manifold runtime — fragment export/merge + adaptive cycles.
//!
//! Protocol: `valhalla-manifold-v1`
//! - Snapshot is **not** monolithic freeze; fragments compose under Fate scheduling.
//! - High transformation cycles are **Fate-owned** (convergence stop authority).

use serde::{Deserialize, Serialize};
use stem_cell::CompleteCell;
use valhalla_tile_quattro::CompletedTile;

use crate::dl_front::{BodyFrontPorts, DlFrontConfig, ModalityInput};
use crate::grinder::grind_text_hash;
use crate::signal_ingress::{
    build_model_patch, BodyTarget, IngressBudget, TriadIncubationSnapshot, TriadSession,
};
use crate::{FateConvergenceState, FateSlice, FateWeightMode, IngressRoutingMode, TripleFateCombined};

pub const MANIFOLD_PROTOCOL: &str = "valhalla-manifold-v1";
pub const FRAGMENT_LIBRARY_VERSION: &str = "valhalla-fragment-library-v1";

/// Metadata recorded when a dynamic density segment closes.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FragmentCloseMeta {
    #[serde(default)]
    pub line_start: usize,
    #[serde(default)]
    pub line_end: usize,
    #[serde(default)]
    pub signal_tick_start: u64,
    #[serde(default)]
    pub signal_tick_end: u64,
    /// Tile-clock window of this fragment (Tile's own global_tick
    /// domain). The Hub Fate signal tick and Tile's completed_tick
    /// advance at different rates, so windowing tiles by Hub ticks
    /// scoped fragments to the wrong material; segment closes record
    /// both clocks and export prefers this one when present.
    #[serde(default)]
    pub tile_tick_start: u64,
    #[serde(default)]
    pub tile_tick_end: u64,
    #[serde(default)]
    pub close_reason: String,
    #[serde(default)]
    pub marginal_density: f64,
    #[serde(default)]
    pub pref_output_rate: f64,
}

/// One composable slice of Triad structure + Fate state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotFragment {
    pub id: String,
    pub label: String,
    pub completed_tiles: Vec<CompletedTile>,
    pub stem_cells: Vec<CompleteCell>,
    /// Snapshot dedup: ids into the snapshot-level shared tables
    /// (`TriadIncubationSnapshot.completed_tiles` / `.stem_cells`).
    /// When non-empty, the payloads above are emptied at compact
    /// export and rehydrated on load (see `rehydrate_fragments`).
    /// Empty = legacy inline payload form.
    #[serde(default)]
    pub tile_ids: Vec<String>,
    #[serde(default)]
    pub cell_ids: Vec<String>,
    pub patch_vector: Vec<f64>,
    pub hub_prefs: Vec<f64>,
    pub hub_routed: [u64; 4],
    pub fate_convergence: FateConvergenceState,
    pub triple_fate_slice: FateSlice,
    #[serde(default)]
    pub corpus_line_affinity: Vec<f64>,
    /// Dynamic segment bounds (density mode).
    #[serde(default)]
    pub line_start: usize,
    #[serde(default)]
    pub line_end: usize,
    #[serde(default)]
    pub signal_tick_start: u64,
    #[serde(default)]
    pub signal_tick_end: u64,
    #[serde(default)]
    pub close_reason: String,
    #[serde(default)]
    pub marginal_density: f64,
    #[serde(default)]
    pub pref_output_rate: f64,
}

/// Fragment recording strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentMode {
    Off,
    /// One fragment per corpus line / modality item.
    Line,
    /// Fate signal-density drop segments (longer, adaptive).
    Density,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FateOrchestratorConfig {
    pub min_cycles: usize,
    pub soft_max_cycles: usize,
    pub hard_max_cycles: usize,
    /// During adapt phase, enable Fate preference routing + evaluated weights.
    pub fate_adapt_unlock: bool,
    /// Soft max for warmup phase (Fate convergence stops earlier when possible).
    pub warmup_cycles: usize,
    /// Prelearn line count (0 = auto from corpus size).
    pub warmup_prelearn_lines: usize,
    /// When false, skip prelearn + warmup (overrides env in unit tests).
    pub warmup_enabled: bool,
}

/// Per-phase cycle bounds — Fate convergence is stop authority within [min, soft_max], capped at hard_max.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FateCycleLimits {
    pub min_cycles: usize,
    pub soft_max_cycles: usize,
    pub hard_max_cycles: usize,
}

impl FateCycleLimits {
    pub fn adapt(cfg: &FateOrchestratorConfig) -> Self {
        Self {
            min_cycles: cfg.min_cycles,
            soft_max_cycles: cfg.soft_max_cycles,
            hard_max_cycles: cfg.hard_max_cycles,
        }
    }

    pub fn warmup(cfg: &FateOrchestratorConfig) -> Self {
        let soft = cfg.warmup_cycles.clamp(1, 64);
        Self {
            min_cycles: env_usize("VALHALLA_FATE_WARMUP_MIN", 1).clamp(1, 32),
            soft_max_cycles: soft,
            hard_max_cycles: env_usize("VALHALLA_FATE_WARMUP_HARD_MAX", soft.saturating_mul(2).max(8))
                .clamp(4, 128),
        }
    }

    pub fn pulse(_cfg: &FateOrchestratorConfig) -> Self {
        Self {
            min_cycles: env_usize("VALHALLA_FATE_PULSE_MIN", 1).clamp(1, 16),
            soft_max_cycles: env_usize("VALHALLA_FATE_PULSE_SOFT_MAX", 4).clamp(1, 32),
            hard_max_cycles: env_usize("VALHALLA_FATE_PULSE_HARD_MAX", 12).clamp(2, 64),
        }
    }
}

impl Default for FateOrchestratorConfig {
    fn default() -> Self {
        Self {
            min_cycles: env_usize("VALHALLA_FATE_ADAPT_MIN", 5).clamp(1, 64),
            soft_max_cycles: env_usize("VALHALLA_FATE_ADAPT_SOFT_MAX", 48).clamp(4, 512),
            hard_max_cycles: env_usize("VALHALLA_FATE_ADAPT_HARD_MAX", 128).clamp(8, 1024),
            fate_adapt_unlock: env_bool("VALHALLA_FATE_ADAPT_UNLOCK", true),
            warmup_cycles: env_usize("VALHALLA_FATE_WARMUP_CYCLES", 4).clamp(1, 64),
            warmup_prelearn_lines: env_usize("VALHALLA_FATE_WARMUP_PRELEARN_LINES", 0),
            warmup_enabled: env_bool("VALHALLA_FATE_WARMUP", true),
        }
    }
}

/// `VALHALLA_FATE_WARMUP=1` (default on) — prelearn lines + warmup cycles before dynamic fragments.
pub fn fate_warmup_enabled() -> bool {
    env_bool("VALHALLA_FATE_WARMUP", true)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FateWarmupOutcome {
    pub cycles_executed: usize,
    pub prelearn_lines: usize,
    pub pref_history_len: usize,
    pub signal_tick: u64,
    pub aggregate_score: f64,
    pub stopped_reason: String,
}

fn warmup_prelearn_count(corpus_lines: usize, cfg: &FateOrchestratorConfig) -> usize {
    if corpus_lines <= 1 {
        return 0;
    }
    let auto = (corpus_lines / 3).clamp(1, 4);
    let configured = if cfg.warmup_prelearn_lines > 0 {
        cfg.warmup_prelearn_lines
    } else {
        auto
    };
    configured.min(corpus_lines.saturating_sub(1))
}

/// Prelearn line count for long-chain Fate warmup (public — no test changes required).
pub fn production_prelearn_count(corpus_lines: usize) -> usize {
    warmup_prelearn_count(corpus_lines, &FateOrchestratorConfig::default())
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FateLinkOutcome {
    pub fragments_in: usize,
    pub fragments_out: usize,
    pub links_merged: usize,
    pub adapt_cycles: usize,
    pub aggregate_score: f64,
}

pub fn fate_link_fragments_enabled() -> bool {
    match std::env::var("VALHALLA_FATE_LINK_FRAGMENTS") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        Err(_) => true,
    }
}

fn fragment_link_score(a: &SnapshotFragment, b: &SnapshotFragment) -> f64 {
    if a.id == "merged" || b.id == "merged" || a.id == "density_default" || b.id == "density_default" {
        return 0.0;
    }
    let gap = b.line_start.saturating_sub(a.line_end);
    let contiguous = gap <= 1;
    if gap > 4 {
        return 0.0;
    }
    let tick_gap = b.signal_tick_start.saturating_sub(a.signal_tick_end);
    let tick_score = 1.0 / (1.0 + tick_gap as f64 * 0.02);
    let dens_a = a.marginal_density.max(1e-9);
    let dens_b = b.marginal_density.max(1e-9);
    let dens_ratio = dens_a.min(dens_b) / dens_a.max(dens_b);
    let line_score = if contiguous { 0.45 } else { 0.12 };
    (line_score + tick_score * 0.35 + dens_ratio * 0.2).clamp(0.0, 1.0)
}

fn link_threshold() -> f64 {
    env_f64("VALHALLA_FATE_LINK_THRESHOLD", 0.42)
}

fn fate_link_max_group_size() -> usize {
    env_usize("VALHALLA_FATE_LINK_MAX_GROUP", 2).clamp(2, 16)
}

pub fn fate_link_each_extend_enabled() -> bool {
    match std::env::var("VALHALLA_FATE_LINK_EACH_EXTEND") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    }
}

/// Fate links adjacent dynamic fragments into chain segments, then adapt.
pub fn fate_link_fragments(
    session: &mut TriadSession,
    cfg: &FateOrchestratorConfig,
) -> FateLinkOutcome {
    if !fate_link_fragments_enabled() {
        return FateLinkOutcome::default();
    }
    let tracks: Vec<SnapshotFragment> = session
        .fragment_tracks
        .iter()
        .filter(|f| is_mergeable_fragment(&f.id) || f.id.starts_with("seg_") || f.id.starts_with("link_"))
        .cloned()
        .collect();
    let tracks_len = tracks.len();
    if tracks.len() < 2 {
        return FateLinkOutcome {
            fragments_in: tracks_len,
            fragments_out: tracks_len,
            ..Default::default()
        };
    }

    let mut sorted = tracks;
    sorted.sort_by_key(|f| (f.line_start, f.signal_tick_start));

    let threshold = link_threshold();
    let max_group = fate_link_max_group_size();
    let mut groups: Vec<Vec<SnapshotFragment>> = Vec::new();
    let mut current = vec![sorted[0].clone()];
    for next in sorted.into_iter().skip(1) {
        let prev = current.last().unwrap();
        let score = fragment_link_score(prev, &next);
        if score >= threshold && current.len() < max_group {
            current.push(next);
        } else if score >= threshold && current.len() >= max_group {
            groups.push(current);
            current = vec![next];
        } else {
            groups.push(current);
            current = vec![next];
        }
    }
    groups.push(current);

    let budget = IngressBudget::experiment();
    let body = session.body;
    let mut merged_tracks: Vec<SnapshotFragment> = Vec::new();
    let mut links_merged = 0usize;
    for (i, group) in groups.iter().enumerate() {
        if group.len() == 1 {
            merged_tracks.push(group[0].clone());
            continue;
        }
        links_merged += group.len().saturating_sub(1);
        let linked = merge_fragments(group, budget.clone(), body);
        let line_start = group.first().map(|f| f.line_start).unwrap_or(0);
        let line_end = group.last().map(|f| f.line_end).unwrap_or(line_start);
        let meta = FragmentCloseMeta {
            line_start,
            line_end,
            close_reason: "fate_link_chain".into(),
            marginal_density: group.iter().map(|f| f.marginal_density).sum::<f64>() / group.len() as f64,
            ..Default::default()
        };
        merged_tracks.push(export_fragment(
            &linked,
            &format!("link_{i}"),
            &format!("fate_link:{line_start}..{line_end}"),
            meta,
        ));
    }

    let preserved: Vec<SnapshotFragment> = session
        .fragment_tracks
        .iter()
        .filter(|f| !is_mergeable_fragment(&f.id) && !f.id.starts_with("seg_") && !f.id.starts_with("link_"))
        .cloned()
        .collect();
    merged_tracks.extend(preserved);
    session.fragment_tracks = merged_tracks;

    let adapt = run_fate_adapt(session, cfg);
    FateLinkOutcome {
        fragments_in: tracks_len,
        fragments_out: session.fragment_tracks.len(),
        links_merged,
        adapt_cycles: adapt.cycles_executed,
        aggregate_score: adapt.aggregate_score,
    }
}

/// Rebuild session from snapshot fragments, link chains, adapt — long-chain finalize.
pub fn fate_link_snapshot(
    snap: &TriadIncubationSnapshot,
    cfg: &FateOrchestratorConfig,
    body: BodyTarget,
) -> (TriadIncubationSnapshot, FateLinkOutcome) {
    let frags = stored_fragments(snap);
    if frags.len() < 2 {
        return (snap.clone(), FateLinkOutcome::default());
    }
    let budget = IngressBudget::experiment();
    let mut session = session_from_stored_fragments(&frags, &snap.corpus_line_affinity, budget, body);
    TriadSession::apply_snapshot_fate_policy(&mut session, snap);
    let outcome = fate_link_fragments(&mut session, cfg);
    (export_session_after_ingest(&session), outcome)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FateSplitOutcome {
    pub splits_done: usize,
    pub callback_steps: usize,
    pub rolled_back: bool,
}

/// Split callback referenced by `valhalla_base`. This tree has the link path;
/// the split body is not present, so the snapshot is returned unchanged.
pub fn fate_split_snapshot(
    snap: &TriadIncubationSnapshot,
    _cfg: &FateOrchestratorConfig,
    _body: BodyTarget,
    _corpus: &[String],
    _question: Option<&str>,
) -> (TriadIncubationSnapshot, FateSplitOutcome) {
    (snap.clone(), FateSplitOutcome::default())
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FatePretrainPassOutcome {
    pub pass: usize,
    pub knowledge_lines: usize,
    pub warmup: FateWarmupOutcome,
    pub link: FateLinkOutcome,
    pub adapt: FateAdaptOutcome,
}

pub fn default_pretrain_passes() -> usize {
    env_usize("VALHALLA_FATE_PRETRAIN_PASSES", 3).clamp(1, 8)
}

/// Pulse prior knowledge + optional explore routing, warmup, link, adapt (one pretrain pass).
pub fn fate_pretrain_pass(
    session: &mut TriadSession,
    knowledge_lines: &[String],
    cfg: &FateOrchestratorConfig,
    explore_routing: bool,
    pass_idx: usize,
) -> FatePretrainPassOutcome {
    let cap = crate::grinder::default_port_cap();
    if explore_routing {
        session.hub.set_fate_policy(
            FateWeightMode::Evaluated,
            true,
            IngressRoutingMode::RoundRobin,
        );
    }
    for line in knowledge_lines {
        let ports = grind_text_hash(line, cap);
        pulse_ingest_line(session, &ports, Some(line.as_str()), cfg, true);
    }
    let warmup = run_fate_warmup(session, cfg, knowledge_lines.len());
    let link = fate_link_fragments(session, cfg);
    let adapt = run_fate_adapt(session, cfg);
    FatePretrainPassOutcome {
        pass: pass_idx,
        knowledge_lines: knowledge_lines.len(),
        warmup,
        link,
        adapt,
    }
}

/// Extra Fate warmup+adapt on persisted session (no new corpus lines).
pub fn fate_pretrain_adapt_snapshot(
    snap: &TriadIncubationSnapshot,
    cfg: &FateOrchestratorConfig,
    body: BodyTarget,
    explore_routing: bool,
) -> (TriadIncubationSnapshot, FateWarmupOutcome, FateLinkOutcome) {
    let frags = stored_fragments(snap);
    let budget = IngressBudget::experiment();
    let mut session = if frags.is_empty() {
        TriadSession::with_body(budget, body)
    } else {
        session_from_stored_fragments(&frags, &snap.corpus_line_affinity, budget, body)
    };
    TriadSession::apply_snapshot_fate_policy(&mut session, snap);
    if explore_routing {
        session.hub.set_fate_policy(
            FateWeightMode::Evaluated,
            true,
            IngressRoutingMode::RoundRobin,
        );
    }
    let warmup = run_fate_warmup(&mut session, cfg, 0);
    let link = fate_link_fragments(&mut session, cfg);
    let _adapt = run_fate_adapt(&mut session, cfg);
    (export_session_after_ingest(&session), warmup, link)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FateAdaptOutcome {
    pub cycles_executed: usize,
    pub stopped_reason: String,
    pub aggregate_score: f64,
    pub hub_converged: bool,
    pub tile_converged: bool,
    pub stem_converged: bool,
    pub hub_pattern: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PretrainStats {
    pub lines_fed: usize,
    pub modality_fragments: usize,
    pub adapt_cycles: usize,
    pub stopped_reason: String,
    pub aggregate_score: f64,
    #[serde(default)]
    pub warmup_cycles: usize,
    #[serde(default)]
    pub warmup_prelearn_lines: usize,
}

/// Offline / runtime fragment store — bake + session preload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifoldFragmentLibrary {
    pub version: String,
    pub protocol: String,
    pub created_at: String,
    pub fragments: Vec<SnapshotFragment>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merged_snapshot: Option<TriadIncubationSnapshot>,
    pub pretrain_stats: PretrainStats,
}

impl ManifoldFragmentLibrary {
    pub fn new(fragments: Vec<SnapshotFragment>, merged: TriadIncubationSnapshot, stats: PretrainStats) -> Self {
        Self {
            version: FRAGMENT_LIBRARY_VERSION.into(),
            protocol: MANIFOLD_PROTOCOL.into(),
            created_at: chrono::Utc::now().to_rfc3339(),
            fragments,
            merged_snapshot: Some(merged),
            pretrain_stats: stats,
        }
    }

    pub fn save_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    pub fn load_json(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn session_from_library(
        &self,
        budget: IngressBudget,
        body: BodyTarget,
    ) -> TriadSession {
        if let Some(ref snap) = self.merged_snapshot {
            TriadSession::from_incubation_snapshot(budget, body, snap.clone())
        } else if !self.fragments.is_empty() {
            merge_fragments(&self.fragments, budget, body)
        } else {
            TriadSession::with_body(budget, body)
        }
    }
}

pub fn multi_fragment_enabled() -> bool {
    !matches!(fragment_mode(), FragmentMode::Off)
}

/// Unified dynamic corpus: **full-line Fate ingest** + **query fragment merge**.
///
/// Replaces the legacy tail window (`VALHALLA_NATIVE_MAX_CORPUS_LINES`, default 32)
/// when active. Disable with `VALHALLA_DYNAMIC_CORPUS_FRAGMENTS=0` (falls back to
/// legacy line cap + optional `VALHALLA_QUERY_FRAGMENT_MERGE`).
pub fn dynamic_corpus_fragments_enabled() -> bool {
    match std::env::var("VALHALLA_DYNAMIC_CORPUS_FRAGMENTS") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        Err(_) => multi_fragment_enabled(),
    }
}

/// Lines to skip at corpus ingest (legacy tail-only window).
/// Returns `0` when dynamic corpus fragments or explicit full ingest is active.
pub fn corpus_ingest_skip(len: usize, explicit_ingest_all: bool) -> usize {
    if explicit_ingest_all || dynamic_corpus_fragments_enabled() {
        return 0;
    }
    let max_ingest = env_usize("VALHALLA_NATIVE_MAX_CORPUS_LINES", 0);
    if max_ingest == 0 {
        return 0;
    }
    len.saturating_sub(max_ingest)
}

/// Eval corpora line cap: `0` = unlimited (dynamic fragment full ingest).
/// Non-zero explicit only for ablation; legacy cap applies only when dynamic fragments off.
pub fn eval_corpus_line_cap(explicit: usize) -> usize {
    if explicit > 0 {
        return explicit;
    }
    if dynamic_corpus_fragments_enabled() {
        return 0;
    }
    env_usize("VALHALLA_NATIVE_MAX_CORPUS_LINES", 0)
}

/// Apply eval corpus cap — no-op when cap is 0 (algebra full-line ingest).
pub fn apply_eval_corpus_cap(corpus: Vec<String>, explicit: usize) -> Vec<String> {
    let cap = eval_corpus_line_cap(explicit);
    if cap == 0 || corpus.len() <= cap {
        corpus
    } else {
        corpus.into_iter().take(cap).collect()
    }
}

/// Chunk size for long-chain ingest: `explicit=0` derives from ingress budget + corpus length.
pub fn eval_chunk_size(corpus_len: usize, explicit: usize) -> usize {
    if explicit > 0 {
        return explicit.max(1).min(corpus_len.max(1));
    }
    let rp = env_usize("VALHALLA_INGRESS_RESOURCE_PERCENT", 70).clamp(1, 100);
    let base = (rp / 2).clamp(8, 128);
    base.min(corpus_len.max(1))
}

pub fn explicit_ingest_all_corpus(request_flag: Option<bool>) -> bool {
    request_flag.unwrap_or(false)
        || env_bool("VALHALLA_INGEST_ALL_CORPUS", false)
}

/// `VALHALLA_FRAGMENT_MODE=density|line|off` (default: density).
pub fn fragment_mode() -> FragmentMode {
    if std::env::var("VALHALLA_MULTI_FRAGMENT")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "0" || v == "false" || v == "off"
        })
        .unwrap_or(false)
    {
        return FragmentMode::Off;
    }
    match std::env::var("VALHALLA_FRAGMENT_MODE")
        .unwrap_or_else(|_| "density".into())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "line" | "per_line" | "corpus_line" => FragmentMode::Line,
        "off" | "none" => FragmentMode::Off,
        _ => FragmentMode::Density,
    }
}

pub fn live_session_enabled() -> bool {
    std::env::var("VALHALLA_LIVE_SESSION")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        })
        .unwrap_or(true)
}

fn truncate_label(s: &str, max: usize) -> String {
    let t = s.trim();
    let n = t.chars().count();
    if n <= max {
        t.to_string()
    } else {
        let take = max.saturating_sub(1);
        let shortened: String = t.chars().take(take).collect();
        format!("{shortened}…")
    }
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Debug, Clone)]
struct FateDensitySample {
    tick: u64,
    pref_hist_len: usize,
    total_system_counts: u64,
    pref_marginal: f64,
}

fn sample_fate_density(session: &TriadSession) -> FateDensitySample {
    let state = session.hub.fate_system.export_convergence_state();
    let pref_marginal = if state.preference_history.len() >= 2 {
        let a = state.preference_history[state.preference_history.len() - 1];
        let b = state.preference_history[state.preference_history.len() - 2];
        (0..4).map(|i| (a[i] - b[i]).abs()).sum()
    } else {
        1.0
    };
    FateDensitySample {
        tick: state.signal_tick,
        pref_hist_len: state.preference_history.len(),
        total_system_counts: state.system_counts.iter().sum(),
        pref_marginal,
    }
}

/// Segments corpus by Fate marginal signal density / preference output rate drops.
#[derive(Debug, Clone, Default)]
pub struct FateDensitySegmenter {
    segment_index: usize,
    segment_start_line: usize,
    line_labels: Vec<String>,
    last_sample: Option<FateDensitySample>,
    open_tick_start: u64,
    open_tile_tick: u64,
    prev_marginal_density: f64,
    low_density_streak: usize,
    low_pref_streak: usize,
}

impl FateDensitySegmenter {
    pub fn new() -> Self {
        Self::default()
    }

    fn min_lines(&self) -> usize {
        env_usize("VALHALLA_FRAGMENT_MIN_LINES", 2).clamp(1, 256)
    }

    fn max_lines(&self) -> usize {
        env_usize("VALHALLA_FRAGMENT_MAX_LINES", 48).clamp(2, 512)
    }

    fn chunk_stride(&self) -> usize {
        env_usize("VALHALLA_FRAGMENT_CHUNK_STRIDE", 8).clamp(2, 64)
    }

    pub fn start_next_segment(&mut self, next_line: usize) {
        self.segment_index += 1;
        self.segment_start_line = next_line;
        self.line_labels.clear();
        self.last_sample = None;
        self.open_tick_start = 0;
        self.open_tile_tick = 0;
        self.prev_marginal_density = 0.0;
        self.low_density_streak = 0;
        self.low_pref_streak = 0;
    }

    fn segment_label(&self) -> (String, String) {
        let id = format!("seg_{}", self.segment_index);
        let joined = self.line_labels.join(" | ");
        let label = if joined.len() > 160 {
            format!(
                "lines {}-{}: {}…",
                self.segment_start_line,
                self.segment_start_line + self.line_labels.len().saturating_sub(1),
                truncate_label(&joined, 120)
            )
        } else if self.line_labels.len() > 1 {
            format!(
                "lines {}-{}: {}",
                self.segment_start_line,
                self.segment_start_line + self.line_labels.len().saturating_sub(1),
                joined
            )
        } else {
            joined
        };
        (id, label)
    }

    fn build_close_meta(
        &self,
        line_end: usize,
        tick_end: u64,
        tile_tick_end: u64,
        marginal_density: f64,
        pref_output_rate: f64,
        reason: &str,
    ) -> FragmentCloseMeta {
        FragmentCloseMeta {
            line_start: self.segment_start_line,
            line_end,
            signal_tick_start: self.open_tick_start,
            signal_tick_end: tick_end,
            tile_tick_start: self.open_tile_tick,
            tile_tick_end,
            close_reason: reason.into(),
            marginal_density,
            pref_output_rate,
        }
    }

    /// Observe Fate after one ingest pulse; returns close metadata when density drops.
    pub fn observe(
        &mut self,
        session: &TriadSession,
        line_idx: usize,
        line_label: String,
    ) -> Option<FragmentCloseMeta> {
        self.line_labels.push(line_label);
        let cur = sample_fate_density(session);
        let lines_in_seg = line_idx + 1 - self.segment_start_line;
        let max_lines = self.max_lines();

        if self.last_sample.is_none() {
            self.open_tick_start = cur.tick;
            self.open_tile_tick = session.tile.global_tick;
            self.last_sample = Some(cur.clone());
            if lines_in_seg >= max_lines {
                return Some(self.build_close_meta(
                    line_idx,
                    cur.tick,
                    session.tile.global_tick,
                    0.0,
                    0.0,
                    "max_lines",
                ));
            }
            return None;
        }

        let prev = self.last_sample.as_ref().expect("sample");
        let dt = cur.tick.saturating_sub(prev.tick).max(1);
        let dc = cur
            .total_system_counts
            .saturating_sub(prev.total_system_counts);
        let marginal_density = dc as f64 / dt as f64;
        let pref_output_rate =
            (cur.pref_hist_len.saturating_sub(prev.pref_hist_len)) as f64 / dt as f64;

        let drop_ratio = env_f64("VALHALLA_FRAGMENT_DENSITY_DROP", 0.35).clamp(0.05, 0.95);
        let pref_eps = env_f64("VALHALLA_FRAGMENT_PREF_EPS", 0.02);

        let density_dropped = self.prev_marginal_density > 1e-9
            && marginal_density < self.prev_marginal_density * (1.0 - drop_ratio);

        if marginal_density + 1e-12 < self.prev_marginal_density * 0.9 {
            self.low_density_streak += 1;
        } else {
            self.low_density_streak = 0;
        }

        if cur.pref_marginal < pref_eps {
            self.low_pref_streak += 1;
        } else {
            self.low_pref_streak = 0;
        }

        self.prev_marginal_density = marginal_density;
        self.last_sample = Some(cur.clone());

        let min_lines = self.min_lines();
        let pref_rate_floor = env_f64("VALHALLA_FRAGMENT_PREF_RATE_FLOOR", 1e-4);
        let stride = self.chunk_stride();
        let stride_close = lines_in_seg >= min_lines
            && stride > 0
            && lines_in_seg % stride == 0;

        let should_close = lines_in_seg >= min_lines
            && ((density_dropped && self.low_density_streak >= 1)
                || self.low_pref_streak >= 2
                || (pref_output_rate < pref_rate_floor && lines_in_seg >= min_lines)
                || lines_in_seg >= max_lines
                || stride_close);

        if should_close {
            let reason = if stride_close && lines_in_seg < max_lines && !density_dropped {
                "chunk_stride"
            } else if lines_in_seg >= max_lines {
                "max_lines"
            } else if density_dropped {
                "density_drop"
            } else if self.low_pref_streak >= 2 {
                "pref_marginal_exhaust"
            } else {
                "pref_rate_drop"
            };
            return Some(self.build_close_meta(
                line_idx,
                cur.tick,
                session.tile.global_tick,
                marginal_density,
                pref_output_rate,
                reason,
            ));
        }
        None
    }

    /// Close trailing segment at corpus end.
    pub fn flush(&mut self, session: &TriadSession, last_line_idx: usize) -> Option<(String, String, FragmentCloseMeta)> {
        if self.line_labels.is_empty() {
            return None;
        }
        let cur = sample_fate_density(session);
        let meta = self.build_close_meta(
            last_line_idx,
            cur.tick,
            session.tile.global_tick,
            self.prev_marginal_density,
            0.0,
            "corpus_end",
        );
        let (id, label) = self.segment_label();
        Some((id, label, meta))
    }
}

/// Fate-controlled fragmentation scope for ingest convergence pulses.
/// Legacy (default): full-manifold convergence cycles after every line.
/// `VALHALLA_FATE_PULSE_SCOPE=segment`: lines only emit; convergence
/// runs when Fate's density segmenter closes a fragment (plus the
/// chunk-end adapt) — no per-line full-volume re-convergence.
pub fn fate_pulse_segment_scoped() -> bool {
    match std::env::var("VALHALLA_FATE_PULSE_SCOPE") {
        Ok(v) => v.trim().eq_ignore_ascii_case("segment"),
        Err(_) => false,
    }
}

fn pulse_ingest_line(
    session: &mut TriadSession,
    ports: &BodyFrontPorts,
    line: Option<&str>,
    cfg: &FateOrchestratorConfig,
    converge: bool,
) {
    session.emit_body_ports_with_line(ports, line);
    if converge {
        let _ = run_fate_convergence_cycles(session, &FateCycleLimits::pulse(cfg));
    }
    if session.hub.is_data_flowing() {
        session.hub.mark_epoch_complete();
    }
}

/// Ingest one corpus line, Fate-convergence pulse, record per-line fragment track.
pub fn ingest_line_record_fragment(
    session: &mut TriadSession,
    ports: &BodyFrontPorts,
    line: Option<&str>,
    idx: usize,
    cfg: &FateOrchestratorConfig,
) {
    pulse_ingest_line(session, ports, line, cfg, true);
    let label = line
        .map(|s| truncate_label(s, 96))
        .unwrap_or_else(|| format!("corpus_line_{idx}"));
    session.record_fragment_with_meta(&format!("corpus_{idx}"), &label, FragmentCloseMeta::default());
}

/// Density-aware ingest — closes segment when Fate output density drops.
pub fn ingest_line_density_fragment(
    session: &mut TriadSession,
    segmenter: &mut FateDensitySegmenter,
    ports: &BodyFrontPorts,
    line: Option<&str>,
    idx: usize,
    cfg: &FateOrchestratorConfig,
) {
    let scoped = fate_pulse_segment_scoped();
    pulse_ingest_line(session, ports, line, cfg, !scoped);
    let label = line
        .map(|s| truncate_label(s, 96))
        .unwrap_or_else(|| format!("corpus_line_{idx}"));
    if let Some(meta) = segmenter.observe(session, idx, label) {
        if scoped {
            // Fate-controlled fragmentation: settle the manifold once
            // per closed fragment, not once per line.
            let _ = run_fate_convergence_cycles(session, &FateCycleLimits::pulse(cfg));
            if session.hub.is_data_flowing() {
                session.hub.mark_epoch_complete();
            }
        }
        let (id, seg_label) = segmenter.segment_label();
        session.record_fragment_with_meta(&id, &seg_label, meta);
        segmenter.start_next_segment(idx + 1);
    }
}

/// Ingest multimodal context item as its own fragment track.
pub fn ingest_modality_record_fragment(
    session: &mut TriadSession,
    item: &ModalityInput,
    front: &DlFrontConfig,
    idx: usize,
    cfg: &FateOrchestratorConfig,
) {
    let _ = session.feed_modality(item.clone(), front);
    let _ = run_fate_convergence_cycles(session, &FateCycleLimits::pulse(cfg));
    if multi_fragment_enabled() && fragment_mode() == FragmentMode::Line {
        let kind = match item {
            ModalityInput::Text { .. } => "text",
            ModalityInput::AudioFeatures { .. } => "audio",
            ModalityInput::ImageFeatures { .. } => "image",
            ModalityInput::ImagePath { .. } => "image_path",
            ModalityInput::Embedding { .. } => "embedding",
        };
        session.record_fragment_with_meta(
            &format!("modality_{kind}_{idx}"),
            &format!("{kind}_item_{idx}"),
            FragmentCloseMeta::default(),
        );
    }
}

/// Corpus ingress with Fate-density or per-line fragments + final Fate adapt.
pub fn ingest_corpus_with_fragments(
    session: &mut TriadSession,
    lines: &[String],
    ports: &[BodyFrontPorts],
    cfg: &FateOrchestratorConfig,
) -> FateAdaptOutcome {
    ingest_corpus_with_fragments_mode(session, lines, ports, cfg, fragment_mode())
}

pub fn ingest_corpus_with_fragments_mode(
    session: &mut TriadSession,
    lines: &[String],
    ports: &[BodyFrontPorts],
    cfg: &FateOrchestratorConfig,
    mode: FragmentMode,
) -> FateAdaptOutcome {
    let (outcome, _) = ingest_corpus_with_fragments_mode_full(session, lines, ports, cfg, mode);
    outcome
}

/// Corpus ingress with optional Fate prelearn + warmup before fragment recording.
pub fn ingest_corpus_with_fragments_mode_full(
    session: &mut TriadSession,
    lines: &[String],
    ports: &[BodyFrontPorts],
    cfg: &FateOrchestratorConfig,
    mode: FragmentMode,
) -> (FateAdaptOutcome, Option<FateWarmupOutcome>) {
    let n = lines.len().min(ports.len());
    let prelearn_n = if cfg.warmup_enabled && fate_warmup_enabled() && mode != FragmentMode::Off {
        warmup_prelearn_count(n, cfg)
    } else {
        0
    };

    let mut warmup_outcome = None;
    let tick_head_start = sample_fate_density(session).tick;
    let tile_head_start = session.tile.global_tick;

    // Phase 1: prelearn — pulse Fate, no fragment tracks yet (the head
    // lines are closed into the first block in Phase 3 below).
    for i in 0..prelearn_n {
        let line = lines[i].trim();
        let line_ref = if line.is_empty() { None } else { Some(line) };
        pulse_ingest_line(session, &ports[i], line_ref, cfg, true);
    }

    // Phase 2: Fate warmup cycles before dynamic segmentation.
    if prelearn_n > 0 {
        warmup_outcome = Some(run_fate_warmup(session, cfg, prelearn_n));
    }
    let tick_head_end = sample_fate_density(session).tick;

    // Phase 3: remainder with fragment recording.
    let mut segmenter = FateDensitySegmenter::new();
    // Block identity must survive appends: seed the segment counter
    // from fragments the session already holds, so seg ids never repeat.
    segmenter.segment_index = session.fragment_tracks.len();
    if prelearn_n > 0 && mode == FragmentMode::Density {
        // Architectural closure (owner design): the prelearned head
        // lines are ingested material like any other — bundle them
        // into the first block instead of leaving them outside every
        // block. Material outside a block is not graspable substrate,
        // so a head needle could never pass a block-local reading.
        for i in 0..prelearn_n {
            let line = lines[i].trim();
            segmenter.line_labels.push(if line.is_empty() {
                format!("corpus_line_{i}")
            } else {
                truncate_label(line, 96)
            });
        }
        segmenter.open_tick_start = tick_head_start;
        segmenter.open_tile_tick = tile_head_start;
        let head_meta = segmenter.build_close_meta(
            prelearn_n - 1,
            tick_head_end,
            session.tile.global_tick,
            0.0,
            0.0,
            "prelearn_head",
        );
        let (head_id, head_label) = segmenter.segment_label();
        session.record_fragment_with_meta(&head_id, &head_label, head_meta);
        segmenter.start_next_segment(prelearn_n);
    }
    for i in prelearn_n..n {
        let line = lines[i].trim();
        let line_ref = if line.is_empty() { None } else { Some(line) };
        match mode {
            FragmentMode::Density => {
                ingest_line_density_fragment(session, &mut segmenter, &ports[i], line_ref, i, cfg);
            }
            FragmentMode::Line => {
                ingest_line_record_fragment(session, &ports[i], line_ref, i, cfg);
            }
            FragmentMode::Off => {
                pulse_ingest_line(session, &ports[i], line_ref, cfg, !fate_pulse_segment_scoped());
            }
        }
    }
    if mode == FragmentMode::Density && n > prelearn_n {
        if let Some((id, label, meta)) = segmenter.flush(session, n - 1) {
            session.record_fragment_with_meta(&id, &label, meta);
        }
    }

    let adapt = run_fate_adapt(session, cfg);
    (adapt, warmup_outcome)
}

/// Incremental corpus append on live session — dynamic density fragments, no prelearn/warmup.
pub fn append_corpus_with_fragments_mode(
    session: &mut TriadSession,
    lines: &[String],
    ports: &[BodyFrontPorts],
    cfg: &FateOrchestratorConfig,
    start_idx: usize,
    mode: FragmentMode,
) -> FateAdaptOutcome {
    let n = lines.len().min(ports.len());
    let mut segmenter = FateDensitySegmenter::new();
    // Block identity must survive appends (see ingest path).
    segmenter.segment_index = session.fragment_tracks.len();
    if mode == FragmentMode::Density && n > 0 {
        segmenter.start_next_segment(start_idx);
    }
    for i in 0..n {
        let line = lines[i].trim();
        let line_ref = if line.is_empty() { None } else { Some(line) };
        let idx = start_idx + i;
        match mode {
            FragmentMode::Density => {
                ingest_line_density_fragment(session, &mut segmenter, &ports[i], line_ref, idx, cfg);
            }
            FragmentMode::Line => {
                ingest_line_record_fragment(session, &ports[i], line_ref, idx, cfg);
            }
            FragmentMode::Off => {
                pulse_ingest_line(session, &ports[i], line_ref, cfg, !fate_pulse_segment_scoped());
            }
        }
    }
    if mode == FragmentMode::Density && n > 0 {
        if let Some((id, label, meta)) = segmenter.flush(session, start_idx + n - 1) {
            session.record_fragment_with_meta(&id, &label, meta);
        }
    }
    run_fate_adapt(session, cfg)
}

/// Production fragment mode — density dynamic segments unless explicitly disabled via env.
pub fn production_fragment_mode() -> FragmentMode {
    fragment_mode()
}

/// Live QA pulse — question on persistent session (no fork).
pub fn pulse_question_live(
    session: &mut TriadSession,
    question_ports: &BodyFrontPorts,
    qa_cycles: usize,
    audio_features: Option<&[f64]>,
    front: &DlFrontConfig,
) {
    session.emit_body_ports(question_ports);
    crate::multimodal::feed_audio(session, audio_features, front);
    session.run_optimize_cycles_fate_owned(qa_cycles.max(1));
    if matches!(session.body, BodyTarget::Hub | BodyTarget::Triad) {
        session.hub.mark_epoch_complete();
    }
}

/// Offline pretrain: corpus → fragments → Fate adapt → fragment library.
pub fn pretrain_corpus_to_library(
    body: BodyTarget,
    lines: &[String],
    cfg: &FateOrchestratorConfig,
) -> ManifoldFragmentLibrary {
    let budget = crate::signal_ingress::IngressBudget::incubation_with_env();
    let mut session = TriadSession::with_body(budget, body);
    let cap = crate::grinder::default_port_cap();
    let ports: Vec<BodyFrontPorts> = lines
        .iter()
        .map(|l| grind_text_hash(l, cap))
        .collect();
    let (outcome, warmup) =
        ingest_corpus_with_fragments_mode_full(&mut session, lines, &ports, cfg, fragment_mode());
    let snap = session.manifold_snapshot();
    let stats = PretrainStats {
        lines_fed: lines.len(),
        modality_fragments: snap
            .fragments
            .iter()
            .filter(|f| f.id.starts_with("modality_"))
            .count(),
        adapt_cycles: outcome.cycles_executed,
        stopped_reason: outcome.stopped_reason,
        aggregate_score: outcome.aggregate_score,
        warmup_cycles: warmup.as_ref().map(|w| w.cycles_executed).unwrap_or(0),
        warmup_prelearn_lines: warmup.as_ref().map(|w| w.prelearn_lines).unwrap_or(0),
    };
    ManifoldFragmentLibrary::new(snap.fragments.clone(), snap, stats)
}

struct FateTracker {
    last_score: f64,
    stable_streak: usize,
}

impl FateTracker {
    fn new() -> Self {
        Self {
            last_score: 0.0,
            stable_streak: 0,
        }
    }

    fn record(&mut self, score: f64) {
        let delta = (score - self.last_score).abs();
        if delta < 0.02 {
            self.stable_streak += 1;
        } else {
            self.stable_streak = 0;
        }
        self.last_score = score;
    }

    fn still_exploring(&self) -> bool {
        self.stable_streak == 0 && self.last_score > 0.05
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        })
        .unwrap_or(default)
}

fn compute_aggregate_score(session: &TriadSession) -> f64 {
    let prefs = session.hub.get_fate_preferences().affinities;
    let pref_spread = if prefs.is_empty() {
        0.0
    } else {
        let sum: f64 = prefs.iter().sum::<f64>().max(1e-9);
        prefs.iter().cloned().fold(0.0_f64, f64::max) / sum
    };
    let tile_energy: f64 = session
        .tile
        .completed_tiles
        .iter()
        .map(|t| t.activation)
        .sum();
    let activation_conc = if tile_energy > 0.0 {
        session
            .tile
            .completed_tiles
            .iter()
            .map(|t| t.activation)
            .fold(0.0_f64, f64::max)
            / tile_energy
    } else {
        0.0
    };
    let avg_conn = if session.stem.cells.is_empty() {
        0.0
    } else {
        session
            .stem
            .cells
            .iter()
            .map(|c| c.connections.len() as f64)
            .sum::<f64>()
            / session.stem.cells.len() as f64
    };
    let stem_cohesion = (avg_conn / 8.0).min(1.0);
    (0.4 * pref_spread + 0.35 * activation_conc + 0.25 * stem_cohesion).clamp(0.0, 1.0)
}

fn fate_should_stop(
    session: &TriadSession,
    tracker: &FateTracker,
    cycle: usize,
    limits: &FateCycleLimits,
    aggregate_score: f64,
) -> Option<&'static str> {
    if cycle < limits.min_cycles {
        return None;
    }
    let exploring = tracker.still_exploring();
    if exploring && cycle < limits.soft_max_cycles {
        return None;
    }
    let pattern = session.hub.get_fate_convergence_pattern();
    let any_converged = session.hub.has_fate_converged()
        || session.tile.has_fate_converged()
        || session.stem.has_fate_converged();

    if matches!(pattern, Some("fixed_point")) && !exploring {
        return Some("fate_fixed_point");
    }
    if matches!(pattern, Some("converging")) && any_converged && !exploring {
        return Some("fate_converged");
    }
    if tracker.stable_streak >= 3 && !exploring {
        return Some("fate_marginal_stable");
    }
    if aggregate_score >= 0.85 && !exploring {
        return Some("aggregate_ready");
    }
    None
}

/// Export one fragment from live session (shard / modality track / corpus slice).
pub fn export_fragment(
    session: &TriadSession,
    id: &str,
    label: &str,
    meta: FragmentCloseMeta,
) -> SnapshotFragment {
    let prefs = session.hub.get_fate_preferences();
    let patch = build_model_patch(
        session.body,
        &prefs.affinities,
        session.hub_routed_snapshot(),
        &session.tile,
        &session.stem,
        &session.stats,
    );
    // Dynamic unload: a fragment carries only the tiles that closed
    // inside its own segment window, not a clone of the whole body's
    // tiles at close time (which made every new fragment heavier than
    // the last and defeated fragment-only persistence on the wire).
    // Falls back to the full set when the window matches nothing
    // (e.g. default fragments with no tick window recorded).
    // Prefer the Tile-clock window when the segment recorded one:
    // Hub Fate ticks and Tile completed_ticks are different clocks
    // advancing at different rates, and windowing by the Hub clock
    // scoped fragments to stale material. Legacy metas without a
    // Tile window keep the Hub-tick behavior.
    let use_tile_window = meta.tile_tick_end > meta.tile_tick_start;
    let windowed_tiles: Vec<_> = session
        .tile
        .completed_tiles
        .iter()
        .filter(|t| {
            if use_tile_window {
                t.completed_tick >= meta.tile_tick_start
                    && t.completed_tick <= meta.tile_tick_end
            } else {
                meta.signal_tick_end > meta.signal_tick_start
                    && t.completed_tick >= meta.signal_tick_start
                    && t.completed_tick <= meta.signal_tick_end
            }
        })
        .cloned()
        .collect();
    let completed_tiles = if windowed_tiles.is_empty() {
        session.tile.completed_tiles.clone()
    } else {
        windowed_tiles
    };
    SnapshotFragment {
        id: id.into(),
        label: label.into(),
        completed_tiles,
        tile_ids: Vec::new(),
        cell_ids: Vec::new(),
        stem_cells: session.stem.cells.clone(),
        patch_vector: patch.patch_vector,
        hub_prefs: prefs.affinities,
        hub_routed: session.hub_routed_snapshot(),
        fate_convergence: session.hub.fate_system.export_convergence_state(),
        triple_fate_slice: session.hub.triple_fate.freeze(),
        corpus_line_affinity: session.corpus_line_affinity_snapshot(),
        line_start: meta.line_start,
        line_end: meta.line_end,
        signal_tick_start: meta.signal_tick_start,
        signal_tick_end: meta.signal_tick_end,
        close_reason: meta.close_reason,
        marginal_density: meta.marginal_density,
        pref_output_rate: meta.pref_output_rate,
    }
}

/// Fate-scheduled merge of fragments → new session substrate.
pub fn merge_fragments(
    fragments: &[SnapshotFragment],
    budget: crate::signal_ingress::IngressBudget,
    body: BodyTarget,
) -> TriadSession {
    assert!(!fragments.is_empty(), "merge_fragments requires ≥1 fragment");
    let mut session = TriadSession::with_body(budget, body);

    let mut all_tiles: Vec<CompletedTile> = Vec::new();
    let mut all_stem: Vec<CompleteCell> = Vec::new();
    let mut hub_routed = [0u64; 4];
    let mut pref_num = [0.0f64; 4];
    let mut pref_den = [0.0f64; 4];
    let mut affinity: Vec<f64> = Vec::new();
    let mut best_fate = FateConvergenceState::default();
    let mut best_hist_len = 0usize;
    let mut merged_triple = fragments[0].triple_fate_slice.clone();

    // Hash dedup (first occurrence wins, same as the previous linear
    // scans): merging all fragments must stay linear in total tiles,
    // not fragments x tiles.
    let mut seen_tiles: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut seen_cells: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for frag in fragments {
        for t in &frag.completed_tiles {
            if seen_tiles.insert(t.id.as_str()) {
                all_tiles.push(t.clone());
            }
        }
        for cell in &frag.stem_cells {
            if seen_cells.insert(cell.cell_id.as_str()) {
                all_stem.push(cell.clone());
            }
        }
        for i in 0..4 {
            hub_routed[i] = hub_routed[i].saturating_add(frag.hub_routed[i]);
            let w = frag.hub_routed[i].max(1) as f64;
            pref_num[i] += frag.hub_prefs.get(i).copied().unwrap_or(0.25) * w;
            pref_den[i] += w;
        }
        for (i, &v) in frag.corpus_line_affinity.iter().enumerate() {
            if i >= affinity.len() {
                affinity.resize(i + 1, 0.0);
            }
            affinity[i] = affinity[i].max(v);
        }
        let hlen = frag.fate_convergence.preference_history.len();
        if hlen >= best_hist_len {
            best_hist_len = hlen;
            best_fate = frag.fate_convergence.clone();
        }
        for (k, v) in &frag.triple_fate_slice.data_store {
            merged_triple.data_store.insert(k.clone(), v.clone());
        }
    }

    let hub_prefs: Vec<f64> = (0..4)
        .map(|i| {
            if pref_den[i] > 0.0 {
                pref_num[i] / pref_den[i]
            } else {
                0.25
            }
        })
        .collect();

    session.tile.completed_tiles = all_tiles;
    session.stem.cells = all_stem;
    session.restore_hub_routed(hub_routed);
    session.restore_corpus_line_affinity(affinity);
    session.hub.fate_system.import_convergence_state(&best_fate);
    session.hub.triple_fate = TripleFateCombined::thaw(merged_triple);
    session
        .hub
        .seed_fate_from_snapshot(&hub_prefs, hub_routed, Some(best_fate.signal_tick));

    session
}

fn signal_cosine(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let n = a.len().min(b.len());
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn keyword_overlap_score(question: &str, text: &str) -> f64 {
    use std::collections::HashSet;
    let q_words: HashSet<String> = question
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .map(|w| w.to_string())
        .collect();
    if q_words.is_empty() {
        return 0.0;
    }
    let t_words: HashSet<String> = text
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .map(|w| w.to_string())
        .collect();
    let overlap = q_words.intersection(&t_words).count();
    overlap as f64 / q_words.len() as f64
}

/// Query-conditioned fragment pick — part of [`dynamic_corpus_fragments_enabled`].
pub fn query_fragment_merge_enabled() -> bool {
    if dynamic_corpus_fragments_enabled() {
        return true;
    }
    env_bool("VALHALLA_QUERY_FRAGMENT_MERGE", false)
}

/// Session persistence uses dynamic fragments only — no monolithic Tile/Stem snapshot blob.
pub fn fragment_only_session_enabled() -> bool {
    match std::env::var("VALHALLA_FRAGMENT_ONLY") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        Err(_) => dynamic_corpus_fragments_enabled(),
    }
}

pub fn mergeable_session_fragments(session: &TriadSession) -> Vec<SnapshotFragment> {
    if session.fragment_tracks.is_empty() {
        return vec![export_fragment(
            session,
            "density_default",
            "default",
            FragmentCloseMeta::default(),
        )];
    }
    session
        .fragment_tracks
        .iter()
        .filter(|f| is_mergeable_fragment(&f.id))
        .cloned()
        .collect()
}

/// Compact session export — fragments + Fate metadata; no monolithic tile/stem freeze.
pub fn compact_manifold_snapshot(session: &TriadSession) -> TriadIncubationSnapshot {
    let mut snap = session.manifold_snapshot();
    snap.tile_total_operations = 0;
    snap.patch_vector = Vec::new();
    let mut fragments = mergeable_session_fragments(session);
    // Snapshot dedup (same discipline as the wire dedupe): the tile /
    // stem-cell tables are stored ONCE at snapshot level and each
    // fragment keeps only the ids of its entries. Previously every
    // fragment embedded a near-global copy of both tables (the
    // dynamic-unload tick window does not prevent this: link chains
    // re-export with a default meta and fall back to the full set,
    // stem_cells is never windowed, and completed_tick stamps are
    // too bursty for windows to scope), so snapshot size scaled as
    // fragments x table and crossed the HTTP body limit in a single
    // extend — measured 3.72MB -> 37.23MB at ~490 wholesale lines.
    let mut tiles: Vec<CompletedTile> = Vec::new();
    let mut cells: Vec<CompleteCell> = Vec::new();
    let mut seen_tiles: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut seen_cells: std::collections::HashSet<String> = std::collections::HashSet::new();
    for frag in &mut fragments {
        if frag.tile_ids.is_empty() {
            frag.tile_ids = frag.completed_tiles.iter().map(|t| t.id.clone()).collect();
        }
        for t in frag.completed_tiles.drain(..) {
            if seen_tiles.insert(t.id.clone()) {
                tiles.push(t);
            }
        }
        if frag.cell_ids.is_empty() {
            frag.cell_ids = frag.stem_cells.iter().map(|c| c.cell_id.clone()).collect();
        }
        for c in frag.stem_cells.drain(..) {
            if seen_cells.insert(c.cell_id.clone()) {
                cells.push(c);
            }
        }
    }
    snap.completed_tiles = tiles;
    snap.stem_cells = cells;
    snap.fragments = fragments;
    snap
}

/// Rehydrate fragment tile/cell payloads from the snapshot-level
/// shared tables (inverse of the compact-export dedup; the dynamic
/// load side). Fragments with empty id lists keep their inline
/// payloads, so legacy snapshots load unchanged.
pub fn rehydrate_fragments(
    fragments: &[SnapshotFragment],
    tiles: &[CompletedTile],
    cells: &[CompleteCell],
) -> Vec<SnapshotFragment> {
    if fragments.iter().all(|f| f.tile_ids.is_empty() && f.cell_ids.is_empty()) {
        return fragments.to_vec();
    }
    let tile_map: std::collections::HashMap<&str, &CompletedTile> =
        tiles.iter().map(|t| (t.id.as_str(), t)).collect();
    let cell_map: std::collections::HashMap<&str, &CompleteCell> =
        cells.iter().map(|c| (c.cell_id.as_str(), c)).collect();
    fragments
        .iter()
        .map(|f| {
            let mut f = f.clone();
            if !f.tile_ids.is_empty() {
                f.completed_tiles = f
                    .tile_ids
                    .iter()
                    .filter_map(|id| tile_map.get(id.as_str()).map(|t| (*t).clone()))
                    .collect();
            }
            if !f.cell_ids.is_empty() {
                f.stem_cells = f
                    .cell_ids
                    .iter()
                    .filter_map(|id| cell_map.get(id.as_str()).map(|c| (*c).clone()))
                    .collect();
            }
            f
        })
        .collect()
}

pub fn stored_fragments(snap: &TriadIncubationSnapshot) -> Vec<SnapshotFragment> {
    let frags: Vec<SnapshotFragment> = snap
        .fragments
        .iter()
        .filter(|f| is_mergeable_fragment(&f.id))
        .cloned()
        .collect();
    rehydrate_fragments(&frags, &snap.completed_tiles, &snap.stem_cells)
}

/// Rebuild live session from persisted dynamic fragments (no monolithic snapshot restore).
pub fn session_from_stored_fragments(
    fragments: &[SnapshotFragment],
    corpus_line_affinity: &[f64],
    budget: IngressBudget,
    body: BodyTarget,
) -> TriadSession {
    let mut session = if fragments.is_empty() {
        TriadSession::with_body(budget, body)
    } else {
        merge_fragments(fragments, budget, body)
    };
    session.restore_corpus_line_affinity(corpus_line_affinity.to_vec());
    session.fragment_tracks = fragments.to_vec();
    session
}

pub fn export_session_after_ingest(session: &TriadSession) -> TriadIncubationSnapshot {
    if fragment_only_session_enabled() {
        compact_manifold_snapshot(session)
    } else {
        session.manifold_snapshot()
    }
}

pub fn query_fragment_top_k(corpus_lines: usize) -> usize {
    let base = env_usize("VALHALLA_QUERY_FRAGMENT_TOP_K", 3).clamp(1, 32);
    if !dynamic_corpus_fragments_enabled() {
        return base;
    }
    // Small corpora: keep every fragment so multi-hop evidence is not truncated to seg_1.
    if corpus_lines <= 16 {
        return 32;
    }
    if corpus_lines <= 32 {
        return base;
    }
    corpus_lines.div_ceil(20).clamp(base, 32)
}

/// Keyword-strong corpus lines always enter the retrieval candidate set
/// (prevents needle / rare-token lines from being dropped by fragment top-k).
fn union_keyword_anchor_lines(
    selection: &mut QueryFragmentSelection,
    corpus: &[String],
    question: &str,
) {
    if !dynamic_corpus_fragments_enabled() || corpus.is_empty() {
        return;
    }
    let min_frac = env_f64("VALHALLA_FRAGMENT_KEYWORD_ANCHOR_FRAC", 0.35);
    for (i, line) in corpus.iter().enumerate() {
        if keyword_overlap_score(question, line) >= min_frac
            && !selection.corpus_line_indices.contains(&i)
        {
            selection.corpus_line_indices.push(i);
        }
    }
    selection.corpus_line_indices.sort_unstable();
}

fn is_mergeable_fragment(id: &str) -> bool {
    id != "merged" && id != "main"
}

/// Rank + select composable fragments for a question (skips merged/main).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryFragmentSelection {
    pub selected_ids: Vec<String>,
    pub scores: Vec<f64>,
    pub top_k: usize,
    pub total_candidates: usize,
    pub corpus_line_indices: Vec<usize>,
}

pub fn score_fragment_for_query(
    frag: &SnapshotFragment,
    question: &str,
    question_sig: &[f64],
    line_align: &[(usize, f64)],
) -> f64 {
    let mut score = 0.0;
    if !frag.patch_vector.is_empty() && !question_sig.is_empty() {
        score += 0.45 * signal_cosine(question_sig, &frag.patch_vector);
    }
    if frag.line_end >= frag.line_start {
        for (idx, align) in line_align {
            if *idx >= frag.line_start && *idx <= frag.line_end {
                let aff = frag.corpus_line_affinity.get(*idx).copied().unwrap_or(0.0);
                score += 0.35 * align * (0.5 + 0.5 * aff);
            }
        }
    } else if !frag.corpus_line_affinity.is_empty() {
        score += 0.25 * frag
            .corpus_line_affinity
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
    }
    score += 0.20 * keyword_overlap_score(question, &frag.label);
    score
}

pub fn select_fragments_for_query(
    fragments: &[SnapshotFragment],
    question: &str,
    question_sig: &[f64],
    line_align: &[(usize, f64)],
    top_k: usize,
) -> QueryFragmentSelection {
    let mut ranked: Vec<(SnapshotFragment, f64)> = fragments
        .iter()
        .filter(|f| is_mergeable_fragment(&f.id))
        .map(|f| {
            let s = score_fragment_for_query(f, question, question_sig, line_align);
            (f.clone(), s)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let take = top_k.min(ranked.len()).max(if ranked.is_empty() { 0 } else { 1 });
    let mut selected_ids = Vec::new();
    let mut scores = Vec::new();
    let mut corpus_line_indices = Vec::new();
    for (frag, s) in ranked.into_iter().take(take) {
        selected_ids.push(frag.id.clone());
        scores.push(s);
        if frag.line_end >= frag.line_start {
            for i in frag.line_start..=frag.line_end {
                if !corpus_line_indices.contains(&i) {
                    corpus_line_indices.push(i);
                }
            }
        }
    }
    corpus_line_indices.sort_unstable();

    QueryFragmentSelection {
        selected_ids,
        scores,
        top_k: take,
        total_candidates: fragments
            .iter()
            .filter(|f| is_mergeable_fragment(&f.id))
            .count(),
        corpus_line_indices,
    }
}

/// Merge only query-relevant fragments into a fresh QA session substrate.
pub fn merge_fragments_for_query(
    fragments: &[SnapshotFragment],
    question: &str,
    question_sig: &[f64],
    corpus_line_signals: &[(usize, Vec<f64>)],
    corpus_lines: &[String],
    budget: crate::signal_ingress::IngressBudget,
    body: BodyTarget,
    top_k: usize,
) -> (TriadSession, QueryFragmentSelection) {
    let line_align: Vec<(usize, f64)> = corpus_line_signals
        .iter()
        .map(|(idx, sig)| (*idx, signal_cosine(question_sig, sig)))
        .collect();
    let mut selection =
        select_fragments_for_query(fragments, question, question_sig, &line_align, top_k);
    union_keyword_anchor_lines(&mut selection, corpus_lines, question);
    let selected: Vec<SnapshotFragment> = selection
        .selected_ids
        .iter()
        .filter_map(|id| fragments.iter().find(|f| &f.id == id).cloned())
        .collect();
    assert!(
        !selected.is_empty(),
        "merge_fragments_for_query requires ≥1 selected fragment"
    );
    let session = merge_fragments(&selected, budget, body);
    (session, selection)
}

pub fn corpus_line_indices_for_selection(
    fragments: &[SnapshotFragment],
    selection: &QueryFragmentSelection,
) -> Vec<usize> {
    if !selection.corpus_line_indices.is_empty() {
        return selection.corpus_line_indices.clone();
    }
    let mut out = Vec::new();
    for id in &selection.selected_ids {
        if let Some(frag) = fragments.iter().find(|f| &f.id == id) {
            if frag.line_end >= frag.line_start {
                for i in frag.line_start..=frag.line_end {
                    if !out.contains(&i) {
                        out.push(i);
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out
}

/// Fate-owned cycles until convergence (or phase limits).
pub fn run_fate_convergence_cycles(
    session: &mut TriadSession,
    limits: &FateCycleLimits,
) -> (usize, String, f64) {
    let mut tracker = FateTracker::new();
    let mut cycles = 0usize;
    let mut aggregate_score = 0.0;
    let mut stopped_reason = "hard_max".to_string();

    while cycles < limits.hard_max_cycles {
        session.run_optimize_cycles_fate_owned(1);
        if matches!(session.body, BodyTarget::Hub | BodyTarget::Triad) {
            session.hub.mark_epoch_complete();
        }
        cycles += 1;
        aggregate_score = compute_aggregate_score(session);
        tracker.record(aggregate_score);

        if let Some(reason) = fate_should_stop(session, &tracker, cycles, limits, aggregate_score) {
            stopped_reason = reason.to_string();
            break;
        }
        if cycles >= limits.soft_max_cycles {
            stopped_reason = "soft_max".into();
            break;
        }
    }

    (cycles, stopped_reason, aggregate_score)
}

/// Fate-owned warmup cycles after prelearn corpus lines (before fragment segmentation).
pub fn run_fate_warmup(
    session: &mut TriadSession,
    cfg: &FateOrchestratorConfig,
    prelearn_lines: usize,
) -> FateWarmupOutcome {
    let saved_mode = session.hub.fate_weight_mode;
    let saved_free = session.hub.fate_routing_free;
    let saved_ingress = session.hub.ingress_routing;

    if cfg.fate_adapt_unlock {
        session.hub.set_fate_policy(
            FateWeightMode::Evaluated,
            true,
            IngressRoutingMode::RoundRobin,
        );
    }

    let limits = FateCycleLimits::warmup(cfg);
    let (cycles, stopped_reason, aggregate_score) =
        run_fate_convergence_cycles(session, &limits);

    session
        .hub
        .set_fate_policy(saved_mode, saved_free, saved_ingress);

    let state = session.hub.fate_system.export_convergence_state();
    FateWarmupOutcome {
        cycles_executed: cycles,
        prelearn_lines,
        pref_history_len: state.preference_history.len(),
        signal_tick: state.signal_tick,
        aggregate_score,
        stopped_reason,
    }
}

pub fn run_fate_adapt(session: &mut TriadSession, cfg: &FateOrchestratorConfig) -> FateAdaptOutcome {
    let saved_mode = session.hub.fate_weight_mode;
    let saved_free = session.hub.fate_routing_free;
    let saved_ingress = session.hub.ingress_routing;

    if cfg.fate_adapt_unlock {
        session.hub.set_fate_policy(
            FateWeightMode::Evaluated,
            true,
            IngressRoutingMode::RoundRobin,
        );
    }

    let limits = FateCycleLimits::adapt(cfg);
    let (cycles, stopped_reason, aggregate_score) =
        run_fate_convergence_cycles(session, &limits);

    session
        .hub
        .set_fate_policy(saved_mode, saved_free, saved_ingress);

    FateAdaptOutcome {
        cycles_executed: cycles,
        stopped_reason,
        aggregate_score,
        hub_converged: session.hub.has_fate_converged(),
        tile_converged: session.tile.has_fate_converged(),
        stem_converged: session.stem.has_fate_converged(),
        hub_pattern: session
            .hub
            .get_fate_convergence_pattern()
            .unwrap_or("unknown")
            .to_string(),
    }
}

/// After corpus ingress: Fate adapt until stable, then export manifold snapshot.
pub fn ingress_and_adapt(
    session: &mut TriadSession,
    fallback_cycles: usize,
    cfg: &FateOrchestratorConfig,
) -> (FateAdaptOutcome, TriadIncubationSnapshot) {
    let outcome = if std::env::var("VALHALLA_FATE_ORCHESTRATE")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            v == "0" || v == "false" || v == "off"
        })
        .unwrap_or(false)
    {
        session.run_optimize_cycles(fallback_cycles.max(1));
        FateAdaptOutcome {
            cycles_executed: fallback_cycles.max(1),
            stopped_reason: "legacy_fixed_cycles".into(),
            aggregate_score: compute_aggregate_score(session),
            hub_converged: session.hub.has_fate_converged(),
            tile_converged: session.tile.has_fate_converged(),
            stem_converged: session.stem.has_fate_converged(),
            hub_pattern: session
                .hub
                .get_fate_convergence_pattern()
                .unwrap_or("unknown")
                .to_string(),
        }
    } else {
        run_fate_adapt(session, cfg)
    };
    let snap = session.manifold_snapshot();
    (outcome, snap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grinder::default_port_cap;
    use crate::signal_ingress::IngressBudget;

    fn smoke_session() -> TriadSession {
        let budget = IngressBudget::experiment();
        let mut session = TriadSession::with_body(budget, BodyTarget::Triad);
        let cap = default_port_cap();
        for line in &[
            "Mercury is the closest planet to the Sun.",
            "Venus is the second planet from the Sun.",
        ] {
            let ports = crate::grinder::grind_text_hash(line, cap);
            session.emit_body_ports_with_line(&ports, Some(line));
        }
        session
    }

    #[test]
    fn fragment_export_has_patch_and_fate_state() {
        let session = smoke_session();
        let frag = export_fragment(&session, "t0", "smoke", FragmentCloseMeta::default());
        assert!(!frag.completed_tiles.is_empty() || !frag.stem_cells.is_empty());
        assert!(!frag.patch_vector.is_empty());
        assert!(frag.hub_prefs.len() >= 4);
    }

    #[test]
    fn merge_fragments_composes_structure() {
        let mut s1 = smoke_session();
        let mut s2 = smoke_session();
        s2.emit_body_ports_with_line(
            &crate::grinder::grind_text_hash("Earth is the third planet.", default_port_cap()),
            Some("Earth is the third planet."),
        );
        let f1 = export_fragment(&s1, "a", "corpus_a", FragmentCloseMeta::default());
        let f2 = export_fragment(&s2, "b", "corpus_b", FragmentCloseMeta::default());
        let merged = merge_fragments(&[f1, f2], IngressBudget::experiment(), BodyTarget::Triad);
        assert!(
            merged.tile.completed_tiles.len() >= s1.tile.completed_tiles.len()
                || merged.stem.cells.len() >= s1.stem.cells.len()
        );
    }

    #[test]
    fn fate_adapt_runs_and_stops() {
        let mut session = smoke_session();
        let cfg = FateOrchestratorConfig {
            min_cycles: 2,
            soft_max_cycles: 12,
            hard_max_cycles: 24,
            fate_adapt_unlock: true,
            ..Default::default()
        };
        let outcome = run_fate_adapt(&mut session, &cfg);
        assert!(outcome.cycles_executed >= cfg.min_cycles);
        assert!(!outcome.stopped_reason.is_empty());
    }

    #[test]
    fn fate_link_max_group_preserves_multiple_tracks() {
        std::env::set_var("VALHALLA_FATE_LINK_MAX_GROUP", "2");
        let mut s1 = smoke_session();
        s1.run_optimize_cycles(2);
        let mut s2 = smoke_session();
        s2.run_optimize_cycles(2);
        let mut s3 = smoke_session();
        s3.run_optimize_cycles(2);
        let mut f1 = export_fragment(&s1, "seg_0", "a", FragmentCloseMeta {
            line_start: 0,
            line_end: 3,
            marginal_density: 0.8,
            ..Default::default()
        });
        let mut f2 = export_fragment(&s2, "seg_1", "b", FragmentCloseMeta {
            line_start: 4,
            line_end: 7,
            marginal_density: 0.75,
            ..Default::default()
        });
        let mut f3 = export_fragment(&s3, "seg_2", "c", FragmentCloseMeta {
            line_start: 8,
            line_end: 11,
            marginal_density: 0.7,
            ..Default::default()
        });
        f1.line_end = 3;
        f2.line_start = 4;
        f2.line_end = 7;
        f3.line_start = 8;
        f3.line_end = 11;
        let budget = IngressBudget::experiment();
        let mut session = TriadSession::with_body(budget, BodyTarget::Triad);
        session.fragment_tracks = vec![f1, f2, f3];
        let cfg = FateOrchestratorConfig {
            min_cycles: 1,
            soft_max_cycles: 4,
            hard_max_cycles: 8,
            ..Default::default()
        };
        let out = fate_link_fragments(&mut session, &cfg);
        assert!(out.fragments_out >= 2, "max_group=2 should keep ≥2 tracks: {:?}", out);
    }

    #[test]
    fn fate_link_merges_contiguous_segments() {
        std::env::set_var("VALHALLA_FATE_LINK_FRAGMENTS", "1");
        let mut s1 = smoke_session();
        let mut s2 = smoke_session();
        s2.emit_body_ports_with_line(
            &crate::grinder::grind_text_hash("Earth is the third planet.", default_port_cap()),
            Some("Earth is the third planet."),
        );
        let mut f1 = export_fragment(
            &s1,
            "seg_0",
            "lines:0..0",
            FragmentCloseMeta {
                line_start: 0,
                line_end: 0,
                signal_tick_start: 1,
                signal_tick_end: 5,
                marginal_density: 0.8,
                ..Default::default()
            },
        );
        let mut f2 = export_fragment(
            &s2,
            "seg_1",
            "lines:1..1",
            FragmentCloseMeta {
                line_start: 1,
                line_end: 1,
                signal_tick_start: 6,
                signal_tick_end: 10,
                marginal_density: 0.75,
                ..Default::default()
            },
        );
        f1.line_start = 0;
        f1.line_end = 0;
        f2.line_start = 1;
        f2.line_end = 1;
        let budget = IngressBudget::experiment();
        let mut session = TriadSession::with_body(budget, BodyTarget::Triad);
        session.fragment_tracks = vec![f1, f2];
        let cfg = FateOrchestratorConfig {
            min_cycles: 1,
            soft_max_cycles: 4,
            hard_max_cycles: 8,
            ..Default::default()
        };
        let out = fate_link_fragments(&mut session, &cfg);
        assert!(out.links_merged >= 1, "expected link merge: {:?}", out);
        assert!(
            session.fragment_tracks.iter().any(|f| f.id.starts_with("link_")),
            "expected link_* fragment"
        );
    }

    #[test]
    fn compact_snapshot_dedups_tables_keeps_fragments() {
        std::env::set_var("VALHALLA_FRAGMENT_ONLY", "1");
        std::env::set_var("VALHALLA_DYNAMIC_CORPUS_FRAGMENTS", "1");
        let mut session = smoke_session();
        session.run_optimize_cycles(2);
        let full = session.manifold_snapshot();
        assert!(!full.completed_tiles.is_empty() || !full.stem_cells.is_empty());
        let compact = compact_manifold_snapshot(&session);
        // Dedup invariant: shared tables live once at snapshot level;
        // fragments keep ids only, payloads rehydrate on load.
        assert!(!compact.completed_tiles.is_empty() || !compact.stem_cells.is_empty());
        assert!(!compact.fragments.is_empty());
        assert!(compact
            .fragments
            .iter()
            .all(|f| f.completed_tiles.is_empty() && f.stem_cells.is_empty()));
        let stored = stored_fragments(&compact);
        assert!(!stored.is_empty());
        assert!(stored
            .iter()
            .any(|f| !f.completed_tiles.is_empty() || !f.stem_cells.is_empty()));
        let restored = session_from_stored_fragments(
            &stored,
            &compact.corpus_line_affinity,
            IngressBudget::experiment(),
            BodyTarget::Triad,
        );
        assert!(!restored.tile.completed_tiles.is_empty() || !restored.stem.cells.is_empty());
        assert!(restored.hub.fate_system().get_logical_time() >= 0);
    }

    #[test]
    fn manifold_snapshot_roundtrip_preserves_fate_history() {
        let mut session = smoke_session();
        let cfg = FateOrchestratorConfig {
            min_cycles: 3,
            soft_max_cycles: 8,
            hard_max_cycles: 16,
            fate_adapt_unlock: true,
            ..Default::default()
        };
        run_fate_adapt(&mut session, &cfg);
        let snap = session.manifold_snapshot();
        assert_eq!(snap.manifold_protocol.as_deref(), Some(MANIFOLD_PROTOCOL));
        assert!(snap.fate_convergence.is_some());
        let hist_before = snap
            .fate_convergence
            .as_ref()
            .map(|f| f.preference_history.len())
            .unwrap_or(0);

        let restored = TriadSession::from_incubation_snapshot(
            IngressBudget::experiment(),
            BodyTarget::Triad,
            snap,
        );
        let hist_after = restored
            .hub
            .fate_system
            .export_convergence_state()
            .preference_history
            .len();
        assert_eq!(hist_before, hist_after);
    }

    #[test]
    fn multi_fragment_per_corpus_line() {
        let budget = IngressBudget::experiment();
        let mut session = TriadSession::with_body(budget, BodyTarget::Triad);
        let lines = vec![
            "Alpha centauri is a star system.".to_string(),
            "Betelgeuse is a red supergiant.".to_string(),
        ];
        let cap = default_port_cap();
        let ports: Vec<_> = lines.iter().map(|l| crate::grinder::grind_text_hash(l, cap)).collect();
        let cfg = FateOrchestratorConfig {
            min_cycles: 2,
            soft_max_cycles: 8,
            hard_max_cycles: 16,
            fate_adapt_unlock: true,
            warmup_enabled: false,
            ..Default::default()
        };
        ingest_corpus_with_fragments_mode(&mut session, &lines, &ports, &cfg, FragmentMode::Line);
        let snap = session.manifold_snapshot();
        assert!(snap.fragments.len() >= 3, "expected line fragments + merged");
    }

    #[test]
    fn density_fragments_group_multiple_lines() {
        std::env::set_var("VALHALLA_FRAGMENT_MAX_LINES", "4");
        let budget = IngressBudget::experiment();
        let mut session = TriadSession::with_body(budget, BodyTarget::Triad);
        let lines: Vec<String> = (0..12)
            .map(|i| format!("Corpus fact number {i} about planetary science and orbital mechanics."))
            .collect();
        let cap = default_port_cap();
        let ports: Vec<_> = lines.iter().map(|l| crate::grinder::grind_text_hash(l, cap)).collect();
        let cfg = FateOrchestratorConfig {
            min_cycles: 2,
            soft_max_cycles: 8,
            hard_max_cycles: 16,
            fate_adapt_unlock: true,
            warmup_enabled: false,
            ..Default::default()
        };
        ingest_corpus_with_fragments_mode(
            &mut session,
            &lines,
            &ports,
            &cfg,
            FragmentMode::Density,
        );
        let snap = session.manifold_snapshot();
        let seg_frags: Vec<_> = snap
            .fragments
            .iter()
            .filter(|f| f.id.starts_with("seg_"))
            .collect();
        assert!(
            !seg_frags.is_empty(),
            "density mode should record seg_* fragments"
        );
        assert!(
            seg_frags.len() < lines.len(),
            "density mode should produce fewer segments than lines: {} segs vs {} lines",
            seg_frags.len(),
            lines.len()
        );
        let multi_line = seg_frags.iter().any(|f| f.line_end > f.line_start);
        assert!(
            multi_line,
            "max_lines=4 over 12 lines should yield multi-line segments: {:?}",
            seg_frags
                .iter()
                .map(|f| (f.id.as_str(), f.line_start, f.line_end, f.close_reason.as_str()))
                .collect::<Vec<_>>()
        );
        std::env::remove_var("VALHALLA_FRAGMENT_MAX_LINES");
    }

    #[test]
    fn query_merge_prefers_matching_fragment() {
        let budget = IngressBudget::experiment();
        let cap = default_port_cap();

        let mut s_mercury = TriadSession::with_body(budget.clone(), BodyTarget::Triad);
        s_mercury.emit_body_ports_with_line(
            &crate::grinder::grind_text_hash("Mercury is the closest planet to the Sun.", cap),
            Some("Mercury is the closest planet to the Sun."),
        );
        s_mercury.run_optimize_cycles_fate_owned(2);
        let f_mercury = export_fragment(
            &s_mercury,
            "seg_mercury",
            "Mercury closest planet Sun",
            FragmentCloseMeta {
                line_start: 0,
                line_end: 0,
                ..Default::default()
            },
        );

        let mut s_venus = TriadSession::with_body(budget.clone(), BodyTarget::Triad);
        s_venus.emit_body_ports_with_line(
            &crate::grinder::grind_text_hash("Venus is the second planet from the Sun.", cap),
            Some("Venus is the second planet from the Sun."),
        );
        s_venus.run_optimize_cycles_fate_owned(2);
        let f_venus = export_fragment(
            &s_venus,
            "seg_venus",
            "Venus second planet Sun",
            FragmentCloseMeta {
                line_start: 1,
                line_end: 1,
                ..Default::default()
            },
        );

        let frags = vec![f_mercury, f_venus];
        let q_ports = crate::grinder::grind_text_hash(
            "Which planet is closest to the Sun?",
            cap,
        );
        let corpus_sigs = vec![
            (
                0,
                crate::grinder::grind_text_hash("Mercury is the closest planet to the Sun.", cap)
                    .embedding
                    .clone(),
            ),
            (
                1,
                crate::grinder::grind_text_hash("Venus is the second planet from the Sun.", cap)
                    .embedding
                    .clone(),
            ),
        ];
        let sel = select_fragments_for_query(
            &frags,
            "Which planet is closest to the Sun?",
            &q_ports.embedding,
            &corpus_sigs
                .iter()
                .map(|(i, s)| (*i, signal_cosine(&q_ports.embedding, s)))
                .collect::<Vec<_>>(),
            1,
        );
        assert_eq!(sel.selected_ids.len(), 1);
        assert!(
            sel.selected_ids[0].contains("mercury"),
            "expected Mercury fragment, got {:?}",
            sel.selected_ids
        );

        let (merged, sel2) = merge_fragments_for_query(
            &frags,
            "Which planet is closest to the Sun?",
            &q_ports.embedding,
            &corpus_sigs,
            &[
                "Mercury is the closest planet to the Sun.".into(),
                "Venus is the second planet from the Sun.".into(),
            ],
            budget,
            BodyTarget::Triad,
            1,
        );
        assert_eq!(sel2.selected_ids, sel.selected_ids);
        assert!(!merged.tile.completed_tiles.is_empty() || !merged.stem.cells.is_empty());
    }

    #[test]
    fn dynamic_fragments_keyword_anchor_keeps_needle_line() {
        std::env::remove_var("VALHALLA_DYNAMIC_CORPUS_FRAGMENTS");
        std::env::remove_var("VALHALLA_QUERY_FRAGMENT_MERGE");
        let needle = "The magic registration code for Project Valhalla is XK-7742.";
        let question = "What is the magic registration code for Project Valhalla?";
        let filler: Vec<String> = (0..62)
            .map(|i| format!("Generic filler fact number {i} about science and geography."))
            .collect();
        let mut corpus = filler;
        corpus[30] = needle.to_string();

        let budget = IngressBudget::experiment();
        let cap = default_port_cap();
        let mut session = TriadSession::with_body(budget.clone(), BodyTarget::Triad);
        let ports: Vec<_> = corpus.iter().map(|l| grind_text_hash(l, cap)).collect();
        let cfg = FateOrchestratorConfig {
            min_cycles: 1,
            soft_max_cycles: 4,
            hard_max_cycles: 8,
            fate_adapt_unlock: false,
            warmup_enabled: false,
            ..Default::default()
        };
        ingest_corpus_with_fragments_mode(
            &mut session,
            &corpus,
            &ports,
            &cfg,
            FragmentMode::Density,
        );
        let snap = session.manifold_snapshot();
        let frags: Vec<_> = snap
            .fragments
            .iter()
            .filter(|f| f.id != "merged" && f.id != "main")
            .cloned()
            .collect();
        assert!(!frags.is_empty(), "expected density fragments");

        let q_ports = grind_text_hash(question, cap);
        let corpus_sigs: Vec<_> = corpus
            .iter()
            .enumerate()
            .map(|(i, l)| (i, grind_text_hash(l, cap).embedding))
            .collect();
        let top_k = query_fragment_top_k(corpus.len());
        let (_, selection) = merge_fragments_for_query(
            &frags,
            question,
            &q_ports.embedding,
            &corpus_sigs,
            &corpus,
            budget,
            BodyTarget::Triad,
            top_k,
        );
        assert!(
            selection.corpus_line_indices.contains(&30),
            "needle line 30 must be anchored: {:?}",
            selection.corpus_line_indices
        );
    }

    #[test]
    fn truncate_label_respects_utf8_boundaries() {
        let long = "Cloud VoIP Service. | With our Cloud VoIP service, you can get VoIP service completely in the cloud with no equipment required for toll-free numbers.";
        let out = truncate_label(long, 120);
        assert!(out.ends_with('…'));
        assert!(out.chars().count() <= 121);
    }

    #[test]
    fn pretrain_library_roundtrip() {
        let lines = vec!["Saturn has prominent rings.".into()];
        let cfg = FateOrchestratorConfig {
            min_cycles: 2,
            soft_max_cycles: 6,
            hard_max_cycles: 12,
            fate_adapt_unlock: true,
            warmup_cycles: 2,
            warmup_prelearn_lines: 0,
            ..Default::default()
        };
        let lib = pretrain_corpus_to_library(BodyTarget::Triad, &lines, &cfg);
        assert_eq!(lib.version, FRAGMENT_LIBRARY_VERSION);
        assert!(!lib.fragments.is_empty());
        let dir = std::env::temp_dir().join("valhalla_manifold_lib_test.json");
        lib.save_json(&dir).unwrap();
        let loaded = ManifoldFragmentLibrary::load_json(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);
        assert!(loaded.merged_snapshot.is_some());
    }

    #[test]
    fn fate_warmup_builds_preference_history() {
        let budget = IngressBudget::experiment();
        let mut session = TriadSession::with_body(budget, BodyTarget::Triad);
        let lines: Vec<String> = (0..8)
            .map(|i| format!("Warmup corpus line {i} about astronomy and orbital dynamics."))
            .collect();
        let cap = default_port_cap();
        let ports: Vec<_> = lines
            .iter()
            .map(|l| crate::grinder::grind_text_hash(l, cap))
            .collect();
        let cfg = FateOrchestratorConfig {
            min_cycles: 2,
            soft_max_cycles: 8,
            hard_max_cycles: 16,
            fate_adapt_unlock: true,
            warmup_enabled: true,
            warmup_cycles: 3,
            warmup_prelearn_lines: 2,
            ..Default::default()
        };
        let (_, warmup) = ingest_corpus_with_fragments_mode_full(
            &mut session,
            &lines,
            &ports,
            &cfg,
            FragmentMode::Density,
        );
        let w = warmup.expect("warmup should run when prelearn > 0");
        assert_eq!(w.prelearn_lines, 2);
        let warmup_limits = FateCycleLimits::warmup(&cfg);
        assert!(w.cycles_executed >= warmup_limits.min_cycles);
        assert!(w.cycles_executed <= warmup_limits.hard_max_cycles);
        assert!(!w.stopped_reason.is_empty());
        assert!(
            w.pref_history_len >= 2,
            "warmup should grow preference_history, got {}",
            w.pref_history_len
        );
    }

    #[test]
    fn convergence_cycles_respect_fate_limits() {
        let mut session = smoke_session();
        let limits = FateCycleLimits {
            min_cycles: 2,
            soft_max_cycles: 12,
            hard_max_cycles: 24,
        };
        let (cycles, reason, score) = run_fate_convergence_cycles(&mut session, &limits);
        assert!(cycles >= limits.min_cycles);
        assert!(cycles <= limits.hard_max_cycles);
        assert!(!reason.is_empty());
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn prelearn_lines_skip_fragments_until_warmup_done() {
        std::env::set_var("VALHALLA_FRAGMENT_MAX_LINES", "2");
        let budget = IngressBudget::experiment();
        let mut session = TriadSession::with_body(budget, BodyTarget::Triad);
        let lines: Vec<String> = (0..6)
            .map(|i| format!("Segment line {i} for density fragment warmup test."))
            .collect();
        let cap = default_port_cap();
        let ports: Vec<_> = lines
            .iter()
            .map(|l| crate::grinder::grind_text_hash(l, cap))
            .collect();
        let cfg = FateOrchestratorConfig {
            min_cycles: 2,
            soft_max_cycles: 6,
            hard_max_cycles: 12,
            fate_adapt_unlock: true,
            warmup_enabled: true,
            warmup_cycles: 2,
            warmup_prelearn_lines: 3,
            ..Default::default()
        };
        ingest_corpus_with_fragments_mode(&mut session, &lines, &ports, &cfg, FragmentMode::Density);
        let snap = session.manifold_snapshot();
        let seg_frags: Vec<_> = snap
            .fragments
            .iter()
            .filter(|f| f.id.starts_with("seg_"))
            .collect();
        assert!(
            seg_frags.iter().all(|f| f.line_start >= 3),
            "fragments should start after prelearn lines: {:?}",
            seg_frags
                .iter()
                .map(|f| (f.id.as_str(), f.line_start, f.line_end))
                .collect::<Vec<_>>()
        );
        std::env::remove_var("VALHALLA_FRAGMENT_MAX_LINES");
    }
}
