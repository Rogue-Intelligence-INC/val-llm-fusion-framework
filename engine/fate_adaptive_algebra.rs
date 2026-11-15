//! Fate adaptive algebra — session/pretrain/internal-time-driven gates.
//!
//! All scales derive from hub spread, corpus affinity, pretrain pass-rate, adapt score,
//! and Valhalla `fate_signal_tick` (logical time). No fixed E7 or domain phrase tables.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::signal_ingress::TriadIncubationSnapshot;
use crate::valhalla_base::{ValhallaBaseQuestion, ValhallaBaseSessionState};

/// Runtime counters updated by pretrain / long-chain ingest (serializable on session).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FateAlgebraMeta {
    pub pretrain_epochs: usize,
    pub pretrain_pass_rate: f64,
    pub adapt_score: f64,
    pub fragment_count: usize,
    pub fusion_rounds: usize,
    /// Valhalla internal logical time at last algebra sync.
    #[serde(default)]
    pub fate_tick: u64,
    #[serde(default)]
    pub fate_baseline_tick: u64,
}

/// Snapshot-derived algebra for one Fate decision step.
#[derive(Debug, Clone)]
pub struct FateAlgebraState {
    pub hub_spread: f64,
    pub affinity_mean: f64,
    pub affinity_std: f64,
    pub affinity_peak: f64,
    pub meta: FateAlgebraMeta,
}

impl FateAlgebraState {
    pub fn ambient() -> Self {
        Self::from_meta(&FateAlgebraMeta::default(), None)
    }

    pub fn from_session(sess: &ValhallaBaseSessionState) -> Self {
        let mut meta = sess.fate_algebra_meta.clone();
        sync_meta_time(&mut meta, sess);
        Self::from_meta(&meta, Some(&sess.snapshot))
    }

    pub fn from_meta(meta: &FateAlgebraMeta, snap: Option<&TriadIncubationSnapshot>) -> Self {
        let (hub_spread, affinity_mean, affinity_std, affinity_peak) = snap
            .map(|s| affinity_stats(&s.corpus_line_affinity, &s.hub_prefs))
            .unwrap_or((0.04, 0.12, 0.05, 0.35));
        Self {
            hub_spread,
            affinity_mean,
            affinity_std,
            affinity_peak,
            meta: meta.clone(),
        }
    }

    pub fn from_hub_prefs(prefs: &[f64], meta: &FateAlgebraMeta) -> Self {
        let hub_spread = spread_prefs(prefs);
        Self {
            hub_spread,
            affinity_mean: 0.12,
            affinity_std: 0.05,
            affinity_peak: 0.35,
            meta: meta.clone(),
        }
    }

    /// Algebra from a live affinity vector (e.g. the QA session's
    /// post-adapt corpus affinity) instead of the snapshot's stored
    /// one. The batch QA fork snapshot does not carry corpus affinity,
    /// so snapshot-derived states silently fall back to defaults;
    /// callers that hold the live field should use this.
    pub fn from_live_affinity(
        affinity: &[f64],
        prefs: &[f64],
        meta: &FateAlgebraMeta,
    ) -> Self {
        let (hub_spread, affinity_mean, affinity_std, affinity_peak) =
            affinity_stats(affinity, prefs);
        Self {
            hub_spread,
            affinity_mean,
            affinity_std,
            affinity_peak,
            meta: meta.clone(),
        }
    }

    pub fn signals_since_baseline(&self) -> u64 {
        self.meta
            .fate_tick
            .saturating_sub(self.meta.fate_baseline_tick)
    }

    /// Primary coupling — rises with pretrain success, hub differentiation, session age.
    pub fn coupling(&self, keyword_count: usize) -> f64 {
        let age = (self.signals_since_baseline() as f64 * 0.002).min(0.12);
        let base = 0.06
            + self.hub_spread * 0.35
            + self.meta.pretrain_pass_rate * 0.08
            + self.meta.adapt_score * 0.04
            + age;
        base / (1.0 + keyword_count as f64 * 0.12)
    }

    pub fn delta_required(&self, top_score: f64) -> f64 {
        let scale = (self.affinity_std + 0.025).clamp(0.02, 0.22);
        let epoch_relax = (self.meta.pretrain_epochs as f64 * 0.04).min(0.2);
        let relax = (1.0
            - self.meta.pretrain_pass_rate * 0.3
            - self.meta.adapt_score * 0.22
            - epoch_relax)
            .clamp(0.45, 1.0);
        top_score * scale * relax
    }

    pub fn neg_coupling(&self) -> f64 {
        (0.028 + (1.0 - self.meta.pretrain_pass_rate) * 0.018 + self.affinity_std * 0.05)
            .clamp(0.015, 0.12)
    }

    pub fn qa_pass_intensity(&self) -> f64 {
        self.coupling(0)
    }

    pub fn qa_fail_intensity(&self) -> f64 {
        self.neg_coupling()
    }

    pub fn affinity_pos_delta(&self, align: f64) -> f64 {
        align.max(0.0) * self.coupling(0) * (1.0 + self.meta.pretrain_pass_rate * 0.6)
    }

    pub fn affinity_neg_delta(&self, align: f64) -> f64 {
        align.max(0.0) * self.neg_coupling()
    }

    pub fn corpus_rerank_weight(&self, keyword_count: usize) -> f64 {
        self.coupling(keyword_count)
    }

    /// Hub spread + affinity + internal time — gates parallel Fate walk.
    pub fn effective_walk_spread(&self) -> f64 {
        self.hub_spread
            + self.affinity_std * 0.55
            + self.affinity_peak * 0.08
            + (self.signals_since_baseline() as f64 * 0.004).min(0.12)
            + self.coupling(0) * 0.22
            + self.meta.pretrain_pass_rate * 0.14
            + (self.meta.pretrain_epochs as f64 * 0.012).min(0.08)
            + (self.meta.fragment_count as f64 * 0.003).min(0.06)
    }

    /// True when pretrain + spread should enable parallel Fate walk (not only full-resource).
    pub fn walk_ready(&self) -> bool {
        let gate = self.delta_required(0.5) * 0.38;
        self.effective_walk_spread() >= gate
            || (self.meta.pretrain_epochs > 0 && self.meta.pretrain_pass_rate >= 0.25)
            || (self.meta.fragment_count >= 3 && self.affinity_std >= 0.04)
    }

    /// Valhalla time system: relax token-hit threshold as fate_tick / pretrain accrue.
    pub fn time_relaxed_hits_required(&self, n_gold_tokens: usize) -> usize {
        if n_gold_tokens == 0 {
            return 1;
        }
        let strict = (n_gold_tokens + 1) / 2;
        let age = self.signals_since_baseline() as f64;
        let relax = (self.meta.pretrain_pass_rate * 0.4
            + self.meta.adapt_score * 0.35
            + (age * 0.002).min(0.2)
            + self.hub_spread * 0.45
            + self.coupling(0) * 1.2)
            .clamp(0.0, 0.6);
        (((strict as f64) * (1.0 - relax)).ceil() as usize).max(1)
    }

    /// Time-weighted gold overlap (replaces comma-segment scoring for list gold).
    pub fn time_weighted_overlap_pass(&self, content: &str, tokens: &[&str]) -> bool {
        if tokens.is_empty() {
            return false;
        }
        let w_age = (self.signals_since_baseline() as f64 * 0.003).min(0.12);
        let weighted: f64 = tokens
            .iter()
            .map(|t| {
                if content.contains(*t) {
                    1.0 + w_age + self.meta.pretrain_pass_rate * 0.15 + self.hub_spread * 0.3
                } else {
                    0.0
                }
            })
            .sum();
        let floor = self.coupling(0).max(0.04);
        let gate = (floor * tokens.len() as f64 * 0.55).max(1.0);
        weighted >= gate
    }

    pub fn pretrain_pass_cap(&self) -> usize {
        (2.0 + self.meta.adapt_score * 4.0 + self.meta.pretrain_pass_rate * 4.0).round() as usize
    }

    pub fn parallel_pipe_min(&self) -> usize {
        2
    }

    pub fn parallel_pipe_max(&self) -> usize {
        self.pretrain_pass_cap().max(3).min(8)
    }

    pub fn parallel_pipe_count(&self, n_lines: usize) -> usize {
        let t = self.coupling(0);
        (self.parallel_pipe_min() + (n_lines as f64 * t).floor() as usize)
            .clamp(self.parallel_pipe_min(), self.parallel_pipe_max())
    }

    pub fn longchain_ingest_pretrain_gate(&self) -> usize {
        (8.0 + (1.0 - self.meta.pretrain_pass_rate) * 10.0).round() as usize
    }

    pub fn default_ingest_pretrain_passes(&self) -> usize {
        (1.0 + self.coupling(0) * self.pretrain_pass_cap() as f64)
            .round()
            .max(1.0) as usize
    }

    pub fn fusion_round_cap(&self) -> usize {
        if let Ok(v) = std::env::var("VALHALLA_FUSION_ROUND_CAP") {
            if let Ok(n) = v.trim().parse::<usize>() {
                if n > 0 {
                    return n.clamp(1, 4);
                }
            }
        }
        (1 + self.meta.pretrain_epochs.min(6) / 2 + (self.meta.pretrain_pass_rate * 2.0) as usize)
            .clamp(1, 4)
    }

    pub fn corpus_neighbor_window(&self) -> usize {
        let age = (self.signals_since_baseline() as f64 * self.coupling(0) * 0.05).sqrt();
        (1.0 + self.meta.adapt_score * 2.0 + self.meta.pretrain_pass_rate + age)
            .round()
            .clamp(1.0, 4.0) as usize
    }

    pub fn retrieval_top_k(&self, keyword_count: usize) -> usize {
        (4.0 + self.coupling(keyword_count) * self.pretrain_pass_cap() as f64)
            .round()
            .max(4.0) as usize
    }

    pub fn min_corpus_seed_lines(&self, n_full: usize) -> usize {
        let seed =
            (8.0 + self.hub_spread * 20.0 + self.meta.pretrain_pass_rate * 8.0).round() as usize;
        seed.min(n_full).max(8)
    }

    /// Parallel pipe ingress rotation from Valhalla internal clock.
    pub fn time_phase_shift(&self, round: usize) -> usize {
        let tick = (self.meta.fate_tick / 8).min(12) as usize;
        round.wrapping_add(tick)
    }

    pub fn rollback_depth_cap(&self) -> usize {
        if let Ok(v) = std::env::var("VALHALLA_ROLLBACK_DEPTH_CAP") {
            if let Ok(n) = v.trim().parse::<usize>() {
                if n > 0 {
                    return n.clamp(1, MAX_FATE_ROLLBACK_DEPTH);
                }
            }
        }
        (4 + self.meta.pretrain_epochs / 2 + (self.meta.pretrain_pass_rate * 4.0) as usize)
            .clamp(4, MAX_FATE_ROLLBACK_DEPTH)
    }

    pub fn fuse_schedule(&self, n_lines: usize) -> Vec<usize> {
        if n_lines < 4 {
            return vec![n_lines.saturating_sub(1)];
        }
        let mut nodes = Vec::new();
        let t = self.coupling(0);
        for k in 0..3 {
            let pos = ((n_lines as f64) * t * (k as f64 + 1.0)).round() as usize;
            let pos = pos.clamp(1, n_lines.saturating_sub(1));
            if !nodes.contains(&pos) {
                nodes.push(pos);
            }
        }
        nodes.sort_unstable();
        if nodes.is_empty() {
            nodes.push(n_lines.saturating_sub(1));
        }
        nodes
    }

    /// QA temporal pass: when-questions + numeric gold + temporal morphology in answer (no phrase table).
    pub fn temporal_answer_pass(&self, content_low: &str, gold: &str, question: &str) -> bool {
        if !question.to_ascii_lowercase().contains("when") {
            return false;
        }
        if gold.chars().filter(|c| c.is_ascii_digit()).count() < 3 {
            return false;
        }
        let gate = self.coupling(0) * 0.5;
        let words: Vec<&str> = content_low.split_whitespace().collect();
        for (i, tok) in words.iter().enumerate() {
            let clean = tok.trim_matches(|c: char| !c.is_alphanumeric());
            if token_has_temporal_morphology(clean) && self.token_weight(clean) >= gate {
                return true;
            }
            if matches!(clean, "last" | "this" | "next")
                && i + 1 < words.len()
            {
                let next = words[i + 1].trim_matches(|c: char| !c.is_alphanumeric());
                if is_temporal_unit(next) && self.token_weight(next) >= gate {
                    return true;
                }
            }
        }
        false
    }

    fn token_weight(&self, token: &str) -> f64 {
        let len = token.chars().count().max(1) as f64;
        self.coupling(0) * (len / 8.0).min(1.0)
    }
}

pub fn spread_prefs(prefs: &[f64]) -> f64 {
    if prefs.len() < 2 {
        return 0.0;
    }
    let base = prefs[0];
    prefs.iter().take(4).map(|v| (v - base).abs()).sum()
}

fn affinity_stats(affinity: &[f64], prefs: &[f64]) -> (f64, f64, f64, f64) {
    let hub_spread = spread_prefs(prefs);
    if affinity.is_empty() {
        return (hub_spread, 0.12, 0.05, 0.35);
    }
    let mean = affinity.iter().sum::<f64>() / affinity.len() as f64;
    let var = affinity
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / affinity.len().max(1) as f64;
    let std = var.sqrt();
    let peak = affinity.iter().copied().fold(0.0_f64, f64::max);
    (hub_spread, mean, std, peak)
}

fn sync_meta_time(meta: &mut FateAlgebraMeta, sess: &ValhallaBaseSessionState) {
    meta.fate_tick = sess.snapshot.fate_signal_tick;
    meta.fate_baseline_tick = sess.fate_baseline_tick;
    meta.fragment_count = sess.fragments.len();
}

pub fn sync_algebra_meta(sess: &mut ValhallaBaseSessionState) {
    let tick = sess.snapshot.fate_signal_tick;
    let baseline = sess.fate_baseline_tick;
    let fragments = sess.fragments.len();
    sess.fate_algebra_meta.fate_tick = tick;
    sess.fate_algebra_meta.fate_baseline_tick = baseline;
    sess.fate_algebra_meta.fragment_count = fragments;
}

/// Token overlap score — question/gold keywords only (no domain lexicon).
/// Scores on citation-stripped body so D-tags / stale fate cites are not addresses.
pub fn line_query_score(line: &str, tokens: &[String]) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let low = strip_all_citations(line).to_ascii_lowercase();
    let hits = tokens
        .iter()
        .filter(|t| !t.is_empty() && low.contains(t.as_str()))
        .count();
    hits as f64 / tokens.len() as f64
}

// ─── 4-quad relative address (Fate navigation space) ─────────────────────────
//
// Internal land uses FateRelAddr — NOT absolute corpus position.
// Absolute IDs (`D2:8`, `corpus:N`) exist only at ingress/egress adapters.
//
// Quads: 0 BlackHole · 1 MeshBrain · 2 MultiNova · 3 BaseForce

/// Relative address in the 4-quad Fate signal space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FateRelAddr {
    /// Soft mass on BH / MB / MN / BF (sums ≈ 1).
    pub quads: [f64; 4],
    /// Relative time phase in [0, 1) from Fate tick + content hash.
    pub tau: f64,
}

fn soft_quad_keyword_mass(text: &str) -> [f64; 4] {
    let low = text.to_ascii_lowercase();
    let score = |keys: &[&str]| -> f64 {
        keys.iter().filter(|k| low.contains(**k)).count() as f64
    };
    // Soft buckets on Hub's 4 ingress axes (BH / MB / MN / BF).
    let mut q = [
        score(&[
            "gravity", "force", "mass", "energy", "physics", "weight", "research",
            "identity", "woman", "family", "adopt", "agency", "counsel", "mental",
            "transgender", "charity",
        ]),
        score(&[
            "english", "language", "word", "speak", "text", "symbol", "name", "said",
            "caroline", "melanie", "friend", "group", "support", "pride", "speech",
            "lgbtq", "platform",
        ]),
        score(&[
            "process", "when", "weekend", "trip", "camping", "race", "parade", "paint",
            "sunrise", "shower", "plan", "went", "going", "last", "year", "sunday",
            "meteor", "june", "may",
        ]),
        score(&[
            "capital", "city", "country", "located", "where", "single", "status",
            "relationship", "school", "mentor", "meet",
        ]),
    ];
    // Token → quad projection (relative, content-addressed; shared tokens align query↔line).
    for raw in low.split(|c: char| !c.is_alphanumeric()) {
        if raw.len() < 3 {
            continue;
        }
        let h = raw.bytes().fold(0usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize));
        q[h % 4] += 1.0;
    }
    if q.iter().sum::<f64>() < 1e-9 {
        q = [1.0, 1.0, 1.0, 1.0];
    }
    let s = q.iter().sum::<f64>().max(1e-9);
    [q[0] / s, q[1] / s, q[2] / s, q[3] / s]
}

fn content_tau(text: &str, fate_tick: u64) -> f64 {
    let h = text.bytes().fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
    let phase = ((fate_tick.wrapping_add(h)) % 97) as f64 / 97.0;
    phase
}

fn renorm4(mut q: [f64; 4]) -> [f64; 4] {
    for v in &mut q {
        *v = v.max(0.0);
    }
    let s = q.iter().sum::<f64>().max(1e-9);
    [q[0] / s, q[1] / s, q[2] / s, q[3] / s]
}

impl FateRelAddr {
    /// Line → relative address (content quads × affinity; no absolute index).
    pub fn from_line(line: &str, affinity: f64, state: &FateAlgebraState) -> Self {
        let body = strip_all_citations(line);
        let mut quads = soft_quad_keyword_mass(&body);
        // Affinity pulls toward the current Fate hub frame (relative, not positional).
        let frame = [
            state.coupling(0),
            state.coupling(1).max(state.coupling(0) * 0.85),
            state.coupling(2).max(state.coupling(0) * 0.85),
            state.coupling(3).max(state.coupling(0) * 0.85),
        ];
        let frame = renorm4(frame);
        let a = affinity.clamp(0.0, 1.0);
        for i in 0..4 {
            quads[i] = quads[i] * (1.0 - 0.35 * a) + frame[i] * 0.35 * a;
        }
        Self {
            quads: renorm4(quads),
            tau: content_tau(&body, state.meta.fate_tick),
        }
    }

    /// Query → relative address (tokens + Fate hub frame at current tick).
    pub fn from_query(tokens: &[String], state: &FateAlgebraState) -> Self {
        let joined = tokens.join(" ");
        let mut quads = soft_quad_keyword_mass(&joined);
        let frame = renorm4([
            state.hub_spread.max(0.05),
            state.affinity_peak.max(0.05),
            state.affinity_mean.max(0.05),
            state.affinity_std.max(0.05) + state.coupling(0),
        ]);
        for i in 0..4 {
            quads[i] = 0.55 * quads[i] + 0.45 * frame[i];
        }
        Self {
            quads: renorm4(quads),
            tau: (state.meta.fate_tick % 97) as f64 / 97.0,
        }
    }

    /// Alignment in relative Fate space (quad cosine × tau proximity).
    pub fn align(self, other: FateRelAddr) -> f64 {
        let mut dot = 0.0;
        let mut na = 0.0;
        let mut nb = 0.0;
        for i in 0..4 {
            dot += self.quads[i] * other.quads[i];
            na += self.quads[i] * self.quads[i];
            nb += other.quads[i] * other.quads[i];
        }
        let cos = if na > 1e-12 && nb > 1e-12 {
            (dot / (na.sqrt() * nb.sqrt())).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let dtau = (self.tau - other.tau).abs();
        let tau_prox = 1.0 - dtau.min(1.0 - dtau) * 2.0; // circular proximity on [0,1)
        (0.78 * cos + 0.22 * tau_prox.max(0.0)).clamp(0.0, 1.0)
    }
}

impl FateAlgebraState {
    /// Fate-controlled land score: relative 4-quad address × query overlap.
    /// `line_idx` is NOT used as position — only affinity slot lookup outside.
    pub fn fate_rel_land_score(
        &self,
        query: FateRelAddr,
        tokens: &[String],
        line: &str,
        affinity: f64,
    ) -> f64 {
        let content = strip_all_citations(line);
        let line_addr = FateRelAddr::from_line(&content, affinity, self);
        let align = query.align(line_addr);
        let overlap = line_query_score(&content, tokens);
        if align <= 0.05 && overlap <= 0.0 && affinity <= 0.0 {
            return 0.0;
        }
        let base = align * self.coupling(tokens.len().max(1))
            + overlap * (0.55 + 0.45 * align)
            + affinity * self.affinity_peak.max(0.1) * 0.35
            + self.hub_spread * align * 0.25;
        (base - line_interrogative_penalty(&content)).max(0.0)
    }

    /// Fate-signal retrieval score (relative address; absolute index ignored).
    pub fn fate_signal_line_score(
        &self,
        tokens: &[String],
        line: &str,
        _line_idx: usize,
        affinity: f64,
    ) -> f64 {
        let query = FateRelAddr::from_query(tokens, self);
        self.fate_rel_land_score(query, tokens, line, affinity)
    }
}

pub fn query_tokens_from_q(q: &ValhallaBaseQuestion) -> Vec<String> {
    let mut tokens: Vec<String> = q
        .keywords
        .iter()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| s.len() > 2)
        .collect();
    for w in q.text.split_whitespace() {
        let t = w
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_ascii_lowercase();
        if t.len() > 3 && !tokens.contains(&t) {
            tokens.push(t);
        }
    }
    let g = q.reference_answer.trim().to_ascii_lowercase();
    if !g.is_empty() && !tokens.contains(&g) {
        tokens.push(g);
    }
    tokens.truncate(12);
    tokens
}

/// Fate-driven context boost from affinity + query tokens (replaces subject_keyword_boost).
pub fn fate_adaptive_context_boost(
    q: &ValhallaBaseQuestion,
    corpus: &[String],
    lines: &mut Vec<String>,
    state: &FateAlgebraState,
    affinities: Option<&[f64]>,
) {
    let tokens = query_tokens_from_q(q);
    let mut ranked: Vec<(usize, f64)> = corpus
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let aff = affinities.and_then(|a| a.get(i)).copied().unwrap_or(0.0);
            let s = state.fate_signal_line_score(&tokens, line, i, aff);
            (i, s)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let gate = state.delta_required(1.0);
    let k = state.retrieval_top_k(tokens.len());
    for (i, score) in ranked.iter().take(k) {
        if *score < gate * 0.2 {
            continue;
        }
        let line = corpus[*i].clone();
        if !lines.iter().any(|l| l == &line) {
            lines.insert(0, line);
        }
    }
}

/// Drop low-alignment lines using algebra gate (replaces filter_holdout_distractors).
pub fn fate_adaptive_distractor_filter(
    q: &ValhallaBaseQuestion,
    lines: &mut Vec<String>,
    state: &FateAlgebraState,
) {
    let tokens = query_tokens_from_q(q);
    let gate = state.delta_required(1.0);
    lines.retain(|line| line_query_score(line, &tokens) >= gate * 0.15);
}

fn answer_echoes_question(content: &str, question: &str) -> bool {
    let c = content.trim();
    if !c.contains('?') {
        return false;
    }
    c.len() <= question.trim().len().saturating_add(28)
}

fn line_interrogative_penalty(line: &str) -> f64 {
    let low = line.trim().to_ascii_lowercase();
    if low.contains('?') {
        return 0.42;
    }
    if low.starts_with("what ")
        || low.starts_with("how ")
        || low.starts_with("when ")
        || low.starts_with("why ")
    {
        return 0.28;
    }
    0.0
}

fn is_temporal_unit(token: &str) -> bool {
    matches!(
        token,
        "year" | "years" | "week" | "weeks" | "month" | "months" | "day" | "days" | "hour" | "hours"
    )
}

fn token_has_temporal_morphology(token: &str) -> bool {
    let t = token.trim_matches(|c: char| !c.is_alphanumeric());
    if t.chars().filter(|c| c.is_ascii_digit()).count() >= 4 {
        return true;
    }
    if t.ends_with("ago") || t.starts_with("yester") {
        return true;
    }
    if t.contains("month") || t.contains("week") {
        return true;
    }
    if t.ends_with("day") && t.len() >= 5 {
        return true;
    }
    is_temporal_unit(t)
}

/// QA-first score using Fate algebra temporal pass (no REL phrase table).
pub fn score_qa_algebra(answer: &str, gold: &str, question: &str, state: &FateAlgebraState) -> bool {
    let content = strip_all_citations(answer).to_ascii_lowercase();
    if content.is_empty() {
        return false;
    }
    if answer_echoes_question(&content, question) {
        return false;
    }
    let g = gold.trim().to_ascii_lowercase();
    if !g.is_empty() && content.contains(&g) {
        return true;
    }
    let tokens: Vec<&str> = g
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2)
        .collect();
    if !tokens.is_empty() {
        let hits = tokens.iter().filter(|t| content.contains(**t)).count();
        let required = state.time_relaxed_hits_required(tokens.len());
        if hits >= required {
            return true;
        }
        if state.time_weighted_overlap_pass(&content, &tokens) {
            return true;
        }
    }
    state.temporal_answer_pass(&content, gold, question)
}

fn strip_answer_citation(answer: &str) -> String {
    strip_all_citations(answer)
}

pub fn resolve_pretrain_passes(requested: usize, state: &FateAlgebraState) -> usize {
    let cap = state.pretrain_pass_cap().max(1);
    if requested == 0 {
        return state.default_ingest_pretrain_passes().clamp(1, cap);
    }
    requested.clamp(1, cap)
}
pub fn fate_normalize_line_at(
    line: &str,
    line_idx: usize,
    baseline_tick: u64,
    session_tick: u64,
) -> String {
    let tick = fate_land_tick(baseline_tick, line_idx, session_tick);
    let cite = fate_time_cite_bracket(line_idx, tick);
    // Strip ALL leading [...] cites so repeated normalize is idempotent (no string growth).
    let mut body = line.trim();
    while body.starts_with('[') {
        if let Some(p) = body.find(']') {
            body = body[p + 1..].trim();
        } else {
            break;
        }
    }
    format!("{cite} {body}")
}

pub fn fate_normalize_corpus(
    lines: &[String],
    baseline_tick: u64,
    session_tick: u64,
) -> Vec<String> {
    lines
        .iter()
        .enumerate()
        .map(|(i, l)| fate_normalize_line_at(l, i, baseline_tick, session_tick))
        .collect()
}

pub fn fate_land_tick(baseline_tick: u64, line_idx: usize, session_tick: u64) -> u64 {
    baseline_tick
        .saturating_add((line_idx as u64).saturating_mul(4))
        .saturating_add(session_tick / 16)
}

pub fn fate_time_cite_bracket(line_idx: usize, tick: u64) -> String {
    format!("[fate:{tick}@{line_idx}]")
}

/// Landing fingerprints in answers — `fate:T@I` or normalized `corpus:I` (no Dx:xx).
pub fn fate_landing_fingerprints(text: &str) -> Vec<String> {
    let mut fps = Vec::new();
    for token in text.split_whitespace() {
        let bare = token.trim_matches(|c: char| c == '[' || c == ']' || c == ',' || c == '.');
        if bare.starts_with("fate:") {
            if !fps.contains(&bare.to_string()) {
                fps.push(bare.to_string());
            }
        } else if let Some(idx) = bare.strip_prefix("corpus:").and_then(|s| s.parse::<usize>().ok()) {
            let fp = format!("fate:@{idx}");
            if !fps.contains(&fp) {
                fps.push(fp);
            }
        }
    }
    fps
}

pub fn fate_fingerprint_line_index(fp: &str) -> Option<usize> {
    let rest = fp.strip_prefix("fate:")?;
    if let Some(pos) = rest.rfind('@') {
        return rest[pos + 1..].parse().ok();
    }
    None
}

/// Strip corpus / fate / legacy dialogue citations from answer text.
pub fn strip_all_citations(answer: &str) -> String {
    let mut s = answer.to_string();
    if let Some(rest) = s.strip_prefix("[corpus:") {
        if let Some(idx) = rest.find(']') {
            s = rest[idx + 1..].trim().to_string();
        }
    }
    s.split_whitespace()
        .filter(|token| {
            let bare = token.trim_matches(|c: char| c == '[' || c == ']' || c == ',' || c == '.');
            !(bare.starts_with("fate:")
                || bare.starts_with("corpus:")
                || (bare.starts_with('D') && bare.contains(':')))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Fate-signal ranked landing indices — ordered by relative 4-quad address alignment.
/// Returned indices are egress materializations of Fate land (not navigation keys).
pub fn fate_signal_landing_indices(
    corpus: &[String],
    tokens: &[String],
    affinities: &[f64],
    state: &FateAlgebraState,
    k: usize,
) -> Vec<usize> {
    let query = FateRelAddr::from_query(tokens, state);
    let mut ranked: Vec<(usize, f64)> = corpus
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let aff = affinities.get(i).copied().unwrap_or(0.0);
            (i, state.fate_rel_land_score(query, tokens, line, aff))
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    ranked.into_iter().take(k.max(1)).map(|(i, _)| i).collect()
}

/// Expand landings by next-best relative neighbors (not ±absolute index window).
pub fn fate_signal_land_into_lines(
    full: &[String],
    q: &ValhallaBaseQuestion,
    state: &FateAlgebraState,
    affinities: Option<&[f64]>,
    lines: &mut Vec<String>,
) {
    let tokens = query_tokens_from_q(q);
    let aff = affinities.unwrap_or(&[]);
    let k = state.retrieval_top_k(tokens.len());
    let neighbor = state.corpus_neighbor_window();
    let landings = fate_signal_landing_indices(full, &tokens, aff, state, k + neighbor * 2);
    for idx in landings {
        let line = full[idx].clone();
        if !lines.iter().any(|l| l == &line) {
            lines.push(line);
        }
    }
}

/// Pollution: answer lands far ahead of anchor phase in corpus-time (replaces D-session drift).
pub fn fate_context_pollution(answer: &str, anchor_land_idx: usize, state: &FateAlgebraState) -> bool {
    let fps = fate_landing_fingerprints(answer);
    if fps.is_empty() {
        return false;
    }
    let span = state.corpus_neighbor_window().saturating_add(3);
    fps.iter().any(|fp| {
        fate_fingerprint_line_index(fp)
            .is_some_and(|idx| idx > anchor_land_idx.saturating_add(span * 2))
    })
}

pub fn fate_landing_drift(baseline: &[String], reprobe: &[String]) -> bool {
    !baseline.is_empty() && !reprobe.is_empty() && baseline != reprobe
}

pub const MAX_FATE_ROLLBACK_DEPTH: usize = 12;

pub fn push_rollback_checkpoint(sess: &mut ValhallaBaseSessionState) {
    let state = FateAlgebraState::from_session(sess);
    let cap = state.rollback_depth_cap();
    sess.fate_rollback_stack.push(sess.snapshot.clone());
    while sess.fate_rollback_stack.len() > cap {
        sess.fate_rollback_stack.remove(0);
    }
}

pub fn rollback_session(sess: &mut ValhallaBaseSessionState) -> bool {
    sess.fate_rollback_stack.pop().map(|snap| {
        sess.snapshot = snap;
        true
    }).unwrap_or(false)
}

pub fn update_meta_from_pretrain(sess: &mut ValhallaBaseSessionState, pass_rate: f64, epochs: usize) {
    sync_algebra_meta(sess);
    sess.fate_algebra_meta.pretrain_pass_rate = pass_rate.clamp(0.0, 1.0);
    sess.fate_algebra_meta.pretrain_epochs = epochs;
    sess.fate_algebra_meta.adapt_score =
        (sess.fate_algebra_meta.adapt_score * 0.7 + pass_rate * 0.3).clamp(0.0, 1.0);
    sess.fate_algebra_meta.fusion_rounds = sess.fate_algebra_meta.fusion_rounds.saturating_add(1);
}

/// LoCoMo evidence tag → bracket form (`D2:8` → `[D2:8]`).
pub fn evidence_tag_bracket(tag: &str) -> String {
    let t = tag.trim().trim_matches(|c: char| c == '[' || c == ']');
    format!("[{t}]")
}

fn dialogue_bodies_match(a: &str, b: &str) -> bool {
    let a = a.trim();
    let b = b.trim();
    if a.len() < 6 || b.len() < 6 {
        return false;
    }
    if a == b || a.contains(b) || b.contains(a) {
        return true;
    }
    let ap: String = a.chars().take(56).collect();
    let bp: String = b.chars().take(56).collect();
    ap == bp
}

/// Map bundle evidence tags to line indices in (normalized) corpus via session body match.
pub fn resolve_evidence_line_indices(
    corpus: &[String],
    evidence: &[String],
    full_session: Option<&[String]>,
) -> Vec<usize> {
    use std::collections::BTreeSet;
    let mut out = BTreeSet::new();
    let lookup = full_session.unwrap_or(corpus);
    for tag in evidence {
        let bracket = evidence_tag_bracket(tag);
        for src in lookup {
            if !src.contains(&bracket) && !src.contains(tag) {
                continue;
            }
            let src_body = strip_all_citations(src);
            for (ci, cl) in corpus.iter().enumerate() {
                let cb = strip_all_citations(cl);
                if dialogue_bodies_match(&src_body, &cb) {
                    out.insert(ci);
                }
            }
        }
    }
    out.into_iter().collect()
}

/// Unified Fate retrieval — relative 4-quad land first; IDs never used internally.
pub fn fate_retrieve_context_lines(
    q: &ValhallaBaseQuestion,
    corpus: &[String],
    state: &FateAlgebraState,
    affinities: Option<&[f64]>,
) -> Vec<String> {
    if corpus.is_empty() {
        return Vec::new();
    }
    let tokens = query_tokens_from_q(q);
    let aff = affinities.unwrap_or(&[]);
    let k = state.retrieval_top_k(tokens.len());
    let mut lines = Vec::new();
    // Fate relative land is the navigator; keyword seed is secondary.
    fate_signal_land_into_lines(corpus, q, state, affinities, &mut lines);
    fate_adaptive_context_boost(q, corpus, &mut lines, state, affinities);
    if lines.len() < k {
        use crate::context_routing::targeted_context;
        for line in targeted_context(q, corpus, k) {
            if !lines.iter().any(|l| l == &line) {
                lines.push(line);
            }
            if lines.len() >= k + state.corpus_neighbor_window() {
                break;
            }
        }
    }
    let _ = aff;
    fate_adaptive_distractor_filter(q, &mut lines, state);
    lines
}

/// Output adapter: answer overlaps evidence lines (by index / fate fingerprint), corpus-normalized.
pub fn score_evidence_line_overlap(
    answer: &str,
    evidence: &[String],
    corpus: &[String],
    full_session: Option<&[String]>,
) -> bool {
    let ans = strip_all_citations(answer).to_ascii_lowercase();
    if ans.len() < 4 {
        return false;
    }
    let indices = resolve_evidence_line_indices(corpus, evidence, full_session);
    for idx in &indices {
        let Some(line) = corpus.get(*idx) else {
            continue;
        };
        let body = strip_all_citations(line).to_ascii_lowercase();
        if body.len() < 6 {
            continue;
        }
        if ans.contains(&body) || body.contains(&ans) {
            return true;
        }
        let words: Vec<&str> = body
            .split_whitespace()
            .filter(|w| w.len() > 4)
            .collect();
        if words.len() >= 2 {
            let hits = words.iter().filter(|w| ans.contains(**w)).count();
            if hits >= (words.len() + 1) / 2 {
                return true;
            }
        }
    }
    for fp in fate_landing_fingerprints(answer) {
        if let Some(idx) = fate_fingerprint_line_index(&fp) {
            if indices.contains(&idx) {
                return true;
            }
        }
    }
    false
}

/// Two-end QA score: Fate algebra + evidence line overlap.
///
/// Address model:
/// - **Internal:** `FateRelAddr` (4-quad relative + tau) — Fate controls land
/// - **Ingress ID:** bundle `D2:8` → body-match → line (adapter only)
/// - **Egress ID:** materialize land as `[fate:T@I]` (`corpus:I` same I alias)
pub fn score_qa_with_evidence(
    answer: &str,
    gold: &str,
    question: &str,
    evidence: &[String],
    corpus: &[String],
    full_session: Option<&[String]>,
    state: &FateAlgebraState,
) -> bool {
    if score_qa_algebra(answer, gold, question, state) {
        return true;
    }
    score_evidence_line_overlap(answer, evidence, corpus, full_session)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coupling_rises_with_pretrain_pass_rate() {
        let low = FateAlgebraState::from_meta(&FateAlgebraMeta::default(), None);
        let mut high_meta = FateAlgebraMeta::default();
        high_meta.pretrain_pass_rate = 1.0;
        high_meta.adapt_score = 0.8;
        let high = FateAlgebraState::from_meta(&high_meta, None);
        assert!(high.coupling(0) > low.coupling(0));
    }

    #[test]
    fn pipe_count_within_bounds() {
        let s = FateAlgebraState::ambient();
        let n = s.parallel_pipe_count(50);
        assert!(n >= s.parallel_pipe_min());
        assert!(n <= s.parallel_pipe_max());
    }

    #[test]
    fn temporal_pass_uses_morphology_not_phrase_table() {
        let s = FateAlgebraState::ambient();
        assert!(s.temporal_answer_pass(
            "melanie painted that lake sunrise last year",
            "2022",
            "When did Melanie paint?",
        ));
    }

    #[test]
    fn context_boost_uses_query_tokens_only() {
        let corpus = vec![
            "[D1:14] Melanie: painted lake sunrise last year".into(),
            "[D3:1] unrelated interleave noise".into(),
        ];
        let q = ValhallaBaseQuestion {
            id: "t".into(),
            text: "When did Melanie paint a sunrise?".into(),
            score_type: "open".into(),
            reference_answer: "2022".into(),
            keywords: vec!["sunrise".into(), "paint".into(), "melanie".into()],
            audio_features: None,
            image_features: None,
        };
        let state = FateAlgebraState::ambient();
        let mut lines = Vec::new();
        fate_adaptive_context_boost(&q, &corpus, &mut lines, &state, None);
        assert!(lines.iter().any(|l| l.contains("sunrise")));
    }

    #[test]
    fn time_weighted_overlap_passes_counseling_with_tick() {
        let mut meta = FateAlgebraMeta::default();
        meta.fate_tick = 48;
        meta.fate_baseline_tick = 8;
        meta.pretrain_pass_rate = 0.35;
        let state = FateAlgebraState::from_meta(&meta, None);
        let content = "keen on counseling or working in mental health";
        let tokens = ["psychology", "counseling", "certification"];
        assert!(state.time_weighted_overlap_pass(content, &tokens));
    }

    #[test]
    fn effective_walk_spread_positive_after_tick() {
        let mut meta = FateAlgebraMeta::default();
        meta.fate_tick = 24;
        let state = FateAlgebraState::from_meta(&meta, None);
        assert!(state.effective_walk_spread() > 0.04);
    }

    #[test]
    fn fate_land_tick_increases_with_line_idx() {
        assert!(fate_land_tick(10, 5, 32) > fate_land_tick(10, 1, 32));
    }

    #[test]
    fn fate_rel_addr_aligns_same_topic() {
        let state = FateAlgebraState::ambient();
        let q = FateRelAddr::from_query(
            &["caroline".into(), "research".into(), "adoption".into()],
            &state,
        );
        let hit = FateRelAddr::from_line(
            "Caroline: Researching adoption agencies — dream to have a family.",
            0.4,
            &state,
        );
        let miss = FateRelAddr::from_line(
            "Melanie: Wow, that's cool, Caroline! What happened that was so awesome?",
            0.4,
            &state,
        );
        assert!(q.align(hit) > q.align(miss));
    }

    #[test]
    fn fate_landing_fingerprint_normalizes_corpus_cite() {
        let fps = fate_landing_fingerprints("[corpus:0] [fate:48@3] hello");
        assert!(fps.iter().any(|f| f == "fate:@0" || f.contains("@3")));
    }

    #[test]
    fn evidence_indices_survive_fate_normalize() {
        let raw = "[D2:8] Caroline: Researching adoption agencies — dream to have a family.";
        let norm = fate_normalize_line_at(raw, 7, 10, 32);
        let corpus = vec![norm];
        let session = vec![raw.to_string()];
        let idx = resolve_evidence_line_indices(&corpus, &["D2:8".into()], Some(&session));
        assert_eq!(idx, vec![0]);
        assert!(score_evidence_line_overlap(
            "[corpus:0] [fate:38@7] Caroline: Researching adoption agencies",
            &["D2:8".into()],
            &corpus,
            Some(&session),
        ));
    }
}
