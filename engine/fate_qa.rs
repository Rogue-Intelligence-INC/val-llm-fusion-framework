//! Fate-native QA readout (owner architecture).
//!
//! P1 (exposure only, no behaviour change): every QA row carries the
//! Fate grasp record — which blocks (fragments) the question stimulus
//! activated, their Fate scores, the full candidate score series
//! (s1..sk, descending) produced by the field readout, and the grasped
//! corpus line. Downstream (gateway, tests) must read this record
//! instead of reconstructing grasp from answer text or lexical tracks.
//!
//! P1b (decision form): the gate reading is `true_block_reading` —
//! the grasp is judged against competition inside its own block
//! (stored fragment line range), never against the whole field.
//! `native_qa` applies it to supersede global-margin abstains only.
//! The separation threshold is Fate's own: the session cluster
//! equation's `bundle_fits` under its adapted threshold. No
//! calibrated constant lives in this gate.

use serde::{Deserialize, Serialize};

/// Fate grasp record for one question.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FateGrasp {
    /// Activated block (fragment) ids, best first.
    pub selected_fragments: Vec<String>,
    /// Fate score per selected block, aligned with `selected_fragments`.
    pub fragment_scores: Vec<f64>,
    /// Candidate line score series, descending (s1 >= s2 >= ...).
    pub candidate_scores: Vec<f64>,
    /// Corpus index per candidate, aligned with `candidate_scores`
    /// (None when the match carries no corpus citation).
    #[serde(default)]
    pub candidate_indices: Vec<Option<usize>>,
    /// Corpus line indices inside the activated blocks for this
    /// question (the field's chosen substrate, before ranking).
    #[serde(default)]
    pub activated_indices: Vec<usize>,
    /// Corpus index of the grasped line, when the field produced one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grasped_idx: Option<usize>,
    /// Line range [start, end] of the stored block that contains the
    /// grasped line (true block membership, not the activated union).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grasp_block: Option<[usize; 2]>,
    /// Within-block relative margin of the grasp: competition read
    /// inside its own block only. See `true_block_reading`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fate_block_rel: Option<f64>,
    /// How many series candidates fall inside the grasped block.
    #[serde(default)]
    pub block_candidate_count: usize,
    /// Fate-state telemetry: the session snapshot's corpus-affinity
    /// dispersion, from which the Fate algebra derives its scales.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alg_affinity_std: Option<f64>,
    /// What the session's own Fate algebra would require as relative
    /// margin here: delta_required(s_grasp) / s_grasp (default meta —
    /// the strict bound; session pretrain/adapt only relaxes it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alg_delta_rel: Option<f64>,
    /// The grasped block's own Fate equation threshold, frozen in its
    /// triple_fate_slice when the block closed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_cluster_threshold: Option<f64>,
    /// The session's LIVE global Fate equation at the gate point
    /// (hub.triple_fate.cluster_equation): coefficients, its own
    /// threshold, and how many times Fate has adapted it. This is
    /// the quantity that should control the gate threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eq_coefs: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eq_threshold: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eq_adaptations: Option<u64>,
}

impl FateGrasp {
    pub fn build(
        selected_fragments: Vec<String>,
        fragment_scores: Vec<f64>,
        candidate_scores: Vec<f64>,
        candidate_indices: Vec<Option<usize>>,
        activated_indices: Vec<usize>,
        grasped_idx: Option<usize>,
    ) -> Option<Self> {
        if selected_fragments.is_empty() && candidate_scores.is_empty() {
            return None;
        }
        Some(Self {
            selected_fragments,
            fragment_scores,
            candidate_scores,
            candidate_indices,
            activated_indices,
            grasped_idx,
            grasp_block: None,
            fate_block_rel: None,
            block_candidate_count: 0,
            alg_affinity_std: None,
            alg_delta_rel: None,
            block_cluster_threshold: None,
            eq_coefs: None,
            eq_threshold: None,
            eq_adaptations: None,
        })
    }

    /// Attach the Fate-state scales recorded at the gate point.
    pub fn set_fate_scales(
        &mut self,
        affinity_std: f64,
        delta_rel: Option<f64>,
        block_cluster_threshold: Option<f64>,
    ) {
        self.alg_affinity_std = Some(affinity_std);
        self.alg_delta_rel = delta_rel;
        self.block_cluster_threshold = block_cluster_threshold;
    }

    /// Attach the session's live Fate equation state.
    pub fn set_live_equation(&mut self, coefs: [f64; 3], threshold: f64, adaptations: u64) {
        self.eq_coefs = Some(coefs);
        self.eq_threshold = Some(threshold);
        self.eq_adaptations = Some(adaptations);
    }

    /// Attach the true block reading (see `true_block_reading`).
    pub fn set_block_reading(&mut self, reading: &BlockReading) {
        self.grasp_block = Some([reading.line_start, reading.line_end]);
        self.fate_block_rel = Some(reading.rel_margin);
        self.block_candidate_count = reading.n_candidates;
    }

    /// Candidate series restricted to the activated blocks: the
    /// field's own substrate, where a block-local gate should read.
    pub fn block_series(&self) -> Vec<(usize, f64)> {
        let active: std::collections::HashSet<usize> =
            self.activated_indices.iter().copied().collect();
        self.candidate_indices
            .iter()
            .zip(self.candidate_scores.iter())
            .filter_map(|(idx, s)| idx.filter(|i| active.contains(i)).map(|i| (i, *s)))
            .collect()
    }

    /// Within-block relative margin of the grasped candidate:
    /// (s_grasp - s_best_other_in_block) / s_grasp. None when the
    /// grasp is not inside the activated substrate or has no score.
    pub fn block_rel_margin(&self) -> Option<f64> {
        let g = self.grasped_idx?;
        let series = self.block_series();
        let s_grasp = series.iter().find(|(i, _)| *i == g).map(|(_, s)| *s)?;
        let s_other = series
            .iter()
            .filter(|(i, _)| *i != g)
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        if s_grasp <= 0.0 {
            return None;
        }
        let margin = if s_other.is_finite() {
            s_grasp - s_other
        } else {
            s_grasp // no competitor inside the activated blocks
        };
        Some(margin / s_grasp)
    }

    /// Adjacent differences of the candidate series: [s1-s2, s2-s3, ...].
    /// This is the series a Fate convergence judgement would read.
    pub fn diff_series(&self) -> Vec<f64> {
        self.candidate_scores
            .windows(2)
            .map(|w| w[0] - w[1])
            .collect()
    }

    /// Top-two margin (s1 - s2), 0 when fewer than two candidates.
    pub fn margin(&self) -> f64 {
        if self.candidate_scores.len() >= 2 {
            self.candidate_scores[0] - self.candidate_scores[1]
        } else {
            0.0
        }
    }
}

/// Parse the corpus index out of a native match citation
/// ("[corpus:12] ..." -> Some(12)); other citations -> None.
pub fn parse_corpus_idx(match_text: &str) -> Option<usize> {
    let inner = match_text.strip_prefix('[')?.split(']').next()?;
    inner.strip_prefix("corpus:")?.parse().ok()
}

/// The Fate block reading for one grasp (owner architecture).
///
/// Grasp is judged against competition inside its own block only:
/// lines outside the block that contains the grasped line do not
/// compete with it. This replaces the global top-2 margin reading,
/// which abstains whenever two lines from *different* blocks score
/// alike even though they never shared a substrate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BlockReading {
    pub line_start: usize,
    pub line_end: usize,
    /// (s_grasp - s_best_other_in_block) / s_grasp; 1.0 when no other
    /// series candidate falls inside the block.
    pub rel_margin: f64,
    /// Series candidates (incl. the grasp) inside the block.
    pub n_candidates: usize,
    /// Score of the grasped line itself.
    pub s_grasp: f64,
    /// Best competing score inside the block, if any.
    pub s_other: Option<f64>,
}

/// Compute the block reading. None when the grasped line belongs to
/// no stored block (unclosed material — the grasp is not yet on
/// bundled substrate) or carries no score in the series.
pub fn true_block_reading(
    candidate_indices: &[Option<usize>],
    candidate_scores: &[f64],
    grasped_idx: usize,
    block_ranges: &[(usize, usize)],
) -> Option<BlockReading> {
    let (ls, le) = block_ranges
        .iter()
        .find(|(s, e)| grasped_idx >= *s && grasped_idx <= *e)
        .copied()?;
    let mut s_grasp: Option<f64> = None;
    let mut s_other = f64::NEG_INFINITY;
    let mut n = 0usize;
    for (idx, score) in candidate_indices.iter().zip(candidate_scores.iter()) {
        if let Some(i) = idx {
            if *i >= ls && *i <= le {
                n += 1;
                if *i == grasped_idx {
                    s_grasp = Some(*score);
                } else if *score > s_other {
                    s_other = *score;
                }
            }
        }
    }
    let s_grasp = s_grasp?;
    if s_grasp <= 0.0 {
        return None;
    }
    let rel_margin = if s_other.is_finite() {
        (s_grasp - s_other) / s_grasp
    } else {
        1.0
    };
    Some(BlockReading {
        line_start: ls,
        line_end: le,
        rel_margin,
        n_candidates: n,
        s_grasp,
        s_other: if s_other.is_finite() { Some(s_other) } else { None },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Recorded offline vectors (valhalla-server probes, 2026-11-14):
    // multimodal q1: grasp correct, gate abstains on a near-tie series.
    #[test]
    fn recorded_q1_near_tie_series() {
        let g = FateGrasp::build(
            vec!["seg_1".into(), "seg_2".into()],
            vec![0.31, 0.29],
            vec![1.3665087882942228, 1.3662427025488026],
            vec![Some(0), Some(2)],
            vec![0, 2, 3],
            Some(0),
        )
        .unwrap();
        assert_eq!(g.grasped_idx, Some(0));
        assert!((g.margin() - 0.0002660857454202).abs() < 1e-12);
        assert_eq!(g.diff_series().len(), 1);
        assert_eq!(g.block_series().len(), 2);
    }

    #[test]
    fn empty_grasp_is_none() {
        assert!(FateGrasp::build(vec![], vec![], vec![], vec![], vec![], None).is_none());
    }

    #[test]
    fn corpus_idx_parse() {
        assert_eq!(parse_corpus_idx("[corpus:12] hello"), Some(12));
        assert_eq!(parse_corpus_idx("[fate:3@1] hello"), None);
        assert_eq!(parse_corpus_idx("plain"), None);
    }

    // Post-head-closure q1 shape: the global tie competitor (line 2)
    // lives in a different block, so inside the grasped head block
    // [0,1] the grasp stands clear.
    #[test]
    fn block_reading_cross_block_tie_is_clear() {
        let idx: Vec<Option<usize>> = vec![Some(0), Some(2), Some(5), Some(3), Some(4), Some(1)];
        let scores = vec![1.4003, 1.3913, 1.3062, 1.2806, 1.2746, 1.10];
        let ranges = vec![(0, 1), (2, 4), (5, 6)];
        let r = true_block_reading(&idx, &scores, 0, &ranges).unwrap();
        assert_eq!((r.line_start, r.line_end), (0, 1));
        assert_eq!(r.n_candidates, 2);
        assert!((r.rel_margin - (1.4003 - 1.10) / 1.4003).abs() < 1e-9);
    }

    // Conflict shape: both conflicting lines share one block, so the
    // block reading stays a near-tie and silence is preserved.
    #[test]
    fn block_reading_in_block_tie_stays_tied() {
        let idx: Vec<Option<usize>> = vec![Some(1), Some(0), Some(2), Some(5), Some(7)];
        let scores = vec![1.5283, 1.5107, 1.4123, 1.4106, 1.3892];
        let ranges = vec![(0, 1), (2, 4), (5, 7)];
        let r = true_block_reading(&idx, &scores, 1, &ranges).unwrap();
        assert!((r.rel_margin - (1.5283 - 1.5107) / 1.5283).abs() < 1e-9);
        assert_eq!(r.s_grasp, 1.5283);
        assert_eq!(r.s_other, Some(1.5107));
        // Under the birth equation (thr 0.01) the pair still fits one
        // bundle: (s1/s2 - 1)^2 ~= 0.00013 < 0.01.
        let x = r.s_grasp / r.s_other.unwrap();
        assert!((x - 1.0).powi(2) < 0.01);
    }

    #[test]
    fn block_reading_unclosed_material_is_none() {
        let idx: Vec<Option<usize>> = vec![Some(0), Some(2)];
        let scores = vec![1.40, 1.39];
        assert!(true_block_reading(&idx, &scores, 0, &[(2, 4)]).is_none());
    }
}
